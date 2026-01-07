import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from layers.MoE import MoE
from layers.RevIN import RevIN


class Model(nn.Module):
    def __init__(self, configs):
        super(Model, self).__init__()
        self.layer_nums = configs.layer_nums
        self.num_nodes = configs.num_nodes
        self.pre_len = configs.pred_len
        self.seq_len = configs.seq_len
        self.k = configs.k
        self.num_experts_list = configs.num_experts_list
        self.patch_size_list = configs.patch_size_list
        self.d_model = configs.d_model
        self.d_ff = configs.d_ff
        self.residual_connection = configs.residual_connection
        self.revin = configs.revin
        if self.revin:
            self.revin_layer = RevIN(num_features=configs.num_nodes, affine=False, subtract_last=False)

        self.start_fc = nn.Linear(in_features=1, out_features=self.d_model)
        self.AMS_lists = nn.ModuleList()
        self.device = torch.device('cuda:{}'.format(configs.gpu))
        self.batch_norm = configs.batch_norm
        self.temp = configs.temp
        self.use_graph = getattr(configs, 'use_graph', False)
        self.graph_dynamic = getattr(configs, 'graph_dynamic', False)
        self.graph_emb_dim = getattr(configs, 'graph_emb_dim', 8)
        self.graph_dropout = getattr(configs, 'graph_dropout', 0.0)
        self.graph_alpha = getattr(configs, 'graph_alpha', 0.1)
        if self.use_graph:
            self.node_emb1 = nn.Parameter(torch.randn(self.num_nodes, self.graph_emb_dim))
            self.node_emb2 = nn.Parameter(torch.randn(self.num_nodes, self.graph_emb_dim))
            self.graph_drop = nn.Dropout(self.graph_dropout)
        
        for num in range(self.layer_nums):
            self.AMS_lists.append(
                MoE(self.seq_len, self.seq_len, self.num_experts_list[num], self.device, k=self.k,
                    num_nodes=self.num_nodes, patch_size=self.patch_size_list[num], noisy_gating=True,
                    d_model=self.d_model, d_ff=self.d_ff, layer_number=num + 1,
                    residual_connection=self.residual_connection, batch_norm=self.batch_norm, temp=self.temp,
                    ))
        self.projections = nn.Sequential(
            nn.Linear(self.seq_len * self.d_model, self.pre_len)
        )

    def _graph_adj(self, x_norm):
        base = torch.relu(self.node_emb1 @ self.node_emb2.t())
        if self.graph_dynamic:
            x_mean = x_norm.mean(dim=1)
            corr = torch.einsum('bi,bj->bij', x_mean, x_mean)
            base = base.unsqueeze(0) + self.graph_alpha * corr
            adj = F.softmax(base, dim=-1)
        else:
            adj = F.softmax(base, dim=-1)
        return self.graph_drop(adj)

    def _apply_graph(self, x_norm):
        adj = self._graph_adj(x_norm)
        if adj.dim() == 2:
            return torch.einsum('ij,blj->bli', adj, x_norm)
        return torch.einsum('bij,blj->bli', adj, x_norm)

    def forward(self, x, return_features: bool = False, return_x_norm: bool = False):
        # x: [bz, seq, d]
        balance_loss = 0
        contrastive_loss = []
        x_norm = self.revin_layer(x, 'norm')
        if self.use_graph:
            x_norm = self._apply_graph(x_norm)
        out = self.start_fc(x_norm.unsqueeze(-1))  # out: [bz, seq, d, d_model]

        batch_size = x.shape[0]
        
        for layer in self.AMS_lists:
            out, aux_loss, con_loss = layer(out)
            balance_loss += aux_loss
            contrastive_loss.append(con_loss)

        cond_features = out
        out = out.permute(0,2,1,3).reshape(batch_size, self.num_nodes, -1)
        out = self.projections(out).transpose(2, 1)

        out = self.revin_layer(out, 'denorm')

        contrastive_loss = torch.stack(contrastive_loss)
        contrastive_loss = torch.mean(contrastive_loss)

        if return_features or return_x_norm:
            return out, balance_loss, contrastive_loss, cond_features, (x_norm if return_x_norm else None)
        return out, balance_loss, contrastive_loss
