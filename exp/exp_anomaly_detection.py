from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, confusion_matrix
import torch.multiprocessing
from models import MTCL
from models.diffusion import ConditionalDiffusion
from models.ebm import EBMScorer
from torch.optim import lr_scheduler

torch.multiprocessing.set_sharing_strategy('file_system')
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
import torch.nn.functional as F
import math
from scipy.stats import genpareto
from sklearn.neighbors import NearestNeighbors

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        self.train_losses = []
        self.val_losses = []
        self.lambda_contrastive = args.lambda_contrastive
        self.use_diffusion = getattr(args, 'use_diffusion', False)
        self.use_ebm = getattr(args, 'use_ebm', False)
        self.knn_index = None
        if self.use_diffusion:
            cond_dim = args.num_nodes * args.d_model
            self.diffusion = ConditionalDiffusion(
                num_nodes=args.num_nodes,
                cond_dim=cond_dim,
                model_dim=getattr(args, 'diffusion_dim', 64),
                num_heads=getattr(args, 'diffusion_heads', 4),
                depth=getattr(args, 'diffusion_depth', 3),
                timesteps=getattr(args, 'diffusion_steps', 100),
                beta_schedule=getattr(args, 'diffusion_beta_schedule', 'linear'),
                beta_start=getattr(args, 'diffusion_beta_start', 1e-4),
                beta_end=getattr(args, 'diffusion_beta_end', 2e-2),
                cond_reconstruction=getattr(args, 'lambda_rec', 0.0) > 0,
            ).to(self.device)
        else:
            self.diffusion = None
        if self.use_ebm:
            self.ebm = EBMScorer(input_dim=args.num_nodes,
                                 hidden_dim=getattr(args, 'ebm_hidden', 64)).to(self.device)
        else:
            self.ebm = None
        
    def _build_model(self):
        model_dict = {
            'MTCL': MTCL,
        }
        model = model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader

    def _select_optimizer(self):
        params = list(self.model.parameters())
        if self.diffusion is not None:
            params += list(self.diffusion.parameters())
        if self.ebm is not None:
            params += list(self.ebm.parameters())
        model_optim = optim.Adam(params, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def _ddim_denoise(self, x_norm, cond_flat):
        """
        DDIM sampling from noisy x_norm back to clean signal.
        """
        steps = min(getattr(self.args, 'ddim_steps', 20), self.diffusion.timesteps)
        eta = getattr(self.args, 'ddim_eta', 0.0)
        t_seq = torch.linspace(self.diffusion.timesteps - 1, 0, steps, device=self.device).long()
        t_start = t_seq[0].item()

        noise = torch.randn_like(x_norm)
        alpha_bar_start = self.diffusion.alpha_bars[t_start].view(1, 1, 1)
        x_t = torch.sqrt(alpha_bar_start) * x_norm + torch.sqrt(1 - alpha_bar_start) * noise

        for idx, t in enumerate(t_seq):
            t_int = t.item()
            t_batch = torch.full((x_norm.size(0),), t_int, device=self.device, dtype=torch.long)
            eps_pred = self.diffusion(x_t, t_batch, cond_flat)

            alpha_bar_t = self.diffusion.alpha_bars[t_int].view(1, 1, 1)
            alpha_t = self.diffusion.alphas[t_int].view(1, 1, 1)

            if idx < len(t_seq) - 1:
                t_prev = t_seq[idx + 1].item()
                alpha_bar_prev = self.diffusion.alpha_bars[t_prev].view(1, 1, 1)
                sigma_t = eta * torch.sqrt(
                    torch.clamp((1 - alpha_bar_prev) / (1 - alpha_bar_t) * (1 - alpha_t / alpha_bar_prev), min=0.0)
                )
            else:
                alpha_bar_prev = torch.ones_like(alpha_bar_t)
                sigma_t = torch.zeros_like(alpha_bar_t)

            pred_x0 = (x_t - torch.sqrt(1 - alpha_bar_t) * eps_pred) / torch.sqrt(alpha_bar_t)

            if idx < len(t_seq) - 1:
                c1 = torch.sqrt(alpha_bar_prev)
                c2 = torch.sqrt(torch.clamp(1 - alpha_bar_prev - sigma_t ** 2, min=0.0))
                noise = torch.randn_like(x_t)
                x_t = c1 * pred_x0 + c2 * eps_pred + sigma_t * noise
            else:
                x_t = pred_x0

        return x_t

    def _reconstruct_batch(self, batch_x, t_eval_default, sampling_mode):
        batch_x = batch_x.float().to(self.device)
        if self.use_diffusion:
            outputs, _, _, cond_feats, x_norm = self.model(
                batch_x, return_features=True, return_x_norm=True
            )
            cond_flat = cond_feats.view(cond_feats.shape[0], cond_feats.shape[1], -1)
            if sampling_mode == 'ddim':
                recon_norm = self._ddim_denoise(x_norm, cond_flat)
            else:
                t_eval = min(getattr(self.args, 'diffusion_eval_step', t_eval_default), t_eval_default)
                t_tensor = torch.full((batch_x.size(0),), t_eval, device=self.device, dtype=torch.long)
                noise = torch.randn_like(x_norm)
                alpha_bar = self.diffusion.alpha_bars[t_eval]
                z_t = torch.sqrt(alpha_bar) * x_norm + torch.sqrt(1 - alpha_bar) * noise
                eps_pred = self.diffusion(z_t, t_tensor, cond_flat)
                recon_norm = (z_t - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
            revin_layer = self.model.module.revin_layer if hasattr(self.model, 'module') else self.model.revin_layer
            recon = revin_layer(recon_norm, 'denorm')
        else:
            outputs, _, _ = self.model(batch_x)
            recon = outputs
        return recon

    def _noise_score_batch(self, batch_x):
        """
        Single-step noise score: predict noise at small t and take L2 energy.
        """
        return self._noise_score_batch_mode(batch_x, score_mode='l2')

    def _noise_feature_batch(self, batch_x):
        batch_x = batch_x.float().to(self.device)
        if not (self.use_diffusion and self.diffusion is not None):
            return None, None
        t_noise = min(1, self.diffusion.timesteps - 1)
        outputs, _, _, cond_feats, x_norm = self.model(
            batch_x, return_features=True, return_x_norm=True
        )
        cond_flat = cond_feats.view(cond_feats.shape[0], cond_feats.shape[1], -1)
        t_tensor = torch.full((batch_x.size(0),), t_noise, device=self.device, dtype=torch.long)
        alpha_bar = self.diffusion.alpha_bars[t_noise]
        noise = torch.randn_like(x_norm)
        z_t = torch.sqrt(alpha_bar) * x_norm + torch.sqrt(1 - alpha_bar) * noise
        eps_pred = self.diffusion(z_t, t_tensor, cond_flat)
        l2_score = torch.mean(eps_pred ** 2, dim=(1, 2))
        feat = eps_pred.mean(dim=1)
        return l2_score, feat

    def _noise_score_batch_mode(self, batch_x, score_mode='l2'):
        l2_score, feat = self._noise_feature_batch(batch_x)
        if l2_score is None:
            return None
        if score_mode == 'ebm':
            if self.ebm is None:
                return l2_score.detach().cpu().numpy()
            energy = self.ebm(feat)
            return energy.detach().cpu().numpy()
        if score_mode == 'knn':
            if self.knn_index is None:
                return l2_score.detach().cpu().numpy()
            feat_np = feat.detach().cpu().numpy()
            dists, _ = self.knn_index.kneighbors(feat_np, return_distance=True)
            return dists.mean(axis=1)
        return l2_score.detach().cpu().numpy()

    def _build_knn_index(self, train_loader, max_samples=50000, knn_k=5):
        feats = []
        with torch.no_grad():
            for batch_x, _, _, _ in train_loader:
                _, feat = self._noise_feature_batch(batch_x)
                if feat is None:
                    continue
                feats.append(feat.detach().cpu().numpy())
        if not feats:
            return
        feats = np.concatenate(feats, axis=0)
        if feats.shape[0] > max_samples:
            idx = np.random.choice(feats.shape[0], size=max_samples, replace=False)
            feats = feats[idx]
        knn_k = max(1, min(knn_k, feats.shape[0]))
        self.knn_index = NearestNeighbors(n_neighbors=knn_k)
        self.knn_index.fit(feats)

    def vali(self, vali_data, vali_loader, criterion):
        torch.cuda.empty_cache() 
        total_loss = []
        self.model.eval()
        if self.diffusion is not None:
            self.diffusion.eval()
        if self.ebm is not None:
            self.ebm.eval()
        with torch.no_grad():
            for i, (batch_x, _, _, _) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.model == 'MTCL':
                            if self.use_diffusion:
                                outputs, balance_loss, contrastive_loss, cond_feats, x_norm = self.model(
                                    batch_x, return_features=True, return_x_norm=True
                                )
                            else:
                                outputs, balance_loss, contrastive_loss = self.model(batch_x)
                                cond_feats, x_norm = None, None
                        else:
                            outputs, balance_loss, contrastive_loss, cond_feats, x_norm = self.model(batch_x), 0, 0, None, None
                else:
                    if self.args.model == 'MTCL':
                        if self.use_diffusion:
                            outputs, balance_loss, contrastive_loss, cond_feats, x_norm = self.model(
                                batch_x, return_features=True, return_x_norm=True
                            )
                        else:
                            outputs, balance_loss, contrastive_loss = self.model(batch_x)
                            cond_feats, x_norm = None, None
                    else:
                        outputs, balance_loss, contrastive_loss, cond_feats, x_norm = self.model(batch_x), 0, 0, None, None

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                pred = outputs.detach().cpu()
                true = batch_x.detach().cpu()

                reconstruction_loss = criterion(pred, true)
                loss = reconstruction_loss + self.lambda_contrastive * contrastive_loss

                if self.use_diffusion:
                    t = torch.randint(
                        low=0,
                        high=self.diffusion.timesteps,
                        size=(batch_x.size(0),),
                        device=self.device,
                        dtype=torch.long,
                    )
                    noise = torch.randn_like(x_norm)
                    alpha_bar = self.diffusion.alpha_bars[t].view(-1, 1, 1)
                    z_t = torch.sqrt(alpha_bar) * x_norm + torch.sqrt(1 - alpha_bar) * noise
                    cond_flat = cond_feats.view(cond_feats.shape[0], cond_feats.shape[1], -1)
                    eps_pred = self.diffusion(z_t, t, cond_flat)
                    diff_loss = F.mse_loss(eps_pred, noise)
                    loss = loss + self.args.lambda_diff * diff_loss
                    if getattr(self.args, 'lambda_rec', 0.0) > 0:
                        rec_pred = self.diffusion.reconstruct_from_cond(cond_flat)
                        rec_loss = criterion(rec_pred, x_norm)
                        loss = loss + self.args.lambda_rec * rec_loss
                    if self.use_ebm and self.ebm is not None:
                        t0 = torch.zeros(batch_x.size(0), device=self.device, dtype=torch.long)
                        eps_pos = self.diffusion(x_norm, t0, cond_flat)
                        feat_pos = eps_pos.mean(dim=1)
                        feat_neg = eps_pred.mean(dim=1)
                        ebm_loss = (self.ebm(feat_pos) - self.ebm(feat_neg)).mean()
                        loss = loss + self.args.lambda_ebm * ebm_loss
                
                total_loss.append(loss.item())

        average_loss = np.mean(total_loss)
        self.model.train()
        if self.diffusion is not None:
            self.diffusion.train()
        self.val_losses.append(average_loss)
        return average_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')
        accum_steps = max(1, getattr(self.args, 'accum_steps', 1))

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        eff_steps = math.ceil(train_steps / accum_steps)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=eff_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
        

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_balance_loss = []
            train_contrastive_loss = []

            self.model.train()
            if self.ebm is not None:
                self.ebm.train()
            epoch_time = time.time()
            for i, (batch_x, _, _, _) in enumerate(train_loader):
                iter_count += 1
                if i % accum_steps == 0:
                    model_optim.zero_grad()

                batch_x = batch_x.float().to(self.device)

                if self.args.model == 'MTCL':
                    if self.use_diffusion:
                        outputs, balance_loss, contrastive_loss, cond_feats, x_norm = self.model(
                            batch_x, return_features=True, return_x_norm=True
                        )
                    else:
                        outputs, balance_loss, contrastive_loss = self.model(batch_x)
                        cond_feats, x_norm = None, None
                else:
                    outputs, balance_loss, contrastive_loss, cond_feats, x_norm = self.model(batch_x), 0, 0, None, None
                

                f_dim = -1 if self.args.features == 'MS' else 0
                outputs = outputs[:, :, f_dim:]
                construct_loss = criterion(outputs, batch_x)
                loss = construct_loss + balance_loss + self.lambda_contrastive * contrastive_loss

                diff_loss = torch.tensor(0.0, device=self.device)
                rec_loss = torch.tensor(0.0, device=self.device)
                ebm_loss = torch.tensor(0.0, device=self.device)
                if self.use_diffusion:
                    t = torch.randint(
                        low=0,
                        high=self.diffusion.timesteps,
                        size=(batch_x.size(0),),
                        device=self.device,
                        dtype=torch.long,
                    )
                    noise = torch.randn_like(x_norm)
                    alpha_bar = self.diffusion.alpha_bars[t].view(-1, 1, 1)
                    z_t = torch.sqrt(alpha_bar) * x_norm + torch.sqrt(1 - alpha_bar) * noise
                    cond_flat = cond_feats.view(cond_feats.shape[0], cond_feats.shape[1], -1)
                    eps_pred = self.diffusion(z_t, t, cond_flat)
                    diff_loss = F.mse_loss(eps_pred, noise)
                    loss = loss + self.args.lambda_diff * diff_loss

                    if getattr(self.args, 'lambda_rec', 0.0) > 0:
                        rec_pred = self.diffusion.reconstruct_from_cond(cond_flat)
                        rec_loss = criterion(rec_pred, x_norm)
                        loss = loss + self.args.lambda_rec * rec_loss

                    if self.use_ebm and self.ebm is not None:
                        t0 = torch.zeros(batch_x.size(0), device=self.device, dtype=torch.long)
                        eps_pos = self.diffusion(x_norm, t0, cond_flat)
                        feat_pos = eps_pos.mean(dim=1)
                        feat_neg = eps_pred.mean(dim=1)
                        ebm_loss = (self.ebm(feat_pos) - self.ebm(feat_neg)).mean()
                        loss = loss + self.args.lambda_ebm * ebm_loss

                train_loss.append(loss.item())
                self.train_losses.append(loss.item())
                train_balance_loss.append(balance_loss.item())
                train_contrastive_loss.append(contrastive_loss.item())
                if self.use_diffusion:
                    train_contrastive_loss[-1] = contrastive_loss.item()

                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                step_now = ((i + 1) % accum_steps == 0) or (i + 1 == train_steps)
                if self.args.use_amp:
                    scaler.scale(loss / accum_steps).backward()
                    if step_now:
                        scaler.unscale_(model_optim)
                        params_to_clip = list(self.model.parameters())
                        if self.diffusion is not None:
                            params_to_clip += list(self.diffusion.parameters())
                        torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=4.0)
                        scaler.step(model_optim)
                        scaler.update()
                        if self.args.lradj == 'TST':
                            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                            scheduler.step()
                else:
                    (loss / accum_steps).backward()
                    if step_now:
                        params_to_clip = list(self.model.parameters())
                        if self.diffusion is not None:
                            params_to_clip += list(self.diffusion.parameters())
                        torch.nn.utils.clip_grad_norm_(params_to_clip, max_norm=4.0)
                        model_optim.step()
                        if self.args.lradj == 'TST':
                            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args, printout=False)
                            scheduler.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)
            balance_loss_ep = np.average(train_balance_loss)
            contrastive_loss_ep = np.average(train_contrastive_loss)
            
            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f} Balance: {5:.7f} Contrast: {6:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss, balance_loss_ep, contrastive_loss_ep))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(model_optim, scheduler, epoch + 1, self.args)

        best_model_path = path + '/' + 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        if self.diffusion is not None:
            diffusion_path = path + '/' + 'diffusion.pth'
            torch.save(self.diffusion.state_dict(), diffusion_path)
        if self.ebm is not None:
            ebm_path = path + '/' + 'ebm.pth'
            torch.save(self.ebm.state_dict(), ebm_path)

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')
        threshold_mode = getattr(self.args, 'threshold_mode', 'train_test')
        sampling_mode = getattr(self.args, 'diffusion_sampling', 'one_step')
        use_noise_score = getattr(self.args, 'use_noise_score', False)
        use_evt = getattr(self.args, 'use_evt', False)
        evt_tail_frac = max(0.01, min(0.5, getattr(self.args, 'evt_tail_frac', 0.1)))
        evt_conf = max(0.9, min(0.999, getattr(self.args, 'evt_conf', 0.99)))
        noise_score_mode = getattr(self.args, 'noise_score_mode', 'l2')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            if self.diffusion is not None:
                diffusion_path = os.path.join('./checkpoints/' + setting, 'diffusion.pth')
                if os.path.exists(diffusion_path):
                    self.diffusion.load_state_dict(torch.load(diffusion_path, map_location=self.device))
            if self.ebm is not None:
                ebm_path = os.path.join('./checkpoints/' + setting, 'ebm.pth')
                if os.path.exists(ebm_path):
                    self.ebm.load_state_dict(torch.load(ebm_path, map_location=self.device))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        if self.diffusion is not None:
            self.diffusion.eval()
        if self.ebm is not None:
            self.ebm.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        t_eval_default = self.diffusion.timesteps // 2 if self.diffusion is not None else 0
        if use_noise_score and noise_score_mode == 'ebm' and self.ebm is None:
            print('Warning: noise_score_mode=ebm but EBM is disabled; fallback to l2.')
        if use_noise_score and noise_score_mode == 'knn' and self.use_diffusion:
            self._build_knn_index(
                train_loader,
                max_samples=getattr(self.args, 'knn_max_samples', 50000),
                knn_k=getattr(self.args, 'knn_k', 5)
            )
        if use_noise_score and noise_score_mode == 'knn' and self.knn_index is None:
            print('Warning: kNN index not built; fallback to l2.')

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(train_loader):
                if use_noise_score:
                    score = self._noise_score_batch_mode(batch_x, noise_score_mode)
                    if score is None:
                        recon = self._reconstruct_batch(batch_x, t_eval_default, sampling_mode)
                        score = torch.mean(self.anomaly_criterion(batch_x.float().to(self.device), recon), dim=-1).detach().cpu().numpy()
                else:
                    recon = self._reconstruct_batch(batch_x, t_eval_default, sampling_mode)
                    score = torch.mean(self.anomaly_criterion(batch_x.float().to(self.device), recon), dim=-1).detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        val_energy = []
        if threshold_mode == 'val_test':
            with torch.no_grad():
                for i, (batch_x, batch_y, _, _) in enumerate(val_loader):
                    if use_noise_score:
                        score = self._noise_score_batch_mode(batch_x, noise_score_mode)
                        if score is None:
                            recon = self._reconstruct_batch(batch_x, t_eval_default, sampling_mode)
                            score = torch.mean(self.anomaly_criterion(batch_x.float().to(self.device), recon), dim=-1).detach().cpu().numpy()
                    else:
                        recon = self._reconstruct_batch(batch_x, t_eval_default, sampling_mode)
                        score = torch.mean(self.anomaly_criterion(batch_x.float().to(self.device), recon), dim=-1).detach().cpu().numpy()
                    val_energy.append(score)

            val_energy = np.concatenate(val_energy, axis=0).reshape(-1)
            val_energy = np.array(val_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        test_data = []

        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
                if use_noise_score:
                    score = self._noise_score_batch_mode(batch_x, noise_score_mode)
                    if score is None:
                        recon = self._reconstruct_batch(batch_x, t_eval_default, sampling_mode)
                        score = torch.mean(self.anomaly_criterion(batch_x.float().to(self.device), recon), dim=-1).detach().cpu().numpy()
                else:
                    recon = self._reconstruct_batch(batch_x, t_eval_default, sampling_mode)
                    score = torch.mean(self.anomaly_criterion(batch_x.float().to(self.device), recon), dim=-1).detach().cpu().numpy()
                attens_energy.append(score)
                # window-level labels: mark window as anomalous if any timestep is non-zero
                batch_y_window = (batch_y.view(batch_y.shape[0], -1) != 0).any(dim=1).int().detach().cpu().numpy()
                test_labels.append(batch_y_window)
                batch_x = batch_x.detach().cpu().numpy()
                test_data.append(batch_x)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        if threshold_mode == 'test_only':
            combined_energy = test_energy
        elif threshold_mode == 'val_test':
            combined_energy = np.concatenate([val_energy, test_energy], axis=0)
        else:
            combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        if use_evt and combined_energy.size > 10:
            tail_len = max(5, int(len(combined_energy) * evt_tail_frac))
            tail = np.sort(combined_energy)[-tail_len:]
            tail_min = tail.min()
            tail_shift = tail - tail_min
            try:
                c, loc, scale = genpareto.fit(tail_shift, floc=0)
                threshold = tail_min + genpareto.ppf(evt_conf, c, loc=0, scale=scale)
                print('EVT threshold:', threshold, 'tail_frac:', evt_tail_frac, 'conf:', evt_conf)
            except Exception as e:
                threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
                print('EVT failed, fallback percentile. Err:', e)
        else:
            threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print('Threshold mode:', threshold_mode, 'Threshold :', threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        # Align lengths if windowing causes mismatch
        if pred.shape[0] != gt.shape[0]:
            min_len = min(pred.shape[0], gt.shape[0])
            print(f'Warning: pred/gt length mismatch {pred.shape[0]} vs {gt.shape[0]}, truncating to {min_len}')
            pred = pred[:min_len]
            gt = gt[:min_len]

        print('pred:   ', pred.shape)
        print('gt:     ', gt.shape)
        print('Positive labels after windowing:', gt.sum())

        gt = np.where(gt != 0, 1, 0)

        # (4) detection adjustment
        gt, pred = adjustment(gt, pred)

        pred = np.array(pred)
        gt = np.array(gt)
        print('pred: ', pred.shape)
        print('gt:   ', gt.shape)

        accuracy = accuracy_score(gt, pred)
        precision, recall, f_score, support = precision_recall_fscore_support(gt, pred, average='binary')
        print('Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}'.format(
            accuracy, precision,
            recall, f_score))

        f = open('result_anomaly_detection.txt', 'a')
        f.write(setting + '  \n')
        f.write('Accuracy : {:0.4f}, Precision : {:0.4f}, Recall : {:0.4f}, F-score : {:0.4f}'.format(
            accuracy, precision,
            recall, f_score))
        f.write('\n')
        f.write('\n')
        f.close()
        return
