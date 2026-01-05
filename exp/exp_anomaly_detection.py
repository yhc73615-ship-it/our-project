from data_provider.data_factory import data_provider
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate, adjustment
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score, confusion_matrix
import torch.multiprocessing
from models import MTCL
from models.diffusion import ConditionalDiffusion
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

warnings.filterwarnings('ignore')


class Exp_Anomaly_Detection(Exp_Basic):
    def __init__(self, args):
        super(Exp_Anomaly_Detection, self).__init__(args)
        self.train_losses = []
        self.val_losses = []
        self.lambda_contrastive = args.lambda_contrastive
        self.use_diffusion = getattr(args, 'use_diffusion', False)
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
        model_optim = optim.Adam(params, lr=self.args.learning_rate)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        torch.cuda.empty_cache() 
        total_loss = []
        self.model.eval()
        if self.diffusion is not None:
            self.diffusion.eval()
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

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()

        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        model_optim = self._select_optimizer()
        criterion = self._select_criterion()
        
        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        scheduler = lr_scheduler.OneCycleLR(optimizer=model_optim,
                                            steps_per_epoch=train_steps,
                                            pct_start=self.args.pct_start,
                                            epochs=self.args.train_epochs,
                                            max_lr=self.args.learning_rate)
        

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            train_balance_loss = []
            train_contrastive_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, _, _, _) in enumerate(train_loader):
                iter_count += 1
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

                loss.backward()
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

        return self.model

    def test(self, setting, test=0):
        test_data, test_loader = self._get_data(flag='test')
        train_data, train_loader = self._get_data(flag='train')
        val_data, val_loader = self._get_data(flag='val')
        threshold_mode = getattr(self.args, 'threshold_mode', 'train_test')

        if test:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting, 'checkpoint.pth')))
            if self.diffusion is not None:
                diffusion_path = os.path.join('./checkpoints/' + setting, 'diffusion.pth')
                if os.path.exists(diffusion_path):
                    self.diffusion.load_state_dict(torch.load(diffusion_path, map_location=self.device))

        attens_energy = []
        folder_path = './test_results/' + setting + '/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        self.model.eval()
        if self.diffusion is not None:
            self.diffusion.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        t_eval_default = self.diffusion.timesteps // 2 if self.diffusion is not None else 0

        # (1) stastic on the train set
        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(train_loader):
                batch_x = batch_x.float().to(self.device)
                if self.use_diffusion:
                    outputs, _, _, cond_feats, x_norm = self.model(
                        batch_x, return_features=True, return_x_norm=True
                    )
                    t_eval = min(getattr(self.args, 'diffusion_eval_step', t_eval_default), t_eval_default)
                    t_tensor = torch.full((batch_x.size(0),), t_eval, device=self.device, dtype=torch.long)
                    noise = torch.randn_like(x_norm)
                    alpha_bar = self.diffusion.alpha_bars[t_eval]
                    z_t = torch.sqrt(alpha_bar) * x_norm + torch.sqrt(1 - alpha_bar) * noise
                    cond_flat = cond_feats.view(cond_feats.shape[0], cond_feats.shape[1], -1)
                    eps_pred = self.diffusion(z_t, t_tensor, cond_flat)
                    x0_pred = (z_t - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
                    recon = self.model.revin_layer(x0_pred, 'denorm')
                else:
                    outputs, _, _ = self.model(batch_x)
                    recon = outputs

                score = torch.mean(self.anomaly_criterion(batch_x, recon), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        val_energy = []
        if threshold_mode == 'val_test':
            with torch.no_grad():
                for i, (batch_x, batch_y, _, _) in enumerate(val_loader):
                    batch_x = batch_x.float().to(self.device)
                    if self.use_diffusion:
                        outputs, _, _, cond_feats, x_norm = self.model(
                            batch_x, return_features=True, return_x_norm=True
                        )
                        t_eval = min(getattr(self.args, 'diffusion_eval_step', t_eval_default), t_eval_default)
                        t_tensor = torch.full((batch_x.size(0),), t_eval, device=self.device, dtype=torch.long)
                        noise = torch.randn_like(x_norm)
                        alpha_bar = self.diffusion.alpha_bars[t_eval]
                        z_t = torch.sqrt(alpha_bar) * x_norm + torch.sqrt(1 - alpha_bar) * noise
                        cond_flat = cond_feats.view(cond_feats.shape[0], cond_feats.shape[1], -1)
                        eps_pred = self.diffusion(z_t, t_tensor, cond_flat)
                        x0_pred = (z_t - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
                        recon = self.model.revin_layer(x0_pred, 'denorm')
                    else:
                        outputs, _, _ = self.model(batch_x)
                        recon = outputs

                    score = torch.mean(self.anomaly_criterion(batch_x, recon), dim=-1)
                    score = score.detach().cpu().numpy()
                    val_energy.append(score)

            val_energy = np.concatenate(val_energy, axis=0).reshape(-1)
            val_energy = np.array(val_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        test_data = []

        with torch.no_grad():
            for i, (batch_x, batch_y, _, _) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                if self.use_diffusion:
                    outputs, _, _, cond_feats, x_norm = self.model(
                        batch_x, return_features=True, return_x_norm=True
                    )
                    t_eval = min(getattr(self.args, 'diffusion_eval_step', t_eval_default), t_eval_default)
                    t_tensor = torch.full((batch_x.size(0),), t_eval, device=self.device, dtype=torch.long)
                    noise = torch.randn_like(x_norm)
                    alpha_bar = self.diffusion.alpha_bars[t_eval]
                    z_t = torch.sqrt(alpha_bar) * x_norm + torch.sqrt(1 - alpha_bar) * noise
                    cond_flat = cond_feats.view(cond_feats.shape[0], cond_feats.shape[1], -1)
                    eps_pred = self.diffusion(z_t, t_tensor, cond_flat)
                    x0_pred = (z_t - torch.sqrt(1 - alpha_bar) * eps_pred) / torch.sqrt(alpha_bar)
                    recon = self.model.revin_layer(x0_pred, 'denorm')
                else:
                    outputs, _, _ = self.model(batch_x)
                    recon = outputs

                score = torch.mean(self.anomaly_criterion(batch_x, recon), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)
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
        threshold = np.percentile(combined_energy, 100 - self.args.anomaly_ratio)
        print('Threshold mode:', threshold_mode, 'Threshold :', threshold)

        # (3) evaluation on the test set
        pred = (test_energy > threshold).astype(int)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_labels = np.array(test_labels)
        gt = test_labels.astype(int)

        print('pred:   ', pred.shape)
        print('gt:     ', gt.shape)

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
