import logging
from sklearn.model_selection import TimeSeriesSplit
import logging
logging.basicConfig(format='%(asctime)s,%(msecs)03d %(levelname)-8s [%(filename)s:%(lineno)d] %(message)s',
    datefmt='%Y-%m-%d:%H:%M:%S',
    level=logging.INFO)
from data.data_factory import data_provider
from data.data_loader import Dataset_Custom
from torch.utils.data import DataLoader
from exp.exp_basic import Exp_Basic
from utils.tools import EarlyStopping, adjust_learning_rate
from utils.vis import plot_loss, visual_interval
from torch.nn import functional as F

from utils.metrics import metric
import torch
import torch.nn as nn
from torch import optim
import os
import time
import warnings
import numpy as np
from utils.dtw_metric import dtw, accelerated_dtw
from utils.augmentation import run_augmentation, run_augmentation_single
from torch.optim.lr_scheduler import ReduceLROnPlateau, OneCycleLR, CosineAnnealingWarmRestarts
from utils.losses import R2Loss
from sklearn.preprocessing import StandardScaler, MinMaxScaler


warnings.filterwarnings('ignore')

# This is the main experiment class for time series forecasting models.
class Exp_Main(Exp_Basic):
    """
    Main class for time series forecasting experiments.
    This class extends the Exp_Basic class and implements methods for building the model,
    getting data, selecting the optimizer and criterion, training, validating, and testing the model.
    It supports both k-fold cross-validation and standard training.
    Attributes:
        args: Argument parser containing model parameters.
        model_dict: Dictionary mapping model names to their respective classes.
        device: The device (CPU or GPU) on which the model will run.
        model: The initialized model instance.
    """
    def __init__(self, args):
        super(Exp_Main, self).__init__(args)

    def _build_model(self):
        model = self.model_dict[self.args.model].Model(self.args).float()

        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)
        return model

    def _get_data(self, flag):
        data_set, data_loader = data_provider(self.args, flag)
        return data_set, data_loader


    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate, weight_decay=1e-2)
        return model_optim

    def _select_criterion(self):
        criterion = nn.MSELoss()
        # criterion = R2Loss()    
        return criterion

    def _predict(self, batch_x, batch_y, batch_x_mark, batch_y_mark):
        # decoder input
        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
        # encoder - decoder

        def _run_model():
            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            if getattr(self.args, 'output_attention', False):
                outputs = outputs[0]
            return outputs

        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                outputs = _run_model()
        else:
            outputs = _run_model()

        f_dim = -2 if self.args.features == 'MS' else 0
        outputs = outputs[:, -self.args.pred_len:, f_dim:]
        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

        return outputs, batch_y
 

    def vali(self, vali_data, vali_loader, criterion):
        """
        Validate the model on the validation dataset.
        Args:
            vali_data: Validation dataset.
            vali_loader: DataLoader for the validation dataset.
            criterion: Loss function to evaluate the model's performance.
        Returns:
            avg_loss: Average validation loss.
        """
        total_loss = []
        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(vali_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float()

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():

                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -2 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        pred = outputs[:, -self.args.pred_len:, :]
                        true = batch_y[:, -self.args.pred_len:, :]
                        
                        loss_lower = criterion(pred[:, :, 0], true[:, :, 0])
                        loss_upper = criterion(pred[:, :, 1], true[:, :, 1])

                        consistency_penalty = (F.relu(pred[:, :, 0] - pred[:, :, 1]) ** 2).mean()
                        loss = (1/2)*(loss_lower + loss_upper) + consistency_penalty
                        total_loss.append(loss.item())
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -2 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    pred = outputs[:, -self.args.pred_len:, :]
                    true = batch_y[:, -self.args.pred_len:, :]
                    
                    loss_lower = criterion(pred[:, :, 0], true[:, :, 0])
                    loss_upper = criterion(pred[:, :, 1], true[:, :, 1])

                    consistency_penalty = (F.relu(pred[:, :, 0] - pred[:, :, 1]) ** 2).mean()
                    loss = (1/2)*(loss_lower + loss_upper) + consistency_penalty
                    total_loss.append(loss.item())


        avg_loss = np.average(total_loss)
        self.model.train()

        return avg_loss

    def train(self, setting):
        """
        Train the model based on the specified setting.
        Args:
            setting: A string indicating the training setting (e.g., 'kfold', 'standard').
            plot: A boolean indicating whether to plot the training and validation losses.
        Returns:
            val_losses: A list of validation losses for each fold if k-fold is used, or the final validation loss if standard training is used.
        """
        if self.args.kfold > 0:
            return self.train_kfold(setting)
        elif self.args.kfold == 0:
            return self.train_standard(setting)
        



    def train_standard(self, setting, checkpoint_base="./checkpoints/tmp"):
        """
        Train the model using standard training procedure.
        Args:
            setting: A string indicating the training setting (e.g., 'standard').
            plot: A boolean indicating whether to plot the training and validation losses.
            checkpoint_base: Base directory for saving checkpoints.
        Returns:
            val_loss: The final validation loss after training.
        """
        model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        logging.info(f"[MODEL] Trainable parameters: {model_params:,}")
        
        train_data, train_loader = self._get_data(flag='train')
        vali_data, vali_loader = self._get_data(flag='val')
        test_data, test_loader = self._get_data(flag='test')

        # print(f"[INFO] Dataset Shapes:")
        # print(f">> Train set  : {len(train_loader)} batches, each with shape {[batch_x.shape for (batch_x, _, _, _) in train_loader][0]}")
        # print(f">> Val set    : {len(vali_loader)} batches, each with shape {[batch_x.shape for (batch_x, _, _, _) in vali_loader][0]}")
        # print(f">> Test set   : {len(test_loader)} batches, each with shape {[batch_x.shape for (batch_x, _, _, _) in test_loader][0]}")
        # for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
            
        #         print("=" * 30 + f" Loader training data " + "=" * 30)
        #         print(f"\n>>> BATCH {i + 1}")

        #         # batch_x
        #         print(f"batch_x shape: {batch_x.shape}")
        #         print(batch_x.squeeze().cpu().numpy())

        #         # batch_y
        #         print(f"batch_y shape: {batch_y.shape}")
        #         print(batch_y.squeeze().cpu().numpy())

        #         # batch_x_mark
        #         print(f"batch_x_mark shape: {batch_x_mark.shape}")
        #         print(batch_x_mark.squeeze().cpu().numpy())

        #         # batch_y_mark
        #         print(f"batch_y_mark shape: {batch_y_mark.shape}")
        #         print(batch_y_mark.squeeze().cpu().numpy())
                
        #         if i >= 5:
        #             break
        
        time_now = time.time()

        train_steps = len(train_loader)

        if checkpoint_base:
            best_model_path = os.path.join(checkpoint_base, f'{setting}', f'checkpoint.pth')
            os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
        else:
            path = os.path.join(self.args.checkpoints, f'{setting}', f'checkpoint')
            os.makedirs(path, exist_ok=True)
            best_model_path = os.path.join(path, 'checkpoint.pth')
        
        # Early stopping to prevent overfitting
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)

        # Select optimizer and learning rate scheduler
        model_optim = self._select_optimizer()
        if self.args.lradj == 'type3':
            scheduler = OneCycleLR(
                model_optim,
                max_lr=self.args.learning_rate,
                epochs=self.args.train_epochs,
                steps_per_epoch=len(train_loader),
                pct_start=0.3,        # 30 % warm-up
                anneal_strategy='cos',
                final_div_factor=1e4, # lr cuối = max_lr / 1e4
            )
        elif self.args.lradj == 'type4':
            scheduler = ReduceLROnPlateau(
                model_optim,
                mode='min',          # vì loss
                factor=0.5,
                patience=3,
                min_lr=1e-7,
            )          
        else:
            scheduler = None


        criterion = self._select_criterion()

        if self.args.use_amp:
            amp_scaler = torch.cuda.amp.GradScaler()

        train_losses = []
        val_losses, val_mses = [], []

        for epoch in range(self.args.train_epochs):

            iter_count = 0
            train_loss = []

            self.model.train()
            epoch_time = time.time()
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                model_optim.zero_grad()
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        
                        outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                        f_dim = -2 if self.args.features == 'MS' else 0
                        outputs = outputs[:, -self.args.pred_len:, f_dim:]
                        batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                        pred = outputs[:, -self.args.pred_len:, :]
                        true = batch_y[:, -self.args.pred_len:, :]
                        
                        loss_lower = criterion(pred[:, :, 0], true[:, :, 0])
                        loss_upper = criterion(pred[:, :, 1], true[:, :, 1])

                        consistency_penalty = (F.relu(pred[:, :, 0] - pred[:, :, 1]) ** 2).mean()
                        loss = (1/2)*(loss_lower + loss_upper) + consistency_penalty
                        train_loss.append(loss.item())
                else:

                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                    f_dim = -2 if self.args.features == 'MS' else 0
                    outputs = outputs[:, -self.args.pred_len:, f_dim:]
                    batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                    pred = outputs[:, -self.args.pred_len:, :]
                    true = batch_y[:, -self.args.pred_len:, :]
                    
                    loss_lower = criterion(pred[:, :, 0], true[:, :, 0])
                    loss_upper = criterion(pred[:, :, 1], true[:, :, 1])

                    consistency_penalty = (F.relu(pred[:, :, 0] - pred[:, :, 1]) ** 2).mean()
                    loss = (1/2)*(loss_lower + loss_upper) + consistency_penalty
                    train_loss.append(loss.item())


                if (i + 1) % 100 == 0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()

                if self.args.use_amp:
                    amp_scaler.scale(loss).backward()
                    amp_scaler.step(model_optim)
                    amp_scaler.update()
                else:
                    # Backward pass and optimization
                    loss.backward()
                    model_optim.step()

                if self.args.lradj == 'type3':
                    adjust_learning_rate(model_optim, i, self.args, scheduler)

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            train_losses.append(train_loss)
            val_losses.append(vali_loss)
            np.save(os.path.join(checkpoint_base, f"{setting}", "train_losses.npy"), train_losses)
            np.save(os.path.join(checkpoint_base, f"{setting}", "vali_losses.npy"), val_losses)


            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss))

            early_stopping(vali_loss, self.model, best_model_path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.lradj in ['type1', 'type2', 'type3']:
                adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler, vali_loss)

        self.model.load_state_dict(torch.load(best_model_path))
        
        time_train = time.time() - time_now
        peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MiB
        logging.info(f"[TIME] Training time: {time_train:.2f}s, Peak GPU memory used: {peak_mem:.2f} MB")

        # return self.model
        return vali_loss




    def test(self, setting, test=0):
        """
        Test the model on the test dataset.
        Args:
            setting: A string indicating the testing setting (e.g., 'kfold', 'standard').
            test: An integer indicating whether to load a pre-trained model (1) or not (0).
            plot: A boolean indicating whether to plot the test results.
        Returns:
            None
        """
        start_time = time.time()
        test_data, test_loader = self._get_data(flag='test')


        if self.args.kfold:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting + ".pth")))
        else:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting + "/checkpoint.pth")))

        preds = []
        trues = []

        plot_path = './test_results/test/' + setting + '/'
        os.makedirs(plot_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(test_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)

                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                outputs, batch_y = self._predict(batch_x, batch_y, batch_x_mark, batch_y_mark)

                pred = outputs  # outputs.detach().cpu().numpy()  # .squeeze()
                true = batch_y  # batch_y.detach().cpu().numpy()  # .squeeze()

                preds.append(pred)
                trues.append(true)
                
                if test == 1:
                    shape = outputs.shape  # (B, T, 2)
                    # input = batch_x[0, :, -2:].detach().cpu().numpy()
                    input = batch_x[0].detach().cpu().numpy()  # [seq_len, C]

                    if self.args.scale and self.args.inverse:
                        input = test_data.inverse_transform(input.reshape(-1, 2)).reshape(input.shape)
                        # inverse prediction
                        scaled_pred = np.stack([outputs[:, :, 0], outputs[:, :, 1]], axis=-1).reshape(-1, 2)
                        outputs = test_data.inverse_transform(scaled_pred).reshape(shape)

                        # inverse ground truth
                        scaled_true = np.stack([batch_y[:, :, 0], batch_y[:, :, 1]], axis=-1).reshape(-1, 2)
                        batch_y = test_data.inverse_transform(scaled_true).reshape(shape)

                    
                    true = batch_y #.detach().cpu().numpy()
                    pred = outputs #.detach().cpu().numpy()

                    input_low, input_high = input[:, -2], input[:, -1]
                    true_low, true_high = true[0, :, -2], true[0, :, -1]
                    pred_low, pred_high = pred[0, :, -2], pred[0, :, -1]

                    gt = np.stack([np.concatenate((input_low, true_low)),
                                   np.concatenate((input_high, true_high))], axis=1)
                    pd = np.stack([np.concatenate((input_low, pred_low)),
                                   np.concatenate((input_high, pred_high))], axis=1)

                    visual_interval(gt, pd, os.path.join(plot_path, f'{i}.pdf'), input_len=self.args.seq_len)


        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # dtw calculation
        if self.args.use_dtw:
            dtw_list = []
            manhattan_distance = lambda x, y: np.abs(x - y)
            for i in range(preds.shape[0]):
                x = preds[i].reshape(-1, 1)
                y = trues[i].reshape(-1, 1)
                if i % 100 == 0:
                    print("calculating dtw iter:", i)
                d, _, _, _ = accelerated_dtw(x, y, dist=manhattan_distance)
                dtw_list.append(d)
            dtw = np.array(dtw_list).mean()
        else:
            dtw = 'Not calculated'
        
        # Save results
        folder_path = './results/test/' + setting + '/'
        os.makedirs(folder_path, exist_ok=True)
        mae0, mse0, rmse0, mape0, mspe0, nse0 = metric(preds[...,0], trues[...,0])
        print(f'TESTING LOWER: mse:{mse0:.4f} | mae:{mae0:.4f} | dtw:{dtw} | nse:{nse0:.4f}')
        
        mae1, mse1, rmse1, mape1, mspe1, nse1 = metric(preds[...,1], trues[...,1])
        print(f'TESTING HIGHER mse:{mse1:.4f} | mae:{mae1:.4f} | dtw:{dtw} | nse:{nse1:.4f}')
        
        np.save(os.path.join(folder_path, 'metrics_Low.npy'), np.array([mae0, mse0, rmse0, mape0, mspe0, nse0]))
        np.save(os.path.join(folder_path, 'metrics_High.npy'), np.array([mae1, mse1, rmse1, mape1, mspe1, nse1]))

        np.save(os.path.join(folder_path, 'pred.npy'), preds)
        np.save(os.path.join(folder_path, 'true.npy'), trues)

        elapsed_time = time.time() - start_time
        print(f"[TIME] Testing time: {elapsed_time:.2f} seconds")

        for sample_idx in range(preds.shape[0]):
            print("TimeStep | True_Low  | Pred_Low | True_High | Pred_High")
            for t in range(preds.shape[1]):
                true_low  = trues[sample_idx, t, 0]
                pred_low  = preds[sample_idx, t, 0]
                true_high = trues[sample_idx, t, 1]
                pred_high = preds[sample_idx, t, 1]
                print(f"{t:>8} | {true_low:>9.4f} | {pred_low:>8.4f} | {true_high:>10.4f} | {pred_high:>9.4f}")

        return
    
    
    

    def predict(self,  setting, load=False):
        """
        Test the model on the test dataset.
        Args:
            setting: A string indicating the testing setting (e.g., 'kfold', 'standard').
            test: An integer indicating whether to load a pre-trained model (1) or not (0).
            plot: A boolean indicating whether to plot the test results.
        Returns:
            None
        """
        start_time = time.time()
        pred_data, pred_loader = self._get_data(flag='pred')


        if self.args.kfold:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting + ".pth")))
        else:
            print('loading model')
            self.model.load_state_dict(torch.load(os.path.join('./checkpoints/' + setting + "/checkpoint.pth")))

        preds = []

        plot_path = './test_results/pred/' + setting + '/'
        os.makedirs(plot_path, exist_ok=True)

        self.model.eval()
        with torch.no_grad():
            for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(pred_loader):
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                batch_x_mark = batch_x_mark.float().to(self.device)
                batch_y_mark = batch_y_mark.float().to(self.device)

                # decoder input
                dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)
                # encoder - decoder
                if self.args.use_amp:
                    with torch.cuda.amp.autocast():
                        if self.args.output_attention:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)[0]
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                outputs = outputs.detach().cpu().numpy()

                if pred_data.scale and self.args.inverse:
                    shape = outputs.shape
                    outputs = pred_data.inverse_transform(outputs.squeeze(0)).reshape(shape)
                preds.append(outputs)

        preds = np.array(preds)                          # list -> ndarray
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])   # (N, T, 2)

        for i in range(preds.shape[0]):
            print(f"\n[Sample {i}] TimeStep | Pred_Low | Pred_High")
            for t in range(preds.shape[1]):
                pred_low  = preds[i, t, 0]
                pred_high = preds[i, t, 1]
                print(f"{t:>11} | {pred_low:>8.4f} | {pred_high:>9.4f}")

                visual_interval(
                    true=None,
                    preds=preds,                # truyền cả mảng (N,T,2)
                    sample_idx=i,               # chọn sample cần vẽ
                    input_len=self.args.seq_len,
                    name=os.path.join(plot_path, f'{i}.pdf'),
                )

        return










    def train_kfold(self, setting, checkpoint_base="./checkpoints/tmp"):
            """
            Train the model using k-fold cross-validation.
            Args:
                setting: A string indicating the training setting (e.g., 'kfold').
                plot: A boolean indicating whether to plot the training and validation losses.
                checkpoint_base: Base directory for saving checkpoints.
            Returns:
                val_losses: A list of validation losses for each fold.
            """
            model_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            logging.info(f"[MODEL] Trainable parameters: {model_params:,}")
            
            # Get training, validation, and test data loaders
            train_data_full, train_loader_full = self._get_data(flag='train')
            

            # Chia k-fold từ 80% training
            dataset = train_data_full
            indices = np.arange(len(dataset))
            tscv = TimeSeriesSplit(n_splits=self.args.kfold)

            val_losses = []

            for fold, (trainval_idx, _) in enumerate(tscv.split(train_data_full)):

                import pandas as pd

                df_fold = train_data_full.df_target.iloc[trainval_idx].reset_index(drop=True)
                n_val = int(len(df_fold) * 0.2)
                val_df = df_fold.iloc[-n_val:].reset_index(drop=True)
                subtrain_df = df_fold.iloc[:-n_val].reset_index(drop=True)

                logging.info(f"[Fold {fold}] Total: {len(df_fold)} → Subtrain: {len(subtrain_df)}, Val: {len(val_df)}")

                # Kiểm tra số mẫu hợp lệ
                min_len = self.args.seq_len + self.args.pred_len
                if len(subtrain_df) < min_len or len(val_df) < min_len:
                    logging.warning(f"[Fold {fold}] Dataset too small: len(train)={len(subtrain_df)}, len(val)={len(val_df)}; seq+pred={min_len}")
                    continue
                
                self.scaler = StandardScaler()
                
                target_cols = self.args.target
                if isinstance(target_cols, str) and ',' in target_cols:
                    target_cols = [col.strip() for col in target_cols.split(',')]

                self.scaler.fit(subtrain_df[target_cols])

                subtrain_scaled = subtrain_df.copy()
                subtrain_scaled[target_cols] = self.scaler.transform(subtrain_df[target_cols])

                val_scaled = val_df.copy()
                val_scaled[target_cols] = self.scaler.transform(val_df[target_cols])


                train_dataset = Dataset_Custom(
                    args=self.args, root_path=None, flag='train',
                    size=(self.args.seq_len, self.args.label_len, self.args.pred_len),
                    features=self.args.features, data_path=None, target=self.args.target,
                    scale=False, freq=self.args.freq, embed=self.args.embed, kfold=self.args.kfold,
                    dataframe=subtrain_scaled
                )

                val_dataset = Dataset_Custom(
                    args=self.args, root_path=None, flag='train',
                    size=(self.args.seq_len, self.args.label_len, self.args.pred_len),
                    features=self.args.features, data_path=None, target=self.args.target,
                    scale=False, freq=self.args.freq, embed=self.args.embed, kfold=self.args.kfold,
                    dataframe = val_scaled
                )

                train_loader = DataLoader(train_dataset, batch_size=self.args.batch_size, shuffle=False)
                vali_loader = DataLoader(val_dataset, batch_size=self.args.batch_size, shuffle=False)

                # print(f"[INFO] Dataset Shapes:")
                # print(f"  ➤ Train set  : {len(train_loader)} batches, each with shape {[batch_x.shape for (batch_x, _, _, _) in train_loader][0]}")
                # print(f"  ➤ Val set    : {len(vali_loader)} batches, each with shape {[batch_x.shape for (batch_x, _, _, _) in vali_loader][0]}")
                # print(f"  ➤ Test set   : {len(test_loader)} batches, each with shape {[batch_x.shape for (batch_x, _, _, _) in test_loader][0]}")

                time_now = time.time()

                train_steps = len(train_loader)

                if checkpoint_base:
                    best_model_path = os.path.join(checkpoint_base, f'{setting}', f'fold{fold}.pth')
                    os.makedirs(os.path.dirname(best_model_path), exist_ok=True)
                else:
                    path = os.path.join(self.args.checkpoints, f'{setting}', f'fold{fold}')
                    os.makedirs(path, exist_ok=True)
                    best_model_path = os.path.join(path, 'checkpoint.pth')
                
                early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
                
                
                model_optim = self._select_optimizer()
                if self.args.lradj == 'type3':
                    scheduler = OneCycleLR(
                        model_optim,
                        max_lr=1e-3, 
                        steps_per_epoch=len(train_loader), 
                        epochs=self.args.train_epochs,
                        pct_start=0.3,
                        anneal_strategy='cos'
                    )
                elif self.args.lradj == 'type4':
                    scheduler = CosineAnnealingWarmRestarts(model_optim, T_0=10, T_mult=2, eta_min=1e-6)

                else:
                    scheduler = None

                criterion = self._select_criterion()

                if self.args.use_amp:
                    amp_scaler = torch.cuda.amp.GradScaler()

                train_losses_this_fold = []
                val_losses_this_fold = []

                for epoch in range(self.args.train_epochs):
                    fold_start = time.time()

                    iter_count = 0
                    train_loss = []

                    self.model.train()
                    epoch_time = time.time()
                    for i, (batch_x, batch_y, batch_x_mark, batch_y_mark) in enumerate(train_loader):
                        iter_count += 1
                        model_optim.zero_grad()
                        batch_x = batch_x.float().to(self.device)
                        batch_y = batch_y.float().to(self.device)
                        batch_x_mark = batch_x_mark.float().to(self.device)
                        batch_y_mark = batch_y_mark.float().to(self.device)

                        # decoder input
                        dec_inp = torch.zeros_like(batch_y[:, -self.args.pred_len:, :]).float()
                        dec_inp = torch.cat([batch_y[:, :self.args.label_len, :], dec_inp], dim=1).float().to(self.device)

                        # encoder - decoder
                        if self.args.use_amp:
                            with torch.cuda.amp.autocast():

                                outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                                f_dim = -2 if self.args.features == 'MS' else 0
                                outputs = outputs[:, -self.args.pred_len:, f_dim:]
                                batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                                pred = outputs[:, -self.args.pred_len:, :]
                                true = batch_y[:, -self.args.pred_len:, :]
                                
                                loss_lower = criterion(pred[:, :, 0], true[:, :, 0])
                                loss_upper = criterion(pred[:, :, 1], true[:, :, 1])
                                # loss = loss_lower + loss_upper
                                # consistency_penalty = torch.clamp(pred[:, :, 0] - pred[:, :, 1], min=0) ** 2
                                consistency_penalty = F.relu(pred[:, :, 0] - pred[:, :, 1]) ** 2
                                consistency_loss = consistency_penalty.mean()
                                loss = loss_lower + loss_upper + consistency_loss
                                train_loss.append(loss.item())
                        else:
                            outputs = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

                            f_dim = -2 if self.args.features == 'MS' else 0
                            outputs = outputs[:, -self.args.pred_len:, f_dim:]
                            batch_y = batch_y[:, -self.args.pred_len:, f_dim:].to(self.device)

                            pred = outputs[:, -self.args.pred_len:, :]
                            true = batch_y[:, -self.args.pred_len:, :]
                            
                            loss_lower = criterion(pred[:, :, 0], true[:, :, 0])
                            loss_upper = criterion(pred[:, :, 1], true[:, :, 1])
                            # loss = loss_lower + loss_upper
                            consistency_penalty = torch.clamp(pred[:, :, 0] - pred[:, :, 1], min=0) ** 2
                            consistency_loss = consistency_penalty.mean()
                            loss = loss_lower + loss_upper + 10*consistency_loss
                            train_loss.append(loss.item())

                        if (i + 1) % 100 == 0:
                            print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                            speed = (time.time() - time_now) / iter_count
                            left_time = speed * ((self.args.train_epochs - epoch) * train_steps - i)
                            print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                            iter_count = 0
                            time_now = time.time()

                        if self.args.use_amp:
                            amp_scaler.scale(loss).backward()
                            amp_scaler.step(model_optim)
                            amp_scaler.update()
                        else:
                            loss.backward()
                            model_optim.step()

                    print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
                    train_loss = np.average(train_loss)
                    vali_loss = self.vali(None, vali_loader, criterion)
                    
                    train_losses_this_fold.append(train_loss)
                    val_losses_this_fold.append(vali_loss)
                    print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} | Vali Loss: {3:.7f}".format(
                        epoch + 1, train_steps, train_loss, vali_loss))

                    early_stopping(vali_loss, self.model, best_model_path)

                    if early_stopping.early_stop:
                        print("Early stopping")
                        break

                    adjust_learning_rate(model_optim, epoch + 1, self.args, scheduler, vali_loss)
                
                self.model.load_state_dict(torch.load(best_model_path))
                np.save(os.path.join(checkpoint_base, f"{setting}", f"train_losses_{fold}.npy"), train_losses_this_fold)
                np.save(os.path.join(checkpoint_base, f"{setting}", f"vali_losses_{fold}.npy"), val_losses_this_fold)

                fold_time = time.time() - fold_start
                peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024  # MiB
                logging.info(f"[TIME] Fold {fold+1} training time: {fold_time:.2f}s, Peak GPU memory used: {peak_mem:.2f} MB")

                val_losses.append(val_losses_this_fold[-1])

            return val_losses