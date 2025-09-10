# -- coding: utf-8 --
"""

run_optun.py
===============

Tập lệnh này thực hiện tối ưu siêu tham số cho các mô hình dự báo chuỗi thời gian bằng thư viện Optuna.
Sau khi tìm ra cấu hình tốt nhất, mô hình sẽ được huấn luyện lại, đánh giá và sinh kết quả trực quan hoá,
kèm theo giải thích mô hình sử dụng XAI (giải thích bằng gradient).

Các bước thực hiện
------------------
1. Khởi tạo và chạy quá trình tối ưu với Optuna.
2. Truy xuất trial tốt nhất và xây dựng toàn bộ tập tham số.
3. Sao chép checkpoint từ thư mục tạm sang thư mục kết quả chính thức.
4. Kiểm tra mô hình tốt nhất, vẽ biểu đồ loss huấn luyện/validation.
5. Lưu kết quả dự đoán và cấu hình của mô hình.
6. Thực thi phân tích giải thích mô hình (XAI).

Kết quả trả về
--------------
Không trả về giá trị, nhưng tạo ra các tệp kết quả bao gồm:
- Biểu đồ loss huấn luyện/validation
- Biểu đồ scatter giữa dự đoán và giá trị thực tế
- Biểu đồ giải thích từ SmoothGrad
- File YAML lưu lại cấu hình mô hình tốt nhất

Ví dụ sử dụng
-------------
$ python run_optun.py --model PatchTST --data_path mydata.csv --target Low,High

Ghi chú
------
- Cần có thư mục `./checkpoints`, `./results`, và `./test_results` trước khi chạy.
- Phải đảm bảo các module `exp`, `utils`, `data` có đầy đủ class và hàm tương ứng.
- Để dùng XAI, Captum phải được cài đặt.
"""
import argparse
import optuna
import yaml
import random
import torch
import numpy as np
from datetime import datetime
from exp.exp_main import Exp_Main
from utils.tools import set_seed
import os
from utils.params import suggest_model_specific
from utils.vis import *
import shutil
import time
import glob



import argparse
import optuna
import torch
from utils.params import suggest_model_specific
import os
import numpy as np
from exp.exp_main import Exp_Main
from utils.tools import set_seed



# ========================
# Objective function for Optuna
# ========================

def objective(trial):
    """
    Hàm mục tiêu cho quá trình tối ưu hóa Optuna.

    Hàm này thiết lập toàn bộ tập tham số cần thiết cho mô hình dự báo chuỗi thời gian,
    bao gồm cả các tham số được tìm kiếm (như dropout, learning rate, cấu trúc mô hình) 
    và các tham số cố định. Sau đó khởi tạo mô hình và huấn luyện nó, đánh giá hiệu quả 
    bằng loss trên tập validation.

    Tham số
    -------
    trial : optuna.trial.Trial
        Đối tượng trial được Optuna truyền vào để đề xuất các giá trị siêu tham số.

    Trả về
    -------
    float
        Giá trị loss dùng để tối thiểu hóa trong quá trình tìm kiếm. Nếu sử dụng k-fold thì là:
        `median_val_loss + 0.2 * var_val_loss`; nếu không thì là `val_loss` trên tập validation.

    Ghi chú
    -------
    - Hàm này hỗ trợ cả huấn luyện tiêu chuẩn và k-fold cross-validation.
    - Các tham số được đề xuất từ trial bao gồm: dropout, learning rate, activation, lradj, seq/label_len,...
    - Các kiểu augmentation (biến đổi dữ liệu) như jitter, scaling, dtw-warp,... cũng có thể được bật.
    - Dùng `suggest_model_specific()` để bổ sung các tham số riêng của từng loại mô hình (LSTM, PatchTST,...).
    """
    parser = argparse.ArgumentParser()

    
    # Learning
    parser.add_argument('--lradj', type=str, default=trial.suggest_categorical('lradj', ['type1', 'type3', 'type4']), help='adjust learning rate')
    parser.add_argument('--batch_size', type=int, default=trial.suggest_int('batch_size', 2,4), help='batch size of train input data')

    # basic config
    parser.add_argument('--task_name', type=str, required=True, default='long_term_forecast',help='task name, options:[long_term_forecast, short_term_forecast, imputation, classification, anomaly_detection]')
    parser.add_argument('--is_training', type=int, required=True, default=1, help='status')
    parser.add_argument('--model_id', type=str, default='test', help='model id')
    parser.add_argument('--model', type=str, required=True, default='Autoformer',help='model name, options: [Autoformer, Transformer, TimesNet]')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention')
    # data loader
    parser.add_argument('--data', type=str, default='ETTh1', help='dataset type')
    parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
    parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')
    parser.add_argument('--features', type=str, default='MS', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
    parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
    parser.add_argument('--freq', type=str, default='d',help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')
    # forecasting task
    parser.add_argument('--pred_len', type=int, default=1, help='prediction sequence length')
    parser.add_argument('--seasonal_patterns', type=str, default='Monthly', help='subset for M4')
    parser.add_argument('--inverse', action='store_true', help='inverse output data', default=True)
    parser.add_argument('--scale', default=True, type=bool, help='whether to scale the data')
    # inputation task
    parser.add_argument('--mask_rate', type=float, default=0.25, help='mask ratio')
    # anomaly detection task
    parser.add_argument('--anomaly_ratio', type=float, default=0.25, help='prior anomaly ratio (%%)')
    # model define
    parser.add_argument('--expand', type=int, default=2, help='expansion factor for Mamba')
    parser.add_argument('--d_conv', type=int, default=4, help='conv kernel size for Mamba')
    parser.add_argument('--top_k', type=int, default=5, help='for TimesBlock')
    # INPUTs
    parser.add_argument('--num_kernels', type=int, default=6, help='for Inception')
    parser.add_argument('--enc_in', type=int, default=2, help='encoder input size')
    parser.add_argument('--dec_in', type=int, default=2, help='decoder input size')
    parser.add_argument('--c_out', type=int, default=2, help='output size')
    parser.add_argument('--distil', action='store_false',help='whether to use distilling in encoder, using this argument means not using distilling',default=True)
    parser.add_argument('--embed', type=str, default='timeF',help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--channel_independence', type=int, default=1,help='0: channel dependence 1: channel independence for FreTS model')
    parser.add_argument('--decomp_method', type=str, default='moving_avg',help='method of series decompsition, only support moving_avg or dft_decomp')
    parser.add_argument('--down_sampling_layers', type=int, default=0, help='num of down sampling layers')
    parser.add_argument('--down_sampling_window', type=int, default=1, help='down sampling window size')
    parser.add_argument('--down_sampling_method', type=str, default=None,help='down sampling method, only support avg, max, conv')
    parser.add_argument('--kfold', type=int, default=0,help='Number of folds for k-fold cross validation (0 = no k-fold, >=2 = use k-fold)')

    # optimization
    parser.add_argument('--num_workers', type=int, default=10, help='data loader num workers')
    parser.add_argument('--itr', type=int, default=1, help='experiments times')
    parser.add_argument('--train_epochs', type=int, default=100, help='train epochs')
    parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
    parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
    parser.add_argument('--des', type=str, default='test', help='exp description')
    parser.add_argument('--loss', type=str, default='MSE', help='loss function')
    parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
    parser.add_argument('--remove', action='store_true',default=True)
    # GPU
    parser.add_argument('--use_gpu', type=bool, default=False, help='use gpu')
    parser.add_argument('--gpu', type=int, default=0, help='gpu')
    parser.add_argument('--gpu_type', type=str, default='cuda', help='gpu type')  # cuda or mps
    parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
    parser.add_argument('--devices', type=str, default='0,1,2,3', help='device ids of multile gpus')
    # de-stationary projector params
    parser.add_argument('--p_hidden_dims', type=int, nargs='+', default=[128, 128],help='hidden layer dimensions of projector (List)')
    parser.add_argument('--p_hidden_layers', type=int, default=2, help='number of hidden layers in projector')
    # metrics (dtw)
    parser.add_argument('--use_dtw', type=bool, default=True,help='the controller of using dtw metric (dtw is time consuming, not suggested unless necessary)')
    # Augmentation
    parser.add_argument('--augmentation_ratio', type=int, default=0, help="How many times to augment")
    parser.add_argument('--seed', type=int, default=2, help="Randomization seed")
    parser.add_argument('--jitter', default=False, action="store_true", help="Jitter preset augmentation")
    parser.add_argument('--scaling', default=False, action="store_true", help="Scaling preset augmentation")
    parser.add_argument('--permutation', default=False, action="store_true",help="Equal Length Permutation preset augmentation")
    parser.add_argument('--randompermutation', default=False, action="store_true",help="Random Length Permutation preset augmentation")
    parser.add_argument('--magwarp', default=False, action="store_true", help="Magnitude warp preset augmentation")
    parser.add_argument('--timewarp', default=False, action="store_true", help="Time warp preset augmentation")
    parser.add_argument('--windowslice', default=False, action="store_true", help="Window slice preset augmentation")
    parser.add_argument('--windowwarp', default=False, action="store_true", help="Window warp preset augmentation")
    parser.add_argument('--rotation', default=False, action="store_true", help="Rotation preset augmentation")
    parser.add_argument('--spawner', default=False, action="store_true", help="SPAWNER preset augmentation")
    parser.add_argument('--dtwwarp', default=False, action="store_true", help="DTW warp preset augmentation")
    parser.add_argument('--shapedtwwarp', default=False, action="store_true", help="Shape DTW warp preset augmentation")
    parser.add_argument('--wdba', default=False, action="store_true", help="Weighted DBA preset augmentation")
    parser.add_argument('--discdtw', default=False, action="store_true",help="Discrimitive DTW warp preset augmentation")
    parser.add_argument('--discsdtw', default=False, action="store_true",help="Discrimitive shapeDTW warp preset augmentation")
    parser.add_argument('--extra_tag', type=str, default="", help="Anything extra")




    # Fixed value
    args, _ = parser.parse_known_args()


    # Random Seed
    set_seed(args.seed + trial.number)
    specific_params=suggest_model_specific(trial, args.model)
    args.__dict__.update(specific_params)

    # Detect Device
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
        print('Using GPU')
    else:
        args.device = torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')
        print('Using CPU or MPS')

    if args.use_gpu and args.use_multi_gpu:
        args.devices = args.devices.replace(' ', '').split(',')
        args.device_ids = [int(id_) for id_ in args.devices]
        args.gpu = args.device_ids[0]
    
    # Create checkpoints directory
    exp = Exp_Main(args)
    setting = f"{args.model_id}_{args.data_path}_{args.model}_trial{trial.number}"

    # Create a unique directory for this trial
    if args.kfold > 0:
        checkpoint_tmp = os.path.join(args.checkpoints, "tmp")
        val_losses = exp.train_kfold(
            setting,
            checkpoint_base=checkpoint_tmp
        )
        best_fold = int(np.argmin(val_losses)) # can luu y sua lai lay trung binh (Ensamble learning)
        trial.set_user_attr("best_fold", best_fold)
        trial.set_user_attr("val_losses", val_losses)

        
        median_val_loss = np.median(val_losses)
        var_val_loss = np.var(val_losses)
        print(f"[Trial {trial.number}] Median Loss: {median_val_loss:.4f}, Var: {var_val_loss:.4f}")
        return median_val_loss + .2*var_val_loss
    

    else:
        checkpoint_tmp = os.path.join(args.checkpoints, "tmp")
        val_loss = exp.train_standard(
            setting, checkpoint_base=checkpoint_tmp
        )
        print(f"[Trial {trial.number}] Total Loss: {val_loss:.4f}")

    args_save_path = f'checkpoints/tmp/{setting}'
    torch.save(vars(args), os.path.join(args_save_path, 'best_args.pt'))


    return val_loss
    


# ========================
# Main function to run the optimization
# ========================
if __name__ == '__main__':
    # Set up the Optuna study
    sampler = optuna.samplers.TPESampler(
        seed=42,
        multivariate=True,
        group=True,
        n_startup_trials=5
    )

    pruner = optuna.pruners.MedianPruner(
        n_startup_trials=5, 
        n_warmup_steps=3)



    study = optuna.create_study(
        direction="minimize",
        sampler=sampler,
        pruner=pruner,
        study_name="optuna_best_search",
        load_if_exists=True
    )

    # NUM_TRIALS
    study.optimize(objective, n_trials=30, catch=(RuntimeError, ValueError, TypeError))
    print(">>>> FINISHED OPTIMIZATION <<<<")

    # Save the study
    best_trial = study.best_trial
    print("Best Trial Result:")
    print(study.best_trial)

   # Tìm tất cả file best.yaml
    best_trial = study.best_trial
    trial_no = best_trial.number

 
    # 1. Lấy tham số Optuna
    best_params = study.best_trial.params

    # 2. Load toàn bộ args đã lưu
    pattern = os.path.join('checkpoints/tmp', f'*trial{study.best_trial.number}', 'best_args.pt')
    matches = glob.glob(pattern)

    args_dict = torch.load(matches[0], weights_only=True)   # now a plain dict
    parser = argparse.ArgumentParser()
    args = parser.parse_args([])        # empty namespace
    args.__dict__.update(args_dict)

    # 3. Merge thêm siêu tham số Optuna
    args.__dict__.update(best_params)

    # Giờ args đã chứa đầy đủ
    print(args)


    if args.remove:

        tmp_dir = os.path.join(args.checkpoints, "tmp")
        setting_name = f'{args.model_id}_{args.data_path}_{args.model}_trial{best_trial.number}'
        tmp_setting_dir = os.path.join(tmp_dir, setting_name)
        final_setting_dir = os.path.join(args.checkpoints, setting_name)

        start_copy = time.time()
        if os.path.exists(tmp_setting_dir):
            print(f"[INFO] Copying full directory for best trial: {setting_name}")
            shutil.copytree(tmp_setting_dir, final_setting_dir, dirs_exist_ok=True)
            print(f"[INFO] Copied in {time.time() - start_copy:.2f}s")
        else:
            print(f"[WARNING] Temp directory {tmp_setting_dir} does not exist!")


    # copy and paste the best trial parameters
    start_rm = time.time()
    shutil.rmtree(tmp_dir, ignore_errors=True)
    print(f"[INFO] Removed tmp_dir in {time.time() - start_rm:.2f}s")



    
    if torch.cuda.is_available() and args.use_gpu:
        args.device = torch.device(f'cuda:{args.gpu}')
    else:
        args.device = torch.device('mps' if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available() else 'cpu')

    set_seed(args.seed + best_trial.number)



    # =========================
    # Run the experiment with the best parameters
    # =========================
    print("Run experiment with best params and plot result")
    Exp = Exp_Main
    exp = Exp(args)


    # Check if the checkpoint directory exists, if not, create it
    best = False

    if args.kfold > 0:
        if best:
            best_fold = best_trial.user_attrs["best_fold"]
            setting = f"{args.model_id}_{args.data_path}_{args.model}_trial{best_trial.number}"
            fold_dir = os.path.join(setting, f"fold{best_fold}")
            print(f"{'>'*50} Testing/Predicting {setting} {'<'*50}")
            train_losses = np.load(f"./checkpoints/{setting}/train_losses_{best_fold}.npy", allow_pickle=True)
            val_losses   = np.load(f"./checkpoints/{setting}/vali_losses_{best_fold}.npy", allow_pickle=True)
            plot_path = f"./test_results/test/{setting}/fold{best_fold}/loss.pdf"

        else:
            setting = f"{args.model_id}_{args.data_path}_{args.model}_trial{best_trial.number}"
            for fold in range(args.kfold):
                fold_dir = os.path.join(setting, f"fold{fold}")
                print(f"{'>'*50} Testing/Predicting {setting} Fold {fold} {'<'*50}")
                # Plot training/validation loss
                train_losses = np.load(f"./checkpoints/{setting}/train_losses_{fold}.npy", allow_pickle=True)
                val_losses   = np.load(f"./checkpoints/{setting}/vali_losses_{fold}.npy", allow_pickle=True)
                plot_path = f"./test_results/test/{setting}/fold{fold}/loss.pdf"
        exp.test(fold_dir, test=1)
        plot_loss(train_losses, val_losses, name=plot_path)

    else:
        setting = f'{args.model_id}_{args.data_path}_{args.model}_trial{best_trial.number}'

        print(f"{'>'*50} Testing/Predicting {setting} {'<'*50}")
        exp.test(setting, test=1)

        train_losses = np.load(f"./checkpoints/{setting}/train_losses.npy", allow_pickle=True)
        val_losses   = np.load(f"./checkpoints/{setting}/vali_losses.npy", allow_pickle=True)
        plot_loss(train_losses, val_losses, name=f"./test_results/test/{setting}/loss.pdf")


    # Clean GPU Cache
    if args.gpu_type == 'mps':
        torch.backends.mps.empty_cache()
    elif args.gpu_type == 'cuda':
        torch.cuda.empty_cache()

    print(f"{'>'*50} Finish Experiments {setting} {'<'*50}")


    # ========================
    # Plotting results
    # ========================

    # plot_input(dataset, save_path=f'./test_results/{setting}/INplot.pdf')
    # dataset, train_loader = exp._get_data(flag='train')
    # plot_metrics(study, './test_results/' + setting)
    plot_optimization_history(study, f'test_results/test/{setting}/OptHis.pdf')
    plot_hyperparameter_importance(study, f'test_results/test/{setting}/HyperImpo.pdf')
    # plot_input(args, f'test_results/test/{setting}/InputPlot.pdf')


    if args.kfold > 0:
        if best:
            setting = f'{args.model_id}_{args.data_path}_{args.model}_trial{best_trial.number}/fold{best_fold}'
            y_true = np.load(f'./results/{setting}/true.npy')
            y_pred = np.load(f'./results/{setting}/pred.npy')

            plot_scatter_truth_vs_pred(y_true, y_pred, save_path=f'./test_results/test/{setting}/PredScatter.pdf')
            # plot_residual_acf(y_true, y_pred, save_path=f'./test_results/{setting}/ACF.pdf')

            args_save_path = f'./test_results/test/{setting}/best.yaml'
            args_dict = vars(args)
            with open(args_save_path, 'w') as f:
                yaml.dump({
                    'args': vars(args)
                }, f, sort_keys=False)
        else:
            setting = f"{args.model_id}_{args.data_path}_{args.model}_trial{best_trial.number}"
            for fold in range(args.kfold):
                fold_dir = os.path.join(setting, f"fold{fold}")
                print(f"{'>'*50} Testing/Predicting {setting} Fold {fold} {'<'*50}")
                exp.test(fold_dir, test=1, plot=True)

                train_losses = np.load(f"./checkpoints/{setting}/train_losses_{fold}.npy", allow_pickle=True)
                val_losses   = np.load(f"./checkpoints/{setting}/vali_losses_{fold}.npy", allow_pickle=True)
                plot_path = f"./test_results/test/{setting}/fold{fold}/loss.pdf"
                plot_loss(train_losses, val_losses, name=plot_path)

                y_true = np.load(f'./results/{setting}/test/fold{fold}/true.npy')
                y_pred = np.load(f'./results/{setting}/test/fold{fold}/pred.npy')
                plot_scatter_truth_vs_pred(y_true, y_pred, save_path=f'./test_results/{setting}/fold{fold}/PredScatter.pdf')

                args_save_path = f'./test_results/test/{setting}/fold{fold}/best.yaml'
                with open(args_save_path, 'w') as f:
                    yaml.dump({'args': vars(args)}, f, sort_keys=False)
    else:
        setting = f'{args.model_id}_{args.data_path}_{args.model}_trial{best_trial.number}'

        y_true = np.load(f'./results/test/{setting}/true.npy')
        y_pred = np.load(f'./results/test/{setting}/pred.npy')

        plot_scatter_truth_vs_pred(y_true, y_pred, save_path=f'./test_results/test/{setting}/PredScatter.pdf')
        # plot_residual_acf(y_true, y_pred, save_path=f'./test_results/{setting}/ACF.pdf')

    # ========================
    # Predict steps and visualize
    # ========================
    # print("[PREDICT] Running n-step forecast and drawing visual...")

    # setting = f'{args.model_id}_{args.data_path}_{args.model}_trial{best_trial.number}'
    # exp.predict(setting=setting, load=True)


    # ========================
    # Run XAI for best model
    # ========================
    # print("[XAI] Start running SHAPTime explanation...")

    # from utils.xai import *
    # from data.data_factory import data_provider
    # import numpy as np
    # import torch
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # import seaborn as sns

    # # Load dữ liệu test
    # data_set, data_loader = data_provider(args, flag='test')
    # for batch_x, batch_y, batch_x_mark, batch_y_mark in data_loader:
    #     x_enc = batch_x.to(torch.float32)
    #     x_mark_enc = batch_x_mark.to(torch.float32)
    #     x_dec = batch_y[:, :args.label_len, :].to(torch.float32)
    #     x_mark_dec = batch_y_mark[:, :args.label_len, :].to(torch.float32)
    #     break

    # # Khởi tạo lại mô hình và load checkpoint
    # exp = Exp_Main(args)
    # model_raw = exp._build_model()
    # ckpt_path = f"./checkpoints/{setting}/fold{fold}.pth" if args.kfold > 0 else f"./checkpoints/{setting}/checkpoint.pth"
    # ckpt = torch.load(ckpt_path, map_location=args.device)
    # model_raw.load_state_dict(ckpt)
    # model_raw.to(args.device)

    # from captum.attr import NoiseTunnel, Saliency

    # print("[XAI] Running SmoothGrad with Captum...")

    # class WrapperForward(torch.nn.Module):
    #     def __init__(self, model, model_name, target_channel=0):
    #         super().__init__()
    #         self.model = model
    #         self.model_name = model_name.lower()
    #         self.target_channel = target_channel

    #     def forward(self, x, x_mark_enc, x_dec, x_mark_dec):
    #         out = self.model(x, x_mark_enc, x_dec, x_mark_dec)
    #         out = out[..., self.target_channel]
    #         return out



    # # Dùng lại model đã load
    # model_raw.eval()
    # wrapped_model = WrapperForward(model_raw, args.model, target_channel=0).to(args.device)

    # x = x_enc.detach().to(args.device).requires_grad_()  # [B, seq_len, C]
    # saliency = Saliency(wrapped_model)
    # noise_tunnel = NoiseTunnel(saliency)

    # attributions = noise_tunnel.attribute(
    #     x,
    #     nt_type='smoothgrad',
    #     stdevs=0.1,
    #     nt_samples=50,
    #     additional_forward_args=(x_mark_enc, x_dec, x_mark_dec),  # <- quan trọng!,
    #     target=0
    # )

    # # Chọn sample đầu tiên
    # attr = attributions[0].detach().cpu().numpy()  # [seq_len, C]

    # # Gói model cho channel 1 (High)
    # wrapped_model_high = WrapperForward(model_raw, args.model, target_channel=1).to(args.device)

    # # Tạo lại saliency và noise tunnel cho High
    # saliency_high = Saliency(wrapped_model_high)
    # noise_tunnel_high = NoiseTunnel(saliency_high)

    # # Attribution cho High
    # attributions_high = noise_tunnel_high.attribute(
    #     x,
    #     nt_type='smoothgrad',
    #     stdevs=0.1,
    #     nt_samples=50,
    #     additional_forward_args=(x_mark_enc, x_dec, x_mark_dec),
    #     target=0  
    # )

    # # Chọn sample đầu tiên
    # attr_high = attributions_high[0].detach().cpu().numpy()  # [seq_len, C]


    # # Gộp attribution lại: [2, seq_len]
    # attr_combined = np.stack([
    #     attr[:, 0],         # Attribution cho Low (channel 0)
    #     attr_high[:, 1]     # Attribution cho High (channel 1)
    # ], axis=0)

    # # Vẽ heatmap gộp
    # plt.figure(figsize=(12, 3))
    # sns.heatmap(attr_combined, cmap='viridis',
    #             xticklabels=[f"$t={{-{attr.shape[0] - 1 - i}}}$" for i in range(attr.shape[0])],
    #             yticklabels=["Low", "High"])
    # plt.xlabel("Time Steps")
    # plt.ylabel("Output Channels")
    # plt.title("SmoothGrad Attribution Heatmap")
    # plt.tight_layout()
    # plt.savefig(f"./test_results/test/{setting}/XAI.pdf")
