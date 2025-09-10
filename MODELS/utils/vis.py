import optuna.visualization.matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np
import matplotlib.cm as cm
import torch
from statsmodels.graphics.tsaplots import plot_acf


def set_plot_style():
    plt.rcParams.update({
        'font.size': 15,
        'font.family': 'serif',
        'mathtext.fontset': 'cm', 
        'axes.linewidth': 0.8,
        'lines.linewidth': 2,
        'lines.markersize': 6,
        'legend.frameon': False,
        'legend.fontsize': 13,
        'axes.grid': True,
        'grid.alpha': 0.4,
        'xtick.direction': 'in',
        'ytick.direction': 'in',
        'xtick.major.size': 4,
        'ytick.major.size': 4,
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.05,
        'figure.facecolor': 'white',     
    })

def plot_metrics(study, save_dir):
    set_plot_style()
    trials = study.trials_dataframe()

    best_trial_idx = trials['value'].idxmin()
    best_val = trials['value'].min()

    plt.figure(figsize=(8, 5))
    plt.plot(trials['value'], marker='o', label='Validation Loss (MSE)', color=cm.viridis(0.5))
    plt.scatter(best_trial_idx, best_val, color=cm.viridis(0.9), zorder=5, label=...)
    plt.axvline(x=best_trial_idx, color=cm.viridis(0.9), linestyle='--', alpha=0.6)


    plt.xlabel('Trial')
    plt.ylabel('Validation Loss (MSE)')
    plt.title('Validation Loss per Trial')
    plt.legend()
    plt.tight_layout()

    os.makedirs(save_dir, exist_ok=True)
    plt.savefig(os.path.join(save_dir, 'ValLoss.pdf'))
    plt.close()

    for metric in ['user_attrs_dtw', 'user_attrs_mae']:
        if metric in trials.columns:
            plt.figure()
            plt.plot(trials[metric], marker='o')
            plt.xlabel('Trial')
            plt.ylabel(metric.upper())
            plt.title(f'{metric.upper()} per Trial')
            plt.grid()
            plt.tight_layout()
            plt.savefig(os.path.join(save_dir, f'{metric.upper()}_trial.pdf'))
            plt.close()




def plot_residual_acf(y_true, y_pred, save_path='./ResidualACF.pdf', lags=48):
    set_plot_style()
    residuals = y_true.flatten() - y_pred.flatten()
    plt.figure(figsize=(10,5))
    plot_acf(residuals, lags=lags)
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# ========================== USING ====================================

def visual_interval(true=None, preds=None, name='./pic/test_interval.pdf', input_len=None, sample_idx: int | None = None):
    """
    Vẽ khoảng (lower, upper) cho ground-truth (nếu có) và prediction.
    - true:  (T, 2) hoặc (N, T, 2) hoặc None
    - preds: (T, 2) hoặc (N, T, 2) hoặc None
    - input_len: vị trí vạch ngăn giữa phần input và phần dự báo (tính theo trục x, 0-based)
    - sample_idx: nếu mảng 3D thì chọn sample để vẽ; nếu None và dữ liệu 3D -> lấy sample 0
    """
    set_plot_style()
    plt.figure(figsize=(15, 6))

    def _pick_sample(x):
        if x is None:
            return None
        x = np.asarray(x)
        if x.ndim == 3:
            idx = 0 if sample_idx is None else int(sample_idx)
            x = x[idx]
        return x  # shape (T, 2)

    true_  = _pick_sample(true)
    preds_ = _pick_sample(preds)

    # Vẽ TRUE nếu có
    if true_ is not None:
        true_lower = true_[..., 0]
        true_upper = true_[..., 1]
        # mask NaN nếu có
        m = np.isfinite(true_lower) & np.isfinite(true_upper)
        x_idx = np.arange(len(true_lower))[m]
        plt.plot(x_idx, true_lower[m], label='GroundTruth Lower', color='blue', linewidth=2)
        plt.plot(x_idx, true_upper[m], label='GroundTruth Upper', color='blue', linestyle='--', linewidth=2)
        plt.fill_between(x_idx, true_lower[m], true_upper[m], color='blue', alpha=0.2)

    # Vẽ PREDS nếu có
    if preds_ is not None:
        pred_lower = preds_[..., 0]
        pred_upper = preds_[..., 1]
        m = np.isfinite(pred_lower) & np.isfinite(pred_upper)
        x_idx = np.arange(len(pred_lower))[m]
        plt.plot(x_idx, pred_lower[m], label='Prediction Lower', color='orange', linewidth=2)
        plt.plot(x_idx, pred_upper[m], label='Prediction Upper', color='orange', linestyle='--', linewidth=2)
        plt.fill_between(x_idx, pred_lower[m], pred_upper[m], color='orange', alpha=0.2)

    # Vạch input_len (hỗ trợ cả khi không có true)
    series_len = None
    if preds_ is not None:
        series_len = len(preds_)
    elif true_ is not None:
        series_len = len(true_)

    if input_len is not None and series_len is not None and input_len < series_len:
        plt.axvline(input_len - 1, color='gray', linestyle='--', alpha=0.7)
        ylim = plt.gca().get_ylim()
        plt.text(input_len - 2, ylim[1] * 0.98, 'Input', ha='right', va='top', color='gray')
        plt.text(input_len + 2, ylim[1] * 0.98, 'Prediction', ha='left', va='top', color='gray')

    plt.xlabel("Time Steps")
    plt.ylabel("Target Values")
    plt.legend()
    plt.grid(True)
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.savefig(name, bbox_inches='tight')
    plt.close()






def plot_loss(train_losses=None, val_losses=None, name='./pic/loss.pdf'):
    set_plot_style()
    plt.figure(figsize=(10, 6))
    epochs = range(1, len(train_losses) + 1)
    plt.semilogy(epochs, train_losses, label='Train Loss (log)', color=cm.viridis(0.1), linestyle='--', linewidth=3)
    if val_losses is not None:
        plt.semilogy(epochs, val_losses, label='Validation Loss (log)', color=cm.viridis(0.8), linewidth=4)

    plt.xlabel('Epochs')
    plt.ylabel('Loss (MSE)')
    plt.legend()
    os.makedirs(os.path.dirname(name), exist_ok=True)
    plt.savefig(name, bbox_inches='tight')
    plt.close()



def plot_optimization_history(study, save_path):
    """
    Vẽ và lưu biểu đồ lịch sử tối ưu của Optuna.

    Parameters
    ----------
    study : optuna.study.Study
        Đối tượng study sau khi optimize.
    save_path : str
        Đường dẫn file ảnh đầu ra (có đuôi .png, .pdf, …).
    """
    set_plot_style()
    fig = optuna.visualization.matplotlib.plot_optimization_history(study)
    # Lưu figure
    fig.figure.savefig(save_path, bbox_inches='tight')
    plt.close(fig.figure)

def plot_hyperparameter_importance(study, save_path):
    set_plot_style()
    plt.figure(figsize=(15, 6))
    fig = optuna.visualization.matplotlib.plot_param_importances(study)
    fig.figure.savefig(save_path, bbox_inches='tight')
    plt.close(fig.figure)


def plot_scatter_truth_vs_pred(y_true, y_pred, save_path='./PredScatter.pdf'):
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import r2_score
    from utils.metrics import metric
    import matplotlib.pyplot as plt
    import numpy as np
    import matplotlib.cm as cm

    set_plot_style()

    # Tính metrics tổng thể
    mae0, mse0, rmse0, mape0, mspe0, nse0 = metric(y_pred[..., 0], y_true[..., 0])
    mae1, mse1, rmse1, mape1, mspe1, nse1 =  metric(y_pred[..., 1], y_true[..., 1])

    y_true = np.array(y_true)
    y_pred = np.array(y_pred)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    titles = ['Low', 'High']
    components = [0, 1]

    for ax, comp_idx, title in zip(axes, components, titles):
        y_true_comp = y_true[..., comp_idx]
        y_pred_comp = y_pred[..., comp_idx]

        y_true_flat = y_true_comp.flatten().reshape(-1, 1)
        y_pred_flat = y_pred_comp.flatten().reshape(-1, 1)

        # Hồi quy tuyến tính
        reg = LinearRegression().fit(y_true_flat, y_pred_flat)
        y_fit = reg.predict(y_true_flat)
        r2 = r2_score(y_true_flat, y_pred_flat)

        # Màu theo batch
        num_batches = y_true.shape[0]
        colors = cm.get_cmap('viridis', num_batches)

        for i in range(num_batches):
            ax.scatter(
                y_true_comp[i].flatten(),
                y_pred_comp[i].flatten(),
                alpha=0.5,
                color=cm.viridis(0.5)
                # label=f'Batch {i+1}'
            )

        min_val = min(y_true_comp.min(), y_pred_comp.min())
        max_val = max(y_true_comp.max(), y_pred_comp.max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', label='Ideal')

        ax.plot(y_true_flat, y_fit, color='black', linewidth=2, label=f'Fit (R²={r2:.4f})')

        ax.set_xlabel('Observed')
        ax.set_ylabel('Forecasted')
        ax.set_title(f'{title} Values')
        ax.legend(loc='lower right', fontsize=8)

        # Ghi metric trên subplot "Low"
        if comp_idx == 0:
            ax.annotate(
                f'MAE:  {mae0:.4f}\n'
                f'MSE:  {mse0:.4f}\n'
                f'RMSE: {rmse0:.4f}\n'
                f'MAPE: {mape0:.2f}%\n'
                f'MSPE: {mspe0:.2f}%\n'
                f'NSE:  {nse0:.4f}\n',
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top',
                fontsize=10,
            )
                # Ghi metric trên subplot "Low"
        if comp_idx == 1:
            ax.annotate(
                f'MAE:  {mae1:.4f}\n'
                f'MSE:  {mse1:.4f}\n'
                f'RMSE: {rmse1:.4f}\n'
                f'MAPE: {mape1:.2f}%\n'
                f'MSPE: {mspe1:.2f}%\n'
                f'NSE:  {nse1:.4f}\n',
                xy=(0.05, 0.95), xycoords='axes fraction',
                ha='left', va='top',
                fontsize=10,
            )

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_input(args, save_path='./figs/input_scaled_plot.pdf'):
    import matplotlib.pyplot as plt
    import pandas as pd
    import numpy as np
    from data.data_factory import data_provider
    import os

    # Load datasets
    dataset_train, _ = data_provider(args, flag='train')
    dataset_val, _ = data_provider(args, flag='val')
    dataset_test, _ = data_provider(args, flag='test')



    # Read full CSV
    file_path = os.path.join(args.root_path, args.data_path)
    df_raw = pd.read_csv(file_path)
    low_col, high_col = dataset_train.target

    low_raw = df_raw[low_col].values
    high_raw = df_raw[high_col].values
    n = len(df_raw)

    # Borders
    num_train = dataset_train.num_train
    num_val = dataset_val.num_vali
    train_border = num_train
    val_border = train_border + num_val
    test_border = n

    # Split
    low_train, high_train = low_raw[:train_border], high_raw[:train_border]
    low_val, high_val = low_raw[train_border:val_border], high_raw[train_border:val_border]
    low_test, high_test = low_raw[val_border:], high_raw[val_border:]


#    # Tô màu vùng mẫu dự báo (ở cuối tập train)
#     seq_len = args.seq_len
#     label_len = args.label_len
#     pred_len = args.pred_len

#     # Bắt đầu sample ngay trước khi kết thúc tập train
#     sample_start = train_border - (seq_len + label_len + pred_len)
#     seq_end = sample_start + seq_len
#     label_end = seq_end + label_len
#     pred_end = label_end + pred_len


    # Vẽ
    plt.figure(figsize=(18, 6))

    # --- Gốc ---
    plt.subplot(1, 2, 1)
    plt.plot(low_raw, label='Low (raw)', color='blue')
    plt.plot(high_raw, label='High (raw)', color='red')
    plt.axvline(train_border, color='gray', linestyle='--')
    plt.axvline(val_border, color='gray', linestyle='--')
    plt.title("Original Low & High")
    plt.text(train_border / 2, max(high_raw)*0.95, 'Train', ha='center', color='gray')
    plt.text((train_border + val_border) / 2, max(high_raw)*0.95, 'Val', ha='center', color='gray')
    plt.text((val_border + test_border) / 2, max(high_raw)*0.95, 'Test', ha='center', color='gray')
    plt.legend()

    # --- Scaled ---
    plt.subplot(1, 2, 2)

        # Scaler
    if args.scale:
        scaler = dataset_train.scaler
        scaler.fit(low_train, high_train)

        low_scaled_train, high_scaled_train = scaler.transform(low_train, high_train)
        low_scaled_val, high_scaled_val = scaler.transform(low_val, high_val)
        low_scaled_test, high_scaled_test = scaler.transform(low_test, high_test)

        low_scaled_full = np.concatenate([low_scaled_train, low_scaled_val, low_scaled_test])
        high_scaled_full = np.concatenate([high_scaled_train, high_scaled_val, high_scaled_test])

        plt.plot(low_scaled_full, label='Low (scaled)', color='blue')
        plt.plot(high_scaled_full, label='High (scaled)', color='red')
        plt.axvline(train_border, color='gray', linestyle='--')
        plt.axvline(val_border, color='gray', linestyle='--')
        plt.title("Scaled Low & High")
        plt.text(train_border / 2, 1.0, 'Train', ha='center', color='gray')
        plt.text((train_border + val_border) / 2, 1.0, 'Val', ha='center', color='gray')
        plt.text((val_border + test_border) / 2, 1.0, 'Test', ha='center', color='gray')
        plt.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
