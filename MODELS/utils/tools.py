import numpy as np
import torch
import matplotlib.pyplot as plt
import os
import random

plt.switch_backend('agg')
import torch.optim as optim

def adjust_learning_rate(optimizer, epoch, args, scheduler=None, vali_loss=None):
    if args.lradj == 'type1':
        # Manual step decay
        lr = args.learning_rate * (0.5 ** ((epoch - 1) // 5))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        print(f"[INFO] Type1 LR Scheduler: updated learning rate to {lr}")

    elif args.lradj == 'type2':
        # Predefined epoch mapping
        lr_adjust = {
            2: 5e-5, 4: 1e-5, 6: 5e-6, 8: 1e-6,
            10: 5e-7, 15: 1e-7, 20: 5e-8
        }
        if epoch in lr_adjust.keys():
            lr = lr_adjust[epoch]
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            print(f"[INFO] Type2 LR Scheduler: updated learning rate to {lr}")

    elif args.lradj == 'type3':
        # OneCycleLR tự điều chỉnh mỗi step → không cần làm gì thêm
        pass
    elif args.lradj == 'type4':
        if scheduler is not None:
            scheduler.step(epoch)

class EarlyStopping:
    def __init__(self, patience=8, verbose=False, delta=1e-5):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.inf
        self.delta = delta

    def __call__(self, val_loss, model, path_to_file):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path_to_file)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path_to_file)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path_to_file):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}).  Saving model ...')
        os.makedirs(os.path.dirname(path_to_file), exist_ok=True)
        torch.save(model.state_dict(), path_to_file)
        self.val_loss_min = val_loss


class dotdict(dict):
    """dot.notation access to dictionary attributes"""
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def set_seed(seed=42):
    """Set random seed for reproducibility across Python, NumPy, and PyTorch."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # For deterministic behavior (slower but reproducible)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

