import optuna

def suggest_model_specific(trial, model, seq_len=None):
    # ---------- 1. Các tham số chung ----------
    frac = trial.suggest_float('label_frac', 0.1, 0.9, step=0.1)  
    seq_len = trial.suggest_int('seq_len', 24, 48, 6)      
    label_len = (max(12, int(seq_len * frac)) // 12) * 12                        # làm tròn 12    label_len = None
    

    d_ff = [128, 256, 512, 1024, 2048]

    # Đảm bảo d_model chia hết cho n_heads
    d_model_choices = [256, 512, 1024, 2048]
    n_heads_choices = [4, 8, 16, 32]
 

    # ---------- 2. Từng mô hình ----------
    if model == 'PatchTST':
        patch_len = trial.suggest_categorical('patch_len', [4, 8, 16, 32])
        if patch_len > seq_len:          
            raise optuna.exceptions.TrialPruned()

        return {
            'seq_len': seq_len,
            'label_len': label_len,
            'd_model': trial.suggest_categorical('d_model', d_model_choices),
            'n_heads': trial.suggest_categorical('n_heads', n_heads_choices),
            'e_layers': trial.suggest_int('e_layers', 2, 3),
            'd_layers': trial.suggest_int('d_layers', 1, 2),
            'd_ff': trial.suggest_categorical('d_ff', d_ff),
            'activation': trial.suggest_categorical('activation', ['gelu','relu']),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.05),
            'patch_len': patch_len,
            'factor': trial.suggest_int('factor', 1, 4)
        }

    elif model in ['LSTM', 'GRU']:
        label_len = 0
        return {
            'seq_len': seq_len,
            'label_len': label_len,
            'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.1),
            'd_model': trial.suggest_categorical('d_model', d_model_choices),
            'e_layers': trial.suggest_int('e_layers', 2,3),
            'activation': trial.suggest_categorical('activation', ['gelu','relu']),
            'bidirectional': trial.suggest_categorical('bidirectional', [True, False]),
        }

    elif model == 'Nonstationary_Transformer':
        hidden_unit = trial.suggest_categorical('hidden_unit', [128, 256])
        p_hidden_layers = trial.suggest_int('p_hidden_layers', 2, 3)
        return {
            'seq_len': seq_len,
            'label_len': label_len,
            'd_model': trial.suggest_categorical('d_model', d_model_choices),
            'n_heads': trial.suggest_categorical('n_heads', n_heads_choices),
            'e_layers': trial.suggest_int('e_layers', 2, 3),
            'd_layers': trial.suggest_int('d_layers', 1, 2),
            'd_ff': trial.suggest_categorical('d_ff', d_ff),
            'factor': trial.suggest_int('factor', 2, 4),
            'p_hidden_dims': [hidden_unit] * p_hidden_layers,
            'p_hidden_layers': p_hidden_layers,
            'top_k': trial.suggest_categorical('top_k', [5, 10, 15]),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.05),
            'activation': trial.suggest_categorical('activation', ['gelu','relu']),
        }

    elif model in ['iTransformer', 'TSMixer', 'Transformer', 'Informer', 'Crossformer']:
        return {
            'seq_len': seq_len,
            'label_len': label_len,
            'd_model': trial.suggest_categorical('d_model', d_model_choices),
            'n_heads': trial.suggest_categorical('n_heads', n_heads_choices),
            'e_layers': trial.suggest_int('e_layers', 2, 3),
            'd_layers': trial.suggest_int('d_layers', 1, 2),
            'd_ff': trial.suggest_categorical('d_ff', d_ff),
            'factor': trial.suggest_int('factor', 1, 4),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.05),
            'activation': trial.suggest_categorical('activation', ['gelu','relu']),
        }
 

    elif model == 'DLinear':
        return {
            'seq_len': seq_len,
            'label_len': label_len,
            'moving_avg': trial.suggest_categorical('moving_avg', [3, 5, 7, 9, 11, 13]),
        }

    if model == 'Autoformer':
        return {
            'seq_len': seq_len,
            'label_len': label_len,
            'd_model': trial.suggest_categorical('d_model', d_model_choices),
            'n_heads': trial.suggest_categorical('n_heads', n_heads_choices),
            'e_layers': trial.suggest_int('e_layers', 2, 3),
            'd_layers': trial.suggest_int('d_layers', 1, 2),
            'd_ff': trial.suggest_categorical('d_ff', d_ff),
            'moving_avg': trial.suggest_categorical('moving_avg', [3, 5, 7, 9, 11, 13]),
            'factor': trial.suggest_int('factor', 1, 3),
            'dropout': trial.suggest_float('dropout', 0.1, 0.5, step=0.05),
            'activation': trial.suggest_categorical('activation', ['gelu','relu']),
        }

    return {}        # fallback
