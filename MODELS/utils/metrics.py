import numpy as np

def RSE(pred, true):
    return np.sqrt(np.sum((true - pred) ** 2)) / np.sqrt(np.sum((true - true.mean()) ** 2))

def CORR(pred, true, eps=1e-6):
    u = ((true - true.mean(0)) * (pred - pred.mean(0))).sum(0)
    d = np.sqrt(((true - true.mean(0)) ** 2 * (pred - pred.mean(0)) ** 2).sum(0))
    return (u / (d + eps)).mean(-1)

def MAE(pred, true):
    return np.mean(np.abs(true - pred))

def MSE(pred, true):
    return np.mean((true - pred) ** 2)

def RMSE(pred, true):
    return np.sqrt(MSE(pred, true))

def MAPE(pred, true, eps=1e-6):
    return np.mean(np.abs((true - pred) / (true + eps)))

def MSPE(pred, true, eps=1e-6):
    return np.mean(np.square((true - pred) / (true + eps)))

def NSE(pred, true):
    ss_res = np.sum((true - pred) ** 2)
    ss_tot = np.sum((true - np.mean(true)) ** 2)
    return 1 - ss_res / ss_tot if ss_tot != 0 else np.nan

def metric(pred, true):
    mae = MAE(pred, true)
    mse = MSE(pred, true)
    rmse = RMSE(pred, true)
    mape = MAPE(pred, true)
    mspe = MSPE(pred, true)
    nse = NSE(pred, true)

    return mae, mse, rmse, mape, mspe, nse
