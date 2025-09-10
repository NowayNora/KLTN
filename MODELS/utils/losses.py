import torch
import torch.nn as nn


def divide_no_nan(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Safe division: a / b with NaN or Inf replaced by 0.
    """
    result = a / b
    result[result != result] = 0.0  # NaN
    result[result == float("inf")] = 0.0  # Inf
    return result


class SMAPELoss(nn.Module):
    """
    Symmetric Mean Absolute Percentage Error (sMAPE)
    https://robjhyndman.com/hyndsight/smape/
    """
    def __init__(self):
        super(SMAPELoss, self).__init__()

    def forward(self, forecast: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        numerator = torch.abs(forecast - target)
        denominator = torch.abs(forecast) + torch.abs(target) + 1e-8
        smape = 200. * numerator / denominator
        return torch.mean(smape)


class MAPELoss(nn.Module):
    """
    Mean Absolute Percentage Error (MAPE)
    https://en.wikipedia.org/wiki/Mean_absolute_percentage_error
    """
    def __init__(self):
        super(MAPELoss, self).__init__()

    def forward(self, forecast: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs((forecast - target) / (target + 1e-8))) * 100


class MASELoss(nn.Module):
    """
    Mean Absolute Scaled Error (MASE)
    https://robjhyndman.com/papers/mase.pdf
    """
    def __init__(self, freq: int):
        super(MASELoss, self).__init__()
        self.freq = freq

    def forward(self, insample: torch.Tensor,
                forecast: torch.Tensor,
                target: torch.Tensor) -> torch.Tensor:
        """
        insample: (batch, time_i)
        forecast, target: (batch, time_o)
        """
        # Mean absolute difference in insample
        scale = torch.mean(torch.abs(insample[:, self.freq:] - insample[:, :-self.freq]), dim=1)
        scale = scale.unsqueeze(1)  # shape: (batch, 1)
        scale = scale + 1e-8  # avoid division by zero

        mase = torch.abs(target - forecast) / scale
        return torch.mean(mase)

class R2Loss(nn.Module):
    """
    R-squared (Coefficient of Determination) Loss
    Higher R² is better (1 is perfect). Loss is (1 - R²), so lower is better.
    """
    def __init__(self):
        super(R2Loss, self).__init__()

    def forward(self, forecast: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        target_mean = torch.mean(target)
        ss_tot = torch.sum((target - target_mean) ** 2)
        ss_res = torch.sum((target - forecast) ** 2)
        r2 = 1 - divide_no_nan(ss_res, ss_tot + 1e-8)  # safe division
        return 1 - r2  # return as a loss (lower is better)
