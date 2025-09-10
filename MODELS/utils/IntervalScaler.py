import numpy as np

class IntervalScaler:
    def __init__(self):
        self.low_min = None
        self.high_max = None
        self.range = None

    def fit(self, low: np.ndarray, high: np.ndarray):
        self.low_min = np.min(low)
        self.high_max = np.max(high)
        self.range = self.high_max - self.low_min + 1e-8  # tránh chia cho 0

    def transform(self, low: np.ndarray, high: np.ndarray):
        low_scaled = (low - self.low_min) / self.range
        high_scaled = (high - self.low_min) / self.range
        return low_scaled, high_scaled

    def inverse_transform(self, low_scaled: np.ndarray, high_scaled: np.ndarray):
        low = low_scaled * self.range + self.low_min
        high = high_scaled * self.range + self.low_min
        return low, high
