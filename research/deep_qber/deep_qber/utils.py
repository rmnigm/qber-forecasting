import random
import statistics
from collections import deque

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from torchmetrics.functional import mean_squared_error


def seed_everything(seed: int) -> None:
    """Fix all the random seeds we can for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class OutlierDetector:
    def __init__(self, mode='median', window_size=10, alpha=3):
        self.mode = 'median'
        self.window_size = window_size
        self.alpha = alpha

    def fit(self, ts):
        anomalies = []
        window = deque(ts[:self.window_size])
        for i in tqdm(range(len(ts[self.window_size:]))):
            item = ts[i]
            med = statistics.median(window)
            std = statistics.stdev(window)
            diff = np.abs(item - med)
            if diff > std * self.alpha:
                anomalies.append(i)
            else:
                window.append(item)
                window.popleft()
        return anomalies
    
    def fit_transform(self, ts):
        new_ts = list(ts[:self.window_size].copy())
        window = deque(ts[:self.window_size])
        for item in tqdm(ts[self.window_size:]):
            med = statistics.median(window)
            std = statistics.stdev(window)
            diff = np.abs(item - med)
            if diff > std * self.alpha:
                new_ts.append(med)
            else:
                new_ts.append(item)
                window.append(item)
                window.popleft()
        return new_ts


def setup_metric(path):
    dataframe = pd.read_csv(path)
    x = dataframe['delta'].values
    y = dataframe['f_ec'].values
    X = np.stack([x ** k for k in range(5)]).T
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    
    def correction_effectivenes(predictions, labels):
        error = mean_squared_error(predictions, labels)
        x = np.power(error.cpu().detach().numpy(), range(5))
        return x @ w
    
    return correction_effectivenes
