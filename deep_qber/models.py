import numpy as np
import torch
import torch.nn


class ExpSmoothing:
    def __init__(self, start_value, window_size):
        assert window_size >= 0
        self.val = start_value
        self.alpha = 2. / (window_size + 1)
        self.window_size = window_size

    def update(self, val):
        self.val = self.alpha * val + (1. - self.alpha) * self.val

    def get(self):
        return self.val
