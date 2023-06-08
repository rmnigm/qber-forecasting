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


class Extractor(nn.Module):
    def __init__(self, look_back, output_size, hidden_size):
        super().__init__() 
        self.input_size = look_back
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.autoregressive_dense = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.classifier = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.output_size)
        )
        
    def forward(self, data):
        x, x_current = data
        autoregressive_features = self.autoregressive_dense(x[:, :, 0])
        return self.classifier(autoregressive_features)


class ExtractorExod(nn.Module):
    def __init__(self, look_back, output_size, hidden_size):
        super().__init__() 
        self.input_size = look_back
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.autoregressive_dense = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.dense_exod = nn.Sequential(
            nn.Linear(6, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, 64),
            )
        self.classifier = nn.Sequential(
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size + 64, self.output_size)
        )
        
    def forward(self, data):
        x, x_current = data
        autoregressive_features = self.autoregressive_dense(x[:, :, 0])
        exod_features = self.dense_exod(x_current)[:, -1, :]
        return self.classifier(torch.cat((autoregressive_features, exod_features), 1))
    
    
class ExtractorLSTM(nn.Module):
    def __init__(self, input_size, output_size, hid_size=128):
        super().__init__() 
        self.input_size = input_size
        self.output_size = output_size
        self.hid_size = hid_size
        self.lstm = nn.LSTM(input_size,
                            hid_size,
                            batch_first=True
                            )
        self.dense = nn.Sequential(
            nn.Linear(input_size - 1, hid_size),
            nn.LeakyReLU(),
            nn.Linear(hid_size, hid_size),
        )
        self.regressor = nn.Linear(2 * hid_size, output_size)
        
    def forward(self, data):
        x, x_current = data
        x, _ = self.lstm(x)
        past_features = x[:, -1, :]
        current_features = self.dense(x_current)[:, -1, :]
        features = torch.cat((past_features, current_features), 1)
        return self.regressor(features)
