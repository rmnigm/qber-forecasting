import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.functional import mean_squared_error, mean_absolute_percentage_error


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


def setup_metric(path):
    dataframe = pd.read_csv(path)
    x = dataframe['delta'].values
    y = dataframe['f_ec'].values
    X = np.stack([x ** k for k in range(5)]).T
    w, *_ = np.linalg.lstsq(X, y, rcond=None)
    
    def correction_effectivenes(predictions, labels):
        error = mean_squared_error(predictions, labels)
        x = np.power(error.cpu().numpy(), range(5))
        return x @ w

    return correction_effectivenes


class ModelInterfaceTS(nn.Module):
    def __init__(self, model, metric_filepath):
        super().__init__()
        self.model = model
        self.metric = setup_metric(metric_filepath)

    def forward(self, x):
        return self.model(x)
    
    def get_metrics(self, predictions, labels):
        return {
            "MSE": mean_squared_error(predictions.float(), labels.float()),
            "MAPE": mean_absolute_percentage_error(predictions.float(), labels.float()),
            "F_EC": self.metric(predictions.float(), labels.float()),
        }


class ModuleTS(pl.LightningModule):
    def __init__(self, model, loss, lr=1e-5):
        super().__init__()
        self.model = model
        self.loss = loss
        self.lr = lr
        self.loss_multiplier = 1e4
        self.save_hyperparameters(ignore=['model'])

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        data, target = train_batch
        predictions = self.forward(data)
        loss = self.loss_multiplier * self.loss(predictions, target)
        self.log("Train Loss", loss, prog_bar=True)
        metrics = self.model.get_metrics(predictions, target)
        self.log("Train MSE", metrics["MSE"], prog_bar=True)
        self.log("Train MAPE", metrics["MAPE"], prog_bar=True)
        self.log("Train F_EC", metrics["F_EC"], prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        data, target = val_batch
        preds = self.forward(data)
        loss = self.loss_multiplier * self.loss(preds, target)
        metrics = self.model.get_metrics(preds, target)
        self.log("Validation Loss", loss, prog_bar=True)
        self.log("Validation MSE", metrics["MSE"], prog_bar=True)
        self.log("Validation MAPE", metrics["MAPE"], prog_bar=True)
        self.log("Validation F_EC", metrics["F_EC"], prog_bar=True)
