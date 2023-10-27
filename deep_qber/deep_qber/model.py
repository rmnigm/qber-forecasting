import torch
import torch.nn as nn
import pytorch_lightning as pl
import torchmetrics
from torchmetrics import R2Score, MeanSquaredError, MeanAbsolutePercentageError


class Extractor(nn.Module):
    def __init__(self, look_back, output_size, hidden_size, input_size):
        super().__init__() 
        self.look_back = look_back
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.autoregressive_dense = nn.Sequential(
            nn.Linear(self.look_back, self.hidden_size),
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
    def __init__(self, look_back, output_size, hidden_size, input_size):
        super().__init__() 
        self.look_back = look_back
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.autoregressive_dense = nn.Sequential(
            nn.Linear(self.look_back, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.dense_exod = nn.Sequential(
            nn.Linear(self.input_size - 1, self.hidden_size),
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
        exod_features = self.dense_exod(x_current)
        return self.classifier(torch.cat((autoregressive_features, exod_features), 1))
    
    
class ExtractorLSTM(nn.Module):
    def __init__(self, input_size, output_size, look_back, hidden_size=128):
        super().__init__() 
        self.look_back = look_back
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(self.input_size,
                            self.hidden_size,
                            batch_first=True
                            )
        self.dense = nn.Sequential(
            nn.Linear(self.input_size - 1, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
        )
        self.regressor = nn.Linear(2 * self.hidden_size, output_size)
        
    def forward(self, data):
        x, x_current = data
        x, _ = self.lstm(x)
        past_features = x[:, -1, :]
        current_features = self.dense(x_current)
        features = torch.cat((past_features, current_features), 1)
        return self.regressor(features)


class Reshape(nn.Module):
    def __init__(self, *args):
        super(Reshape, self).__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class ConvARNN(nn.Module):
    def __init__(self, look_back, output_size, hidden_size, input_size):
        super().__init__() 
        self.look_back = look_back
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.input_size = input_size
        self.batch_size = config["batch_size"]
        self.conv = nn.Sequential(
            nn.Conv1d(7, 16, 5),
            nn.Softplus(),
            nn.BatchNorm1d(16),

            nn.Conv1d(16, 32, 5),
            nn.Softplus(),
            nn.BatchNorm1d(32),
    
            nn.Conv1d(32, 64, 5),
            nn.Softplus(),
            nn.BatchNorm1d(64),
    
            nn.Conv1d(64, 64, 3),
            nn.Softplus(),
            nn.AdaptiveAvgPool1d(1),
            Reshape(-1, 64),
    
            nn.Linear(64, 1)
        )
        self.autoregressive = nn.Sequential(
            nn.Linear(self.look_back, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, 1),
        )
        
    def forward(self, data):
        x, x_current = data
        ar_component = self.autoregressive(x[:, :, 0])
        conv_component = self.conv(x.transpose(1, 2))
        return ar_component + conv_component


class ModelInterfaceTS(nn.Module):
    def __init__(self, model, device):
        super().__init__()
        self.model = model
        self.metrics = {
            "MSE": torchmetrics.MeanSquaredError().to(device),
            "R2Score": torchmetrics.R2Score().to(device),
            "MAPE": torchmetrics.MeanAbsolutePercentageError().to(device),
        }

    def forward(self, x):
        return self.model(x)
    
    def get_metrics(self, predictions, labels):
        preds_f, labels_f = predictions.float(), labels.float()
        return {k: v(preds_f, labels_f) for k, v in self.metrics.items()}


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
        for k, v in metrics.items():
            self.log(f"Train {k}", v, prog_bar=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        data, target = val_batch
        preds = self.forward(data)
        loss = self.loss_multiplier * self.loss(preds, target)
        metrics = self.model.get_metrics(preds, target)
        self.log("Validation Loss", loss, prog_bar=True)
        for k, v in metrics.items():
            self.log(f"Validation {k}", v, prog_bar=True)
