import random

import torch
import numpy as np
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader
from torchmetrics.functional import mean_squared_error, mean_absolute_percentage_error


def seed_everything(seed: int) -> None:
    """Fix all the random seeds we can for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


class TorchTSDataset(Dataset):
    def __init__(self,
                 dataset,
                 target_index=0,
                 look_back=1,
                 device='cpu'):
        length = dataset.shape[0] - look_back - 1
        width = dataset.shape[1]
        mask = np.array([i != target_index for i in range(width)])
        x_current = np.empty((length, 1, width - 1))
        x, y = np.empty((length, look_back, width)), np.empty((length, 1))
        for i in range(length):
            x[i] = dataset[i:(i + look_back), :]
            x_current[i] = dataset[i + look_back, mask]
            y[i] = dataset[i + look_back, target_index]
        self.X = torch.tensor(x).float().to(device)
        self.y = torch.tensor(y).float().to(device)
        self.X_current = torch.tensor(x_current).float().to(device)
    
    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx):
        return (self.X[idx], self.X_current[idx]), self.y[idx]


def setup_dataset(dataset,
                  look_back: int = 5,
                  train_size: float = 0.8,
                  scaler=None,
                  batch_size: int = 64,
                  shuffle: bool = False,
                  device: str = 'cpu'):
    train_size = int(len(dataset) * train_size)
    test_size = len(dataset) - train_size
    data_train, data_test = dataset[0:train_size, :], dataset[train_size:len(dataset), :]
    print("Training set size = {}, testing set size = {}".format(train_size, test_size))

    if scaler is not None:
        scaler.fit(data_train)
        data_train = scaler.transform(data_train)
        data_test = scaler.transform(data_test)

    train_set = TorchTSDataset(data_train,
                               target_index=0,
                               look_back=look_back,
                               device=device)
    train_loader = DataLoader(train_set,
                              batch_size=batch_size,
                              shuffle=shuffle)
    test_set = TorchTSDataset(data_test,
                              target_index=0,
                              look_back=look_back,
                               device=device)
    test_loader = DataLoader(test_set,
                             batch_size=batch_size,
                             shuffle=shuffle)
    return train_loader, test_loader
