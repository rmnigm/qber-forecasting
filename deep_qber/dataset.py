from abc import ABC, abstractmethod

import torch
import pandas as pd
from tqdm import trange
from catboost import Pool


class ABCDataset(ABC):
    def __init__(self, dataframe, target_column, window_size, train_size):
        self.dataframe = dataframe
        self.target_column = target_column
        self.window_size = window_size
        self.train_size = train_size
    
    @staticmethod
    def split(data, train_size):
        train_size = int(len(data) * train_size)
        data_train = data[0:train_size]
        data_test = data[train_size:len(data)]
        return data_train, data_test
    
    @abstractmethod
    def transform(self, subset):
        pass
    
    @abstractmethod
    def assemble(self, dataset):
        pass
    
    def setup(self, start, end):
        if start < self.window_size:
            raise ValueError('Start should be greater than window size!')
        dataset = []
        for i in trange(start, end):
            subset = self.dataframe.loc[i - self.window_size:i]
            x = self.transform(subset)
            dataset.append((i, x))
        model_format_dataset = self.assemble(dataset)
        return model_format_dataset


class ClassicModelDataset(ABCDataset):
    def __init__(self, dataframe, target_column, window_size=10, train_size=0.75):
        super().__init__(dataframe, target_column, window_size, train_size)
        
        self.window_options = [self.window_size // 2, self.window_size]
        if self.window_size >= 20:
            self.window_options.append(5)
        
        self.train = None
        self.test = None
        self.schema = self.set_schema()
    
    def set_schema(self):
        schema = []
        cols = self.dataframe.columns
        for window in self.window_options:
            schema += [col + f'_w{window}_mean' for col in cols]
            schema += [col + f'_w{window}_std' for col in cols]
            schema += [col + f'_w{window}_minmax_delta' for col in cols]
            schema += [col + f'_w{window}_ema' for col in cols]
        schema += [f'{self.window_size - 1 - i}_lag_{self.target_column}' for i in range(self.window_size)]
        schema += list(cols)
        return schema
    
    def transform(self, subset):
        features = []
        for window in self.window_options:
            x = subset.iloc[:window]
            
            means = x.mean()
            means.index = [col + f'_w{window}_mean' for col in means.index]
            
            stds = x.std()
            stds.index = [col + f'_w{window}_std' for col in stds.index]
            
            deltas = x.max() - x.min()
            deltas.index = [col + f'_w{window}_minmax_delta' for col in deltas.index]
            
            emas = x.ewm(alpha=0.2).mean().iloc[-1]
            emas.index = [col + f'_w{window}_ema' for col in emas.index]
            
            features += [means, deltas, stds, emas]
        
        lag_vals = subset[self.target_column].iloc[:-1]
        lag_vals.index = [
            f'{self.window_size - 1 - i}_lag_{self.target_column}' for i in range(self.window_size)
        ]
        latest_vals = subset.iloc[-1]
        features += [lag_vals, latest_vals]
        
        series_features = pd.concat(features)
        return series_features
    
    def assemble(self, dataset):
        indices, rows = zip(*dataset)
        dataset = pd.DataFrame(rows, columns=self.schema)
        dataset.index = indices
        
        train, test = self.split(dataset, self.train_size)
        self.train = train
        self.test = test
        train_pool = Pool(train.drop(columns=self.target_column),
                          train[self.target_column])
        test_pool = Pool(test.drop(columns=self.target_column),
                         test[self.target_column])
        
        return train_pool, test_pool


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.indices, self.X, self.Y = [], [], []
        for point in data:
            i, datarow = point
            x_lag, x_latest, y = datarow
            self.indices.append(i)
            self.X.append((x_lag, x_latest))
            self.Y.append(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class TorchDatasetInterface(ABCDataset):
    def __init__(self,
                 dataframe,
                 target_column,
                 shuffle=True,
                 train_size=0.75,
                 batch_size=64,
                 window_size=10,
                 dtype=torch.float32,
                 ):
        super().__init__(dataframe, target_column, window_size, train_size)
        self.dtype = dtype
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.train = None
        self.test = None
    
    def transform(self, subset):
        lag, latest = subset.iloc[:-1], subset.iloc[-1]
        y = torch.tensor(
            latest[self.target_column],
            dtype=self.dtype
        )
        x_latest = torch.tensor(
            latest.drop(columns=self.target_column).values,
            dtype=self.dtype
        )
        x_lag = torch.tensor(
            lag.values,
            dtype=self.dtype
        )
        return x_lag, x_latest, y
    
    def assemble(self, dataset):
        train, test = self.split(dataset, train_size=0.75)
        
        train_dataset = TorchDataset(train)
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   shuffle=self.shuffle,
                                                   batch_size=self.batch_size)
        
        test_dataset = TorchDataset(test)
        test_loader = torch.utils.data.DataLoader(test_dataset,
                                                  shuffle=self.shuffle,
                                                  batch_size=self.batch_size)
        
        self.train = train_dataset
        self.test = test_dataset
        
        return train_loader, test_loader
