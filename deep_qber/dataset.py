from abc import ABC, abstractmethod

import torch
import pandas as pd
from tqdm import trange
from catboost import Pool


class BaseDataset(ABC):
    def __init__(self, dataframe, target_column, window_size, train_size, anomaly_column):
        self.dataframe = dataframe
        self.target_column = target_column
        self.window_size = window_size
        self.train_size = train_size
        self.anomaly_column = anomaly_column
        self.anomalies_present = (dataframe[anomaly_column] == 1).sum() > 0
        self.dataset = None

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

    @abstractmethod
    def load(self):
        pass

    def setup(self, start, end, from_files=False):
        if not from_files:
            if start < self.window_size:
                raise ValueError('Start should be greater than window size!')
            dataset = []
            for i in trange(start, end):
                subset = self.dataframe.loc[i - self.window_size:i]
                x = self.transform(subset.drop(columns=self.anomaly_column))
                anomaly_label = subset[self.anomaly_column].iloc[-1]
                dataset.append((i, x, anomaly_label))
            self.assemble(dataset)
        else:
            self.load()
        return self.dataset


class ClassicModelDataset(BaseDataset):
    def __init__(self,
                 dataframe,
                 target_column,
                 anomaly_column,
                 window_size=10,
                 train_size=0.75):
        super().__init__(dataframe, target_column, window_size, train_size, anomaly_column)

        self.window_options = [self.window_size // 2, self.window_size]
        if self.window_size >= 20:
            self.window_options.append(5)

        self.schema = []
        self.set_schema()

    def set_schema(self):
        cols = self.dataframe.columns
        cols = list(filter(lambda x: x != self.anomaly_column, cols))
        for window in self.window_options:
            self.schema += [col + f'_w{window}_mean' for col in cols]
            self.schema += [col + f'_w{window}_std' for col in cols]
            self.schema += [col + f'_w{window}_minmax_delta' for col in cols]
            self.schema += [col + f'_w{window}_ema' for col in cols]
        self.schema += [f'{self.window_size - 1 - i}_lag_{self.target_column}' for i in range(self.window_size)]
        self.schema += list(cols)

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

    def load(self):
        self.dataset = pd.read_csv(f'catboost_dataset.csv', index_col='index')

    def assemble(self, dataset):
        indices, rows, anomaly_labels = zip(*dataset)
        dataset = pd.DataFrame(rows, columns=self.schema)
        dataset.index = indices
        dataset['_anomaly_label'] = anomaly_labels

        self.dataset = dataset
        self.dataset.reset_index().to_csv(f'catboost_dataset.csv', index=False)

    def get_catboost_pools(self, anomaly_split=True):
        pools = []
        targets = []
        if anomaly_split and self.anomalies_present:
            normal_subset = self.dataset[self.dataset['_anomaly_label'] == 0].drop(columns='_anomaly_label')
            anomaly_subset = self.dataset[self.dataset['_anomaly_label'] == 1].drop(columns='_anomaly_label')
            subsets = [normal_subset, anomaly_subset]
        else:
            subsets = [self.dataset.drop(columns='_anomaly_label')]
        for subset in subsets:
            train, test = self.split(subset, self.train_size)
            train_y = train[self.target_column]
            test_y = test[self.target_column]

            train_pool = Pool(train.drop(columns=self.target_column), train_y)
            test_pool = Pool(test.drop(columns=self.target_column), test_y)
            pools.append((train_pool, test_pool))
            targets.append((train_y, test_y))
        return pools, targets


class TorchDataset(torch.utils.data.Dataset):
    def __init__(self, data):
        self.indices, self.X, self.Y = [], [], []
        for point in data:
            i, data = point
            x_lag, x_latest, y = data
            self.indices.append(i)
            self.X.append((x_lag, x_latest))
            self.Y.append(y)
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        return self.X[idx], self.Y[idx]


class TorchDatasetInterface(BaseDataset):
    def __init__(self,
                 dataframe,
                 target_column,
                 anomaly_column,
                 shuffle=True,
                 train_size=0.75,
                 batch_size=64,
                 window_size=10,
                 dtype=torch.float32,
                 ):
        super().__init__(dataframe=dataframe,
                         target_column=target_column,
                         anomaly_column=anomaly_column,
                         window_size=window_size,
                         train_size=train_size)
        self.dtype = dtype
        self.shuffle = shuffle
        self.batch_size = batch_size
        self.train = None
        self.test = None
    
    def transform(self, subset):
        lag, latest = subset.iloc[:-1], subset.iloc[-1]
        y = torch.tensor(
            [latest[self.target_column]],
            dtype=self.dtype
        )
        x_latest = torch.tensor(
            latest.drop(self.target_column).values,
            dtype=self.dtype
        )
        x_lag = torch.tensor(
            lag.values,
            dtype=self.dtype
        )
        return x_lag, x_latest, y
    
    def assemble(self, dataset):
        self.dataset = dataset
        torch.save(self.dataset, 'qber_dataset.pt')
    
    def load(self):
        self.dataset = torch.load('qber_dataset.pt')
    
    def get_dataloaders(self, anomaly_split=True):
        loaders = []
        normal, anomaly = [], []
        for point in self.dataset:
            i, data, anomaly_label = point
            if anomaly_split and anomaly_label:
                anomaly.append((i, data))
            else:
                normal.append((i, data))
        subsets = [normal]
        if anomaly:
            subsets.append(anomaly)
        for subset in subsets:
            train, test = self.split(subset, train_size=self.train_size)
            train_dataset = TorchDataset(train)
            train_loader = torch.utils.data.DataLoader(train_dataset,
                                                       shuffle=self.shuffle,
                                                       batch_size=self.batch_size)
            
            test_dataset = TorchDataset(test)
            test_loader = torch.utils.data.DataLoader(test_dataset,
                                                      shuffle=self.shuffle,
                                                      batch_size=self.batch_size)
            loaders.append((train_loader, test_loader))
        
        return loaders
