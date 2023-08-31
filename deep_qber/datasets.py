from abc import ABC, abstractmethod

import pandas as pd
from tqdm import trange


class BaseDataset(ABC):
    def __init__(self, dataframe, target_column, window_size):
        self.dataframe = dataframe
        self.target_column = target_column
        self.window_size = window_size
    
    @abstractmethod
    def transform(self, subset):
        pass
    
    @abstractmethod
    def assemble(self, dataset):
        pass
    
    @staticmethod
    def split(data, train_size):
        train_size = int(len(data) * train_size)
        data_train, data_test = data[0:train_size], data[train_size:len(data)]
        return data_train, data_test

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


class ClassicModelDataset(BaseDataset):
    def __init__(self, dataframe, target_column, window_size=10):
        super().__init__(dataframe, target_column, window_size)
        
        self.window_options = [self.window_size // 2, self.window_size]
        if self.window_size >= 20:
            self.window_options.append(5)
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
        return dataset
