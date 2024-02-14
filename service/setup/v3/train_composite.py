import random
import pathlib

import joblib
import polars as pl
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error


seed = 123456
random.seed(seed)
np.random.seed(seed)


def calculate_offset_limit(offset, limit, length) -> tuple[int, int]:
    if offset is None:
        offset = 0
    else:
        offset = offset if offset >= 1 else int(offset * length)
    if limit is None:
        limit = length
    else:
        limit = limit if limit >= 1 else int(limit * length)
    return offset, limit


def build(
    data_path: str | pathlib.Path,
    window_size: int,
    dtype: np.dtype = np.float32,
    columns: list[str] | None = None,
    offset: int | float = None,
    limit: int | float = None) -> np.ndarray:
    dataframe = pl.scan_csv(data_path)
    length = dataframe.select(pl.count()).collect().item()
    offset, limit = calculate_offset_limit(offset, limit, length)
    columns = columns or dataframe.columns
    dataframe = (
        dataframe
        .select(columns)
        .slice(offset, limit)
    )
    data_array = dataframe.collect().to_numpy()
    dataset = np.lib.stride_tricks.sliding_window_view(
        data_array,
        window_size + 1,
        axis=0
        )
    return dataset


qber_path = pathlib.Path('.')
columns = ['e_mu_current', 'e_mu_estimated', 'e_nu_1', 'e_nu_2', 'q_mu', 'q_nu1', 'q_nu2']

train_data = build(qber_path / 'datasets' / 'data.csv', 20, columns=columns, limit=0.75)
test_data = build(qber_path / 'datasets' / 'data.csv', 20, columns=columns, offset=0.75)

train_x, train_y = train_data[:, 0, :-1], train_data[:, 0, -1]
test_x, test_y = test_data[:, 0, :-1], test_data[:, 0, -1]


class CompositeModel(RegressorMixin, BaseEstimator):
    def __init__(self):
        super().__init__()
        self.base = LinearRegression()
        self.boost = LGBMRegressor(verbose=-1)

    def fit(self, X, y):
        self.base.fit(X, y)
        predictions = self.base.predict(X)
        diff = y - predictions
        self.boost.fit(X, diff)

    def predict(self, X):
        return self.base.predict(X) + self.boost.predict(X)
    
    def save(self, path_prefix: str):
        linear_filename = path_prefix + '_linear_model.joblib'
        joblib.dump(self.base, open(linear_filename, "wb"))
        boost_filename = path_prefix + '_boost_model.joblib'
        joblib.dump(self.boost, open(boost_filename, "wb"))

    def load(self, path_prefix: str):
        linear_filename = path_prefix + '_linear_model.joblib'
        self.base = joblib.load(linear_filename)
        boost_filename = path_prefix + '_boost_model.joblib'
        self.boost = joblib.load(boost_filename)


model = CompositeModel()
model.fit(train_x, train_y)
train_preds = model.predict(train_x)
test_preds = model.predict(test_x)
print(f'train R2 = {r2_score(train_y, train_preds):.7f}')
print(f'test R2 = {r2_score(test_y, test_preds):.7f}')
print(f'train MSE = {mean_squared_error(train_y, train_preds):.7f}')
print(f'test MSE = {mean_squared_error(test_y, test_preds):.7f}')
print(f'train RMSE = {mean_squared_error(train_y, train_preds, squared=False):.7f}')
print(f'test RMSE = {mean_squared_error(test_y, test_preds, squared=False):.7f}')
print(f'train MAPE = {mean_absolute_percentage_error(train_y, train_preds):.7f}')
print(f'test MAPE = {mean_absolute_percentage_error(test_y, test_preds):.7f}')
print('-' * 60)


model.save('models/composite')
model = CompositeModel()
model.load('models/composite')
train_preds = model.predict(train_x)
test_preds = model.predict(test_x)
print(f'train R2 = {r2_score(train_y, train_preds):.7f}')
print(f'test R2 = {r2_score(test_y, test_preds):.7f}')
print(f'train MSE = {mean_squared_error(train_y, train_preds):.7f}')
print(f'test MSE = {mean_squared_error(test_y, test_preds):.7f}')
print(f'train RMSE = {mean_squared_error(train_y, train_preds, squared=False):.7f}')
print(f'test RMSE = {mean_squared_error(test_y, test_preds, squared=False):.7f}')
print(f'train MAPE = {mean_absolute_percentage_error(train_y, train_preds):.7f}')
print(f'test MAPE = {mean_absolute_percentage_error(test_y, test_preds):.7f}')
print('-' * 60)
