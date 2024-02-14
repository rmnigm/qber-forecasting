import json
import dataclasses
from collections import deque
from itertools import chain

import numpy as np
from catboost import CatBoostRegressor, FeaturesData  # type: ignore


record_tuple = tuple[float | None, ...]
record_list = list[record_tuple]


@dataclasses.dataclass
class Record:
    """
    Time series point, containing frame statistics values.
    eMu is optional, because it is known only after processing the frame
    """
    eMuEma: float
    eMu: float | None
    eNu1: float
    eNu2: float
    qMu: float
    qNu1: float
    qNu2: float


class SlidingWindow:
    """
    Data structure for sliding window of N Records, adapted for creation of catboost features
    """
    def __init__(self, size: int,
                 alpha: float | None = None,
                 feature_config: list[str] | None = None) -> None:
        """
        Creates the SlidingWindow object
        :param size: number of records to store simultaneously
        :param alpha: parameter for exponential smoothing feature calculation
        :param feature_config: list of features, passed to model from SlidingWindow
        """
        self.data: deque[Record] = deque()
        self.size = size
        self.latest: Record | None = None
        self.alpha = alpha if alpha is not None else 2. / (size + 1)
        self.ema_weights = np.array(
            [self.alpha * (1 - self.alpha) ** i for i in range(size)][::-1],
            dtype=np.float32
        )
        self.feature_names = self.build_feature_names()
        if feature_config is None:
            self.feature_config = list(chain(*self.feature_names.values()))
        else:
            self.feature_config = feature_config
    
    def build_feature_names(self) -> dict[str, list[str]]:
        """
        Creates dict of feature names for model:
            key: feature subtype (stddev, mean, lag, etc.)
            value: list of feature names
        TODO: change to static dict with predefined values
        :return: dict of feature names
        """
        latest_exod = ['eNu1', 'eNu2', 'qMu', 'qNu1', 'qNu2']
        exp_smoothed = [f'{ts}_ema' for ts in latest_exod]
        deltas = [f'{ts}_delta' for ts in latest_exod]
        means = [f'{ts}_mean' for ts in latest_exod]
        stds = [f'{ts}_std' for ts in latest_exod]
        target_lags = [f'eMu_lag_{i}' for i in reversed(range(self.size))]
        lags = []
        for row in latest_exod:
            lags.extend([f'{row}_lag_{i}' for i in reversed(range(self.size))])
        return {
            'latest': latest_exod,
            'exp_smoothed': exp_smoothed,
            'deltas': deltas,
            'means': means,
            'stds': stds,
            'target_lags': target_lags,
            'lags': lags,
        }
    
    def is_full(self) -> bool:
        """
        Checks if SlidingWindow contains self.size records
        :return: bool, true if full
        """
        return len(self.data) == self.size
    
    def update(self, e_mu_prev: float | None, e_mu_ema: float, e_nu1: float, e_nu2: float, q_mu: float, q_nu1: float,
               q_nu2: float) -> None:
        """
        Inserts new data into structure: adds latest record for all values except eMu
        and fills eMu value in previous record, pushes it to the SlidingWindow.
        :return: None
        """
        record = Record(eMu=None, eMuEma=e_mu_ema, eNu1=e_nu1, eNu2=e_nu2, qMu=q_mu, qNu1=q_nu1, qNu2=q_nu2)
        if self.latest is None:
            self.latest = record
        else:
            self.latest.eMu = e_mu_prev
            if self.is_full():
                self.data.popleft()
            self.data.append(self.latest)
            self.latest = record
    
    def to_tuple_list(self) -> tuple[list, tuple]:
        """
        Transforms stored data from SlidingWindow, returns tuple of two objects:
        - tuple of latest time series point, every statistic except eMu
        - list of tuples of all stored time points, every value filled
        :return: tuple[tuple[float | None, ...], list[tuple[float | None, ...]]]
        """
        latest: record_tuple = dataclasses.astuple(self.latest)[2:]
        data: record_list = [dataclasses.astuple(elem)[1:] for elem in self.data]
        return data, latest
    
    def get_features(self) -> dict[str, np.float32]:
        """
        Transforms data stored data from SlidingWindow, returns flat dict:
        - key = str, feature name
        - value = np.float32, feature value
        :return: dict[str, np.float32], flat features
        """
        data, latest = self.to_tuple_list()
        matrix = np.array(data, dtype=np.float32).T
        ema_features = (matrix * self.ema_weights).sum(axis=1)
        delta_features = matrix.max(axis=1) - matrix.min(axis=1)
        std_features, mean_features = matrix.std(axis=1), matrix.mean(axis=1)
        target_lag_features = matrix[0]
        lag_features = []
        for row in matrix[1:]:
            lag_features.extend(row)
        features = {k: v for k, v in zip(self.feature_names['deltas'], delta_features)}
        features.update(dict(zip(self.feature_names['exp_smoothed'], ema_features)))
        features.update(dict(zip(self.feature_names['stds'], std_features)))
        features.update(dict(zip(self.feature_names['means'], mean_features)))
        features.update(dict(zip(self.feature_names['latest'], latest)))
        features.update(dict(zip(self.feature_names['target_lags'], target_lag_features)))
        features.update(dict(zip(self.feature_names['lags'], lag_features)))
        return {k: features[k] for k in self.feature_config}
    
    def get_boost_features(self) -> FeaturesData:
        """
        Transforms data stored in SlidingWindow into Catboost FeatureData object.
        :return: FeatureData, flat features for current timepoint
        """
        features = self.get_features()
        num_feature = np.array(list(features.values()), dtype=np.float32)[None, :]
        num_feature_names = list(features.keys())
        features_data = FeaturesData(
            num_feature_data=num_feature,
            num_feature_names=num_feature_names,
        )
        return features_data
    
    def get_latest_ema(self) -> float | None:
        """
        Return exponentially averaged eMu from latest record, if not empty.
        :return: float | None, latest eMuEma value
        """
        if self.latest is not None:
            return self.latest.eMuEma
        return None


class Estimator:
    """
    CatBoost model interface for sequential forecasting with SlidingWindow as data storage.
    """
    def __init__(self, size=20, feature_config_path: str | None = None):
        self.model = CatBoostRegressor()
        self.feature_config = None
        if feature_config_path is not None:
            with open(feature_config_path, 'r') as f:
                self.feature_config = json.load(f)
        self.data = SlidingWindow(size=size, feature_config=self.feature_config)
    
    def load_model(self, model_path: str, model_format: str = 'cbm') -> None:
        """
        Loads model from file.
        :param model_path: str, path to model file
        :param model_format: str, model file extension
        :return: None
        """
        self.model.load_model(model_path, format=model_format)
    
    def update(self, *args):
        """
        Updates SlidingWindow with new record, see SlidingWindow.update
        """
        self.data.update(*args)

    def predict(self) -> float | None:
        """
        Forecasts the new eMu value:
        - if sliding window if full, then calculate the features and pass them to model, return model output
        - if sliding window is not full, but there is at least one record, return latest exponentially averaged value
        - else return None
        :return: float | None, resulting prediction for current timepoint
        """
        if self.data.is_full():
            features = self.data.get_boost_features()
            prediction = self.model.predict(data=features)
        else:
            prediction = self.data.get_latest_ema()
        return prediction
