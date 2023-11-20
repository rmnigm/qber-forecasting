import dataclasses
from collections import deque

import numpy as np
from catboost import FeaturesData


record_tuple = tuple[float | None, ...]
record_list = list[record_tuple] | None


@dataclasses.dataclass
class Record:
    eMuEma: float
    eMu: float | None
    eNu1: float
    eNu2: float
    qMu: float
    qNu1: float
    qNu2: float


class SlidingWindow:
    def __init__(self, size: int,
                 alpha: float | None = None,
                 feature_config: list[str] | None = None) -> None:
        self.data: deque[Record] = deque()
        self.size = size
        self.latest: Record | None = None
        self.alpha = alpha if alpha is not None else 2. / (size + 1)
        self.ema_weights = np.array(
            [self.alpha * (1 - self.alpha) ** i for i in range(size)][::-1],
            dtype=np.float32
        )
        self.feature_names = self.build_feature_config()
        if feature_config is None:
            self.feature_config = set().union(*self.feature_names.values())
        else:
            self.feature_config = feature_config

    def build_feature_config(self):
        latest_exod = frozenset(['eNu1', 'eNu2', 'qMu', 'qNu1', 'qNu2'])
        exp_smoothed = frozenset([f'{ts}_ema' for ts in latest_exod])
        deltas = frozenset([f'{ts}_delta' for ts in latest_exod])
        means = frozenset([f'{ts}_mean' for ts in latest_exod])
        stds = frozenset([f'{ts}_std' for ts in latest_exod])
        target_lags = frozenset([f'eMu_lag_{i}' for i in reversed(range(self.size))])
        return {
            'latest': latest_exod,
            'exp_smoothed': exp_smoothed,
            'deltas': deltas,
            'means': means,
            'stds': stds,
            'target_lags': target_lags,
        }
        
    def is_full(self) -> bool:
        return len(self.data) == self.size

    def update(self, e_mu_prev: float | None, e_mu_ema: float, e_nu1: float, e_nu2: float, q_mu: float, q_nu1: float, q_nu2: float):
        record = Record(eMu=None, eMuEma=e_mu_ema, eNu1=e_nu1, eNu2=e_nu2, qMu=q_mu, qNu1=q_nu1, qNu2=q_nu2)
        if self.latest is None:
            self.latest = record
        else:
            self.latest.eMu = e_mu_prev
            if self.is_full():
                self.data.popleft()
            self.data.append(self.latest)
            self.latest = record

    def to_tuple_list(self) -> tuple[record_list, record_tuple]:
        latest, data = None, None
        if self.latest is not None:
            latest = dataclasses.astuple(self.latest)[2:]
            if self.is_full():
                data = [dataclasses.astuple(elem)[1:] for elem in self.data]
        return data, latest
    
    def get_features(self) -> dict[str, np.float32]:
        data, latest = self.to_tuple_list()
        matrix = np.array(data, dtype=np.float32).T
        ema_features = (matrix * self.ema_weights).sum(axis=1)
        delta_features = matrix.max(axis=1) - matrix.min(axis=1)
        std_features, mean_features = matrix.std(axis=1), matrix.mean(axis=1)
        lag_features = matrix[0]
        features = {k: v for k, v in zip(self.feature_names['deltas'], delta_features)}
        features.update({k: v for k, v in zip(self.feature_names['exp_smoothed'], ema_features)})
        features.update({k: v for k, v in zip(self.feature_names['stds'], std_features)})
        features.update({k: v for k, v in zip(self.feature_names['means'], mean_features)})
        features.update({k: v for k, v in zip(self.feature_names['latest'], latest)})
        features.update({k: v for k, v in zip(self.feature_names['target_lags'], lag_features)})
        return features
    
    def get_boost_features(self) -> FeaturesData:
        features = {k: v for k, v in self.get_features().items() if k in self.feature_config}
        num_feature = np.array(list(features.values()), dtype=np.float32)[None, :]
        num_feature_names = list(features.keys())
        features_data = FeaturesData(
            num_feature_data=num_feature,
            num_feature_names=num_feature_names,
        )
        return features_data
    
    def get_latest_ema(self) -> float | None:
        if self.latest is not None:
            return self.latest.eMuEma
