import dataclasses
from collections import deque

import joblib
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.linear_model import LinearRegression


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
    def __init__(self, size: int) -> None:
        """
        Creates the SlidingWindow object
        :param size: number of records to store simultaneously
        :param alpha: parameter for exponential smoothing feature calculation
        :param feature_config: list of features, passed to model from SlidingWindow
        """
        self.data: deque[Record] = deque()
        self.size = size
        self.latest: Record | None = None

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

    def get_features(self) -> np.ndarray:
        """
        Transforms data stored data from SlidingWindow, returns flat dict:
        - key = str, feature name
        - value = np.float32, feature value
        :return: dict[str, np.float32], flat features
        """
        return np.array([elem.eMu for elem in self.data])
    
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
    def __init__(self, size=20):
        self.model = CompositeModel()
        self.data = SlidingWindow(size=size)
    
    def load_model(self, path_prefix: str) -> None:
        """
        Loads model from file.
        :param path_prefix: str, path prefix to model files
        :return: None
        """
        self.model.load(path_prefix)

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
            features = self.data.get_features().reshape(1, -1)
            prediction = self.model.predict(features)
        else:
            prediction = self.data.get_latest_ema()
        return prediction
