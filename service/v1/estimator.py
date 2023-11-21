import json
from catboost import CatBoostRegressor
from processing import SlidingWindow


class Estimator:
    def __init__(self, size=20, feature_config_path: str | None = None):
        self.model = CatBoostRegressor()
        self.feature_config = None
        if feature_config_path is not None:
            with open(feature_config_path, 'r') as f:
                self.feature_config = json.load(f)
        self.data = SlidingWindow(size=size, feature_config=self.feature_config)
    
    def load_model(self, model_path: str, model_format='cbm'):
        self.model.load_model(model_path, format=model_format)
    
    def update(self, *args):
        self.data.update(*args)

    def predict(self) -> float:
        if self.data.is_full():
            features = self.data.get_boost_features()
            prediction = self.model.predict(data=features)
        else:
            prediction = self.data.get_latest_ema()
        return prediction
