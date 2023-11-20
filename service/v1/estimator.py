from catboost import CatBoostRegressor
from processing import SlidingWindow


class Estimator:
    def __init__(self, size=20):
        self.model = CatBoostRegressor()
        self.data = SlidingWindow(size=size)
    
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
