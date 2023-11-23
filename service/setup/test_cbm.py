import json
import polars as pl
import numpy as np
from tqdm import tqdm
from processing import SlidingWindow
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor, FeaturesData, Pool


def get_data(feature_config: list | None = None):
    filename = 'datasets/data.csv'
    df = pl.read_csv(filename)
    
    buffer = SlidingWindow(20, feature_config=feature_config)
    features, targets = [], []
    e_mu_prev = None
    for i, row in tqdm(enumerate(df.rows()), total=len(df)):
        row_id, e_mu, *values = row
        e_mu_ema, e_nu_1, e_nu_2, q_mu, q_nu1, q_nu2 = values
        buffer.update(e_mu_prev, e_mu_ema, e_nu_1, e_nu_2, q_mu, q_nu1, q_nu2)
        if buffer.is_full():
            features.append(buffer.get_features())
            targets.append(e_mu)
        e_mu_prev = e_mu
    
    train_x, test_x, train_y, test_y = train_test_split(features, targets, train_size=0.75, shuffle=False)
    train_num_features = FeaturesData(
        num_feature_data=np.array([list(row.values()) for row in train_x], dtype=np.float32),
        num_feature_names=list(buffer.feature_config)
    )
    test_num_features = FeaturesData(
        num_feature_data=np.array([list(row.values()) for row in test_x], dtype=np.float32),
        num_feature_names=list(buffer.feature_config)
    )
    train_pool = Pool(train_num_features, train_y)
    test_pool = Pool(test_num_features, test_y)
    return train_pool, test_pool, train_y, test_y


with open('config.json', 'r') as f:
    features_config = json.load(f)

train_pool, test_pool, train_y, test_y = get_data(features_config)

model = CatBoostRegressor()
model.load_model('best_unscaled.cbm')
test_predictions = model.predict(test_pool)
print(f'R2 score = {r2_score(test_y, test_predictions)}')