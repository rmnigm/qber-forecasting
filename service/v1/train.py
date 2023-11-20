import polars as pl
import numpy as np
from tqdm import tqdm, trange
from processing import SlidingWindow
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from catboost import CatBoostRegressor, FeaturesData, Pool


filename = 'datasets/data.csv'
df = pl.read_csv(filename)

buffer = SlidingWindow(20)
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


train_x, test_x, train_y, test_y = train_test_split(features, targets, train_size=0.75)
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


scores = []
feature_importances = None
for i in trange(20):
    model = CatBoostRegressor()
    model.fit(train_pool, eval_set=test_pool, verbose=False)

    predictions = model.predict(test_pool)
    scores.append(r2_score(test_y, predictions))
    if feature_importances is None:
        feature_importances = np.zeros_like(model.feature_importances_)
    feature_importances += model.feature_importances_


print(f'Mean R2 score: {np.mean(scores):.4f}, stddev: {np.std(scores):.4f}')
for a, b in sorted(zip(feature_importances, model.feature_names_))[::-1]:
    print(f'{b}: feat imp = {a:.6f}')
