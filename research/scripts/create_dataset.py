import logging

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_percentage_error
from catboost import CatBoostRegressor, EFeaturesSelectionAlgorithm, EShapCalcType

from deep_qber import ClassicModelDataset


logging.basicConfig(filename='experiment.log',
                    filemode='a',
                    format='%(asctime)s,%(msecs)d %(name)s %(levelname)s %(message)s',
                    datefmt='%H:%M:%S',
                    level=logging.DEBUG)

raw_dataframe = pd.read_csv("datasets/qber_with_outliers.csv")
info_dataframe = pd.read_csv("datasets/outliers_info.csv")

with_steps = (
    raw_dataframe
    .set_index('index')
    .join(info_dataframe.set_index('index')[['steps_to_anomaly']],
          on='index',
          how='left',
          rsuffix='_info'
          )
    )
with_steps['outliers'] = (with_steps['steps_to_anomaly'] == 0).astype(int)
dataframe = with_steps.drop(columns='steps_to_anomaly')

target_column = 'e_mu_current'
anomaly_column = 'outliers'
window_size = 20
train_size = 0.75
end = len(dataframe)

scaler = RobustScaler()

train, _ = train_test_split(dataframe.drop(columns=anomaly_column).to_numpy(), train_size=train_size)
scaler.fit(train)

scaled_dataset = ClassicModelDataset(dataframe=dataframe,
                                     target_column=target_column,
                                     anomaly_column=anomaly_column,
                                     window_size=window_size,
                                     train_size=train_size,
                                     scaler=scaler)

scaled_dataset.setup(window_size, end, from_files=True)
pools, targets = scaled_dataset.get_catboost_pools(anomaly_split=False)
(scaled_train_pool, scaled_test_pool), = pools
(scaled_train_targets, scaled_test_targets), = targets

scaled_model = CatBoostRegressor()
# scaled_model.fit(scaled_train_pool, eval_set=scaled_test_pool)

summary = scaled_model.select_features(
    scaled_train_pool,
    eval_set=scaled_test_pool,
    features_for_select=f'0-{scaled_test_pool.num_col()-1}',
    num_features_to_select=50,
    steps=15,
    algorithm=EFeaturesSelectionAlgorithm.RecursiveByShapValues,
    shap_calc_type=EShapCalcType.Regular,
    train_final_model=True,
    plot=True
)

predictions = scaled_model.predict(scaled_test_pool)
predictions_array = np.zeros((predictions.shape[0], 7))
predictions_array[:, 0] = predictions
unscaled_predictions = scaler.inverse_transform(predictions_array)[:, 0]


targets_array = np.zeros((predictions.shape[0], 7))
targets_array[:, 0] = scaled_test_targets
unscaled_targets = scaler.inverse_transform(targets_array)[:, 0]


plt.figure(figsize=(10, 5))
sns.lineplot(x=range(100), y=unscaled_targets[200:300], label='QBER')
sns.lineplot(x=range(70, 100), y=unscaled_predictions[270:300], label='Prediction')
plt.savefig('prediction_example.png', dpi=200)
plt.show()


print(f'Catboost Model metrics:')
print(f'MAPE = {mean_absolute_percentage_error(unscaled_targets, unscaled_predictions):.5f}')
print(f'MSE = {mean_squared_error(unscaled_targets, unscaled_predictions):.5f}')
print(f'R2 = {r2_score(unscaled_targets, unscaled_predictions):.5f}')