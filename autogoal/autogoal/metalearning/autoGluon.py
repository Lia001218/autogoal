from autogluon.tabular import TabularDataset, TabularPredictor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


# pd_data.rename(columns={'415': 'target'}, inplace=True)
# print(pd_data)
# data_root_Y = np.load('X_train.npy')
train_data = pd.read_csv('text_train_data.csv')

data = TabularDataset(train_data)
# test_data = TabularDataset(data_root_Y)
print(train_data)

predictor = TabularPredictor(label='394').fit(train_data=data, presets='medium_quality')

results = predictor.evaluate('text_test_data.csv')
print("results: \n", results)