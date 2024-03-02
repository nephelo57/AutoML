import pandas as pd
import numpy as np
from autogluon.tabular import TabularDataset, TabularPredictor
import warnings

warnings.filterwarnings('ignore')

path='userPred/data/'
train_data = pd.read_csv(path + 'train.csv')
test_data = pd.read_csv(path + 'test.csv')

train = TabularDataset(train_data.drop(['uuid'], axis=1))
test = TabularDataset(test_data.drop(['uuid'], axis=1))

predictor = TabularPredictor(label='target',
                             problem_type='binary',
                             eval_metric='f1').fit(train_data=train,
                                                   time_limit=60,
                                                   presets='medium_quality',
                                                   excluded_model_types=['CAT', 'NN_TORCH', 'FASTAI'])
