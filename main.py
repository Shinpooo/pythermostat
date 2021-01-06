import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, KFold
from statsmodels.graphics.tsaplots import plot_pacf

from darkgreybox.model import DarkGreyModel, Ti, TiTe, TiTh, TiTeTh, TiTeThRia
from darkgreybox.fit import darkgreyfit
from utils import plot
from utils import rmse

# the duration of a record
rec_duration = 1 # hour

train_df = pd.read_csv('./data/demo_data.csv', index_col=0, parse_dates=True)

input_X = train_df[['Ph', 'Ta', 'Th']]
input_y = train_df['Ti']

input_X['Ti0'] = input_y
input_X['Th0'] = input_y
input_X['Te0'] = input_y - 2 

X_train, X_test, y_train, y_test = train_test_split(input_X, input_y, test_size=5 / 33, shuffle=False)

train_params_Ti = {
    'Ti0': {'value': X_train.iloc[0]['Ti0'], 'vary': False},
    'Ci': {'value': 1},
    'Ria': {'value': 1},
}

train_params_TiTe = {
    'Ti0': {'value': X_train.iloc[0]['Ti0'], 'vary': False},
    'Te0': {'value': X_train.iloc[0]['Ti0'], 'vary': True},
    'Ci': {'value': 1},
    'Ce': {'value': 1},
    'Rie': {'value': 1},
    'Rea': {'value': 1},
}

train_params_TiTh = {
    'Ti0': {'value': X_train.iloc[0]['Ti0'], 'vary': False},
    'Th0': {'value': X_train.iloc[0]['Th0'], 'vary': False},    
    'Ci': {'value': 1},
    'Ch': {'value': 2.55, 'vary': False},
    'Ria': {'value': 1},
    'Rih': {'value': 0.65, 'vary': False}
}

train_params_TiTeTh = {
    'Ti0': {'value': X_train.iloc[0]['Ti0'], 'vary': False},
    'Te0': {'value': X_train.iloc[0]['Te0'], 'vary': True, 'min': 10, 'max': 25},
    'Th0': {'value': X_train.iloc[0]['Th0'], 'vary': False},    
    'Ci': {'value': 1},
    'Ch': {'value': 2.55, 'vary': False},
    'Ce': {'value': 1},
    'Rie': {'value': 1},
    'Rea': {'value': 1},
    'Rih': {'value': 0.65, 'vary': False}
}

train_params_TiTeThRia = {
    'Ti0': {'value': X_train.iloc[0]['Ti0'], 'vary': False},
    'Te0': {'value': X_train.iloc[0]['Te0'], 'vary': True, 'min': 10, 'max': 25},
    'Th0': {'value': X_train.iloc[0]['Th0'], 'vary': False},    
    'Ci': {'value': 1},
    'Ch': {'value': 2.55, 'vary': False},
    'Ce': {'value': 1},
    'Rie': {'value': 1},
    'Rea': {'value': 1},
    'Ria': {'value': 1},
    'Rih': {'value': 0.65, 'vary': False}
}


# ?
ic_params_map = {
    'Ti0': lambda X_test, y_test, train_result: y_test.iloc[0],
    'Th0': lambda X_test, y_test, train_result: y_test.iloc[0],
    'Te0': lambda X_test, y_test, train_result: train_result.Te[-1],
}

models = [Ti(train_params_Ti, rec_duration),
          TiTe(train_params_TiTe, rec_duration),
          TiTh(train_params_TiTh, rec_duration),
          TiTeTh(train_params_TiTeTh, rec_duration),
          TiTeThRia(train_params_TiTeThRia, rec_duration)]

prefit_splits = KFold(n_splits=int(len(X_train) / 24), shuffle=False).split(X_train)
# for x,y in prefit_splits:
#     print(x, y)

error_metric = rmse
prefit_filter = lambda error: abs(error) < 2
method = 'nelder'
df = darkgreyfit(models, X_train, y_train, X_test, y_test, ic_params_map, error_metric,
                 prefit_splits=prefit_splits, prefit_filter=prefit_filter, reduce_train_results=True, 
                 method=method, n_jobs=-1, verbose=10)

select_idx = df[('train', 'error')].argmin()

model = df.loc[select_idx, ('train', 'model')]
train_results = df.loc[select_idx, ('train', 'model_result')]
test_results = df.loc[select_idx, ('test', 'model_result')]

print(model.result.params)
plot(y_train, train_results, 'TiTeThRia (Train)')
plot(y_test, test_results, 'TiTeThRia (Test)')
