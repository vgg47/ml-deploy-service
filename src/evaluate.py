import sys
import os
import pickle
import json
import pandas as pd
import numpy as np
import yaml

from sklearn.metrics import (
    mean_squared_error as mse,
    mean_absolute_error as mae,
    mean_squared_log_error as msle,
    accuracy_score as acc
)

if len(sys.argv) != 5:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write('\tpython evaluate.py model-directory model-name featurized-directory metrics-directory\n')
    sys.exit(1)

params = yaml.safe_load(open('params.yaml'))['evaluate']

version = params['model_version']

model_input = os.path.join(sys.argv[1], f'{sys.argv[2]}_{version}.pkl')
test_input = os.path.join(sys.argv[3], 'test.csv')
loss_file = os.path.join(sys.argv[4], 'loss.json')
scores_file = os.path.join(sys.argv[4], 'scores.json')

os.makedirs(sys.argv[3], exist_ok=True)


def rmsle(y_true, y_pred):
    return msle(y_true, y_pred) ** 0.5


with open(model_input, 'rb') as fd:
    model = pickle.load(fd)

test = pd.read_csv(test_input, index_col=0)
x_test = test.drop(columns=['Times'])
y_true = test.Times

y_pred = model.predict(x_test)

mse_loss = mse(y_true, y_pred)
mae_loss = mae(y_true, y_pred)
rmsle_loss = rmsle(y_true, y_pred)

with open(loss_file, 'w') as f:
    json.dump({
        'mse': mse_loss,
        'mae': mae_loss,
        'rmsle': rmsle_loss,
    }, f, indent=4)

acc_score = acc(y_true, np.round(y_pred))

with open(scores_file, 'w') as f:
    json.dump({
        'accuracy': acc_score,
    }, f, indent=4)
