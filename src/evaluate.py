import sys
import os
import pickle
import json
import pandas as pd
import numpy as np

from sklearn.metrics import (
    mean_squared_error as mse,
    mean_absolute_error as mae,
    mean_squared_log_error as msle,
    accuracy_score as acc
)

if len(sys.argv) != 4:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write('\tpython evaluate.py model-file featurized-directory metrics-file\n')
    sys.exit(1)

model_input = sys.argv[1]
test_input = os.path.join(sys.argv[2], 'test.csv')
metrics_file = sys.argv[3]


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

acc_score = acc(y_true, np.round(y_pred))

with open(metrics_file, 'w') as f:
    json.dump({
        'mse_loss': mse_loss,
        'mae_loss': mae_loss,
        'rmsle_loss': rmsle_loss,
        'accuracy_score': acc_score,
    }, f, indent=4)
