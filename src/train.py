import sys
import os
import pickle
import yaml
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write('\tpython train.py featurized-directory model-file\n')
    sys.exit(1)

params = yaml.safe_load(open('params.yaml'))['train']
objective = params['objective']
nthread = params['nthread']
eval_metric = params['eval_metric']
n_estimators = params['n_estimators']

train_input = os.path.join(sys.argv[1], 'train.csv')
model_output = sys.argv[2]
os.makedirs(os.path.dirname(model_output), exist_ok=True)

train = pd.read_csv(train_input, index_col=0)
fit, eval = train_test_split(train, train_size=0.9)
x_fit = fit.drop(columns=['Times'])
y_fit = fit.Times
x_eval = eval.drop(columns=['Times'])
y_eval = eval.Times

param = {
    'objective': objective,
    'nthread': nthread,
    'eval_metric': eval_metric,
    'n_estimators': n_estimators,
}

model = xgb.XGBRegressor(**param)
model.fit(x_fit, y_fit, eval_set=[(x_eval, y_eval)], early_stopping_rounds=30)

with open(model_output, 'wb') as fd:
    pickle.dump(model, fd)
