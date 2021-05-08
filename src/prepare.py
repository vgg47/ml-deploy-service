import sys
import random
import os
import yaml

import numpy as np
import pandas as pd

if len(sys.argv) != 3:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write('\tpython prepare.py data-file output-directory\n')
    sys.exit(1)

params = yaml.safe_load(open('params.yaml'))['prepare']
split = params['split']
random.seed(params['seed'])

data_input = sys.argv[1]
train_output = os.path.join(sys.argv[2], 'train.csv')
test_output = os.path.join(sys.argv[2], 'test.csv')
os.makedirs(sys.argv[2], exist_ok=True)

data = pd.read_csv(data_input)
data.dropna(inplace=True)
test_indices = np.random.rand(len(data)) < split
data[test_indices].to_csv(test_output)
data[~test_indices].to_csv(train_output)
