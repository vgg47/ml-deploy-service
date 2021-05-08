import os
import sys
import pandas as pd
import re

if len(sys.argv) != 3 and len(sys.argv) != 5:
    sys.stderr.write('Arguments error. Usage:\n')
    sys.stderr.write('\tpython featurize.py input-directory output-directory\n')
    sys.exit(1)

train_input = os.path.join(sys.argv[1], 'train.csv')
test_input = os.path.join(sys.argv[1], 'test.csv')
train_output = os.path.join(sys.argv[2], 'train.csv')
test_output = os.path.join(sys.argv[2], 'test.csv')
os.makedirs(os.path.join(sys.argv[2]), exist_ok=True)

def number_of_characters(pswrd):
    return len(pswrd) * 4

def uppercase_letters(pswrd):
    return (len(pswrd) - sum(1 for c in pswrd if c.isupper())) * 2

def lowercase_letters(pswrd):
    return (len(pswrd) - sum(1 for c in pswrd if c.islower())) * 2

def number_of_numbers(pswrd):
    return len(re.findall(r'\d', pswrd)) * 4

def number_of_symbols(pswrd):
    return (len(pswrd) - len(re.findall(r'\w', pswrd))) * 6

def middel_numbers_or_symbols(pswrd):
    if len(pswrd) > 2:
        pswrd = pswrd[1:-1]
        return ((len(pswrd) - len(re.findall(r'\w', pswrd))) + len(re.findall(r'\d', pswrd))) * 2
    else:
        return 0

def letters_only(pswrd):
    return len(pswrd) * pswrd.isalpha()

def numbers_only(pswrd):
    return len(pswrd) * pswrd.isdigit()

def consecutive_uppercase_letters(pswrd):
    score = 0
    for l, r in zip(pswrd, pswrd[1:]):
        score += l.isupper() and r.isupper()
    return score * 2

def consecutive_lowercase_letters(pswrd):
    score = 0
    for l, r in zip(pswrd, pswrd[1:]):
        score += l.islower() and r.islower()
    return score * 2

def consecutive_numbers(pswrd):
    score = 0
    for l, r in zip(pswrd, pswrd[1:]):
        score += l.isdigit() and r.isdigit()
    return score * 2

feature_funcs = [
    number_of_characters,
    uppercase_letters,
    lowercase_letters,
    number_of_numbers,
    number_of_symbols,
    middel_numbers_or_symbols,
    letters_only,
    numbers_only,
    consecutive_uppercase_letters,
    consecutive_lowercase_letters,
    consecutive_numbers
]

data = pd.read_csv(train_input, index_col=0, dtype={'Password': str, 'Times': int})
for f in feature_funcs:
    data[f.__qualname__] = data.Password.apply(f).astype('int8', copy=False)
data.drop(columns=['Password']).to_csv(train_output)

data = pd.read_csv(test_input, index_col=0, dtype={'Password': str, 'Times': int})
for f in feature_funcs:
    data[f.__qualname__] = data.Password.apply(f).astype('int8', copy=False)
data.drop(columns=['Password']).to_csv(test_output)
