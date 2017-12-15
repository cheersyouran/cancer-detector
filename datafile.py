import numpy as np
import pandas as pd
from keras.utils import np_utils
import os
from sklearn.utils import resample

PATH = '/Users/wangchengming/Documents/HKUST/6000B/Project3'
CATEGORY = 'L_ML'

np.random.seed(27)

train = pd.read_table(PATH + '/Dataset_A/train.txt', header=None)
train.columns = ['id', 'label']
y_train = train.label.values.reshape((-1, 1))
y_train = np_utils.to_categorical(y_train, 2)
val = pd.read_table(PATH + '/Dataset_A/val.txt', header=None)
val.columns = ['id', 'label']
y_val = val.label.values.reshape((-1, 1))
y_val = np_utils.to_categorical(y_val, 2)
test = pd.read_table(PATH + 'Dataset_A/test.txt', header=None)
test.columns = ['id']

img_path_li = os.listdir(PATH + '/Dataset_A/data/')
dicts = dict(zip(pd.Series(img_path_li).apply(lambda x: str(x)[:8]).values, img_path_li))
full_train = train.copy()
full_val = val.copy()
full_test = test.copy()
full_train['path'] = full_train['id'].astype(str).map(dicts)
full_val['path'] = full_val['id'].astype(str).map(dicts)
full_test['path'] = full_test['id'].astype(str).map(dicts)

# Split different views of the same patient
R_CC_path, L_CC_path, R_ML_path, L_ML_path = [], [], [], []
for img_path in img_path_li:
    if 'R_CC' in img_path:
        R_CC_path.append(img_path)
    elif 'L_CC' in img_path:
        L_CC_path.append(img_path)
    elif 'R_ML' in img_path:
        R_ML_path.append(img_path)
    elif 'L_ML' in img_path:
        L_ML_path.append(img_path)

R_CC_train = full_train[full_train.path.isin(R_CC_path)]
L_CC_train = full_train[full_train.path.isin(L_CC_path)]
R_ML_train = full_train[full_train.path.isin(R_ML_path)]
L_ML_train = full_train[full_train.path.isin(L_ML_path)]

R_CC_val = full_val[full_val.path.isin(R_CC_path)]
L_CC_val = full_val[full_val.path.isin(L_CC_path)]
R_ML_val = full_val[full_val.path.isin(R_ML_path)]
L_ML_val = full_val[full_val.path.isin(L_ML_path)]

R_CC_test = full_test[full_test.path.isin(R_CC_path)]
L_CC_test = full_test[full_test.path.isin(L_CC_path)]
R_ML_test = full_test[full_test.path.isin(R_ML_path)]
L_ML_test = full_test[full_test.path.isin(L_ML_path)]


# Balance 0 and 1
if CATEGORY == 'L_ML':
	target = L_ML_train
elif CATEGORY == 'R_ML':
	target = R_ML_train
elif CATEGORY == 'L_CC':
	target = L_CC_train
elif CATEGORY == 'R_CC':
	target = R_CC_train


pos = target[target.label == 1]
neg = target[target.label == 0]
pos = resample(pos, replace=True, n_samples=len(neg)-1)
target = pd.concat([pos, neg])
target = target.iloc[np.random.permutation(list(range(len(target))))]
