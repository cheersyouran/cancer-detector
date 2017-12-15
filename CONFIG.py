from datafile import *

CATEGORY = 'L_ML'
PATH = '/Users/wangchengming/Documents/HKUST/6000B/Project3'


if CATEGORY == 'L_ML':
	PATCH_SIZE = 50
	SIGMA = 1
	THRESH = 90
	LR = 0.0001
	DROPOUT = 0.5
	train_sub = L_ML_train
	val_sub = L_ML_val
	test_sub = L_ML_test
elif CATEGORY == 'R_ML':
	PATCH_SIZE = 50
	SIGMA = 1
	THRESH = 90
	LR = 0.0001
	DROPOUT = 0.5
	train_sub = R_ML_train
	val_sub = R_ML_val
	test_sub = R_ML_test
elif CATEGORY == 'L_CC':
	PATCH_SIZE = 100
	SIGMA = 4
	THRESH = 75
	LR = 0.0008
	DROPOUT = 0.7
	train_sub = L_CC_train
	val_sub = L_CC_val
	test_sub = L_CC_test
elif CATEGORY == 'R_CC':
	PATCH_SIZE = 100
	SIGMA = 4
	THRESH = 75
	LR = 0.0008
	DROPOUT = 0.7
	train_sub = R_CC_train
	val_sub = R_CC_val
	test_sub = R_CC_test


NUM_SAMPLE = len(train_sub)
NUM_VAL = len(val_sub)
NUM_TEST = len(test_sub)

