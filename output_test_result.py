
import pandas as pd
from CONFIG import *

L_ML = pd.read_csv(PATH + 'L_ML.csv')
R_ML = pd.read_csv(PATH + 'R_ML.csv')
L_CC = pd.read_csv(PATH + 'L_CC.csv')
R_CC = pd.read_csv(PATH + 'R_CC.csv')

result = pd.concat([L_ML, R_ML, L_CC, R_CC])
test = pd.read_table(PATH + '/Dataset_A/test.txt', header=None)

# result.to_csv('test_result.txt', index=None, header=None, sep=' ')