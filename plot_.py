import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import csv
from scipy.interpolate import splprep, splev
names = ['time_stamp','q0','q1','q2','q3','q4','q5','qd0','qd1','qd2','qd3','qd4','qd5']
dataset = pandas.read_csv('/home/deepthi/folder_plot/joint_check.log', names = names, delim_whitespace = True)
dataset['time_stamp'] = pandas.to_datetime(dataset['time_stamp'], errors = 'coerce')
dataset_ = dataset.set_index('time_stamp')
del dataset_['qd0']
del dataset_['qd1']
del dataset_['qd2']
del dataset_['qd3']
del dataset_['qd4']
del dataset_['qd5']
q0 = dataset_.q0.resample('1L')
q0 = pandas.Series(q0)
q1 = dataset_.q1.resample('1L')
q1 = pandas.Series(q1)
q2 = dataset_.q2.resample('1L')
q2 = pandas.Series(q2)
q3 = dataset_.q3.resample('1L')
q3 = pandas.Series(q3)
q4 = dataset_.q4.resample('1L')
q4 = pandas.Series(q4)
q5 = dataset_.q5.resample('1L')
q5 = pandas.Series(q5)
q0_ =  q0.fillna(method = 'ffill')
q1_ =  q0.fillna(method = 'ffill')
q2_ =  q0.fillna(method = 'ffill')
q3_ =  q0.fillna(method = 'ffill')
q4_ =  q0.fillna(method = 'ffill')
q5_ =  q0.fillna(method = 'ffill')
dl_ = {'q0':q0_, 'q1':q1_,'q2':q2_,'q3':q3_,'q4':q4_, 'q5':q5_}
data = pandas.concat(dl_.values(), axis = 1, keys = dl_.keys())
data = data[['q0','q1','q2','q3','q4','q5']]
q_0 = np.empty(q0_, dtype = object)
q_1 = np.empty(q1_, dtype = object)
x = [q_0, q_1]
#print q_0
tck, u = splprep(x, s=1)


