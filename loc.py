import pandas
from pandas.tools.plotting import scatter_matrix
from scipy.interpolate import splprep, splev
import matplotlib.pyplot as plt
import numpy as np
import csv
names = ['eventtime', 'msg_stamp', 'symbols', 'j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'q0', 'q1', 'q2' ,'q3', 'q4', 'q5', 'qd0','qd1','qd2','qd3','qd4','qd5']
dataset = pandas.read_csv('/home/deepthi/c_6.log', names = names, delim_whitespace = False)#read the final_wrench.log file
dataset['time_stamp'] = dataset['eventtime'].map(str)+'.'+ dataset['msg_stamp'].map(str)
dataset['time_stamp']
#dataset_ = dataset.set_index('time_stamp')
del dataset['eventtime']
del dataset['msg_stamp']
del dataset['symbols']
del dataset['j0']
del dataset['j1']
del dataset['j2']
del dataset['j3']
del dataset['j4']
del dataset['j5']
del dataset['qd0']
del dataset['qd1']
del dataset['qd2']
del dataset['qd3']
del dataset['qd4']
del dataset['qd5']
dataset['time_stamp'] = pandas.to_datetime(dataset['time_stamp'], errors = 'coerce')
dataset_ = dataset.set_index('time_stamp')
q0_ = dataset_.q0.resample('1L')
q0_ = pandas.Series(q0_)
q1_ = dataset_.q1.resample('1L')
q1_ = pandas.Series(q1_)
q2_ = dataset_.q2.resample('1L')
q2_ = pandas.Series(q2_)
q3_ = dataset_.q3.resample('1L')
q3_ = pandas.Series(q3_)
q4_ = dataset_.q4.resample('1L')
q4_ = pandas.Series(q4_)
q5_ = dataset_.q5.resample('1L')
q5_ = pandas.Series(q5_)
dl = {'0':q0_,'1':q1_, '2':q2_, '3':q3_, '4':q4_, '5':q5_}
data = pandas.concat(dl.values(), axis = 1, keys = dl.keys())
#q_0 = q0_.fillna(method = 'ffill')
#q_1 = q1_.fillna(method = 'ffill')
#q_2 = q2_.fillna(method = 'ffill')
#q_3 = q3_.fillna(method = 'ffill')
#q_4 = q4_.fillna(method = 'ffill')
#q_5 = q5_.fillna(method = 'ffill')
x = [dataset_['q0'],dataset_['q1'],dataset_['q2'],dataset_['q3'],dataset_['q4'],dataset_['q5']]
tck, u = splprep(x, s=0)
dataset_['eig_force'] = [0, -0.01124, 0.02616, 0.02616, 0.02616, 0.02616, 0.01774, 0.0177, -0.01688, -0.06637, -0.06637, -0.06637 -0.3394, -0.4698, -0.4906, -0.4906, -0.486, -0.533, -0.5765, -0.5765, -0.683, -0.352, -0.4388, -0.2588, -0.3131, -0.09655, -0.09655, -0.1595, 0.0295, -0.1625, -0.283, 0.2022, 0.0061]
dataset_['red.'] = u
plt.plot(dataset_['red.'], dataset_['eig_force'], 'r', lw = 2)
plt.xlabel('Spline parameter u')
plt.ylabel('Eigen Force')
plt.title('Eigen force vs position parameter')
plt.show()
