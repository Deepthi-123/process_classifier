import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal._peak_finding import argrelmax, argrelmin
import csv
#names = ['time_stamp','eigen_force']
#dataset = pandas.read_csv('/home/deepthi/eigen_resamples.log', names = names, sep = ',')
#dataset_ = dataset.set_index('time_stamp')
#eigen = pandas.Series(dataset_['eigen_force'])
names1 = ['eventtime', 'msg_stamp', 'symbols','eigen_force']
dataset1 = pandas.read_csv('/home/deepthi/task1_eig/label1_0.log', names = names1, sep = ',')#read the final_wrench.log file
dataset1.to_csv('/home/deepthi/new_test/lab1_0.log',index=True, sep=',', mode='w') #and write to final_wrench_mod. Dataframe created for ease of data manip
lt = []
for i in dataset1['msg_stamp'].map(str):
   lt.append('.' + i.zfill(3))
dataset1['ros_time'] = pandas.Series(lt)
dataset1['time_stamp'] = dataset1['eventtime'].map(str) + dataset1['ros_time']
del dataset1['eventtime']
del dataset1['msg_stamp']
del dataset1['ros_time']
del dataset1['symbols']
dataset1['time_stamp'] = pandas.to_datetime(dataset1['time_stamp'], errors = 'coerce')
dataset1_ = dataset1.set_index('time_stamp')
eigen = dataset1_.eigen_force.resample('1L')
eigen.fillna(method = 'ffill', inplace = True)
#print dataset1_
y = np.array(eigen)
p = argrelmin(y)
q = argrelmax(y)
p = list(p)
q = list(q)
for i in p:
   c = i.flatten()
for i in q:
   d = i.flatten()
c = list(c)
d = list(d)
fig, ax = plt.subplots()
#ax2.plot(range(len(skill)), skill, 'g', lw = 2)
#ax.set_xlim([0, len(eigen)])
#ax.plot(range(len(eigen)), eigen, 'b', lw = 2)
ax.set_xlim([0, len(eigen)])
ax.plot(range(len(eigen)), eigen, 'r', lw = 2)
#ax3.axvline(87)
#ax3.axvline(65)
#ax3.axvline(81)
plt.show()
#print p
#print d

