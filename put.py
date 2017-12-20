import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from peakutils.plot import plot as pplot
import numpy as np
import csv
import plotly.plotly as py
import scipy
from sklearn.decomposition import TruncatedSVD
from sklearn.random_projection import sparse_random_matrix
names = ['ind','eventtime', 'msg_stamp', 'symbols','q0_n','q1_n','q2_n','q3_n','q4_n','q5_n','q0','q1','q2','q3','q4','q5','q0_d','q1_d','q2_d','q3_d','q4_d','q5_d']
dataset = pandas.read_csv('/home/deepthi/spl.log', names = names, delim_whitespace = True)
#print dataset['q0']
dataset.loc[dataset.index[5000:5050],'q0':'q5'].to_csv('/home/deepthi/tes_2.log', sep ='\t')
fig, ax1 = plt.subplots()
ax1.plot(dataset.index[5000:15000], dataset['q0'][5000:15000], 'r')
ax1.plot(dataset.index[5000:15000], dataset['q1'][5000:15000], 'b')
ax1.plot(dataset.index[5000:15000], dataset['q2'][5000:15000], 'y')
ax1.plot(dataset.index[5000:15000], dataset['q3'][5000:15000], 'g')
ax1.plot(dataset.index[5000:15000], dataset['q4'][5000:15000], 'm')
ax1.plot(dataset.index[5000:15000], dataset['q5'][5000:15000], 'k')
name = ['ind','q_0','q_1','q_2','q_3','q_4','q_5']
data_2 = pandas.read_csv('/home/deepthi/check_2.log',names = name, delim_whitespace = True)
del data_2['ind']
data_p2 = data_2.as_matrix()
data_3 = pandas.read_csv('/home/deepthi/check_3.log',names = name, delim_whitespace = True)
del data_3['ind']
data_p3 = data_3.as_matrix()
data_4 = pandas.read_csv('/home/deepthi/check_4.log',names = name, delim_whitespace = True)
del data_4['ind']
data_p4 = data_4.as_matrix()
data_5 = pandas.read_csv('/home/deepthi/check_5.log',names = name, delim_whitespace = True)
del data_5['ind']
data_p5 = data_5.as_matrix()
#print len(data_p)
svd = TruncatedSVD(n_components = 1, n_iter = 7, random_state = 42)
svd.fit(data_p2)
print(svd.singular_values_)
svd.fit(data_p3)
print(svd.singular_values_)
svd.fit(data_p4)
print(svd.singular_values_)
svd.fit(data_p5)
print(svd.singular_values_)
#print data_p
  
