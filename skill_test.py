import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib.lines as lines
from peakutils.plot import plot as pplot
import numpy as np
import csv
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.tools import FigureFactory as FF
import scipy
import peakutils
names1 = ['index', 'skill', 'time','eig_val']
dataset1 = pandas.read_csv('/home/deepthi/class_27.log', names = names1, delim_whitespace = True)
eig_1 = pandas.Series(dataset1['eig_val'])
names2 = ['index', 'skill', 'time','eig_val']
dataset2 = pandas.read_csv('/home/deepthi/class_27_2.log', names = names2, delim_whitespace = True)
eig_2 = pandas.Series(dataset2['eig_val'])
names3 = ['index', 'skill', 'time','eig_val']
dataset3 = pandas.read_csv('/home/deepthi/class_27_uu3.log', names = names3, delim_whitespace = True)
eig_3 = pandas.Series(dataset3['eig_val'])
#names4 = ['index','skill', 'time','eig_val']
#dataset4 = pandas.read_csv('/home/deepthi/class_27_4.log', names = names4, delim_whitespace = True)
#eig_4 = pandas.Series(dataset4['eig_val'])
dict_ = {'val_1':eig_1,'val_2': eig_2,'val_3':eig_3}
#print dataset4['eig_val']
dataset = pandas.concat(dict_.values(), axis = 1, keys=dict_.keys())
dataset['max'] = dataset[['val_1','val_2','val_3']].max(axis = 1) + 0.2
dataset['min'] = dataset[['val_1','val_2','val_3']].min(axis = 1) - 0.2
data = dataset.fillna(method= 'ffill')
fig, ax1 = plt.subplots()
ax2 = ax1.twiny()
ax3 = ax1.twiny()
ax4 = ax1.twiny()
dataset['ma_set1'] = pandas.rolling_mean(dataset['val_1'], window = 30, center = True)
dataset['ma_set2'] = pandas.rolling_mean(dataset['val_2'], window = 30, center = True)
dataset['ma_set3'] = pandas.rolling_mean(dataset['val_3'], window = 30, center = True)
ax1.plot(dataset.index, dataset['ma_set1'], 'g', lw = 2)
#ax1.set_xlim([0,535])
ax2.plot(dataset.index, dataset['ma_set2'], 'y', lw = 2)
#ax2.set_xlim([0,539])
ax3.plot(dataset.index, dataset['ma_set3'], 'b', lw = 2)
#ax3.set_xlim([0,543])
dataset['max'] = dataset[['ma_set1','ma_set2','ma_set3']].max(axis = 1) + 0.2
dataset['min'] = dataset[['ma_set1','ma_set2','ma_set3']].min(axis = 1) - 0.2
dataset['ma_ma1'] = pandas.rolling_mean(dataset['max'], window = 30, center = True)
dataset['ma_ma2'] = pandas.rolling_mean(dataset['min'], window = 30, center = True)
ax3.plot(dataset.index, dataset['ma_ma1'], 'r', lw = 2)
ax3.plot(dataset.index, dataset['ma_ma2'], 'k', lw = 2)
#ax1.plot(dataset.index, dataset['ma_set3'], 'b', lw = 2)
print len(dataset1)
print len(dataset2)
print len(dataset3)
print len(dataset)
#ax1.set_xlim([0,535])
#ax1.plot(dataset2.index, dataset2['eig_val'], 'r')
#ax2.set_xlim([0,539])
#ax1.plot(dataset3.index, dataset3['eig_val'], 'm')
#ax1.plot(dataset4.index, dataset4['eig_val'], 'g')
#dataset['max']  = pandas.rolling_mean(dataset['max'], window = 20, center = True)
#indices = peakutils.indexes(data['MA_max'], thres=0.01*max(data['MA_max']), min_dist=1)
#plt.figure(figsize=(10,6))
#pplot(data.index, data['MA_max'], indices)
#plt.title('First estimate')
#data['MA_max']  = pandas.rolling_mean(data['max'], window = 50, center = True)
#plt.plot(data.index, data['MA_max'],'g', lw = 2)
#data['MA_min']  = pandas.rolling_mean(data['min'], window = 50, center  = True)
#plt.plot(dataset.index, data['MA_min'],'y', lw = 2)
#plt.tight_layout()
plt.show()
#print dataset





