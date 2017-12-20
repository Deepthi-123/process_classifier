import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import pylab as py
from scipy.stats import norm
from scipy import optimize
import numpy as np
import csv
names = ['index', 'skill', 'time','eigen_force']
dataset = pandas.read_csv('/home/deepthi/class_26.log', names = names, delim_whitespace = True)
del dataset['index']
names1 = ['index', 'skill', 'time','eigen_force']
dataset1 = pandas.read_csv('/home/deepthi/class_26_2.log', names = names1, delim_whitespace = True)
del dataset1['index']
names2 = ['index', 'skill', 'time','eigen_force']
dataset2 = pandas.read_csv('/home/deepthi/class_data_3.log', names = names2, delim_whitespace = True)
del dataset2['index']
#print dataset
e1 = pandas.Series(dataset['eigen_force'])
e2 = pandas.Series(dataset1['eigen_force'])
e3 = pandas.Series(dataset2['eigen_force'])
ex = e1.append(e2)
data = ex.append(e3)
print len(data)
fig, ax1 = plt.subplots()
mu, sigma = norm.fit(data)
n, bins, patches = plt.hist(data, 20, normed=1, facecolor='green', alpha=0.75)
y = mlab.normpdf( bins, mu, sigma)
#l = plt.plot(bins, y, 'r--', linewidth=2)
ser = pandas.Series(pandas.rolling_mean(bins, window = 3))
#ax1.plot(dataset['eigen_force'], dataset['skill'],'rx', label = 'for time 10:09:57')
#ax1.plot(dataset1['eigen_force'], dataset1['skill'],'bx', label = 'for time 10:10:50')
#ax1.plot(dataset2['eigen_force'], dataset2['skill'],'mx', label = 'for time 10:12:37')
#mu, sigma = norm.fit(dataset['eigen_force'])
#xmin, xmax = plt.xlim()
#data_ = plt.hist(data, bins = 10)
#data2 = plt.hist(dataset2['eigen_force'], bins = 100)
#p = norm.pdf(x,mu,sigma)
#ax1.plot(x, p, 'r', lw = 3)
#print pcov
#plt.plot(bins, ser , 'm--', lw = 3)
print ser
#print mu
#print xmin
#ax1.plot(dataset['eigen_force'], dataset['skill'],'rx')
#ax1.set_ylim([25,27])
#plt.xlabel('Eigen Force')
#plt.ylabel('skill')
#plt.title('Skill vs Eigen Force')
#ax1.legend(loc = 'best')
#plt.show()
#print data
