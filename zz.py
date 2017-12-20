import numpy as np
import pandas
import csv
from scipy.interpolate import splprep, splev
names = ['ind','q0','q1','q2','q3','q4','q5']
dataset = pandas.read_csv('/home/deepthi/check_.log', names = names, delim_whitespace = True)
del dataset['ind']
q0 = np.array(dataset['q0'])
q1 = np.array(dataset['q1'])
q2 = np.array(dataset['q2'])
q3 = np.array(dataset['q3'])
q4 = np.array(dataset['q4'])
q5 = np.array(dataset['q5'])
x = [q0,q1,q2,q3,q4,q5]
tck, u = splprep(x, s=0)
print u

