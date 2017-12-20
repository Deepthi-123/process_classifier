import pandas as pd
import sys
import scipy
import numpy
import matplotlib
import sklearn
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

with open('/home/ipa325/eigenforce3d.log') as infile:
     with open('/home/ipa325/eigmodified.log','w') as ofile:
          for i in range(8000):
             line = infile.next()
             ofile.write(line.replace('[dx, dy, dz, da, db, dc, joint_0, joint_1, joint_2, joint_3, joint_4, joint_5], ', '').replace('[','').replace(']','').replace(', ',','))
          names = ['eventtime', 'rostime.now', 'symbols', 'eigforce3d']
          dataset = pd.read_csv('/home/ipa325/eigmodified.log', names = names) 
dataset.to_csv('/home/ipa325/neweig.log', header=True, index=False, sep='\t', mode='w')
df1 = pd.DataFrame(dataset, columns =['rostime.now','eigforce3d'])
df1[['eigforce3d','rostime.now']] = df1[['eigforce3d','rostime.now']].apply(pd.to_numeric, errors = 'ignore')
df1.iloc[100:1000].plot(y='eigforce3d', style='b')
plt.ylabel('Eigen Force 3d')
plt.xlabel('ROS time')
plt.title('Eigen Force Plot')
plt.show()
