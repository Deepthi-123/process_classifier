import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
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
import csv


with open('/home/ipa325/newlog.log') as i_file:
     with open('/home/ipa325/newl.log','w') as o_file:   
       for i in range(1000):
          line = i_file.next()
          o_file.write(line.replace('[dx, dy, dz, da, db, dc, joint_0, joint_1, joint_2, joint_3, joint_4, joint_5], ', '').replace('[','').replace(']','').replace(', ',','))
       names = ['eventtime', 'rostime.now', 'symbols', 'dx', 'dy', 'dz', 'da', 'db', 'dc', 'joint_0', 'joint_1', 'joint_2', 'joint_3', 'joint_4', 'joint_5']
       dataset = pandas.read_csv('/home/ipa325/newl.log', names = names) 
dataset.to_csv('/home/ipa325/newx.log', header=True, index=False, sep='\t', mode='w')



          

   
    

   
