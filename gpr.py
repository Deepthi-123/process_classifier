import pandas
import csv
import numpy as np
from sklearn import datasets
from sklearn.model_selection import cross_val_predict
from sklearn import linear_model
import matplotlib.pyplot as plt
lr = linear_model.LinearRegression()
names = ['time1', 'eig1', 'time2','eig2','time3','eig3','max','min']
dataset = pandas.read_csv('/home/deepthi/eig_rep_final.log', names = names, delim_whitespace = True)
p = np.array(dataset['time1']).reshape(-1,1)
q = np.array(dataset['eig1']).reshape(-1,1)
X_train = p[:-80]
X_test = p[-20:]
Y_train = q[:-80]
Y_test  = q[-20:]
regr = linear_model.LinearRegression()
# Train the model using the training sets
regr.fit(X_train, Y_train)
# Make predictions using the testing set
diabetes_y_pred = regr.predict(X_test)
# The coefficients
print p
