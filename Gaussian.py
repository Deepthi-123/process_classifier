import numpy as np
from matplotlib import pyplot as plt
import pandas
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
# Observations
names = ['time1','eig1','time2','eig2','time3','eig3','max','min']
dataset = pandas.read_csv('/home/deepthi/eig_rep_final.log', names = names, delim_whitespace = True)
names1 = ['index','time_1','val_1']
dat = pandas.read_csv('/home/deepthi/chui.log', names = names1, delim_whitespace = True, index_col=False)
time_1 = np.atleast_2d(dat['time_1'])
eig = dat['val_1'].map(float)
eig_1 = np.atleast_2d(eig)
kernel = RBF(10, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=9)
gp.fit(time_1, eig_1)
x = np.atleast_2d(np.arange(763.367, 765.845, 0.001))
y_pred, sigma = gp.predict(x, return_std = True)
fig = plt.figure()
p =  pandas.Series(x.flatten())
q =  pandas.Series(y_pred.flatten())
po = x.flatten()[::150]
qo = y_pred.flatten()
#print x.flatten()[::2000]
#plt.errorbar(po, qo, xerr = 0, yerr = sigma)
#plt.fill_between(np.concatenate([]))
#w = np.concatenate([x, x[::-1]])
#t = np.concatenate([y_pred - 1.9600 * sigma, (y_pred + 1.9600 * sigma)[::-1]]
w =qo+1.9600*sigma
t = (qo-1.9600*sigma)[::-1]
plt.plot(x.flatten(),w,'b')
plt.plot(x.flatten(), y_pred.flatten(),'r.')
plt.fill_between(x.flatten(),w,t)
#print w.size
#print t.size
dl_ = {'p':p,'q':q}
data = pandas.concat(dl_.values(), axis = 1, keys = dl_.keys())
data = data[['p','q']]
#plt.plot(data['p'], data['q'],'b-', label = u'prediction')
plt.xlim(763.367,765.845)
plt.show()
#params  = []
#params = gp.get_params(deep = True).hyperparameters

#print params
