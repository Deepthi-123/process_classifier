import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import csv
names1 = ['eventtime', 'rostime.now', 'symbols', 'eigforce3d']
dataset1 = pandas.read_csv('/home/deepthi/topic_log/_20170420/20170420-10_08_01_eigenforce3d.log', names = names1) 
dataset1.to_csv('/home/deepthi/eig3d_tirol.log', header=False, index=False, sep='\t', mode='w')
dataset1['time_stamp'] = dataset1['eventtime'].map(str)+'.'+dataset1['rostime.now'].map(str)
del dataset1['eventtime']
del dataset1['rostime.now']
del dataset1['symbols']
cols = dataset1.columns.tolist()
cols = cols[-1:] + cols[:-1]
dataset1 = dataset1[cols]
dataset1['time_stamp'] = pandas.to_datetime(dataset1['time_stamp'], errors = 'coerce',  infer_datetime_format=True)
dataset_ = dataset1[1:].set_index(dataset1[1:].time_stamp)
del dataset_['time_stamp']
eig = dataset_.eigforce3d.resample('1L')
eig = pandas.Series(eig) 
eig.fillna(method = 'ffill', inplace = 'True')
x = dataset_.index.map(str)
xp = []
xp += ((float(i[14:16]))*60 + float(i[17:]) for i in x)
ser_time = pandas.Series(xp, index = dataset_.index)
ser_t = ser_time.resample('1L')
ser_t.index.map(str)
interpolated = ser_t.interpolate(method = 'linear')
#print interpolated
dl = {'eig': eig,'time': interpolated}
df = pandas.concat(dl.values(),axis = 1, keys = dl.keys())
df = df[['time','eig']]
df.to_csv('/home/deepthi/eig_rep.log',header=False, index=False, sep='\t', mode='w')
df_ = df.set_index(np.arange(len(eig)))
mask = ((df_['time'] >= 602.836) & (df_['time'] <= 605.339))
p = df_[mask]
mask_ = ((df_['time'] >= 763.367) & (df_['time'] <= 765.845 ))
q = df_[mask_]
_mask = ((df_['time'] >= 537.806) & (df_['time'] <= 541.198))
t = df_[_mask]
cols = ['time','eig']
keys = ['time1','val1','time2','val2','time3','val3']
l = p['time'].reset_index(drop = 'True')
m = q['time'].reset_index(drop = 'True')
u = t['time'].reset_index(drop = 'True')
r = p['eig'].reset_index(drop = 'True')
s = q['eig'].reset_index(drop = 'True')
v = t['eig'].reset_index(drop= 'True')
dfs = [r,s]
datfra = pandas.concat([l,r,m,s,u,v], axis = 1, keys = keys)
datfra['max_val'] = datfra[['val1','val2']].max(axis = 1)
datfra['min_val'] = datfra[['val1','val2']].min(axis = 1)
datfra.to_csv('/home/deepthi/eig_rep_final.log',header=False, index=False, sep='\t', mode='w')
#print datfra
datfra['MA_max']  = pandas.rolling_mean(datfra['max_val'], window = 15, center = True)
datfra['MA_min']  = pandas.rolling_mean(datfra['min_val'], window = 28, center  = True)
fig, ax1= plt.subplots()
ax2 = ax1.twiny()
ax3 = ax1.twiny()
ax4 = ax1.twiny()
ax5 = ax1.twiny()
#a = ax1.plot(datfra['time1'], datfra['val1'], 'k', label = 'eigen val1')
#ax1.set_xlim([602.836,605.337])
#b = ax2.plot(datfra['time2'],datfra['val2'], 'r', label = 'eigen val2')
#ax2.set_xlim([763.367,765.843])
c = ax3.plot(datfra['time2'],datfra['MA_max'], 'm', label = 'max_val')
ax3.set_xlim([763.367,765.843])
d = ax4.plot(datfra['time2'],datfra['MA_min'],'y', label = 'min_val')
ax4.set_xlim([763.367,765.843])
#e = ax5.plot(datfra['time3'],datfra['val3'],'m')
#ax5.set_xlim([537.806,541.198])
ax1.legend(loc = 'upper left')
ax2.legend(loc = 'upper right')
ax3.legend(loc = 'lower left')
ax4.legend(loc = 'lower right')
ax5.legend(loc = 'lower left')
plt.title('Eigen force vs time for one cyle of process')
plt.ylabel('Eigen-force')
plt.xlabel('time')
plt.show()
datfra['test'] = np.nan
#print datfra
def threshold_check(x):
   if ((x['val3'] < x['max_val']) & (x['val3'] > x['min_val'])):
      x['test'] = 1
   else:
      x['test'] = 2
   return x
datfra.apply(threshold_check, axis = 1)
#print datfra['val1']
