import pandas
import numpy as np
import os, os.path, sys, shutil
import glob
from shutil import copyfile
from cStringIO import StringIO
src = '/home/ipa325/Pew1'
filelist = os.listdir(src)
lst = []
for i in range(len(filelist)):
   if 'production_video' in filelist[i]:
      lst += filelist[i].replace(filelist[i][17:],',').replace('_',':').replace('-',' ')
lst = ''.join(lst)
lst_ = lst.split(',')
df1 = pandas.DataFrame(lst_, columns = ['time_stamp'])
df1['time_stamp'] = pandas.to_datetime(df1.time_stamp) 
time_ = pandas.Series(df1.sort('time_stamp'))
ser = pandas.Series(['True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True',
'True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True',
'True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True','True',
'True','True','True','True','True','False','False','False','False','False','False','False','False','False','False','True','True','True','True','True','True','True','True','True','True'])
dl_list = {'time': time_,
           'val': ser }
dataf = pandas.concat(dl_list.values(), axis=1, keys=dl_list.keys())
dataf = dataf[['time','val']]
data_fin = dataf.reset_index().set_index('time')
data_fin = data_fin.apply(pandas.to_numeric, errors = 'coerce')
del data_fin['index']
val = data_fin.val.resample('1L')
val_ = pandas.Series(val)
#val.loc['2017-04-20 10:42:56.000'].fillna('False',inplace = 'True')
#val_.loc['20170420 10:42:56.000':'20170420 10:43:18.000'] = ['False']
#val.loc['2017-04-20 10:09:57.000'].fillna('True', inplace = True)
val_.drop(val_.index['2017-04-20 10:42:56.000'])
print val_.loc['2017-04-20 10:42:56.000']
   
#val_.to_csv('/home/ipa325/resampled.log', mode = 'w')
#data_fin.to_csv('/home/ipa325/boolVsDateresampled.log', mode = 'w')





  
