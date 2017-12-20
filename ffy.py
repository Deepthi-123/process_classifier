import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import csv
names_n = ['eventtime', 'msg_stamp', 'symbols','status','skill']
datan = pandas.read_csv('/home/ipa325/topic_log/_20170420/20170420-10_08_01_status.log', names = names_n, delim_whitespace = False)#read the final_wrench.log file
datan.to_csv('/home/ipa325/status_test_check_1.log',index=True, sep='\t', mode='w') #and write to final_wrench_mod. Dataframe created for ease of data manip
datan['time_stamp'] = datan['eventtime'].map(str)+'.'+ datan['msg_stamp'].map(str)
datan['time_stamp']
del datan['eventtime']
del datan['msg_stamp']
del datan['symbols']
cols = datan.columns.tolist()
cols = cols[-1:] + cols[:-1]
datan = datan[cols]
datan['time_stamp'] = pandas.to_datetime(datan['time_stamp'], errors = 'coerce')
datan_ = datan.set_index('time_stamp')
print datan_

