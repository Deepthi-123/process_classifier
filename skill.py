import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import csv
names_n = ['eventtime', 'msg_stamp', 'symbols','status','skill']
datan = pandas.read_csv('/home/ipa325/topic_log/_20170420/20170420-10_08_01_status.log', names = names_n, delim_whitespace = False)#read the final_wrench.log file
datan['time_stamp'] = datan['eventtime'].map(str)+'.'+datan['msg_stamp'].map(str)
datan = datan[['time_stamp','symbols', 'status','skill','eventtime','msg_stamp']]
del datan['eventtime']
del datan['symbols']
del datan['status']
datan['time_stamp'] = pandas.to_datetime(datan['time_stamp'], errors = 'coerce')
dataset = datan.set_index('time_stamp')
skill = dataset.skill.resample('1L')
skill.fillna(method= 'ffill', inplace = True)
dataset = dataset[['msg_stamp','skill']]
time = dataset.time.resample()
print dataset
#fig, ax = plt.subplots

