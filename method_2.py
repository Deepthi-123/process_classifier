import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import csv
names1 = ['eventtime','msg_stamp','symbols','skill-name','skill id']
dataset1 = pandas.read_csv('/home/ipa325/method_2/status_log.log', names = names1, delim_whitespace = False)#read the final_wrench.log file
dataset1.to_csv('/home/ipa325/method_2/new_status_log.log',index=True, sep='\t', mode='w') #and write to final_wrench_mod. Dataframe created for ease of data manip
dataset1['time_stamp'] = dataset1['eventtime'].map(str)+'.'+ dataset1['msg_stamp'].map(str)
dataset1['time_stamp']
del dataset1['eventtime']
del dataset1['msg_stamp']
del dataset1['skill-name']
del dataset1['symbols']
dataset1['time_stamp','skill id']
print dataset1
