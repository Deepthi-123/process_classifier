import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import csv
names = ['eventtime', 'msg_stamp', 'symbols','force_x', 'force.y', 'force.z', 'torque.x', 'torque.y', 'torque.z']
dataset = pandas.read_csv('/home/ipa325/final_wrench.log', names = names, delim_whitespace = False)#read the final_wrench.log file
dataset.to_csv('/home/ipa325/final_wrench_mod.log',index=True, sep='\t', mode='w') #and write to final_wrench_mod. Dataframe created for ease of data manip
names_ = ['eventtime', 'msg_stamp', 'symbols', '01','02','03','04','05','06','11','12','13','14','15','16','21','22','23','24','25','26','31','32','33','34','35','36','41','42','43','44','45','46','51','52','53','54','55','56']
datset = pandas.read_csv('/home/ipa325/topic_log/_20170420/20170420-10_08_01_Jq.log', names = names_, delim_whitespace = False) #read and write jacobian file
datset.to_csv('/home/ipa325/jacob_mod.log',index=True, sep='\t', mode='w')
name = ['','','']
dataset.reset_index().set_index(pandas.to_datetime(dataset['eventtime'], errors = 'coerce', infer_datetime_format = True))
df_a = pandas.DataFrame(data = dataset)
datset.reset_index().set_index(pandas.to_datetime(datset['eventtime'], errors = 'coerce', infer_datetime_format = True))
df_b = pandas.DataFrame(data = datset)
dq = np.array(datset.loc[:,'01':'56']) #create  6x6 jacobian matrices
dataList = {'a':df_a,
            'b':df_b}
for p in dq[1:]:
   B = [float(i) for i in p] 
   Jq = np.reshape(B,(6,6))
pt = np.array(dataset.loc[:,'force_x':'torque.z'])#create 6x1 matrices of forces and torques
prod_s = pandas.DataFrame()
prod = []
for t in pt[1:]:
   C =[float(i) for i in t]
   ar = np.reshape(C,(6,1))
   prod += ((np.dot(Jq,ar)).T).tolist()#get the product of forces and jacobians
df_ = pandas.DataFrame(prod, columns =['x1','x2','x3','x4','x5','x6'])
dl = {'p':df_a,
      'q':df_b,
      'r':df_}
dh = pandas.concat(dl.values(), axis = 1, keys=dl.keys()) 
dict_ = {'rostime':dh.iloc[:,40], 'x1':dh.iloc[:,48]}
dhp = pandas.concat(dict_.values(), axis = 1, keys=dict_.keys()) 
dhp.iloc[:].plot(y= 'x1', style = 'r')
plt.axis([0,1500,-0.2,0.8])
plt.show()
#print dhp['x1']
#print dhp2
