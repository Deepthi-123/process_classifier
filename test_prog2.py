import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import csv
names = ['eventtime', 'msg_stamp', 'symbols','force_x', 'force_y', 'force_z', 'torque_x', 'torque_y', 'torque_z']
dataset = pandas.read_csv('/home/deepthi/topic_log/20170420/20170420-10_42_08_wrench.log', names = names, delim_whitespace = False)#read the final_wrench.log file
dataset.to_csv('/home/deepthi/wrench_test_check22.log',index=True, sep='\t', mode='w') #and write to final_wrench_mod. Dataframe created for ease of data manip
dataset['time_stamp'] = dataset['eventtime'].map(str)+'.'+ dataset['msg_stamp'].map(str)
dataset['time_stamp']
del dataset['eventtime']
del dataset['msg_stamp']
cols = dataset.columns.tolist()
cols = cols[-1:] + cols[:-1]
dataset = dataset[cols]
dataset['time_stamp'] = pandas.to_datetime(dataset['time_stamp'], errors = 'coerce')
dataset_ = dataset.set_index('time_stamp')
dataset_ = dataset_.apply(pandas.to_numeric, errors = 'coerce')
names_n = ['eventtime', 'msg_stamp', 'symbols','status','skill']
datan = pandas.read_csv('/home/deepthi/topic_log/20170420/20170420-10_42_08_status.log', names = names_n, delim_whitespace = False)#read the final_wrench.log file
datan.to_csv('/home/deepthi/status_test_check_1.log',index=True, sep='\t', mode='w') #and write to final_wrench_mod. Dataframe created for ease of data manip
datan['time_stamp'] = datan['eventtime'].map(str)+'.'+ datan['msg_stamp'].map(str)
datan['time_stamp']
del datan['eventtime']
del datan['msg_stamp']
del datan['symbols']
#print datan
cols = datan.columns.tolist()
cols = cols[-1:] + cols[:-1]
datan = datan[cols]
datan['time_stamp'] = pandas.to_datetime(datan['time_stamp'], errors = 'coerce')
datan_ = datan.set_index('time_stamp')
del datan_['status']
names_ = ['eventtime', 'msg_stamp', 'symbols', 'jq01','jq02','jq03','jq04','jq05','jq06','jq11','jq12','jq13','jq14','jq15','jq16','jq21','jq22','jq23','jq24','jq25','jq26','jq31','jq32','jq33','jq34','jq35','jq36','jq41','jq42','jq43','jq44','jq45','jq46','jq51','jq52','jq53','jq54','jq55','jq56']
datset = pandas.read_csv('/home/deepthi/topic_log/20170420/20170420-10_42_08_Jq.log', names = names_, delim_whitespace = False) #read and write jacobian file
datset.to_csv('/home/deepthi/jacob_mod_true22.log',index=True, sep='\t', mode='w')
datset['time_stamp'] = datset['eventtime'].map(str)+'.'+ datset['msg_stamp'].map(str)
datset['time_stamp']
del datset['eventtime']
del datset['msg_stamp']
del datset['symbols']
cols_ = datset.columns.tolist()
cols_ = cols_[-1:] + cols_[:-1]
datset = datset[cols_]
datset['time_stamp'] = pandas.to_datetime(datset['time_stamp'], errors = 'coerce')
datset_ = datset.set_index('time_stamp')
datset_ = datset_[1:].astype(float)
#print dataset_
skill = datan_.skill.resample('1L')
skill = pandas.Series(skill)
f_x = dataset_.force_x.resample('1L')
f_x = pandas.Series(f_x)
f_y = dataset_.force_y.resample('1L')
f_y = pandas.Series(f_y)
f_z = dataset_.force_z.resample('1L')
f_z = pandas.Series(f_z)
t_x = dataset_.torque_x.resample('1L')
t_x = pandas.Series(t_x)
t_y = dataset_.torque_y.resample('1L')
t_y = pandas.Series(t_y)
t_z = dataset_.torque_z.resample('1L')
t_z = pandas.Series(t_z)
jq01 = datset_.jq01.resample('1L')
jq01 = pandas.Series(jq01)
jq02 = datset_.jq02.resample('1L')
jq02 = pandas.Series(jq02)
jq03 = datset_.jq03.resample('1L')
jq03 = pandas.Series(jq03)
jq04 = datset_.jq04.resample('1L')
jq04 = pandas.Series(jq04)
jq05 = datset_.jq05.resample('1L')
jq05 = pandas.Series(jq05)
jq06 = datset_.jq06.resample('1L')
jq06 = pandas.Series(jq06)
jq11 = datset_.jq11.resample('1L')
jq11 = pandas.Series(jq11)
jq12 = datset_.jq12.resample('1L')
jq12 = pandas.Series(jq12)
jq13 = datset_.jq13.resample('1L')
jq13 = pandas.Series(jq13)
jq14 = datset_.jq14.resample('1L')
jq14 = pandas.Series(jq14)
jq15 = datset_.jq15.resample('1L')
jq15 = pandas.Series(jq15)
jq16 = datset_.jq16.resample('1L')
jq16 = pandas.Series(jq16)
jq21 = datset_.jq21.resample('1L')
jq21 = pandas.Series(jq21)
jq22 = datset_.jq22.resample('1L')
jq22 = pandas.Series(jq22)
jq23 = datset_.jq23.resample('1L')
jq23 = pandas.Series(jq23)
jq24 = datset_.jq24.resample('1L')
jq24 = pandas.Series(jq24)
jq25 = datset_.jq25.resample('1L')
jq25 = pandas.Series(jq25)
jq26 = datset_.jq26.resample('1L')
jq26 = pandas.Series(jq26)
jq31 = datset_.jq31.resample('1L')
jq31 = pandas.Series(jq31)
jq32 = datset_.jq32.resample('1L')
jq32 = pandas.Series(jq32)
jq33 = datset_.jq33.resample('1L')
jq33 = pandas.Series(jq33)
jq34 = datset_.jq34.resample('1L')
jq34 = pandas.Series(jq34)
jq35 = datset_.jq35.resample('1L')
jq35 = pandas.Series(jq35)
jq36 = datset_.jq36.resample('1L')
jq36 = pandas.Series(jq36)
jq41 = datset_.jq41.resample('1L')
jq41 = pandas.Series(jq41)
jq42 = datset_.jq42.resample('1L')
jq42 = pandas.Series(jq42)
jq43 = datset_.jq43.resample('1L')
jq43 = pandas.Series(jq43)
jq44 = datset_.jq44.resample('1L')
jq44 = pandas.Series(jq44)
jq45 = datset_.jq45.resample('1L')
jq45 = pandas.Series(jq45)
jq46 = datset_.jq46.resample('1L')
jq46 = pandas.Series(jq46)
jq51 = datset_.jq51.resample('1L')
jq51 = pandas.Series(jq51)
jq52 = datset_.jq52.resample('1L')
jq52 = pandas.Series(jq52)
jq53 = datset_.jq53.resample('1L')
jq53 = pandas.Series(jq53)
jq54 = datset_.jq54.resample('1L')
jq54 = pandas.Series(jq54)
jq55 = datset_.jq55.resample('1L')
jq55 = pandas.Series(jq55)
jq56 = datset_.jq56.resample('1L')
jq56 = pandas.Series(jq56)
dict_ = {'f_x': f_x,
         'f_y': f_y,
         'f_z': f_z,
         't_x': t_x,
         't_y': t_y,
         't_z': t_z,
         'jq01': jq01,
         'jq02': jq02,
         'jq03': jq03,
         'jq04': jq04,
         'jq05': jq05,
         'jq06': jq06,
         'jq11': jq11,
         'jq12': jq12,
         'jq13': jq13,
         'jq14': jq14,
         'jq15': jq15,
         'jq16': jq16,
         'jq21': jq21,
         'jq22': jq22,
         'jq23': jq23,
         'jq24': jq24,
         'jq25': jq25,
         'jq26': jq26,
         'jq31': jq31,
         'jq32': jq32,
         'jq33': jq33,
         'jq34': jq34,
         'jq35': jq35,
         'jq36': jq36,
         'jq41': jq41,
         'jq42': jq42,
         'jq43': jq43,
         'jq44': jq44,
         'jq45': jq45,
         'jq46': jq46,
         'jq51': jq51,
         'jq52': jq52,
         'jq53': jq53,
         'jq54': jq54,
         'jq55': jq55,
         'jq56': jq56}
data = pandas.concat(dict_.values(), axis =1, keys = dict_.keys())
data = data[['f_x','f_y','f_z','t_x','t_y','t_z','jq01','jq02','jq03','jq04','jq05','jq06','jq11','jq12','jq13','jq14','jq15','jq16','jq21','jq22','jq23','jq24','jq25','jq26','jq31','jq32','jq33','jq34','jq35','jq36','jq41','jq42','jq43','jq44','jq45','jq46','jq51','jq52','jq53','jq54','jq55','jq56']]
data['f_x'].fillna(method = 'ffill', inplace = True)
data['f_y'].fillna(method = 'ffill', inplace = True)
data['f_z'].fillna(method = 'ffill', inplace = True)
data['t_x'].fillna(method = 'ffill', inplace = True)
data['t_y'].fillna(method = 'ffill', inplace = True)
data['t_z'].fillna(method = 'ffill', inplace = True)
data['jq01'].fillna(method = 'ffill', inplace = True)
data['jq02'].fillna(method = 'ffill', inplace = True)
data['jq03'].fillna(method = 'ffill', inplace = True)
data['jq04'].fillna(method = 'ffill', inplace = True)
data['jq05'].fillna(method = 'ffill', inplace = True)
data['jq06'].fillna(method = 'ffill', inplace = True)
data['jq11'].fillna(method = 'ffill', inplace = True)
data['jq12'].fillna(method = 'ffill', inplace = True)
data['jq13'].fillna(method = 'ffill', inplace = True)
data['jq14'].fillna(method = 'ffill', inplace = True)
data['jq15'].fillna(method = 'ffill', inplace = True)
data['jq16'].fillna(method = 'ffill', inplace = True)
data['jq21'].fillna(method = 'ffill', inplace = True)
data['jq22'].fillna(method = 'ffill', inplace = True)
data['jq23'].fillna(method = 'ffill', inplace = True)
data['jq24'].fillna(method = 'ffill', inplace = True)
data['jq25'].fillna(method = 'ffill', inplace = True)
data['jq26'].fillna(method = 'ffill', inplace = True)
data['jq31'].fillna(method = 'ffill', inplace = True)
data['jq32'].fillna(method = 'ffill', inplace = True)
data['jq33'].fillna(method = 'ffill', inplace = True)
data['jq34'].fillna(method = 'ffill', inplace = True)
data['jq35'].fillna(method = 'ffill', inplace = True)
data['jq36'].fillna(method = 'ffill', inplace = True)
data['jq41'].fillna(method = 'ffill', inplace = True)
data['jq42'].fillna(method = 'ffill', inplace = True)
data['jq43'].fillna(method = 'ffill', inplace = True)
data['jq44'].fillna(method = 'ffill', inplace = True)
data['jq45'].fillna(method = 'ffill', inplace = True)
data['jq46'].fillna(method = 'ffill', inplace = True)
data['jq51'].fillna(method = 'ffill', inplace = True)
data['jq52'].fillna(method = 'ffill', inplace = True)
data['jq53'].fillna(method = 'ffill', inplace = True)
data['jq54'].fillna(method = 'ffill', inplace = True)
data['jq55'].fillna(method = 'ffill', inplace = True)
data['jq56'].fillna(method = 'ffill', inplace = True)
dq = np.array(data.loc[:,'jq01':'jq56'])
for p in dq[:]:
   B = [float(i) for i in p] 
   Jq = np.reshape(B,(6,6))
pt = np.array(data.loc[:,'f_x':'t_z'])#create 6x1 matrices of forces and torques
prod = []
for t in pt[:]:
   C =[float(i) for i in t]
   ar = np.reshape(C,(6,1))
   prod += ((np.dot(Jq,ar)).T).tolist()#get the product of forces and jacobians
df_ = pandas.DataFrame(prod, columns =['x1','x2','x3','x4','x5','x6'], index = data.index)
x1 = pandas.Series(df_['x1'])
x2 = pandas.Series(df_['x2'])
x3 = pandas.Series(df_['x3'])
x4 = pandas.Series(df_['x4'])
x5 = pandas.Series(df_['x5'])
x6 = pandas.Series(df_['x6'])
x = data.index.map(str)
xp = []
xp += ((float(i[14:16]))*60 + float(i[17:]) for i in x)
xpn = pandas.Series(xp, index = data.index)
names1 = ['eventtime', 'rostime.now', 'symbols', 'eigforce3d']
dataset1 = pandas.read_csv('/home/deepthi/topic_log/20170420/20170420-10_42_08_eigenforced.log', names = names1) 
dataset1.to_csv('/home/deepthi/final_eig3d_this22.log', header=False, index=False, sep='\t', mode='w')
dataset1['time_stamp'] = dataset1['eventtime'].map(str)+'.'+dataset1['rostime.now']
del dataset1['symbols']
del dataset1['eventtime']
del dataset1['rostime.now']
#print dataset1
dataset1['time_stamp'] = pandas.to_datetime(dataset1['time_stamp'], errors= 'coerce')
dataset2 = dataset1.set_index('time_stamp')
eig_force = dataset2.eigforce3d.resample('1L')
eig_force = pandas.Series(eig_force)
dl_list = {'f_x': f_x,
         'f_y': f_y,
         'f_z': f_z,
         't_x': t_x,
         't_y': t_y,
         't_z': t_z,
         'jq01': jq01,
         'jq02': jq02,
         'jq03': jq03,
         'jq04': jq04,
         'jq05': jq05,
         'jq06': jq06,
         'jq11': jq11,
         'jq12': jq12,
         'jq13': jq13,
         'jq14': jq14,
         'jq15': jq15,
         'jq16': jq16,
         'jq21': jq21,
         'jq22': jq22,
         'jq23': jq23,
         'jq24': jq24,
         'jq25': jq25,
         'jq26': jq26,
         'jq31': jq31,
         'jq32': jq32,
         'jq33': jq33,
         'jq34': jq34,
         'jq35': jq35,
         'jq36': jq36,
         'jq41': jq41,
         'jq42': jq42,
         'jq43': jq43,
         'jq44': jq44,
         'jq45': jq45,
         'jq46': jq46,
         'jq51': jq51,
         'jq52': jq52,
         'jq53': jq53,
         'jq54': jq54,
         'jq55': jq55,
         'jq56': jq56,
         'x1'  : x1,
         'x2'  : x2,
         'x3'  : x3,
         'x4'  : x4,
         'x5'  : x5,
         'x6'  : x6,
         'time_x' : xpn,
         'eigen_force' : eig_force,
         'skill': skill}
#print eig_force
data_final = pandas.concat(dl_list.values(),axis=1,keys=dl_list.keys())
data_final = data_final[['f_x','f_y','f_z','t_x','t_y','t_z','jq01','jq02','jq03','jq04','jq05','jq06','jq11','jq12','jq13','jq14','jq15','jq16','jq21','jq22','jq23','jq24','jq25','jq26','jq31','jq32','jq33','jq34','jq35','jq36','jq41','jq42','jq43','jq44','jq45','jq46','jq51','jq52','jq53','jq54','jq55','jq56','x1','x2','x3','x4','x5','x6','time_x','eigen_force','skill']]
data_final['f_x'].fillna(method = 'ffill', inplace = True)
data_final['f_y'].fillna(method = 'ffill', inplace = True)
data_final['f_z'].fillna(method = 'ffill', inplace = True)
data_final['t_x'].fillna(method = 'ffill', inplace = True)
data_final['t_y'].fillna(method = 'ffill', inplace = True)
data_final['t_z'].fillna(method = 'ffill', inplace = True)
data_final['jq01'].fillna(method = 'ffill', inplace = True)
data_final['jq02'].fillna(method = 'ffill', inplace = True)
data_final['jq03'].fillna(method = 'ffill', inplace = True)
data_final['jq04'].fillna(method = 'ffill', inplace = True)
data_final['jq05'].fillna(method = 'ffill', inplace = True)
data_final['jq06'].fillna(method = 'ffill', inplace = True)
data_final['jq11'].fillna(method = 'ffill', inplace = True)
data_final['jq12'].fillna(method = 'ffill', inplace = True)
data_final['jq13'].fillna(method = 'ffill', inplace = True)
data_final['jq14'].fillna(method = 'ffill', inplace = True)
data_final['jq15'].fillna(method = 'ffill', inplace = True)
data_final['jq16'].fillna(method = 'ffill', inplace = True)
data_final['jq21'].fillna(method = 'ffill', inplace = True)
data_final['jq22'].fillna(method = 'ffill', inplace = True)
data_final['jq23'].fillna(method = 'ffill', inplace = True)
data_final['jq24'].fillna(method = 'ffill', inplace = True)
data_final['jq25'].fillna(method = 'ffill', inplace = True)
data_final['jq26'].fillna(method = 'ffill', inplace = True)
data_final['jq31'].fillna(method = 'ffill', inplace = True)
data_final['jq32'].fillna(method = 'ffill', inplace = True)
data_final['jq33'].fillna(method = 'ffill', inplace = True)
data_final['jq34'].fillna(method = 'ffill', inplace = True)
data_final['jq35'].fillna(method = 'ffill', inplace = True)
data_final['jq36'].fillna(method = 'ffill', inplace = True)
data_final['jq41'].fillna(method = 'ffill', inplace = True)
data_final['jq42'].fillna(method = 'ffill', inplace = True)
data_final['jq43'].fillna(method = 'ffill', inplace = True)
data_final['jq44'].fillna(method = 'ffill', inplace = True)
data_final['jq45'].fillna(method = 'ffill', inplace = True)
data_final['jq46'].fillna(method = 'ffill', inplace = True)
data_final['jq51'].fillna(method = 'ffill', inplace = True)
data_final['jq52'].fillna(method = 'ffill', inplace = True)
data_final['jq53'].fillna(method = 'ffill', inplace = True)
data_final['jq54'].fillna(method = 'ffill', inplace = True)
data_final['jq55'].fillna(method = 'ffill', inplace = True)
data_final['jq56'].fillna(method = 'ffill', inplace = True)
data_final['x1'].fillna(method = 'ffill', inplace = True)
data_final['x2'].fillna(method = 'ffill', inplace = True)
data_final['x3'].fillna(method = 'ffill', inplace = True)
data_final['x4'].fillna(method = 'ffill', inplace = True)
data_final['x5'].fillna(method = 'ffill', inplace = True)
data_final['x6'].fillna(method = 'ffill', inplace = True)
data_final['skill'].fillna(method = 'ffill', inplace = True)
data_final['skill'].fillna(0, inplace = True)
data_final['eigen_force'].fillna(method = 'ffill', inplace = True)
data_final['eigen_force'].fillna(0, inplace = True)
data_final.to_csv('/home/deepthi/wrench_jacobian_prod22.log',index=True, sep='\t', mode='w')
dt_ = {'skill': skill, 'eig_val': eig_force}
dats = pandas.concat(dt_.values(), axis = 1, keys = dt_.keys())
dats['skill'].fillna(method = 'ffill', inplace = True)
dats['skill'].fillna(0, inplace = True)
dats['eig_val'].fillna(method = 'ffill', inplace = True)
dats = dats.reset_index()
dats.to_csv('/home/deepthi/skill_eig_ch.log', index= False, sep = '\t', mode ='w')
axes = plt.gca()
axes.set_xlim([2600, 2950])
fig,ax1 = plt.subplots()
#ax1.plot(data_final['time_x'],data_final['f_x'], 'r', label = 'force x')
#ax1.plot(data_final['time_x'],data_final['f_y'], 'b', label = 'force y')
#ax1.plot(data_final['time_x'],data_final['f_z'], 'g', label = 'force z')
ax2 = ax1.twinx()
ax2.set_xlim([2750,2950])
ax2.plot(data_final['time_x'],data_final['skill'], 'b', label = 'skill')
#plt.plot(data_final['time_x'],data_final['x5'], 'm', label = 'tau5q')
#plt.plot(data_final['time_x'],data_final['x6'], 'y', label = 'tau6q')
ax1.plot(data_final['time_x'],data_final['eigen_force'], 'k', label = 'eigen force')
plt.legend(loc = 'best')
plt.xlabel('Time')
plt.ylabel('Eigen-Force and Forces')
plt.title('Eigen-Force and Forces vs time')
#plt.show()

                    
