import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import csv
names =['f_x','f_y','f_z','t_x','t_y','t_z','jq01','jq02','jq03','jq04','jq05','jq06','jq11','jq12','jq13','jq14','jq15','jq16','jq21','jq22','jq23','jq24','jq25','jq26','jq31','jq32','jq33','jq34','jq35','jq36','jq41','jq42','jq43','jq44','jq45','jq46','jq51','jq52','jq53','jq54','jq55','jq56','x1','x2','x3','x4','x5','x6','time_x','eigen_force','skill']
datfr = pandas.read_csv('/home/ipa325/wrench_jacobian_prod.log', names = names, delim_whitespace = False)
datfr.to_csv('/home/ipa325/wrench_jacobian_prodI_test.log',index=True, sep='\t', mode='w')
datfr.apply(pandas.to_numeric)
plt.plot(datfr['time_x'],datfr['eigen_force'], 'r')
plt.show()
