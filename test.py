import pandas as pd
import sys
import scipy
import numpy
import matplotlib.pyplot as plt

with open('/home/ipa325/eigenforce3d.log') as infile:
   with open('/home/ipa325/eigmodified_.log','w') as ofile:
      for i in range(8000):
             line = infile.next()
             ofile.write(line.replace('[dx, dy, dz, da, db, dc, joint_0, joint_1, joint_2, joint_3, joint_4, joint_5], ', '').replace('[','').replace(']','').replace(', ',','))
      names = ['eventtime', 'rostime.now', 'symbols', 'eigforce3d']
      dataset = pd.read_csv('/home/ipa325/eigmodified.log', names = names) 
dataset.to_csv('/home/ipa325/neweig.log', header=True, index=False, sep='\t', mode='w')
df1 = pd.DataFrame(dataset, columns =['rostime.now','eigforce3d'])
df1[['eigforce3d','rostime.now']] = df1[['eigforce3d','rostime.now']].apply(pd.to_numeric, errors = 'ignore')
df1.iloc[100:1000].plot(y='eigforce3d', style='b')
plt.ylabel('Eigen Force 3d')
plt.xlabel('ROS time')
plt.title('Eigen Force Plot')
plt.show()
