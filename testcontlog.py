import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
from sklearn import model_selection
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import csv


with open('/home/ipa325/_20170420/control_op/cont_log_l.log') as in_file:
    with open('/home/ipa325/_20170420/control_op/cont_log_two.log','w') as o_file:
        for i in range (10):
           line = in_file.next()
           o_file.write(line.replace(', [_contact1_fix, 140263760625936/force_chain/z, contact_h_vac_bck, 140263760625936/force_chain/y, 140263760625936/force_chain/b, 140263760625936/force_chain/c, 140263760625936/force_chain/a, marker, joint_4, joint_5, joint_2, joint_3, joint_0, joint_1, copy_semiclosed_lid_frame, contact_h_vac, copy_fullyclosed_lid_frame, _hold_temp_frame, copy_open_lid_frame, startpos, contact_h_vac_deviation4, contact_h_vac_deviation2, contact_h_vac_deviation3, contact_h_vac_deviation1, 140263760625936/force_chain/x]', '').replace('[','').replace(']','').replace(', ',','))
        names = ['eventtime', 'rostime.now', 'symbols','_contact1_fix', 'force_chain/z', 'contact_h_vac_bck', 'force_chain/y', 'force_chain/b', 'force_chain/c', 'force_chain/a', 'marker', 'joint_4', 'joint_5', 'joint_2', 'joint_3', 'joint_0', 'joint_1', 'copy_semiclosed_lid_frame', 'contact_h_vac', 'copy_fullyclosed_lid_frame', '_hold_temp_frame', 'copy_open_lid_frame', 'startpos', 'contact_h_vac_deviation4', 'contact_h_vac_deviation2', 'contact_h_vac_deviation3', 'contact_h_vac_deviation1', 'force_chain/x']
        dataset = pandas.read_csv('/home/ipa325/_20170420/control_op/cont_log_m.log', names = names, delim_whitespace = False)
        #print dataset.head(6)
dataset.to_csv('/home/ipa325/_20170420/control_op/control_log_formatted_test.log',index=True, sep='\t', mode='w')


