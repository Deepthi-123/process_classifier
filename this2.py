import pandas
from pandas.tools.plotting import scatter_matrix
import matplotlib.pyplot as plt
import numpy as np
import csv
names1 = ['index','skill', 'time_check', 'eigen_force']
dataset1 = pandas.read_csv('/home/ipa325/check_eig_skill.log', names = names1, delim_whitespace = False) 
print dataset1.reset_index(drop = 'True')

