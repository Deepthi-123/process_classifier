import numpy as np
import pandas
from io import StringIO
import matplotlib.pyplot as plt

def list2String(f_x):
    f_x_s = "["
    i = 0;
    for elem in f_x:
        if ( i == 0) :
            f_x_s = f_x_s + repr(elem)
            i = i +1;
        else:
            f_x_s = f_x_s + ", " + repr(elem)
    f_x_s = f_x_s + "]"
    return f_x_s

def extract_file(filename):
   names = ['date', 'time', 'f_x', 'f_y', 'f_z', 't_x', 't_y', 't_z']
   dataset = pandas.read_csv(filename, sep = ',', names = names)
   return dataset

def convert_dataset(dataset, time_stamp):
   f_x = dataset['f_x'].tolist()
   f_y = dataset['f_y'].tolist()
   f_z = dataset['f_z'].tolist()
   f_x_s = list2String(f_x) 
   f_y_s = list2String(f_y)
   f_z_s = list2String(f_z)
   if (str(time_stamp) != "[]"):
      labels = ['l1', 'l2']
   elif (str(time_stamp) == "[]"):
      labels = "[]"
   dataset = f_x_s + ";" + f_y_s + ";" + f_z_s + ";" + str(labels) + ";" + str(time_stamp)
   return dataset

   
dataset_0 = extract_file("/home/deepthi/to_dataset/dataset_0.log")
dataset_1 = extract_file("/home/deepthi/to_dataset/dataset_1.log")
dataset_2 = extract_file("/home/deepthi/to_dataset/dataset_2.log")
dataset_3 = extract_file("/home/deepthi/to_dataset/dataset_3.log")
dataset_4 = extract_file("/home/deepthi/to_dataset/dataset_4.log")
dataset_5 = extract_file("/home/deepthi/to_dataset/dataset_5.log")
dataset_6 = extract_file("/home/deepthi/to_dataset/dataset_6.log")
dataset_7 = extract_file("/home/deepthi/to_dataset/dataset_7.log")
dataset_8 = extract_file("/home/deepthi/to_dataset/dataset_8.log")
dataset_9 = extract_file("/home/deepthi/to_dataset/dataset_9.log")
dataset_10 = extract_file("/home/deepthi/to_dataset/dataset_10.log")
dataset_11 = extract_file("/home/deepthi/to_dataset/dataset_11.log")
dataset_12 = extract_file("/home/deepthi/to_dataset/dataset_12.log")
dataset_13 = extract_file("/home/deepthi/to_dataset/dataset_13.log")
dataset_14 = extract_file("/home/deepthi/to_dataset/dataset_14.log")
dataset_15 = extract_file("/home/deepthi/to_dataset/dataset_15.log")
dataset_16 = extract_file("/home/deepthi/to_dataset/dataset_16.log")
dataset_17 = extract_file("/home/deepthi/to_dataset/dataset_17.log")
dataset_18 = extract_file("/home/deepthi/to_dataset/dataset_18.log")
dataset_19 = extract_file("/home/deepthi/to_dataset/dataset_19.log")
dataset_20 = extract_file("/home/deepthi/to_dataset/dataset_20.log")
dataset_21 = extract_file("/home/deepthi/to_dataset/dataset_21.log")
dataset_22 = extract_file("/home/deepthi/to_dataset/dataset_22.log")
dataset_23 = extract_file("/home/deepthi/to_dataset/dataset_23.log")
dataset_24 = extract_file("/home/deepthi/to_dataset/dataset_24.log")
dataset_25 = extract_file("/home/deepthi/to_dataset/dataset_25.log")
dataset_26 = extract_file("/home/deepthi/to_dataset/dataset_26.log")
dataset_27 = extract_file("/home/deepthi/to_dataset/dataset_27.log")
dataset_28 = extract_file("/home/deepthi/to_dataset/dataset_28.log")
dataset_29 = extract_file("/home/deepthi/to_dataset/dataset_29.log")
dataset_30 = extract_file("/home/deepthi/to_dataset/dataset_30.log")
dataset_31 = extract_file("/home/deepthi/to_dataset/dataset_31.log")
dataset_32 = extract_file("/home/deepthi/to_dataset/dataset_32.log")
dataset_33 = extract_file("/home/deepthi/to_dataset/dataset_33.log")
dataset_34 = extract_file("/home/deepthi/to_dataset/dataset_34.log")
dataset_35 = extract_file("/home/deepthi/to_dataset/dataset_35.log")
dataset_36 = extract_file("/home/deepthi/to_dataset/dataset_36.log")
dataset_37 = extract_file("/home/deepthi/to_dataset/dataset_37.log")
dataset_38 = extract_file("/home/deepthi/to_dataset/dataset_38.log")
dataset_39 = extract_file("/home/deepthi/to_dataset/dataset_39.log")
dataset_nt = extract_file("/home/deepthi/to_dataset/dataset_no_tar.log")
dataset_nt1 = extract_file("/home/deepthi/to_dataset/dataset_nt1.log")
dataset_nt2 = extract_file("/home/deepthi/to_dataset/dataset_nt2.log") 
dataset_nt3 = extract_file("/home/deepthi/to_dataset/dataset_nt3.log") 
dataset_nt4 = extract_file("/home/deepthi/to_dataset/dataset_nt4.log")
dataset_nt5 = extract_file("/home/deepthi/to_dataset/dataset_nt5.log") 
dataset_nt6 = extract_file("/home/deepthi/to_dataset/dataset_nt6.log")  
 
data_list = []
time_stamp_0 = [[65],[82]]
time_stamp_1 = [[67],[87]]
time_stamp_2 = [[63],[83]]
time_stamp_3 = [[66],[82]]
time_stamp_4 = [[67],[81]]
time_stamp_5 = [[66],[83]]
time_stamp_6 = [[64],[80]]
time_stamp_7 = [[68],[86]]
time_stamp_8 = [[66],[80]]
time_stamp_9 = [[68],[88]]
time_stamp_10 = [[62],[80]]
time_stamp_11 = [[71],[85]]
time_stamp_12 = [[69],[86]]
time_stamp_13 = [[60],[79]]
time_stamp_14 = [[64],[82]]
time_stamp_15 = [[64],[81]]
time_stamp_16 = [[67],[83]]
time_stamp_17 = [[66],[84]]
time_stamp_18 = [[69],[84]]
time_stamp_19 = [[64],[79]]
time_stamp_20 = [[65],[82]]
time_stamp_21 = [[64],[79]]
time_stamp_22 = [[61],[78]]
time_stamp_23 = [[59],[76]]
time_stamp_24 = [[67],[83]]
time_stamp_25 = [[68],[85]]
time_stamp_26 = [[67],[86]]
time_stamp_27 = [[65],[82]]
time_stamp_28 = [[63],[78]]
time_stamp_29 = [[58],[72]]
time_stamp_30 = [[67],[85]]
time_stamp_31 = [[65],[84]]
time_stamp_32 = [[59],[78]]
time_stamp_33 = [[63],[79]]
time_stamp_34 = [[65],[74]]
time_stamp_35 = [[56],[71]]
time_stamp_36 = [[63],[84]]
time_stamp_37 = [[68],[84]]
time_stamp_38 = [[69],[86]]
time_stamp_39 = [[70],[87]]
time_stamp_nt = []

to_dat_0 = convert_dataset(dataset_0, time_stamp_0)
data_list.append(to_dat_0)
to_dat_1 = convert_dataset(dataset_1, time_stamp_1)
data_list.append(to_dat_1)
to_dat_2 = convert_dataset(dataset_2, time_stamp_2)
data_list.append(to_dat_2)
to_dat_3 = convert_dataset(dataset_3, time_stamp_3)
data_list.append(to_dat_3)
to_dat_4 = convert_dataset(dataset_4, time_stamp_4)
data_list.append(to_dat_4)
to_dat_5 = convert_dataset(dataset_5, time_stamp_5)
data_list.append(to_dat_5)
to_dat_6 = convert_dataset(dataset_6, time_stamp_6)
data_list.append(to_dat_6)
to_dat_7 = convert_dataset(dataset_7, time_stamp_7)
data_list.append(to_dat_7)
to_dat_8 = convert_dataset(dataset_8, time_stamp_8)
data_list.append(to_dat_8)
to_dat_9 = convert_dataset(dataset_9, time_stamp_9)
data_list.append(to_dat_9)
to_dat_10 = convert_dataset(dataset_10, time_stamp_10)
data_list.append(to_dat_10)
to_dat_11 = convert_dataset(dataset_11, time_stamp_11)
data_list.append(to_dat_11)
to_dat_12 = convert_dataset(dataset_12, time_stamp_12)
data_list.append(to_dat_12)
to_dat_13 = convert_dataset(dataset_13, time_stamp_13)
data_list.append(to_dat_13)
to_dat_14 = convert_dataset(dataset_14, time_stamp_14)
data_list.append(to_dat_14)
to_dat_15 = convert_dataset(dataset_15, time_stamp_15)
data_list.append(to_dat_15)
to_dat_16 = convert_dataset(dataset_16, time_stamp_16)
data_list.append(to_dat_16)
to_dat_17 = convert_dataset(dataset_17, time_stamp_17)
data_list.append(to_dat_17)
to_dat_18 = convert_dataset(dataset_18, time_stamp_18)
data_list.append(to_dat_18)
to_dat_19 = convert_dataset(dataset_19, time_stamp_19)
data_list.append(to_dat_19)
to_dat_20 = convert_dataset(dataset_20, time_stamp_20)
data_list.append(to_dat_20)
to_dat_21 = convert_dataset(dataset_21, time_stamp_21)
data_list.append(to_dat_21)
to_dat_22 = convert_dataset(dataset_22, time_stamp_22)
data_list.append(to_dat_22)
to_dat_23 = convert_dataset(dataset_23, time_stamp_23)
data_list.append(to_dat_23)
to_dat_24 = convert_dataset(dataset_24, time_stamp_24)
data_list.append(to_dat_24)
to_dat_25 = convert_dataset(dataset_25, time_stamp_25)
data_list.append(to_dat_25)
to_dat_26 = convert_dataset(dataset_26, time_stamp_26)
data_list.append(to_dat_26)
to_dat_27 = convert_dataset(dataset_27, time_stamp_27)
data_list.append(to_dat_27)
to_dat_28 = convert_dataset(dataset_28, time_stamp_28)
data_list.append(to_dat_28)
to_dat_29 = convert_dataset(dataset_29, time_stamp_29)
data_list.append(to_dat_29)
to_dat_30 = convert_dataset(dataset_30, time_stamp_30)
data_list.append(to_dat_30)
to_dat_31 = convert_dataset(dataset_31, time_stamp_31)
data_list.append(to_dat_31)
to_dat_32 = convert_dataset(dataset_32, time_stamp_32)
data_list.append(to_dat_32)
to_dat_33 = convert_dataset(dataset_33, time_stamp_33)
data_list.append(to_dat_33)
to_dat_34 = convert_dataset(dataset_34, time_stamp_34)
data_list.append(to_dat_34)
to_dat_35 = convert_dataset(dataset_35, time_stamp_35)
data_list.append(to_dat_35)
to_dat_36 = convert_dataset(dataset_36, time_stamp_36)
data_list.append(to_dat_36)
to_dat_37 = convert_dataset(dataset_37, time_stamp_37)
data_list.append(to_dat_37)
to_dat_38 = convert_dataset(dataset_38, time_stamp_38)
data_list.append(to_dat_38)
to_dat_39 = convert_dataset(dataset_39, time_stamp_39)
data_list.append(to_dat_39)
to_dat_nt = convert_dataset(dataset_nt, time_stamp_nt)
data_list.append(to_dat_nt)
to_dat_nt1 = convert_dataset(dataset_nt1, time_stamp_nt)
data_list.append(to_dat_nt1)
to_dat_nt2 = convert_dataset(dataset_nt2, time_stamp_nt)
data_list.append(to_dat_nt2)
to_dat_nt3 = convert_dataset(dataset_nt3, time_stamp_nt)
data_list.append(to_dat_nt3)
to_dat_nt5 = convert_dataset(dataset_nt5, time_stamp_nt)
data_list.append(to_dat_nt5)
to_dat_nt6 = convert_dataset(dataset_nt6, time_stamp_nt)
data_list.append(to_dat_nt6)

dataset_file = open("/home/deepthi/dataset_file.log", "w")

for elem in data_list:
   dataset_file.write("%s\n"%elem )

print len(dataset_39)
