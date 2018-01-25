#!/usr/bin/env python
import rospy
import message_filters
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String
import pandas
import sys

tick_list = []
# This function is called whenever a wrench data set arrives (now does nothing)
def callback_wrench(data):
    global t
    global tick
    global tick_list
    global lst
#    global p = 0
    #rospy.loginfo("I heard %s", data.wrench)
    t = data.header.stamp
    tick = tick + 1
    pass
    

# Global variables to simplify running numbers of states/skills
dict_status_num = {}
cnt = 0



# This function is called whenever a information about the active state arrives
def callback_log(data):
    global dict_status_num, cnt

    # Checks if a new state is entered
    if data.data.startswith("Entering state") and data.data.endswith("Place Erebus"):
        statename = data.data[15:] # Cuts the first 16 characters of the message
       # rospy.logwarn("I heard %s", data1.data)

        # Checks if statename as already occured, if not gives a unique id and stores it

        if not statename in dict_status_num:
            dict_status_num[statename] = cnt
            cnt += 1

#        rospy.logwarn("%s, state %s", dict_status_num[statename], statename)

#def callback(data1):
#    global cache
#    if (callback1(data1)): 
#        rospy.loginfo("I heard %s", data1.data)
#        rospy.loginfo("%s", cache.getElemAfterTime(cache.getLastestTime()))

lst = []
x_list = []
y_list = []
z_list = []
count = 0

def data_formatting(list_):
   global cache
   global t
   global x_list
   global y_list
   global z_list
   
   for i, elem in enumerate(list_):
      list_[i] = (str(list_[i]).split("torque:"))[0]
      list_[i] = list_[i].split("force:")[1]
      list_[i] = list_[i].replace("\n", ",")
      x = (list_[i].split(","))[1].replace("x: ", "")
      y = (list_[i].split(","))[2].replace("y: ", "")
      z = (list_[i].split(","))[3].replace("z: ", "")
      x_list.append(float(x))
      y_list.append(float(y))
      z_list.append(float(z))
n = 0
def check_for_count(data1):
   global count
   global cache
   global lst
   global t
   global tick
   global t_list
   global dat
   global n
   t_list = []
   global x_list
   #lst = []
   if (count == 1): 
       tick = 1
       for i in range(200):
           rospy.logwarn(t)
#           rospy.logwarn(tick)
           rospy.logwarn(cache.getElemAfterTime(t))
           if not tick in t_list:
               t_list.append(tick)
           if not (cache.getElemAfterTime(t)) in lst:
               lst.append(cache.getElemAfterTime(t))
               n = n+1
      # rospy.loginfo(lst)

   elif (count == 0):
       rospy.loginfo("I heard %s", data1.data)
       rospy.loginfo("the count value is %d", count)
       pass



def check_for_state(data1):
    global count
    global cache
    if data1.data.startswith("Entering state") and data1.data.endswith("Pitasc-Sub - White Part Mount Tilted"):
        count = 1

    elif data1.data.startswith("Leaving state") and data1.data.endswith("Pitasc-Sub - White Part Mount Tilted"): 
        count = 0

    elif not (data1.data.startswith("Leaving state") and data1.data.endswith("Pitasc-Sub - White Part Mount Tilted")): 
        if (count == 1):
            count = 1
        elif (count == 0):
            count = 0

    

        
def listener():
    global cache
    global t
    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    wrench = message_filters.Subscriber("/wrench", WrenchStamped)
    rospy.Subscriber("/wrench", WrenchStamped, callback_wrench)

    cache = message_filters.Cache(wrench, cache_size=10, allow_headerless=True)

    #t = cache.getLastestTime()
#    rospy.Subscriber("/dnb_executor/log", String, callback1)
#   rospy.Subscriber("/dnb_executor/log", String, callback)
    rospy.Subscriber("/dnb_executor/log", String, check_for_state)
    rospy.Subscriber("/dnb_executor/log", String, check_for_count)
    #rospy.Subscriber("/wrench", WrenchStamped, data_formatting)
   # rospy.Subscriber("/dnb_executor/log", String, callback_log)
   # rospy.Subscriber("/dnb_executor/log", String, check_state)
    #rospy.Subscriber("/wrench", WrenchStamped, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
    data_formatting(lst)
    rospy.loginfo(x_list)
    rospy.loginfo(len(x_list))
    rospy.loginfo(len(y_list))
    rospy.loginfo(len(lst))
    #thefile = open("/home/deepthi/online_test.log", "w")
    #for item in lst:
    rospy.loginfo(n)
      #  thefile.write("%s\n" % item)

