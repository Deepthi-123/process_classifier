#!/usr/bin/env python
import rospy
import message_filters
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String
import pandas
import sys


# This function is called whenever a wrench data set arrives (now does nothing)
def callback_wrench(data):
    rospy.loginfo("I heard %s", data.wrench)
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
        rospy.logwarn("I heard %s", data1.data)

        # Checks if statename as already occured, if not gives a unique id and stores it

        if not statename in dict_status_num:
            dict_status_num[statename] = cnt
            cnt += 1

        rospy.logwarn("%s, state %s", dict_status_num[statename], statename)

#def callback(data1):
#    global cache
#    if (callback1(data1)): 
#        rospy.loginfo("I heard %s", data1.data)
#        rospy.loginfo("%s", cache.getElemAfterTime(cache.getLastestTime()))

lst = []
count = 0

def check_for_count(data1):
   global count
   global cache
   global lst
   
   if (count == 1):
       lst.append(cache.getElemAfterTime(cache.getLastestTime()))
       rospy.loginfo("I heard %s", data1.data)
       rospy.loginfo("the count value is %d", count)
       thefile = open("/home/deepthi/online.log")
       for item in lst:
          thefile.write("%s\n" % item)

   elif (count == 0):
       rospy.loginfo("I heard %s", data1.data)
       rospy.loginfo("the count value is %d", count)
       pass

#rospy.loginfo('[%s]' % ', '.join(map(str, lst)))


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

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    wrench = message_filters.Subscriber("/wrench", WrenchStamped)


    cache = message_filters.Cache(wrench, cache_size=10, allow_headerless=True)


#    rospy.Subscriber("/dnb_executor/log", String, callback1)
#   rospy.Subscriber("/dnb_executor/log", String, callback)
    rospy.Subscriber("/dnb_executor/log", String, check_for_state)
    rospy.Subscriber("/dnb_executor/log", String, check_for_count)
   # rospy.Subscriber("/dnb_executor/log", String, check_state)
    #rospy.Subscriber("/wrench", WrenchStamped, callback)
    rospy.spin()

if __name__ == '__main__':
    listener()
