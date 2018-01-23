#!/usr/bin/env python
import rospy
import message_filters
from geometry_msgs.msg import WrenchStamped
from std_msgs.msg import String
import sys


# This function is called whenever a wrench data set arrives (now does nothing)
def callback_wrench(data):
    rospy.loginfo("I heard %s", data.wrench)
    pass

# Global variables to simplify running numbers of states/skills
dict_status_num = {}
cnt = 0



# This function is called whenever a information about the active state arrives
def callback_log(data1, data2):
    global dict_status_num, cnt


    # Checks if a new state is entered
    if data.data.startswith("Entering state") and data.data.endswith("Place Erebus"):
        statename = data.data[15:] # Cuts the first 16 characters of the message
        rospy.logwarn("I heard %s", data1.data)
        rospy.loginfo("%s", data2.wrench)
        # Checks if statename as already occured, if not gives a unique id and stores it

        if not statename in dict_status_num:
            dict_status_num[statename] = cnt
            cnt += 1

        rospy.logwarn("%s, state %s", dict_status_num[statename], statename)

def callback(data1):
    global cache
    rospy.logwarn("I heard %s", data1.data)
    rospy.logwarn(str(cache.getElemBeforeTime(cache.getLastestTime())))
    #rospy.logwarn("%s", data2.wrench)
    

def listener():
    global cache

    # In ROS, nodes are uniquely named. If two nodes with the same
    # node are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('listener', anonymous=True)

    wrench = message_filters.Subscriber("/wrench", WrenchStamped)
    #swrench.registerCallback(callback_log)
    #log = message_filters.Subscriber("/dnb_executor/log", String)

    cache = message_filters.Cache(wrench, cache_size=10, allow_headerless=True)
    rospy.Subscriber("/dnb_executor/log", String, callback)
    #cache.registerCallback(callback)
    #total = message_filters.ApproximateTimeSynchronizer([wrench,log],10,1, allow_headerless=True)
    #total.registerCallback(callback)

    #rospy.Subscriber("/wrench", WrenchStamped, callback_wrench)
    #rospy.Subscriber("/dnb_executor/log", String, callback_log)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    listener()
