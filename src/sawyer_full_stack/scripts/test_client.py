#!/usr/bin/env python
import rospy
from set_msgs.srv import TargetPosition
from geometry_msgs.msg import Point


def patrol_client():
    rospy.init_node('test_target_position_client')
    rospy.wait_for_service('/sawyer_target_card')
    try:
        patrol_proxy = rospy.ServiceProxy('/sawyer_target_card', TargetPosition)
        rospy.loginfo('test_target_position_client ready')
        pos = Point(0.0, 0.0, 0.1)
        patrol_proxy(pos)

        input("ENTER to continue")
        pos = Point(0.1, 0.1, 0.3)
        patrol_proxy(pos)

        input("ENTER to continue")
        pos = Point(0.2, 0.2, 0.1)
        patrol_proxy(pos)
    except rospy.ServiceException as e:
        rospy.loginfo(e)


if __name__ == '__main__':
    patrol_client()

