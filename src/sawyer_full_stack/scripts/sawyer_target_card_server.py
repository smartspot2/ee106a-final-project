#!/usr/bin/env python

import sys

import intera_interface
import rospy
from paths.path_planner import PathPlanner
from sawyer_pykdl import sawyer_kinematics
from trac_ik_python.trac_ik import IK
import numpy as np
from utils.utils import *
from utils.shared import *

from set_msgs.srv import TargetPosition

AR_MARKER = "ar_marker_3"
CONTROLLER = "pid"
LOOP_RATE = 200  # ms
TIMEOUT = 60  # seconds
NUM_WAYPOINTS = 10  # for robot trajectory

CARD_PICKUP_OFFSET = [-0.0, 0.02, 0.0]
AR_Z_POS = -0.13 # -0.14

# Setup
rospy.init_node("sawyer_target_card_server")

ik_solver = IK("base", "right_gripper_tip")
limb = intera_interface.Limb("right")
kin = sawyer_kinematics("right")
planner = PathPlanner("right_arm")  # moveit! wrapper (used to go to start position)
controller = get_controller(CONTROLLER, limb, kin)
# tuck()
tag_pos = lookup_tag(AR_MARKER)


def sawyer_target_card_callback(request):
    try:
        goal = np.array([request.position.x, request.position.y, request.position.z])
        robot_trajectory = (
            get_trajectory_ar_frame(
                limb, kin, ik_solver, tag_pos, NUM_WAYPOINTS, goal, CARD_PICKUP_OFFSET, AR_Z_POS
            )
            if request.use_ar_frame
            else get_trajectory(
                limb, kin, ik_solver, NUM_WAYPOINTS, goal, CARD_PICKUP_OFFSET
            )
        )

        # Move to the trajectory start position
        plan = planner.plan_to_joint_pos(
            robot_trajectory.joint_trajectory.points[0].positions
        )

        plan = planner.retime_trajectory(plan, 0.3)
        planner.execute_plan(plan[1])

        # execute the path using your own controller.
        done = controller.execute_path(
            robot_trajectory, rate=LOOP_RATE, timeout=TIMEOUT
        )
        if done:
            return "ok"
        else:
            print("Failed to move to position")
            sys.exit(0)

    except rospy.ServiceException as e:
        rospy.loginfo(e)


def sawyer_target_card_server():
    # rospy.init_node('sawyer_target_card_server')
    rospy.Service("/sawyer_target_card", TargetPosition, sawyer_target_card_callback)
    rospy.loginfo("Running sawyer_target_card server...")
    rospy.spin()  # Spin the node until Ctrl-C


if __name__ == "__main__":
    # setup()
    sawyer_target_card_server()
