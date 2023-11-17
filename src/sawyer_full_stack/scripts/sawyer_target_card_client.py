#!/usr/bin/env python

import argparse
import sys

import intera_interface
import rospy
from moveit_msgs.msg import DisplayTrajectory, RobotState
from paths.path_planner import PathPlanner
from sawyer_pykdl import sawyer_kinematics
from trac_ik_python.trac_ik import IK
from utils.utils import *
from utils.shared import *

AR_MARKER = "ar_marker_16"
CONTROLLER = "pid"
LOOP_RATE = 200 # ms
TIMEOUT = 60 # seconds
NUM_WAYPOINTS = 50 # for robot trajectory


def main():

    rospy.init_node("moveit_node")
    tuck()

    # this is used for sending commands (velocity, torque, etc) to the robot
    ik_solver = IK("base", "right_gripper_tip")
    limb = intera_interface.Limb("right")
    kin = sawyer_kinematics("right")

    # Lookup the AR tag position.
    tag_pos = [lookup_tag(AR_MARKER)]

    robot_trajectory = get_trajectory(limb, kin, ik_solver, tag_pos, NUM_WAYPOINTS, GOALLLLL)

    # moveit! wrapper (used to go to start position)
    planner = PathPlanner("right_arm")

    # By publishing the trajectory to the move_group/display_planned_path topic, you should
    # be able to view it in RViz.  You will have to click the "loop animation" setting in
    # the planned path section of MoveIt! in the menu on the left side of the screen.
    pub = rospy.Publisher(
        "move_group/display_planned_path", DisplayTrajectory, queue_size=10
    )
    disp_traj = DisplayTrajectory()
    disp_traj.trajectory.append(robot_trajectory)
    disp_traj.trajectory_start = RobotState()
    pub.publish(disp_traj)

    # Move to the trajectory start position
    plan = planner.plan_to_joint_pos(
        robot_trajectory.joint_trajectory.points[0].positions
    )

    plan = planner.retime_trajectory(plan, 0.3)
    planner.execute_plan(plan[1])

    controller = get_controller(CONTROLLER, limb, kin)
    try:
        input("Press <Enter> to execute the trajectory using YOUR OWN controller")
    except KeyboardInterrupt:
        sys.exit()
    # execute the path using your own controller.
    done = controller.execute_path(
        robot_trajectory, rate=LOOP_RATE, timeout=TIMEOUT
    )
    if not done:
        print("Failed to move to position")
        sys.exit(0)


if __name__ == "__main__":
    main()
