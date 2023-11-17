#!/usr/bin/env python
"""
based on 106a lab7
"""
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


def main():
    """
    python scripts/main.py --help <------This prints out all the help messages
    """
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-ar_marker",
        "-ar",
        default="ar_marker_16",
        nargs="+",
        help="Which AR marker to use.",
    )
    parser.add_argument(
        "-controller_name",
        "-c",
        type=str,
        default="pid",
        help="Options: moveit, open_loop, pid.  Default: pid",
    )
    parser.add_argument(
        "-rate",
        type=int,
        default=200,
        help="""
        This specifies how many ms between loops.  It is important to use a rate
        and not a regular while loop because you want the loop to refresh at a
        constant rate, otherwise you would have to tune your PD parameters if 
        the loop runs slower / faster.  Default: 200""",
    )
    parser.add_argument(
        "-timeout",
        type=int,
        default=None,
        help="""after how many seconds should the controller terminate if it hasn\'t already.  
        Default: None""",
    )
    parser.add_argument(
        "-waypoints",
        type=int,
        default=50,
        help="How many waypoints for the :obj:`moveit_msgs.msg.RobotTrajectory`.  Default: 50",
    )
    parser.add_argument(
        "-goal",
        type=str,
        default="0.0,0.0,0.5",
        help="Goal in terms of offset from aruco marker. x forward/back, y right/left, z up/down.  Default: 0.0,0.0,0.5",
    )
    args = parser.parse_args()
    goal_offset = [float(n) for n in args.goal.split(",")]

    rospy.init_node("moveit_node")

    tuck()

    # this is used for sending commands (velocity, torque, etc) to the robot
    ik_solver = IK("base", "right_gripper_tip")
    limb = intera_interface.Limb("right")
    kin = sawyer_kinematics("right")

    # Lookup the AR tag position.
    tag_pos = [lookup_tag(marker) for marker in args.ar_marker]

    robot_trajectory = get_trajectory(limb, kin, ik_solver, tag_pos, args.waypoints, goal_offset)

    # This is a wrapper around MoveIt! for you to use.  We use MoveIt! to go to the start position
    # of the trajectory
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
    if args.controller_name != "moveit":
        plan = planner.retime_trajectory(plan, 0.3)
    planner.execute_plan(plan[1])

    if args.controller_name == "moveit":
        try:
            input("Press <Enter> to execute the trajectory using MOVEIT")
        except KeyboardInterrupt:
            sys.exit()
        # Uses MoveIt! to execute the trajectory.
        planner.execute_plan(robot_trajectory)
    else:
        controller = get_controller(args.controller_name, limb, kin)
        try:
            input("Press <Enter> to execute the trajectory using YOUR OWN controller")
        except KeyboardInterrupt:
            sys.exit()
        # execute the path using your own controller.
        done = controller.execute_path(
            robot_trajectory, rate=args.rate, timeout=args.timeout
        )
        if not done:
            print("Failed to move to position")
            sys.exit(0)


if __name__ == "__main__":
    main()
