import numpy as np
import roslaunch
import rospkg
import rospy
import tf2_ros
from controllers.controllers import FeedforwardJointVelocityController, PIDJointVelocityController
from paths.paths import MotionPath
from paths.trajectories import LinearTrajectory
from utils.utils import *


def tuck():
    """
    Tuck the robot arm to the start position. Use with caution
    """
    if input("Would you like to tuck the arm? (y/n): ") == "y":
        rospack = rospkg.RosPack()
        path = rospack.get_path("sawyer_full_stack")
        launch_path = path + "/launch/custom_sawyer_tuck.launch"
        uuid = roslaunch.rlutil.get_or_generate_uuid(None, False)
        roslaunch.configure_logging(uuid)
        launch = roslaunch.parent.ROSLaunchParent(uuid, [launch_path])
        launch.start()
    else:
        print("Canceled. Not tucking the arm.")



def lookup_transform(a, b):
    tfBuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfBuffer)
    trans = None
    try:
        trans = tfBuffer.lookup_transform(
            str(a), str(b), rospy.Time(0), rospy.Duration(10.0)
        )
    except Exception as e:
        print("error in lookup_transform:", e)
    return trans


def lookup_tag(tag_name):
    trans = lookup_transform("base", tag_name)
    tag_pos = [getattr(trans.transform.translation, dim) for dim in ("x", "y", "z")]
    return np.array(tag_pos)


def get_trajectory(limb, kin, ik_solver, tag_pos, num_waypoints, goal_offset):

    trans = lookup_transform("base", "right_hand")

    current_position = np.array(
        [getattr(trans.transform.translation, dim) for dim in ("x", "y", "z")]
    )
    print("Current Position:", current_position)

    target_pos = tag_pos[0]
    target_pos += goal_offset

    print("Target Position:", target_pos)
    trajectory = LinearTrajectory(
        start_position=current_position,
        goal_position=target_pos,
        goal_orientation=np.array([0.5, -0.5, 0.5, -0.5]),
        total_time=9,
    )

    path = MotionPath(limb, kin, ik_solver, trajectory)
    return path.to_robot_trajectory(num_waypoints, True)


def get_controller(controller_name, limb, kin):
    if controller_name == "open_loop":
        controller = FeedforwardJointVelocityController(limb, kin)
    elif controller_name == "pid":
        Kp = 0.1 * np.array([0.4, 2, 1.7, 1.5, 2, 2, 3])
        Kd = 0.02 * np.array([2, 1, 2, 0.5, 0.8, 0.8, 0.8])
        Ki = 0.02 * np.array([1.4, 1.4, 1.4, 1, 0.6, 0.6, 0.6])
        Kw = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        controller = PIDJointVelocityController(limb, kin, Kp, Ki, Kd, Kw)
    else:
        raise ValueError("Controller {} not recognized".format(controller_name))
    return controller

