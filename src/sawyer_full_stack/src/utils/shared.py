import numpy as np
import roslaunch
import rospkg
import rospy
import tf2_ros
import tf
from controllers.controllers import (
    FeedforwardJointVelocityController,
    PIDJointVelocityController,
)
from paths.paths import MotionPath
from paths.trajectories import LinearTrajectory
from utils.utils import *

MOVE_TIME = 4.0 # seconds

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
    return lookup_transform("base", tag_name).transform


def get_trajectory_ar_frame(limb, kin, ik_solver, tag_transform, num_waypoints, ar_tag_goal, offset, z_pos):
    """
    Paramters:
        tag_transform - transformation between base and AR tag (rotation + position)
        ar_tag_goal - goal position in AR tag frame (non-homogeneous coordinate)
    """

    trans = lookup_transform("base", "right_gripper_tip")
    rospy.loginfo(f"transformation {trans}")

    current_position = np.array(
        [getattr(trans.transform.translation, dim) for dim in ("x", "y", "z")]
    )
    rospy.loginfo(f"Current Position: {current_position}")
    rospy.loginfo(f"tag_transform {tag_transform}")

    
    tag_to_base = tf.transformations.quaternion_matrix(
        [
            tag_transform.rotation.x,
            tag_transform.rotation.y,
            tag_transform.rotation.z,
            tag_transform.rotation.w,
        ]
    )

    tag_to_base[:3, 3] = [
        tag_transform.translation.x,
        tag_transform.translation.y,
        tag_transform.translation.z,
    ]

    # tag_to_base_trans = np.array(
    #     [
    #         [
    #             tag_transform.translation.x,
    #             tag_transform.translation.y,
    #             tag_transform.translation.z,
    #             1,
    #         ]
    #     ]
    # ).T

    rospy.loginfo(f"tag_to_base {tag_to_base}")

    ar_tag_goal_hom = np.array([*ar_tag_goal, 1])

    rospy.loginfo(f"ar_tag_goal_hom {ar_tag_goal_hom}")

    target_pos = tag_to_base @ ar_tag_goal_hom + np.array([*offset, 0])
    target_pos[2] = z_pos # hard-coded z position

    print("Offset:", offset)
    print("Target Position:", target_pos)
    trajectory = LinearTrajectory(
        start_position=current_position,
        goal_position=target_pos[:3],
        # goal_position=np.array([0.576, -0.047, 0.0]),
        # goal_position=current_position,
        goal_orientation=np.array([0.5, 0.5, 0.5, 0.5]),
        # goal_orientation=np.array(
        #     [
        #         trans.transform.rotation.x,
        #         trans.transform.rotation.y,
        #         trans.transform.rotation.z,
        #         trans.transform.rotation.w,
        #     ]
        # ),
        total_time=MOVE_TIME,
    )

    path = MotionPath(limb, kin, ik_solver, trajectory)
    return path.to_robot_trajectory(num_waypoints, True)


def get_trajectory(limb, kin, ik_solver, num_waypoints, goal, offset=[0, 0, 0]):
    """
    Paramters:
        goal - goal position in base frame
    """

    trans = lookup_transform("base", "right_gripper_tip")
    rospy.loginfo(f"transformation {trans}")

    current_position = np.array(
        [getattr(trans.transform.translation, dim) for dim in ("x", "y", "z")]
    )
    rospy.loginfo(f"Current Position: {current_position}")

    rospy.loginfo(f"goal {goal}")

    target_pos = goal + offset

    print("Offset:", offset)
    print("Target Position:", target_pos)
    trajectory = LinearTrajectory(
        start_position=current_position,
        goal_position=target_pos,
        goal_orientation=np.array([0.5, 0.5, 0.5, 0.5]),
        total_time=MOVE_TIME,
    )

    path = MotionPath(limb, kin, ik_solver, trajectory)
    return path.to_robot_trajectory(num_waypoints, True)


def get_controller(controller_name, limb, kin):
    if controller_name == "open_loop":
        controller = FeedforwardJointVelocityController(limb, kin)
    elif controller_name == "pid":
        # Kp = 0.1 * np.array([0.4, 2, 1.7, 1.5, 2, 2, 3])
        Kp = 0.01 * np.array([0.4, 2, 1.7, 1.5, 2, 2, 3])
        # Kd = 0.02 * np.array([2, 1, 2, 0.5, 0.8, 0.8, 0.8])
        Kd = 0.03 * np.array([2, 1, 2, 0.5, 0.8, 0.8, 0.8])
        # Ki = 0.02 * np.array([1.4, 1.4, 1.4, 1, 0.6, 0.6, 0.6])
        Ki = 0.04 * np.array([1.4, 1.4, 1.4, 1, 0.6, 0.6, 0.6])
        Kw = np.array([0.9, 0.9, 0.9, 0.9, 0.9, 0.9, 0.9])
        controller = PIDJointVelocityController(limb, kin, Kp, Ki, Kd, Kw)
    else:
        raise ValueError("Controller {} not recognized".format(controller_name))
    return controller
