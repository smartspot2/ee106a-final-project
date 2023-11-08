#!/usr/bin/env python

import matplotlib

# Python imports
import numpy as np

matplotlib.use("TkAgg")
import matplotlib.pyplot as plt

# ROS imports
import rospy

# Lab imports
from utils.utils import *

NUM_JOINTS = 7


class Controller:
    def __init__(self, limb, kin):
        """
        Constructor for the superclass. All subclasses should call the superconstructor

        Parameters
        ----------
        limb : :obj:`sawyer_interface.Limb` or :obj:`intera_interface.Limb`
        kin : :obj:`sawyer_pykdl.sawyer_kinematics`
            must be the same arm as limb
        """

        # Run the shutdown function when the ros node is shutdown
        rospy.on_shutdown(self.shutdown)
        self._limb = limb
        self._kin = kin

        # Set this attribute to True if the present controller is a jointspace controller.
        self.is_joinstpace_controller = False

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        makes a call to the robot to move according to it's current position and the desired position
        according to the input path and the current time. Each Controller below extends this
        class, and implements this accordingly.

        Parameters
        ----------
        target_position : 7x' or 6x' :obj:`numpy.ndarray`
            desired positions
        target_velocity : 7x' or 6x' :obj:`numpy.ndarray`
            desired velocities
        target_acceleration : 7x' or 6x' :obj:`numpy.ndarray`
            desired accelerations
        """
        pass

    def interpolate_path(self, path, t, current_index=0):
        """
        interpolates over a :obj:`moveit_msgs.msg.RobotTrajectory` to produce desired
        positions, velocities, and accelerations at a specified time

        Parameters
        ----------
        path : :obj:`moveit_msgs.msg.RobotTrajectory`
        t : float
            the time from start
        current_index : int
            waypoint index from which to start search

        Returns
        -------
        target_position : 7x' or 6x' :obj:`numpy.ndarray`
            desired positions
        target_velocity : 7x' or 6x' :obj:`numpy.ndarray`
            desired velocities
        target_acceleration : 7x' or 6x' :obj:`numpy.ndarray`
            desired accelerations
        current_index : int
            waypoint index at which search was terminated
        """

        # a very small number (should be much smaller than rate)
        epsilon = 0.0001

        max_index = len(path.joint_trajectory.points) - 1

        # If the time at current index is greater than the current time,
        # start looking from the beginning
        if path.joint_trajectory.points[current_index].time_from_start.to_sec() > t:
            current_index = 0

        # Iterate forwards so that you're using the latest time
        while (
            not rospy.is_shutdown()
            and current_index < max_index
            and path.joint_trajectory.points[current_index + 1].time_from_start.to_sec()
            < t + epsilon
        ):
            current_index = current_index + 1

        # Perform the interpolation
        if current_index < max_index:
            time_low = path.joint_trajectory.points[
                current_index
            ].time_from_start.to_sec()
            time_high = path.joint_trajectory.points[
                current_index + 1
            ].time_from_start.to_sec()

            target_position_low = np.array(
                path.joint_trajectory.points[current_index].positions
            )
            target_velocity_low = np.array(
                path.joint_trajectory.points[current_index].velocities
            )
            target_acceleration_low = np.array(
                path.joint_trajectory.points[current_index].accelerations
            )

            target_position_high = np.array(
                path.joint_trajectory.points[current_index + 1].positions
            )
            target_velocity_high = np.array(
                path.joint_trajectory.points[current_index + 1].velocities
            )
            target_acceleration_high = np.array(
                path.joint_trajectory.points[current_index + 1].accelerations
            )

            target_position = target_position_low + (t - time_low) / (
                time_high - time_low
            ) * (target_position_high - target_position_low)
            target_velocity = target_velocity_low + (t - time_low) / (
                time_high - time_low
            ) * (target_velocity_high - target_velocity_low)
            target_acceleration = target_acceleration_low + (t - time_low) / (
                time_high - time_low
            ) * (target_acceleration_high - target_acceleration_low)

        # If you're at the last waypoint, no interpolation is needed
        else:
            target_position = np.array(
                path.joint_trajectory.points[current_index].positions
            )
            target_velocity = np.array(
                path.joint_trajectory.points[current_index].velocities
            )
            target_acceleration = np.array(
                path.joint_trajectory.points[current_index].velocities
            )

        return (target_position, target_velocity, target_acceleration, current_index)

    def shutdown(self):
        """
        Code to run on shutdown. This is good practice for safety
        """
        rospy.loginfo("Stopping Controller")

        # Set velocities to zero
        self.stop_moving()
        rospy.sleep(0.1)

    def stop_moving(self):
        """
        Set robot joint velocities to zero
        """
        zero_vel_dict = joint_array_to_dict(np.zeros(NUM_JOINTS), self._limb)
        self._limb.set_joint_velocities(zero_vel_dict)


    def execute_path(self, path, rate=200, timeout=None, log=False):
        """
        takes in a path and moves the sawyer in order to follow the path.

        Parameters
        ----------
        path : :obj:`moveit_msgs.msg.RobotTrajectory`
        rate : int
            This specifies how many ms between loops.  It is important to
            use a rate and not a regular while loop because you want the
            loop to refresh at a constant rate, otherwise you would have to
            tune your PD parameters if the loop runs slower / faster
        timeout : int
            If you want the controller to terminate after a certain number
            of seconds, specify a timeout in seconds.
        log : bool
            whether or not to display a plot of the controller performance

        Returns
        -------
        bool
            whether the controller completes the path or not
        """

        # For interpolation
        max_index = len(path.joint_trajectory.points) - 1
        current_index = 0

        # For timing
        start_t = rospy.Time.now()
        r = rospy.Rate(rate)

        while not rospy.is_shutdown():
            # Find the time from start
            t = (rospy.Time.now() - start_t).to_sec()

            # If the controller has timed out, stop moving and return false
            if timeout is not None and t >= timeout:
                # Set velocities to zero
                self.stop_moving()
                return False

            current_position = get_joint_positions(self._limb)
            current_velocity = get_joint_velocities(self._limb)

            # Get the desired position, velocity, and effort
            (
                target_position,
                target_velocity,
                target_acceleration,
                current_index,
            ) = self.interpolate_path(path, t, current_index)

            # Run controller
            self.step_control(target_position, target_velocity, target_acceleration)

            # Sleep for a bit (to let robot move)
            r.sleep()

            if current_index >= max_index:
                self.stop_moving()
                break

        return True


class FeedforwardJointVelocityController(Controller):
    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        Parameters
        ----------
        target_position: 7x' ndarray of desired positions
        target_velocity: 7x' ndarray of desired velocities
        target_acceleration: 7x' ndarray of desired accelerations
        """
        # TODO: Implement Feedforward control
        controller_velocity = target_velocity

        self._limb.set_joint_velocities(
            joint_array_to_dict(controller_velocity, self._limb)
        )


class PIDJointVelocityController(Controller):
    """
    Look at the comments on the Controller class above.  This controller turns the desired workspace position and velocity
    into desired JOINT position and velocity.  Then it compares the difference between the sawyer's
    current JOINT position and velocity and desired JOINT position and velocity to come up with a
    joint velocity command and sends that to the sawyer.
    """

    def __init__(self, limb, kin, Kp, Ki, Kd, Kw):
        """
        Parameters
        ----------
        limb : :obj:`sawyer_interface.Limb`
        kin : :obj:`sawyerKinematics`
        Kp : 7x' :obj:`numpy.ndarray` of proportional constants
        Ki: 7x' :obj:`numpy.ndarray` of integral constants
        Kd : 7x' :obj:`numpy.ndarray` of derivative constants
        Kw : 7x' :obj:`numpy.ndarray` of anti-windup constants
        """
        Controller.__init__(self, limb, kin)
        self.Kp = np.diag(Kp)
        self.Ki = np.diag(Ki)
        self.Kd = np.diag(Kd)

        self.Kw = Kw

        self.integ_error = np.zeros(7)

        self.is_joinstpace_controller = True
        # --------
        self.current_error = np.zeros(7)
        self.current_timestamp = 0.0
        # --------

    def step_control(self, target_position, target_velocity, target_acceleration):
        """
        makes a call to the robot to move according to it's current position and the desired position
        according to the input path and the current time. Each Controller below extends this
        class, and implements this accordingly. This method should call
        self._limb.joint_angle and self._limb.joint_velocity to get the current joint position and velocity
        and self._limb.set_ current_joint_velocities() to set the joint velocity to something.  You may find
        joint_array_to_dict() in utils.py useful

        Parameters
        ----------
        target_position: 7x' :obj:`numpy.ndarray` of desired positions
        target_velocity: 7x' :obj:`numpy.ndarray` of desired velocities
        target_acceleration: 7x' :obj:`numpy.ndarray` of desired accelerations
        """
        current_position = get_joint_positions(self._limb)
        current_velocity = get_joint_velocities(self._limb)

        # TODO: implement PID control to set the joint velocities.

        prev_timestamp = self.current_timestamp
        self.current_timestamp = rospy.get_time()
        time_interval = self.current_timestamp - prev_timestamp
        prev_error = self.current_error
        self.current_error = target_position - current_position
        self.integ_error = self.Kw * self.integ_error + self.current_error

        p = self.Kp @ self.current_error
        i = self.Ki @ self.integ_error
        d = self.Kd @ (self.current_error - prev_error) / time_interval

        controller_velocity = target_velocity + p + i + d
        controller_velocity = controller_velocity.astype(float)

        # -------

        self._limb.set_joint_velocities(
            joint_array_to_dict(controller_velocity, self._limb)
        )
