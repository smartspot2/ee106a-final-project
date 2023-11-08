#!/usr/bin/env/python

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from mpl_toolkits.mplot3d import Axes3D

"""
Set of classes for defining SE(3) trajectories for the end effector of a robot 
manipulator
"""

class Trajectory:
    def __init__(self, total_time):
        """
        Parameters
        ----------
        total_time : float
        	desired duration of the trajectory in seconds 
        """
        self.total_time = total_time

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        pass

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.

        The function get_g_matrix from utils may be useful to perform some frame
        transformations.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        pass

    def display_trajectory(self, num_waypoints=67, show_animation=False, save_animation=False):
        """
        Displays the evolution of the trajectory's position and body velocity.

        Parameters
        ----------
        num_waypoints : int
            number of waypoints in the trajectory
        show_animation : bool
            if True, displays the animated trajectory
        save_animatioon : bool
            if True, saves a gif of the animated trajectory
        """
        trajectory_name = self.__class__.__name__
        times = np.linspace(0, self.total_time, num=num_waypoints)
        target_positions = np.vstack([self.target_pose(t)[:3] for t in times])
        target_velocities = np.vstack([self.target_velocity(t)[:3] for t in times])
        
        fig = plt.figure(figsize=plt.figaspect(0.5))
        colormap = plt.cm.brg(np.fmod(np.linspace(0, 1, num=num_waypoints), 1))

        # Position plot
        ax0 = fig.add_subplot(1, 2, 1, projection='3d')
        pos_boundaries = [[-2, 2],
                           [-2, 2],
                           [-2, 2]]
        pos_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax0.set_xlim3d([max(pos_boundaries[0][0], min(target_positions[:, 0]) + pos_padding[0][0]), 
                        min(pos_boundaries[0][1], max(target_positions[:, 0]) + pos_padding[0][1])])
        ax0.set_xlabel('X')
        ax0.set_ylim3d([max(pos_boundaries[1][0], min(target_positions[:, 1]) + pos_padding[1][0]), 
                        min(pos_boundaries[1][1], max(target_positions[:, 1]) + pos_padding[1][1])])
        ax0.set_ylabel('Y')
        ax0.set_zlim3d([max(pos_boundaries[2][0], min(target_positions[:, 2]) + pos_padding[2][0]), 
                        min(pos_boundaries[2][1], max(target_positions[:, 2]) + pos_padding[2][1])])
        ax0.set_zlabel('Z')
        ax0.set_title("%s evolution of\nend-effector's position." % trajectory_name)
        line0 = ax0.scatter(target_positions[:, 0], 
                        target_positions[:, 1], 
                        target_positions[:, 2], 
                        c=colormap,
                        s=2)

        # Velocity plot
        ax1 = fig.add_subplot(1, 2, 2, projection='3d')
        vel_boundaries = [[-2, 2],
                           [-2, 2],
                           [-2, 2]]
        vel_padding = [[-0.1, 0.1],
                        [-0.1, 0.1],
                        [-0.1, 0.1]]
        ax1.set_xlim3d([max(vel_boundaries[0][0], min(target_velocities[:, 0]) + vel_padding[0][0]), 
                        min(vel_boundaries[0][1], max(target_velocities[:, 0]) + vel_padding[0][1])])
        ax1.set_xlabel('X')
        ax1.set_ylim3d([max(vel_boundaries[1][0], min(target_velocities[:, 1]) + vel_padding[1][0]), 
                        min(vel_boundaries[1][1], max(target_velocities[:, 1]) + vel_padding[1][1])])
        ax1.set_ylabel('Y')
        ax1.set_zlim3d([max(vel_boundaries[2][0], min(target_velocities[:, 2]) + vel_padding[2][0]), 
                        min(vel_boundaries[2][1], max(target_velocities[:, 2]) + vel_padding[2][1])])
        ax1.set_zlabel('Z')
        ax1.set_title("%s evolution of\nend-effector's translational body-frame velocity." % trajectory_name)
        line1 = ax1.scatter(target_velocities[:, 0], 
                        target_velocities[:, 1], 
                        target_velocities[:, 2], 
                        c=colormap,
                        s=2)

        if show_animation or save_animation:
            def func(num, line):
                line[0]._offsets3d = target_positions[:num].T
                line[0]._facecolors = colormap[:num]
                line[1]._offsets3d = target_velocities[:num].T
                line[1]._facecolors = colormap[:num]
                return line

            # Creating the Animation object
            line_ani = animation.FuncAnimation(fig, func, frames=num_waypoints, 
                                                          fargs=([line0, line1],), 
                                                          interval=max(1, int(1000 * self.total_time / (num_waypoints - 1))), 
                                                          blit=False)
        plt.show()
        if save_animation:
            line_ani.save('%s.gif' % trajectory_name, writer='pillow', fps=60)
            print("Saved animation to %s.gif" % trajectory_name)

class LinearTrajectory(Trajectory):
    def __init__(self, start_position, goal_position, goal_orientation, total_time):

        Trajectory.__init__(self, total_time)
        self.start_position = start_position
        self.goal_position = goal_position
        self.distance = self.goal_position - self.start_position
        self.acceleration = (self.distance * 4.0) / (self.total_time ** 2) # keep constant magnitude acceleration
        self.v_max = (self.total_time / 2.0) * self.acceleration # maximum velocity magnitude
        self.desired_orientation = goal_orientation

    def target_pose(self, time):
        """
        Returns where the arm end effector should be at time t, in the form of a 
        7D vector [x, y, z, qx, qy, qz, qw]. i.e. the first three entries are 
        the desired end-effector position, and the last four entries are the 
        desired end-effector orientation as a quaternion, all written in the 
        world frame.

        Hint: The end-effector pose with the gripper pointing down corresponds 
        to the quaternion [0, 1, 0, 0]. 

        Parameters
        ----------
        time : float        
    
        Returns
        -------
        7x' :obj:`numpy.ndarray`
            desired configuration in workspace coordinates of the end effector
        """
        if time <= self.total_time / 2.0:
            pos = self.start_position + self.acceleration * time * time / 2.0
        else:
            _t = (time - self.total_time/2.0)
            pos = self.start_position + self.distance/2.0 + self.v_max * _t - self.acceleration * _t * _t / 2.0
        return np.hstack((pos, self.desired_orientation))

    def target_velocity(self, time):
        """
        Returns the end effector's desired body-frame velocity at time t as a 6D
        twist. Note that this needs to be a rigid-body velocity, i.e. a member 
        of se(3) expressed as a 6D vector.

        The function get_g_matrix from utils may be useful to perform some frame
        transformations.

        Parameters
        ----------
        time : float

        Returns
        -------
        6x' :obj:`numpy.ndarray`
            desired body-frame velocity of the end effector
        """
        if time <= self.total_time / 2.0:
            linear_vel = np.minimum(self.acceleration * time, self.v_max)
        else:
            _t = time - self.total_time / 2.0
            linear_vel = self.v_max - np.minimum(self.acceleration * _t, self.v_max)
        return np.hstack((linear_vel, np.zeros(3)))
