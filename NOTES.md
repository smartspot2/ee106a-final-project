
```bash
source ~ee106a/sawyer_setup.bash

# starts intera action server
rosrun intera_interface joint_trajectory_action_server.py
# starts moveit
roslaunch sawyer_moveit_config sawyer_moveit.launch electric_gripper:=true
# start camera
roslaunch sawyer_full_stack sawyer_camera_track.launch

python main.py -task line -ar_marker ar_marker_16 -c pid
```
