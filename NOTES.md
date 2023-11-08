
```bash
source ~ee106a/sawyer_setup.bash

# starts intera action server
rosrun intera_interface joint_trajectory_action_server.py
# starts intera action server
roslaunch sawyer_moveit_config sawyer_moveit.launch electric_gripper:=true
```
