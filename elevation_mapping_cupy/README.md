# Example 1: Turtle Simple Example
Set the env in the Dockerfile
```dockerfile
ENV TURTLEBOT3_MODEL=waffle_realsense_depth
```
or in the terminal
```bash
export TURTLEBOT3_MODEL=waffle_realsense_depth
```
If using zenoh as rmw then start one terminal up and run the router
```bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```

Now launch the turtlebot3 in Gazebo with the following command:
```bash
source /usr/share/gazebo/setup.bash
ros2 launch turtlebot3_gazebo turtlebot3_world.launch.py
``` 

Launch the elevation mapping node with the configs for the turtle. Set use_python_node to true to use it instead of the cpp node:
```bash
ros2 launch elevation_mapping_cupy elevation_mapping_turtle.launch.py use_python_node:=false
```

If you want to drive the turtlebot around using the keyboard then run:
```bash
ros2 run turtlebot3_teleop teleop_keyboard 
```

# Example 2: Vortex Studio CAT 299 D3 CTL Example
We assume properly sourced terminals, with `RMW_IMPLEMENTATION=rmw_zenoh_cpp`, and the same `ROS_DOMAIN_ID=12`.

If using zenoh as rmw then start one terminal up and run the router.
```bash
ros2 run rmw_zenoh_cpp rmw_zenohd
```

Connect an XBOX controller to the linux machine and run the joy node
```bash
ros2 run joy joy_node --ros-args -p use_sim_time:=true
ros2 launch ctl_blade_controller joy-launch.py use_sim_time:=true
```

On the Windows Machine launch the Vortex :
```bash
ros2 run ros2_vortex_SIL CTL_SIL_node
``` 

Launch the elevation mapping node with the configs for the CTL:
```bash
ros2 launch elevation_mapping_cupy elevation_mapping_vortex_CTL.launch.py
ros2 launch elevation_mapping_cupy elevation_mapping_real_CTL.launch.py
```

Launch the controller
```bash
ros2 launch ctl_blade_controller launch_blade_control.launch.py use_sim:=true
```