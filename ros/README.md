# ViPlanner ROS Node

## Overview

ROS Node to run ViPlanner on the LeggedRobot Platform ANYmal.
The implementation consists of
- the `planner` itself, running a semantic segmentation network and ViPlanner in parallel
- a `visualizer` to project the path in the RGB and depth camera stream of the robot
- a `pathFollower` to translate the path into twist commands that can be executed by the robot
- an RViz plugin to set the waypoints for the planner


## Installation

Please refer to [Installation Instructions](./INSTALL.md) where details about the included docker and a manual install is given.

## Usage

For the legged platform ANYmal/ similar platforms, we provide a configuration file based on a mounted RGB-D camera ([Realsense d435i](https://www.intelrealsense.com/depth-camera-d435i/)). The configuration file is located in the [config](./planner/config/) folder. Before running the planner, make sure to adjust the configuration file to your needs. In the case a different camera or significantly different platform is used, please retrain the model.

After launching the ANYmal software stack, run the VIPlanner without visualization:

```bash
roslaunch viplanner_node viplanner.launch
```

By enabling the `WaypointTool` in RViz, you can set waypoints for the planner. The planner will track these waypoints.
It is recommended to visualize the path and the waypoints in RViz to verify the correct behavior.

## SmartJoystick

Press the **LB** button on the joystick, when seeing the output on the screen:

    Switch to Smart Joystick mode ...

Now the smartjoystick feature is enabled. It takes the joy stick command as motion intention and runs the VIPlanner in the background for low-level obstacle avoidance.
