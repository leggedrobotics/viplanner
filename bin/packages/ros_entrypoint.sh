#!/bin/bash

# Adapted from
# https://github.com/dusty-nv/jetson-containers/blob/eb2307d40f0884d66310e9ac34633a4c5ef2e083/packages/ros_entrypoint.sh

figlet VIPlanner

set -e

ros_env_setup="/opt/ros/$ROS_DISTRO/setup.bash"
echo "sourcing   $ros_env_setup"
source "$ros_env_setup"

echo "ROS_ROOT   $ROS_ROOT"
echo "ROS_DISTRO $ROS_DISTRO"

exec "$@"
