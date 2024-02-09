#!/usr/bin/env bash
#
# Builds the self-contained JetPack Docker container, including development
# headers/libraries/samples for CUDA Toolkit.
# This container differs from original jetpack for the OpenCV version which is not
# 4.5 but 4.2.0 for better compatibility with ROS Noetic packages.
# It also has already pre-installed ROS Noetic distro.
#
# Run this script as follows:
#
#   $ cd jetson-containers-rsl
#   $ cd jetson-containers
#   $ ./../scripts/docker_build_ros_base_rsl.sh
#
PKGROOT="$( realpath "$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd )"/../ )"
echo -e "\e[1;32m[build.sh]: Package root is '$PKGROOT'.\e[0m"

set -e

source $PKGROOT/bin/scripts/jetson_l4t_version.sh
source $PKGROOT/bin/scripts/jetson_docker_base.sh

CONTAINER="rslethz/jetpack-5:r$L4T_VERSION-viplanner_test_new"
DOCKERFILE="$PKGROOT/Dockerfile"

# build container
echo -e "\e[1;32m[build.sh]: Building container '$CONTAINER'.\e[0m"
echo -e "\e[1;32m[build.sh]: BASE_IMAGE='$BASE_IMAGE_L4T'.\e[0m"

sudo docker build --network=host -t $CONTAINER -f $DOCKERFILE --build-arg BASE_IMAGE=$BASE_IMAGE_L4T ./..
