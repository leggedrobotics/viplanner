#!/bin/bash

PKGROOT="$( realpath "$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null 2>&1 && pwd )"/../ )"
echo -e "\e[1;32m[run.sh]: Package root is '$PKGROOT'.\e[0m"

set -e

source $PKGROOT/bin/scripts/jetson_l4t_version.sh
IMAGE="rslethz/jetpack-5:r$L4T_VERSION-viplanner_test_new"

echo -e "[run.sh]: \e[1;32mSetting max fan speed to increase performance.\e[0m"
sudo /usr/bin/jetson_clocks --fan

echo -e "[run.sh]: \e[1;32mRunning docker image '$IMAGE'.\e[0m"

RUN_COMMAND="docker run \
  -it \
  --net=host \
  --runtime nvidia \
  --dns 8.8.8.8 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix/:/tmp/.X11-unix \
  -v $HOME/git/:/root/git \
  -v $HOME/catkin_ws/:/root/catkin_ws/ \
  -v /etc/timezone:/etc/timezone \
  -v /etc/localtime:/etc/localtime \
  -v $HOME/git/viplanner:/viplanner \
  $IMAGE
"
echo -e "[run.sh]: \e[1;32mThe final run command is\n\e[0;35m$RUN_COMMAND\e[0m."
$RUN_COMMAND
echo -e "[run.sh]: \e[1;32mDocker terminal closed.\e[0m"
