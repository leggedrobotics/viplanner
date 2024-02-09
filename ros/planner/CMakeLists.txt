cmake_minimum_required(VERSION 2.8.3)
project(viplanner_node)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  roscpp
  rospy
  std_msgs
  image_transport
  sensor_msgs
  geometry_msgs
  message_generation
)

## System dependencies are found with CMake's conventions
find_package(Boost REQUIRED COMPONENTS system)

catkin_package(
#  INCLUDE_DIRS include
 CATKIN_DEPENDS
 roscpp
 rospy
 std_msgs
 geometry_msgs
#  DEPENDS system_lib
)

catkin_install_python(PROGRAMS src/viplanner_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)
