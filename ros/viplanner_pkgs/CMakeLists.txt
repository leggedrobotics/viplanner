cmake_minimum_required(VERSION 3.16)
project(viplanner_pkgs)

#==
# Dependencies
#==

find_package(catkin REQUIRED COMPONENTS
    joy
    ps3joy
    path_follower
    viplanner_node
    viplanner_viz
    waypoint_rviz_plugin
)

#==
# Package Configuration
#==

catkin_package()

# EOF
