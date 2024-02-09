cmake_minimum_required(VERSION 3.16)
project(viplanner_viz)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
add_compile_options(-O2)

if(POLICY CMP0091)
    # https://stackoverflow.com/a/56490614
    cmake_policy(SET CMP0091 NEW)
endif()


set(CATKIN_PACKAGE_DEPENDENCIES
  roscpp
  roslib
  std_msgs
  image_transport
  sensor_msgs
  geometry_msgs
  message_generation
  tf
  tf2
  tf2_ros
  tf2_geometry_msgs
  rosbag
  cv_bridge
)

# Check whether open3d can be found
find_package(Boost REQUIRED COMPONENTS system)
find_package(Eigen3 REQUIRED)
find_package(OpenCV REQUIRED)

find_package(catkin REQUIRED COMPONENTS
  ${CATKIN_PACKAGE_DEPENDENCIES}
)

catkin_package(
 CATKIN_DEPENDS
  ${CATKIN_PACKAGE_DEPENDENCIES}
)

include_directories(
  ${catkin_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIRS}
)

add_executable(${PROJECT_NAME}_node src/viplannerViz.cpp)
target_link_libraries(${PROJECT_NAME}_node ${catkin_LIBRARIES} ${OpenCV_LIBS})  #  ${PROJECT_NAME}
