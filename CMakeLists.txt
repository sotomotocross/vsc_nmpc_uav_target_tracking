cmake_minimum_required(VERSION 3.0.2)
project(vsc_nmpc_uav_target_tracking)

## Compile as C++11, supported in ROS Kinetic and newer
set(CMAKE_CXX_STANDARD 11)

## Find catkin macros and libraries
## if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
## is used, also find other catkin packages
find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  message_generation
  roscpp
  rospy
  std_msgs
  img_seg_cnn
  mavros
  mavros_msgs
  sensor_msgs
  yaml-cpp
)

## System dependencies are found with CMake's conventions
## find_package(Boost REQUIRED COMPONENTS system)

## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
## catkin_python_setup()

## Declare ROS messages, services, and actions
add_message_files(
  FILES
  rec.msg
)

## Generate added messages and services with any dependencies listed here
generate_messages(
  DEPENDENCIES
  geometry_msgs
  std_msgs
)

## Declare a catkin package
catkin_package(
  INCLUDE_DIRS include
  LIBRARIES ${PROJECT_NAME}
  CATKIN_DEPENDS message_runtime geometry_msgs message_generation roscpp rospy std_msgs img_seg_cnn mavros mavros_msgs sensor_msgs
  DEPENDS system_lib eigen3
)

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${YAML_CPP_INCLUDE_DIR}
)

## Add source files
set(SOURCE_FILES
#   src/Controller_old.cpp
  src/DynamicsCalculator.cpp
  # src/FeatureData.cpp
  src/NMPCController.cpp
  src/NMPCProblem.cpp
  src/VelocityTransformer.cpp
)

## Add header files
set(HEADER_FILES
#   include/vsc_nmpc_uav_target_tracking/Controller_old.hpp
  include/vsc_nmpc_uav_target_tracking/DynamicsCalculator.hpp
  # include/vsc_nmpc_uav_target_tracking/FeatureData.hpp
  include/vsc_nmpc_uav_target_tracking/NMPCController.hpp
  include/vsc_nmpc_uav_target_tracking/NMPCProblem.hpp
  include/vsc_nmpc_uav_target_tracking/VelocityTransformer.hpp
)

## Declare a C++ library
add_library(${PROJECT_NAME}
  ${SOURCE_FILES}
  ${HEADER_FILES}
)

## Specify libraries to link a library or executable target against
target_link_libraries(${PROJECT_NAME}
  ${catkin_LIBRARIES}
  ${YAML_CPP_LIBRARIES}
)

## Add executables
add_executable(track_ibvs_nmpc src/et_ibvs_nmpc/track_ibvs_nmpc.cpp)
target_link_libraries(track_ibvs_nmpc
  ${PROJECT_NAME}
  nlopt
)

add_executable(et_track_ibvs_nmpc src/et_ibvs_nmpc/et_track_ibvs_nmpc.cpp)
target_link_libraries(et_track_ibvs_nmpc
  ${PROJECT_NAME}
  nlopt
)

add_executable(event_triggered_nmpc_img_moments_coast_tracking_OF_tracker src/et_img_moments_nmpc/event_triggered_nmpc_img_moments_coast_tracking_OF_tracker.cpp)
target_link_libraries(event_triggered_nmpc_img_moments_coast_tracking_OF_tracker
  ${PROJECT_NAME}
  nlopt
)

add_executable(event_triggered_nmpc_img_moments_coast_tracking src/et_img_moments_nmpc/event_triggered_nmpc_img_moments_coast_tracking.cpp)
target_link_libraries(event_triggered_nmpc_img_moments_coast_tracking
  ${PROJECT_NAME}
  nlopt
)

add_executable(nmpc_img_moments_coast_tracking_with_cbf src/img_moments_nmpc/nmpc_img_moments_coast_tracking_with_cbf.cpp)
target_link_libraries(nmpc_img_moments_coast_tracking_with_cbf
  ${PROJECT_NAME}
  nlopt
)

add_executable(nmpc_img_moments_coast_tracking src/img_moments_nmpc/nmpc_img_moments_coast_tracking.cpp)
target_link_libraries(nmpc_img_moments_coast_tracking
  ${PROJECT_NAME}
  nlopt
)

add_executable(event_triggered_nmpc_img_moments_coast_tracking_OF_tracker_with_bfs src/et_img_moments_nmpc/event_triggered_nmpc_img_moments_coast_tracking_OF_tracker_with_bfs.cpp)
target_link_libraries(event_triggered_nmpc_img_moments_coast_tracking_OF_tracker_with_bfs
  ${PROJECT_NAME}
  nlopt
)

add_executable(event_triggered_nmpc_img_moments_coast_tracking_with_bfs src/et_img_moments_nmpc/event_triggered_nmpc_img_moments_coast_tracking_with_bfs.cpp)
target_link_libraries(event_triggered_nmpc_img_moments_coast_tracking_with_bfs
  ${PROJECT_NAME}
  nlopt
)

add_executable(ibvs_nmpc_node src/ibvs_nmpc_node.cpp)
target_link_libraries(ibvs_nmpc_node ${PROJECT_NAME} nlopt)

## Specify dependencies for executables
add_dependencies(${PROJECT_NAME}
  ${${PROJECT_NAME}_EXPORTED_TARGETS}
  ${catkin_EXPORTED_TARGETS}
  ${NLOPT_LIBRARIES}
  img_seg_cnn_generate_messages_cpp
  vsc_nmpc_uav_target_tracking_generate_messages_cpp
)
