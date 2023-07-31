cmake_minimum_required(VERSION 2.8.3)
project(rrt_planner)

## Compile as C++11, supported in ROS Kinetic and newer
add_compile_options(-std=c++11)

find_package(catkin REQUIRED COMPONENTS
  geometry_msgs
  nav_msgs
  roscpp

      message_generation
      rospy
      std_msgs
      img_seg_cnn
      mavros
      mavros_msgs
      sensor_msgs
)

add_message_files(
      FILES
      rec.msg
)

find_package(Boost 1.65 REQUIRED)

find_package(OpenCV REQUIRED)

include_directories(${OpenCV_INCLUDE_DIRS})


generate_messages(
      DEPENDENCIES
      geometry_msgs
      std_msgs
)


catkin_package(
  INCLUDE_DIRS include
  LIBRARIES rrt_planner
  CATKIN_DEPENDS
    nav_msgs
    geometry_msgs
    roscpp
    message_runtime 
    message_generation 
    rospy std_msgs 
    img_seg_cnn 
    mavros 
    mavros_msgs 
    sensor_msgs
    DEPENDS system_lib eigen3
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME}
  src/rrt_planner.cpp
)

add_executable(${PROJECT_NAME}_node src/rrt_planner_node.cpp)

target_link_libraries(${PROJECT_NAME}
  nlopt
  ${catkin_LIBRARIES}
  ${OpenCV_LIBS}
)

target_link_libraries(${PROJECT_NAME}_node
  nlopt
  ${catkin_LIBRARIES}
  ${PROJECT_NAME}
)

install(TARGETS ${PROJECT_NAME}_node
  RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

add_dependencies(${PROJECT_NAME}
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      rrt_planner_generate_messages_cpp)

add_dependencies(${PROJECT_NAME}_node
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      rrt_planner_generate_messages_cpp)