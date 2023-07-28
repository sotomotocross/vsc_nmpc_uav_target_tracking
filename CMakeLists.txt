cmake_minimum_required(VERSION 3.0.2)
project(mpcpack)

# # Compile as C++11, supported in ROS Kinetic and newer
# add_compile_options(-std=c++11)

# # Find catkin macros and libraries
# # if COMPONENTS list like find_package(catkin REQUIRED COMPONENTS xyz)
# # is used, also find other catkin packages
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
)

# # System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)

# # Uncomment this if the package has a setup.py. This macro ensures
# # modules and global scripts declared therein get installed
# # See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

# ###############################################
# # Declare ROS messages, services and actions ##
# ###############################################

# # To declare and build messages, services or actions from within this
# # package, follow these steps:
# # * Let MSG_DEP_SET be the set of packages whose message types you use in
# #   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
# # * In the file package.xml:
# #   * add a build_depend tag for "message_generation"
# #   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
# #   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
# #     but can be declared for certainty nonetheless:
# #     * add a exec_depend tag for "message_runtime"
# # * In this file (CMakeLists.txt):
# #   * add "message_generation" and every package in MSG_DEP_SET to
# #     find_package(catkin REQUIRED COMPONENTS ...)
# #   * add "message_runtime" and every package in MSG_DEP_SET to
# #     catkin_package(CATKIN_DEPENDS ...)
# #   * uncomment the add_*_files sections below as needed
# #     and list every .msg/.srv/.action file to be processed
# #   * uncomment the generate_messages entry below
# #   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

# # Generate messages in the 'msg' folder
add_message_files(
      FILES

      # Twist.msg
      rec.msg
)

# # Generate services in the 'srv' folder
# add_service_files(
# FILES
# Service1.srv
# Service2.srv
# )

# # Generate actions in the 'action' folder
# add_action_files(
# FILES
# Action1.action
# Action2.action
# )

# # Generate added messages and services with any dependencies listed here
generate_messages(
      DEPENDENCIES
      geometry_msgs
      std_msgs
)

# ###############################################
# # Declare ROS dynamic reconfigure parameters ##
# ###############################################

# # To declare and build dynamic reconfigure parameters within this
# # package, follow these steps:
# # * In the file package.xml:
# #   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
# # * In this file (CMakeLists.txt):
# #   * add "dynamic_reconfigure" to
# #     find_package(catkin REQUIRED COMPONENTS ...)
# #   * uncomment the "generate_dynamic_reconfigure_options" section below
# #     and list every .cfg file to be processed

# # Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
# cfg/DynReconf1.cfg
# cfg/DynReconf2.cfg
# )

# ##################################
# # catkin specific configuration ##
# ##################################
# # The catkin_package macro generates cmake config files for your package
# # Declare things to be passed to dependent projects
# # INCLUDE_DIRS: uncomment this if your package contains header files
# # LIBRARIES: libraries you create in this project that dependent projects also need
# # CATKIN_DEPENDS: catkin_packages dependent projects also need
# # DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
      INCLUDE_DIRS include
      LIBRARIES mpcpack
      CATKIN_DEPENDS message_runtime geometry_msgs message_generation roscpp rospy std_msgs img_seg_cnn mavros mavros_msgs sensor_msgs
      DEPENDS system_lib eigen3
)

# ##########
# # Build ##
# ##########

# # Specify additional locations of header files
# # Your package locations should be listed before other locations
include_directories(include ${catkin_INCLUDE_DIRS})

# # Declare a C++ library
# add_library(${PROJECT_NAME}
# src/${PROJECT_NAME}/mpcpack.cpp
# )

# # Add cmake target dependencies of the library
# # as an example, code may need to be generated before libraries
# # either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# # Declare a C++ executable
# # With catkin_make all packages are built within a single CMake context
# # The recommended prefix ensures that target names across packages don't collide
# add_executable(${PROJECT_NAME}_node src/mpcpack_node.cpp)

# # Rename C++ executable without prefix
# # The above recommended prefix causes long target names, the following renames the
# # target back to the shorter version for ease of user use
# # e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

# # Add cmake target dependencies of the executable
# # same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

# # Specify libraries to link a library or executable target against
# target_link_libraries(${PROJECT_NAME}_node
# ${catkin_LIBRARIES}
# )

# ############
# # Install ##
# ############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

# # Mark executable scripts (Python etc.) for installation
# # in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
# scripts/my_python_script
# DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# # Mark executables for installation
# # See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
# install(TARGETS ${PROJECT_NAME}_node
# RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

# # Mark libraries for installation
# # See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
# ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
# LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
# RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )

# # Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
# DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
# FILES_MATCHING PATTERN "*.h"
# PATTERN ".svn" EXCLUDE
# )

# # Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
# # myfile1
# # myfile2
# DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

# ############
# # Testing ##
# ############

# # Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_mpcpack.cpp)
# if(TARGET ${PROJECT_NAME}-test)
# target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

# # Add folders to be run by python nosetests
# catkin_add_nosetests(test)
add_executable(stab_ibvs_nmpc src/stab_ibvs_nmpc.cpp)
target_link_libraries(stab_ibvs_nmpc nlopt ${catkin_LIBRARIES})
add_dependencies(stab_ibvs_nmpc
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(track_ibvs_nmpc src/track_ibvs_nmpc.cpp)
target_link_libraries(track_ibvs_nmpc nlopt ${catkin_LIBRARIES})
add_dependencies(track_ibvs_nmpc
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(et_stab_ibvs_nmpc src/et_stab_ibvs_nmpc.cpp)
target_link_libraries(et_stab_ibvs_nmpc nlopt ${catkin_LIBRARIES})
add_dependencies(et_stab_ibvs_nmpc
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(et_track_ibvs_nmpc src/et_track_ibvs_nmpc.cpp)
target_link_libraries(et_track_ibvs_nmpc nlopt ${catkin_LIBRARIES})
add_dependencies(et_track_ibvs_nmpc
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(et_tracking_pvs_barycenter_nmpc src/et_tracking_pvs_barycenter_nmpc.cpp)
target_link_libraries(et_tracking_pvs_barycenter_nmpc nlopt ${catkin_LIBRARIES})
add_dependencies(et_tracking_pvs_barycenter_nmpc
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(tracking_pvs_barycenter_nmpc src/tracking_pvs_barycenter_nmpc.cpp)
target_link_libraries(tracking_pvs_barycenter_nmpc nlopt ${catkin_LIBRARIES})
add_dependencies(tracking_pvs_barycenter_nmpc
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(tracking_pvs_barycenter_nmpc_4_state_vars src/tracking_pvs_barycenter_nmpc_4_state_vars.cpp)
target_link_libraries(tracking_pvs_barycenter_nmpc_4_state_vars nlopt ${catkin_LIBRARIES})
add_dependencies(tracking_pvs_barycenter_nmpc_4_state_vars
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(pvs_nmpc_barycenter_6_dof_cam src/pvs_nmpc_barycenter_6_dof_cam.cpp)
target_link_libraries(pvs_nmpc_barycenter_6_dof_cam nlopt ${catkin_LIBRARIES})
add_dependencies(pvs_nmpc_barycenter_6_dof_cam
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(pvs_nmpc_barycenter_4_dof_cam_4_state_vars src/pvs_nmpc_barycenter_4_dof_cam_4_state_vars.cpp)
target_link_libraries(pvs_nmpc_barycenter_4_dof_cam_4_state_vars nlopt ${catkin_LIBRARIES})
add_dependencies(pvs_nmpc_barycenter_4_dof_cam_4_state_vars
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(pvs_nmpc_barycenter_4_dof_full_res_cam_4_state_vars src/pvs_nmpc_barycenter_4_dof_full_res_cam_4_state_vars.cpp)
target_link_libraries(pvs_nmpc_barycenter_4_dof_full_res_cam_4_state_vars nlopt ${catkin_LIBRARIES})
add_dependencies(pvs_nmpc_barycenter_4_dof_full_res_cam_4_state_vars
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(pvs_nmpc_barycenter_6_dof_cam_4_state_vars src/pvs_nmpc_barycenter_6_dof_cam_4_state_vars.cpp)
target_link_libraries(pvs_nmpc_barycenter_6_dof_cam_4_state_vars nlopt ${catkin_LIBRARIES})
add_dependencies(pvs_nmpc_barycenter_6_dof_cam_4_state_vars
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(pvs_nmpc_baryc_6_dof_6_state_vars src/pvs_nmpc_baryc_6_dof_6_state_vars.cpp)
target_link_libraries(pvs_nmpc_baryc_6_dof_6_state_vars nlopt ${catkin_LIBRARIES})
add_dependencies(pvs_nmpc_baryc_6_dof_6_state_vars
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(img_moments_ibvs_chaumette src/img_moments_ibvs_chaumette.cpp)
target_link_libraries(img_moments_ibvs_chaumette nlopt ${catkin_LIBRARIES})
add_dependencies(img_moments_ibvs_chaumette
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(uav_tracking_pvs_nmpc_with_img_moments_centroid src/uav_tracking_pvs_nmpc_with_img_moments_centroid.cpp)
target_link_libraries(uav_tracking_pvs_nmpc_with_img_moments_centroid nlopt ${catkin_LIBRARIES})
add_dependencies(uav_tracking_pvs_nmpc_with_img_moments_centroid
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(ibvs_img_moments_chaumette_coast_tracking src/uav_simulator/ibvs_img_moments_chaumette_coast_tracking.cpp)
target_link_libraries(ibvs_img_moments_chaumette_coast_tracking nlopt ${catkin_LIBRARIES})
add_dependencies(ibvs_img_moments_chaumette_coast_tracking
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(nmpc_img_moments_coast_tracking_with_cbf src/uav_simulator/nmpc_img_moments_coast_tracking_with_cbf.cpp)
target_link_libraries(nmpc_img_moments_coast_tracking_with_cbf nlopt ${catkin_LIBRARIES})
add_dependencies(nmpc_img_moments_coast_tracking_with_cbf
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(nmpc_img_moments_coast_tracking src/uav_simulator/nmpc_img_moments_coast_tracking.cpp)
target_link_libraries(nmpc_img_moments_coast_tracking nlopt ${catkin_LIBRARIES})
add_dependencies(nmpc_img_moments_coast_tracking
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(ibvs_img_moments_chaumette_target_tracking src/6_dof_cam_system/ibvs_img_moments_chaumette_target_tracking.cpp)
target_link_libraries(ibvs_img_moments_chaumette_target_tracking nlopt ${catkin_LIBRARIES})
add_dependencies(ibvs_img_moments_chaumette_target_tracking
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(nmpc_img_moments_target_tracking_with_cbf src/6_dof_cam_system/nmpc_img_moments_target_tracking_with_cbf.cpp)
target_link_libraries(nmpc_img_moments_target_tracking_with_cbf nlopt ${catkin_LIBRARIES})
add_dependencies(nmpc_img_moments_target_tracking_with_cbf
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(nmpc_img_moments_target_tracking src/6_dof_cam_system/nmpc_img_moments_target_tracking.cpp)
target_link_libraries(nmpc_img_moments_target_tracking nlopt ${catkin_LIBRARIES})
add_dependencies(nmpc_img_moments_target_tracking
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(ibvs_nmpc_testing_of_convergence src/ibvs_nmpc_testing_of_convergence.cpp)
target_link_libraries(ibvs_nmpc_testing_of_convergence nlopt ${catkin_LIBRARIES})
add_dependencies(ibvs_nmpc_testing_of_convergence
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(event_triggered_nmpc_img_moments_coast_tracking src/uav_simulator/event_triggered_nmpc_img_moments_coast_tracking.cpp)
target_link_libraries(event_triggered_nmpc_img_moments_coast_tracking nlopt ${catkin_LIBRARIES})
add_dependencies(event_triggered_nmpc_img_moments_coast_tracking
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

add_executable(event_triggered_nmpc_img_moments_coast_tracking_OF_tracker src/uav_simulator/event_triggered_nmpc_img_moments_coast_tracking_OF_tracker.cpp)
target_link_libraries(event_triggered_nmpc_img_moments_coast_tracking_OF_tracker nlopt ${catkin_LIBRARIES})
add_dependencies(event_triggered_nmpc_img_moments_coast_tracking_OF_tracker
      ${${PROJECT_NAME}_EXPORTED_TARGETS}
      ${catkin_EXPORTED_TARGETS}
      ${NLOPT_LIBRARIES}
      img_seg_cnn_generate_messages_cpp
      mpcpack_generate_messages_cpp)

      add_executable(event_triggered_nmpc_img_moments_coast_tracking_OF_tracker_param_version src/uav_simulator/event_triggered_nmpc_img_moments_coast_tracking_OF_tracker_param_version.cpp)
      target_link_libraries(event_triggered_nmpc_img_moments_coast_tracking_OF_tracker_param_version nlopt ${catkin_LIBRARIES})
      add_dependencies(event_triggered_nmpc_img_moments_coast_tracking_OF_tracker_param_version
            ${${PROJECT_NAME}_EXPORTED_TARGETS}
            ${catkin_EXPORTED_TARGETS}
            ${NLOPT_LIBRARIES}
            img_seg_cnn_generate_messages_cpp
            mpcpack_generate_messages_cpp)