#include <ros/ros.h>
#include "vsc_nmpc_uav_target_tracking/NMPCController.hpp"

int main(int argc, char** argv) {
    // Initialize ROS node
    ros::init(argc, argv, "vsc_nmpc_uav_target_tracking");

    // Create ROS node handles for the main node and private node
    ros::NodeHandle nh;
    ros::NodeHandle pnh("~");

    // Print a message indicating the start of the node
    ROS_INFO("Starting IBVS-NMPC node");

    // Create NMPC controller object
    NMPCController controller(nh, pnh);

    // Enter the ROS spin loop
    ros::spin();

    return 0;
}
