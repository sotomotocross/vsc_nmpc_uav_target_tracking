#include <ros/ros.h>

#include "vsc_nmpc_uav_target_tracking/ROSPublisher.hpp"
#include "vsc_nmpc_uav_target_tracking/VelocityTransformer.hpp"

#include "mavros_msgs/PositionTarget.h"
#include "vsc_nmpc_uav_target_tracking/rec.h"

#include <yaml-cpp/yaml.h>

ROSPublisher::ROSPublisher(ros::NodeHandle& nh) : nh_(nh) {

    // std::string yaml_filename;
    // if (!nh.getParam("yaml_filename", yaml_filename)) {
    //     ROS_ERROR("Failed to retrieve yaml_filename parameter");
    //     // Handle error
    //     return;
    // }
    // loadParametersFromYAML(yaml_filename);

    vel_pub_ = nh_.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);
    rec_pub_ = nh_.advertise<vsc_nmpc_uav_target_tracking::rec>("/vsc_nmpc_uav_target_tracking/msg/rec", 1);    
}

// void ROSPublisher::loadParametersFromYAML(const std::string& filename) {
//     YAML::Node config = YAML::LoadFile(filename);

//     // Check if the YAML node exists and load parameters
//     if (config["vsc_nmpc_uav_target_tracking"])
//     {
//         nh_.param<double>("vsc_nmpc_uav_target_tracking/gain_tx", gain_tx_, config["vsc_nmpc_uav_target_tracking"]["gain_tx"].as<double>());
//         nh_.param<double>("vsc_nmpc_uav_target_tracking/gain_ty", gain_ty_, config["vsc_nmpc_uav_target_tracking"]["gain_ty"].as<double>());
//         nh_.param<double>("vsc_nmpc_uav_target_tracking/gain_tz", gain_tz_, config["vsc_nmpc_uav_target_tracking"]["gain_tz"].as<double>());
//         nh_.param<double>("vsc_nmpc_uav_target_tracking/gain_yaw", gain_yaw_, config["vsc_nmpc_uav_target_tracking"]["gain_yaw"].as<double>());
//     } else {
//         ROS_WARN("Failed to load parameters from YAML file. Using default values.");
//         // Set default values
//         gain_tx_ = 1.0;
//         gain_ty_ = 1.0;
//         gain_tz_ = 0.0;
//         gain_yaw_ = 1.5;
//         std::cout << "gain_tx_ = " << gain_tx_ << std::endl;
//         std::cout << "gain_ty_ = " << gain_ty_ << std::endl;
//         std::cout << "gain_tz_ = " << gain_tz_ << std::endl;
//         std::cout << "gain_yaw_ = " << gain_yaw_ << std::endl;
//     }
// }

void ROSPublisher::publishVelocities(const std::vector<double>& inputs) {
    
    mavros_msgs::PositionTarget dataMsg;
    Eigen::Matrix<double, 4, 1> caminputs;
    caminputs(0, 0) = inputs[0];
    caminputs(1, 0) = inputs[1];
    caminputs(2, 0) = inputs[2];
    caminputs(3, 0) = inputs[3];

    dataMsg.coordinate_frame = 8;
    dataMsg.type_mask = 1479;
    dataMsg.header.stamp = ros::Time::now();
    double Tx = velocity_transformer_.VelTrans1(velocity_transformer_.VelTrans(caminputs))(0, 0);
    double Ty = velocity_transformer_.VelTrans1(velocity_transformer_.VelTrans(caminputs))(1, 0);
    double Tz = velocity_transformer_.VelTrans1(velocity_transformer_.VelTrans(caminputs))(2, 0);
    double Oz = velocity_transformer_.VelTrans1(velocity_transformer_.VelTrans(caminputs))(5, 0);

    if (Tx >= 0.5) {
        Tx = 0.3;
    }
    if (Tx <= -0.5) {
        Tx = -0.3;
    }
    if (Ty >= 0.5) {
        Ty = 0.4;
    }
    if (Ty <= -0.4) {
        Ty = -0.4;
    }
    if (Oz >= 0.3) {
        Oz = 0.2;
    }
    if (Oz <= -0.3) {
        Oz = -0.2;
    }

    double gain_tx_ = 10.0;
    double gain_ty_ = 5.0;
    double gain_tz_ = 0.0;
    double gain_yaw_ = 2.5;

    dataMsg.velocity.x = gain_tx_ * Tx;
    dataMsg.velocity.y = gain_ty_ * Ty;
    dataMsg.velocity.z = gain_tz_ * Tz;
    dataMsg.yaw_rate = gain_yaw_ * Oz;

    printf("\n\nDrone Velocities Tx,Ty,Tz,Oz(%g,%g,%g,%g) = ", Tx, Ty, Tz, Oz);

    vel_pub_.publish(dataMsg);
}

void ROSPublisher::publishRecData(double minJ, double optNum, double x0, double g0, double x1, double g1, double x2, double g2, double x3, double g3, double timer, double dt) {
    
    vsc_nmpc_uav_target_tracking::rec fdataMsg;

    fdataMsg.J = minJ;
    fdataMsg.optNUM = optNum;

    fdataMsg.u1 = x0;
    fdataMsg.v1 = g0;
    fdataMsg.u2 = x1;
    fdataMsg.v2 = g1;
    fdataMsg.u3 = x2;
    fdataMsg.v3 = g2;
    fdataMsg.u4 = x3;
    fdataMsg.v4 = g3;

    fdataMsg.time = timer;
    fdataMsg.dtloop = dt;

    rec_pub_.publish(fdataMsg);
}