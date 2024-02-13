#pragma once

#include <ros/ros.h>
#include "mavros_msgs/PositionTarget.h"
#include "vsc_nmpc_uav_target_tracking/rec.h"

#include "vsc_nmpc_uav_target_tracking/VelocityTransformer.hpp"

class ROSPublisher {
public:
    // ROSPublisher(ros::NodeHandle& nh);
    ROSPublisher(ros::NodeHandle& nh);
    void publishVelocities(const std::vector<double>& inputs);
    void publishRecData(double minJ, double optNum, double x0, double g0, double x1, double g1, double x2, double g2, double x3, double g3, double timer, double dt);

private:
    ros::Publisher vel_pub_;
    ros::Publisher rec_pub_;
    ros::NodeHandle nh_;
    VelocityTransformer velocity_transformer_;

    // Parameters loaded from YAML
    double gain_tx_;
    double gain_ty_;
    double gain_tz_;
    double gain_yaw_;

    // Declaration of the method to load parameters from YAML file
    void loadParametersFromYAML(const std::string& filename);
};