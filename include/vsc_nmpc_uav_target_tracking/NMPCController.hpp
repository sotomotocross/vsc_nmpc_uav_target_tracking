#ifndef NMPC_CONTROLLER_HPP
#define NMPC_CONTROLLER_HPP

#include <ros/ros.h>
#include <thread>

#include "vsc_nmpc_uav_target_tracking/NMPCProblem.hpp"
#include "vsc_nmpc_uav_target_tracking/DynamicsCalculator.hpp"
#include "vsc_nmpc_uav_target_tracking/VelocityTransformer.hpp"
#include "vsc_nmpc_uav_target_tracking/ROSListener.hpp" 
#include "vsc_nmpc_uav_target_tracking/ROSPublisher.hpp"

#include "img_seg_cnn/PredData.h"
#include "std_msgs/Float64.h"
#include <eigen3/Eigen/Dense>
#include <nlopt.hpp>

class NMPCController
{
public:
    NMPCController(ros::NodeHandle &nh, ros::NodeHandle &pnh);
    ~NMPCController();

    void solve();

private:

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    NMPCProblem nmpcProblem_;
    VelocityTransformer velocity_transformer_;
    std::shared_ptr<ROSListener> ros_listener_; 
    ROSPublisher publisher_; 

    int features_n;
    int dim_inputs;
    int dim_s;
    int mpc_hrz;

    double optNum;

    double minJ;
    Eigen::VectorXd inputs;

    double x0, g0, Z0;
    double x1, g1, Z1;
    double x2, g2, Z2;
    double x3, g3, Z3;

    double Tx, Ty, Tz, Oz;
};

#endif // NMPC_CONTROLLER_HPP
