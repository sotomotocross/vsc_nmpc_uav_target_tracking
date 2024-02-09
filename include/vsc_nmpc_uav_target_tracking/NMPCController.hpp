#ifndef NMPC_CONTROLLER_HPP
#define NMPC_CONTROLLER_HPP

#include <ros/ros.h>
#include <thread>

#include "vsc_nmpc_uav_target_tracking/NMPCProblem.hpp"
#include "vsc_nmpc_uav_target_tracking/DynamicsCalculator.hpp"
#include "vsc_nmpc_uav_target_tracking/VelocityTransformer.hpp"
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

    void altitudeCallback(const std_msgs::Float64::ConstPtr &alt_message);
    void featureCallback(const img_seg_cnn::PredData::ConstPtr &s_message);

private:
    void loadParametersYAML(const std::string& filename);

    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    ros::Subscriber feature_sub_;
    ros::Subscriber alt_sub_;

    ros::Publisher vel_pub_;
    ros::Publisher rec_pub_;

    std::thread control_loop_thread;

    NMPCProblem nmpcProblem_;
    VelocityTransformer velocity_transformer_;

    bool altitude_received;
    bool features_received;

    double gain_tx, gain_ty, gain_tz, gain_yaw;
    int features_n;
    int dim_inputs;
    int dim_s;
    int mpc_hrz;
    double mpc_dt;
    double l;
    double a;
    double optNum;
    int flag;
    double minJ;
    Eigen::VectorXd inputs;

    double umax;
    double umin;
    double vmax;
    double vmin;
    double cu;
    double cv;

    double f0, h0;
    double f1, h1;
    double f2, h2;
    double f3, h3;
    double d01, d03;

    double u0, v0;
    double u1, v1;
    double u2, v2;
    double u3, v3;

    double x0, g0, Z0;
    double x1, g1, Z1;
    double x2, g2, Z2;
    double x3, g3, Z3;

    double u0d, v0d;
    double u1d, v1d;
    double u2d, v2d;
    double u3d, v3d;

    double x0d, g0d;
    double x1d, g1d;
    double x2d, g2d;
    double x3d, g3d;

    double Tx, Ty, Tz, Oz;
};

#endif // NMPC_CONTROLLER_HPP
