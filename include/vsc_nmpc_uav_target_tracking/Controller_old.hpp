#pragma once

#include <ros/ros.h>
#include "vsc_nmpc_uav_target_tracking/FeatureData.hpp"
#include "vsc_nmpc_uav_target_tracking/VelocityTransformer.hpp"
#include "vsc_nmpc_uav_target_tracking/DynamicsCalculator.hpp"

#include <geometry_msgs/TwistStamped.h>
#include "geometry_msgs/Twist.h"
#include "mavros_msgs/PositionTarget.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"

#include "img_seg_cnn/PredData.h"

#include "vsc_nmpc_uav_target_tracking/rec.h"

#include <thread>
#include <vector>
#include <algorithm>
#include <iostream>
#include <fstream>
#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <eigen3/Eigen/Dense>
#include <nlopt.hpp>

using namespace Eigen;

namespace vsc_nmpc_uav_target_tracking
{
  class DynamicsCalculator; // Forward declaration

  class Controller
  {
  public:
    // Constructor and Destructor
    Controller(ros::NodeHandle &nh, ros::NodeHandle &pnh);
    ~Controller();

    // Callbacks
    void altitudeCallback(const std_msgs::Float64::ConstPtr &alt_message);
    void featureCallback(const img_seg_cnn::PredData::ConstPtr &s_message);

    static double costFunction(unsigned int n, const double *x, double *grad, void *data,
                                  int dim_inputs, int mpc_hrz,
                                  double Z0, double Z1, double Z2, double Z3,
                                  double x0, double x1, double x2, double x3,
                                  double x0d, double x1d, double x2d, double x3d,
                                  double g0, double g1, double g2, double g3,
                                  double g0d, double g1d, double g2d, double g3d,
                                  double l, int dim_s, double mpc_dt, 
                                  Matrix s_des, MatrixXd Q, 
                                  MatrixXd R, MatrixXd P, VectorXd ek);
    static void constraints(unsigned int m, double *c, unsigned int n, const double *x, double *grad, void *data,
                            int dim_inputs, int mpc_hrz,
                            double x0, double x1, double x2, double x3,
                            double g0, double g1, double g2, double g3,
                            double l, int dim_s, double mpc_dt, VectorXd s_abs);

    // Update function
    void update();

  private:
    // ROS NodeHandles
    ros::NodeHandle nh_;
    ros::NodeHandle pnh_;

    // ROS Subscribers
    ros::Subscriber feature_sub_;
    ros::Subscriber alt_sub_;

    // ROS Publishers
    ros::Publisher vel_pub_;
    ros::Publisher rec_pub_;

    // Feature data
    FeatureData feature_data_;

    // Declare a static instance of UtilityFunctions
    static DynamicsCalculator dynamics_calculator;

    // Update loop thread
    std::thread control_loop_thread;

    //****CREATE NLOPT OPTIMIZATION OBJECT, ALGORITHM & TOLERANCES****//
    nlopt_opt opt;

    // Control gains
    double gain_tx, gain_ty, gain_tz, gain_yaw;

    // Global MPC Variables
    int features_n;
    int dim_inputs;
    int dim_s;
    int mpc_hrz;
    double mpc_dt;
    double l;
    double a;
    double optNum;

    MatrixXd s_des;
    VectorXd s_abs;
    MatrixXd Q;
    MatrixXd R;
    MatrixXd P;
    VectorXd t;
    VectorXd ek;

    // Simulator camera parameters
    double umax;
    double umin;
    double vmax;
    double vmin;
    double cu;
    double cv;

    int flag;

    // Camera Frame Update Callback Variables
    double x0, g0, Z0;
    double x1, g1, Z1;
    double x2, g2, Z2;
    double x3, g3, Z3;
    double f0, h0;
    double f1, h1;
    double f2, h2;
    double f3, h3;
    double Tx, Ty, Tz, Oz;
    double d01, d03;

    double u0, v0, u1, v1, u2, v2, u3, v3;

    // Camera Frame Desired Features
    // Simulator (720*480)
    double u0d;
    double v0d;
    double u1d;
    double v1d;
    double u2d;
    double v2d;
    double u3d;
    double v3d;

    double x0d;
    double g0d;
    double x1d;
    double g1d;
    double x2d;
    double g2d;
    double x3d;
    double g3d;

    double sc_x;
    double sc_y;
  };
} // namespace vsc_nmpc_uav_target_tracking
