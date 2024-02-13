#include "ros/ros.h"
#include <ros/package.h>

#include "vsc_nmpc_uav_target_tracking/NMPCController.hpp"
#include "vsc_nmpc_uav_target_tracking/NMPCProblem.hpp"
#include "vsc_nmpc_uav_target_tracking/DynamicsCalculator.hpp"
#include "vsc_nmpc_uav_target_tracking/VelocityTransformer.hpp"

#include "vsc_nmpc_uav_target_tracking/rec.h"
#include "img_seg_cnn/PredData.h"
#include "mavros_msgs/PositionTarget.h"
#include "std_msgs/Float64.h"

#include <cstdlib>
#include <iostream>
#include <thread>
#include <stdio.h>
#include <math.h>
#include <nlopt.hpp>
#include <eigen3/Eigen/Dense>

#include <yaml-cpp/yaml.h>

using namespace std;
using namespace Eigen;

// Constructor
NMPCController::NMPCController(ros::NodeHandle &nh, ros::NodeHandle &pnh)
    : nh_(nh), pnh_(pnh)
{
    //****INITIALIZE INPUT VECTOR****//
    // Define dimensions and other parameters
    dim_inputs = 4;
    features_n = 4;
    dim_s = 2 * features_n;
    mpc_hrz = 6;
    mpc_dt = 0.1;
    l = 252.07;
    a = 10;
    minJ = -1.0; //****DEFINE COST FUNCTION VARIABLE****//
    flag = 0;

    // Simulator camera parameters
    umax = 720;
    umin = 0;
    vmax = 480;
    vmin = 0;
    cu = 0.5 * (umax + umin);
    cv = 0.5 * (vmax + vmin);

    // Initialize inputs array
    inputs.resize(dim_inputs * mpc_hrz);
    inputs.setZero(); // Initialize all elements to zero

    // Retrieve filename parameter from ROS parameter server
    std::string filename;
    if (!pnh_.getParam("yaml_filename", filename))
    {
        ROS_ERROR("Failed to retrieve yaml_filename parameter");
        // Handle error
        return;
    }

    // Load parameters from YAML file
    loadParametersYAML(filename);

    // Set up ROS subscribers
    feature_sub_ = nh_.subscribe<img_seg_cnn::PredData>("/pred_data", 10, &NMPCController::featureCallback, this);
    alt_sub_ = nh_.subscribe<std_msgs::Float64>("/mavros/global_position/rel_alt", 10, &NMPCController::altitudeCallback, this);

    // Set up ROS publishers
    vel_pub_ = nh_.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);
    rec_pub_ = nh_.advertise<vsc_nmpc_uav_target_tracking::rec>("/vsc_nmpc_uav_target_tracking/msg/rec", 1);

    // Initialize NMPC problem
    nmpcProblem_.setup();

    // Create a thread for the control loop
    control_loop_thread = std::thread([this]()
                                      {
                                          ros::Rate rate(35); // Adjust the rate as needed
                                          while (ros::ok())
                                          {
                                              if (Z0 != 0.0)
                                              {
                                                  NMPCController::solve();
                                              }
                                              rate.sleep();
                                          }
                                      });
}

// Destructor
NMPCController::~NMPCController()
{
    // Shutdown ROS publishers...
    vel_pub_.shutdown();
}

// Load parameters from YAML file
void NMPCController::loadParametersYAML(const std::string &filename)
{
    // Load parameters from YAML file
    YAML::Node config = YAML::LoadFile(filename);

    // Check if the YAML node exists and load parameters
    if (config["vsc_nmpc_uav_target_tracking"])
    {
        nh_.param<double>("vsc_nmpc_uav_target_tracking/gain_tx", gain_tx, config["vsc_nmpc_uav_target_tracking"]["gain_tx"].as<double>());
        nh_.param<double>("vsc_nmpc_uav_target_tracking/gain_ty", gain_ty, config["vsc_nmpc_uav_target_tracking"]["gain_ty"].as<double>());
        nh_.param<double>("vsc_nmpc_uav_target_tracking/gain_tz", gain_tz, config["vsc_nmpc_uav_target_tracking"]["gain_tz"].as<double>());
        nh_.param<double>("vsc_nmpc_uav_target_tracking/gain_yaw", gain_yaw, config["vsc_nmpc_uav_target_tracking"]["gain_yaw"].as<double>());
    }
    else
    {
        ROS_WARN("Failed to load parameters from YAML file. Using default values.");
    }
}

// Callback for altitude data
void NMPCController::altitudeCallback(const std_msgs::Float64::ConstPtr &alt_message)
{
    // Handle altitude data...
    Z0 = alt_message->data;
    Z1 = alt_message->data;
    Z2 = alt_message->data;
    Z3 = alt_message->data;
    flag = 1;
    // Set altitude flag to true
    altitude_received = true;
}

// Callback for image feature data
void NMPCController::featureCallback(const img_seg_cnn::PredData::ConstPtr &s_message)
{
    // Process feature data...
    f0 = s_message->box_1[0];
    h0 = s_message->box_1[1];

    f1 = s_message->box_2[0];
    h1 = s_message->box_2[1];

    f2 = s_message->box_3[0];
    h2 = s_message->box_3[1];

    f3 = s_message->box_4[0];
    h3 = s_message->box_4[1];

    if (f0 > umax)
    {
        f0 = umax;
    }
    if (f1 > umax)
    {
        f1 = umax;
    }
    if (f2 > umax)
    {
        f2 = umax;
    }
    if (f3 > umax)
    {
        f3 = umax;
    }

    if (f0 < umin)
    {
        f0 = umin;
    }
    if (f1 < umin)
    {
        f1 = umin;
    }
    if (f2 < umin)
    {
        f2 = umin;
    }
    if (f3 < umin)
    {
        f3 = umin;
    }

    if (h0 > vmax)
    {
        h0 = vmax;
    }
    if (h1 > vmax)
    {
        h1 = vmax;
    }
    if (h2 > vmax)
    {
        h2 = vmax;
    }
    if (h3 > vmax)
    {
        h3 = vmax;
    }

    if (h0 < vmin)
    {
        h0 = vmin;
    }
    if (h1 < vmin)
    {
        h1 = vmin;
    }
    if (h2 < vmin)
    {
        h2 = vmin;
    }
    if (h3 < vmin)
    {
        h3 = vmin;
    }

    d01 = sqrt((f0 - f1) * (f0 - f1) + (h0 - h1) * (h0 - h1));
    d03 = sqrt((f0 - f3) * (f0 - f3) + (h0 - h3) * (h0 - h3));

    if (d01 > d03)
    {
        u0 = f0;
        v0 = h0;

        u1 = f1;
        v1 = h1;

        u2 = f2;
        v2 = h2;

        u3 = f3;
        v3 = h3;
    }

    if (d01 < d03)
    {

        u0 = f1;
        v0 = h1;

        u1 = f2;
        v1 = h2;

        u2 = f3;
        v2 = h3;

        u3 = f0;
        v3 = h0;
    }

    x0 = (u0 - cu) / l;
    g0 = (v0 - cv) / l;
    x1 = (u1 - cu) / l;
    g1 = (v1 - cv) / l;
    x2 = (u2 - cu) / l;
    g2 = (v2 - cv) / l;
    x3 = (u3 - cu) / l;
    g3 = (v3 - cv) / l;

    flag = 1;
    // cout << "Feature callback flag: " << flag << endl;

    // Set features flag to true
    features_received = true;
}


void NMPCController::solve()
{
    cout << "\n"
         << endl;
    //****INITIALIZE TIME VARIABLES****//
    double t0 = ros::WallTime::now().toSec();
    double realtime = 0;

    //****RUNNING LOOP****//
    while (ros::ok())
    {
        // Check if all required values are received
        if (x0 != 0 && g0 != 0 && Z0 != 0 && x1 != 0 && g1 != 0 && Z1 != 0 &&
            x2 != 0 && g2 != 0 && Z2 != 0 && x3 != 0 && g3 != 0 && Z3 != 0)
        {
            double start = ros::Time::now().toSec();

            // cout << "(x0,g0,Z0): (" << x0 << "," << g0 << "," << Z0 << ")" << endl;
            // cout << "(x1,g1,Z1): (" << x1 << "," << g1 << "," << Z1 << ")" << endl;
            // cout << "(x2,g2,Z2): (" << x2 << "," << g2 << "," << Z2 << ")" << endl;
            // cout << "(x3,g3,Z3): (" << x3 << "," << g3 << "," << Z3 << ")" << endl;

            if (flag)
            {
                // cout << "Going to the solve of NMPC???" << endl;

                // Pass values to NMPCProblem
                nmpcProblem_.setValues(x0, g0, Z0, x1, g1, Z1, x2, g2, Z2, x3, g3, Z3);

                // Call NMPCProblem::solve() and store the return values
                auto result = nmpcProblem_.solve(inputs.data(), minJ);
                double optNum = result.first; // Extract optNum from the result
                std::vector<double> inputs = result.second;
                printf("minJ - found minimum at J(%g,%g,%g,%g) = %g\n", inputs[0], inputs[1], inputs[2], inputs[3], minJ);
                printf("optNum - found minimum at J(%g,%g,%g,%g) = %g\n", inputs[0], inputs[1], inputs[2], inputs[3], optNum);
            }

            printf("found minimum at J(%g,%g,%g,%g) = %g\n", inputs[0], inputs[1], inputs[2], inputs[3], optNum);

            double end = ros::Time::now().toSec();
            double tf = ros::WallTime::now().toSec();
            double timer = tf - t0;
            double dt = end - start;
            realtime = realtime + dt;

            //****SEND VELOCITIES TO AUTOPILOT THROUGH MAVROS****//
            mavros_msgs::PositionTarget dataMsg;

            Matrix<double, 4, 1> caminputs;
            caminputs(0, 0) = inputs[0];
            caminputs(1, 0) = inputs[1];
            caminputs(2, 0) = inputs[2];
            caminputs(3, 0) = inputs[3];

            dataMsg.coordinate_frame = 8;
            dataMsg.type_mask = 1479;
            dataMsg.header.stamp = ros::Time::now();
            Tx = velocity_transformer_.VelTrans1(velocity_transformer_.VelTrans(caminputs))(0, 0);
            Ty = velocity_transformer_.VelTrans1(velocity_transformer_.VelTrans(caminputs))(1, 0);
            Tz = velocity_transformer_.VelTrans1(velocity_transformer_.VelTrans(caminputs))(2, 0);
            Oz = velocity_transformer_.VelTrans1(velocity_transformer_.VelTrans(caminputs))(5, 0);

            dataMsg.velocity.x = gain_tx * Tx;
            dataMsg.velocity.y = gain_ty * Ty;
            dataMsg.velocity.z = gain_tz * Tz;
            dataMsg.yaw_rate = gain_yaw * Oz;

            if (Tx >= 0.5)
            {
                Tx = 0.3;
            }
            if (Tx <= -0.5)
            {
                Tx = -0.3;
            }
            if (Ty >= 0.5)
            {
                Ty = 0.4;
            }
            if (Ty <= -0.4)
            {
                Ty = -0.4;
            }
            if (Oz >= 0.3)
            {
                Oz = 0.2;
            }
            if (Oz <= -0.3)
            {
                Oz = -0.2;
            }

            printf("Drone Velocities Tx,Ty,Tz,Oz(%g,%g,%g,%g) = ", Tx, Ty, Tz, Oz);

            //****SAVE DATA****//
            vsc_nmpc_uav_target_tracking::rec fdataMsg;

            fdataMsg.J = minJ;
            fdataMsg.optNUM = optNum;
            fdataMsg.Z = Z0;

            fdataMsg.u1d = u0d;
            fdataMsg.v1d = v0d;
            fdataMsg.u2d = u1d;
            fdataMsg.v2d = v1d;
            fdataMsg.u3d = u2d;
            fdataMsg.v3d = v2d;
            fdataMsg.u4d = u3d;
            fdataMsg.v4d = v3d;

            fdataMsg.u1 = u0;
            fdataMsg.v1 = v0;
            fdataMsg.u2 = u1;
            fdataMsg.v2 = v1;
            fdataMsg.u3 = u2;
            fdataMsg.v3 = v2;
            fdataMsg.u4 = u3;
            fdataMsg.v4 = v3;

            fdataMsg.Eu1 = u0 - u0d;
            fdataMsg.Ev1 = v0 - v0d;
            fdataMsg.Eu2 = u1 - u1d;
            fdataMsg.Ev2 = v1 - v1d;
            fdataMsg.Eu3 = u2 - u2d;
            fdataMsg.Ev3 = v2 - v2d;
            fdataMsg.Eu4 = u3 - u3d;
            fdataMsg.Ev4 = v3 - v3d;

            fdataMsg.Tx = Tx;
            fdataMsg.Ty = Ty;
            fdataMsg.Tz = Tz;
            fdataMsg.Oz = Oz;

            fdataMsg.time = timer;
            fdataMsg.dtloop = dt;

            rec_pub_.publish(fdataMsg);
            // vel_pub_.publish(dataMsg);
        }
        else
        {
            ROS_ERROR("Optimization failed!");
            //...Handle failure case...
        }
        //     //     //****REMEMBER TO FREE ALLOCATED MEMORY AND DESTROY THE OPTIMIZATION OBJECT****//
        //     // nlopt_destroy(opt);
    }
}