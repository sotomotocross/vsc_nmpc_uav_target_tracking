#include "ros/ros.h"
#include <ros/package.h>

#include "vsc_nmpc_uav_target_tracking/NMPCController.hpp"
#include "vsc_nmpc_uav_target_tracking/NMPCProblem.hpp"
#include "vsc_nmpc_uav_target_tracking/DynamicsCalculator.hpp"
#include "vsc_nmpc_uav_target_tracking/VelocityTransformer.hpp"
#include "vsc_nmpc_uav_target_tracking/ROSListener.hpp"
#include "vsc_nmpc_uav_target_tracking/ROSPublisher.hpp"

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
    : nh_(nh), pnh_(pnh), publisher_(nh)
{
    //****INITIALIZE INPUT VECTOR****//
    // Define dimensions and other parameters
    dim_inputs = 4;
    features_n = 4;
    dim_s = 2 * features_n;
    mpc_hrz = 6;
    minJ = -1.0; //****DEFINE COST FUNCTION VARIABLE****//

    // Initialize inputs array
    inputs.resize(dim_inputs * mpc_hrz);
    inputs.setZero(); // Initialize all elements to zero

    // Initialize ROSListener
    ros_listener_ = std::make_shared<ROSListener>(nh_); // Construct ROSListener with nh_ only

    // Initialize NMPC problem
    nmpcProblem_.setup();
}

// Destructor
NMPCController::~NMPCController()
{

}

void NMPCController::solve()
{
    double t0 = ros::WallTime::now().toSec();
    double realtime = 0;

    // Access data from ROSListener using its getter methods
    x0 = ros_listener_->getX0();
    x1 = ros_listener_->getX1();
    x2 = ros_listener_->getX2();
    x3 = ros_listener_->getX3();

    g0 = ros_listener_->getG0();
    g1 = ros_listener_->getG1();
    g2 = ros_listener_->getG2();
    g3 = ros_listener_->getG3();

    Z0 = ros_listener_->getZ0();
    Z1 = ros_listener_->getZ1();
    Z2 = ros_listener_->getZ2();
    Z3 = ros_listener_->getZ3();

    // Check if all required values are received
    if (x0 != 0 && g0 != 0 && Z0 != 0 && x1 != 0 && g1 != 0 && Z1 != 0 &&
        x2 != 0 && g2 != 0 && Z2 != 0 && x3 != 0 && g3 != 0 && Z3 != 0)
    {
        double start = ros::Time::now().toSec();

        // Pass values to NMPCProblem
        nmpcProblem_.setValues(x0, g0, Z0, x1, g1, Z1, x2, g2, Z2, x3, g3, Z3);

        // Call NMPCProblem::solve() and store the return values
        auto result = nmpcProblem_.solve(inputs.data(), minJ);
        double optNum = result.first; // Extract optNum from the result
        std::vector<double> inputs = result.second;
        // printf("\nminJ - found minimum at J(%g,%g,%g,%g) = %g\n", inputs[0], inputs[1], inputs[2], inputs[3], minJ);
        // printf("\noptNum - found minimum at J(%g,%g,%g,%g) = %g\n", inputs[0], inputs[1], inputs[2], inputs[3], optNum);

        printf("\nfound minimum at J(%g,%g,%g,%g) = %g\n", inputs[0], inputs[1], inputs[2], inputs[3], optNum);

        double end = ros::Time::now().toSec();
        double tf = ros::WallTime::now().toSec();
        double timer = tf - t0;
        double dt = end - start;
        realtime = realtime + dt;

        // Publish velocities and data using ROSPublisher
        publisher_.publishVelocities(inputs);

        publisher_.publishRecData(minJ, optNum,
                                  x0, g0, x1, g1, x2, g2, x3, g3, timer, dt);
    }
    else
    {
        ROS_ERROR("Optimization failed!");
        //...Handle failure case...
    }
}