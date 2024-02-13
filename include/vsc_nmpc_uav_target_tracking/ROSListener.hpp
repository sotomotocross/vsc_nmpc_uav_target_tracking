#pragma once

#include <ros/ros.h>
#include <std_msgs/Float64.h>
#include <img_seg_cnn/PredData.h>
#include <string>

class ROSListener
{
public:
    ROSListener(ros::NodeHandle& nh);
    ~ROSListener();

    // Callback for altitude data
    void altitudeCallback(const std_msgs::Float64::ConstPtr& alt_message);

    // Callback for image feature data
    void featureCallback(const img_seg_cnn::PredData::ConstPtr& s_message);

    // Member functions to retrieve variables
    double getX0() const { return x0; }
    double getX1() const { return x1; }
    double getX2() const { return x2; }
    double getX3() const { return x3; }

    double getG0() const { return g0; }
    double getG1() const { return g1; }
    double getG2() const { return g2; }
    double getG3() const { return g3; }
    
    double getZ0() const { return Z0; }
    double getZ1() const { return Z1; }
    double getZ2() const { return Z2; }
    double getZ3() const { return Z3; }

private:
    ros::NodeHandle nh_;

    ros::Subscriber feature_sub_;
    ros::Subscriber alt_sub_;

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

    bool altitude_received;
    bool features_received;

    double l;
    int flag;
};
