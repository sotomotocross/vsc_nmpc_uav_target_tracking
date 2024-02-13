// ROSListener.cpp
#include "vsc_nmpc_uav_target_tracking/ROSListener.hpp"

#include "img_seg_cnn/PredData.h"
#include "std_msgs/Float64.h"
#include <Eigen/Dense>

using namespace std;

ROSListener::ROSListener(ros::NodeHandle &nh) : nh_(nh)
{
    l = 252.07;
    flag = 0;
    umax = 720;
    umin = 0;
    vmax = 480;
    vmin = 0;
    cu = 0.5 * (umax + umin);
    cv = 0.5 * (vmax + vmin);
    altitude_received = false;
    features_received = false;

    // Set up ROS subscribers
    feature_sub_ = nh_.subscribe<img_seg_cnn::PredData>("/pred_data", 10, &ROSListener::featureCallback, this);
    alt_sub_ = nh_.subscribe<std_msgs::Float64>("/mavros/global_position/rel_alt", 10, &ROSListener::altitudeCallback, this);
}

ROSListener::~ROSListener()
{
    // Shutdown subscribers
    alt_sub_.shutdown();
    feature_sub_.shutdown();
}

void ROSListener::featureCallback(const img_seg_cnn::PredData::ConstPtr &s_message)
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

    // cout << "Feature callback data" << endl;
    // cout << "(x0,g0): (" << x0 << "," << g0 << ")" << endl;
    // cout << "(x1,g1): (" << x1 << "," << g1 << ")" << endl;
    // cout << "(x2,g2): (" << x2 << "," << g2 << ")" << endl;
    // cout << "(x3,g3): (" << x3 << "," << g3 << ")" << endl;

    // Set features flag to true
    features_received = true;
    // cout << "features_received: " << features_received << endl;
}

void ROSListener::altitudeCallback(const std_msgs::Float64::ConstPtr &alt_message)
{
    // Handle altitude data...
    Z0 = alt_message->data;
    Z1 = alt_message->data;
    Z2 = alt_message->data;
    Z3 = alt_message->data;

    flag = 1;
    // cout << "Altitude callback flag: " << flag << endl;

    // cout << "Altitude callback data" << endl;
    // cout << "Z0: " << Z0 << endl;
    // cout << "Z1: " << Z1 << endl;
    // cout << "Z2: " << Z2 << endl;
    // cout << "Z3: " << Z3 << endl;

    // Set altitude flag to true
    altitude_received = true;
    // cout << "altitude_received: " << altitude_received << endl;
}
