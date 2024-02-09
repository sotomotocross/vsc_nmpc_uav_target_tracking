#include "vsc_nmpc_uav_target_tracking/Controller.hpp"
#include "vsc_nmpc_uav_target_tracking/DynamicsCalculator.hpp"
#include "vsc_nmpc_uav_target_tracking/VelocityTransformer.hpp"
#include "vsc_nmpc_uav_target_tracking/FeatureData.hpp"

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

using namespace std;
using namespace Eigen;

namespace vsc_nmpc_uav_target_tracking
{
  // Initialize the static member of the UtilityFunctions
  DynamicsCalculator Controller::dynamics_calculator;

  // Constructor
  Controller::Controller(ros::NodeHandle &nh, ros::NodeHandle &pnh)
      : nh_(nh), pnh_(pnh)
  {
    features_n = 4;
    dim_inputs = 4;
    dim_s = 2 * features_n;
    mpc_hrz = 6;
    mpc_dt = 0.1;
    l = 252.07;
    a = 10;
    // optNum;
    flag = 0;

    // Simulator camera parameters
    umax = 720;
    umin = 0;
    vmax = 480;
    vmin = 0;
    cu = 0.5 * (umax + umin);
    cv = 0.5 * (vmax + vmin);

    // Camera Frame Desired Features
    // Simulator (720*480)
    u0d = 340;
    v0d = 480;
    u1d = 340;
    v1d = 0;
    u2d = 380;
    v2d = 0;
    u3d = 380;
    v3d = 480;

    x0d = (u0d - cu) / l;
    g0d = (v0d - cv) / l;
    x1d = (u1d - cu) / l;
    g1d = (v1d - cv) / l;
    x2d = (u2d - cu) / l;
    g2d = (v2d - cv) / l;
    x3d = (u3d - cu) / l;
    g3d = (v3d - cv) / l;

    sc_x = (umax - cu) / l;
    sc_y = (vmax - cv) / l;

    // Initialize MPC Variables
    s_des.setZero(dim_s, mpc_hrz + 1);
    ek.setZero(dim_s);
    s_abs.setZero(dim_s);

    //****SET MPC COST FUNCTION MATRICES****//
    Q.setIdentity(dim_s, dim_s);
    R.setIdentity(dim_inputs, dim_inputs);
    P.setIdentity(dim_s, dim_s);

    // Load parameters from ROS parameter server
    pnh_.param<double>("vsc_nmpc_uav_target_tracking/gain_tx", gain_tx, 1.0);
    pnh_.param<double>("vsc_nmpc_uav_target_tracking/gain_ty", gain_ty, 1.0);
    pnh_.param<double>("vsc_nmpc_uav_target_tracking/gain_tz", gain_tz, 2.0);
    pnh_.param<double>("vsc_nmpc_uav_target_tracking/gain_yaw", gain_yaw, 1.0);

    // Set up ROS subscribers
    feature_sub_ = nh.subscribe<img_seg_cnn::PredData>("/pred_data", 10, &Controller::featureCallback, this);
    alt_sub_ = nh.subscribe<std_msgs::Float64>("/mavros/global_position/rel_alt", 10, &Controller::altitudeCallback, this);

    // Set up ROS publishers
    vel_pub_ = nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);
    rec_pub_ = nh.advertise<vsc_nmpc_uav_target_tracking::rec>("/vsc_nmpc_uav_target_tracking/msg/rec", 1);

    // Create a thread for the control loop
    control_loop_thread = std::thread([this]()
                                      {
      ros::Rate rate(50); // Adjust the rate as needed
      while (ros::ok())
      {
        update();
        rate.sleep();
      } });
  }

  // Destructor
  Controller::~Controller()
  {
    // Shutdown ROS publishers...
    vel_pub_.shutdown();
  }

  // Implement costFunction and constraints with additional parameters for dim_s and mpc_dt
  double Controller::costFunction(unsigned int n, const double *x, double *grad, void *data,
                                  int dim_inputs, int mpc_hrz,
                                  double Z0, double Z1, double Z2, double Z3,
                                  double x0, double x1, double x2, double x3,
                                  double x0d, double x1d, double x2d, double x3d,
                                  double g0, double g1, double g2, double g3,
                                  double g0d, double g1d, double g2d, double g3d,
                                  double l, int dim_s, double mpc_dt, Matrix s_des,
                                  MatrixXd Q, MatrixXd R, MatrixXd P, VectorXd ek)
  {

    //   MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz> > ((double*) x);
    MatrixXd inputs(dim_inputs, mpc_hrz);
    Map<MatrixXd> inputs_map((double *)x, dim_inputs, mpc_hrz);
    inputs = inputs_map;

    // Trajectory of States (image features)
    MatrixXd traj_s(dim_s, mpc_hrz + 1);
    traj_s.setZero(dim_s, mpc_hrz + 1);
    traj_s.col(0) << x0, g0, x1, g1, x2, g2, x3, g3;
    // cout << "traj_s: \n" << traj_s << endl;
    // cout << "inputs: " << inputs << endl;

    // Progate the model (IBVS with Image Jacobian)
    for (int k = 0; k < mpc_hrz; k++)
    {
      // cout << "Mpike to gamidi!!!" << endl;
      VectorXd sdot = dynamics_calculator.IBVS_system(inputs.col(k), dim_s, dim_inputs,
                                                      Z0, Z1, Z2, Z3,
                                                      x0, x1, x2, x3,
                                                      g0, g1, g2, g3);
      //   cout << "s_dot" << sdot << endl;
      traj_s.col(k + 1) = traj_s.col(k) + sdot * mpc_dt;
    }

    // Calculate Running Costs
    double Ji = 0.0;

    //   cout << "traj_s =" << traj_s << endl;

    //****DEFINE INITIAL DESIRED V****//
    // VectorXd s_des(dim_s);
    
    s_des.col(0) << x0d, g0d, x1d, g1d, x2d, g2d, x3d, g3d;

    //****SET V DESIRED VELOCITY FOR THE VTA****//
    double b = 15;
    VectorXd s_at(dim_s);
    s_at.setZero(dim_s);
    s_at << 0, b / l, 0, b / l, 0, b / l, 0, b / l;

    //****PROPOGATE THE V DESIRED IN THE HORIZON N FOR dt TIMESTEP SIZE****//
    for (int k = 0; k < mpc_hrz; k++)
    {
      s_des.col(k + 1) = s_des.col(k) + s_at;
    }

    // cout << "s_des = " << s_des << endl;
    // cout << "v0d FUNCTION = " << g0d*l + cv << endl;
    // printf("g0d FUNCTION:%lf\n", g0d);

    for (int k = 0; k < mpc_hrz; k++)
    {
      ek = traj_s.col(k) - s_des.col(k);

      Ji += ek.transpose() * Q * ek;
      Ji += inputs.col(k).transpose() * R * inputs.col(k);
    }

    // cout << "ek = " << ek << endl;

    // Calculate Terminal Costs
    double Jt;
    VectorXd et = traj_s.col(mpc_hrz) - s_des.col(mpc_hrz);

    Jt = et.transpose() * P * et;
    // cout << "et = " << et << endl;

    // cout << "Ji" << Ji << "+" << "Jt" << Jt << endl;
    return Ji + Jt;
  }

  //****DEFINE FOV CONSTRAINTS****//
  void Controller::constraints(unsigned int m, double *c, unsigned int n, const double *x, double *grad, void *data,
                             int dim_inputs, int mpc_hrz,
                             double x0, double x1, double x2, double x3,
                             double g0, double g1, double g2, double g3,
                             double l, int dim_s, double mpc_dt, VectorXd s_abs)
  {
    //   MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz> > ((double*) x);
    MatrixXd inputs(dim_inputs, mpc_hrz);
    Map<MatrixXd> inputs_map((double *)x, dim_inputs, mpc_hrz);
    inputs = inputs_map;

    MatrixXd traj_s(dim_s, mpc_hrz + 1);
    traj_s.setZero(dim_s, mpc_hrz + 1);
    traj_s.col(0) << x0, g0, x1, g1, x2, g2, x3, g3;

    // Progate the model (IBVS with Image Jacobian)
    for (int k = 0; k < mpc_hrz; k++)
    {
      VectorXd sdot = dynamics_calculator.IBVS_system(inputs.col(k), dim_s, dim_inputs,
                                                      Z0, Z1, Z2, Z3,
                                                      x0, x1, x2, x3,
                                                      g0, g1, g2, g3);
      traj_s.col(k + 1) = traj_s.col(k) + sdot * mpc_dt;
    }

    // cout << "traj_s" << traj_s << endl;

    // Calculate Field Of View (Linear inequality constraints.)
    for (int i = 0; i < mpc_hrz + 1; i++)
    {
      // t = (traj_s.col(i) - s_bc).cwiseAbs() - s_br;
      t = (traj_s.col(i)).cwiseAbs() - s_abs;
      for (int j = 0; j < dim_s; j++)
      {
        c[dim_s * i + j] = t(j);
      }
    }
    // cout << "t = " << t << endl;
    // cout << "C FOV constraints" << c << endl;
  }

  // Callback for altitude data
  void Controller::altitudeCallback(const std_msgs::Float64::ConstPtr &alt_message)
  {
    // Handle altitude data...
    Z0 = alt_message->data;
    Z1 = alt_message->data;
    Z2 = alt_message->data;
    Z3 = alt_message->data;
    flag = 1;
    // cout << "flag = " << flag << endl;
    cout << "Z0: " << Z0 << endl;
  }

  //****UPDATE IMAGE FEATURE COORDINATES****//
  void Controller::featureCallback(const img_seg_cnn::PredData::ConstPtr &s_message)
  {
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
  }

  // Main update function
  void Controller::update()
  {
    // Add print statements for debugging
    ROS_INFO("Update function called...");

    Q = 10 * Q;
    R = 5 * R;
    P = 1 * Q;

    R(0, 0) = 15;
    R(2, 2) = 500;
    R(3, 3) = 15;

    s_abs << sc_x, sc_y, sc_x, sc_y, sc_x, sc_y, sc_x, sc_y;
    // cout << "s_abs: " << s_abs << endl;

    //****DEFINE INPUT CONSTRAINTS****//
    double inputs_lb[dim_inputs * mpc_hrz];
    double inputs_ub[dim_inputs * mpc_hrz];

    for (int k = 0; k < mpc_hrz; k++)
    {
      inputs_lb[dim_inputs * k] = -0.5;
      inputs_lb[dim_inputs * k + 1] = -3;
      inputs_lb[dim_inputs * k + 2] = -0.1;
      inputs_lb[dim_inputs * k + 3] = -1;
      inputs_ub[dim_inputs * k] = 0.5;
      inputs_ub[dim_inputs * k + 1] = 3;
      inputs_ub[dim_inputs * k + 2] = 0.1;
      inputs_ub[dim_inputs * k + 3] = 1;
    }

    //****CREATE NLOPT OPTIMIZATION OBJECT, ALGORITHM & TOLERANCES****//

    opt = nlopt_create(NLOPT_LN_BOBYQA, dim_inputs * mpc_hrz); // algorithm and dimensionality
    nlopt_set_lower_bounds(opt, inputs_lb);
    nlopt_set_upper_bounds(opt, inputs_ub);
    // nlopt_set_min_objective(opt, Controller::costFunction, NULL);
    nlopt_set_min_objective(opt, Controller::costFunction,
                            &dim_inputs, &mpc_hrz,
                            &Z0, &Z1, &Z2, &Z3,
                            &x0, &x1, &x2, &x3, 
                            &x0d, &x1d, &x2d, &x3d,
                            &g0, &g1, &g2, &g3,
                            &g0d, &g1d, &g2d, &g3d,
                            &l, &dim_s, &mpc_dt, &s_des,
                            &Q, &R, &P, &ek);
    nlopt_set_ftol_abs(opt, 0.0001);
    nlopt_set_xtol_abs1(opt, 0.0001);
    // nlopt_set_maxtime(opt, 0.25);

    //****DEFINE CONSTRAINTS****//
    double constraints_tol[dim_s * (mpc_hrz + 1)];
    for (int k = 0; k < dim_s * (mpc_hrz + 1); k++)
    {
      constraints_tol[k] = 0.001;
    }

    // add constraints
    // nlopt_add_inequality_constraint(opt, dim_s * (mpc_hrz + 1), constraints, NULL, constraints_tol);
    nlopt_add_inequality_constraint(opt, dim_s * (mpc_hrz + 1), Controller::constraints,
                                    &dim_inputs, &mpc_hrz,
                                    &Z0, &Z1, &Z2, &Z3,
                                    &x0, &x1, &x2, &x3 & x0d, &x1d, &x2d, &x3d,
                                    &g0, &g1, &g2, &g3,
                                    &g0d, &g1d, &g2d, &g3d,
                                    &l, &dim_s, &mpc_dt, &s_abs);

    //****INITIALIZE INPUT VECTOR****//
    double inputs[dim_inputs * mpc_hrz];
    for (int i = 0; i < dim_inputs * mpc_hrz; i++)
    {
      inputs[i] = 0.0;
    }

    //****OPTIMIZE****//
    double minJ; /* the minimum objective value, upon return */

    double t0 = ros::WallTime::now().toSec();
    // printf("Start time:%lf\n", t0);
    double realtime = 0;
    if (x0 != 0 && g0 != 0)
    {
      double start = ros::Time::now().toSec();
        // printf("Start time:%lf\n", start);

        //****EXECUTE OPTIMIZATION****//
        if (flag)
        {
          optNum = nlopt_optimize(opt, inputs, &minJ);
          cout << "Optimization Return Code: " << nlopt_optimize(opt, inputs, &minJ) << endl;
          // cout << "Optimization Return Code: " << optNum << endl;
        }
        printf("found minimum at J(%g,%g,%g,%g) = %g\n", inputs[0], inputs[1], inputs[2], inputs[3], minJ);

        double end = ros::Time::now().toSec();
        double tf = ros::WallTime::now().toSec();
        double timer = tf - t0;
        double dt = end - start;
        realtime = realtime + dt;

        ROS_INFO("Optimization success!");
        //...Publish velocity commands...
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

        Tx = VelocityTransformer::VelTrans1(VelocityTransformer::VelTrans(caminputs))(0, 0);
        Ty = VelocityTransformer::VelTrans1(VelocityTransformer::VelTrans(caminputs))(1, 0);
        Tz = VelocityTransformer::VelTrans1(VelocityTransformer::VelTrans(caminputs))(2, 0);
        Oz = VelocityTransformer::VelTrans1(VelocityTransformer::VelTrans(caminputs))(5, 0);

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

        printf("Drone Velocities Tx,Ty,Tz,Oz(%g,%g,%g,%g) =", Tx, Ty, Tz, Oz);

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
    //****REMEMBER TO FREE ALLOCATED MEMORY AND DESTROY THE OPTIMIZATION OBJECT****//
    nlopt_destroy(opt);
  }

} // namespace vsc_nmpc_uav_target_tracking
