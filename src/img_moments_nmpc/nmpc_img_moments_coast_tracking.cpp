#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "mavros_msgs/PositionTarget.h"
#include "img_seg_cnn/PREDdata.h"
#include "img_seg_cnn/POLYcalc_custom.h"
#include "img_seg_cnn/POLYcalc_custom_tf.h"
#include "std_msgs/Float64.h"
#include "vsc_nmpc_uav_target_tracking/rec.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <nlopt.hpp>
#include <eigen3/Eigen/Dense>

// Distance definition for CBF
#include <bits/stdc++.h>
// To store the point
#define Point pair<double, double>
#define F first
#define S second

using namespace std;
using namespace Eigen;

double cX, cY;
int cX_int, cY_int;

// Global MPC Variables
//  int features_n = 4;
int features_n = 1;
const int dim_inputs = 4;
// const int dim_inputs = 6;
int dim_s = 4;
// int dim_s = 2*features_n;
const int mpc_hrz = 10; // 10
double mpc_dt = 0.001;  // 0.001
double l = 252.07;
double a = 10;
double optNum;

MatrixXd s_des(dim_s, mpc_hrz + 1);
VectorXd stored_s_des(dim_s);
VectorXd stored_traj_s(dim_s);
MatrixXd s_des_test(dim_s, mpc_hrz + 1);
// VectorXd s_des(dim_s);
VectorXd s_ub(dim_s);
VectorXd s_lb(dim_s);
VectorXd s_abs(dim_s);
VectorXd s_bc;
VectorXd s_br;
MatrixXd Q;
MatrixXd R;
MatrixXd P;
MatrixXd gains;
VectorXd t;
VectorXd ek;
VectorXd ek_1;
VectorXd feature_vector;
VectorXd transformed_features;
VectorXd opencv_moments;
MatrixXd polygon_features;
MatrixXd transformed_polygon_features;

// Simulator camera parameters
double umax = 720;
double umin = 0;
double vmax = 480;
double vmin = 0;
double cu = 360.5;
double cv = 240.5;

int flag = 0;

// Camera Frame Update Callback Variables
double Z0, Z1, Z2, Z3;
double Tx, Ty, Tz, Oz;

double s_bar_x, s_bar_y;
double first_min_index, second_min_index;
double custom_sigma, custom_sigma_square, custom_sigma_square_log;
double angle_tangent, angle_radian, angle_deg;

double transformed_s_bar_x, transformed_s_bar_y;
double transformed_first_min_index, transformed_second_min_index;
double transformed_sigma, transformed_sigma_square, transformed_sigma_square_log;
double transformed_tangent, transformed_angle_radian, transformed_angle_deg;

double s_bar_x_des = Z0 * (cu - cu / l);
double s_bar_y_des = Z0 * (cv - cv / l);

// double sigma_des = 18500.0;
double sigma_des = 18.5;
double sigma_square_des = sqrt(sigma_des);
double sigma_log_des = log(sigma_square_des);

double angle_deg_des = 0;
double angle_des_tan = tan((angle_deg_des / 180) * 3.14);

// double sigma_constraints = 25000.0;
double sigma_constraints = 25.0;
double sigma_constraints_square = sqrt(sigma_constraints);
double sigma_constraints_square_log = log(sigma_constraints_square);

double angle_deg_constraint = 45;
double angle_deg_constraint_tan = tan((angle_deg_constraint / 180) * 3.14);

// distance between two 2D points
double distance(double x1, double y1, double x2, double y2)
{
   return sqrt(pow(x2 - x1, 2) + pow(y2 - y1, 2));
}

// distance between a point and a line segment
double distance(double px, double py, double x1, double y1, double x2, double y2, double &projx, double &projy)
{
   double dx = x2 - x1;
   double dy = y2 - y1;
   double dot = dx * (px - x1) + dy * (py - y1);
   if (dot <= 0)
   {
      projx = x1;
      projy = y1;
      return distance(px, py, x1, y1);
   }
   double len = distance(x1, y1, x2, y2);
   if (dot >= len * len)
   {
      projx = x2;
      projy = y2;
      return distance(px, py, x2, y2);
   }
   projx = x1 + (dot / (len * len)) * dx;
   projy = y1 + (dot / (len * len)) * dy;
   return distance(px, py, projx, projy);
}

// distance between a point and a polygon
double distance(double px, double py, const vector<pair<double, double>> &polygon, double &closestx, double &closesty)
{
   double minDist = distance(px, py, polygon[0].first, polygon[0].second);
   closestx = polygon[0].first;
   closesty = polygon[0].second;
   for (int i = 0; i < polygon.size() - 1; i++)
   {
      double projx, projy;
      double dist = distance(px, py, polygon[i].first, polygon[i].second, polygon[i + 1].first, polygon[i + 1].second, projx, projy);
      if (dist < minDist)
      {
         minDist = dist;
         closestx = projx;
         closesty = projy;
      }
   }
   return minDist;
}

// Function to return the minimum distance
// between a line segment AB and a point E
double minDistance(Point A, Point B, Point E)
{

   // vector AB
   pair<double, double> AB;
   AB.F = B.F - A.F;
   AB.S = B.S - A.S;

   // vector BP
   pair<double, double> BE;
   BE.F = E.F - B.F;
   BE.S = E.S - B.S;

   // vector AP
   pair<double, double> AE;
   AE.F = E.F - A.F,
   AE.S = E.S - A.S;

   // Variables to store dot product
   double AB_BE, AB_AE;

   // Calculating the dot product
   AB_BE = (AB.F * BE.F + AB.S * BE.S);
   AB_AE = (AB.F * AE.F + AB.S * AE.S);

   // Minimum distance from
   // point E to the line segment
   double reqAns = 0;

   // Case 1
   if (AB_BE > 0)
   {

      // Finding the magnitude
      double y = E.S - B.S;
      double x = E.F - B.F;
      reqAns = sqrt(x * x + y * y);
   }

   // Case 2
   else if (AB_AE < 0)
   {
      double y = E.S - A.S;
      double x = E.F - A.F;
      reqAns = sqrt(x * x + y * y);
   }

   // Case 3
   else
   {

      // Finding the perpendicular distance
      double x1 = AB.F;
      double y1 = AB.S;
      double x2 = AE.F;
      double y2 = AE.S;
      double mod = sqrt(x1 * x1 + y1 * y1);
      reqAns = abs(x1 * y2 - y1 * x2) / mod;
   }
   // cout << "reqAns = " << reqAns << endl;
   return reqAns;
}

double state_bar_fnct_calc(VectorXd camTwist)
{

   double inputs_lb_Tx = -0.5;
   double inputs_lb_Ty = -3;
   double inputs_lb_Tz = -0.1;
   double inputs_lb_Oz = -1;

   double inputs_ub_Tx = 0.5;
   double inputs_ub_Ty = 3;
   double inputs_ub_Tz = 0.1;
   double inputs_ub_Oz = 1;

   double bv_Tx = (-2 / inputs_ub_Tx) + (-2 / inputs_lb_Tx) + (1 / (inputs_ub_Tx - camTwist[0])) + (1 / (camTwist[0] + inputs_ub_Tx)) + (1 / (inputs_lb_Tx - camTwist[0])) + (1 / (camTwist[0] + inputs_lb_Tx));
   double bv_Ty = (-2 / inputs_ub_Ty) + (-2 / inputs_lb_Ty) + (1 / (inputs_ub_Ty - camTwist[1])) + (1 / (camTwist[1] + inputs_ub_Ty)) + (1 / (inputs_lb_Ty - camTwist[1])) + (1 / (camTwist[1] + inputs_lb_Ty));
   double bv_Tz = (-2 / inputs_ub_Tz) + (-2 / inputs_lb_Tz) + (1 / (inputs_ub_Tz - camTwist[2])) + (1 / (camTwist[2] + inputs_ub_Tz)) + (1 / (inputs_lb_Tz - camTwist[2])) + (1 / (camTwist[2] + inputs_lb_Tz));
   double bv_Oz = (-2 / inputs_ub_Oz) + (-2 / inputs_lb_Oz) + (1 / (inputs_ub_Oz - camTwist[3])) + (1 / (camTwist[3] + inputs_ub_Oz)) + (1 / (inputs_lb_Oz - camTwist[3])) + (1 / (camTwist[3] + inputs_lb_Oz));

   double bv = bv_Tx + bv_Ty + bv_Tz + bv_Oz;
   // cout << "bv: " << bv << endl;

   return bv;
}

VectorXd barrier_function_calculation()
{
   double a = 1.5;
   double sigma_lb = 3.8;
   double sigma_ub = 5.5;
   double b = 0.9;

   vector<pair<double, double>> polygon = {{-1.43, -0.954}, {1.43, -0.954}, {1.43, 0.954}, {-1.43, 0.954}}; // square

   double px = ((opencv_moments[1] / opencv_moments[0]) - cu) / l, py = ((opencv_moments[2] / opencv_moments[0]) - cv) / l; // point
   double closestx, closesty;
   double dist = distance(px, py, polygon, closestx, closesty);
   // cout << "Minimum distance from point (" << px << ", " << py << ") to polygon is " << dist << endl;
   // cout << "Closest point on polygon is (" << closestx << ", " << closesty << ")" << endl;

   double px_des = 0.0, py_des = 0.0; // point
   double closestx_des, closesty_des;
   double dist_des = distance(px_des, py_des, polygon, closestx_des, closesty_des);
   // cout << "Minimum distance from point (" << px_des << ", " << py_des << ") to polygon is " << dist_des << endl;
   // cout << "Closest point on polygon is (" << closestx_des << ", " << closesty_des << ")" << endl;

   VectorXd x(2);
   x << px, py;
   VectorXd z(2);
   z << closestx, closesty;
   VectorXd x_des(2);
   x_des << px_des, py_des;
   VectorXd z_des(2);
   z_des << closestx_des, closesty_des;

   double b_1 = 0.0, b_1_d = 0.0, dl1_dd = 0.0;
   double b_2 = 0.0, b_2_d = 0.0, dl2_dd = 0.0;
   double dd_area_ds = 0.0;
   double grad_b2_s = 0.0;

   if (dist <= a)
   {
      b_1 = 1 / (1 - exp(-pow(dist / (dist - a), 2)));
      b_1_d = 1 / (1 - exp(-pow(dist_des / (dist_des - a), 2)));
      dl1_dd = (2 * a * dist_des * exp(-(dist_des / pow(a - dist_des, 2)))) / pow(a - dist_des, 3);
   }
   else if (dist > a)
   {
      b_1 = 1;
      b_1_d = 1;
      dl1_dd = 0;
   }

   double dist_area = min(abs(log(sqrt(opencv_moments[0])) - sigma_lb), abs(log(sqrt(opencv_moments[0])) - sigma_ub));
   double dist_area_des = min(abs(5.0 - sigma_lb), abs(5.0 - sigma_ub));

   if (dist_area <= b)
   {
      // cout << "Mikrotero apo b!!!" << endl;
      b_2 = 1 / (1 - exp(-pow(dist_area / (dist_area - b), 2)));
      b_2_d = 1 / (1 - exp(-pow(dist_area_des / (dist_area_des - b), 2)));
      dl2_dd = (2 * b * dist_area_des * exp(-(dist_area_des / pow(b - dist_area_des, 2)))) / pow(b - dist_area_des, 3);
      dd_area_ds = 1;
      grad_b2_s = -(1 / pow(b_2_d, 2)) * dl2_dd * dd_area_ds;
   }
   else if (dist_area > b)
   {
      // cout << "Megalytero apo b!!!" << endl;
      b_2 = 1;
      b_2_d = 1;
      dl2_dd = 0;
      dd_area_ds = -1;
      grad_b2_s = -(1 / pow(b_2_d, 2)) * dl2_dd * dd_area_ds;
   }

   double dd_dsx = (px_des - closestx_des) / (x_des - z_des).norm();
   double dd_dsy = (py_des - closesty_des) / (x_des - z_des).norm();

   double grad_b1_sx = -(1 / pow(b_1_d, 2)) * dl1_dd * dd_dsx;
   double grad_b1_sy = -(1 / pow(b_1_d, 2)) * dl1_dd * dd_dsy;
   double grad_b1 = grad_b1_sx * (px - px_des) + grad_b1_sy * (py - py_des);

   double grad_b2 = grad_b2_s * (log(sqrt(opencv_moments[0])) - 5.0);

   double r_1 = b_1 - b_1_d + grad_b1;
   double r_2 = b_2 - b_2_d + grad_b2;

   // cout << "dist = " << dist << endl;
   // cout << "dist_des = " << dist_des << endl;
   // cout << "a = " << a << endl;
   // cout << "b_1 = " << b_1 << endl;
   // cout << "b_1_d = " << b_1_d << endl;
   // cout << "grad_b1_sx = " << grad_b1_sx << endl;
   // cout << "grad_b1_sy = " << grad_b1_sy << endl;
   // cout << "grad_b1 = " << grad_b1 << endl;

   // cout << "log(sqrt(opencv_moments[0])) = " << log(sqrt(opencv_moments[0])) << endl;
   // cout << "dist_area = " << dist_area << endl;
   // cout << "dist_area_des = " << dist_area_des << endl;
   // cout << "b = " << b << endl;
   // cout << "b_2 = " << b_2 << endl;
   // cout << "b_2_d = " << b_2_d << endl;
   // cout << "grad_b2_s = " << grad_b2_s << endl;
   // cout << "grad_b2 = " << grad_b2 << endl;

   VectorXd barrier_function(2);
   barrier_function.setZero(2);

   barrier_function << r_1, r_2;
   // cout << "barrier_function: " << barrier_function.transpose() << endl;

   return barrier_function;
}

VectorXd calculate_moments(VectorXd feat_u, VectorXd feat_v)
{
   VectorXd moments(10);
   moments.setZero(10);

   // cout << "feat_u = " << feat_u << endl;
   // cout << "feat_u.size() = " << feat_u.size() << endl;
   // cout << "feat_v = " << feat_v << endl;
   // cout << "feat_v.size() = " << feat_v.size() << endl;

   int N = feat_u.size();

   for (int k = 0; k < N - 1; k++)
   {
      moments[0] += (feat_u[k] * feat_v[k + 1] - feat_u[k + 1] * feat_v[k]); // m00 = area
      // cout << "moments[0] = " << moments[0] << endl;
      moments[1] += (feat_u[k] * feat_v[k + 1] - feat_u[k + 1] * feat_v[k]) * (feat_u[k] + feat_u[k + 1]); // m10
      // cout << "moments[1] = " << moments[1] << endl;
      moments[2] += (feat_u[k] * feat_v[k + 1] - feat_u[k + 1] * feat_v[k]) * (feat_v[k] + feat_v[k + 1]); // m01
      // cout << "moments[2] = " << moments[2] << endl;
      moments[3] += (feat_u[k] * feat_v[k + 1] - feat_u[k + 1] * feat_v[k]) * (pow(feat_u[k], 2) + feat_u[k] * feat_u[k + 1] + pow(feat_u[k + 1], 2)); // m20
      // cout << "moments[3] = " << moments[3] << endl;
      moments[4] += (feat_u[k] * feat_v[k + 1] - feat_u[k + 1] * feat_v[k]) * (pow(feat_v[k], 2) + feat_v[k] * feat_v[k + 1] + pow(feat_v[k + 1], 2)); // m02
      // cout << "moments[4] = " << moments[4] << endl;
      moments[5] += (feat_u[k] * feat_v[k + 1] - feat_u[k + 1] * feat_v[k]) * (2 * feat_u[k] * feat_v[k] + feat_u[k] * feat_v[k + 1] + feat_u[k + 1] * feat_v[k] + 2 * feat_u[k + 1] * feat_v[k + 1]); // m11
      // cout << "moments[5] = " << moments[5] << endl;
      moments[6] += (feat_u[k] * feat_v[k + 1] - feat_u[k + 1] * feat_v[k]) * (3 * pow(feat_u[k], 2) * feat_v[k] + 2 * feat_u[k + 1] + feat_u[k] * feat_v[k] + pow(feat_u[k + 1], 2) + pow(feat_u[k], 2) * feat_v[k + 1] + 2 * feat_u[k + 1] * feat_u[k] * feat_v[k + 1] + 3 * pow(feat_u[k + 1], 2) * feat_v[k + 1]); // m21
      // cout << "moments[6] = " << moments[6] << endl;
      moments[7] += (feat_u[k] * feat_v[k + 1] - feat_u[k + 1] * feat_v[k]) * (3 * pow(feat_v[k], 2) * feat_u[k] + 2 * feat_u[k] * feat_v[k + 1] * feat_v[k] + feat_u[k] * pow(feat_v[k + 1], 2) + feat_u[k + 1] * feat_u[k] * pow(feat_v[k], 2) + 2 * feat_u[k + 1] * feat_v[k + 1] * feat_v[k] + 3 * feat_u[k + 1] * pow(feat_v[k + 1], 2)); // m12
      // cout << "moments[7] = " << moments[7] << endl;
      moments[8] += (feat_u[k] * feat_v[k + 1] - feat_u[k + 1] * feat_v[k]) * (pow(feat_u[k], 3) + feat_u[k + 1] * pow(feat_u[k], 2) + pow(feat_u[k + 1], 2) * feat_u[k] + pow(feat_u[k + 1], 3)); // m30
      // cout << "moments[8] = " << moments[8] << endl;
      moments[9] += (feat_u[k] * feat_v[k + 1] - feat_u[k + 1] * feat_v[k]) * (pow(feat_v[k], 3) + feat_v[k + 1] * pow(feat_v[k], 2) + pow(feat_v[k + 1], 2) * feat_v[k] + pow(feat_v[k + 1], 3)); // m03
                                                                                                                                                                                                   // cout << "moments[9] = " << moments[9] << endl;
   }

   moments[0] = 0.5 * (moments[0] + (feat_u[N - 1] * feat_v[0] - feat_u[0] * feat_v[N - 1])); // m00 = area
   // cout << "moments[0] = " << moments[0] << endl;
   moments[1] = (moments[1] + (feat_u[N - 1] * feat_v[0] - feat_u[0] * feat_v[N - 1]) * (feat_u[N - 1] + feat_u[0]));
   // cout << "moments[1] = " << moments[1] << endl;
   moments[2] = (moments[2] + (feat_u[N - 1] * feat_v[0] - feat_u[0] * feat_v[N - 1]) * (feat_v[N - 1] + feat_v[0]));
   // cout << "moments[2] = " << moments[2] << endl;
   moments[3] = (moments[3] + (feat_u[N - 1] * feat_v[0] - feat_u[0] * feat_v[N - 1]) * (pow(feat_u[N - 1], 2) + feat_u[N - 1] * feat_u[0] + pow(feat_u[0], 2)));
   // cout << "moments[3] = " << moments[3] << endl;
   moments[4] = (moments[4] + (feat_u[N - 1] * feat_v[0] - feat_u[0] * feat_v[N - 1]) * (pow(feat_v[N - 1], 2) + feat_v[N - 1] * feat_v[0] + pow(feat_v[0], 2)));
   // cout << "moments[4] = " << moments[4] << endl;
   moments[5] = (moments[5] + (feat_u[N - 1] * feat_v[0] - feat_u[0] * feat_v[N - 1]) * (2 * feat_u[N - 1] * feat_v[N - 1] + feat_u[N - 1] * feat_v[0] + feat_u[0] * feat_v[N - 1] + 2 * feat_u[0] * feat_v[0]));
   // cout << "moments[5] = " << moments[5] << endl;
   moments[6] = (moments[6] + (feat_u[N - 1] * feat_v[0] - feat_u[0] * feat_v[N - 1]) * (3 * pow(feat_u[N - 1], 2) * feat_v[N - 1] + 2 * feat_u[0] + feat_u[N - 1] * feat_v[N - 1] + pow(feat_u[0], 2) + pow(feat_u[N - 1], 2) * feat_v[0] + 2 * feat_u[0] * feat_u[N - 1] * feat_v[0] + 3 * pow(feat_u[0], 2) * feat_v[0])); // m21
   // cout << "moments[6] = " << moments[6] << endl;
   moments[7] = (moments[7] + (feat_u[N - 1] * feat_v[0] - feat_u[0] * feat_v[N - 1]) * (3 * pow(feat_v[N - 1], 2) * feat_u[N - 1] + 2 * feat_u[N - 1] * feat_v[0] * feat_v[N - 1] + feat_u[N - 1] * pow(feat_v[0], 2) + feat_u[0] * feat_u[N - 1] * pow(feat_v[N - 1], 2) + 2 * feat_u[0] * feat_v[0] * feat_v[N - 1] + 3 * feat_u[0] * pow(feat_v[0], 2))); // m12
   // cout << "moments[7] = " << moments[7] << endl;
   moments[8] = (moments[8] + (feat_u[N - 1] * feat_v[0] - feat_u[0] * feat_v[N - 1]) * (pow(feat_u[N - 1], 3) + feat_u[0] * pow(feat_u[N - 1], 2) + pow(feat_u[0], 2) * feat_u[N - 1] + pow(feat_u[0], 3)));
   // cout << "moments[8] = " << moments[8] << endl;
   moments[9] = (moments[9] + (feat_u[N - 1] * feat_v[0] - feat_u[0] * feat_v[N - 1]) * (pow(feat_v[N - 1], 3) + feat_v[0] * pow(feat_v[N - 1], 2) + pow(feat_v[0], 2) * feat_v[N - 1] + pow(feat_v[0], 3)));
   // cout << "moments[9] = " << moments[9] << endl;

   // cout << "moments[0] = " << moments[0] << endl;
   moments[1] = moments[1] / 6.0; // m10
   // cout << "moments[1] = " << moments[1] << endl;
   moments[2] = moments[2] / 6.0; // m01
   // cout << "moments[2] = " << moments[2] << endl;
   moments[3] = moments[3] / 12.0; // m20
   // cout << "moments[3] = " << moments[3] << endl;
   moments[4] = moments[4] / 12.0; // m02
   // cout << "moments[4] = " << moments[4] << endl;
   moments[5] = moments[5] / 24.0; // m11
   // cout << "moments[5] = " << moments[5] << endl;
   moments[6] = moments[6] / 60.0; // m21
   // cout << "moments[6] = " << moments[6] << endl;
   moments[7] = moments[7] / 60.0; // m12
   // cout << "moments[7] = " << moments[7] << endl;
   moments[8] = moments[8] / 20.0; // m30
   // cout << "moments[8] = " << moments[8] << endl;
   moments[9] = moments[9] / 20.0; // m03
   // cout << "moments[9] = " << moments[9] << endl;

   // cout << "Ntaks, upologistikan ta moments????" << endl;

   return moments;
}

VectorXd calculate_central_moments(VectorXd moments)
{
   VectorXd central_moments(10);
   central_moments.setZero(10);

   double xg = moments[1] / moments[0]; // x-axis centroid
   // cout << "xg: " << xg << endl;
   double yg = moments[2] / moments[0]; // y-axis centroid
   // cout << "yg: " << yg << endl;

   // cout << "barx_meas[0]: " << barx_meas[0] << endl;
   // cout << "barx_meas[1]: " << barx_meas[1] << endl;
   // cout << "barx_meas[2]: " << barx_meas[2] << endl;

   central_moments[0] = abs(moments[0]); // μ00
   // cout << "central_abs(moments[0]) = " << central_abs(moments[0]) << endl;
   central_moments[1] = 0; // μ10
   // cout << "central_moments[1] = " << central_moments[1] << endl;
   central_moments[2] = 0; // μ01
   // cout << "central_moments[2] = " << central_moments[2] << endl;
   central_moments[3] = moments[3] - xg * moments[1]; // μ20
   // cout << "central_moments[3] = " << central_moments[3] << endl;
   central_moments[4] = moments[4] - yg * moments[2]; // μ02
   // cout << "central_moments[4] = " << central_moments[4] << endl;
   central_moments[5] = moments[5] - xg * moments[2]; // μ11
   // cout << "central_moments[5] = " << central_moments[5] << endl;
   central_moments[6] = moments[6] - 2 * xg * moments[5] - yg * moments[3] + 2 * pow(xg, 2) * moments[2]; // μ21
   // cout << "central_moments[6] = " << central_moments[6] << endl;
   central_moments[7] = moments[7] - 2 * yg * moments[5] - xg * moments[4] + 2 * pow(yg, 2) * moments[1]; // μ12
   // cout << "central_moments[7] = " << central_moments[7] << endl;
   central_moments[8] = moments[8] - 3 * xg * moments[3] + 2 * pow(xg, 2) * moments[1]; // μ30
   // cout << "central_moments[8] = " << central_moments[8] << endl;
   central_moments[9] = moments[9] - 3 * yg * moments[4] + 2 * pow(yg, 2) * moments[2]; // μ03
   // cout << "central_moments[9] = " << central_moments[9] << endl;

   // cout << "Ntaks, upologistikan kai ta central moments????" << endl;

   return central_moments;
}

VectorXd img_moments_system(VectorXd camTwist, VectorXd moments)
{
   MatrixXd model_mat(dim_s, dim_inputs);
   MatrixXd Le(4, 6);
   Le.setZero(4, 6);

   double gamma_1 = 1.0;
   double gamma_2 = 1.0;

   double A = -gamma_1 / Z0;
   double B = -gamma_2 / Z0;
   double C = 1 / Z0;

   VectorXd L_area(6);
   L_area.setZero(6);

   double xg = ((moments[1] / moments[0]) - cu) / l; // x-axis centroid
   // cout << "xg = " << xg << endl;
   double yg = ((moments[2] / moments[0]) - cv) / l; // y-axis centroid
   // cout << "yg = " << yg << endl;
   double area = abs(log(sqrt(opencv_moments[0]))); // area

   double n20 = moments[17];
   double n02 = moments[19];
   double n11 = moments[18];

   VectorXd L_xg(6);
   VectorXd L_yg(6);
   L_xg.setZero(6);
   L_yg.setZero(6);

   double mu20_ux = -3 * A * moments[10] - 2 * B * moments[11]; // μ20_ux
   double mu02_uy = -2 * A * moments[11] - 3 * B * moments[12]; // μ02_uy
   double mu11_ux = -2 * A * moments[11] - B * moments[12];     // μ11_ux
   double mu11_uy = -2 * B * moments[11] - A * moments[10];     // μ11_uy
   double s20 = -7 * xg * moments[10] - 5 * moments[13];
   double t20 = 5 * (yg * moments[10] + moments[14]) + 2 * xg * moments[11];
   double s02 = -5 * (xg * moments[12] + moments[15]) - 2 * yg * moments[11];
   double t02 = 7 * yg * moments[12] + 5 * moments[16];
   double s11 = -6 * xg * moments[11] - 5 * moments[14] - yg * moments[10];
   double t11 = 6 * yg * moments[11] + 5 * moments[15] + xg * moments[12];
   double u20 = -A * s20 + B * t20 + 4 * C * moments[10];
   double u02 = -A * s02 + B * t02 + 4 * C * moments[12];
   double u11 = -A * s11 + B * t11 + 4 * C * moments[11];

   VectorXd L_mu20(6);
   VectorXd L_mu02(6);
   VectorXd L_mu11(6);

   L_mu20.setZero(6);
   L_mu02.setZero(6);
   L_mu11.setZero(6);

   L_mu20 << mu20_ux, -B * moments[10], u20, t20, s20, 2 * moments[11];
   L_mu02 << -A * moments[12], mu02_uy, u02, t02, s02, -2 * moments[11];
   L_mu11 << mu11_ux, mu11_uy, u11, t11, s11, moments[12] - moments[10];

   double angle = 0.5 * atan(2 * moments[11] / (moments[10] - moments[12]));
   double Delta = pow(moments[10] - moments[12], 2) + 4 * pow(moments[11], 2);

   double a = moments[11] * (moments[10] + moments[12]) / Delta;
   double b = (2 * pow(moments[11], 2) + moments[12] * (moments[12] - moments[10])) / Delta;
   double c = (2 * pow(moments[11], 2) + moments[10] * (moments[10] - moments[12])) / Delta;
   double d = 5 * (moments[15] * (moments[10] - moments[12]) + moments[11] * (moments[16] - moments[14])) / Delta;
   double e = 5 * (moments[14] * (moments[12] - moments[10]) + moments[11] * (moments[13] - moments[15])) / Delta;

   double angle_ux = area * A + b * B;
   double angle_uy = -c * A - area * B;
   double angle_wx = -b * xg + a * yg + d;
   double angle_wy = a * xg - c * yg + e;
   double angle_uz = -A * angle_wx + B * angle_wy;

   VectorXd L_angle(6);
   L_angle.setZero(6);

   double c1 = moments[10] - moments[12];
   double c2 = moments[16] - 3 * moments[14];
   double s1 = 2 * moments[11];
   double s2 = moments[13] - 3 * moments[15];
   double I1 = pow(c1, 2) + pow(s1, 2);
   double I2 = pow(c2, 2) + pow(s2, 2);
   double I3 = moments[10] + moments[12];
   double Px = I1 / pow(I3, 2);
   double Py = area * I2 / pow(I3, 3);

   // Le.row(0) = L_xg;
   // Le.row(1) = L_yg;
   // Le.row(2) = L_area;
   // Le.row(3) = L_Px;
   // Le.row(4) = L_Py;
   // Le.row(5) = L_angle;

   L_xg << -1 / Z0, 0, (xg / Z0) + 4 * (A * n20 + B * n11), xg * yg + 4 * n11, -(1 + pow(xg, 2) + 4 * n20), yg;
   L_yg << 0, -1 / Z0, (yg / Z0) + 4 * (A * n11 + B * n02), 1 + pow(yg, 2) + 4 * n02, -xg * yg - 4 * n11, -xg;
   L_area << -area * A, -area * B, area * ((3 / Z0) - C), 3 * area * yg, -3 * area * xg, 0;
   L_angle << angle_ux, angle_uy, angle_uz, angle_wx, angle_wy, -1;

   MatrixXd Int_matrix(4, 4);
   Int_matrix << -1 / Z0, 0, (xg / Z0) + 4 * (A * n20 + B * n11), yg,
       0, -1 / Z0, (yg / Z0) + 4 * (A * n11 + B * n02), -xg,
       -area * A, -area * B, area * ((3 / Z0) - C), 0,
       angle_ux, angle_uy, angle_uz, -1;

   gains.setIdentity(dim_s, dim_inputs);

   gains(0, 0) = 1.0;
   gains(1, 1) = 1.0;
   gains(2, 2) = 1.0;
   gains(3, 3) = 1.0;

   // cout << "Int_matrix shape: (" << Int_matrix.rows() << "," << Int_matrix.cols() << ")" << endl;
   // cout << "camTwist shape: (" << camTwist.rows() << "," << camTwist.cols() << ")" << endl;
   // cout << "camTwist: " << camTwist.transpose() << endl;
   // cout << "Int_matrix = \n"
   //      << Int_matrix << endl;
   // cout << "gains * Int_matrix = " << gains * Int_matrix << endl;
   // cout << "Int_matrix*camTwist = " << gains * Int_matrix * camTwist << endl;
   return gains * Int_matrix * camTwist;
}

VectorXd Dynamic_System_x_y_reverted(VectorXd camTwist, VectorXd feat_prop)
{
   MatrixXd model_mat(dim_s, dim_inputs);
   //    cout << "model_mat shape: (" << model_mat.rows() << "," << model_mat.cols() << ")" << endl;

   //    cout << "feat prop inside Dynamic system: " << feat_prop << endl;

   // Barycenter dynamics calculation
   double term_1_4 = 0.0;
   double term_1_5 = 0.0;
   double term_2_4 = 0.0;
   double term_2_5 = 0.0;

   int N;
   N = transformed_features.size() / 2;
   //    cout << "feat_prop: \n" << feat_prop << endl;
   //    cout << "feat_prop[0]: " << feat_prop[0] << endl;
   // cout << "transformed_features: \n"
   //   << transformed_features.transpose() << endl;
   //    cout << "transformed_features[0]: " << transformed_features[0] << endl;
   //    cout << "length feature vector: " << transformed_features.size() << "\n" << endl;
   //    cout << "length feat_prop vector: " << feat_prop.size() << "\n" << endl;

   for (int i = 0; i < N - 1; i += 2)
   {
      //   cout << "1st Vector Index: " << i << "\n" << endl;
      //   cout << "feat_prop[i]*feat_prop[i+1]: " << feat_prop[i]*feat_prop[i+1] << "\n" << endl;
      term_1_4 = term_1_4 + feat_prop[i] * feat_prop[i + 1];
      term_1_5 = term_1_5 + (1 + pow(feat_prop[i], 2));
      term_2_4 = term_2_4 + (1 + pow(feat_prop[i + 1], 2));
      term_2_5 = term_2_5 + feat_prop[i] * feat_prop[i + 1];
   }

   term_1_4 = term_1_4 / N;
   term_1_5 = -term_1_5 / N;
   term_2_4 = term_2_4 / N;
   term_2_5 = -term_2_5 / N;

   double g_4_4, g_4_5, g_4_6;

   // Angle dynamics calculation
   // Fourth term
   double term_4_4_1, term_4_4_2, term_4_4_3, term_4_4_4;
   double sum_4_4_1 = 0.0, sum_4_4_2 = 0.0;

   double k = 0;
   VectorXd x(N);
   VectorXd y(N);

   //    cout << "2N: " << 2*N << endl;
   //    cout << "feat_prop: " << feat_prop.transpose() << endl;

   for (int i = 0; i < 2 * N - 1; i += 2)
   {
      //   cout << "index for x: " << i << endl;
      x[k] = feat_prop[i];
      k++;
   }

   k = 0;

   for (int i = 1; i < 2 * N; i += 2)
   {
      //   cout << "index for y: " << i << endl;
      y[k] = feat_prop[i];
      k++;
   }

   //    cout << "x: " << x.transpose() << endl;
   //    cout << "y: " << y.transpose() << endl;

   for (int i = 0; i < N - 1; i += 2)
   {
      //   cout << "2nd Vector Index: " << i << "\n" << endl;
      //   cout << "feat_prop[i]*feat_prop[i+1]: " << feat_prop[i]*feat_prop[i+1] << "\n" << endl;
      sum_4_4_1 = sum_4_4_1 + pow(feat_prop[i + 1], 2);
      sum_4_4_2 = sum_4_4_2 + feat_prop[i] * feat_prop[i + 1];
   }

   term_4_4_1 = transformed_tangent / (y[transformed_first_min_index] + y[transformed_second_min_index] - 2 * transformed_s_bar_y);
   term_4_4_2 = (pow(y[transformed_first_min_index], 2) + pow(y[transformed_second_min_index], 2) - (2 / N) * sum_4_4_1);
   term_4_4_3 = -1 / (y[transformed_first_min_index] + y[transformed_second_min_index] - 2 * transformed_s_bar_y);
   term_4_4_4 = (x[transformed_first_min_index] * y[transformed_first_min_index] + x[transformed_second_min_index] * y[transformed_second_min_index] - (2 / N) * sum_4_4_2);

   g_4_4 = term_4_4_1 * term_4_4_2 + term_4_4_3 * term_4_4_4;
   //    cout << "g_4_4: " << g_4_4 << endl;

   // Fifth term
   double term_4_5_1, term_4_5_2, term_4_5_3, term_4_5_4;
   double sum_4_5_1 = 0.0, sum_4_5_2 = 0.0;

   for (int i = 0; i < N - 1; i += 2)
   {
      //   cout << "3rd Vector Index: " << i << "\n" << endl;
      //   cout << "feat_prop[i]*feat_prop[i+1]: " << feat_prop[i]*feat_prop[i+1] << "\n" << endl;
      sum_4_5_1 = sum_4_5_1 + pow(feat_prop[i], 2);
      sum_4_5_2 = sum_4_5_2 + feat_prop[i] * feat_prop[i + 1];
   }

   term_4_5_1 = 1 / (y[transformed_first_min_index] + y[transformed_second_min_index] - 2 * transformed_s_bar_y);
   term_4_5_2 = (pow(x[transformed_first_min_index], 2) + pow(x[transformed_second_min_index], 2) - (2 / N) * sum_4_5_1);
   term_4_5_3 = -transformed_tangent / (y[transformed_first_min_index] + y[transformed_second_min_index] - 2 * transformed_s_bar_y);
   term_4_5_4 = (x[transformed_first_min_index] * y[transformed_first_min_index] + x[transformed_second_min_index] * y[transformed_second_min_index] - (2 / N) * sum_4_5_2);

   g_4_5 = term_4_5_1 * term_4_5_2 + term_4_5_3 * term_4_5_4;
   //    cout << "g_4_5: " << g_4_5 << endl;

   // Fifth term
   g_4_6 = pow(transformed_tangent, 2) + 1;
   // cout << "g_4_6: " << g_4_6 << endl;

   // MatrixXd full_model_mat(dim_s, 6);
   // full_model_mat <<
   //          -1/Z0, 0.0, transformed_s_bar_x/Z0, term_1_4, term_1_5, transformed_s_bar_y,
   //          0.0, -1/Z0, transformed_s_bar_y/Z0, term_2_4, term_2_5, -transformed_s_bar_x,
   //          0.0, 0.0, 2/Z0, (3/2)*transformed_s_bar_y, -(3/2)*transformed_s_bar_x, 0.0,
   //          0.0, 0.0, 0.0, g_4_4, g_4_5, g_4_6,
   //          0.0, 0.0, 1.0, 0.0, 0.0, 0.0;

   // cout << "Full model matrix:\n" << full_model_mat << endl;

   //    model_mat <<
   //             -1/Z0, 0.0, transformed_s_bar_x/Z0, transformed_s_bar_y,
   //             0.0, -1/Z0, transformed_s_bar_y/Z0, -transformed_s_bar_x,
   //             0.0, 0.0, 2/Z0, 0.0,
   //             0.0, 0.0, 0.0, g_4_6,
   //             0.0, 0.0, 1.0, 0.0;

   model_mat << -1 / Z0, 0.0, transformed_s_bar_x / Z0, transformed_s_bar_y,
       0.0, -1 / Z0, transformed_s_bar_y / Z0, -transformed_s_bar_x,
       0.0, 0.0, 2 / Z0, 0.0,
       0.0, 0.0, 0.0, g_4_6;

   // model_mat <<
   //          -1/Z0, 0.0, transformed_s_bar_x/Z0, transformed_s_bar_y,
   //          0.0, -1/Z0, transformed_s_bar_y/Z0, -transformed_s_bar_x,
   //          0.0, 0.0, 2/Z0, 0.0,
   //          0.0, 0.0, 0.0, 0.0,
   //          0.0, 0.0, 1.0, 0.0;

   // cout << "model_mat: \n"
   //   << model_mat << endl;

   //    cout << "model_mat shape: (" << model_mat.rows() << "," << model_mat.cols() << ")" << endl;
   //    cout << "Interaction Matrix:\n" << model_mat << endl;
   //    cout << "camTwist shape: (" << camTwist.rows() << "," << camTwist.cols() << ")" << endl;
   //    cout << "camTwist: " << camTwist << endl;
   //    cout << "Interaction Matrix" << model_mat * camTwist << endl;

   return model_mat * camTwist;
}

// Camera-UAV Velocity Transform VelUAV
MatrixXd VelTrans(MatrixXd CameraVel)
{
   Matrix<double, 3, 1> tt;
   tt(0, 0) = 0;
   tt(1, 0) = 0;
   tt(2, 0) = 0;

   Matrix<double, 3, 3> Tt;
   Tt(0, 0) = 0;
   Tt(1, 0) = tt(2, 0);
   Tt(2, 0) = -tt(1, 0);
   Tt(0, 1) = -tt(2, 0);
   Tt(1, 1) = 0;
   Tt(2, 1) = tt(0, 0);
   Tt(0, 2) = tt(1, 0);
   Tt(1, 2) = -tt(0, 0);
   Tt(2, 2) = 0;
   double thx = M_PI_2;
   double thy = M_PI;
   double thz = M_PI_2;

   Matrix<double, 3, 3> Rx;
   Rx(0, 0) = 1;
   Rx(1, 0) = 0;
   Rx(2, 0) = 0;
   Rx(0, 1) = 0;
   Rx(1, 1) = cos(thx);
   Rx(2, 1) = sin(thx);
   Rx(0, 2) = 0;
   Rx(1, 2) = -sin(thx);
   Rx(2, 2) = cos(thx);

   Matrix<double, 3, 3> Ry;
   Ry(0, 0) = cos(thy);
   Ry(1, 0) = 0;
   Ry(2, 0) = -sin(thy);
   Ry(0, 1) = 0;
   Ry(1, 1) = 1;
   Ry(2, 1) = 0;
   Ry(0, 2) = sin(thy);
   Ry(1, 2) = 0;
   Ry(2, 2) = cos(thy);

   Matrix<double, 3, 3> Rz;
   Rz(0, 0) = cos(thz);
   Rz(1, 0) = sin(thz);
   Rz(2, 0) = 0;
   Rz(0, 1) = -sin(thz);
   Rz(1, 1) = cos(thz);
   Rz(2, 1) = 0;
   Rz(0, 2) = 0;
   Rz(1, 2) = 0;
   Rz(2, 2) = 1;

   Matrix<double, 3, 3> Rth;
   Rth.setZero(3, 3);
   Rth = Rz * Ry * Rx;

   Matrix<double, 6, 1> VelCam;
   VelCam(0, 0) = CameraVel(0, 0);
   VelCam(1, 0) = CameraVel(1, 0);
   VelCam(2, 0) = CameraVel(2, 0);
   VelCam(3, 0) = 0;
   VelCam(4, 0) = 0;
   VelCam(5, 0) = CameraVel(3, 0);

   Matrix<double, 3, 3> Zeroes;
   Zeroes.setZero(3, 3);

   Matrix<double, 6, 6> Vtrans;
   Vtrans.block(0, 0, 3, 3) = Rth;
   Vtrans.block(0, 3, 3, 3) = Tt * Rth;
   Vtrans.block(3, 0, 3, 3) = Zeroes;
   Vtrans.block(3, 3, 3, 3) = Rth;

   Matrix<double, 6, 1> VelUAV;
   VelUAV.setZero(6, 1);
   VelUAV = Vtrans * VelCam;

   return VelUAV;

   // printf("Camera velocities are (%g,%g,%g,%g)=", VelCam(0,0), VelCam(1,0), VelCam(2,0), VelCam(5,0));
   // printf("UAV velocities are (%g,%g,%g,%g)=", VelUAV(0,0), VelUAV(1,0), VelUAV(2,0), VelUAV(5,0));
}

// Camera-UAV Velocity Transform VelUAV
MatrixXd VelTrans1(MatrixXd CameraVel1)
{
   Matrix<double, 3, 1> tt1;
   tt1(0, 0) = 0;
   tt1(1, 0) = 0;
   tt1(2, 0) = -0.14;

   Matrix<double, 3, 3> Tt1;
   Tt1(0, 0) = 0;
   Tt1(1, 0) = tt1(2, 0);
   Tt1(2, 0) = -tt1(1, 0);
   Tt1(0, 1) = -tt1(2, 0);
   Tt1(1, 1) = 0;
   Tt1(2, 1) = tt1(0, 0);
   Tt1(0, 2) = tt1(1, 0);
   Tt1(1, 2) = -tt1(0, 0);
   Tt1(2, 2) = 0;

   double thx1 = 0;
   double thy1 = M_PI_2;
   double thz1 = 0;

   Matrix<double, 3, 3> Rx1;
   Rx1(0, 0) = 1;
   Rx1(1, 0) = 0;
   Rx1(2, 0) = 0;
   Rx1(0, 1) = 0;
   Rx1(1, 1) = cos(thx1);
   Rx1(2, 1) = sin(thx1);
   Rx1(0, 2) = 0;
   Rx1(1, 2) = -sin(thx1);
   Rx1(2, 2) = cos(thx1);

   Matrix<double, 3, 3> Ry1;
   Ry1(0, 0) = cos(thy1);
   Ry1(1, 0) = 0;
   Ry1(2, 0) = -sin(thy1);
   Ry1(0, 1) = 0;
   Ry1(1, 1) = 1;
   Ry1(2, 1) = 0;
   Ry1(0, 2) = sin(thy1);
   Ry1(1, 2) = 0;
   Ry1(2, 2) = cos(thy1);

   Matrix<double, 3, 3> Rz1;
   Rz1(0, 0) = cos(thz1);
   Rz1(1, 0) = sin(thz1);
   Rz1(2, 0) = 0;
   Rz1(0, 1) = -sin(thz1);
   Rz1(1, 1) = cos(thz1);
   Rz1(2, 1) = 0;
   Rz1(0, 2) = 0;
   Rz1(1, 2) = 0;
   Rz1(2, 2) = 1;

   Matrix<double, 3, 3> Rth1;
   Rth1.setZero(3, 3);
   Rth1 = Rz1 * Ry1 * Rx1;

   Matrix<double, 6, 1> VelCam1;
   VelCam1(0, 0) = CameraVel1(0, 0);
   VelCam1(1, 0) = CameraVel1(1, 0);
   VelCam1(2, 0) = CameraVel1(2, 0);
   VelCam1(3, 0) = CameraVel1(3, 0);
   VelCam1(4, 0) = CameraVel1(4, 0);
   VelCam1(5, 0) = CameraVel1(5, 0);

   Matrix<double, 3, 3> Zeroes1;
   Zeroes1.setZero(3, 3);

   Matrix<double, 6, 6> Vtrans1;
   Vtrans1.block(0, 0, 3, 3) = Rth1;
   Vtrans1.block(0, 3, 3, 3) = Tt1 * Rth1;
   Vtrans1.block(3, 0, 3, 3) = Zeroes1;
   Vtrans1.block(3, 3, 3, 3) = Rth1;

   Matrix<double, 6, 1> VelUAV1;
   VelUAV1.setZero(6, 1);
   VelUAV1 = Vtrans1 * VelCam1;

   return VelUAV1;
   // printf("Camera velocities are (%g,%g,%g,%g)=", VelCam(0,0), VelCam(1,0), VelCam(2,0), VelCam(5,0));
   // printf("UAV velocities are (%g,%g,%g,%g)=", VelUAV(0,0), VelUAV(1,0), VelUAV(2,0), VelUAV(5,0));
}

// IBVS Feature Rate Le*Vk
VectorXd IBVSSystem(VectorXd camTwist)
{
   //    cout << "Mpike kai sto feature propagation to gamidi!!!" << endl;
   MatrixXd Le(transformed_features.size(), dim_inputs);
   //,Le.setZero(dim_s,dim_inputs);
   Le.setZero(transformed_features.size(), dim_inputs);
   // cout << "Le: \n" << Le << endl;

   // Le.row(0) << -1/Z0, 0.0, transformed_features[0]/Z0, transformed_features[1];
   // Le.row(1) << 0.0, -1/Z0, transformed_features[1]/Z0, -transformed_features[0];

   // cout << "after Le: \n" << Le << endl;

   for (int k = 0, kk = 0; k < transformed_features.size() && kk < transformed_features.size(); k++, kk++)
   {
      Le.row(k) << -1 / Z0, 0.0, transformed_features[kk] / Z0, transformed_features[kk + 1];
      Le.row(k + 1) << 0.0, -1 / Z0, transformed_features[kk + 1] / Z0, -transformed_features[kk];
      k++;
      kk++;
   }
   // cout << "after Le: \n"
   //   << Le << endl;
   // cout << "transformed_features.size(): " << transformed_features.size() << endl;
   // cout << "transformed_features: " << transformed_features << endl;

   return Le * camTwist;
}

// PVS-MPC Cost Function
double costFunction(unsigned int n, const double *x, double *grad, void *data)
{
   cout << "\nGia pame ligo!!!" << endl;
   MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz>>((double *)x);
   // Trajectory of States (image features)
   MatrixXd traj_s(dim_s, mpc_hrz + 1);
   MatrixXd traj_s_test(dim_s, mpc_hrz + 1);
   MatrixXd feature_hrz_prop(transformed_features.size(), mpc_hrz + 1);
   traj_s.setZero(dim_s, mpc_hrz + 1);

   traj_s.col(0) << ((opencv_moments[1] / opencv_moments[0]) - cu) / l, ((opencv_moments[2] / opencv_moments[0]) - cv) / l, log(sqrt(opencv_moments[0])), atan(2 * opencv_moments[11] / (opencv_moments[10] - opencv_moments[12]));
   traj_s_test.col(0) << transformed_s_bar_x, transformed_s_bar_y, transformed_sigma_square_log, transformed_tangent;

   // Progate the model (PVS with Image Jacobian)
   for (int k = 0; k < mpc_hrz; k++)
   {
      VectorXd feat_prop = IBVSSystem(inputs.col(k));
      feature_hrz_prop.col(k + 1) = feature_hrz_prop.col(k) + feat_prop * mpc_dt;
      VectorXd sdot = Dynamic_System_x_y_reverted(inputs.col(k), feature_hrz_prop.col(k));

      VectorXd sdot_test = img_moments_system(inputs.col(k), opencv_moments);

      traj_s.col(k + 1) = traj_s.col(k) + sdot_test * mpc_dt;
      traj_s_test.col(k + 1) = traj_s_test.col(k) + sdot * mpc_dt;
   }

   // Calculate Running Costs
   double Ji = 0.0;
   double Ji_1 = 0.0;

   //****DEFINE INITIAL DESIRED V****//
   s_des.col(0) << 0.0, 0.0, 5.0, angle_des_tan;
   s_des_test.col(0) << s_bar_x_des, s_bar_y_des, sigma_log_des, angle_des_tan;

   //****SET V DESIRED VELOCITY FOR THE VTA****//
   double b = 0.0;
   VectorXd s_at(dim_s);
   s_at.setZero(dim_s);
   s_at << b / l, 0, 0, 0;

   //****PROPOGATE THE V DESIRED IN THE HORIZON N FOR dt TIMESTEP SIZE****//
   for (int k = 0; k < mpc_hrz; k++)
   {
      s_des.col(k + 1) = s_des.col(k) + s_at;
      s_des_test.col(k + 1) = s_des_test.col(k) + s_at;
   }

   for (int k = 0; k < mpc_hrz; k++)
   {
      ek = -s_des.col(k) + traj_s.col(k);
      ek_1 = -s_des_test.col(k) + traj_s_test.col(k);

      Ji += ek.transpose() * Q * ek;
      Ji_1 += ek_1.transpose() * Q * ek_1;

      Ji += inputs.col(k).transpose() * R * inputs.col(k);
      Ji_1 += inputs.col(k).transpose() * R * inputs.col(k);
   }
   // cout << "Ji: \n" << Ji << endl;

   // Calculate Terminal Costs
   double Jt;
   double Jt_1;
   s_des.col(mpc_hrz - 1) << 0.0, 0.0, 5.0, angle_des_tan;
   s_des.col(mpc_hrz) << 0.0, 0.0, 5.0, angle_des_tan;
   s_des_test.col(mpc_hrz - 1) << s_bar_x_des, s_bar_y_des, 5.0, angle_des_tan;
   s_des_test.col(mpc_hrz) << s_bar_x_des, s_bar_y_des, 5.0, angle_des_tan;
   VectorXd et = traj_s.col(mpc_hrz) - s_des.col(mpc_hrz);
   VectorXd et_1 = traj_s_test.col(mpc_hrz) - s_des_test.col(mpc_hrz);
   // cout << "traj_s.col(mpc_hrz): " << traj_s.col(mpc_hrz) << endl;

   // cout << "Jt: \n" << Jt << endl;

   Jt = et.transpose() * P * et;
   //   cout << "Ji = " << Ji << " + " << "Jt = " << Jt << endl;\

   VectorXd bar_fnct = barrier_function_calculation();
   double state_bar_fnct = state_bar_fnct_calc(inputs.col(0));

   cout << "traj_s.col(0): " << traj_s.col(0).transpose() << endl;
   cout << "traj_s_test.col(0): " << traj_s_test.col(0).transpose() << endl;

   cout << "s_des.col(0): " << s_des.col(0).transpose() << endl;
   cout << "s_des_test.col(0): " << s_des_test.col(0).transpose() << endl;

   cout << "ek = " << ek.transpose() << endl;
   cout << "et = " << et.transpose() << endl;

   // cout << "ek_1 = " << ek_1.transpose() << endl;
   // cout << "et_1 = " << et_1.transpose() << endl;

   // cout << "bar_fnct: " << bar_fnct.transpose() << endl;
   // cout << "state_bar_fnct: " << state_bar_fnct << endl;

   // cout << "Ji + Jt + bar_fnct[0] + bar_fnct[1] + state_bar_fnct = " << Ji + Jt + bar_fnct[0] + bar_fnct[1] + state_bar_fnct << endl;
   // cout << "Ji_1 + Jt_1 + bar_fnct[0] + bar_fnct[1] + state_bar_fnct = " << Ji_1 + Jt_1 + bar_fnct[0] + bar_fnct[1] + state_bar_fnct << endl;

   cX = opencv_moments[1] / opencv_moments[0];
   cY = opencv_moments[2] / opencv_moments[0];

   cX_int = (int)cX;
   cY_int = (int)cY;

   // cout << "cX = " << cX << endl;
   // cout << "cY = " << cY << endl;
   // cout << "(cX - cu)/l = " << (cX - cu) / l << endl;
   // cout << "(cY - cv)/l = " << (cY - cv) / l << endl;
   // cout << "cX_int = " << cX_int << endl;
   // cout << "cY_int = " << cY_int << endl;
   // cout << "(cX_int - cu)/l = " << (cX_int - cu) / l << endl;
   // cout << "(cY_int - cv)/l = " << (cY_int - cv) / l << endl;

   stored_s_des = s_des.col(0);
   stored_traj_s = traj_s.col(0);

   // return Ji + Jt + bar_fnct[0] + bar_fnct[1] + state_bar_fnct;
   return Ji + Jt;
}

// PVS-MPC Cost Function
double costFunction_alter(unsigned int n, const double *x, double *grad, void *data)
{
   cout << "\nGia pame ligo!!!" << endl;
   MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz>>((double *)x);
   // Trajectory of States (image features)
   MatrixXd traj_s(dim_s, mpc_hrz + 1);
   MatrixXd traj_s_test(dim_s, mpc_hrz + 1);
   MatrixXd feature_hrz_prop(transformed_features.size(), mpc_hrz + 1);
   traj_s.setZero(dim_s, mpc_hrz + 1);

   traj_s.col(0) << ((opencv_moments[1] / opencv_moments[0]) - cu) / l, ((opencv_moments[2] / opencv_moments[0]) - cv) / l, log(sqrt(opencv_moments[0])), atan(2 * opencv_moments[11] / (opencv_moments[10] - opencv_moments[12]));
   traj_s_test.col(0) << transformed_s_bar_x, transformed_s_bar_y, transformed_sigma_square_log, transformed_tangent;

   // Progate the model (PVS with Image Jacobian)
   for (int k = 0; k < mpc_hrz; k++)
   {
      VectorXd feat_prop = IBVSSystem(inputs.col(k));
      feature_hrz_prop.col(k + 1) = feature_hrz_prop.col(k) + feat_prop * mpc_dt;
      VectorXd sdot = Dynamic_System_x_y_reverted(inputs.col(k), feature_hrz_prop.col(k));

      VectorXd sdot_test = img_moments_system(inputs.col(k), opencv_moments);

      traj_s.col(k + 1) = traj_s.col(k) + sdot_test * mpc_dt;
      traj_s_test.col(k + 1) = traj_s_test.col(k) + sdot * mpc_dt;
   }

   // Calculate Running Costs
   double Ji = 0.0;
   double Ji_1 = 0.0;

   //****DEFINE INITIAL DESIRED V****//
   s_des.col(0) << 0.0, 0.0, 5.0, angle_des_tan;
   s_des_test.col(0) << s_bar_x_des, s_bar_y_des, sigma_log_des, angle_des_tan;

   //****SET V DESIRED VELOCITY FOR THE VTA****//
   double b = 0.0;
   VectorXd s_at(dim_s);
   s_at.setZero(dim_s);
   s_at << b / l, 0, 0, 0;

   //****PROPOGATE THE V DESIRED IN THE HORIZON N FOR dt TIMESTEP SIZE****//
   for (int k = 0; k < mpc_hrz; k++)
   {
      s_des.col(k + 1) = s_des.col(k) + s_at;
      s_des_test.col(k + 1) = s_des_test.col(k) + s_at;
   }

   for (int k = 0; k < mpc_hrz; k++)
   {
      ek = -s_des.col(k) + traj_s.col(k);
      ek_1 = -s_des_test.col(k) + traj_s_test.col(k);

      Ji += ek.transpose() * Q * ek;
      Ji_1 += ek_1.transpose() * Q * ek_1;

      Ji += inputs.col(k).transpose() * R * inputs.col(k);
      Ji_1 += inputs.col(k).transpose() * R * inputs.col(k);
   }
   // cout << "Ji: \n" << Ji << endl;

   // Calculate Terminal Costs
   double Jt;
   double Jt_1;
   s_des.col(mpc_hrz - 1) << 0.0, 0.0, 5.0, angle_des_tan;
   s_des.col(mpc_hrz) << 0.0, 0.0, 5.0, angle_des_tan;
   s_des_test.col(mpc_hrz - 1) << s_bar_x_des, s_bar_y_des, 5.0, angle_des_tan;
   s_des_test.col(mpc_hrz) << s_bar_x_des, s_bar_y_des, 5.0, angle_des_tan;
   VectorXd et = traj_s.col(mpc_hrz) - s_des.col(mpc_hrz);
   VectorXd et_1 = traj_s_test.col(mpc_hrz) - s_des_test.col(mpc_hrz);
   // cout << "traj_s.col(mpc_hrz): " << traj_s.col(mpc_hrz) << endl;

   // cout << "Jt: \n" << Jt << endl;

   Jt = et.transpose() * P * et;
   //   cout << "Ji = " << Ji << " + " << "Jt = " << Jt << endl;\

   VectorXd bar_fnct = barrier_function_calculation();
   double state_bar_fnct = state_bar_fnct_calc(inputs.col(0));

   cout << "traj_s.col(0): " << traj_s.col(0).transpose() << endl;
   cout << "traj_s_test.col(0): " << traj_s_test.col(0).transpose() << endl;

   cout << "s_des.col(0): " << s_des.col(0).transpose() << endl;
   cout << "s_des_test.col(0): " << s_des_test.col(0).transpose() << endl;

   // cout << "ek = " << ek.transpose() << endl;
   // cout << "et = " << et.transpose() << endl;

   cout << "ek_1 = " << ek_1.transpose() << endl;
   cout << "et_1 = " << et_1.transpose() << endl;

   // cout << "bar_fnct: " << bar_fnct.transpose() << endl;
   // cout << "state_bar_fnct: " << state_bar_fnct << endl;

   // cout << "Ji + Jt + bar_fnct[0] + bar_fnct[1] + state_bar_fnct = " << Ji + Jt + bar_fnct[0] + bar_fnct[1] + state_bar_fnct << endl;
   // cout << "Ji_1 + Jt_1 + bar_fnct[0] + bar_fnct[1] + state_bar_fnct = " << Ji_1 + Jt_1 + bar_fnct[0] + bar_fnct[1] + state_bar_fnct << endl;

   cX = opencv_moments[1] / opencv_moments[0];
   cY = opencv_moments[2] / opencv_moments[0];

   cX_int = (int)cX;
   cY_int = (int)cY;

   // cout << "cX = " << cX << endl;
   // cout << "cY = " << cY << endl;
   // cout << "(cX - cu)/l = " << (cX - cu) / l << endl;
   // cout << "(cY - cv)/l = " << (cY - cv) / l << endl;
   // cout << "cX_int = " << cX_int << endl;
   // cout << "cY_int = " << cY_int << endl;
   // cout << "(cX_int - cu)/l = " << (cX_int - cu) / l << endl;
   // cout << "(cY_int - cv)/l = " << (cY_int - cv) / l << endl;

   // return Ji + Jt + bar_fnct[0] + bar_fnct[1] + state_bar_fnct;
   // return Ji_1 + Jt_1 + bar_fnct[0] + bar_fnct[1] + state_bar_fnct;
   return Ji_1 + Jt_1;
}

//****DEFINE FOV CONSTRAINTS****//
void constraints(unsigned int m, double *c, unsigned int n, const double *x, double *grad, void *data)
{
   MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz>>((double *)x);
   // Trajectory of States (image features)
   MatrixXd traj_s(dim_s, mpc_hrz + 1);
   MatrixXd traj_s_test(dim_s, mpc_hrz + 1);
   MatrixXd feature_hrz_prop(transformed_features.size(), mpc_hrz + 1);
   traj_s.setZero(dim_s, mpc_hrz + 1);
   //    cout << "angle_tangent: " << angle_tangent << endl;
   //    cout << "sigma_square: " << sigma_square << endl;

   traj_s.col(0) << ((opencv_moments[1] / opencv_moments[0]) - cu) / l, ((opencv_moments[2] / opencv_moments[0]) - cv) / l, log(sqrt(opencv_moments[0])), atan(2 * opencv_moments[11] / (opencv_moments[10] - opencv_moments[12]));
   traj_s_test.col(0) << transformed_s_bar_x, transformed_s_bar_y, transformed_sigma_square_log, transformed_tangent;
   // cout << "traj_s.col(0): " << traj_s.col(0).transpose() << endl;

   // Progate the model (PVS with Image Jacobian)
   for (int k = 0; k < mpc_hrz; k++)
   {
      VectorXd feat_prop = IBVSSystem(inputs.col(k));
      feature_hrz_prop.col(k + 1) = feature_hrz_prop.col(k) + feat_prop * mpc_dt;
      VectorXd sdot = Dynamic_System_x_y_reverted(inputs.col(k), feature_hrz_prop.col(k));

      VectorXd sdot_test = img_moments_system(inputs.col(k), opencv_moments);
      // cout << "sdot_test = " << sdot_test.transpose() << endl;
      traj_s.col(k + 1) = traj_s.col(k) + sdot_test * mpc_dt;
      traj_s_test.col(k + 1) = traj_s_test.col(k) + sdot * mpc_dt;
      //   cout << "traj_s.col(k + 1) =" << traj_s.col(k + 1) << endl;
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

//****UPDATE IMAGE FEATURE COORDINATES****//
void featureCallback_poly_custom(const img_seg_cnn::POLYcalc_custom::ConstPtr &s_message)
{
   feature_vector.setZero(s_message->features.size());
   polygon_features.setZero(s_message->features.size() / 2, 2);

   for (int i = 0; i < s_message->features.size() - 1; i += 2)
   {
      feature_vector[i] = s_message->features[i];
      feature_vector[i + 1] = s_message->features[i + 1];
   }

   for (int i = 0, j = 0; i < s_message->features.size() - 1 && j < s_message->features.size() / 2; i += 2, ++j)
   {
      polygon_features(j, 0) = feature_vector[i];
      polygon_features(j, 1) = feature_vector[i + 1];
   }

   s_bar_x = s_message->barycenter_features[0];
   s_bar_y = s_message->barycenter_features[1];

   first_min_index = s_message->d;
   second_min_index = s_message->f;

   custom_sigma = s_message->custom_sigma;
   custom_sigma_square = s_message->custom_sigma_square;
   custom_sigma_square_log = s_message->custom_sigma_square_log;

   angle_tangent = s_message->tangent;
   angle_radian = s_message->angle_radian;
   angle_deg = s_message->angle_deg;

   // cout << "------------------------------------------------------------------" << endl;
   // cout << "------------ Features shape and angle of the polygon -------------" << endl;
   // cout << "------------------------------------------------------------------" << endl;

   // cout << "feature_vector: " << feature_vector << endl;
   // cout << "polygon_features: " << polygon_features << endl;

   // cout << "s_bar_x: " << s_bar_x << endl;
   // cout << "s_bar_y: " << s_bar_y << endl;

   // cout << "first_min_index: " << first_min_index << endl;
   // cout << "second_min_index: " << second_min_index << endl;

   // cout << "custom_sigma: " << custom_sigma << endl;
   // cout << "custom_sigma_square: " << custom_sigma_square << endl;
   // cout << "custom_sigma_square_log: " << custom_sigma_square_log << endl;

   // cout << "angle_tangent: " << angle_tangent << endl;
   // cout << "angle_radian: " << angle_radian << endl;
   // cout << "angle_deg: " << angle_deg << endl;

   flag = 1;
   // cout << "Feature callback flag: " << flag << endl;
}

//****UPDATE IMAGE FEATURE COORDINATES****//
void featureCallback_poly_custom_tf(const img_seg_cnn::POLYcalc_custom_tf::ConstPtr &s_message)
{
   transformed_features.setZero(s_message->transformed_features.size());
   transformed_polygon_features.setZero(s_message->transformed_features.size() / 2, 2);

   for (int i = 0; i < s_message->transformed_features.size() - 1; i += 2)
   {
      transformed_features[i] = s_message->transformed_features[i];
      transformed_features[i + 1] = s_message->transformed_features[i + 1];
   }

   for (int i = 0, j = 0; i < s_message->transformed_features.size() - 1 && j < s_message->transformed_features.size() / 2; i += 2, ++j)
   {
      transformed_polygon_features(j, 0) = transformed_features[i];
      transformed_polygon_features(j, 1) = transformed_features[i + 1];
   }

   transformed_s_bar_x = s_message->transformed_barycenter_features[0];
   transformed_s_bar_y = s_message->transformed_barycenter_features[1];

   transformed_first_min_index = s_message->d_transformed;
   transformed_second_min_index = s_message->f_transformed;

   transformed_sigma = s_message->transformed_sigma;
   transformed_sigma_square = s_message->transformed_sigma_square;
   transformed_sigma_square_log = s_message->transformed_sigma_square_log;

   transformed_tangent = s_message->transformed_tangent;
   transformed_angle_radian = s_message->transformed_angle_radian;
   transformed_angle_deg = s_message->transformed_angle_deg;

   opencv_moments.setZero(s_message->moments.size());
   // cout << "opencv_moments before subscription: " << opencv_moments.transpose() << endl;
   for (int i = 0; i < s_message->moments.size(); i++)
   {
      // cout << "i = " << i << endl;
      opencv_moments[i] = s_message->moments[i];
   }
   cout << "opencv_moments after subscription: " << opencv_moments.transpose() << endl;

   cout << "opencv_moments[1]/opencv_moments[0] = " << opencv_moments[1] / opencv_moments[0] << endl;
   cout << "(opencv_moments[1]/opencv_moments[0]-cu)/l = " << (opencv_moments[1] / opencv_moments[0] - cu) / l << endl;

   cout << "opencv_moments[2]/opencv_moments[0] = " << opencv_moments[2] / opencv_moments[0] << endl;
   cout << "(opencv_moments[2]/opencv_moments[0]-cv)/l = " << (opencv_moments[2] / opencv_moments[0] - cv) / l << endl;

   cX = opencv_moments[1] / opencv_moments[0];
   cY = opencv_moments[2] / opencv_moments[0];

   cX_int = (int)cX;
   cY_int = (int)cY;

   cout << "cX = " << cX << endl;
   cout << "cY = " << cY << endl;
   cout << "(cX - cu)/l = " << (cX - cu) / l << endl;
   cout << "(cY - cv)/l = " << (cY - cv) / l << endl;
   cout << "cX_int = " << cX_int << endl;
   cout << "cY_int = " << cY_int << endl;
   cout << "(cX_int - cu)/l = " << (cX_int - cu) / l << endl;
   cout << "(cY_int - cv)/l = " << (cY_int - cv) / l << endl;

   cout << "------------------------------------------------------------------" << endl;
   cout << "------ Transformed features shape and angle of the polygon -------" << endl;
   cout << "------------------------------------------------------------------" << endl;

   cout << "transformed_features: " << transformed_features.transpose() << endl;
   cout << "transformed_polygon_features: " << transformed_polygon_features.transpose() << endl;

   cout << "transformed_s_bar_x: " << transformed_s_bar_x << endl;
   cout << "transformed_s_bar_y: " << transformed_s_bar_y << endl;

   cout << "transformed_first_min_index: " << transformed_first_min_index << endl;
   cout << "transformed_second_min_index: " << transformed_second_min_index << endl;

   cout << "transformed_sigma: " << transformed_sigma << endl;
   cout << "transformed_sigma_square: " << transformed_sigma_square << endl;
   cout << "transformed_sigma_square_log: " << transformed_sigma_square_log << endl;

   cout << "transformed_tangent: " << transformed_tangent << endl;
   cout << "transformed_angle_radian: " << transformed_angle_radian << endl;
   cout << "transformed_angle_deg: " << transformed_angle_deg << endl;

   flag = 1;
   // cout << "Feature callback flag: " << flag << endl;
}

//****UPDATE ALTITUDE****//
void altitudeCallback(const std_msgs::Float64::ConstPtr &alt_message)
{

   Z0 = alt_message->data;
   Z1 = alt_message->data;
   Z2 = alt_message->data;
   Z3 = alt_message->data;
   flag = 1;
   // cout << "Altitude callback flag: " << flag << endl;
   // printf("Relative altitude is (%g,%g,%g,%g) =", Z0, Z1, Z2, Z3);
}

//****MAIN****//
int main(int argc, char **argv)
{
   ros::init(argc, argv, "mpc");
   ros::NodeHandle nh;
   ros::Rate loop_rate(10.0);

   // Create publishers
   // ros::Subscriber feature_sub = nh.subscribe<img_seg_cnn::PREDdata>("/pred_data", 10, featureCallback);
   ros::Subscriber feature_sub_poly_custom = nh.subscribe<img_seg_cnn::POLYcalc_custom>("/polycalc_custom", 10, featureCallback_poly_custom);
   ros::Subscriber feature_sub_poly_custom_tf = nh.subscribe<img_seg_cnn::POLYcalc_custom_tf>("/polycalc_custom_tf", 10, featureCallback_poly_custom_tf);
   ros::Subscriber alt_sub = nh.subscribe<std_msgs::Float64>("/mavros/global_position/rel_alt", 10, altitudeCallback);

   // Create subscribers
   ros::Publisher vel_pub = nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);
   ros::Publisher rec_pub = nh.advertise<vsc_nmpc_uav_target_tracking::rec>("/vsc_nmpc_uav_target_tracking/msg/rec", 1);

   // Initialize MPC Variables
   s_des.setZero(dim_s, mpc_hrz + 1);
   s_abs.setZero(dim_s);
   s_abs << umax - cu, vmax - cv, sigma_constraints_square_log, angle_deg_constraint_tan;

   //****SET MPC COST FUNCTION MATRICES****//
   Q.setIdentity(dim_s, dim_s);
   R.setIdentity(dim_inputs, dim_inputs);
   P.setIdentity(dim_s, dim_s);

   Q = 10 * Q;
   R = 5 * R;
   P = 1 * Q;

   // cout << "Q shape: (" << Q.rows() << "," << Q.cols() << ")" << endl;
   // cout << "R shape: (" << R.rows() << "," << R.cols() << ")" << endl;
   // cout << "P shape: (" << P.rows() << "," << P.cols() << ")" << endl;

   Q(0, 0) = 650;
   Q(1, 1) = 650;
   Q(2, 2) = 1;   // 1.0;
   Q(3, 3) = 180; // 180
   //    Q(4,4) = 1;

   R(0, 0) = 5;   // 5;
   R(1, 1) = 10;  // 10;
   R(2, 2) = 300; /// 750;
   R(3, 3) = 5;   // 18;

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
   nlopt_opt opt;
   opt = nlopt_create(NLOPT_LN_BOBYQA, dim_inputs * mpc_hrz); // algorithm and dimensionality
   nlopt_set_lower_bounds(opt, inputs_lb);
   nlopt_set_upper_bounds(opt, inputs_ub);
   nlopt_set_min_objective(opt, costFunction, NULL);
   // nlopt_set_min_objective(opt, costFunction_alter, NULL);
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
   nlopt_add_inequality_mconstraint(opt, dim_s * (mpc_hrz + 1), constraints, NULL, constraints_tol);
   // nlopt_add_inequality_constraint(opt, myconstraint, &data[0], 1e-8);
   // nlopt_add_inequality_constraint(opt, myconstraint, &data[1], 1e-8);

   // nlopt_set_xtol_rel(opt, 1e-4);

   //****INITIALIZE INPUT VECTOR****//
   double inputs[dim_inputs * mpc_hrz]; // some initial guess
   for (int k = 0; k < dim_inputs * mpc_hrz; k++)
   {
      inputs[k] = 0.0;
   }

   //****DEFINE COST FUNCTION VARIABLE****//
   double minJ; // the minimum objective value, upon return

//****INITIALIZE TIME VARIABLES****//
startlabel:
   double t0 = ros::WallTime::now().toSec();
   // printf("Start time:%lf\n", t0);
   double realtime = 0;

   //****RUNNING LOOP****//
   while (ros::ok())
   {
      if (s_bar_x != 0 && s_bar_y != 0)
      {
         double start = ros::Time::now().toSec();
         // printf("Start time:%lf\n", start);

         // ****EXECUTE OPTIMIZATION****//
         if (flag)
         {
            optNum = nlopt_optimize(opt, inputs, &minJ);
            // cout << "Optimization Return Code: " << nlopt_optimize(opt, inputs, &minJ) << endl;
            // cout << "Optimization Return Code: " << optNum << endl;
         }
         printf("found minimum at J(%g,%g,%g,%g) = %g\n", inputs[0], inputs[1], inputs[2], inputs[3], minJ);

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
         // printf("Inputs are (%g,%g,%g,%g)", VelTrans(caminputs)(0,0), VelTrans(caminputs)(1,0), VelTrans(caminputs)(2,0),VelTrans(caminputs)(5,0));

         dataMsg.coordinate_frame = 8;
         dataMsg.type_mask = 1479;
         dataMsg.header.stamp = ros::Time::now();

         Tx = VelTrans1(VelTrans(caminputs))(0, 0);
         Ty = VelTrans1(VelTrans(caminputs))(1, 0);
         Tz = VelTrans1(VelTrans(caminputs))(2, 0);
         Oz = VelTrans1(VelTrans(caminputs))(5, 0);

         // dataMsg.velocity.x = (1.0)*Tx+1.0;
         // dataMsg.velocity.y = (3.8)*Ty;
         // dataMsg.velocity.z = (1.0)*Tz;
         // dataMsg.yaw_rate = (12.0)*Oz;

         // dataMsg.velocity.x = (1.0) * Tx + 2.5;
         // dataMsg.velocity.x = 0.0;
         // dataMsg.velocity.y = (0.8) * Ty;
         // dataMsg.velocity.z = (1.0) * Tz;
         // dataMsg.yaw_rate = (2.5) * Oz;

         // dataMsg.velocity.x = (1.0) * Tx + 1.5;
         // dataMsg.velocity.x = 0.0;
         // dataMsg.velocity.y = (0.5) * Ty;
         // dataMsg.velocity.z = (1.0) * Tz;
         // dataMsg.yaw_rate = (0.5) * Oz;

         dataMsg.velocity.x = 0.0;
         dataMsg.velocity.y = 0.0;
         dataMsg.velocity.z = 0.0;
         dataMsg.yaw_rate = 0.0;

         // if (Tx >= 0.5)
         // {
         //    Tx = 0.5;
         // }
         // if (Tx <= -0.5)
         // {
         //    Tx = -0.5;
         // }
         // if (Ty >= 0.5)
         // {
         //    Ty = 0.4;
         // }
         // if (Ty <= -0.4)
         // {
         //    Ty = -0.4;
         // }
         // if (Oz >= 0.3)
         // {
         //    Oz = 0.2;
         // }
         // if (Oz <= -0.3)
         // {
         //    Oz = -0.2;
         // }
         // printf("Camera Velocity is u,v,z,Oz(%g,%g,%g,%g)", inputs[0], inputs[1], inputs[2], inputs[3]);
         // cout << "\n" << endl;
         // printf("Drone Velocities Tx,Ty,Tz,Oz(%g,%g,%g,%g)", Tx, Ty, Tz, Oz);
         // cout << "\n"
         //      << endl;

         //****SAVE DATA****//
         vsc_nmpc_uav_target_tracking::rec fdataMsg;

         fdataMsg.J = minJ;
         // fdataMsg.optNUM = nlopt_optimize(opt, inputs, &minJ);
         fdataMsg.optNUM = optNum;
         fdataMsg.Z = Z0;

         fdataMsg.Tx = Tx;
         fdataMsg.Ty = Ty;
         fdataMsg.Tz = Tz;
         fdataMsg.Oz = Oz;

         // fdataMsg.transformed_features = transformed_features;

         // fdataMsg.first_min_index = first_min_index;
         // fdataMsg.second_min_index = second_min_index;

         // fdataMsg.s_bar_x = s_bar_x;
         // fdataMsg.s_bar_y = s_bar_y;
         // fdataMsg.s_bar_x_des = s_bar_x_des;
         // fdataMsg.transformed_sigma = transformed_sigma;
         // fdataMsg.transformed_sigma_square = transformed_sigma_square;
         // fdataMsg.transformed_sigma_square_log = transformed_sigma_square_log;
         // fdataMsg.sigma_des = sigma_des;
         // fdataMsg.sigma_square_des = sigma_square_des;
         // fdataMsg.sigma_log_des = sigma_log_des;

         // fdataMsg.angle_deg_des = angle_deg_des;
         // fdataMsg.angle_des_tan = angle_des_tan;
         // fdataMsg.transformed_tangent = transformed_tangent;
         // fdataMsg.transformed_angle_radian = transformed_angle_radian;
         // fdataMsg.transformed_angle_deg = transformed_angle_deg;

         fdataMsg.traj_s_1 = stored_traj_s[0];
         fdataMsg.traj_s_2 = stored_traj_s[1];
         fdataMsg.traj_s_3 = stored_traj_s[2];
         fdataMsg.traj_s_4 = stored_traj_s[3];

         fdataMsg.s_des_1 = stored_s_des[0];
         fdataMsg.s_des_2 = stored_s_des[1];
         fdataMsg.s_des_3 = stored_s_des[2];
         fdataMsg.s_des_4 = stored_s_des[3];

         fdataMsg.time = timer;
         fdataMsg.dtloop = dt;

         printf("Drone Velocities Tx,Ty,Tz,Oz(%g,%g,%g,%g)", fdataMsg.Tx, fdataMsg.Ty, fdataMsg.Tz, fdataMsg.Oz);
         cout << "\n"
              << endl;

         rec_pub.publish(fdataMsg);
         vel_pub.publish(dataMsg);
      }
      ros::spinOnce();
      // ros::spin;
      loop_rate.sleep();
   }

   nlopt_destroy(opt);
   return 0;
}
