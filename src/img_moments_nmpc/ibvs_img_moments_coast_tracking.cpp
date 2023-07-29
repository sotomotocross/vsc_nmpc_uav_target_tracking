#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "mavros_msgs/PositionTarget.h"
#include "img_seg_cnn/PREDdata.h"
#include "img_seg_cnn/POLYcalc_custom.h"
#include "img_seg_cnn/POLYcalc_custom_tf.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"
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

int features_n = 1;
const int dim_inputs = 4;
int dim_s = 4;
double l = 252.07;
double umax = 720;
double umin = 0;
double vmax = 480;
double vmin = 0;
double cu = 360.5;
double cv = 240.5;

MatrixXd gains;
VectorXd feature_vector;
VectorXd transformed_features;
VectorXd opencv_moments;
MatrixXd polygon_features;
MatrixXd transformed_polygon_features;

VectorXd feat_u_vector;
VectorXd feat_v_vector;
VectorXd feat_vector;

VectorXd state_vector(dim_s);
VectorXd state_vector_des(dim_s);
VectorXd cmd_vel(dim_inputs);
VectorXd error(dim_s);
VectorXd barx_meas(dim_s);
VectorXd barx_des(dim_s);

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

// int flag = 0;

int flag;

MatrixXd img_moments_system(VectorXd moments)
{
   MatrixXd model_mat(dim_s, dim_inputs);
   MatrixXd Le(dim_s, dim_inputs);
   Le.setZero(dim_s, dim_inputs);

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

   MatrixXd Int_matrix(dim_s, dim_inputs);
   Int_matrix << -1 / Z0, 0, (xg / Z0) + 4 * (A * n20 + B * n11), yg,
       0, -1 / Z0, (yg / Z0) + 4 * (A * n11 + B * n02), -xg,
       -area * A, -area * B, area * ((3 / Z0) - C), 0,
       angle_ux, angle_uy, angle_uz, -1;

   gains.setIdentity(dim_s, dim_inputs);

   gains(0, 0) = 1.0;
   gains(1, 1) = 0.1;
   gains(2, 2) = 0.1;
   gains(3, 3) = 0.1;

   // cout << "Int_matrix shape: (" << Int_matrix.rows() << "," << Int_matrix.cols() << ")" << endl;
   // cout << "Int_matrix = \n"
   // << Int_matrix << endl;
   // cout << "gains * Int_matrix = \n" << gains * Int_matrix << endl;

   return gains * Int_matrix;
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

   // cX = opencv_moments[1]/opencv_moments[0];
   // cY = opencv_moments[2]/opencv_moments[0];

   // cX_int = (int)cX;
   // cY_int = (int)cY;

   // cout << "cX = " << cX << endl;
   // cout << "cY = " << cY << endl;
   // cout << "(cX - cu)/l = " << (cX - cu)/l << endl;
   // cout << "(cY - cv)/l = " << (cY - cv)/l<< endl;
   // cout << "cX_int = " << cX_int << endl;
   // cout << "cY_int = " << cY_int << endl;
   // cout << "(cX_int - cu)/l = " << (cX_int - cu)/l << endl;
   // cout << "(cY_int - cv)/l = " << (cY_int - cv)/l << endl;

   // cout << "------------------------------------------------------------------" << endl;
   // cout << "------ Transformed features shape and angle of the polygon -------" << endl;
   // cout << "------------------------------------------------------------------" << endl;

   // cout << "transformed_features: " << transformed_features.transpose() << endl;
   // cout << "transformed_polygon_features: " << transformed_polygon_features.transpose() << endl;

   // cout << "transformed_s_bar_x: " << transformed_s_bar_x << endl;
   // cout << "transformed_s_bar_y: " << transformed_s_bar_y << endl;

   // cout << "transformed_first_min_index: " << transformed_first_min_index << endl;
   // cout << "transformed_second_min_index: " << transformed_second_min_index << endl;

   // cout << "transformed_sigma: " << transformed_sigma << endl;
   // cout << "transformed_sigma_square: " << transformed_sigma_square << endl;
   // cout << "transformed_sigma_square_log: " << transformed_sigma_square_log << endl;

   // cout << "transformed_tangent: " << transformed_tangent << endl;
   // cout << "transformed_angle_radian: " << transformed_angle_radian << endl;
   // cout << "transformed_angle_deg: " << transformed_angle_deg << endl;

   // cout << "opencv_moments after subscription: " << opencv_moments.transpose() << endl;
   // cout << "opencv_moments[1]/opencv_moments[0] = " << opencv_moments[1]/opencv_moments[0] << endl;
   // cout << "(opencv_moments[1]/opencv_moments[0]-cu)/l = " << (opencv_moments[1]/opencv_moments[0]-cu)/l << endl;
   // cout << "opencv_moments[2]/opencv_moments[0] = " << opencv_moments[2]/opencv_moments[0] << endl;
   // cout << "(opencv_moments[2]/opencv_moments[0]-cv)/l = " << (opencv_moments[2]/opencv_moments[0]-cv)/l << endl;

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
   ros::Subscriber feature_sub_poly_custom = nh.subscribe<img_seg_cnn::POLYcalc_custom>("/polycalc_custom", 10, featureCallback_poly_custom);
   ros::Subscriber feature_sub_poly_custom_tf = nh.subscribe<img_seg_cnn::POLYcalc_custom_tf>("/polycalc_custom_tf", 10, featureCallback_poly_custom_tf);
   ros::Subscriber alt_sub = nh.subscribe<std_msgs::Float64>("/mavros/global_position/rel_alt", 10, altitudeCallback);

   // Create publishers
   ros::Publisher vel_pub = nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);
   ros::Publisher rec_pub = nh.advertise<vsc_nmpc_uav_target_tracking::rec>("/vsc_nmpc_uav_target_tracking/msg/rec", 1);
   ros::Publisher cmd_vel_pub = nh.advertise<std_msgs::Float64MultiArray>("/cmd_vel", 1);
   ros::Publisher state_vec_pub = nh.advertise<std_msgs::Float64MultiArray>("/state_vec", 1);
   ros::Publisher state_vec_des_pub = nh.advertise<std_msgs::Float64MultiArray>("/state_vec_des", 1);
   ros::Publisher img_moments_error_pub = nh.advertise<std_msgs::Float64MultiArray>("/img_moments_error", 1);
   ros::Publisher moments_pub = nh.advertise<std_msgs::Float64MultiArray>("/moments", 1);
   ros::Publisher central_moments_pub = nh.advertise<std_msgs::Float64MultiArray>("/central_moments", 1);

   //****INITIALIZE INPUT VECTOR****//
   double inputs[dim_inputs]; // some initial guess
   for (int k = 0; k < dim_inputs; k++)
   {
      inputs[k] = 0.0;
   }

   //****RUNNING LOOP****//
   while (ros::ok())
   {

      double start = ros::Time::now().toSec();
      // printf("Start time:%lf\n", start);
      // cout << "flag: " << flag << endl;

      // ****EXECUTE OPTIMIZATION****//
      if (flag == 1)
      {
         // cout << "flag: " << flag << endl;
         MatrixXd IM = img_moments_system(opencv_moments);
         // cout << "IM: " << IM << endl;

         state_vector << ((opencv_moments[1] / opencv_moments[0]) - cu) / l, ((opencv_moments[2] / opencv_moments[0]) - cv) / l, log(sqrt(opencv_moments[0])), atan(2 * opencv_moments[11] / (opencv_moments[10] - opencv_moments[12]));
         state_vector_des << 0.0, 0.0, 5.0, angle_des_tan;

         cout << "state_vector = " << state_vector.transpose() << endl;
         cout << "state_vector_des = " << state_vector_des.transpose() << endl;
         // cout << "barx_meas = " << barx_meas.transpose() << endl;
         // cout << "barx_des = " << barx_des.transpose() << endl;

         error.setZero(6);

         error = state_vector - state_vector_des;
         cout << "error = " << error.transpose() << endl;

         // cout << "error shape: (" << error.rows() << "," << error.cols() << ")" << endl;
         // cout << "IM shape: (" << IM.rows() << "," << IM.cols() << ")" << endl;

         MatrixXd pinv = IM.completeOrthogonalDecomposition().pseudoInverse();
         // cout << "pinv: " << pinv << endl;
         // cout << "pinv shape: (" << pinv.rows() << "," << pinv.cols() << ")" << endl;
         VectorXd velocities = pinv * error;
         // cout << "velocities = " << velocities << endl;
         // cout << "velocities shape: (" << velocities.rows() << "," << velocities.cols() << ")" << endl;
         cmd_vel = pinv * error;
         // inputs = pinv * error;
      }
      // cout << "inputs = " << inputs << endl;
      VectorXd tmp_cmd_vel(dim_inputs);
      tmp_cmd_vel << cmd_vel[0], cmd_vel[1], cmd_vel[2], cmd_vel[3];
      // cout << "tmp_cmd_vel = " << tmp_cmd_vel.transpose() << endl;
      cout << "cmd_vel: " << cmd_vel.transpose() << endl;

      //****SEND VELOCITIES TO AUTOPILOT THROUGH MAVROS****//
      mavros_msgs::PositionTarget dataMsg;
      Matrix<double, 4, 1> caminputs;
      caminputs(0, 0) = cmd_vel[0];
      caminputs(1, 0) = cmd_vel[1];
      caminputs(2, 0) = cmd_vel[2];
      caminputs(3, 0) = cmd_vel[3];

      dataMsg.coordinate_frame = 8;
      dataMsg.type_mask = 1479;
      dataMsg.header.stamp = ros::Time::now();

      Tx = VelTrans1(VelTrans(caminputs))(0, 0);
      Ty = VelTrans1(VelTrans(caminputs))(1, 0);
      Tz = VelTrans1(VelTrans(caminputs))(2, 0);
      Oz = VelTrans1(VelTrans(caminputs))(5, 0);

      double gain_tx;
      nh.getParam("/gain_tx", gain_tx);
      double gain_ty;
      nh.getParam("/gain_ty", gain_ty);
      double gain_tz;
      nh.getParam("/gain_tz", gain_tz);
      double gain_yaw;
      nh.getParam("/gain_yaw", gain_yaw);

      // Τracking tuning
      dataMsg.velocity.x = gain_tx * Tx + 0.08;
      dataMsg.velocity.y = gain_ty * Ty;
      dataMsg.velocity.z = gain_tz * Tz;
      dataMsg.yaw_rate = gain_yaw * Oz;

      // Stabilization tuning
      // dataMsg.velocity.x = 0.0;
      // dataMsg.velocity.y = (-0.025) * Ty;
      // dataMsg.velocity.z = (0.0008) * Tz;
      // dataMsg.yaw_rate = (-0.015) * Oz;

      printf("Drone Velocities Tx,Ty,Tz,Oz(%g,%g,%g,%g)", dataMsg.velocity.x, dataMsg.velocity.y, dataMsg.velocity.z, dataMsg.yaw_rate);
      cout << "\n"
           << endl;

      std_msgs::Float64MultiArray cmd_vel_Msg;
      for (int i = 0; i < cmd_vel.size(); i++)
      {
         cmd_vel_Msg.data.push_back(cmd_vel[i]);
      }

      std_msgs::Float64MultiArray state_vecMsg;
      for (int i = 0; i < state_vector.size(); i++)
      {
         state_vecMsg.data.push_back(state_vector[i]);
      }

      std_msgs::Float64MultiArray state_vec_desMsg;
      for (int i = 0; i < state_vector_des.size(); i++)
      {
         state_vec_desMsg.data.push_back(state_vector_des[i]);
      }

      std_msgs::Float64MultiArray error_Msg;
      for (int i = 0; i < error.size(); i++)
      {
         error_Msg.data.push_back(error[i]);
      }

      cmd_vel_pub.publish(cmd_vel_Msg);
      state_vec_pub.publish(state_vecMsg);
      state_vec_des_pub.publish(state_vec_desMsg);
      img_moments_error_pub.publish(error_Msg);
      vel_pub.publish(dataMsg);

      ros::spinOnce();
      // ros::spin;
      loop_rate.sleep();
   }

   return 0;
}
