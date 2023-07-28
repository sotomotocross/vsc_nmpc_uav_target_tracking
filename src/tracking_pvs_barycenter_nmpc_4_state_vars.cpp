#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "mavros_msgs/PositionTarget.h"
#include "img_seg_cnn/PREDdata.h"
#include "img_seg_cnn/POLYcalc_custom.h"
#include "img_seg_cnn/POLYcalc_custom_tf.h"
#include "std_msgs/Float64.h"
#include "mpcpack/rec.h"

#include <cstdlib>
#include <iostream>
#include <stdio.h>

#include <math.h>
#include <cmath>
#include <nlopt.hpp>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

// Global MPC Variables
//  int features_n = 4;
int features_n = 1;
const int dim_inputs = 4;
// const int dim_inputs = 6;
int dim_s = 4;
// int dim_s = 2*features_n;
const int mpc_hrz = 10;  // 10
double mpc_dt = 0.001; //0.001
double l = 252.07;
double a = 10;
double optNum;

MatrixXd s_des(dim_s, mpc_hrz + 1);
// VectorXd s_des(dim_s);
VectorXd s_ub(dim_s);
VectorXd s_lb(dim_s);
VectorXd s_abs(dim_s);
VectorXd s_bc;
VectorXd s_br;
MatrixXd Q;
MatrixXd R;
MatrixXd P;
VectorXd t;
VectorXd ek;
VectorXd ek_1;
VectorXd feature_vector;
VectorXd transformed_features;
VectorXd moments;
MatrixXd polygon_features;
MatrixXd transformed_polygon_features;

// Simulator camera parameters
double umax = 720;
double umin = 0;
double vmax = 480;
double vmin = 0;
double cu = 360.5;
double cv = 240.5;

// ZED 2 stereocamera parameters
// double umax = 672;
// double umin = 0;
// double vmax = 376;
// double vmin = 0;
// // double cu = 0.5*(umax+umin);
// // double cv = 0.5*(vmax+vmin);
// double cu = 336.45;
// double cv = 190.73;

int flag = 0;

// Camera Frame Update Callback Variables
double Z0, Z1, Z2, Z3;
double Tx,Ty,Tz,Oz;

double s_bar_x, s_bar_y;
double first_min_index, second_min_index;
double custom_sigma, custom_sigma_square, custom_sigma_square_log;
double angle_tangent, angle_radian, angle_deg;

double transformed_s_bar_x, transformed_s_bar_y;
double transformed_first_min_index, transformed_second_min_index;
double transformed_sigma, transformed_sigma_square, transformed_sigma_square_log;
double transformed_tangent, transformed_angle_radian, transformed_angle_deg;

double s_bar_x_des = Z0*(cu-cu/l);
double s_bar_y_des = Z0*(cv-cv/l);

// double sigma_des = 18500.0;
double sigma_des = 18.5;
double sigma_square_des = sqrt(sigma_des);
double sigma_log_des = log(sigma_square_des);

double angle_deg_des = 0;
double angle_des_tan = tan((angle_deg_des/180) * 3.14);

// double sigma_constraints = 25000.0;
double sigma_constraints = 25.0;
double sigma_constraints_square = sqrt(sigma_constraints);
double sigma_constraints_square_log = log(sigma_constraints_square);

double angle_deg_constraint = 45;
double angle_deg_constraint_tan = tan((angle_deg_constraint/180) * 3.14);


VectorXd Dynamic_System(VectorXd camTwist, VectorXd feat_prop)
{
   // cout << "Mpike ki edo (PVS) to gamidi!!!" << endl;
   MatrixXd model_mat(dim_s, dim_inputs);
   // cout << "model_mat shape: (" << model_mat.rows() << "," << model_mat.cols() << ")" << endl;

   // cout << "feat prop inside Dynamic system: " << feat_prop << endl;

   // Barycenter dynamics calculation
   double term_1_4 = 0.0;
   double term_1_5 = 0.0;
   double term_2_4 = 0.0;
   double term_2_5 = 0.0;

   int N;
   N = transformed_features.size()/2;
   // cout << "feat_prop: \n" << feat_prop << endl;
   // cout << "feat_prop[0]: " << feat_prop[0] << endl;
   // cout << "transformed_features: \n" << transformed_features << endl;
   // cout << "transformed_features[0]: " << transformed_features[0] << endl;
   // cout << "length feature vector: " << transformed_features.size() << "\n" << endl;
   // cout << "length feat_prop vector: " << feat_prop.size() << "\n" << endl;

   for (int i = 0; i < N - 1; i += 2)
   {
      // cout << "1st Vector Index: " << i << "\n" << endl;
      // cout << "feat_prop[i]*feat_prop[i+1]: " << feat_prop[i]*feat_prop[i+1] << "\n" << endl;
      term_1_4 = term_1_4 + feat_prop[i] * feat_prop[i + 1];
      term_1_5 = term_1_5 + (1 + pow(feat_prop[i], 2));
      term_2_4 = term_2_4 + (1 + pow(feat_prop[i + 1], 2));
      term_2_5 = term_2_5 + feat_prop[i] * feat_prop[i + 1];
   }

   term_1_4 = term_1_4/N;
   term_1_5 = -term_1_5/N;
   term_2_4 = term_2_4/N;
   term_2_5 = -term_2_5/N;

   double g_4_4, g_4_5, g_4_6;

   // Angle dynamics calculation
   // Fourth term
   double term_4_4_1, term_4_4_2, term_4_4_3, term_4_4_4;
   double sum_4_4_1 = 0.0, sum_4_4_2 = 0.0;

   double k = 0;
   VectorXd x(N);
   VectorXd y(N);

   // cout << "2N: " << 2*N << endl;
   // cout << "feat_prop: " << feat_prop.transpose() << endl;

   for (int i = 0; i < 2*N - 1; i += 2)
   {
      // cout << "index for x: " << i << endl;
      x[k] = feat_prop[i];
      k++;
   }

   k = 0;

   for (int i = 1; i < 2*N ; i += 2)
   {
      // cout << "index for y: " << i << endl;
      y[k] = feat_prop[i];
      k++;
   }

   // cout << "x: " << x.transpose() << endl;
   // cout << "y: " << y.transpose() << endl;

   for (int i = 0; i < N - 1; i += 2)
   {
      // cout << "2nd Vector Index: " << i << "\n" << endl;
      // cout << "feat_prop[i]*feat_prop[i+1]: " << feat_prop[i]*feat_prop[i+1] << "\n" << endl;
      sum_4_4_1 = sum_4_4_1 + pow(feat_prop[i + 1], 2);
      sum_4_4_2 = sum_4_4_2 + feat_prop[i] * feat_prop[i + 1];
   }

   term_4_4_1 = 1 / (x[transformed_first_min_index] + x[transformed_second_min_index] - 2 * transformed_s_bar_x);
   term_4_4_2 = (pow(y[transformed_first_min_index], 2) + pow(y[transformed_second_min_index], 2) - (2 / N) * sum_4_4_1);
   term_4_4_3 = -transformed_tangent / (x[transformed_first_min_index] + x[transformed_second_min_index] - 2 * transformed_s_bar_x);
   term_4_4_4 = (x[transformed_first_min_index] * y[transformed_first_min_index] + x[transformed_second_min_index] * y[transformed_second_min_index] - (2 / N) * sum_4_4_2);

   g_4_4 = term_4_4_1 * term_4_4_2 + term_4_4_3 * term_4_4_4;
   // cout << "g_4_4: " << g_4_4 << endl; 

   // Fifth term
   double term_4_5_1, term_4_5_2, term_4_5_3, term_4_5_4;
   double sum_4_5_1 = 0.0, sum_4_5_2 = 0.0;

   for (int i = 0; i < N - 1; i += 2)
   {
      // cout << "3rd Vector Index: " << i << "\n" << endl;
      // cout << "feat_prop[i]*feat_prop[i+1]: " << feat_prop[i]*feat_prop[i+1] << "\n" << endl;
      sum_4_5_1 = sum_4_5_1 + pow(feat_prop[i], 2);
      sum_4_5_2 = sum_4_5_2 + feat_prop[i] * feat_prop[i + 1];
   }

   term_4_5_1 = transformed_tangent / (x[transformed_first_min_index] + x[transformed_second_min_index] - 2 * transformed_s_bar_x);
   term_4_5_2 = (pow(x[transformed_first_min_index], 2) + pow(x[transformed_second_min_index], 2) - (2 / N) * sum_4_5_1);
   term_4_5_3 = -1 / (x[transformed_first_min_index] + x[transformed_second_min_index] - 2 * transformed_s_bar_x);
   term_4_5_4 = (x[transformed_first_min_index] * y[transformed_first_min_index] + x[transformed_second_min_index] * y[transformed_second_min_index] - (2 / N) * sum_4_5_2);

   g_4_5 = term_4_5_1 * term_4_5_2 + term_4_5_3 * term_4_5_4;
   // cout << "g_4_5: " << g_4_5 << endl; 

   // Fifth term
   g_4_6 = -pow(transformed_tangent, 2) - 1;
   // cout << "g_4_6: " << g_4_6 << endl; 
   
   // MatrixXd full_model_mat(dim_s, 6);
   // full_model_mat << 
   //          -1/Z0, 0.0, transformed_s_bar_x/Z0, term_1_4, term_1_5, transformed_s_bar_y,
   //          0.0, -1/Z0, transformed_s_bar_y/Z0, term_2_4, term_2_5, -transformed_s_bar_x,
   //          0.0, 0.0, 2/Z0, (3/2)*transformed_s_bar_y, -(3/2)*transformed_s_bar_x, 0.0,
   //          0.0, 0.0, 0.0, g_4_4, g_4_5, g_4_6,
   //          0.0, 0.0, 1.0, 0.0, 0.0, 0.0;

   // cout << "Full model matrix:\n" << full_model_mat << endl;

   model_mat << 
            -1/Z0, 0.0, transformed_s_bar_x/Z0, transformed_s_bar_y,
            0.0, -1/Z0, transformed_s_bar_y/Z0, -transformed_s_bar_x,
            0.0, 0.0, 2/Z0, 0.0,
            0.0, 0.0, 0.0, g_4_6,
            0.0, 0.0, 1.0, 0.0;
   // model_mat << 
   //          -1/Z0, 0.0, transformed_s_bar_x/Z0, transformed_s_bar_y,
   //          0.0, -1/Z0, transformed_s_bar_y/Z0, -transformed_s_bar_x,
   //          0.0, 0.0, 2/Z0, 0.0,
   //          0.0, 0.0, 0.0, 0.0,
   //          0.0, 0.0, 1.0, 0.0;

   // cout << "model_mat: " << model_mat << endl;

   // cout << "model_mat shape: (" << model_mat.rows() << "," << model_mat.cols() << ")" << endl;
   // cout << "Interaction Matrix:\n" << model_mat << endl;
   // cout << "Interaction Matrix" << model_mat * camTwist << endl;

   return model_mat * camTwist;
}


VectorXd Dynamic_System_x_y_reverted(VectorXd camTwist, VectorXd feat_prop)
{
   cout << "Mpike ki edo (PVS) to gamidi!!!" << endl;
   MatrixXd model_mat(dim_s, dim_inputs);
//    cout << "model_mat shape: (" << model_mat.rows() << "," << model_mat.cols() << ")" << endl;

//    cout << "feat prop inside Dynamic system: " << feat_prop << endl;

   // Barycenter dynamics calculation
   double term_1_4 = 0.0;
   double term_1_5 = 0.0;
   double term_2_4 = 0.0;
   double term_2_5 = 0.0;

   int N;
   N = transformed_features.size()/2;
//    cout << "feat_prop: \n" << feat_prop << endl;
//    cout << "feat_prop[0]: " << feat_prop[0] << endl;
   cout << "transformed_features: \n" << transformed_features.transpose() << endl;
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

   term_1_4 = term_1_4/N;
   term_1_5 = -term_1_5/N;
   term_2_4 = term_2_4/N;
   term_2_5 = -term_2_5/N;

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

   for (int i = 0; i < 2*N - 1; i += 2)
   {
    //   cout << "index for x: " << i << endl;
      x[k] = feat_prop[i];
      k++;
   }

   k = 0;

   for (int i = 1; i < 2*N ; i += 2)
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

    model_mat << 
            -1/Z0, 0.0, transformed_s_bar_x/Z0, transformed_s_bar_y,
            0.0, -1/Z0, transformed_s_bar_y/Z0, -transformed_s_bar_x,
            0.0, 0.0, 2/Z0, 0.0,
            0.0, 0.0, 0.0, g_4_6;

   // model_mat << 
   //          -1/Z0, 0.0, transformed_s_bar_x/Z0, transformed_s_bar_y,
   //          0.0, -1/Z0, transformed_s_bar_y/Z0, -transformed_s_bar_x,
   //          0.0, 0.0, 2/Z0, 0.0,
   //          0.0, 0.0, 0.0, 0.0,
   //          0.0, 0.0, 1.0, 0.0;

//    cout << "model_mat: " << model_mat << endl;

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
   MatrixXd Le(transformed_features.size(),dim_inputs);
   //,Le.setZero(dim_s,dim_inputs);
   Le.setZero(transformed_features.size(),dim_inputs);
   // cout << "Le: \n" << Le << endl;

   // Le.row(0) << -1/Z0, 0.0, transformed_features[0]/Z0, transformed_features[1];
   // Le.row(1) << 0.0, -1/Z0, transformed_features[1]/Z0, -transformed_features[0];

   // cout << "after Le: \n" << Le << endl;

   for (int k = 0, kk = 0; k<transformed_features.size() && kk < transformed_features.size() ; k++, kk++){
      Le.row(k) << -1/Z0, 0.0, transformed_features[kk]/Z0, transformed_features[kk+1];
      Le.row(k+1) << 0.0, -1/Z0, transformed_features[kk+1]/Z0, -transformed_features[kk];
      k++;
      kk++;
   }
   cout << "after Le: \n" << Le << endl;
   // cout << "transformed_features.size(): " << transformed_features.size() << endl;
   cout << "transformed_features: " << transformed_features << endl;
   
   return Le*camTwist;
}

// PVS-MPC Cost Function
double costFunction(unsigned int n, const double *x, double *grad, void *data)
{
   MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz>>((double *)x); 

   // Trajectory of States (image features)
   MatrixXd traj_s(dim_s, mpc_hrz + 1);
   MatrixXd feature_hrz_prop(transformed_features.size(), mpc_hrz + 1);
   traj_s.setZero(dim_s, mpc_hrz + 1);
//    cout << "angle_tangent: " << angle_tangent << endl;
//    cout << "sigma_square: " << sigma_square << endl;
   traj_s.col(0) << transformed_s_bar_x, transformed_s_bar_y, transformed_sigma_square_log, transformed_tangent;
//    cout << "traj_s.col(0): \n" << traj_s.col(0) << endl;
   // cout << "inputs: " << inputs << endl;
   // cout << "length of feature vector: " << transformed_features.size()   
   // VectorXd feat(transformed_features.size());

   // for (int k=0; k<transformed_features.size(); k++){
   //    feat.col(k) << transformed_features(k);
   // }
   // cout << "feat.col(0): " << feat.col(0) << endl;
   // cout << "feat: " << feat << endl;

   // Progate the model (PVS with Image Jacobian)
   for (int k = 0; k < mpc_hrz; k++)
   {
    //   cout << "Mpike to gamidi!!!" << endl;
      VectorXd feat_prop = IBVSSystem(inputs.col(k));
    //   cout << "feat_prop: " << feat_prop << endl;
      feature_hrz_prop.col(k+1) = feature_hrz_prop.col(k) + feat_prop*mpc_dt;
      VectorXd sdot = Dynamic_System_x_y_reverted(inputs.col(k),feature_hrz_prop.col(k));
      // VectorXd sdot = Dynamic_System(inputs.col(k),feature_hrz_prop.col(k));
      // sdot.setZero(dim_s);
    //   cout << "s_dot" << sdot << endl;
      traj_s.col(k + 1) = traj_s.col(k) + sdot * mpc_dt;
    //   cout << "traj_s.col(k + 1) =" << traj_s.col(k + 1) << endl;
   }
//    cout << "traj_s = \n" << traj_s << endl;

   // Calculate Running Costs
   double Ji = 0.0; 

   //****DEFINE INITIAL DESIRED V****//
   // VectorXd s_des(dim_s);
//    cout << "angle_des_tan: " << angle_des_tan << endl;
//    cout << "sigma_square_des: " << sigma_square_des << endl;
   // s_des.col(0) << s_bar_x_des, s_bar_y_des, sigma_log_des, angle_des_tan, Z0;
   s_des.col(0) << s_bar_x_des, s_bar_y_des, sigma_log_des, angle_des_tan;
//    cout << "s_des.col(0): \n" << s_des.col(0) << endl;

   //****SET V DESIRED VELOCITY FOR THE VTA****//
   double b = 0.0;
   VectorXd s_at(dim_s);
   s_at.setZero(dim_s);
//    s_at << b/l, 0, 0, 0, 0;
   s_at << b/l, 0, 0, 0;
   // s_at << 0, 0, 0, 0, 0;
   // cout << "s_at: \n" << s_at << endl;

   //****PROPOGATE THE V DESIRED IN THE HORIZON N FOR dt TIMESTEP SIZE****//
   for (int k = 0; k < mpc_hrz; k++)
   {
      s_des.col(k + 1) = s_des.col(k) + s_at;
   }

   for (int k = 0; k < mpc_hrz; k++)
   {
      // cout << "traj_s.col(k): " << traj_s.col(k) << endl;
      // cout << "s_des.col(k): " << s_des.col(k) << endl;
      ek = -s_des.col(k) + traj_s.col(k);
      // cout << "ek: " << ek << endl ;
      // ek = traj_s.col(k)
      // cout << "traj_s.col(k): " << traj_s.col(k) << endl;
      Ji += ek.transpose() * Q * ek;
      Ji += inputs.col(k).transpose() * R * inputs.col(k);
   }
   // cout << "Ji: \n" << Ji << endl;
   cout << "ek: \n" << ek << endl;

   // Calculate Terminal Costs
   double Jt;
   VectorXd et = traj_s.col(mpc_hrz) - s_des.col(mpc_hrz);
   // cout << "traj_s.col(mpc_hrz): " << traj_s.col(mpc_hrz) << endl;
   cout << "et: \n" << et << endl;
   // cout << "Jt: \n" << Jt << endl;

   Jt = et.transpose() * P * et;
   //   cout << "Ji = " << Ji << " + " << "Jt = " << Jt << endl;

   return Ji + Jt;
}

//****DEFINE FOV CONSTRAINTS****//
void constraints(unsigned int m, double *c, unsigned int n, const double *x, double *grad, void *data)
{
   // Propagate the model.
   MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz>>((double *)x);
   // cout << "inputs = " << inputs << endl;
   // Trajectory of States (image features)
   MatrixXd traj_s(dim_s, mpc_hrz + 1);
   MatrixXd feature_hrz_prop(transformed_features.size(), mpc_hrz + 1);
   traj_s.setZero(dim_s, mpc_hrz + 1);
   traj_s.col(0) << s_bar_x_des, s_bar_y_des, sigma_log_des, angle_des_tan;
   // cout << "traj_s.col(0): " << traj_s.col(0) << endl;

   // Progate the model (IBVS with Image Jacobian)
   for (int k = 0; k < mpc_hrz; k++)
   {
      // cout << "Mpike to gamidi!!!" << endl;
      VectorXd feat_prop = IBVSSystem(inputs.col(k));
      // cout << "feat_prop: " << feat_prop << endl;
      feature_hrz_prop.col(k+1) = feature_hrz_prop.col(k) + feat_prop*mpc_dt;
      VectorXd sdot = Dynamic_System_x_y_reverted(inputs.col(k),feature_hrz_prop.col(k));
      // VectorXd sdot = Dynamic_System(inputs.col(k),feature_hrz_prop.col(k));
      // sdot.setZero(dim_s);
      // cout << "s_dot" << sdot << endl;
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
   
   moments.setZero(s_message->moments.size());
   cout << "moments before subscription: " << moments << endl;
   for (int i =0 ; i < s_message->moments.size(); i++){
      cout << "i = " << i << endl;
      moments[i] = s_message->moments[i];
   }
   cout << "moments after subscription: " << moments << endl;

   // cout << "------------------------------------------------------------------" << endl;
   // cout << "------ Transformed features shape and angle of the polygon -------" << endl;
   // cout << "------------------------------------------------------------------" << endl;

   cout << "transformed_features: " << transformed_features << endl;
   cout << "transformed_polygon_features: " << transformed_polygon_features << endl;

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
   ros::Publisher rec_pub = nh.advertise<mpcpack::rec>("/mpcpack/msg/rec", 1);

   // Initialize MPC Variables
   s_des.setZero(dim_s, mpc_hrz + 1);
   s_abs.setZero(dim_s);  
   s_abs << umax-cu, vmax-cv, sigma_constraints_square_log, angle_deg_constraint_tan;
   

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

   Q(0,0) = 650;
	Q(1,1) = 650;
	Q(2,2) = 1;//1.0;
   Q(3,3) = 180; //180
//    Q(4,4) = 1;

   R(0,0) = 5;//5;
   R(1,1) = 10;//10;
	R(2,2) = 300;///750;
	R(3,3) = 5;//18;

   P(0,0) = 1;
   P(1,1) = 1;
   P(2,2) = 1;
   P(3,3) = 18;
//    P(4,4) = 1;

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
         printf("found minimum at J(%g,%g,%g,%g) = %g\n", inputs[0], inputs[1],inputs[2], inputs[3], minJ);

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
         // printf("Inputs are (%g,%g,%g,%g) =", VelTrans(caminputs)(0,0), VelTrans(caminputs)(1,0), VelTrans(caminputs)(2,0),VelTrans(caminputs)(5,0));

         dataMsg.coordinate_frame = 8;
         dataMsg.type_mask = 1479;
         dataMsg.header.stamp = ros::Time::now();
         Tx = dataMsg.velocity.x = VelTrans1(VelTrans(caminputs))(0, 0)+1.5;
         // Tx = dataMsg.velocity.x = 0.0;
         Ty = dataMsg.velocity.y = VelTrans1(VelTrans(caminputs))(1, 0);
         // Ty = dataMsg.velocity.y = 0.0;
         Tz = dataMsg.velocity.z = VelTrans1(VelTrans(caminputs))(2, 0);
         // Tz = dataMsg.velocity.z = 0.0;
         Oz = dataMsg.yaw_rate = VelTrans1(VelTrans(caminputs))(5, 0);
         // Oz = dataMsg.yaw_rate = 0.0;

         if (Tx >= 0.5)
         {
            Tx = 0.5;
         }
         if (Tx <= -0.5)
         {
            Tx = -0.5;
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
         printf("Camera Velocity is u,v,z,Oz(%g,%g,%g,%g) =", inputs[0], inputs[1], inputs[2], inputs[3]);
         cout << "\n" << endl;
         printf("Drone Velocities Tx,Ty,Tz,Oz(%g,%g,%g,%g) =", Tx, Ty, Tz, Oz);
         cout << "\n" << endl;

         //****SAVE DATA****//
         mpcpack::rec fdataMsg;

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
         
         fdataMsg.s_bar_x = s_bar_x;
         fdataMsg.s_bar_y = s_bar_y;
         fdataMsg.s_bar_x_des = s_bar_x_des;
         fdataMsg.transformed_sigma = transformed_sigma;
         fdataMsg.transformed_sigma_square = transformed_sigma_square;
         fdataMsg.transformed_sigma_square_log = transformed_sigma_square_log;
         fdataMsg.sigma_des = sigma_des;
         fdataMsg.sigma_square_des = sigma_square_des;
         fdataMsg.sigma_log_des = sigma_log_des;

         fdataMsg.angle_deg_des = angle_deg_des;
         fdataMsg.angle_des_tan = angle_des_tan;
         fdataMsg.transformed_tangent = transformed_tangent;
         fdataMsg.transformed_angle_radian = transformed_angle_radian;
         fdataMsg.transformed_angle_deg = transformed_angle_deg;
         
         fdataMsg.time = timer;
         fdataMsg.dtloop = dt;
         
        //  printf("Drone Velocities Tx,Ty,Tz,Oz(%g,%g,%g,%g) =", fdataMsg.Tx, fdataMsg.Ty, fdataMsg.Tz, fdataMsg.Oz);
        //  cout << "\n" << endl;

         rec_pub.publish(fdataMsg);
         // vel_pub.publish(dataMsg);
      }
      ros::spinOnce();
      // ros::spin;
      loop_rate.sleep();
   }

   nlopt_destroy(opt);
   return 0;
}
