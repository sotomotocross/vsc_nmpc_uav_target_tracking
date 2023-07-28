#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "std_msgs/Float64.h"
#include "std_msgs/Float64MultiArray.h"

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
// const int dim_inputs = 4;
const int dim_inputs = 6;
// int dim_s = 4;
int dim_s = 6;
int dim_pol_feat = 6;
// int dim_s = 2*features_n;
const int mpc_hrz = 10; // 10
double mpc_dt = 0.001;  // 0.001
// double mpc_dt = 0.005;
double l = 252.07;
double cu = 960;
double cv = 540;
double a = 10;
double optNum;
double dist;

MatrixXd s_des(dim_s, mpc_hrz + 1);
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

VectorXd barx_meas(dim_s);
VectorXd barx_des(dim_s);

// VectorXd feat_u_vector(dim_pol_feat);
// VectorXd feat_v_vector(dim_pol_feat);
// VectorXd feat_vector(2*dim_pol_feat);

VectorXd feat_u_vector;
VectorXd feat_v_vector;
VectorXd feat_vector;

VectorXd cam_translation_vector(3);
VectorXd cam_rotation_vector(3);

VectorXd stored_state_vector;
VectorXd stored_desired_state_vector;
VectorXd error_k;
VectorXd terminal_error;

VectorXd calculated_moments(10);
VectorXd calculated_central_moments(10);

int flag = 0;

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

MatrixXd calculate_IM(VectorXd camTwist, VectorXd moments, VectorXd central_moments)
{
   MatrixXd Le(6, 6);
   Le.setZero(6, 6);

   // MatrixXd Le(4,6);
   // Le.setZero(4,6);

   // cout << "Calculate Interaction Matrices!!!" << endl;
   // cout << "dist = " << dist << endl;

   double gamma_1 = 1.0;
   double gamma_2 = 1.0;

   double A = -gamma_1 / dist;
   double B = -gamma_2 / dist;
   double C = 1 / dist;

   VectorXd L_area(6);
   L_area.setZero(6);

   double xg = moments[1] / moments[0]; // x-axis centroid
   double yg = moments[2] / moments[0]; // y-axis centroid
   double area = abs(moments[0]);       // area

   L_area << -area * A, -area * B, area * ((3 / dist) - C), 3 * area * yg, -3 * area * xg, 0;

   double n20 = central_moments[3] / area;
   double n02 = central_moments[4] / area;
   double n11 = central_moments[5] / area;

   VectorXd L_xg(6);
   VectorXd L_yg(6);
   L_xg.setZero(6);
   L_yg.setZero(6);

   L_xg << -1 / dist, 0, (xg / dist) + 4 * (A * n20 + B * n11), xg * yg + 4 * n11, -(1 + pow(xg, 2) + 4 * n20), yg;
   L_yg << 0, -1 / dist, (yg / dist) + 4 * (A * n11 + B * n02), 1 + pow(yg, 2) + 4 * n02, -xg * yg - 4 * n11, -xg;

   double mu20_ux = -3 * A * central_moments[3] - 2 * B * central_moments[5]; // μ20_ux
   double mu02_uy = -2 * A * central_moments[5] - 3 * B * central_moments[4]; // μ02_uy
   double mu11_ux = -2 * A * central_moments[5] - B * central_moments[4];     // μ11_ux
   double mu11_uy = -2 * B * central_moments[5] - A * central_moments[3];     // μ11_uy
   double s20 = -7 * xg * central_moments[3] - 5 * central_moments[8];
   double t20 = 5 * (yg * central_moments[3] + central_moments[6]) + 2 * xg * central_moments[5];
   double s02 = -5 * (xg * central_moments[4] + central_moments[7]) - 2 * yg * central_moments[5];
   double t02 = 7 * yg * central_moments[4] + 5 * central_moments[9];
   double s11 = -6 * xg * central_moments[5] - 5 * central_moments[6] - yg * central_moments[3];
   double t11 = 6 * yg * central_moments[5] + 5 * central_moments[7] + xg * central_moments[4];
   double u20 = -A * s20 + B * t20 + 4 * C * central_moments[3];
   double u02 = -A * s02 + B * t02 + 4 * C * central_moments[4];
   double u11 = -A * s11 + B * t11 + 4 * C * central_moments[5];

   VectorXd L_mu20(6);
   VectorXd L_mu02(6);
   VectorXd L_mu11(6);

   L_mu20.setZero(6);
   L_mu02.setZero(6);
   L_mu11.setZero(6);

   L_mu20 << mu20_ux, -B * central_moments[3], u20, t20, s20, 2 * central_moments[5];
   L_mu02 << -A * central_moments[4], mu02_uy, u02, t02, s02, -2 * central_moments[5];
   L_mu11 << mu11_ux, mu11_uy, u11, t11, s11, central_moments[4] - central_moments[3];

   double angle = 0.5 * atan(2 * central_moments[5] / (central_moments[3] - central_moments[4]));
   double Delta = pow(central_moments[3] - central_moments[4], 2) + 4 * pow(central_moments[5], 2);

   double a = central_moments[5] * (central_moments[3] + central_moments[4]) / Delta;
   double b = (2 * pow(central_moments[5], 2) + central_moments[4] * (central_moments[4] - central_moments[3])) / Delta;
   double c = (2 * pow(central_moments[5], 2) + central_moments[3] * (central_moments[3] - central_moments[4])) / Delta;
   double d = 5 * (central_moments[7] * (central_moments[3] - central_moments[4]) + central_moments[5] * (central_moments[9] - central_moments[6])) / Delta;
   double e = 5 * (central_moments[6] * (central_moments[4] - central_moments[3]) + central_moments[5] * (central_moments[8] - central_moments[7])) / Delta;

   double angle_ux = area * A + b * B;
   double angle_uy = -c * A - area * B;
   double angle_wx = -b * xg + a * yg + d;
   double angle_wy = a * xg - c * yg + e;
   double angle_uz = -A * angle_wx + B * angle_wy;

   VectorXd L_angle(6);
   L_angle.setZero(6);
   L_angle << angle_ux, angle_uy, angle_uz, angle_wx, angle_wy, -1;

   double c1 = central_moments[3] - central_moments[4];
   double c2 = central_moments[9] - 3 * central_moments[6];
   double s1 = 2 * central_moments[5];
   double s2 = central_moments[8] - 3 * central_moments[7];
   double I1 = pow(c1, 2) + pow(s1, 2);
   double I2 = pow(c2, 2) + pow(s2, 2);
   double I3 = central_moments[3] + central_moments[4];
   double Px = I1 / pow(I3, 2);
   double Py = area * I2 / pow(I3, 3);

   VectorXd L_Px(6);
   VectorXd L_Py(6);

   L_Px.setZero(6);
   L_Py.setZero(6);

   L_Px << 0, 0, 0, 0.08, -0.011, 0;
   L_Py << 0, 0, 0, 0.04, 0.05, 0;

   // Le.row(0) = L_xg;
   // Le.row(1) = L_yg;
   // Le.row(2) = L_area;
   // Le.row(3) = L_Px;
   // Le.row(4) = L_Py;
   // Le.row(5) = L_angle;

   Le.row(0) = L_xg;
   Le.row(1) = L_yg;
   Le.row(2) = L_area;
   Le.row(3) = L_angle;
   Le.row(4) << 0.0, 0.0, 0.0, 1.0, 0.0, 0.0;
   Le.row(5) << 0.0, 0.0, 0.0, 0.0, 1.0, 0.0;

   gains.setIdentity(dim_s, dim_inputs);

   gains(0, 0) = 1.0;
   gains(1, 1) = 1.0;
   gains(2, 2) = 1.0;
   gains(3, 3) = 1.0;
   gains(4, 4) = 1.0;
   gains(5, 5) = 1.0;

   // cout << "Le shape: (" << Le.rows() << "," << Le.cols() << ")" << endl;
   // cout << "camTwist shape: (" << camTwist.rows() << "," << camTwist.cols() << ")" << endl;
   // cout << "camTwist: " << camTwist.transpose() << endl;
   // cout << "Le = " << Le << endl;
   // cout << "gains * Le = " << gains * Le << endl;
   // cout << "Le*camTwist = " << gains * Le * camTwist << endl;

   return gains * Le * camTwist;
}

// PVS-MPC Cost Function
double costFunction(unsigned int n, const double *x, double *grad, void *data)
{
   MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz>>((double *)x);

   // cout << "\nBegin of PVS-NMPC cost function" << endl;

   // Trajectory of States (image features)
   // cout << "feat_u_vector.size() = " << feat_u_vector.size() << endl;
   // cout << "feat_v_vector.size() = " << feat_v_vector.size() << endl;
   int N = feat_u_vector.size();
   MatrixXd traj_s(dim_s, mpc_hrz + 1);
   MatrixXd feature_hrz_prop(2 * N, mpc_hrz + 1);
   // cout << "feature_hrz_prop.size() = " << feature_hrz_prop.size() << endl;
   traj_s.setZero(dim_s, mpc_hrz + 1);
   // cout << "barx_meas: " << barx_meas << "\n" << endl;

   calculated_moments.setZero(10);
   calculated_central_moments.setZero(10);

   calculated_moments = calculate_moments(feat_u_vector, feat_v_vector);
   calculated_central_moments = calculate_central_moments(calculated_moments);

   // cout << "calculated_moments: " << calculated_moments.transpose() << endl;
   // cout << "central calculated_moments: " << calculated_central_moments.transpose() << endl;

   
   // cout << "IM: " << IM << endl;
   traj_s.col(0) << barx_meas[0], barx_meas[1], barx_meas[2], barx_meas[3], cos(barx_meas[5]), cos(barx_meas[6]);
   // traj_s.col(0) << calculated_moments[1] / calculated_moments[0], calculated_moments[2] / calculated_moments[0], abs(calculated_moments[0]), 0.5*atan(2*calculated_central_moments[5]/(calculated_central_moments[3]-calculated_central_moments[4])), cos(barx_meas[5]), cos(barx_meas[6]);
   // traj_s.col(0) << calculated_moments[1] / calculated_moments[0], calculated_moments[2] / calculated_moments[0], abs(calculated_moments[0]), 0.5*atan(2*calculated_central_moments[5]/(calculated_central_moments[3]-calculated_central_moments[4])), cos(barx_meas[5]), cos(barx_meas[6]);
   // cout << "traj_s.col(0): " << traj_s.col(0).transpose() << endl;
   // cout << "abs(calculated_moments[0]) = " << abs(calculated_moments[0]) << endl;

   // cout << "traj_s before: \n" << traj_s << endl;

   // Progate the model (PVS with Image Jacobian)
   for (int k = 0; k < mpc_hrz; k++)
   {
      VectorXd sdot_test = calculate_IM(inputs.col(k), calculated_moments, calculated_central_moments);
      // cout << "sdot_test = " << sdot_test.transpose() << endl;
      traj_s.col(k + 1) = traj_s.col(k) + sdot_test * mpc_dt;
      //   cout << "traj_s.col(k + 1) =" << traj_s.col(k + 1) << endl;
   }
   // cout << "traj_s after: \n" << traj_s << endl;

   // Calculate Running Costs
   double Ji = 0.0;

   //****DEFINE INITIAL DESIRED V****//
   // s_des.col(0) << 0.0, 0.0, 0.1, barx_des[3], cos(barx_des[5]), cos(barx_des[6]);
   s_des.col(0) << barx_des[0], barx_des[1], barx_des[2], barx_des[3], cos(barx_des[5]), cos(barx_des[6]);
   
   // cout << "barx_des[0] = " << barx_des[0] << endl;
   // cout << "barx_des[1] = " << barx_des[1] << endl;
   // cout << "barx_des[2] = " << barx_des[2] << endl;
   // cout << "barx_des[3] = " << barx_des[3] << endl;
   // cout << "barx_des[4] = " << barx_des[4] << endl;
   // cout << "barx_des[5] = " << barx_des[5] << endl;
   // cout << "barx_des[6] = " << barx_des[6] << endl;

   // cout << "s_des before: \n" << s_des << endl;   
   
   for (int k = 0; k < mpc_hrz; k++)
   {
      // cout << "k: " << k << endl;
      s_des.col(k) << barx_des[0], barx_des[1], barx_des[2], barx_des[3], cos(barx_des[5]), cos(barx_des[6]);
      // cout << "traj_s.col(k): " << traj_s.col(k).transpose() << endl;
      // cout << "s_des.col(k): " << s_des.col(k).transpose() << endl;
      ek = -s_des.col(k) + traj_s.col(k);
      // cout << "ek: " << ek.transpose() << endl;
      // cout << "inside for loop ek: " << ek << endl ;
      // ek = traj_s.col(k)
      // cout << "s_des.col(k): " << s_des.col(k) << endl;
      // cout << "traj_s.col(k): " << traj_s.col(k) << endl;
      // cout << "traj_s.col(k): " << traj_s.col(k) << endl;
      Ji += ek.transpose() * Q * ek;
      Ji += inputs.col(k).transpose() * R * inputs.col(k);
   }
   // cout << "Ji: \n" << Ji << endl;
   // cout << "s_des after: \n" << s_des << endl;
   // cout << "traj_s: " << traj_s.col(0).transpose() << endl;
   // cout << "ek: " << ek.transpose() << endl;

   // Calculate Terminal Costs
   double Jt;
   // cout << "traj_s.col(mpc_hrz): " << traj_s.col(mpc_hrz).transpose() << endl;
   // cout << "s_des.col(mpc_hrz): " << s_des.col(mpc_hrz).transpose() << endl;
   s_des.col(mpc_hrz-1) << barx_des[0], barx_des[1], barx_des[2], barx_des[3], cos(barx_des[5]), cos(barx_des[6]);
   s_des.col(mpc_hrz) << barx_des[0], barx_des[1], barx_des[2], barx_des[3], cos(barx_des[5]), cos(barx_des[6]);
   VectorXd et = traj_s.col(mpc_hrz) - s_des.col(mpc_hrz);
   // cout << "et: " << et.transpose() << endl;
   // cout << "Jt: \n" << Jt << endl;

   Jt = et.transpose() * P * et;
   //   cout << "Ji = " << Ji << " + " << "Jt = " << Jt << endl;
   // cout << "\n" << endl;

   stored_state_vector.setZero(dim_s);
   stored_desired_state_vector.setZero(dim_s);
   error_k.setZero(dim_s);
   terminal_error.setZero(dim_s);

   stored_state_vector = traj_s.col(0);
   stored_desired_state_vector = s_des.col(0);
   error_k = ek;
   terminal_error = et;

   cout << "stored_state_vector: " << stored_state_vector.transpose() << endl;
   cout << "stored_desired_state_vector: " << stored_desired_state_vector.transpose() << endl;
   cout << "error_k: " << error_k.transpose() << endl;
   cout << "terminal_error: " << terminal_error.transpose() << endl;

   return (Ji + Jt);
}

//****DEFINE FOV CONSTRAINTS****//
void constraints(unsigned int m, double *c, unsigned int n, const double *x, double *grad, void *data)
{
   MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz>>((double *)x);

   // cout << "\nBegin of PVS-NMPC cost function" << endl;

   // Trajectory of States (image features)
   // cout << "feat_u_vector.size() = " << feat_u_vector.size() << endl;
   // cout << "feat_v_vector.size() = " << feat_v_vector.size() << endl;
   int N = feat_u_vector.size();
   MatrixXd traj_s(dim_s, mpc_hrz + 1);
   MatrixXd feature_hrz_prop(2 * N, mpc_hrz + 1);
   // cout << "feature_hrz_prop.size() = " << feature_hrz_prop.size() << endl;
   traj_s.setZero(dim_s, mpc_hrz + 1);
   // cout << "barx_meas: " << barx_meas << "\n" << endl;

   calculated_moments.setZero(10);
   calculated_central_moments.setZero(10);

   calculated_moments = calculate_moments(feat_u_vector, feat_v_vector);
   calculated_central_moments = calculate_central_moments(calculated_moments);

   // cout << "calculated_moments: " << calculated_moments.transpose() << endl;
   // cout << "central calculated_moments: " << calculated_central_moments.transpose() << endl;

   
   // cout << "IM: " << IM << endl;
   traj_s.col(0) << barx_meas[0], barx_meas[1], barx_meas[2], barx_meas[3], cos(barx_meas[5]), cos(barx_meas[6]);
   // traj_s.col(0) << calculated_moments[1] / calculated_moments[0], calculated_moments[2] / calculated_moments[0], abs(calculated_moments[0]), 0.5*atan(2*calculated_central_moments[5]/(calculated_central_moments[3]-calculated_central_moments[4])), cos(barx_meas[5]), cos(barx_meas[6]);
   // cout << "traj_s.col(0): " << traj_s.col(0).transpose() << endl;

   // Progate the model (PVS with Image Jacobian)
   for (int k = 0; k < mpc_hrz; k++)
   {
      VectorXd sdot_test = calculate_IM(inputs.col(k), calculated_moments, calculated_central_moments);
      // cout << "sdot_test = " << sdot_test.transpose() << endl;
      traj_s.col(k + 1) = traj_s.col(k) + sdot_test * mpc_dt;
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

void state_vec_Callback(const std_msgs::Float64MultiArray::ConstPtr &state_vec_msg)
{
   // cout << "Starting state vector" << "\n" << endl;
   barx_meas.setZero(dim_s + 1);
   // cout << "barx_meas before: " << barx_meas << "\n" << endl;

   //  cout << "state_vec_msg->data.size() = " << state_vec_msg->data.size() << endl;

   for (int i = 0; i < dim_s + 1; i += 1)
   {
      barx_meas[i] = state_vec_msg->data[i];
      // cout << "i: " << i << endl;
      // cout << "barx_meas[i]: " << barx_meas[i] << "\n" << endl;
      //   cout << "after add barx_meas: " << barx_meas << "\n" << endl;
   }

   dist = state_vec_msg->data[4];
   //  cout << "dist: " << dist << endl;
   // cout << "barx_meas[0]: " << state_vec_msg->data[0] << endl;
   // cout << "barx_meas[1]: " << state_vec_msg->data[1] << endl;
   // cout << "barx_meas[2]: " << state_vec_msg->data[2] << endl;
   // cout << "barx_meas[3]: " << state_vec_msg->data[3] << endl;
   // cout << "barx_meas[4]: " << state_vec_msg->data[4] << endl;
   // cout << "barx_meas[5]: " << state_vec_msg->data[5] << endl;
   // cout << "barx_meas[6]: " << state_vec_msg->data[6] << endl;
   //  cout << "barx_meas: " << barx_meas.transpose() << "\n" << endl;
   //  cout << "Done with state vector" << "\n" << endl;
   flag = 1;
}

void state_vec_des_Callback(const std_msgs::Float64MultiArray::ConstPtr &state_vec_des_msg)
{
   // cout << "Starting state vector" << "\n" << endl;
   barx_des.setZero(dim_s + 1);
   // cout << "barx_des before: " << barx_des << "\n" << endl;

   //  cout << "state_vec_des_msg->data.size() = " << state_vec_des_msg->data.size() << endl;

   for (int i = 0; i < dim_s + 1; i += 1)
   {
      barx_des[i] = state_vec_des_msg->data[i];
      // cout << "i: " << i << endl;
      // cout << "barx_des[i]: " << barx_des[i] << "\n" << endl;
      //   cout << "after add barx_des: " << barx_des << "\n" << endl;
   }
   //  cout << "barx_des[0]: " << state_vec_des_msg->data[0] << endl;
   // cout << "barx_des[1]: " << state_vec_des_msg->data[1] << endl;
   // cout << "barx_des[2]: " << state_vec_des_msg->data[2] << endl;
   // cout << "barx_des[3]: " << state_vec_des_msg->data[3] << endl;
   // cout << "barx_des[4]: " << state_vec_des_msg->data[4] << endl;
   // cout << "barx_des[5]: " << state_vec_des_msg->data[5] << endl;
   // cout << "barx_des[6]: " << state_vec_des_msg->data[6] << endl;
   //  cout << "barx_des: " << barx_des.transpose() << "\n" << endl;
   //  cout << "Done with state vector" << "\n" << endl;
   flag = 1;
}

void feat_u_vec_Callback(const std_msgs::Float64MultiArray::ConstPtr &feat_u_vec_msg)
{
   // cout << "Starting feature u vector" << "\n" << endl;
   feat_u_vector.setZero(feat_u_vec_msg->data.size());
   // cout << "feat_u_vector before: " << feat_u_vector << "\n" << endl;

   //  cout << "feat_u_vec_msg->data.size() = " << feat_u_vec_msg->data.size() << endl;

   for (int i = 0; i < feat_u_vec_msg->data.size(); i += 1)
   {
      feat_u_vector[i] = feat_u_vec_msg->data[i];
      //   cout << "feat_u_vector[i]: " << feat_u_vector[i] << "\n" << endl;
      //   cout << "after add feat_u_vector: " << feat_u_vector << "\n" << endl;
   }
   //  cout << "feat_u_vector: " << feat_u_vector.transpose() << "\n" << endl;
   // cout << "Done with feature u vector" << "\n" << endl;
   flag = 1;
}

void feat_v_vec_Callback(const std_msgs::Float64MultiArray::ConstPtr &feat_v_vec_msg)
{
   // cout << "Starting feature v vector" << "\n" << endl;
   feat_v_vector.setZero(feat_v_vec_msg->data.size());
   // cout << "feat_v_vector before: " << feat_v_vector << "\n" << endl;

   //  cout << "feat_v_vec_msg->data.size() = " << feat_v_vec_msg->data.size() << endl;

   for (int i = 0; i < feat_v_vec_msg->data.size(); i += 1)
   {
      feat_v_vector[i] = feat_v_vec_msg->data[i];
      //   cout << "feat_v_vector[i]: " << feat_v_vector[i] << "\n" << endl;
      //   cout << "after add feat_v_vector: " << feat_v_vector << "\n" << endl;
   }
   //  cout << "feat_v_vector: " << feat_v_vector.transpose() << "\n" << endl;
   // cout << "Done with feature v vector" << "\n" << endl;
   flag = 1;
}

void cam_trans_Callback(const std_msgs::Float64MultiArray::ConstPtr &cam_trans_vec_msg)
{
   cam_translation_vector.setZero(3);
   cam_translation_vector << cam_trans_vec_msg->data[0], cam_trans_vec_msg->data[1], cam_trans_vec_msg->data[2];
   // cout << "cam_translation_vector: " << cam_translation_vector << endl;
   flag = 1;
}

void cam_rotation_Callback(const std_msgs::Float64MultiArray::ConstPtr &cam_rot_vec_msg)
{
   cam_rotation_vector.setZero(3);
   cam_rotation_vector << cam_rot_vec_msg->data[0], cam_rot_vec_msg->data[1], cam_rot_vec_msg->data[2];
   // cout << "cam_rotation_vector: " << cam_rotation_vector << endl;
   flag = 1;
}

//****MAIN****//
int main(int argc, char **argv)
{
   ros::init(argc, argv, "mpc");
   ros::NodeHandle nh;
   ros::Rate loop_rate(10.0);

   // Create publishers
   ros::Subscriber state_vec_sub = nh.subscribe<std_msgs::Float64MultiArray>("/state_vector", 10, state_vec_Callback);
   ros::Subscriber state_vec_des_sub = nh.subscribe<std_msgs::Float64MultiArray>("/state_vector_des", 10, state_vec_des_Callback);
   ros::Subscriber feat_u_vector_sub = nh.subscribe<std_msgs::Float64MultiArray>("/feat_u_vector", 10, feat_u_vec_Callback);
   ros::Subscriber feat_v_vector_sub = nh.subscribe<std_msgs::Float64MultiArray>("/feat_v_vector", 10, feat_v_vec_Callback);
   ros::Subscriber cam_trans_vec_sub = nh.subscribe<std_msgs::Float64MultiArray>("/cam_translation", 10, cam_trans_Callback);
   ros::Subscriber cam_rot_vec_sub = nh.subscribe<std_msgs::Float64MultiArray>("/cam_rotation", 10, cam_rotation_Callback);

   // Create subscribers
   ros::Publisher cmd_vel_pub = nh.advertise<std_msgs::Float64MultiArray>("/cmd_vel", 1);
   ros::Publisher state_vec_pub = nh.advertise<std_msgs::Float64MultiArray>("/state_vec", 1);
   ros::Publisher state_vec_des_pub = nh.advertise<std_msgs::Float64MultiArray>("/state_vec_des", 1);
   ros::Publisher ek_nmpc = nh.advertise<std_msgs::Float64MultiArray>("/nmpc_error", 1);
   ros::Publisher et_nmpc = nh.advertise<std_msgs::Float64MultiArray>("/nmpc_terminal_error", 1);

   // Initialize MPC Variables
   s_des.setZero(dim_s, mpc_hrz + 1);
   s_abs.setZero(dim_s);
   // s_abs << 1024-512, 1024-512, 1.0, 3.14, 0.8;
   //    s_abs << 1920-960, 1080-540, 1.0, 3.14;
   s_abs << 1920 - 960, 1080 - 540, 1.0, 3.14, cos(2.044), cos(2.044);

   //****SET MPC COST FUNCTION MATRICES****//
   Q.setIdentity(dim_s, dim_s);
   R.setIdentity(dim_inputs, dim_inputs);
   P.setIdentity(dim_s, dim_s);

   Q = 10 * Q;
   R = 5 * R;
   P = 1 * Q;

   Q(0,0) = 1500;
   Q(1,1) = 1500;
   Q(2, 2) = 35;
   Q(4, 4) = 1500;
   Q(5, 5) = 1500;

   //****DEFINE INPUT CONSTRAINTS****//
   double inputs_lb[dim_inputs * mpc_hrz];
   double inputs_ub[dim_inputs * mpc_hrz];

   for (int k = 0; k < mpc_hrz; k++)
   {
      inputs_lb[dim_inputs * k] = -0.5;
      inputs_lb[dim_inputs * k + 1] = -3;
      inputs_lb[dim_inputs * k + 2] = -0.1;
      inputs_lb[dim_inputs * k + 3] = -1;
      inputs_lb[dim_inputs * k + 4] = -1;
      inputs_lb[dim_inputs * k + 5] = -1;
      inputs_ub[dim_inputs * k] = 0.5;
      inputs_ub[dim_inputs * k + 1] = 3;
      inputs_ub[dim_inputs * k + 2] = 0.1;
      inputs_ub[dim_inputs * k + 3] = 1;
      inputs_ub[dim_inputs * k + 4] = 1;
      inputs_ub[dim_inputs * k + 5] = 1;

      // inputs_lb[dim_inputs * k] = -0.5;
      // inputs_lb[dim_inputs * k + 1] = -3;
      // inputs_lb[dim_inputs * k + 2] = -0.1;
      // inputs_lb[dim_inputs * k + 3] = -1;
      // inputs_ub[dim_inputs * k] = 0.5;
      // inputs_ub[dim_inputs * k + 1] = 3;
      // inputs_ub[dim_inputs * k + 2] = 0.1;
      // inputs_ub[dim_inputs * k + 3] = 1;
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
      double start = ros::Time::now().toSec();
      // printf("Start time:%lf\n", start);
      // cout << "flag: " << flag;

      // ****EXECUTE OPTIMIZATION****//
      if (flag)
      {
         // cout << "Ela mastora ola kala me to flag!!!!" << endl;
         optNum = nlopt_optimize(opt, inputs, &minJ);
         // cout << "Optimization Return Code: " << nlopt_optimize(opt, inputs, &minJ) << endl;
         // cout << "Optimization Return Code: " << optNum << endl;
      }
      // printf("found minimum at J(%g,%g,%g,%g,%g,%g) = %g\n", inputs[0], inputs[1],inputs[2], inputs[3], inputs[4], inputs[5], minJ);
      // cout << "\n" << endl;

      double end = ros::Time::now().toSec();
      double tf = ros::WallTime::now().toSec();
      double timer = tf - t0;
      double dt = end - start;
      realtime = realtime + dt;

      // VectorXd cmd_vel(dim_inputs+2);
      VectorXd cmd_vel(dim_inputs);
      // VectorXd tmp_cmd_vel(dim_inputs);
      // tmp_cmd_vel << inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5];
      // cout << "tmp_cmd_vel = " << tmp_cmd_vel.transpose() << endl;
      // cmd_vel << 1000*inputs[0], 1000*inputs[1], 250*inputs[2], 850*inputs[3], -850*inputs[4], -1000*inputs[5];

      cmd_vel << 350 * inputs[0], 350 * inputs[1], 1500 * inputs[2], (1.0) * inputs[3],  (1.0) * inputs[4], (1600) * inputs[5]; // without roll and pitch motions
      // cmd_vel << 350 * inputs[0], 350 * inputs[1], 500 * inputs[2], (250) * inputs[3],  (-250) * inputs[4], (2000) * inputs[5]; // with roll and pitch motions
      // cmd_vel << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;

      // cmd_vel << 150*inputs[0], 150*inputs[1], 25*inputs[2], 85*inputs[3], -85*inputs[4], -100*inputs[5];
      // cout << "cmd_vel: " << cmd_vel << endl;

      std_msgs::Float64MultiArray cmd_vel_Msg;
      for (int i = 0; i < cmd_vel.size(); i++)
      {
         cmd_vel_Msg.data.push_back(cmd_vel[i]);
      }

      std_msgs::Float64MultiArray state_var_Msg;
      for (int i = 0; i < stored_state_vector.size(); i++)
      {
         state_var_Msg.data.push_back(stored_state_vector[i]);
      }

      std_msgs::Float64MultiArray desired_state_var_Msg;
      for (int i = 0; i < stored_desired_state_vector.size(); i++)
      {
         desired_state_var_Msg.data.push_back(stored_desired_state_vector[i]);
      }

      std_msgs::Float64MultiArray errork_Msg;
      for (int i = 0; i < error_k.size(); i++)
      {
         errork_Msg.data.push_back(error_k[i]);
      }

      std_msgs::Float64MultiArray errort_Msg;
      for (int i = 0; i < terminal_error.size(); i++)
      {
         errort_Msg.data.push_back(terminal_error[i]);
      }

      cout << "Cost function value is: " << minJ << endl;
      printf("Camera Velocity is u,v,z,Ox,Oy,Oz(%g,%g,%g,%g,%g,%g)", cmd_vel[0], cmd_vel[1], cmd_vel[2], cmd_vel[3], cmd_vel[4], cmd_vel[5]);
      cout << "\n"
           << endl;

      cmd_vel_pub.publish(cmd_vel_Msg);
      state_vec_pub.publish(state_var_Msg);
      state_vec_des_pub.publish(desired_state_var_Msg);
      ek_nmpc.publish(errork_Msg);
      et_nmpc.publish(errort_Msg);

      ros::spinOnce();
      // ros::spin;
      loop_rate.sleep();
   }

   nlopt_destroy(opt);
   return 0;
}
