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
#include <armadillo>

using namespace std;
using namespace Eigen;

// Global MPC Variables
//  int features_n = 4;
int features_n = 1;
const int dim_inputs = 6;
int dim_s = 6;
int dim_pol_feat = 6;
double dist;

VectorXd ek;
VectorXd ek_1;

MatrixXd gains;

VectorXd feat_u_vector;
VectorXd feat_v_vector;
VectorXd feat_vector;

VectorXd state_vector(6);
VectorXd state_vector_des(6);
VectorXd cmd_vel(dim_inputs);

VectorXd barx_meas(dim_s);
VectorXd barx_des(dim_s);

VectorXd calculated_moments(10);
VectorXd calculated_central_moments(10);

VectorXd error(6);

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

MatrixXd calculate_IM(VectorXd moments, VectorXd central_moments)
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

  // cout << "Le = \n"
  //      << Le << endl;

  return Le;
}

void state_vec_Callback(const std_msgs::Float64MultiArray::ConstPtr &state_vec_msg)
{
  barx_meas.setZero(dim_s + 1);
  for (int i = 0; i < dim_s + 1; i += 1)
  {
    barx_meas[i] = state_vec_msg->data[i];
  }

  dist = state_vec_msg->data[4];
  // cout << "barx_meas: " << barx_meas.transpose() << "\n" << endl;
  // cout << "barx_meas: " << barx_meas.transpose() << endl;
  flag = 1;
}

void state_vec_des_Callback(const std_msgs::Float64MultiArray::ConstPtr &state_vec_des_msg)
{
  // cout << "Starting state vector" << "\n" << endl;
  barx_des.setZero(dim_s + 1);
  for (int i = 0; i < dim_s + 1; i += 1)
  {
    barx_des[i] = state_vec_des_msg->data[i];
  }
  // cout << "barx_des: " << barx_des.transpose() << "\n" << endl;
  // cout << "barx_des: " << barx_des.transpose() << endl;
  flag = 1;
}

void feat_u_vec_Callback(const std_msgs::Float64MultiArray::ConstPtr &feat_u_vec_msg)
{
  feat_u_vector.setZero(feat_u_vec_msg->data.size());
  for (int i = 0; i < feat_u_vec_msg->data.size(); i += 1)
  {
    feat_u_vector[i] = feat_u_vec_msg->data[i];
  }
  // cout << "feat_u_vector: " << feat_u_vector.transpose() << "\n" << endl;
  flag = 1;
}

void feat_v_vec_Callback(const std_msgs::Float64MultiArray::ConstPtr &feat_v_vec_msg)
{
  feat_v_vector.setZero(feat_v_vec_msg->data.size());
  for (int i = 0; i < feat_v_vec_msg->data.size(); i += 1)
  {
    feat_v_vector[i] = feat_v_vec_msg->data[i];
  }
  //  cout << "feat_v_vector: " << feat_v_vector.transpose() << "\n" << endl;
  flag = 1;
}

//****MAIN****//
int main(int argc, char **argv)
{
  ros::init(argc, argv, "mpc");
  ros::NodeHandle nh;
  ros::Rate loop_rate(10.0);

  // Create subscribers
  ros::Subscriber state_vec_sub = nh.subscribe<std_msgs::Float64MultiArray>("/state_vector", 10, state_vec_Callback);
  ros::Subscriber state_vec_des_sub = nh.subscribe<std_msgs::Float64MultiArray>("/state_vector_des", 10, state_vec_des_Callback);
  ros::Subscriber feat_u_vector_sub = nh.subscribe<std_msgs::Float64MultiArray>("/feat_u_vector", 10, feat_u_vec_Callback);
  ros::Subscriber feat_v_vector_sub = nh.subscribe<std_msgs::Float64MultiArray>("/feat_v_vector", 10, feat_v_vec_Callback);

  // Create publishers
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
  // cout << "inputs: " << inputs << endl;
  //****RUNNING LOOP****//
  while (ros::ok())
  {
    double start = ros::Time::now().toSec();
    // printf("Start time:%lf\n", start);
    // cout << "flag: " << flag;

    // ****EXECUTE OPTIMIZATION****//
    if (flag)
    {

      feat_vector.setZero(2 * dim_pol_feat);
      for (int k = 0, kk = 0; k < 2 * dim_pol_feat && kk < dim_pol_feat; k++, kk++)
      {
        feat_vector[k] = feat_u_vector[kk];
        feat_vector[k + 1] = feat_v_vector[kk];
        k++;
      }

      // cout << "feat_u_vector: " << feat_u_vector.transpose() << endl;
      // cout << "feat_v_vector: " << feat_v_vector.transpose() << endl;
      // cout << "feat_vector: " << feat_vector.transpose() << endl;

      calculated_moments.setZero(10);
      calculated_central_moments.setZero(10);

      calculated_moments = calculate_moments(feat_u_vector, feat_v_vector);
      calculated_central_moments = calculate_central_moments(calculated_moments);

      // cout << "calculated_moments: " << calculated_moments.transpose() << endl;
      // cout << "central calculated_moments: " << calculated_central_moments.transpose() << endl;

      MatrixXd IM = calculate_IM(calculated_moments, calculated_central_moments);
      // cout << "IM: " << IM << endl;

      // state_vector << calculated_moments[1]/calculated_moments[0], calculated_moments[2]/calculated_moments[0], abs(calculated_moments[0]), 0.5*atan(2*calculated_central_moments[5]/(calculated_central_moments[3]-calculated_central_moments[4])), cos(barx_des[5]), cos(barx_des[6]);
      state_vector << calculated_moments[1] / calculated_moments[0], calculated_moments[2] / calculated_moments[0], abs(calculated_moments[0]), barx_meas[3], cos(barx_meas[5]), cos(barx_meas[6]);
      state_vector_des << 0.0, 0.0, 0.1, barx_des[3], cos(barx_des[5]), cos(barx_des[6]);

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
      gains.setIdentity(6, 6);
      // gains(0, 0) = -0.1;
      // gains(1, 1) = 0.1;
      // gains(2, 2) = -0.8;
      // gains(3, 3) = 1.0;
      // gains(4, 4) = 1.0;
      // gains(5, 5) = 4.5;

      gains(0, 0) = -1.3;
      gains(1, 1) = -1.3;
      gains(2, 2) = -1.3;
      gains(3, 3) = 20.0;
      gains(4, 4) = -20.0;
      gains(5, 5) = -20.0;

      

      VectorXd velocities = gains * pinv * error;
      cout << "velocities = " << velocities.transpose() << endl;
      // cout << "velocities shape: (" << velocities.rows() << "," << velocities.cols() << ")" << endl;
      cmd_vel = pinv * error;
    }

    // cmd_vel << (-0.1) * cmd_vel[0], (-0.1) * cmd_vel[1], (-1.5) * cmd_vel[2], (1.0) * cmd_vel[3], (1.0) * cmd_vel[4], (3.5) * cmd_vel[5]; // without roll and pitch motions

    // cmd_vel << (-0.1) * cmd_vel[0], (-0.1) * cmd_vel[1], (-1.5) * cmd_vel[2], (1.0) * cmd_vel[3], (1.0) * cmd_vel[4], (3.5) * cmd_vel[5]; // UAV-like testing

    cmd_vel << (-0.1) * cmd_vel[0], (0.1) * cmd_vel[1], (-1.5) * cmd_vel[2], (0.0) * cmd_vel[3], (0.0) * cmd_vel[4], (-3.5) * cmd_vel[5]; // UAV-like testing

    // cout << "dist = " << dist << endl;
    // cmd_vel << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    // cout << "cmd_vel: " << cmd_vel.transpose() << endl;
    printf("Camera Velocity is u,v,z,Ox,Oy,Oz(%g,%g,%g,%g,%g,%g)", cmd_vel[0], cmd_vel[1], cmd_vel[2], cmd_vel[3], cmd_vel[4], cmd_vel[5]);
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

    std_msgs::Float64MultiArray momentsMsg;
    for (int i = 0; i < calculated_moments.size(); i++)
    {
      momentsMsg.data.push_back(calculated_moments[i]);
    }

    std_msgs::Float64MultiArray central_momentsMsg;
    for (int i = 0; i < calculated_central_moments.size(); i++)
    {
      central_momentsMsg.data.push_back(calculated_central_moments[i]);
    }

    std_msgs::Float64MultiArray error_Msg;
    for (int i = 0; i < error.size(); i++)
    {
      error_Msg.data.push_back(error[i]);
    }

    cmd_vel_pub.publish(cmd_vel_Msg);
    state_vec_pub.publish(state_vecMsg);
    state_vec_des_pub.publish(state_vec_desMsg);
    moments_pub.publish(momentsMsg);
    central_moments_pub.publish(central_momentsMsg);
    img_moments_error_pub.publish(error_Msg);

    ros::spinOnce();
    // ros::spin;
    loop_rate.sleep();
  }

  return 0;
}
