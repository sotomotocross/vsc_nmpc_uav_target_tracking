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
int dim_s = 4;
int dim_pol_feat = 6;
// int dim_s = 2*features_n;
const int mpc_hrz = 10;  // 10
double mpc_dt = 0.001; //0.001
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
VectorXd t;

VectorXd ek;
VectorXd ek_1;

VectorXd barx_meas(dim_s);
VectorXd barx_des(dim_s);

VectorXd feat_u_vector(dim_pol_feat);
VectorXd feat_v_vector(dim_pol_feat);
VectorXd feat_vector(2*dim_pol_feat);

VectorXd cam_translation_vector(3);
VectorXd cam_rotation_vector(3);

VectorXd cartesian_from_pixel_feat_vec(2*dim_pol_feat);
VectorXd virtual_img_plane_transformation_vector(3*dim_pol_feat);
VectorXd transformed_feat_vector(3*dim_pol_feat);

int flag = 0;

// VectorXd featuresTransformation(VectorXd ft_vc, VectorXd translation, VectorXd rotation)
// {
//    // cout << "Mpikame feature transformation function!!!!" << endl;

//    // cout << "ft_vc: " << ft_vc << endl;
//    // cout << "translation: " << translation << endl;
//    // cout << "rotation: " << rotation << endl;

//    MatrixXd Tt(3,3);
//    Tt.setZero(3,3);
//    // cout << "Tt before fill: " << Tt << endl;
//    Tt(0, 0) = 0;
//    Tt(1, 0) = translation[2];
//    Tt(2, 0) = -translation[1];
//    Tt(0, 1) = -translation[2];
//    Tt(1, 1) = 0;
//    Tt(2, 1) = translation[0];
//    Tt(0, 2) = translation[1];
//    Tt(1, 2) = -translation[0];
//    Tt(2, 2) = 0;
//    // cout << "Tt after fill: " << Tt << endl;

//    MatrixXd Rx(3,3);
//    Rx.setZero(3,3);
//    // cout << "Rx before fill: " << Rx << endl;
//    Rx(0, 0) = 1;
//    Rx(1, 0) = 0;
//    Rx(2, 0) = 0;
//    Rx(0, 1) = 0;
//    Rx(1, 1) = cos(rotation[0]);
//    Rx(2, 1) = sin(rotation[0]);
//    Rx(0, 2) = 0;
//    Rx(1, 2) = -sin(rotation[0]);
//    Rx(2, 2) = cos(rotation[0]);
//    // cout << "Rx after fill: " << Rx << endl;

//    MatrixXd Ry(3,3);
//    Ry.setZero(3,3);
//    // cout << "Ry before fill: " << Ry << endl;
//    Ry(0, 0) = cos(rotation[1]);
//    Ry(1, 0) = 0;
//    Ry(2, 0) = -sin(rotation[1]);
//    Ry(0, 1) = 0;
//    Ry(1, 1) = 1;
//    Ry(2, 1) = 0;
//    Ry(0, 2) = sin(rotation[1]);
//    Ry(1, 2) = 0;
//    Ry(2, 2) = cos(rotation[1]);
//    // cout << "Ry after fill: " << Ry << endl;
      
//    MatrixXd Rz(3,3);   
//    Rz.setZero(3,3);
//    // cout << "Rz before fill: " << Rz << endl;
//    Rz(0, 0) = cos(rotation[2]);
//    Rz(1, 0) = sin(rotation[2]);
//    Rz(2, 0) = 0;
//    Rz(0, 1) = -sin(rotation[2]);
//    Rz(1, 1) = cos(rotation[2]);
//    Rz(2, 1) = 0;
//    Rz(0, 2) = 0;
//    Rz(1, 2) = 0;
//    Rz(2, 2) = 1;
//    // cout << "Rz after fill: " << Rz << endl;

//    MatrixXd Rft(3,3);
//    Rft.setZero(3,3);
//    // cout << "Rft before fill: " << Rft << endl;
//    Rft = Rz*Ry*Rx;
//    // cout << "Rft after fill: " << Rft << endl;

//    VectorXd virtual_img_plane_features(3*dim_pol_feat);
//    virtual_img_plane_features.setZero(3*dim_pol_feat);
//    VectorXd temp(3);
//    temp.setZero(3);
//    VectorXd temp_tf(3);
//    temp_tf.setZero(3);

//    // cout << "ft_vc: " << ft_vc.transpose() << endl;
//    // cout << "temp: " << temp.transpose() << endl;
//    // cout << "temp_tf: " << temp_tf.transpose() << endl;
//    // cout << "virtual_img_plane_features: " << virtual_img_plane_features.transpose() << endl;

//    for(int k=0, kk=0;k<3*dim_pol_feat,kk<2*dim_pol_feat;k+=3, kk+=2){
//       // cout << "Mpikame edo?" << endl;
//       // cout << "k: " << k << endl;
//       // cout << "kk: " << kk << endl;
//       temp[0] = ft_vc[kk];
//       temp[1] = ft_vc[kk+1];
//       temp[2] = dist;
//       temp_tf = Rft*temp;
//       virtual_img_plane_features[k] = temp_tf[0];
//       virtual_img_plane_features[k+1] = temp_tf[1];
//       virtual_img_plane_features[k+2] = temp_tf[2];
//       // cout << "Transformation Telos!!!!" << endl;
//    }

//    // cout << "virtual_img_plane_features: " << virtual_img_plane_features << endl;
//    return virtual_img_plane_features;
// }

// VectorXd cartesian_from_pixel(VectorXd ft_vc){

//    // cout << "Cartesian from pixel function" << endl;

//    VectorXd cart_ft_vec(2*dim_pol_feat);
//    cart_ft_vec.setZero(2*dim_pol_feat);

//    for (int i = 0; i<ft_vc.size(); i++){
//       // cout << "i: " << i << endl;
//       cart_ft_vec[i] = dist*((ft_vc[i]-cu)/l);
//       cart_ft_vec[i+1] = dist*((ft_vc[i+1]-cv)/l);
//       i++;
//    }
//    // cout << "ft_vc: " << ft_vc.transpose() << endl;
//    // cout << "cart_ft_vec: " << cart_ft_vec.transpose() << endl;   
//    return cart_ft_vec;
// }

// VectorXd pixels_from_cartesian(VectorXd ft_vc){
//    // cout << "Mpikame kai sto teleutaio transformation!!!" << endl;
//    // cout << "ft_vc: " << ft_vc.transpose() << endl;   

//    VectorXd last_pixel_tf(3*dim_pol_feat);
//    last_pixel_tf.setZero(3*dim_pol_feat);

//    for(int i=0; i<last_pixel_tf.size() ; i+=3){
//       last_pixel_tf[i] = (ft_vc[i]/ft_vc[i+2])*l + cu;
//       last_pixel_tf[i+1] = (ft_vc[i+1]/ft_vc[i+2])*l + cv;
//       last_pixel_tf[i+2] = ft_vc[i+2];
//    }
//    // cout << "last_pixel_tf: " << last_pixel_tf.transpose() << endl;
//    return last_pixel_tf;
// }

VectorXd Dynamic_System_x_y_reverted(VectorXd camTwist, VectorXd feat_prop)
{
   // cout << "Mpike ki edo (PVS) to gamidi!!!" << endl;
   MatrixXd model_mat(dim_s, dim_inputs);
   // cout << "model_mat shape: (" << model_mat.rows() << "," << model_mat.cols() << ")" << endl;
   // cout << "camTwist shape: (" << camTwist.rows() << "," << camTwist.cols() << ")" << endl;

   // Barycenter dynamics calculation
   double term_1_4 = 0.0;
   double term_1_5 = 0.0;
   double term_2_4 = 0.0;
   double term_2_5 = 0.0;

   for (int i = 0; i < dim_pol_feat; i += 1)
   {
      term_1_4 = term_1_4 + feat_u_vector[i] * feat_v_vector[i];
      term_1_5 = term_1_5 + (1 + pow(feat_u_vector[i], 2));
      term_2_4 = term_2_4 + (1 + pow(feat_v_vector[i], 2));
      term_2_5 = term_2_5 + feat_u_vector[i] * feat_v_vector[i];
   }

   term_1_4 = term_1_4/dim_pol_feat;
   term_1_5 = -term_1_5/dim_pol_feat;
   term_2_4 = term_2_4/dim_pol_feat;
   term_2_5 = -term_2_5/dim_pol_feat;

   // cout << "term_1_4: " << term_1_4 << endl;
   // cout << "term_1_5: " << term_1_5 << endl;
   // cout << "term_2_4: " << term_2_4 << endl;
   // cout << "term_2_5: " << term_2_5 << endl;

   double g_4_4, g_4_5, g_4_6;

   // Angle dynamics calculation
   // Fourth term
   double term_4_4_1, term_4_4_2, term_4_4_3, term_4_4_4;
   double sum_4_4_1 = 0.0, sum_4_4_2 = 0.0;

   double k = 0;
   VectorXd x(dim_pol_feat);
   VectorXd y(dim_pol_feat);

   x.setZero(dim_pol_feat);
   y.setZero(dim_pol_feat);

   // cout << "initial x: " << x << endl;
   // cout << "initial y: " << y << endl;

   for (int i = 0; i < dim_pol_feat; i += 1)
   {
      // cout << "index for x: " << i << endl;
      x[k] = feat_u_vector[i];
      k++;
   }

   k = 0;

   for (int i = 0; i < dim_pol_feat ; i += 1)
   {
      // cout << "index for y: " << i << endl;
      y[k] = feat_v_vector[i];
      k++;
   }
   
   // cout << "feat_u_vector: " << feat_u_vector << endl;
   // cout << "x: " << x.transpose() << endl;
   // cout << "feat_v_vector: " << feat_v_vector << endl;
   // cout << "y: " << y.transpose() << endl;

   for (int i = 0; i < dim_pol_feat; i += 1)
   {
      sum_4_4_1 = sum_4_4_1 + pow(feat_v_vector[i], 2);
      sum_4_4_2 = sum_4_4_2 + feat_u_vector[i] * feat_v_vector[i];
   }
   

   term_4_4_1 = barx_meas[3] / (y[0] + y[1] - 2 * barx_meas[1]);
   term_4_4_2 = (pow(y[0], 2) + pow(y[1], 2) - (2 / dim_pol_feat) * sum_4_4_1);
   term_4_4_3 = -1 / (y[0] + y[1] - 2 * barx_meas[1]);
   term_4_4_4 = (x[0] * y[0] + x[1] * y[1] - (2 / dim_pol_feat) * sum_4_4_2);

   g_4_4 = term_4_4_1 * term_4_4_2 + term_4_4_3 * term_4_4_4;
   // cout << "g_4_4: " << g_4_4 << endl; 

   // Fifth term
   double term_4_5_1, term_4_5_2, term_4_5_3, term_4_5_4;
   double sum_4_5_1 = 0.0, sum_4_5_2 = 0.0;

   for (int i = 0; i < dim_pol_feat; i += 1)
   {
      sum_4_5_1 = sum_4_5_1 + pow(feat_u_vector[i], 2);
      sum_4_5_2 = sum_4_5_2 + feat_u_vector[i] * feat_v_vector[i];
   }

   // cout << "sum_4_5_1: " << sum_4_5_1 << endl;
   // cout << "sum_4_5_2: " << sum_4_5_2 << endl;

   term_4_5_1 = 1 / (y[0] + y[1] - 2 * barx_meas[1]);
   term_4_5_2 = (pow(x[0], 2) + pow(x[1], 2) - (2 / dim_pol_feat) * sum_4_5_1);
   term_4_5_3 = -barx_meas[3] / (y[0] + y[1] - 2 * barx_meas[1]);
   term_4_5_4 = (x[0] * y[0] + x[1] * y[1] - (2 / dim_pol_feat) * sum_4_5_2);

   g_4_5 = term_4_5_1 * term_4_5_2 + term_4_5_3 * term_4_5_4;
   // cout << "g_4_5: " << g_4_5 << endl; 

   // Fifth term
   g_4_6 = pow(barx_meas[3], 2) + 1;
   // cout << "g_4_6: " << g_4_6 << endl; 

   // model_mat << 
   //          -1/dist, 0.0, barx_meas[0]/dist, barx_meas[1],
   //          0.0, -1/dist, barx_meas[1]/dist, -barx_meas[0],
   //          0.0, 0.0, 2/dist, 0.0,
   //          0.0, 0.0, 0.0, g_4_6,
   //          0.0, 0.0, 1.0, 0.0;

   // model_mat << 
   //          -1/dist, 0.0, barx_meas[0]/dist, barx_meas[1],
   //          0.0, -1/dist, barx_meas[1]/dist, -barx_meas[0],
   //          0.0, 0.0, 2/dist, 0.0,
   //          0.0, 0.0, 0.0, g_4_6;
   
   model_mat << 
            -1/dist, 0.0, barx_meas[0]/dist, term_1_4, term_1_5, barx_meas[1],
            0.0, -1/dist, barx_meas[1]/dist, term_2_4, term_2_5, -barx_meas[0],
            0.0, 0.0, 2/dist, (3/2)*barx_meas[1], (-3/2)*barx_meas[0], 0.0,
            0.0, 0.0, 0.0, g_4_4, g_4_5, g_4_6;

   // cout << "after model_mat: " << model_mat << endl;

   // cout << "model_mat shape: (" << model_mat.rows() << "," << model_mat.cols() << ")" << endl;
   // cout << "Interaction Matrix:\n" << model_mat << endl;
   // cout << "Interaction Matrix" << model_mat * camTwist << endl;

   return model_mat * camTwist;
}

// IBVS Feature Rate Le*Vk
VectorXd IBVSSystem(VectorXd camTwist)
{
   // cout << "Mpike kai sto feature propagation to gamidi!!!" << endl;
   MatrixXd Le(2 * dim_pol_feat, dim_inputs);

   //,Le.setZero(dim_s,dim_inputs);
   Le.setZero(2 * dim_pol_feat, dim_inputs);
   // cout << "Le: \n" << Le << endl;

   // Le.row(0) << -1/Z0, 0.0, transformed_features[0]/Z0, transformed_features[1];
   // Le.row(1) << 0.0, -1/Z0, transformed_features[1]/Z0, -transformed_features[0];

   // cout << "before Le: \n" << Le << endl;

   // cout << "dim_pol_feat: " << dim_pol_feat << endl;
   // cout << "dist: " << dist << endl;
   // cout << "feat_u_vector: \n" << feat_u_vector.transpose() << endl;
   // cout << "feat_v_vector: \n" << feat_v_vector.transpose() << endl;
   // cout << "transformed_feat_vector inside feature propagation: " << transformed_feat_vector.transpose() << endl;

   // feat_vector.setZero(2 * dim_pol_feat);
   // for (int k = 0, kk = 0; k < 2 * dim_pol_feat && kk < dim_pol_feat; k++, kk++)
   // {
   //    feat_vector[k] = feat_u_vector[kk];
   //    feat_vector[k + 1] = feat_v_vector[kk];
   //    k++;
   // }

   // cout << "feat_vector: \n" << feat_vector.transpose() << endl;
   // cartesian_from_pixel_feat_vec = cartesian_from_pixel(feat_vector);
   // cout << "cartesian_from_pixel_vec: \n" << cartesian_from_pixel_feat_vec.transpose() << endl;
   // virtual_img_plane_transformation_vector = featuresTransformation(cartesian_from_pixel_feat_vec, cam_translation_vector, cam_rotation_vector);
   // cout << "virtual_img_plane_transformation_vector: \n" << virtual_img_plane_transformation_vector.transpose() << endl;
   // transformed_feat_vector = pixels_from_cartesian(virtual_img_plane_transformation_vector);
   // cout << "transformed_feat_vector after calculation: \n" << transformed_feat_vector.transpose() << endl;

   for (int k = 0, kk = 0; k < 2 * dim_pol_feat && kk < 2 * dim_pol_feat; k++, kk++)
   {
      // Le.row(k) << -1 / dist, 0.0, feat_u_vector[kk] / dist, feat_v_vector[kk];
      // Le.row(k + 1) << 0.0, -1 / dist, feat_v_vector[kk] / dist, -feat_u_vector[kk];
      Le.row(k) << -1 / dist, 0.0, feat_u_vector[kk] / dist, feat_u_vector[kk]*feat_v_vector[kk], -(1+feat_u_vector[kk]*feat_u_vector[kk]), feat_v_vector[kk];
      Le.row(k + 1) << 0.0, -1 / dist, feat_v_vector[kk] / dist, (1+feat_v_vector[kk]*feat_v_vector[kk]), -feat_u_vector[kk]*feat_v_vector[kk], -feat_u_vector[kk];
      k++;
   }
   // cout << "after Le: \n" << Le << endl;
   // cout << "Le shape: (" << Le.rows() << "," << Le.cols() << ")" << endl;
   // cout << "Interaction Matrix" << Le * camTwist << endl;
   return Le * camTwist;
}

// PVS-MPC Cost Function
double costFunction(unsigned int n, const double *x, double *grad, void *data)
{
   MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz>>((double *)x); 

   // cout << "Begin of PVS-NMPC cost function" << endl;

   // Trajectory of States (image features)
   MatrixXd traj_s(dim_s, mpc_hrz + 1);
   MatrixXd feature_hrz_prop(2*dim_pol_feat, mpc_hrz + 1);
   traj_s.setZero(dim_s, mpc_hrz + 1);
   // cout << "barx_meas: " << barx_meas << "\n" << endl;
   traj_s.col(0) << barx_meas[0], barx_meas[1], barx_meas[2], barx_meas[3];
   // cout << "traj_s.col(0): \n" << traj_s.col(0) << endl;
   
   // Progate the model (PVS with Image Jacobian)
   for (int k = 0; k < mpc_hrz; k++)
   {
      // cout << "Mpike to gamidi!!!" << endl;
      VectorXd feat_prop = IBVSSystem(inputs.col(k));
      feature_hrz_prop.col(k+1) = feature_hrz_prop.col(k) + feat_prop*mpc_dt;
      VectorXd sdot = Dynamic_System_x_y_reverted(inputs.col(k),feature_hrz_prop.col(k));
      
      traj_s.col(k + 1) = traj_s.col(k) + sdot * mpc_dt;
      // cout << "traj_s.col(k + 1) =" << traj_s.col(k + 1) << endl;
   }
   // cout << "traj_s = \n" << traj_s << endl;

   // Calculate Running Costs
   double Ji = 0.0; 

   //****DEFINE INITIAL DESIRED V****//
   s_des.col(0) << barx_des[0], barx_des[1], barx_des[2], barx_des[3];
   // cout << "s_des.col(0): \n" << s_des.col(0) << endl;
   // cout << "s_des before: " << s_des << endl;

   for (int k = 0; k < mpc_hrz; k++)
   {
      s_des.col(k) << barx_des[0], barx_des[1], barx_des[2], barx_des[3];
      // cout << "traj_s.col(k): " << traj_s.col(k) << endl;
      // cout << "s_des.col(k): " << s_des.col(k) << endl;
      ek = -s_des.col(k) + traj_s.col(k);
      // cout << "inside for loop ek: " << ek << endl ;
      // ek = traj_s.col(k)
      // cout << "s_des.col(k): " << s_des.col(k) << endl;
      // cout << "traj_s.col(k): " << traj_s.col(k) << endl;
      // cout << "traj_s.col(k): " << traj_s.col(k) << endl;
      Ji += ek.transpose() * Q * ek;
      Ji += inputs.col(k).transpose() * R * inputs.col(k);
   }
   // cout << "Ji: \n" << Ji << endl;
   // cout << "s_des after: " << s_des << endl;
   // cout << "traj_s: " << traj_s << endl;
   // cout << "ek: \n" << ek << endl;

   // Calculate Terminal Costs
   double Jt;
   VectorXd et = traj_s.col(mpc_hrz) - s_des.col(mpc_hrz);
   // cout << "traj_s.col(mpc_hrz): " << traj_s.col(mpc_hrz) << endl;
   // cout << "s_des.col(mpc_hrz): " << s_des.col(mpc_hrz) << endl;
   // cout << "et: \n" << et << endl;
   // cout << "Jt: \n" << Jt << endl;

   Jt = et.transpose() * P * et;
   //   cout << "Ji = " << Ji << " + " << "Jt = " << Jt << endl;

   return Ji + Jt;
}

//****DEFINE FOV CONSTRAINTS****//
void constraints(unsigned int m, double *c, unsigned int n, const double *x, double *grad, void *data)
{
   // cout << "Mpikan stin poli oi oxtroi" << endl;
   // Propagate the model.
   MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz>>((double *)x);
   // cout << "inputs = " << inputs << endl;
   // Trajectory of States (image features)
   MatrixXd traj_s(dim_s, mpc_hrz + 1);
   MatrixXd feature_hrz_prop(2*dim_pol_feat, mpc_hrz + 1);
   traj_s.setZero(dim_s, mpc_hrz + 1);
   traj_s.col(0) << barx_meas[0], barx_meas[1], barx_meas[2], barx_meas[3];
   // cout << "traj_s.col(0): " << traj_s.col(0) << endl;

   // Progate the model (IBVS with Image Jacobian)
   for (int k = 0; k < mpc_hrz; k++)
   {
      // cout << "Mpike to gamidi!!!" << endl;
      VectorXd feat_prop = IBVSSystem(inputs.col(k));
      // cout << "feat_prop: " << feat_prop << endl;
      feature_hrz_prop.col(k+1) = feature_hrz_prop.col(k) + feat_prop*mpc_dt;
      VectorXd sdot = Dynamic_System_x_y_reverted(inputs.col(k),feature_hrz_prop.col(k));
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


void state_vec_Callback(const std_msgs::Float64MultiArray::ConstPtr &state_vec_msg)
{
    // cout << "Starting state vector" << "\n" << endl;
    barx_meas.setZero(dim_s+3);
    // cout << "barx_meas before: " << barx_meas << "\n" << endl;   
    for (int i = 0; i < dim_s+3; i += 1){
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
    barx_des.setZero(dim_s+3);
    // cout << "barx_des before: " << barx_des << "\n" << endl;

    for (int i = 0; i < dim_s+3; i += 1){
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
    feat_u_vector.setZero(dim_pol_feat);
    // cout << "feat_u_vector before: " << feat_u_vector << "\n" << endl;

    for (int i = 0; i < dim_pol_feat; i += 1){
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
    feat_v_vector.setZero(dim_pol_feat);
    // cout << "feat_v_vector before: " << feat_v_vector << "\n" << endl;

    for (int i = 0; i < dim_pol_feat; i += 1){
      feat_v_vector[i] = feat_v_vec_msg->data[i];
    //   cout << "feat_v_vector[i]: " << feat_v_vector[i] << "\n" << endl;
    //   cout << "after add feat_v_vector: " << feat_v_vector << "\n" << endl;

    }
   //  cout << "feat_v_vector: " << feat_v_vector.transpose() << "\n" << endl;
    // cout << "Done with feature v vector" << "\n" << endl;
    flag = 1;
}

void cam_trans_Callback(const std_msgs::Float64MultiArray::ConstPtr &cam_trans_vec_msg){
   cam_translation_vector.setZero(3);
   cam_translation_vector << cam_trans_vec_msg->data[0], cam_trans_vec_msg->data[1], cam_trans_vec_msg->data[2];
   // cout << "cam_translation_vector: " << cam_translation_vector << endl;
   flag = 1;
}

void cam_rotation_Callback(const std_msgs::Float64MultiArray::ConstPtr &cam_rot_vec_msg){
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

   // Initialize MPC Variables
   s_des.setZero(dim_s, mpc_hrz + 1);
   s_abs.setZero(dim_s);  
   // s_abs << 1024-512, 1024-512, 1.0, 3.14, 0.8;
   s_abs << 1920-960, 1080-540, 1.0, 3.14;
   

   //****SET MPC COST FUNCTION MATRICES****//
   Q.setIdentity(dim_s, dim_s);
   R.setIdentity(dim_inputs, dim_inputs);
   P.setIdentity(dim_s, dim_s);

   // Q = 75 * Q;
   // Q = 150 * Q;
   Q = 5 * Q;
   R = 5 * R;
   P = 1 * Q;

   R(3,3) = 25;
   R(4,4) = 25;

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
         optNum = nlopt_optimize(opt, inputs, &minJ);
         // cout << "Optimization Return Code: " << nlopt_optimize(opt, inputs, &minJ) << endl;
         // cout << "Optimization Return Code: " << optNum << endl;
      }
      printf("found minimum at J(%g,%g,%g,%g,%g,%g) = %g\n", inputs[0], inputs[1],inputs[2], inputs[3], inputs[4], inputs[5], minJ);
      
      double end = ros::Time::now().toSec();
      double tf = ros::WallTime::now().toSec();
      double timer = tf - t0;
      double dt = end - start;
      realtime = realtime + dt;
         
      // VectorXd cmd_vel(dim_inputs+2);
      VectorXd cmd_vel(dim_inputs);
      cmd_vel << 50*inputs[0], 50*inputs[1], 100*inputs[2], -1000*inputs[3], 1000*inputs[4], -120*inputs[5];
      // cmd_vel << 1250*inputs[0], 1250*inputs[1], 1200*inputs[2], 0.0, 0.0, -25000*inputs[3];
      // cmd_vel << inputs[0], inputs[1], inputs[2], 0.0, 0.0, inputs[3];
      // cmd_vel << 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
      // cmd_vel << 125*inputs[0], 125*inputs[1], 125*inputs[2], 0.0, 0.0, -1800*inputs[3];
      // cout << "cmd_vel: " << cmd_vel << endl;
            
      std_msgs::Float64MultiArray cmd_vel_Msg;

      for(int i=0; i<cmd_vel.size(); i++){
         cmd_vel_Msg.data.push_back(cmd_vel[i]);
      }
         
      printf("Camera Velocity is u,v,z,Oz(%g,%g,%g,%g,%g,%g) =", cmd_vel[0], cmd_vel[1], cmd_vel[2], cmd_vel[3], cmd_vel[4], cmd_vel[5]);   
      cmd_vel_pub.publish(cmd_vel_Msg);

      ros::spinOnce();
      // ros::spin;
      loop_rate.sleep();
   }

   nlopt_destroy(opt);
   return 0;
}
