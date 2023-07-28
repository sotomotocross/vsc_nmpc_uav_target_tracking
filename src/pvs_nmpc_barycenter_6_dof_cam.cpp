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
const int dim_inputs = 4;
// const int dim_inputs = 6;
int dim_s = 5;
int dim_pol_feat = 6;
// int dim_s = 2*features_n;
const int mpc_hrz = 50;  // 10
double mpc_dt = 0.01; //0.001
// double mpc_dt = 0.001; //0.001
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
VectorXd barx_meas(dim_s);
VectorXd barx_des(dim_s);
VectorXd feat_u_vector(dim_pol_feat);
VectorXd feat_v_vector(dim_pol_feat);

int flag = 0;


VectorXd Dynamic_System_x_y_reverted(VectorXd camTwist, VectorXd feat_prop)
{
   // cout << "Mpike ki edo (PVS) to gamidi!!!" << endl;
   MatrixXd model_mat(dim_s, dim_inputs);
   // cout << "model_mat shape: (" << model_mat.rows() << "," << model_mat.cols() << ")" << endl;

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

   model_mat << 
            -1/barx_meas[4], 0.0, barx_meas[0]/barx_meas[4], barx_meas[1],
            0.0, -1/barx_meas[4], barx_meas[1]/barx_meas[4], -barx_meas[0],
            0.0, 0.0, 2/barx_meas[4], 0.0,
            0.0, 0.0, 0.0, g_4_6,
            0.0, 0.0, 1.0, 0.0;
   // model_mat << 
   //          -1/Z0, 0.0, transformed_s_bar_x/Z0, transformed_s_bar_y,
   //          0.0, -1/Z0, transformed_s_bar_y/Z0, -transformed_s_bar_x,
   //          0.0, 0.0, 2/Z0, 0.0,
   //          0.0, 0.0, 0.0, 0.0,
   //          0.0, 0.0, 1.0, 0.0;

   // cout << "after model_mat: " << model_mat << endl;

   // cout << "model_mat shape: (" << model_mat.rows() << "," << model_mat.cols() << ")" << endl;
   // cout << "Interaction Matrix:\n" << model_mat << endl;
   // cout << "Interaction Matrix" << model_mat * camTwist << endl;

   return model_mat * camTwist;
}

// IBVS Feature Rate Le*Vk
VectorXd IBVSSystem(VectorXd camTwist)
{
//    cout << "Mpike kai sto feature propagation to gamidi!!!" << endl;
   MatrixXd Le(2*dim_pol_feat,dim_inputs);
   
   //,Le.setZero(dim_s,dim_inputs);
   Le.setZero(2*dim_pol_feat,dim_inputs);
   // cout << "Le: \n" << Le << endl;

   // Le.row(0) << -1/Z0, 0.0, transformed_features[0]/Z0, transformed_features[1];
   // Le.row(1) << 0.0, -1/Z0, transformed_features[1]/Z0, -transformed_features[0];

   // cout << "before Le: \n" << Le << endl;

   for (int k = 0, kk = 0; k < 2*dim_pol_feat && kk < 2*dim_pol_feat ; k++, kk++){
      Le.row(k) << -1/barx_meas[4], 0.0, feat_u_vector[kk]/barx_meas[4], feat_v_vector[kk];
      Le.row(k+1) << 0.0, -1/barx_meas[4], feat_v_vector[kk]/barx_meas[4], -feat_u_vector[kk];
      k++;
   }
//    cout << "after Le: \n" << Le << endl;
   return Le*camTwist;
}

// PVS-MPC Cost Function
double costFunction(unsigned int n, const double *x, double *grad, void *data)
{
   MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz>>((double *)x); 

   // Trajectory of States (image features)
   MatrixXd traj_s(dim_s, mpc_hrz + 1);
   MatrixXd feature_hrz_prop(2*dim_pol_feat, mpc_hrz + 1);
   traj_s.setZero(dim_s, mpc_hrz + 1);
   // cout << "barx_meas: " << barx_meas << "\n" << endl;
   traj_s.col(0) << barx_meas[0], barx_meas[1], barx_meas[2], barx_meas[3], barx_meas[4];
   cout << "traj_s.col(0): \n" << traj_s.col(0) << endl;
   
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
   s_des.col(0) << barx_des[0], barx_des[1], barx_des[2], barx_des[3], barx_des[4];
   cout << "s_des.col(0): \n" << s_des.col(0) << endl;

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
   MatrixXd feature_hrz_prop(2*dim_pol_feat, mpc_hrz + 1);
   traj_s.setZero(dim_s, mpc_hrz + 1);
   traj_s.col(0) << barx_meas[0], barx_meas[1], barx_meas[2], barx_meas[3], barx_meas[4];
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
    barx_meas.setZero(dim_s);
    // cout << "barx_meas before: " << barx_meas << "\n" << endl;

    for (int i = 0; i < dim_s; i += 1){
      barx_meas[i] = state_vec_msg->data[i];
    //   cout << "barx_meas[i]: " << barx_meas[i] << "\n" << endl;
    //   cout << "after add barx_meas: " << barx_meas << "\n" << endl;

    }
   //  cout << "barx_meas: " << barx_meas << "\n" << endl;
    // cout << "Done with state vector" << "\n" << endl;
    flag = 1;
   
}

void state_vec_des_Callback(const std_msgs::Float64MultiArray::ConstPtr &state_vec_des_msg)
{
    // cout << "Starting state vector" << "\n" << endl;
    barx_des.setZero(dim_s);
    // cout << "barx_des before: " << barx_des << "\n" << endl;

    for (int i = 0; i < dim_s; i += 1){
      barx_des[i] = state_vec_des_msg->data[i];
    //   cout << "barx_des[i]: " << barx_des[i] << "\n" << endl;
    //   cout << "after add barx_des: " << barx_des << "\n" << endl;

    }
   //  cout << "barx_des: " << barx_des << "\n" << endl;
    // cout << "Done with state vector" << "\n" << endl;
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
   //  cout << "feat_u_vector: " << feat_u_vector << "\n" << endl;
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
   //  cout << "feat_v_vector: " << feat_v_vector << "\n" << endl;
    // cout << "Done with feature v vector" << "\n" << endl;
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

   // Create subscribers
   ros::Publisher cmd_vel_pub = nh.advertise<std_msgs::Float64MultiArray>("/cmd_vel", 1);
   

   // Initialize MPC Variables
   s_des.setZero(dim_s, mpc_hrz + 1);
   s_abs.setZero(dim_s);  
   s_abs << 1024-512, 1024-512, 1.0, 3.14, 0.8;
   

   //****SET MPC COST FUNCTION MATRICES****//
   Q.setIdentity(dim_s, dim_s);
   R.setIdentity(dim_inputs, dim_inputs);
   P.setIdentity(dim_s, dim_s);

   Q = 50 * Q;
   // Q = 150 * Q;
   R = 5 * R;
   P = 1 * Q;

   // cout << "Q shape: (" << Q.rows() << "," << Q.cols() << ")" << endl;
   // cout << "R shape: (" << R.rows() << "," << R.cols() << ")" << endl;
   // cout << "P shape: (" << P.rows() << "," << P.cols() << ")" << endl;

   // Q(0,0) = 650;
	// Q(1,1) = 650;
	// Q(2,2) = 1;//1.0;
   // Q(3,3) = 180; //180
   Q(4,4) = 0.0000001;

   // Q(0,0) = 150;
	// Q(1,1) = 150;
	// Q(2,2) = 150;//1.0;
   // Q(3,3) = 150; //180
   // Q(4,4) = 150;

   // R(0,0) = 5;//5;
   // R(1,1) = 10;//10;
	// R(2,2) = 750;///750;
	// R(3,3) = 5;//18;

   // R(0,0) = 2;//5;
   // R(1,1) = 3;//10;
	// R(2,2) = 5;///750;
	// R(3,3) = 5;//18;

   // P(0,0) = 1;
   // P(1,1) = 1;
   // P(2,2) = 1;
   // P(3,3) = 18;
   // P(4,4) = 1;

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
         
      VectorXd cmd_vel(dim_inputs+2);
      cmd_vel << 50*inputs[0], 50*inputs[1], 25*inputs[2], 0.0, 0.0, -15*inputs[3];
      // cmd_vel << 1000*inputs[0], 1000*inputs[1], -1000*inputs[2], 0.0, 0.0, inputs[3];
      // cmd_vel << inputs[0], inputs[1], 2.0, 0.0, 0.0, inputs[3];
      // cout << "cmd_vel: " << cmd_vel << endl;
            
      std_msgs::Float64MultiArray cmd_vel_Msg;

      for(int i=0; i<cmd_vel.size(); i++){
         cmd_vel_Msg.data.push_back(cmd_vel[i]);
      }
         
      // printf("Camera Velocity is u,v,z,Oz(%g,%g,%g,%g) =", inputs[0], inputs[1], inputs[2], inputs[3]);   
      cmd_vel_pub.publish(cmd_vel_Msg);

      ros::spinOnce();
      // ros::spin;
      loop_rate.sleep();
   }

   nlopt_destroy(opt);
   return 0;
}