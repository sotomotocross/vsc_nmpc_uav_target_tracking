#include "ros/ros.h"
#include "geometry_msgs/Twist.h"
#include "mavros_msgs/PositionTarget.h"
#include "img_seg_cnn/PREDdata.h"
#include "std_msgs/Float64.h"
#include "mpcpack/rec.h"

#include <cstdlib>
#include <iostream>
#include <stdio.h>

#include <math.h>
#include <nlopt.hpp>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

// Global MPC Variables
//  int features_n = 4;
int features_n = 1;
const int dim_inputs = 4;
int dim_s = 4;
// int dim_s = 2*features_n;
const int mpc_hrz = 6;
double mpc_dt = 0.1;
double l = 252.07;
double a = 10;
double optNum;

MatrixXd pred_s(dim_s, mpc_hrz + 1);
MatrixXd s_des(dim_s, mpc_hrz + 1);
MatrixXd traj_s(dim_s, mpc_hrz + 1);
VectorXd f_des(dim_s);
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
VectorXd x_prev;
VectorXd e_prev;
VectorXd u_prev;
VectorXd xk;
VectorXd xk_pred;
VectorXd ekm2;

// Simulator camera parameters
double umax = 720;
double umin = 0;
double vmax = 480;
double vmin = 0;
double cu = 360.5;
double cv = 240.5;

int flag = 0;

// Camera Frame Update Callback Variables
double x0, g0, Z0;
double x1, g1, Z1;
double x2, g2, Z2;
double x3, g3, Z3;
double f0, h0;
double f1, h1;
double f2, h2;
double f3, h3;
double Tx, Ty, Tz, Oz;
double d01, d03;

double u0, v0, u1, v1, u2, v2, u3, v3;
double barycenter_x, barycenter_y;
double barycenter_x_des, barycenter_y_des;
double sigma;
double alpha;
double sigma_des = 14850.0;
double alpha_des = 0.0;

// Camera Frame Desired Features
// Simulator (720*480)
double u0d = 340;
double v0d = 480;
double u1d = 340;
double v1d = 0;
double u2d = 380;
double v2d = 0;
double u3d = 380;
double v3d = 480;

double x0d = (u0d - cu) / l;
double g0d = (v0d - cv) / l;
double x1d = (u1d - cu) / l;
double g1d = (v1d - cv) / l;
double x2d = (u2d - cu) / l;
double g2d = (v2d - cv) / l;
double x3d = (u3d - cu) / l;
double g3d = (v3d - cv) / l;

double sc_x = (umax - cu) / l;
double sc_y = (vmax - cv) / l;

// PVS Feature Rate Lm_cal*Vk
VectorXd PVSSystem(VectorXd camTwist)
{
    // cout << "Mpike ki edo (PVS) to gamidi!!!" << endl;
    MatrixXd Lm_cal(dim_s, dim_inputs);
    // cout << "Lm_cal shape: (" << Lm_cal.rows() << "," << Lm_cal.cols() << ")" << endl;
    //,Lm_cal.setZero(dim_s,dim_inputs);
    // cout << "camTwist shape: (" << camTwist.rows() << "," << camTwist.cols() << ")" << endl;

    // cout << "(x0,g0,Z0): (" << x0 << "," << g0 << "," << Z0 << ")" << endl;
    // cout << "(x1,g1,Z1): (" << x1 << "," << g1 << "," << Z1 << ")" << endl;
    // cout << "(x2,g2,Z2): (" << x2 << "," << g2 << "," << Z2 << ")" << endl;
    // cout << "(x3,g3,Z3): (" << x3 << "," << g3 << "," << Z3 << ")" << endl;

    // Lm_cal <<-1/Z0, 0.0, x0/Z0, g0,
    //          0.0, -1/Z0, g0/Z0, -x0,
    //          -1/Z1, 0.0, x1/Z1, g1,
    //          0.0, -1/Z1, g1/Z1, -x1;
    Lm_cal << -1 / Z0, 0.0, -1 / Z0, 0.0,
        0.0, -1 / Z0, 0.0, -1 / Z0,
        -1 / Z1, 0.0, -1 / Z1, 0.0,
        0.0, -1 / Z1, 0.0, -1 / Z1;

    // cout << "Lm_cal: " << Lm_cal << endl;

    // cout << "Lm_cal shape: (" << Lm_cal.rows() << "," << Lm_cal.cols() << ")" << endl;
    // cout << "Interaction Matrix:\n" << Lm_cal << endl;
    // cout << "Interaction Matrix" << Lm_cal*camTwist << endl;

    return Lm_cal * camTwist;
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

// PVS-MPC Cost Function
double costFunction(unsigned int n, const double *x, double *grad, void *data)
{

    MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz>>((double *)x);

    // cout << inputs << endl;
    // cout << endl;

    // Trajectory of States (image features)
    MatrixXd traj_s(dim_s, mpc_hrz + 1);
    traj_s.setZero(dim_s, mpc_hrz + 1);
    traj_s.col(0) << barycenter_x, barycenter_y, sigma, alpha;
    // cout << "traj_s: \n" << traj_s << endl;
    // cout << "inputs: " << inputs << endl;
    pred_s.setZero(dim_s, mpc_hrz + 1);
    pred_s.col(0) = traj_s.col(0);
    // cout << "before propagation pred_s: " << pred_s << endl;
    // cout << "before propagation pred_s.col(0): " << pred_s.col(0) << endl;

    if (Z0 != 0 && Z1 != 0 && Z2 != 0 && Z3 != 0)
    {
        // Progate the model (PVS with Image Jacobian)
        for (int k = 0; k < mpc_hrz; k++)
        {
            // cout << "Mpike to gamidi!!!" << endl;
            //   VectorXd sdot = IBVSSystem(inputs.col(k));
            VectorXd sdot = PVSSystem(inputs.col(k));

            // cout << "s_dot: " << sdot << endl;
            // cout << "sdot * mpc_dt: " << sdot * mpc_dt << endl;

            // cout << "traj_s: " << traj_s << endl;
            // cout << "traj_s.col(k + 1): " << traj_s.col(k + 1) << endl;

            traj_s.col(k + 1) = traj_s.col(k) + sdot * mpc_dt;

            // cout << "after math traj_s: " << traj_s << endl;
            // cout << "after math traj_s.col(k + 1): " << traj_s.col(k + 1) << endl;

            pred_s.col(k + 1) = traj_s.col(k + 1);

            // cout << "after propagation pred_s: " << pred_s << endl;
            // cout << "after propagation pred_s.col(k + 1): " << pred_s.col(k + 1) << endl;
        }
    }

    // Calculate Running Costs
    double Ji = 0.0;

    //   cout << "traj_s =" << traj_s << endl;

    //****DEFINE INITIAL DESIRED V****//
    // VectorXd s_des(dim_s);
    //   s_des.col(0) << x0d,g0d,x1d,g1d,x2d,g2d,x3d,g3d;
    MatrixXd s_des(dim_s, mpc_hrz + 1);
    s_des.col(0) << barycenter_x_des, barycenter_y_des, sigma_des, alpha_des;
    // cout << "s_des: \n" << s_des << endl;

    //****SET V DESIRED VELOCITY FOR THE VTA****//
    double b = 10;
    VectorXd s_at(dim_s);
    s_at.setZero(dim_s);
    s_at << 0, b / l, 0, 0;
    //   s_at  << 0,b/l,0,b/l,0,b/l,0,b/l;

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
        // ek = traj_s.col(k)
        // cout << "traj_s.col(k): " << traj_s.col(k) << endl;
        // cout << "traj_s.col(k).row(2): " << traj_s.col(k).row(2) << endl;
        // cout<< "sigma: " << sigma << "\n" << endl;
        ek.row(2) << log(sigma / sigma_des);
        // cout << "ek: " << ek << endl;

        Ji += ek.transpose() * Q * ek;
        Ji += inputs.col(k).transpose() * R * inputs.col(k);
    }
    // cout << "whole ek: " << ek << endl;

    // cout << "ek = " << ek << endl;

    // Calculate Terminal Costs
    double Jt;
    VectorXd et = traj_s.col(mpc_hrz) - s_des.col(mpc_hrz);
    // cout << "traj_s.col(mpc_hrz): " << traj_s.col(mpc_hrz) << endl;
    // cout << "traj_s.col(mpc_hrz).row(2): " << traj_s.col(mpc_hrz).row(2) << endl;
    // cout<< "sigma: " << sigma << "\n" << endl;
    et.row(2) << log(sigma / sigma_des);
    // cout << "et = " << et << endl;

    Jt = et.transpose() * P * et;
    //   cout << "Ji = " << Ji << " + " << "Jt = " << Jt << endl;

    // cout << "barycenter_x: " << barycenter_x << endl;
    // cout << "barycenter_y: " << barycenter_y << endl;
    // cout << "barycenter_x_des: " << barycenter_x_des << endl;
    // cout << "barycenter_y_des: " << barycenter_y_des << endl;
    // cout << "alpha: " << alpha << endl;
    // cout << "sigma: " << sigma << endl;
    // cout << "alpha_des: " << alpha_des << endl;
    // cout << "sigma_des: " << sigma_des << endl;

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
    traj_s.setZero(dim_s, mpc_hrz + 1);
    // traj_s.col(0) << x0,g0,x1,g1,x2,g2,x3,g3;
    traj_s.col(0) << barycenter_x, barycenter_y, sigma, alpha;
    // cout << "traj_s.col(0): " << traj_s.col(0) << endl;

    // Progate the model (IBVS with Image Jacobian)
    for (int k = 0; k < mpc_hrz; k++)
    {
        // VectorXd sdot = IBVSSystem(inputs.col(k));
        VectorXd sdot = PVSSystem(inputs.col(k));
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
void featureCallback(const img_seg_cnn::PREDdata::ConstPtr &s_message)
{
    f0 = s_message->box_1[0];
    h0 = s_message->box_1[1];

    f1 = s_message->box_2[0];
    h1 = s_message->box_2[1];

    f2 = s_message->box_3[0];
    h2 = s_message->box_3[1];

    f3 = s_message->box_4[0];
    h3 = s_message->box_4[1];

    // cout << "Feature callback information" << endl;

    barycenter_x = s_message->cX / l;
    // cout << "barycenter_x: " << barycenter_x << endl;
    barycenter_y = s_message->cY / l;
    // cout << "barycenter_y: " << barycenter_y << endl;

    barycenter_x_des = cu / l;
    // cout << "barycenter_x_des: " << barycenter_x_des << endl;
    barycenter_y_des = cv / l;
    // cout << "barycenter_y_des: " << barycenter_y_des << endl;

    alpha = s_message->alpha;
    // cout << "alpha: " << alpha << endl;
    sigma = s_message->sigma;
    // cout << "sigma: " << sigma << endl;
    // cout << "alpha_des: " << alpha_des << endl;
    // cout << "sigma_des: " << sigma_des << "\n" << endl;

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

    // printf("Image Features for Point 2 are (%g,%g) =", u0, v0);
    // printf("Image Features for Point 2 are (%g,%g) =", u1, v1);
    // printf("Image Features for Point 3 are (%g,%g) =", u2, v2);
    // printf("Image Features for Point 4 are (%g,%g) =", u3, v3);
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
    ros::Subscriber feature_sub = nh.subscribe<img_seg_cnn::PREDdata>("/pred_data", 10, featureCallback);
    ros::Subscriber alt_sub = nh.subscribe<std_msgs::Float64>("/mavros/global_position/rel_alt", 10, altitudeCallback);

    // Create subscribers
    ros::Publisher vel_pub = nh.advertise<mavros_msgs::PositionTarget>("/mavros/setpoint_raw/local", 1);
    ros::Publisher rec_pub = nh.advertise<mpcpack::rec>("/mpcpack/msg/rec", 1);

    // cout << "int main initial information" << endl;
    // cout << "barycenter_x: " << barycenter_x << endl;
    // cout << "barycenter_y: " << barycenter_y << endl;
    // cout << "barycenter_x_des: " << barycenter_x_des << endl;
    // cout << "barycenter_y_des: " << barycenter_y_des << endl;
    // cout << "alpha: " << alpha << endl;
    // cout << "sigma: " << sigma << endl;
    // cout << "alpha_des: " << alpha_des << endl;
    // cout << "sigma_des: " << sigma_des << endl;

    // Initialize MPC Variables
    s_des.setZero(dim_s, mpc_hrz + 1);
    s_abs.setZero(dim_s);
    //  s_abs << sc_x,sc_y,sc_x,sc_y,sc_x,sc_y,sc_x,sc_y;
    s_des.setZero(dim_s, mpc_hrz + 1);
    s_bc.setOnes(dim_s);
    s_br.setOnes(dim_s);
    s_abs.setZero(dim_s);
    s_abs << barycenter_x, barycenter_y, sigma, alpha;
    x_prev.setZero(dim_s);
    e_prev.setZero(dim_s);
    u_prev.setZero(dim_inputs);
    xk.setZero(dim_s);
    xk_pred.setZero(dim_s);
    f_des.setZero(dim_s);
    f_des << barycenter_x_des, barycenter_y_des, sigma_des, alpha_des;
    ekm2.setZero(dim_s);

    // cout << "s_abs: " << s_abs << endl;
    // cout << "int main after information" << endl;
    // cout << "barycenter_x: " << barycenter_x << endl;
    // cout << "barycenter_y: " << barycenter_y << endl;
    // cout << "barycenter_x_des: " << barycenter_x_des << endl;
    // cout << "barycenter_y_des: " << barycenter_y_des << endl;
    // cout << "alpha: " << alpha << endl;
    // cout << "sigma: " << sigma << endl;
    // cout << "alpha_des: " << alpha_des << endl;
    // cout << "sigma_des: " << sigma_des << endl;

    // cout << "int main information \n" << endl;
    // cout << "barycenter_x: " << barycenter_x << endl;
    // cout << "barycenter_y: " << barycenter_y << endl;
    // cout << "alpha: " << alpha << endl;
    // cout << "sigma: " << sigma << endl;
    // cout << "barycenter_x_des: " << barycenter_x_des << endl;
    // cout << "barycenter_y_des: " << barycenter_y_des << endl;

    //****SET MPC COST FUNCTION MATRICES****//
    Q.setIdentity(dim_s, dim_s);
    R.setIdentity(dim_inputs, dim_inputs);
    P.setIdentity(dim_s, dim_s);

    Q = 15 * Q;
    R = 5 * R;
    P = 1 * Q;

    R(0, 0) = 5;
    R(1, 1) = 25;
    R(2, 2) = 500;
    R(3, 3) = 25;

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

    //****DEFINE TRIGGERING CONDITION VARIABLES****//
    double trig;
    double cond;

    //****INITIALIZE TIME VARIABLES****//
    double t0 = ros::WallTime::now().toSec();
    // printf("Start time:%lf\n", t0);

////****LABEL INDICATING THE START OF THE OPTIMIZATION****//
optlabel:
    double start = ros::Time::now().toSec();
    //***** OPTIMISATION TRIGGERING	******
    cout << "Optimization Return Code: " << nlopt_optimize(opt, inputs, &minJ) << endl;
    cout << "OPTIMIZATION WAS RUN" << endl;
    double end = ros::Time::now().toSec();

    //*****REINITIALIZE MPC VARIABLES FOR TRIGGERING******
    double realtime = 0;
    double Lzm = 0;
    double Fbar = 0;
    int m = 0;
    VectorXd uv(dim_s);
    uv.setZero(dim_s);
    VectorXd uv_prev(dim_s);
    uv_prev.setZero(dim_s);

    // cout << "Pes mou oti ftanei edw" << endl;

    //****RUNNING LOOP****//
    while (ros::ok())
    {
        // cout << "Pes mou oti mpainei kai edw" << endl;
        if (barycenter_x != 0 && barycenter_y != 0)
        {
            // cout << "Ama den mpainei edw tha gamithoume" << endl;
            //****CREATE MESSAGE TO SAVE DATA****//
            mpcpack::rec fdataMsg;

            //****START OF THE LOOP****//
            // cout << "NEW LOOP FOR m   = " << m << "   HAS STARTED" << endl;

            //****SAVE DATA****//
            // xk << x0, g0, x1, g1, x2, g2, x3, g3;
            xk << barycenter_x, barycenter_y, sigma, alpha;
            // uv << u0, v0, u1, v1, u2, v2, u3, v3;
            // cout << "x current is  = " << xk << endl;
            // cout << "uv current is = " << uv << endl;

            // cout << "traj before swap NaN for 0  = " << traj_s << endl;

            //****SWAP NaN FOR 0****//
            for (int i = 0; i < dim_s; i++)
            {
                for (int j = 0; j < mpc_hrz + 1; j++)
                {
                    if (isnan(traj_s(i, j)))
                    {
                        traj_s(i, j) = 0;
                    }
                }
            }
            // cout << "traj after swap NaN for 0  = " << traj_s << endl;

            //****DEFINE TIMER****//
            double tf = ros::WallTime::now().toSec();
            double timer = tf - t0;
            // cout << "timer = " << timer << endl;

            //****EVENT-TRIGGERING BLOCK - START****//
            //
            if (m > 0)
            {
                // cout << "Edw mpainei????" << endl;
                //****LIPSCHITZ CONSTANT Lf****//
                double Tzmax = inputs_ub[2];
                double Ozmax = inputs_ub[3];

                double Lf1 = 4 * pow(1 + (Tzmax * mpc_dt) / Z0, 2);
                double Lf2 = 4 * pow(Ozmax * mpc_dt, 2);
                double Lf = Lf1;
                if (Lf1 < Lf2)
                {
                    Lf = Lf2;
                }
                Lf = sqrt(2 * Lf);

                /*  ****In case an alternative norm wants to be used*****
                // Lf 1-norm & Inf-norm
                double Lf1 = 1+(Tzmax*mpc_dt)/Z0;
                double Lf2 = Ozmax*mpc_dt;
                double Lf = Lf1;
                if (Lf1 < Lf2){
                Lf = Lf2;
                }
                Lf = 2*Lf;
                //Lf = 1.05;
                */

                // cout << "Lf =  " << Lf << endl;
                // cout << "Lf1 =  " << Lf1 << endl;
                // cout << "Lf2 =  " << Lf2 << endl;

                //****LIPSCHITZ CONSTANT Lc****//
                double EoC = sqrt(features_n * (pow(sc_x, 2) + pow(sc_y, 2)));
                // cout << "EoC = " << EoC << endl;
                // cout << "sc_x, sc_y =" << sc_x << "+" << sc_y <<  endl;
                JacobiSVD<MatrixXd> svd1(Q);
                // cout << "Q singular values are:" << endl << svd1.singularValues() << endl;
                VectorXd Qsvd;
                Qsvd.setZero(dim_s);
                for (int k = 0; k < dim_s; k++)
                {
                    Qsvd(k) = svd1.singularValues()[k];
                }
                double maxQ = Qsvd.maxCoeff();
                double Lc = 2 * EoC * maxQ;
                // cout << "Lc =  " << Lc << endl;

                //****LIPSCHITZ CONSTANT Lv****//
                double EoP = 0.3;
                JacobiSVD<MatrixXd> svd2(P);
                // cout << "P singular values are:" << endl << svd2.singularValues() << endl;
                VectorXd Psvd;
                Psvd.setZero(dim_s);
                for (int k = 0; k < dim_s; k++)
                {
                    Psvd(k) = svd2.singularValues()[k];
                }
                double maxP = Psvd.maxCoeff();
                double Lv = 2 * EoP * maxP;
                // cout << "Lv =  " << Lv << endl;

                //****VTA ERROR MAXIMUM THRESHOLD****//
                double aef = 0;
                double ae = P(0, 0) * pow(EoP, 2);
                double xi = (ae - aef) / (Lv * pow(Lf, mpc_hrz - 1 - (m - 1)));
                // cout << "xi = " << xi << endl;

                //****Lzm TRIGGERING CONSTANT****//
                Lzm = Lv * pow(Lf, mpc_hrz - 1 - (m - 1)) + Lc * (pow(Lf, mpc_hrz - 1 - (m - 1)) - 1) / (Lf - 1);
                // cout << "Lzm =  " << Lzm << endl;

                //****LYAPUNOV FUNCTION FOR THE TRIGGERING CONDITION****//
                u_prev << inputs[(m - 1) * dim_inputs], inputs[(m - 1) * dim_inputs + 1], inputs[(m - 1) * dim_inputs + 2], inputs[(m - 1) * dim_inputs + 3];
                e_prev << x_prev - f_des;
                // cout << "x_prev: " << x_prev << endl;
                // cout << "x_prev.row(2): " << x_prev.row(2) << endl;
                // cout << "sigma: " << sigma << endl;
                // cout << "f_des: " << f_des << endl;
                // cout << "f_des.row(2): " << f_des.row(2) << endl;
                // cout << "sigma_des: " << sigma_des << "\n" << endl;
                e_prev.row(2) << log(sigma / sigma_des);
                // e_prev << x_prev;
                //  cout << "u_prev " << u_prev << endl;

                Fbar = Fbar + R(0, 0) * (e_prev).squaredNorm();

                // cout << "x_prev = " << x_prev << endl;
                // cout << "u_prev =" << u_prev << endl;
                // cout << "e_prev =" << e_prev << endl;
                // cout << "Fbar = " << Fbar << endl;
                // cout << "s_des.col(0) =" << f_des << endl;

                //****PREDICTED VS REAL STATE ERROR NORM e(k+m|k-1)****//
                // cout << "pred_s: " << pred_s << endl;
                xk_pred << pred_s.col(m);
                // cout << "xk_pred: " << xk_pred << endl;
                // cout << "xk: " << xk << endl;

                ekm2 << xk - xk_pred;
                // cout << "ekm2: " << ekm2 << endl;

                double ekm = ekm2.squaredNorm();
                // cout << "ekm: " << ekm << endl;

                ekm2.row(2) << log(sigma / sigma_des);
                // cout << "log ekm2: " << ekm2 << endl;

                double alt_ekm = ekm2.squaredNorm();
                // cout << "alt_ekm: " << alt_ekm << endl;

                // double ekm2 = (xk - xk_pred).squaredNorm();
                // cout << "ekm2: " << ekm2 << endl;
                // cout << "xk_pred.squaredNorm()" << xk_pred.squaredNorm() << endl;
                // cout << "xk.squaredNorm()" << xk.squaredNorm() << endl;

                // VectorXd ekm2_subtraction;
                // ekm2_subtraction << xk - xk_pred;
                // cout << "ekm2_subtraction: " << ekm2_subtraction << endl;
                // ekm2_subtraction.row(2) << log(xk.row(2)./xk_pred.row(2));

                // double ekm2 = (xk-xk_pred).squaredNorm();
                // double ekm2;

                // ekm2.row(0) = xk(0) -xk_pred(0);
                // ekm2.row(1) = xk(1) -xk_pred(1);
                // ekm2(0) = xk(0) -xk_pred(0);
                // ekm2.row(3) = xk(3) -xk_pred(3);
                // ekm2 << xk - xk_pred;
                // cout << "ekm2: " << ekm2 << endl;
                // ekm2.row(2) << log.sigma(xk.row(2)/xk_pred.row(2));
                // ekm2 = ekm2.squaredNorm();

                // cout << "xk = " << xk << endl;
                // cout << "xk_pred = " << xk_pred << endl;
                // cout << " (xk-xk_pred).squaredNorm() = " << (xk-xk_pred).squaredNorm() << endl;
                // cout << " |e(k+m|k-1)|_2 = " << ekm2 << endl;
                // cout << " |e(k+m|k-1)|_infty = " << (xk-xk_pred).lpNorm<Infinity>() << endl;

                //****THEORETICAL FEASIBILITY CHECK****//
                if (alt_ekm <= xi)
                {

                    // cout << "Feasibility is Ensured " << "\n" << " alt_ekm = " << alt_ekm << "<" << "xi = " << xi << endl;
                    //****EVENT-TRIGGERING CONDITION****//
                    // double first_Lzm2 = Lzm * (xk - xk_pred).squaredNorm();
                    // double second_Lzm2 = Lzm * alt_ekm;
                    // double Lzm1 = Lzm*(xk-xk_pred).norm();
                    // double Lzminf = Lzm*(xk-xk_pred).lpNorm<Infinity>();
                    // double trigger_sigma = 1.0;
                    // cout << "Lzm <  " << Lzm2 << " trigger_sigma* Fbar  " << trigger_sigma*Fbar << endl;

                    // if (Lzm2 > trigger_sigma * Fbar)
                    // {
                    //     cout << " EVENT OCCURED, MPC MUST RUN AGAN " << endl;
                    //     trig = 5;
                    //     fdataMsg.trig = trig;
                    //     fdataMsg.time = timer;
                    //     goto optlabel;
                    // }
                    // else
                    // {
                    //     trig = 0;
                    // }
                }
                else
                {
                    // cout << "Feasibility is Violated " << "\n" << "alt_ekm = " << alt_ekm << ">" << "xi = " << xi << endl;
                }

                //****EVENT-TRIGGERING CONDITION****//
                // double first_Lzm2 = Lzm * (xk - xk_pred).squaredNorm();
                // cout << "first Lzm2: " << first_Lzm2 << endl;
                double trigger_sigma = 15;
                // cout << "trigger_sigma: " << trigger_sigma << endl;
                double second_Lzm2 = Lzm * alt_ekm;
                // cout << "second Lzm2: " << second_Lzm2 << endl;
                // cout << "Fbar: " << Fbar << endl;
                // cout << "sigma*Fbar: " << trigger_sigma*Fbar << endl;
                // double Lzm1 = Lzm*(xk-xk_pred).norm();
                // double Lzminf = Lzm*(xk-xk_pred).lpNorm<Infinity>();
                // cout << "Lzm <  " << second_Lzm2 << " trigger_sigma* Fbar  " << trigger_sigma*Fbar << endl;

                if (second_Lzm2 > trigger_sigma * Fbar)
                {
                    // trig = 5;
                    cout << "Trigger!!!" << endl;
                    // cout << "Event triggering - OCP of NMPC must run" << endl;
                    // cout << "with trig: " << trig << endl;
                    // fdataMsg.trig = trig;
                    // cout << "fdataMsg.trig: " << fdataMsg.trig << endl;
                    // fdataMsg.trig = trig;
                    // cout << "fdataMsg.trig: " << fdataMsg.trig << endl;
                    // fdataMsg.time = timer;
                    // cout << "fdataMsg.time: " << fdataMsg.time << endl;
                    trig = 5;
                    fdataMsg.trig = trig;
                    fdataMsg.time = timer;
                    goto optlabel;
                }
                else
                {
                    cout << "No trigger!!!" << endl;
                    trig = 0;
                    // cout << "No Triggering" << endl;
                    // fdataMsg.trig = trig;
                    // cout << "fdataMsg.trig: " << fdataMsg.trig << endl;
                    // cout << "without trig: " << trig << endl;
                }
            }

            //****EVENT-TRIGGERING BLOCK - END****//
            printf("found minimum at J(%g,%g,%g,%g) = %g\n", inputs[dim_inputs * m], inputs[dim_inputs * m + 1], inputs[dim_inputs * m + 2], inputs[dim_inputs * m + 3], minJ);

            double dt = end - start;
            // cout << "Loop dt = " << dt << endl;
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
            Tx = dataMsg.velocity.x = VelTrans1(VelTrans(caminputs))(0, 0) + 0.5;
            Ty = dataMsg.velocity.y = VelTrans1(VelTrans(caminputs))(1, 0);
            // Tz = dataMsg.velocity.z  =  VelTrans1(VelTrans(caminputs))(2,0);
            Tz = 0.0;
            Oz = dataMsg.yaw_rate = VelTrans1(VelTrans(caminputs))(5, 0);

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
            // cout << "Drone Velocities Tx,Ty,Tz,Oz = (" << Tx << "," << Ty << "," << Tz << "," << Oz << ")" << endl;

            dataMsg.velocity.x = 0.0;
            dataMsg.velocity.y = 0.0;
            dataMsg.velocity.z = 0.0;
            dataMsg.yaw_rate = 0.0;

            //****SAVE DATA****//

            fdataMsg.J = minJ;
            //   fdataMsg.optNUM = nlopt_optimize(opt, inputs, &minJ);
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

            // fdataMsg.Tx = inputs[0];
            // fdataMsg.Ty = inputs[1];
            // fdataMsg.Tz = inputs[2];
            // fdataMsg.Oz = inputs[3];

            fdataMsg.Tx = Tx;
            fdataMsg.Ty = Ty;
            fdataMsg.Tz = Tz;
            fdataMsg.Oz = Oz;

            // cout << "before publish trig: " << trig << endl;
            fdataMsg.trig = trig;
            // cout << "before publish fdataMsg.trig: " << fdataMsg.trig << endl;

            fdataMsg.time = timer;
            fdataMsg.dtloop = dt;

            printf("Drone Velocities Tx,Ty,Tz,Oz(%g,%g,%g,%g)", fdataMsg.Tx, fdataMsg.Ty, fdataMsg.Tz, fdataMsg.Oz);
            cout << "\n"
                 << endl;

            // fdataMsg.xi = xi;
            // fdataMsg.ekm2 = alt_ekm;

            // fdataMsg.barycenter_x = barycenter_x;
            // fdataMsg.barycenter_x_des = barycenter_x_des;

            // fdataMsg.barycenter_y = barycenter_y;
            // fdataMsg.barycenter_y_des = barycenter_y_des;

            // fdataMsg.sigma = sigma;
            // fdataMsg.sigma_des = sigma_des;

            // fdataMsg.alpha = alpha;
            // fdataMsg.alpha_des = alpha_des;

            // cout << "\n--------  fdataMsg.trig: " << fdataMsg.trig << "--------\n" << endl;

            rec_pub.publish(fdataMsg);
            vel_pub.publish(dataMsg);
        }
        //****UPDATE "PREVIOUS" STATE TO BE USED IN THE TRIGGERING RULE****//
        x_prev << barycenter_x, barycenter_y, sigma, alpha;
        // uv_prev << u0, v0, u1, v1, u2, v2, u3, v3;
        // cout << "x_prev = " << x_prev << endl;
        // cout << "uv_prev = " << uv_prev << endl;

        //****INCREASE LOOP ITERATION VARIABLE M & CHECK IF THE HORIZON END IS REACHED****//
        m = m + 1;
        if (m == mpc_hrz)
        {
            goto optlabel;
        }

        ros::spinOnce();
        // ros::spin;
        loop_rate.sleep();
    }

    nlopt_destroy(opt);
    return 0;
}
