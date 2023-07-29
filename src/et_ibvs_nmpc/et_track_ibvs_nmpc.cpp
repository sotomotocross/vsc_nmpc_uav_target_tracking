#include "ros/ros.h"

#include "geometry_msgs/Twist.h"
#include "mavros_msgs/PositionTarget.h"
#include "img_seg_cnn/PREDdata.h"
#include "std_msgs/Float64.h"
#include "vsc_nmpc_uav_target_tracking/rec.h"

#include <cstdlib>
#include <iostream>
#include <stdio.h>

#include <math.h>
#include <nlopt.hpp>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

// Global MPC Variables
int features_n = 4;
const int dim_inputs = 4;
int dim_s = 2 * features_n;
const int mpc_hrz = 6;
double mpc_dt = 0.1;
double l = 252.07;
double a = 0;

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
VectorXd x_prev;
VectorXd e_prev;
VectorXd u_prev;
VectorXd xk;
VectorXd xk_pred;

// Simulator camera parameters
double umax = 720;
double umin = 0;
double vmax = 480;
double vmin = 0;
// double cu = 0.5*(umax+umin);
// double cv = 0.5*(vmax+vmin);
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

// Camera Frame Desired Features
// ZED 2 stereocamera (672*376)
//  double u0d = 316;
//  double v0d = 376;
//  double u1d = 316;
//  double v1d = 0;
//  double u2d = 356;
//  double v2d = 0;
//  double u3d = 356;
//  double v3d = 376;

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

// IBVS Feature Rate Le*Vk
VectorXd IBVSSystem(VectorXd camTwist)
{
        MatrixXd Le(dim_s, dim_inputs);
        //,Le.setZero(dim_s,dim_inputs);

        Le << -1 / Z0, 0.0, x0 / Z0, g0,
            0.0, -1 / Z0, g0 / Z0, -x0,
            -1 / Z1, 0.0, x1 / Z1, g1,
            0.0, -1 / Z1, g1 / Z1, -x1,
            -1 / Z2, 0.0, x2 / Z2, g2,
            0.0, -1 / Z2, g2 / Z2, -x2,
            -1 / Z3, 0.0, x3 / Z3, g3,
            0.0, -1 / Z3, g3 / Z3, -x3;

        return Le * camTwist;
        cout << "Interaction Matrix:\n"
             << Le << endl;
        cout << "Interaction Matrix" << Le * camTwist << endl;
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

// IBVS-MPC Cost Function
double costFunction(unsigned int n, const double *x, double *grad, void *data)
{
        MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz>>((double *)x);

        // cout << inputs << endl;
        // cout << endl;

        // Trajectory of States (image features)
        MatrixXd traj_s(dim_s, mpc_hrz + 1);
        traj_s.setZero(dim_s, mpc_hrz + 1);
        traj_s.col(0) << x0, g0, x1, g1, x2, g2, x3, g3;
        pred_s.col(0) = traj_s.col(0);
        cout << "pred_s: " << pred_s << endl;
        cout << "pred_s.col(0): " << pred_s.col(0) << endl;

        // Progate the model (IBVS with Image Jacobian)
        for (int k = 0; k < mpc_hrz; k++)
        {
                VectorXd sdot = IBVSSystem(inputs.col(k));
                // cout << "s_dot" << sdot << endl;
                traj_s.col(k + 1) = traj_s.col(k) + sdot * mpc_dt;
                pred_s.col(k + 1) = traj_s.col(k + 1);
                cout << "pred_s: " << pred_s << endl;
                cout << "pred_s.col(k + 1): " << pred_s.col(k + 1) << endl;
        }

        //****DEFINE INITIAL DESIRED V****//
        MatrixXd s_des(dim_s, mpc_hrz + 1);
        s_des.col(0) << x0d, g0d, x1d, g1d, x2d, g2d, x3d, g3d;

        //****SET V DESIRED VELOCITY FOR THE VTA****//
        double b = 25;
        VectorXd s_at(dim_s);
        s_at.setZero(dim_s);
        s_at << 0, b / l, 0, b / l, 0, b / l, 0, b / l;

        //****PROPOGATE THE V DESIRED IN THE HORIZON N FOR dt TIMESTEP SIZE****//
        for (int k = 0; k < mpc_hrz; k++)
        {
                s_des.col(k + 1) = s_des.col(k) + s_at;
        }

        // Calculate Running Costs
        double Ji = 0.0;

        for (int k = 0; k < mpc_hrz; k++)
        {
                ek = traj_s.col(k) - s_des.col(k);

                Ji += ek.transpose() * Q * ek;
                Ji += inputs.col(k).transpose() * R * inputs.col(k);
        }

        // Calculate Terminal Costs
        double Jt;
        VectorXd et = traj_s.col(mpc_hrz) - s_des.col(mpc_hrz);

        Jt = et.transpose() * P * et;

        // cout << "Ji" << Ji << "+" << "Jt" << Jt << endl;
        return Ji + Jt;
}

//****DEFINE FOV CONSTRAINTS****//
void constraints(unsigned int m, double *c, unsigned int n, const double *x, double *grad, void *data)
{
        // Propagate the model.
        MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz>>((double *)x);
        ;
        // cout << "inputs = " << inputs << endl;
        // Trajectory of States (image features)
        MatrixXd traj_s(dim_s, mpc_hrz + 1);
        traj_s.setZero(dim_s, mpc_hrz + 1);
        traj_s.col(0) << x0, g0, x1, g1, x2, g2, x3, g3;

        // Progate the model (IBVS with Image Jacobian)
        for (int k = 0; k < mpc_hrz; k++)
        {
                VectorXd sdot = IBVSSystem(inputs.col(k));
                traj_s.col(k + 1) = traj_s.col(k) + sdot * mpc_dt;
        }

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
        ros::Publisher rec_pub = nh.advertise<vsc_nmpc_uav_target_tracking::rec>("/vsc_nmpc_uav_target_tracking/msg/rec", 1);

        //****INIATILIZE MPC VARIABLES****//
        s_des.setZero(dim_s, mpc_hrz + 1);
        s_bc.setOnes(dim_s);
        s_br.setOnes(dim_s);

        s_abs.setZero(dim_s);
        s_abs << sc_x, sc_y, sc_x, sc_y, sc_x, sc_y, sc_x, sc_y;
        x_prev.setZero(dim_s);
        e_prev.setZero(dim_s);
        u_prev.setZero(dim_inputs);
        xk.setZero(dim_s);
        xk_pred.setZero(dim_s);
        f_des.setZero(dim_s);
        f_des << x0d, g0d, x1d, g1d, x2d, g2d, x3d, g3d;

        //****SET MPC COST FUNCTION MATRICES****//
        Q.setIdentity(dim_s, dim_s);
        R.setIdentity(dim_inputs, dim_inputs);
        P.setIdentity(dim_s, dim_s);

        Q = 15 * Q;
        R = 5 * R;
        P = 1 * Q;
        R(2, 2) = 50;

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

        // printf("Image Features for Point 1 are (%g,%g) =", x0, g0);
        // printf("Image Features for Point 2 are (%g,%g) =", x1, g1);
        // printf("Image Features for Point 3 are (%g,%g) =", x2, g2);
        // printf("Image Features for Point 4 are (%g,%g) =", x3, g3);

        //****CREATE NLOPT OPTIMIZATION OBJECT, ALGORITHM & TOLERANCES****//
        nlopt_opt opt;
        opt = nlopt_create(NLOPT_LN_BOBYQA, dim_inputs * mpc_hrz); // algorithm and dimensionality
        nlopt_set_lower_bounds(opt, inputs_lb);
        nlopt_set_upper_bounds(opt, inputs_ub);
        nlopt_set_min_objective(opt, costFunction, NULL);
        nlopt_set_ftol_abs(opt, 0.0001);
        nlopt_set_xtol_abs1(opt, 0.0001);

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

        //****RUNNING LOOP****//
        while (ros::ok())
        {

                //****CREATE MESSAGE TO SAVE DATA****//
                vsc_nmpc_uav_target_tracking::rec fdataMsg;

                //****START OF THE LOOP****//
                cout << "NEW LOOP FOR m   = " << m << "   HAS STARTED" << endl;

                //****SAVE DATA****//
                xk << x0, g0, x1, g1, x2, g2, x3, g3;
                uv << u0, v0, u1, v1, u2, v2, u3, v3;
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
                        // e_prev << x_prev;
                        //  cout << "u_prev " << u_prev << endl;

                        Fbar = Fbar + R(0, 0) * (e_prev).squaredNorm();

                        // cout << "x_prev = " << x_prev << endl;
                        // cout << "u_prev =" << u_prev << endl;
                        // cout << "e_prev =" << e_prev << endl;
                        // cout << "Fbar = " << Fbar << endl;
                        // cout << "s_des.col(0) =" << f_des << endl;

                        //****PREDICTED VS REAL STATE ERROR NORM e(k+m|k-1)****//
                        cout << "pred_s: " << pred_s << endl;
                        xk_pred << pred_s.col(m);
                        cout << "xk_pred: " << xk_pred << endl;
                        cout << "xk: " << xk << endl;
                        double ekm2 = (xk - xk_pred).squaredNorm();
                        // cout << "xk = " << xk << endl;
                        // cout << "xk_pred = " << xk_pred << endl;
                        // cout << " (xk-xk_pred).squaredNorm() = " << (xk-xk_pred).squaredNorm() << endl;
                        // cout << " |e(k+m|k-1)|_2 = " << ekm2 << endl;
                        // cout << " |e(k+m|k-1)|_infty = " << (xk-xk_pred).lpNorm<Infinity>() << endl;

                        //****THEORETICAL FEASIBILITY CHECK****//
                        if (ekm2 <= xi)
                        {
                                cout << "Feasibility is Ensured " << endl
                                     << " ekm2 = " << ekm2 << "<"
                                     << "xi = " << xi << endl;
                                //****EVENT-TRIGGERING CONDITION****//
                                double Lzm2 = Lzm * (xk - xk_pred).squaredNorm();
                                // double Lzm1 = Lzm*(xk-xk_pred).norm();
                                // double Lzminf = Lzm*(xk-xk_pred).lpNorm<Infinity>();
                                double sigma = 1.0;
                                // cout << "Lzm <  " << Lzm2 << " sigma* Fbar  " << sigma*Fbar << endl;

                                if (Lzm2 > sigma * Fbar)
                                {
                                        cout << " EVENT OCCURED, MPC MUST RUN AGAN " << endl;
                                        trig = 5;
                                        fdataMsg.trig = trig;
                                        fdataMsg.time = timer;
                                        goto optlabel;
                                }
                                else
                                {
                                        trig = 0;
                                }
                        }
                        else
                        {
                                cout << "Feasibility is Violated " << endl
                                     << " ekm2 = " << ekm2 << ">"
                                     << "xi = " << xi << endl;
                        }
                }
                //****EVENT-TRIGGERING BLOCK - END****//
                printf("found minimum at J(%g,%g,%g,%g) = %g\n", inputs[dim_inputs * m], inputs[dim_inputs * m + 1], inputs[dim_inputs * m + 2], inputs[dim_inputs * m + 3], minJ);

                double dt = end - start;
                // cout << "Loop dt = " << dt << endl;
                realtime = realtime + dt;

                //****SEND VELOCITIES TO AUTOPILOT****//
                Matrix<double, 4, 1> caminputs;
                caminputs(0, 0) = inputs[dim_inputs * m];
                caminputs(1, 0) = inputs[dim_inputs * m + 1];
                caminputs(2, 0) = inputs[dim_inputs * m + 2];
                caminputs(3, 0) = inputs[dim_inputs * m + 3];
                cout << " INPUTS FOR m =    " << m << "  WERE APPLIED" << endl;

                mavros_msgs::PositionTarget dataMsg;

                dataMsg.coordinate_frame = 8;
                dataMsg.type_mask = 1479;
                dataMsg.header.stamp = ros::Time::now();
                Tx = dataMsg.velocity.x = VelTrans1(VelTrans(caminputs))(0, 0);
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

                printf("Drone Velocities Tx,Ty,Tz,Oz(%g,%g,%g,%g) =", Tx, Ty, Tz, Oz);

                //****SAVE DATA****//
                fdataMsg.J = minJ;
                fdataMsg.optNUM = nlopt_optimize(opt, inputs, &minJ);
                fdataMsg.Z = Z0;
                fdataMsg.trig = trig;
                fdataMsg.m = m;

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

                // fdataMsg.Tx = inputs[dim_inputs*m];
                // fdataMsg.Ty = inputs[dim_inputs*m+1];
                // fdataMsg.Tz = inputs[dim_inputs*m+2];
                // fdataMsg.Oz = inputs[dim_inputs*m+3];

                fdataMsg.Tx = Tx;
                fdataMsg.Ty = Ty;
                fdataMsg.Tz = Tz;
                fdataMsg.Oz = Oz;

                // fdataMsg.ekm2 = ekm2;
                // fdataMsg.xi = xi;

                fdataMsg.time = timer;
                fdataMsg.dtloop = dt;

                rec_pub.publish(fdataMsg);
                vel_pub.publish(dataMsg);
                ros::spinOnce();
                loop_rate.sleep();

                //****UPDATE "PREVIOUS" STATE TO BE USED IN THE TRIGGERING RULE****//
                x_prev << x0, g0, x1, g1, x2, g2, x3, g3;
                uv_prev << u0, v0, u1, v1, u2, v2, u3, v3;
                // cout << "x_prev = " << x_prev << endl;
                // cout << "uv_prev = " << uv_prev << endl;

                //****INCREASE LOOP ITERATION VARIABLE M & CHECK IF THE HORIZON END IS REACHED****//
                m = m + 1;
                if (m == mpc_hrz)
                {
                        goto optlabel;
                }
        }

        //****DESTROY NLOPT OPTIMIZATION OBJECT****//
        nlopt_destroy(opt);
        return 0;
}
