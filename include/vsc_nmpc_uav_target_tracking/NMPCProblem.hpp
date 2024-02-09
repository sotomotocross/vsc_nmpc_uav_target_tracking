#pragma once

#include "vsc_nmpc_uav_target_tracking/DynamicsCalculator.hpp"

#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <math.h>
#include <nlopt.hpp>
#include <eigen3/Eigen/Dense>

class NMPCProblem
{
public:
    NMPCProblem();
    ~NMPCProblem();

    // Set up the optimization problem
    void setup();

    // Objective function for the optimization problem
    double costFunction(unsigned int n, const double *x, double *grad, void *data);

    // Constraints for the optimization problem
    void constraints(unsigned int m, double *c, unsigned int n, const double *x, double *grad, void *data);

    // Declaration of cost function wrapper
    static double costFunctionWrapper(unsigned n, const double *x, double *grad, void *data);

    // Declaration of constraints function wrapper
    static void constraintsWrapper(unsigned m, double *result, unsigned n, const double *x, double *grad, void *data);

    // Solve the optimization problem
    std::pair<double, std::vector<double>> solve(double *inputs, double minJ);

    // Method to set x, g, and Z values
    void setValues(double x0, double g0, double Z0, double x1, double g1, double Z1,
                   double x2, double g2, double Z2, double x3, double g3, double Z3);

private:
    // NLopt optimizer
    // nlopt::opt optimizer;
    nlopt_opt opt; // NLopt optimization object

    // Dynamics Calculator
    DynamicsCalculator dynamics_calculator_;

    // Define Eigen variables
    Eigen::VectorXd inputs_lb;
    Eigen::VectorXd inputs_ub;
    Eigen::VectorXd constraints_tol;
    Eigen::VectorXd inputs;

    int mpc_hrz;
    int dim_s;
    int dim_inputs;
    double mpc_dt;
    double a;
    int features_n;
    double optNum;

    Eigen::MatrixXd Q;
    Eigen::MatrixXd R;
    Eigen::MatrixXd P;
    Eigen::MatrixXd s_des;
    Eigen::VectorXd s_abs;

    Eigen::VectorXd s_ub;
    Eigen::VectorXd s_lb;
    Eigen::VectorXd s_bc;
    Eigen::VectorXd s_br;
    Eigen::VectorXd t;
    Eigen::VectorXd ek;

    //Camera Frame Update Callback Variables
    double x0,g0,Z0;
    double x1,g1,Z1;
    double x2,g2,Z2;
    double x3,g3,Z3;

    double d01,d03;

    double u0,v0,u1,v1,u2,v2,u3,v3;
    double l;  

    // Simulator camera parameters
    double umax;
    double umin;
    double vmax;
    double vmin;
    double cu;
    double cv;

    // Camera Frame Desired Features
    // Simulator (720*480)
    double u0d;
    double v0d;
    double u1d;
    double v1d;
    double u2d;
    double v2d;
    double u3d;
    double v3d;

    double x0d;
    double g0d;
    double x1d;
    double g1d;
    double x2d;
    double g2d;
    double x3d;
    double g3d;
    double sc_x;
    double sc_y;
};
