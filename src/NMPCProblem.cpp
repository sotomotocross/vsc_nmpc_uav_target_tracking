// NMPCProblem.cpp
#include "vsc_nmpc_uav_target_tracking/NMPCProblem.hpp"
#include "vsc_nmpc_uav_target_tracking/NMPCController.hpp"
#include "vsc_nmpc_uav_target_tracking/DynamicsCalculator.hpp"

#include <cstdlib>
#include <iostream>
#include <stdio.h>

#include <math.h>
#include <nlopt.hpp>
#include <eigen3/Eigen/Dense>

using namespace std;
using namespace Eigen;

NMPCProblem::NMPCProblem()
{
  // Initialize parameters
  features_n = 4;
  dim_inputs = 4;
  dim_s = 2 * features_n;
  mpc_hrz = 6;
  mpc_dt = 0.1;
  l = 252.07;
  a = 10;
  optNum = 0.0;

  // Simulator camera parameters
  umax = 720;
  umin = 0;
  vmax = 480;
  vmin = 0;
  cu = 0.5 * (umax + umin);
  cv = 0.5 * (vmax + vmin);

  // Camera Frame Desired Features
  // Simulator (720*480)
  u0d = 340;
  v0d = 480;
  u1d = 340;
  v1d = 0;
  u2d = 380;
  v2d = 0;
  u3d = 380;
  v3d = 480;

  x0d = (u0d - cu) / l;
  g0d = (v0d - cv) / l;
  x1d = (u1d - cu) / l;
  g1d = (v1d - cv) / l;
  x2d = (u2d - cu) / l;
  g2d = (v2d - cv) / l;
  x3d = (u3d - cu) / l;
  g3d = (v3d - cv) / l;

  sc_x = (umax - cu) / l;
  sc_y = (vmax - cv) / l;

  // Initialize Q, R, and P matrices
  Q.setIdentity(dim_s, dim_s);
  R.setIdentity(dim_inputs, dim_inputs);
  P.setIdentity(dim_s, dim_s);
  
  s_des.setZero(dim_s,mpc_hrz+1);

  Q *= 10;
  R *= 5;
  P *= 1;
  R(0, 0) = 15;
  R(2, 2) = 500;
  R(3, 3) = 15;

  // Initialize s_abs matrix
  s_abs.setZero(dim_s);
  s_abs << sc_x, sc_y, sc_x, sc_y, sc_x, sc_y, sc_x, sc_y;

  // Initialize Eigen variables
  inputs_lb = Eigen::VectorXd::Zero(dim_inputs * mpc_hrz);
  inputs_ub = Eigen::VectorXd::Zero(dim_inputs * mpc_hrz);
  constraints_tol = Eigen::VectorXd::Constant(dim_s * (mpc_hrz + 1), 0.001);
  inputs = Eigen::VectorXd::Zero(dim_inputs * mpc_hrz);

  // Initialize bounds
  for (int k = 0; k < mpc_hrz; ++k)
  {
    inputs_lb.segment(dim_inputs * k, dim_inputs) << -0.5, -3, -0.1, -1;
    inputs_ub.segment(dim_inputs * k, dim_inputs) << 0.5, 3, 0.1, 1;
  }

  // Initialize the NLopt optimization object
  opt = nlopt_create(NLOPT_LN_BOBYQA, dim_inputs * mpc_hrz); // algorithm and dimensionality
}

NMPCProblem::~NMPCProblem()
{
  // Destructor
  nlopt_destroy(opt);
}

void NMPCProblem::setValues(double x0, double g0, double Z0, double x1, double g1, double Z1,
                            double x2, double g2, double Z2, double x3, double g3, double Z3) {
    // Set the received values
    this->x0 = x0;
    this->g0 = g0;
    this->Z0 = Z0;
    this->x1 = x1;
    this->g1 = g1;
    this->Z1 = Z1;
    this->x2 = x2;
    this->g2 = g2;
    this->Z2 = Z2;
    this->x3 = x3;
    this->g3 = g3;
    this->Z3 = Z3;
}

void NMPCProblem::setup()
{
  // Set bounds for input variables
  // Note: inputs_lb and inputs_ub should already be initialized in the constructor
  nlopt_set_lower_bounds(opt, inputs_lb.data());
  nlopt_set_upper_bounds(opt, inputs_ub.data());

  // Set the objective function
  nlopt_set_min_objective(opt, &NMPCProblem::costFunctionWrapper, this); // Pass the object pointer for the member function

  // Set algorithm-specific parameters
  nlopt_set_ftol_abs(opt, 0.0001);
  nlopt_set_xtol_abs1(opt, 0.0001);

  // Add constraints
  nlopt_add_inequality_mconstraint(opt, dim_s * (mpc_hrz + 1), &NMPCProblem::constraintsWrapper, this, constraints_tol.data()); // Pass the object pointer for the member function
}

// Static wrapper function for cost function
double NMPCProblem::costFunctionWrapper(unsigned int n, const double *x, double *grad, void *data)
{
  NMPCProblem *obj = static_cast<NMPCProblem *>(data); // Cast the void pointer back to NMPCProblem pointer
  return obj->costFunction(n, x, grad, nullptr);       // Call the member function using the object pointer
}

// Static wrapper function for constraints
void NMPCProblem::constraintsWrapper(unsigned int m, double *result, unsigned int n, const double *x, double *grad, void *data)
{
  NMPCProblem *obj = static_cast<NMPCProblem *>(data); // Cast the void pointer back to NMPCProblem pointer
  obj->constraints(m, result, n, x, grad, nullptr);    // Call the member function using the object pointer
}

double NMPCProblem::costFunction(unsigned int n, const double *x, double *grad, void *data)
{
  // cout << "\n\nMpikame loipon stin Cost Function \n" << endl;
  //   MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz> > ((double*) x);
  MatrixXd inputs(dim_inputs, mpc_hrz);
  Map<MatrixXd> inputs_map((double *)x, dim_inputs, mpc_hrz);
  inputs = inputs_map;

  // Trajectory of States (image features)
  MatrixXd traj_s(dim_s, mpc_hrz + 1);
  traj_s.setZero(dim_s, mpc_hrz + 1);
  traj_s.col(0) << x0, g0, x1, g1, x2, g2, x3, g3;
  
  // cout << "traj_s: \n" << traj_s << endl;
  // cout << "inputs: \n" << inputs << endl;

  // cout << "(x0,g0,Z0): (" << x0 << "," << g0 << "," << Z0 << ")" << endl;
  // cout << "(x1,g1,Z1): (" << x1 << "," << g1 << "," << Z1 << ")" << endl;
  // cout << "(x2,g2,Z2): (" << x2 << "," << g2 << "," << Z2 << ")" << endl;
  // cout << "(x3,g3,Z3): (" << x3 << "," << g3 << "," << Z3 << ")" << endl;

  // Progate the model (IBVS with Image Jacobian)
  for (int k = 0; k < mpc_hrz; k++)
  {
    // cout << "Mpike to gamidi!!!" << endl;
    VectorXd sdot = dynamics_calculator_.IBVS_system(inputs.col(k), dim_s, dim_inputs,
                                                     Z0, Z1, Z2, Z3,
                                                     x0, x1, x2, x3,
                                                     g0, g1, g2, g3);
    // cout << "s_dot" << sdot << endl;
    traj_s.col(k + 1) = traj_s.col(k) + sdot * mpc_dt;
  }

  // Calculate Running Costs
  double Ji = 0.0;

  // cout << "traj_s = \n"
  //      << traj_s << endl;

  //****DEFINE INITIAL DESIRED V****//
  // VectorXd s_des(dim_s);
  s_des.col(0) << x0d, g0d, x1d, g1d, x2d, g2d, x3d, g3d;
  // cout << "s_des = " << s_des.transpose() << endl;

  //****SET V DESIRED VELOCITY FOR THE VTA****//
  double b = 15;
  VectorXd s_at(dim_s);
  s_at.setZero(dim_s);
  s_at << 0, b / l, 0, b / l, 0, b / l, 0, b / l;
  // cout << "s_at = " << s_at.transpose() << endl;

  //****PROPOGATE THE V DESIRED IN THE HORIZON N FOR dt TIMESTEP SIZE****//
  for (int k = 0; k < mpc_hrz; k++)
  {
    s_des.col(k + 1) = s_des.col(k) + s_at;
  }

  // cout << "s_des = \n"
      //  << s_des << endl;
  // cout << "v0d FUNCTION = " << g0d*l + cv << endl;
  // printf("g0d FUNCTION:%lf\n", g0d);

  for (int k = 0; k < mpc_hrz; k++)
  {
    ek = traj_s.col(k) - s_des.col(k);

    Ji += ek.transpose() * Q * ek;
    Ji += inputs.col(k).transpose() * R * inputs.col(k);
  }

  // cout << "ek = " << ek << endl;

  // Calculate Terminal Costs
  double Jt;
  VectorXd et = traj_s.col(mpc_hrz) - s_des.col(mpc_hrz);

  Jt = et.transpose() * P * et;
  // cout << "et = " << et << endl;

  // cout << "Ji" << Ji << "+"
  //      << "Jt" << Jt << endl;
  return Ji + Jt;
}

void NMPCProblem::constraints(unsigned int m, double *c, unsigned int n, const double *x, double *grad, void *data)
{
  //   MatrixXd inputs = Map<Matrix<double, dim_inputs, mpc_hrz> > ((double*) x);
  MatrixXd inputs(dim_inputs, mpc_hrz);
  Map<MatrixXd> inputs_map((double *)x, dim_inputs, mpc_hrz);
  inputs = inputs_map;

  MatrixXd traj_s(dim_s, mpc_hrz + 1);
  traj_s.setZero(dim_s, mpc_hrz + 1);
  traj_s.col(0) << x0, g0, x1, g1, x2, g2, x3, g3;

  // Progate the model (IBVS with Image Jacobian)
  for (int k = 0; k < mpc_hrz; k++)
  {
    VectorXd sdot = dynamics_calculator_.IBVS_system(inputs.col(k), dim_s, dim_inputs,
                                                     Z0, Z1, Z2, Z3,
                                                     x0, x1, x2, x3,
                                                     g0, g1, g2, g3);
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

std::pair<double, std::vector<double>> NMPCProblem::solve(double *inputs, double minJ)
{
  double optNum = -1.0;                                      // Initialize optNum with a default value
  std::vector<double> optimizedInputs(dim_inputs * mpc_hrz); // Vector to store the optimized inputs

  // cout << "Lete na petixei to optimization???" << endl;

  // cout << "(x0,g0,Z0): (" << x0 << "," << g0 << "," << Z0 << ")" << endl;
  // cout << "(x1,g1,Z1): (" << x1 << "," << g1 << "," << Z1 << ")" << endl;
  // cout << "(x2,g2,Z2): (" << x2 << "," << g2 << "," << Z2 << ")" << endl;
  // cout << "(x3,g3,Z3): (" << x3 << "," << g3 << "," << Z3 << ")" << endl;

  //****EXECUTE OPTIMIZATION****//
  optNum = nlopt_optimize(opt, inputs, &minJ);
  // cout << "Optimization Return Code: " << optNum << endl;
  // printf("Found minimum at J(%g,%g,%g,%g) = %g\n", inputs[0], inputs[1], inputs[2], inputs[3], minJ);

  // Copy the optimized inputs to the vector
  for (int i = 0; i < dim_inputs * mpc_hrz; ++i)
  {
    optimizedInputs[i] = inputs[i];
  }

  return std::make_pair(minJ, optimizedInputs); // Return optNum and the optimized inputs
}