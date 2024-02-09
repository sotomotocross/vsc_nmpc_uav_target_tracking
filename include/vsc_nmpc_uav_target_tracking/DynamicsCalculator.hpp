#pragma once

#include <cstdlib>
#include <iostream>
#include <stdio.h>
#include <cmath>
#include <eigen3/Eigen/Dense>

class DynamicsCalculator
{
public:
    // Calculate feature rate for an IBVS system
    Eigen::VectorXd IBVS_system(Eigen::VectorXd cameraTwist, int dim_s, int dim_inputs,
                                double Z0, double Z1, double Z2, double Z3,
                                double x0, double x1, double x2, double x3,
                                double g0, double g1, double g2, double g3);

    // Add any other dynamics-related functions here

private:
    // Add any private members or helper functions for dynamics calculations
};
