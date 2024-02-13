#include "vsc_nmpc_uav_target_tracking/DynamicsCalculator.hpp"

#include <cmath>
#include <eigen3/Eigen/Dense>

// Calculate feature rate for an IBVS system
Eigen::VectorXd DynamicsCalculator::IBVS_system(Eigen::VectorXd camTwist, int dim_s, int dim_inputs,
                                                double Z0, double Z1, double Z2, double Z3,
                                                double x0, double x1, double x2, double x3,
                                                double g0, double g1, double g2, double g3)
{
    Eigen::MatrixXd interactionMatrix(dim_s, dim_inputs);

    // Populate the interaction matrix
    interactionMatrix << -1 / Z0, 0.0, x0 / Z0, g0,
                         0.0, -1 / Z0, g0 / Z0, -x0,
                         -1 / Z1, 0.0, x1 / Z1, g1,
                         0.0, -1 / Z1, g1 / Z1, -x1,
                         -1 / Z2, 0.0, x2 / Z2, g2,
                         0.0, -1 / Z2, g2 / Z2, -x2,
                         -1 / Z3, 0.0, x3 / Z3, g3,
                         0.0, -1 / Z3, g3 / Z3, -x3;

    // Calculate the feature rate
    return interactionMatrix * camTwist;
}
