#pragma once 

#include <eigen3/Eigen/Dense>

class VelocityTransformer
{
public:
  static Eigen::MatrixXd VelTrans(Eigen::MatrixXd CameraVel);
  static Eigen::MatrixXd VelTrans1(Eigen::MatrixXd CameraVel1);

private:
  // Add any private members or helper functions related to velocity transformation
};
