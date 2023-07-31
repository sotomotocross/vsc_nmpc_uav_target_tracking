#ifndef RRT_PLANNER_INCLUDE_RRT_PLANNER_RRT_PLANNER_H_
#define RRT_PLANNER_INCLUDE_RRT_PLANNER_RRT_PLANNER_H_

#include <random>
#include <iostream>

#include <ros/ros.h>
#include <nav_msgs/OccupancyGrid.h>
#include <nav_msgs/Path.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>
#include <geometry_msgs/PoseStamped.h>

#include <opencv2/opencv.hpp>

#include "geometry_msgs/Twist.h"
#include "mavros_msgs/PositionTarget.h"
#include "img_seg_cnn/PREDdata.h"
#include "img_seg_cnn/POLYcalc_custom.h"
#include "img_seg_cnn/POLYcalc_custom_tf.h"
#include "std_msgs/Float64.h"
#include "rrt_planner/rec.h"
#include <vector>
#include <algorithm>
#include <iostream>
#include <cstdlib>
#include <stdio.h>
#include <math.h>
#include <cmath>
#include <nlopt.hpp>
#include <eigen3/Eigen/Dense>

// Distance definition for CBF
#include <bits/stdc++.h>
// To store the point
#define Point_blank pair<double, double>
#define F first
#define S second

using namespace std;
using namespace Eigen;


namespace rrt_planner
{

  class NMPC_algorithm{
    public:
      explicit NMPC_algorithm(ros::NodeHandle *);
        ~NMPC_algorithm() = default;

      //****UPDATE IMAGE FEATURE COORDINATES****//
      void featureCallback_poly_custom(const img_seg_cnn::POLYcalc_custom::ConstPtr &s_message);

      //****UPDATE IMAGE FEATURE COORDINATES****//
      void featureCallback_poly_custom_tf(const img_seg_cnn::POLYcalc_custom_tf::ConstPtr &s_message);

      //****UPDATE ALTITUDE****//
      void altitudeCallback(const std_msgs::Float64::ConstPtr &alt_message);

    private:
      // distance between two 2D points
      double distance(double x1, double y1, double x2, double y2);
      
      // distance between a point and a line segment
      double distance(double px, double py, double x1, double y1, double x2, double y2, double &projx, double &projy);
      
      // distance between a point and a polygon
      double distance(double px, double py, const vector<pair<double, double>> &polygon, double &closestx, double &closesty);

      // Function to return the minimum distance
      // between a line segment AB and a point E
      double minDistance(Point_blank A, Point_blank B, Point_blank E);

      double state_bar_fnct_calc(VectorXd camTwist);

      VectorXd barrier_function_calculation();

      VectorXd calculate_moments(VectorXd feat_u, VectorXd feat_v);

      VectorXd calculate_central_moments(VectorXd moments);

      VectorXd img_moments_system(VectorXd camTwist, VectorXd moments);

      VectorXd Dynamic_System_x_y_reverted(VectorXd camTwist, VectorXd feat_prop);

      // Camera-UAV Velocity Transform VelUAV
      MatrixXd VelTrans(MatrixXd CameraVel);

      // Camera-UAV Velocity Transform VelUAV
      MatrixXd VelTrans1(MatrixXd CameraVel1);

      // IBVS Feature Rate Le*Vk
      VectorXd IBVSSystem(VectorXd camTwist);

      // Image-moments-like NMPC Cost Function
      double costFunction(unsigned int n, const double *x, double *grad, void *data);

      // Image-moments-like NMPC Cost Function alternative
      double costFunction_alter(unsigned int n, const double *x, double *grad, void *data);

      //****DEFINE FOV CONSTRAINTS****//
      void constraints(unsigned int m, double *c, unsigned int n, const double *x, double *grad, void *data);

      ros::NodeHandle * nh_;
      ros::NodeHandle private_nh_;
    
      ros::Subscriber feature_sub_poly_custom;
      ros::Subscriber feature_sub_poly_custom_tf;
      ros::Subscriber alt_sub;

      // Create subscribers
      ros::Publisher vel_pub;
      ros::Publisher rec_pub;

      /**
       * Utility parameters from the node server
       */
      double cX, cY;
      int cX_int, cY_int;

      //  int features_n = 4;
      int features_n = 1;
      int dim_inputs = 4;
      int dim_s = 4;
      int mpc_hrz = 10; // 10
      double mpc_dt = 0.001;  // 0.001
      double l = 252.07;
      double a = 10;
      double optNum;

      MatrixXd s_des;
      // MatrixXd s_des(dim_s, mpc_hrz + 1);
      
      VectorXd stored_s_des;
      // VectorXd stored_s_des(dim_s);
      
      VectorXd stored_traj_s;
      // VectorXd stored_traj_s(dim_s);
      
      MatrixXd s_des_test;
      // MatrixXd s_des_test(dim_s, mpc_hrz + 1);
      
      VectorXd s_ub;
      // VectorXd s_ub(dim_s);
      
      VectorXd s_lb;
      // VectorXd s_lb(dim_s);
      
      VectorXd s_abs;
      // VectorXd s_abs(dim_s);
      
      VectorXd s_bc;
      VectorXd s_br;
      MatrixXd Q;
      MatrixXd R;
      MatrixXd P;
      MatrixXd gains;
      VectorXd t;
      VectorXd ek;
      VectorXd ek_1;
      VectorXd feature_vector;
      VectorXd transformed_features;
      VectorXd opencv_moments;
      MatrixXd polygon_features;
      MatrixXd transformed_polygon_features;

      // Simulator camera parameters
      double umax = 720;
      double umin = 0;
      double vmax = 480;
      double vmin = 0;
      double cu = 360.5;
      double cv = 240.5;

      int flag = 0;

      // Camera Frame Update Callback Variables
      double Z0, Z1, Z2, Z3;
      double Tx, Ty, Tz, Oz;

      double s_bar_x, s_bar_y;
      double first_min_index, second_min_index;
      double custom_sigma, custom_sigma_square, custom_sigma_square_log;
      double angle_tangent, angle_radian, angle_deg;

      double transformed_s_bar_x, transformed_s_bar_y;
      double transformed_first_min_index, transformed_second_min_index;
      double transformed_sigma, transformed_sigma_square, transformed_sigma_square_log;
      double transformed_tangent, transformed_angle_radian, transformed_angle_deg;

      double s_bar_x_des = Z0 * (cu - cu / l);
      double s_bar_y_des = Z0 * (cv - cv / l);

      // double sigma_des = 18500.0;
      double sigma_des = 18.5;
      double sigma_square_des = sqrt(sigma_des);
      double sigma_log_des = log(sigma_square_des);

      double angle_deg_des = 0;
      double angle_des_tan = tan((angle_deg_des / 180) * 3.14);

      // double sigma_constraints = 25000.0;
      double sigma_constraints = 25.0;
      double sigma_constraints_square = sqrt(sigma_constraints);
      double sigma_constraints_square_log = log(sigma_constraints_square);

      double angle_deg_constraint = 45;
      double angle_deg_constraint_tan = tan((angle_deg_constraint / 180) * 3.14);

      double gain_tx_;
      double gain_ty_;
      double gain_tz_;
      double gain_yaw_;

  };

/**
 * A utility class to represent a 2D point
 */
class Point2D
{
public:
  Point2D(): x_(0), y_(0) {}
  Point2D(int x, int y): x_(x), y_(y) {}

  int x() const
  {
    return x_;
  }

  int y() const
  {
    return y_;
  }

  void x(int x)
  {
    x_ = x;
  }

  void y(int y)
  {
    y_ = y;
  }

  float getDistance(const Point2D & end_point) {
    return sqrt(pow((x() - end_point.x()), 2) + pow((y() - end_point.y()), 2));
  }

  bool operator==(const Point2D& v) {
    return (x_ == v.x_ && y_ == v.y_ );
  }

  bool operator!=(const Point2D& v) {
    return (x_ != v.x_ || y_ != v.y_) ;
  }

private:
  int x_;
  int y_;
};


class Vertex {
 private:
     Point2D point_;

     /**
      * @brief the index of the vertex
      */
     int index_;

     /**
      * @brief the vertex's parent index
      */
     int parent_index_;

 public:

     Vertex() {}

     Vertex(const Point2D& point, int index, int parent_index): point_(point),index_(index),parent_index_(parent_index) {};

     ~Vertex() {}


     void set_location(const Point2D& point)
     {
       point_ = point;
     };


     void set_index(int index) {
       index_ = index;
     };


     void set_parent(int index) {
       parent_index_ = index;
     };


     Point2D get_location() {
       return point_;
     };


    int get_index() {
      return index_;
    };

    int get_parent() {
      return parent_index_;
    };

    bool operator==(const Vertex& v) {
      return (point_ == v.point_ && parent_index_ == v.parent_index_);
    }

    bool operator!=(const Vertex& v) {
      return (point_ != v.point_ || parent_index_ != v.parent_index_);
    }
};

/**
 * Main class which implements the RRT algorithm
 */
class RRTPlanner
{
public:
  explicit RRTPlanner(ros::NodeHandle *);

  ~RRTPlanner() = default;

  /**
   * Given a map, the initial pose, and the goal, this function will plan
   * a collision-free path through the map from the initial pose to the goal
   * using the RRT algorithm
   *
   */
  void plan();

  /**
   * Callback for map subscriber
   */
  void mapCallback(const nav_msgs::OccupancyGrid::Ptr &);

  /**
   * Callback for initial pose subscriber
   */
  void initPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &);

  /**
   * Callback for goal subscriber
   */
  void goalCallback(const geometry_msgs::PoseStamped::ConstPtr &);

private:

  /**
   * Publishes the path calculated by RRT as a nav_msgs::Path msg
   *
   */
  void publishPath(std::vector<Point2D> points);

  /**
   * Utility function to check if a given point is free/occupied in the map
   * @param p: point in the map
   * @return boolean true if point is unoccupied, false if occupied
   *
   */
  bool isPointUnoccupied(const Point2D & p);

  /**
   * Utility function to build a CV::Mat from a nav_msgs::OccupancyGrid for display
   */
  void buildMapImage();

  /**
   * Utility function to display the CV::Mat map image
   * @param delay
   */
  void displayMapImage(int delay = 1);

  /**
   * Utility function to draw initial pose and goal pose on the map image
   */
  void drawGoalInitPose();

  /**
   * Utility function to draw a circle on the map
   * @param p: center point of the circle
   * @param radius: radius of the circle
   * @param color: color of the circle
   */
  void drawCircle(Point2D & p, int radius, const cv::Scalar & color);

  /**
   * Utility function to draw a line on the map
   * @param p1: starting point of the line
   * @param p2: end point of the line
   * @param color: color of the line
   * @param thickness: thickness of the line
   */
  void drawLine(Point2D & p1, Point2D & p2, const cv::Scalar & color, int thickness = 1);

  /**
   * Utility function to convert a Point2D object to a geometry_msgs::PoseStamped object
   * @return corresponding geometry_msgs::PoseStamped object
   */
  inline geometry_msgs::PoseStamped pointToPose(const Point2D &);

  /**
   * Utility function to convert a geometry_msgs::PoseStamped object to a Point2D object
   */
  inline void poseToPoint(Point2D &, const geometry_msgs::Pose &);

  /**
   * Utility function to convert (x, y) matrix coordinate to corresponding vector coordinate
   */
  inline int toIndex(int, int);

  /**
   * Utility function to draw a new connection
   */
  void drawNewConnection(int vertex_index, std::vector<Vertex> nodes);

  /**
   * Path finding with vanilla RRT
   */
  std::vector<Point2D> rrtPathFinding();

  /**
   * Path finding with Bi-directional RRT
   */
  std::vector<Point2D> biRRTPathFinding();

  /**
   * Utility function to sample a random point inside the map
   */
  Point2D getRandomPoint(double goal_bias);

   /**
   * Utility function to find a viable connection between a new point and an existing vertex
   */
  Point2D getPointForConnection(const Point2D & point1, const Point2D & point2);

   /**
   * Utility function to find the closest vertex
   */
  int getClosestVertex(const Point2D & random_point, std::vector<Vertex> vertex_list);


  ros::NodeHandle * nh_;
  ros::NodeHandle private_nh_;

  bool map_received_;
  std::unique_ptr<cv::Mat> map_;
  nav_msgs::OccupancyGrid::Ptr map_grid_;

  bool init_pose_received_;
  Point2D init_pose_;

  bool goal_received_;
  Point2D goal_;

  ros::Subscriber map_sub_;
  ros::Subscriber init_pose_sub_;
  ros::Subscriber goal_sub_;
  ros::Publisher path_pub_;

   /**
   * Utility parameters from the node server
   */
  int max_iterations_;
  int variation;
  float step_size_ ;
  float delta_;
  bool show_path;

};

}

#endif  // RRT_PLANNER_INCLUDE_RRT_PLANNER_RRT_PLANNER_H_
