#include "rrt_planner/rrt_planner.h"



namespace rrt_planner
{

RRTPlanner::RRTPlanner(ros::NodeHandle * node)
: nh_(node),
  private_nh_("~"),
  map_received_(false),
  init_pose_received_(false),
  goal_received_(false)
{
  // Get map and path topics from parameter server
  std::string map_topic, path_topic;
  private_nh_.param<std::string>("/map_topic", map_topic, "/map");
  private_nh_.param<std::string>("/path_topic", path_topic, "/path");

  private_nh_.param<int>("/max_iteration", max_iterations_, 100000);
  private_nh_.param<int>("/variation", variation, 1);

  private_nh_.param<float>("/step_size", step_size_, 4.5);
  private_nh_.param<float>("/delta", delta_, 1);
  private_nh_.param<bool>("/show_path", show_path, true);

  // Subscribe to map topic
  map_sub_ = nh_->subscribe<const nav_msgs::OccupancyGrid::Ptr &>(
    map_topic, 1, &RRTPlanner::mapCallback, this);

  // Subscribe to initial pose topic that is published by RViz
  init_pose_sub_ = nh_->subscribe<const geometry_msgs::PoseWithCovarianceStamped::ConstPtr &>(
    "/initialpose", 1, &RRTPlanner::initPoseCallback, this);

  // Subscribe to goal topic that is published by RViz
  goal_sub_ = nh_->subscribe<const geometry_msgs::PoseStamped::ConstPtr &>(
    "/move_base_simple/goal", 1, &RRTPlanner::goalCallback, this);

  // Advertise topic where calculated path is going to be published
  path_pub_ = nh_->advertise<nav_msgs::Path>(path_topic, 1, true);

  // This loops until the node is running, will exit when the node is killed
  while (ros::ok()) {
    // if map, initial pose, and goal have been received
    // build the map image, draw initial pose and goal, and plan
    if (map_received_ && init_pose_received_ && goal_received_) {
      buildMapImage();
      drawGoalInitPose();
      plan();
    } else {
      if (map_received_) {
        displayMapImage();
      }
      ros::Duration(0.1).sleep();
      ros::spinOnce();
    }
  }
}

void RRTPlanner::mapCallback(const nav_msgs::OccupancyGrid::Ptr & msg)
{
  map_grid_ = msg;

  // Build and display the map image
  buildMapImage();
  displayMapImage();

  // Reset these values for a new planning iteration
  map_received_ = true;
  init_pose_received_ = false;
  goal_received_ = false;

  ROS_INFO("Map obtained successfully. Please provide initial pose and goal through RViz.");
}

void RRTPlanner::initPoseCallback(const geometry_msgs::PoseWithCovarianceStamped::ConstPtr & msg)
{
  if (init_pose_received_) {
    buildMapImage();
  }

  // Convert mas to Point2D
  poseToPoint(init_pose_, msg->pose.pose);

  // Reject the initial pose if the given point is occupied in the map
  if (!isPointUnoccupied(init_pose_)) {
    init_pose_received_ = false;
    ROS_WARN(
      "The initial pose specified is on or too close to an obstacle please specify another point");
  } else {
    init_pose_received_ = true;
    drawGoalInitPose();
    ROS_INFO("Initial pose obtained successfully.");
  }

  displayMapImage();
}

void RRTPlanner::goalCallback(const geometry_msgs::PoseStamped::ConstPtr & msg)
{
  if (goal_received_) {
    buildMapImage();
  }

  // Convert msg to Point2D
  poseToPoint(goal_, msg->pose);

  // Reject the goal pose if the given point is occupied in the map
  if (!isPointUnoccupied(goal_)) {
    goal_received_ = false;
    ROS_WARN("The goal specified is on or too close to an obstacle please specify another point");
  } else {
    goal_received_ = true;
    drawGoalInitPose();
    ROS_INFO("Goal obtained successfully.");
  }

  displayMapImage();
}

void RRTPlanner::drawGoalInitPose()
{
  if (goal_received_) {
    drawCircle(goal_, 3, cv::Scalar(12, 255, 43));
  }
  if (init_pose_received_) {
    drawCircle(init_pose_, 3, cv::Scalar(255, 200, 0));
  }
}

void RRTPlanner::plan()
{
  // Reset these values so planning only happens once for a
  // given pair of initial pose and goal points
  goal_received_ = false;
  init_pose_received_ = false;


  std::vector<Point2D> path_points;

  // Run until we either find the goal or reach the max iterations
  switch (variation)
  {
  case 1:
    path_points = biRRTPathFinding();
    break;
  
  default:
    path_points = rrtPathFinding();
    break;
  }

  if (path_points.size() > 0) {
    publishPath(path_points);
  } 
}

std::vector<Point2D> RRTPlanner::rrtPathFinding() 
{
  int current_iterations_ = 0;
  std::vector<Point2D> points = {};

  std::vector<Vertex> nodes;

  nodes.push_back(Vertex(init_pose_,0,-1));

  while (current_iterations_ < max_iterations_) {
    // get a random point on the map
    Point2D random_point = getRandomPoint(0.2);

    // find the closest known vertex to that point
    int closest_vertex = getClosestVertex(random_point, nodes);

    Point2D closest_point = nodes[closest_vertex].get_location();

    // try to move from the closest known vertex towards the random point
    Point2D potential_position = getPointForConnection(random_point, closest_point);
    if (potential_position != closest_point) {
      ROS_DEBUG("Distance to goal %.2f", goal_.getDistance(potential_position));

      ROS_DEBUG("Moved, closest vertex: %d", closest_vertex);

      // If successful increase our iterations
      current_iterations_++;

      Vertex new_vertex =Vertex(potential_position, nodes.size(), closest_vertex);
      nodes.push_back(new_vertex);

      int vertex_index = nodes.back().get_index();
      if (show_path) drawNewConnection(vertex_index,nodes);
      // check if we've reached our goal

      if (potential_position == goal_) {
        ROS_INFO("Hey, we reached our goal, index: %d", vertex_index);
        std::deque<int> index_path;
        int current_index = vertex_index;

        while (current_index >= 0 ) 
        {
          index_path.push_front(current_index);
          current_index = nodes.at(current_index).get_parent();
        }
        for (int i : index_path) {
          points.push_back(nodes.at(i).get_location());
        }
        return points;
      }
    }
    if (current_iterations_ == max_iterations_) ROS_INFO("Max iterations reached, no plan found.");
  }
  return points;
}

std::vector<Point2D> RRTPlanner::biRRTPathFinding() 
{
  int current_iterations_ = 0;
  std::vector<Point2D> points = {};

  std::vector<Vertex> v1 = {Vertex(init_pose_,0,-1)};
  std::vector<Vertex> v2 = {Vertex(goal_,0,-1)};

  while (current_iterations_ < max_iterations_) {
    // get a random point on the map
    Point2D random_point = getRandomPoint(0.05);

    // find the closest known vertex to that point
    int closest_vertex_1 = getClosestVertex(random_point, v1);

    Point2D closest_point_1 = v1[closest_vertex_1].get_location();

    // try to move from the closest known vertex towards the random point from the start
    Point2D potential_position_1 = getPointForConnection(random_point, closest_point_1);
    if (potential_position_1 != closest_point_1) {
      // If successful increase our iterations
      current_iterations_++;

      v1.push_back(Vertex(potential_position_1, v1.size(), closest_vertex_1));

      if (show_path) drawNewConnection(v1.back().get_index(),v1);

      int closest_vertex_2 = getClosestVertex(potential_position_1, v2);

      Point2D closest_point_2 = v2[closest_vertex_2].get_location();

      Point2D potential_position_2 = getPointForConnection(random_point, closest_point_2);

      if (potential_position_2 != closest_point_2) { 
        v2.push_back(Vertex(potential_position_2, v2.size(), closest_vertex_2));

        if (show_path) drawNewConnection(v2.back().get_index(),v2);
      }

      if( v1.size() > v2.size()) {
        std::vector<Vertex> temp = v1;
        v1 = v2;
        v2 = temp;
      }


      // check if we've reached our goal
      if (potential_position_2 == potential_position_1) {
        ROS_INFO("Hey, we reached our goal");
        std::deque<int> index_path;
        int current_index_start = v1.back().get_index();

        while (current_index_start >= 0 ) 
        {
          index_path.push_front(current_index_start);
          current_index_start = v1.at(current_index_start).get_parent();
        }
        for (int i : index_path) {
          points.push_back(v1.at(i).get_location());
        }

        int current_index_end = v2.back().get_index();

        while (current_index_end >= 0 ) 
        {
          points.push_back(v2.at(current_index_end).get_location());
          current_index_end = v2.at(current_index_end).get_parent();
        }

        return points;
      }
    }
    if (current_iterations_ == max_iterations_) ROS_INFO("Max iterations reached, no plan found.");
  }
  return points;
}





void RRTPlanner::publishPath(std::vector<Point2D> points)
{
  // Create new Path msg
  nav_msgs::Path path;
  path.header.frame_id = map_grid_->header.frame_id;
  path.header.stamp = ros::Time::now();
  
  for (Point2D point : points) {
    geometry_msgs::PoseStamped pose = pointToPose(point);
    path.poses.push_back(pose);
  }
  
  // Publish the calculated path
  path_pub_.publish(path);

  displayMapImage();
}

bool RRTPlanner::isPointUnoccupied(const Point2D & p)
{
  if (map_grid_->data[toIndex(p.x(), p.y())]) return false;
  return true;
}

void RRTPlanner::buildMapImage()
{
  // Create a new opencv matrix with the same height and width as the received map
  map_ = std::unique_ptr<cv::Mat>(new cv::Mat(map_grid_->info.height,
                                              map_grid_->info.width,
                                              CV_8UC3,
                                              cv::Scalar::all(255)));

  // Fill the opencv matrix pixels with the map points
  for (int i = 0; i < map_grid_->info.height; i++) {
    for (int j = 0; j < map_grid_->info.width; j++) {
      if (map_grid_->data[toIndex(i, j)]) {
        map_->at<cv::Vec3b>(map_grid_->info.height - i - 1, j) = cv::Vec3b(0, 0, 0);
      } else {
        map_->at<cv::Vec3b>(map_grid_->info.height - i - 1, j) = cv::Vec3b(255, 255, 255);
      }
    }
  }
}

void RRTPlanner::drawNewConnection(int vertex_index, std::vector<Vertex> nodes) {
  Point2D new_point= nodes.at(vertex_index).get_location();
  Point2D parent_point = nodes.at(nodes.at(vertex_index).get_parent()).get_location();
  drawLine(new_point,parent_point,cv::Scalar( 255, 0, 0 ),2);
  displayMapImage();
}

void RRTPlanner::displayMapImage(int delay)
{
  cv::imshow("Output", *map_);
  cv::waitKey(delay);
}


void RRTPlanner::drawCircle(Point2D & p, int radius, const cv::Scalar & color)
{
  cv::circle(
    *map_,
    cv::Point(p.y(), map_grid_->info.height - p.x() - 1),
    radius,
    color,
    -1);
}

void RRTPlanner::drawLine(Point2D & p1, Point2D & p2, const cv::Scalar & color, int thickness)
{
  cv::line(
    *map_,
    cv::Point(p2.y(), map_grid_->info.height - p2.x() - 1),
    cv::Point(p1.y(), map_grid_->info.height - p1.x() - 1),
    color,
    thickness);
}

inline geometry_msgs::PoseStamped RRTPlanner::pointToPose(const Point2D & p)
{
  geometry_msgs::PoseStamped pose;
  pose.pose.position.x = p.y() * map_grid_->info.resolution;
  pose.pose.position.y = p.x() * map_grid_->info.resolution;
  pose.header.stamp = ros::Time::now();
  pose.header.frame_id = map_grid_->header.frame_id;
  return pose;
}

inline void RRTPlanner::poseToPoint(Point2D & p, const geometry_msgs::Pose & pose)
{
  p.x(pose.position.y / map_grid_->info.resolution);
  p.y(pose.position.x / map_grid_->info.resolution);
}

inline int RRTPlanner::toIndex(int x, int y)
{
  return x * map_grid_->info.width + y;
}

Point2D RRTPlanner::getRandomPoint(double goal_sample_rate) 
{
  double r = rand() / (double) RAND_MAX;
  if (r < goal_sample_rate) {
      return goal_;
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  float map_width = map_grid_->info.width;
  float map_height = map_grid_->info.height;
  std::uniform_real_distribution<> x(0, map_height);
  std::uniform_real_distribution<> y(0, map_width);

  return Point2D(x(gen), y(gen));

}


int RRTPlanner::getClosestVertex(const Point2D & random_point, std::vector<Vertex> vertex_list) 
{
  int closest = -1;

  // closest_distance will keep track of the closest distance we find
  float closest_distance = std::numeric_limits<float>::infinity();

  // current_distance will keep track of the distance of the current
  float current_distance = std::numeric_limits<float>::infinity();

  // iterate through the vertex list to find the closest
  for (Vertex v : vertex_list) {
    current_distance = v.get_location().getDistance(random_point);

    // If the current distance is closer than what was previously
    // saved, update
    if (current_distance < closest_distance) {
      ROS_DEBUG("Closest distance: %.5f, vertex: %d.",
                current_distance, v.get_index());
      closest = v.get_index();
      closest_distance = current_distance;
    }
  }
  return closest;
}

Point2D RRTPlanner::getPointForConnection(const Point2D & new_point, const Point2D & closest_point) 
{
  // get the angle between the random point and our closest point (in rads)
  float theta = atan2(new_point.y() - closest_point.y(), new_point.x() - closest_point.x());

  // proposed new point step_size_ from our closest vertex towards
  // the random point
  Point2D propose_point = new_point;
  if (propose_point.getDistance(closest_point) > step_size_) {
    propose_point = Point2D(closest_point.x() + step_size_ * cos(theta), 
                            closest_point.y() + step_size_ * sin(theta));
  }

  if (!isPointUnoccupied(propose_point)) {
    return closest_point;
  }

  Point2D current_point = closest_point;

  while (current_point.getDistance(propose_point) > delta_) {
    // increment towards end point
    int new_x = round(current_point.x() + delta_ * cos(theta));
    int new_y = round(current_point.y() + delta_ * sin(theta));
    current_point.x(new_x);
    current_point.y(new_y);

    theta = atan2(propose_point.y() - current_point.y(), propose_point.x() - current_point.x());

    // check for collision
    if (!isPointUnoccupied(current_point))
      return closest_point;
  }
  return propose_point;
}


}  // namespace rrt_planner
