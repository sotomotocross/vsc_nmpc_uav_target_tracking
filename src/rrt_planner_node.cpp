#include "rrt_planner/rrt_planner.h"

int main(int argv, char ** argc)
{
  ros::init(argv, argc, "rrt_planner");
  ros::NodeHandle node;
  // ros::Rate loop_rate(10.0);
  new rrt_planner::RRTPlanner(&node);
}
