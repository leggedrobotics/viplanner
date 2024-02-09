#ifndef WAYPOINT_RVIZ_PLUGIN_WAYPOINT_TOOL_H
#define WAYPOINT_RVIZ_PLUGIN_WAYPOINT_TOOL_H

#include <sstream>
#include <ros/ros.h>
#include <QObject>

#include <sensor_msgs/Joy.h>
#include <nav_msgs/Odometry.h>
#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <tf2/transform_datatypes.h>
#include <tf2_ros/transform_listener.h>

#include "rviz/display_context.h"
#include "rviz/properties/string_property.h"
#include "rviz/default_plugin/tools/pose_tool.h"

namespace rviz
{
class StringProperty;

class WaypointTool : public PoseTool
{
  Q_OBJECT
public:
  WaypointTool();
  virtual ~WaypointTool()
  {
  }
  virtual void onInitialize();

protected:
  virtual void odomHandler(const geometry_msgs::PoseWithCovarianceStampedConstPtr& odom);
  virtual void onPoseSet(double x, double y, double theta);

private Q_SLOTS:
  void updateTopic();

private:
  float vehicle_z;

  ros::NodeHandle nh_;
  ros::Subscriber sub_;
  ros::Publisher pub_;
  ros::Publisher pub_joy_;

  StringProperty* topic_property_;
};
}

#endif  // WAYPOINT_RVIZ_PLUGIN_WAYPOINT_TOOL_H
