/*
 * Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
 * Author: Pascal Roth
 * All rights reserved.
 *
 * SPDX-License-Identifier: BSD-3-Clause
 */

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <stdlib.h>
#include <ros/ros.h>
#include <iostream>

#include <message_filters/subscriber.h>
#include <message_filters/synchronizer.h>
#include <message_filters/sync_policies/approximate_time.h>

#include <std_msgs/Bool.h>
#include <std_msgs/Float32.h>
#include <nav_msgs/Path.h>
#include <nav_msgs/Odometry.h>

#include <geometry_msgs/PointStamped.h>
#include <geometry_msgs/PolygonStamped.h>

#include <sensor_msgs/Image.h>
#include <sensor_msgs/CompressedImage.h>
#include <sensor_msgs/CameraInfo.h>

#include <tf/transform_listener.h>
// #include <pcl_ros/transforms.h>
#include <tf/transform_datatypes.h>
#include <tf/transform_broadcaster.h>
#include <geometry_msgs/PoseWithCovarianceStamped.h>

#include <Eigen/Dense>
#include <Eigen/Core>

#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/calib3d.hpp>
#include <opencv2/imgproc.hpp>

#include <cmath>

#include <cv_bridge/cv_bridge.h>

using namespace std;
using namespace Eigen;

static const double mesh_size = 0.3;
static const int max_waypoints = 50;
Eigen::Matrix3d CAM_TO_ROBOT_FRAME = [] {
    Eigen::Matrix3d m;
    m << 0, 0, 1,
         1, 0, 0,
         0, 1, 0;
    return m;
}();
Eigen::Matrix3d FLIP_MAT = [] {
    Eigen::Matrix3d m;
    m << 1, 0, 0,
         0, -1, 0,
         0, 0, -1;
    return m;
}();
Eigen::Matrix3d ROT_MAT = [] {
    Eigen::Matrix3d m;
    m << 0, 0, 1,
         1, 0, 0,
         0, 1, 0;
    return m;
}();



class VIPlannerViz {
public:
    VIPlannerViz() : nh_("~") {
        // Load parameters from the ROS parameter server
        // parameter_name, variable_name, default_value
        nh_.param<std::string> ("vizTopic",      vizTopic_,      "/viz_path_depth");
        nh_.param<std::string> ("imgTopic",      img_topic_,     "/depth_camera_front_upper/depth/image_rect_raw");
        nh_.param<std::string> ("infoTopic",     info_topic_,    "/depth_camera_front_upper/depth/camera_info");
        nh_.param<std::string> ("pathTopic",     path_topic_,    "/path");
        nh_.param<std::string> ("goalTopic",     goal_topic_,    "/mp_waypoint");
        nh_.param<std::string> ("robot_frame",   robot_frame_,   "base");
        nh_.param<std::string> ("odom_frame",    odom_frame_,    "odom");
        nh_.param<std::string> ("domain",        domain_,        "depth");
        nh_.param<bool>        ("image_flip",    image_flip_,    true);
        nh_.param<float>       ("max_depth",     max_depth_,     10.0);


        // Subscribe to the image and the intrinsic matrix
        if (domain_ == "rgb") {
            subImage_ = nh_.subscribe<sensor_msgs::CompressedImage>(img_topic_, 1, &VIPlannerViz::imageRGBCallback, this);
        } else if (domain_ == "depth") {
            subImage_ = nh_.subscribe<sensor_msgs::Image>(img_topic_, 1, &VIPlannerViz::imageDepthCallback, this);
        } else {
            ROS_ERROR("Domain not supported!");
        }
        subCamInfo_ = nh_.subscribe<sensor_msgs::CameraInfo>(info_topic_, 1, &VIPlannerViz::camInfoCallback, this);

        // Subscribe to the path
        subPath_ = nh_.subscribe<nav_msgs::Path>(path_topic_, 1, &VIPlannerViz::pathCallback, this);
        // Subscribe to the goal
        subGoal_ = nh_.subscribe<geometry_msgs::PointStamped>(goal_topic_, 1, &VIPlannerViz::goalCallback, this);

        // Publish the image with the path
        pubImage_ = nh_.advertise<sensor_msgs::Image>(vizTopic_, 1);

    }

    // CALLBACKS
    void imageRGBCallback(const sensor_msgs::CompressedImage::ConstPtr& rgb_msg)
    {
        ROS_DEBUG_STREAM("Received rgb image " << rgb_msg->header.frame_id << ": " << rgb_msg->header.stamp.toSec());

        // image pose
        poseCallback(rgb_msg->header.frame_id);

        // RGB Image
        try {
            cv::Mat image = cv::imdecode(cv::Mat(rgb_msg->data), cv::IMREAD_COLOR);
            // rotate image 90 degrees counter clockwise
            if (!image_flip_) {
                cv::rotate(image, image, cv::ROTATE_90_COUNTERCLOCKWISE);
            }
            current_image_time_ = rgb_msg->header.stamp;
            image_ = image.clone();
            image_init_ = true;
        }
        catch (cv::Exception& e) {
            ROS_ERROR_STREAM("CvBridge Error: " << e.what());
        }
    }

    void imageDepthCallback(const sensor_msgs::Image::ConstPtr& depth_msg)
    {
        ROS_DEBUG_STREAM("Received depth image " << depth_msg->header.frame_id << ": " << depth_msg->header.stamp.toSec());

        // Image time and pose
        poseCallback(depth_msg->header.frame_id); // Assuming that poseCallback is defined somewhere
        current_image_time_ = depth_msg->header.stamp;

        // Depth image
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_16UC1);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Convert to Eigen matrix and apply operations
        cv::Mat img_mat = cv_ptr->image;
        cv::Mat depth_image_float;
        img_mat.convertTo(depth_image_float, CV_32FC1, 1.0/1000.0);
        cv::Mat mask = cv::Mat::zeros(img_mat.size(), img_mat.type());
        cv::compare(depth_image_float, std::numeric_limits<double>::infinity(), mask, cv::CMP_EQ);
        depth_image_float.setTo(0, mask);

        if (image_flip_)
        {
            cv::flip(depth_image_float, depth_image_float, 0); // 0 indicates vertical flip
        }

        image_ = depth_image_float.clone();
        image_init_ = true;
    }

    void poseCallback(const std::string& frame_id)
    {
        tf::StampedTransform transform;
        try {
            tf_listener.waitForTransform(odom_frame_, frame_id, ros::Time(0), ros::Duration(4.0));
            tf_listener.lookupTransform(odom_frame_, frame_id, ros::Time(0), transform);
        }
        catch (tf::TransformException& e) {
            ROS_ERROR_STREAM("Fail to transfer " << frame_id << " into " << odom_frame_ << " frame: " << e.what());
        }

        tf::poseTFToMsg(transform, pose_);
    }

    void camInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& cam_info_msg)
    {
        if (!intrinsics_init_)
        {
            ROS_INFO("Received camera info");

            // Extract the intrinsic matrix from the CameraInfo message
            intrinsics_.at<double>(0, 0) = cam_info_msg->K[0];
            intrinsics_.at<double>(0, 1) = cam_info_msg->K[1];
            intrinsics_.at<double>(0, 2) = cam_info_msg->K[2];
            intrinsics_.at<double>(1, 0) = cam_info_msg->K[3];
            intrinsics_.at<double>(1, 1) = cam_info_msg->K[4];
            intrinsics_.at<double>(1, 2) = cam_info_msg->K[5];
            intrinsics_.at<double>(2, 0) = cam_info_msg->K[6];
            intrinsics_.at<double>(2, 1) = cam_info_msg->K[7];
            intrinsics_.at<double>(2, 2) = cam_info_msg->K[8];

            intrinsics_init_ = true;
        }
    }

    void pathCallback(const nav_msgs::Path::ConstPtr& path_msg)
    {
        // Create an Eigen matrix with the same number of rows as the path
        Eigen::MatrixXf path_mat_new(max_waypoints, 3);

        // Copy the x, y, and z coordinates from the path message into the matrix
        for (int i = 0; i < path_msg->poses.size(); i++)
        {
            path_mat_new(i, 0) = path_msg->poses[i].pose.position.x;
            path_mat_new(i, 1) = path_msg->poses[i].pose.position.y;
            path_mat_new(i, 2) = path_msg->poses[i].pose.position.z;
        }

        // Assign the new path to the path_ member variable
        path_mat_ = path_mat_new;
        path_init_ = true;
    }

    void goalCallback(const geometry_msgs::PointStamped::ConstPtr& goal_msg)
    {
        // Extract the goal point from the message
        float x = goal_msg->point.x;
        float y = goal_msg->point.y;
        float z = goal_msg->point.z;

        // Assign the goal point to the goal_ member variable
        goal_ << x, y, z;
        goal_init_ = true;

        std::cout << "GOAL Received" << std::endl;
    }

    // HELPER FUNCTIONS
    MatrixXf TransformPoints(Vector3f translation, Quaternionf rotation, MatrixXf points) {
        // Convert the quaternion to a rotation matrix
        Matrix3f rotation_matrix = rotation.toRotationMatrix();
        // Multiply the translated points by the rotation matrix
        points = points * rotation_matrix.transpose();
        // Translate the points by the relative translation vector
        points.rowwise() += translation.transpose();

        return points;
    }

    void getOdom(Eigen::Vector3f& translation, Eigen::Quaternionf& rotation)
    {
        try
        {
            // Get the transformation from the reference frame to the target frame
            tf::StampedTransform transform;
            tf_listener.lookupTransform(odom_frame_, robot_frame_, ros::Time(0), transform);

            // Extract the translation and rotation from the transformation
            translation << transform.getOrigin().x(), transform.getOrigin().y(), transform.getOrigin().z();
            rotation = Eigen::Quaternionf(transform.getRotation().getW(), transform.getRotation().getX(), transform.getRotation().getY(), transform.getRotation().getZ());
        }
        catch (tf::TransformException& ex)
        {
            ROS_ERROR("%s", ex.what());
        }
    }

    // RUN NODE
    void run() {
        Eigen::Vector3f translation;
        Eigen::Quaternionf rotation;

        // Main loop
        while (ros::ok()) {
            if (path_init_ && goal_init_ && image_init_ && intrinsics_init_) {
                // Get the current robot pose
                getOdom(translation, rotation);

                // Transform the path into world frame
                MatrixXf transformed_path = TransformPoints(translation, rotation, path_mat_);

                // orientate camera
                cv::Mat cam_translation = (cv::Mat_<double>(3,1) << pose_.position.x, pose_.position.y, pose_.position.z);
                Quaterniond rotation_matrix = Quaterniond(pose_.orientation.w, pose_.orientation.x, pose_.orientation.y, pose_.orientation.z);
                Eigen::Matrix3d rotation_matrix_robotic_frame = rotation_matrix.toRotationMatrix() * CAM_TO_ROBOT_FRAME;
                cv::Mat cam_rotation;
                cv::eigen2cv(rotation_matrix_robotic_frame, cam_rotation);
                cv::Mat rot_vector;
                cv::Rodrigues(cam_rotation, rot_vector);

                // Project 3D points onto image plane
                std::vector<cv::Point2f> points2d;
                cv::Mat path_points;
                cv::eigen2cv(transformed_path, path_points);
                cv::projectPoints(path_points, rot_vector, cam_translation, intrinsics_, cv::noArray(), points2d);

                // Get the position of the path points in camera frame --> needed to get the radius of the sphere in the image
                std::vector<cv::Point3f> points3d(path_points.rows);
                for (int i = 0; i < path_points.rows; i++) {
                    cv::Mat p = path_points.row(i);  // get i-th row of path_points
                    cv::Mat p_double;
                    p.convertTo(p_double, CV_64F);  // convert to double
                    cv::Mat p_cam = cam_rotation * (p_double.t() - cam_translation);  // transpose row vector and subtract translation
                    cv::Point3f p_cam_3d(p_cam.at<double>(0), p_cam.at<double>(1), p_cam.at<double>(2));  // convert to Point3f
                    points3d[i] = p_cam_3d;
                }

                // Draw points on image
                cv::Mat outputImage;
                if (image_.channels() == 1) {
                    double min_val, max_val;
                    cv::minMaxLoc(image_, &min_val, &max_val);
                    cv::Mat gray_image;
                    cv::normalize(image_, gray_image, 0, 255, cv::NORM_MINMAX, CV_8UC1);
                    cv::cvtColor(gray_image, outputImage, cv::COLOR_GRAY2BGR);
                }
                else {
                    outputImage = image_.clone();
                }

                cv::Mat overlay_image = outputImage.clone();
                for (int i = 0; i < points2d.size(); i++) {
                    cv::Point3f p = points3d[i];
                    cv::Point2f p2d = points2d[i];
                    cv::Point2d center(p.x, p.y);
                    int radius = std::min(std::max(cvRound(5 * intrinsics_.at<double>(0,0) / p.z), 0), 5);
                    cv::circle(overlay_image, p2d, radius, cv::Scalar(0, 255, 0), cv::FILLED);
                }

                // Following line overlays transparent rectangle over the image
                cv::Mat final_img;
                double alpha = 0.4;  // Transparency factor.
                cv::addWeighted(overlay_image, alpha, outputImage, 1 - alpha, 0, final_img);

                // Publish as ROS image
                cv_bridge::CvImage cv_image;
                cv_image.header.stamp = current_image_time_;
                cv_image.encoding = sensor_msgs::image_encodings::BGR8;
                cv_image.image = final_img;

                pubImage_.publish(cv_image.toImageMsg());

                // Show resulting image
                // cv::imshow("Overlay", outputImage);
                // cv::waitKey(1);
            }

            ros::spinOnce();
            loop_rate_.sleep();
        }

    }

private:
    // ROS
    ros::NodeHandle nh_;
    ros::Subscriber subImage_;
    ros::Subscriber subCamInfo_;
    ros::Subscriber subGoal_;
    ros::Subscriber subPath_;
    ros::Publisher pubImage_;
    ros::Rate loop_rate_{10};
    ros::Time current_image_time_;
    tf::TransformListener tf_listener;

    // parameters
    std::string vizTopic_;
    std::string img_topic_;
    std::string info_topic_;
    std::string path_topic_;
    std::string goal_topic_;
    std::string robot_frame_;
    std::string odom_frame_;
    std::string domain_;
    float max_depth_;
    bool image_flip_;

    // Flags
    bool intrinsics_init_ = false;
    bool image_init_ = false;
    bool path_init_ = false;
    bool goal_init_ = false;
    bool renderer_init_ = false;

    // variables
    cv::Mat image_;
    Eigen::Vector3f goal_;
    Eigen::Matrix<float, max_waypoints, 3> path_mat_;
    cv::Mat intrinsics_ = cv::Mat::eye(3, 3, CV_64F);
    geometry_msgs::Pose pose_;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "VIPlannerViz");
    VIPlannerViz node;
    node.run();
    return 0;
}
