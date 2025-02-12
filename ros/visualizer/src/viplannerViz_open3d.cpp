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

#include <open3d/Open3D.h>
#include <open3d/geometry/TriangleMesh.h>
#include <open3d/visualization/gui/Application.h>
#include <open3d/visualization/rendering/MaterialRecord.h>
#include <open3d/visualization/rendering/Camera.h>
#include <open3d/visualization/rendering/filament/FilamentEngine.h>
#include <open3d/visualization/rendering/filament/FilamentRenderer.h>

#include <cmath>

#include <cv_bridge/cv_bridge.h>

using namespace open3d;
using namespace std;
using namespace Eigen;

static const double mesh_size = 0.3;
static const int n_waypoints = 51;
 Eigen::Matrix3f CAM_TO_ROBOT_FRAME = [] {
    Eigen::Matrix3f m;
    m << 0, 0, 1,
         1, 0, 0,
         0, 1, 0;
    return m;
}();


// Headless rendering requires Open3D to be compiled with OSMesa support.
// Add -DENABLE_HEADLESS_RENDERING=ON when you run CMake.
static const bool kUseHeadless = true;

class VIPlannerViz {
public:
    VIPlannerViz(open3d::visualization::gui::Application& app) : app_(app), nh_("~") {
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
        nh_.param<bool>        ("image_flip",    image_flip_,    false);
        nh_.param<float>       ("max_depth",     max_depth_,  10.0);


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

        // Initialize the open3d objects
        if (kUseHeadless) {
            open3d::visualization::rendering::EngineInstance::EnableHeadless();
        }

        mtl.base_color = Eigen::Vector4f(1.f, 1.f, 1.f, 1.f);
        mtl.shader = "defaultUnlit";

        std::vector<std::shared_ptr<open3d::geometry::TriangleMesh>> small_spheres(n_waypoints);
        std::vector<std::shared_ptr<open3d::geometry::TriangleMesh>> small_spheres_fear(n_waypoints);
        std::vector<std::shared_ptr<open3d::geometry::TriangleMesh>> mesh_sphere(n_waypoints);
        std::vector<std::shared_ptr<open3d::geometry::TriangleMesh>> mesh_sphere_fear(n_waypoints);

        mesh_box = open3d::geometry::TriangleMesh::CreateBox(mesh_size/20.0);
        mesh_box->PaintUniformColor(Vector3d(0.0, 0.0, 1.0));  // blue

        for (int i = 0; i < n_waypoints; ++i) {
            small_spheres[i]         = open3d::geometry::TriangleMesh::CreateSphere(mesh_size/20.0);
            small_spheres[i]->PaintUniformColor(Vector3d(0.4, 1.0, 0.1)); // green
            small_spheres_fear[i]    = open3d::geometry::TriangleMesh::CreateSphere(mesh_size/20.0);
            small_spheres_fear[i]->PaintUniformColor(Vector3d(0.99, 0.2, 0.1));  // red
            mesh_sphere[i]           = open3d::geometry::TriangleMesh::CreateSphere(mesh_size/5.0);
            mesh_sphere[i]->PaintUniformColor(Vector3d(0.4, 1.0, 0.1));  // green
            mesh_sphere_fear[i]      = open3d::geometry::TriangleMesh::CreateSphere(mesh_size/5.0);
            mesh_sphere_fear[i]->PaintUniformColor(Vector3d(0.99, 0.2, 0.1));  // red
        }



    }

    // CALLBACKS
    void imageRGBCallback(const sensor_msgs::CompressedImage::ConstPtr& rgb_msg)
    {
        ROS_DEBUG_STREAM("Received rgb image " << rgb_msg->header.frame_id << ": " << rgb_msg->header.stamp.toSec());

        // image pose
        geometry_msgs::Pose pose_ = poseCallback(rgb_msg->header.frame_id);

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
        geometry_msgs::Pose pose_ = poseCallback(depth_msg->header.frame_id); // Assuming that poseCallback is defined somewhere
        current_image_time_ = depth_msg->header.stamp;

        // Depth image
        cv_bridge::CvImagePtr cv_ptr;
        try
        {
            cv_ptr = cv_bridge::toCvCopy(depth_msg, sensor_msgs::image_encodings::TYPE_32FC1);
        }
        catch (cv_bridge::Exception& e)
        {
            ROS_ERROR("cv_bridge exception: %s", e.what());
            return;
        }

        // Convert to Eigen matrix and apply operations
        cv::Mat img_mat = cv_ptr->image;
        img_mat /= 1000;
        cv::Mat mask = cv::Mat::zeros(img_mat.size(), img_mat.type());
        cv::compare(img_mat, std::numeric_limits<double>::infinity(), mask, cv::CMP_EQ);
        img_mat.setTo(0, mask);

        if (image_flip_)
        {
            cv::flip(img_mat, img_mat, 1); // 1 indicates horizontal flip
        }

        image_ = img_mat.clone();
        image_init_ = true;
    }

    geometry_msgs::Pose poseCallback(const std::string& frame_id, const std::string& target_frame_id = "")
    {
        std::string target_frame = target_frame_id.empty() ? odom_frame_ : target_frame_id;

        tf::StampedTransform transform;
        try {
            tf_listener.waitForTransform(target_frame, frame_id, ros::Time(0), ros::Duration(4.0));
            tf_listener.lookupTransform(target_frame, frame_id, ros::Time(0), transform);
        }
        catch (tf::TransformException& e) {
            ROS_ERROR_STREAM("Fail to transfer " << frame_id << " into " << target_frame << " frame: " << e.what());
        }

        geometry_msgs::Pose pose;
        tf::poseTFToMsg(transform, pose);

        return pose;
    }

    void camInfoCallback(const sensor_msgs::CameraInfo::ConstPtr& cam_info_msg)
    {
        if (!intrinsics_init_)
        {
            ROS_INFO("Received camera info");

            // Extract the camera intrinsic matrix from the message
            intrinsics_ << cam_info_msg->K[0], cam_info_msg->K[1], cam_info_msg->K[2],
                           cam_info_msg->K[3], cam_info_msg->K[4], cam_info_msg->K[5],
                           cam_info_msg->K[6], cam_info_msg->K[7], cam_info_msg->K[8];

            intrinsics_init_ = true;
        }
    }

    void pathCallback(const nav_msgs::Path::ConstPtr& path_msg)
    {
        // Create an Eigen matrix with the same number of rows as the path
        Eigen::MatrixXf path_mat_new(path_msg->poses.size(), 3);

        // check if path length is same as expected length
        if (path_mat_new.rows() != n_waypoints)
        {
            ROS_ERROR("Path length is not same as expected length");
            return;
        }

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
        // Print the transformed points
        std::cout << points << std::endl;

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
                std::cout << "All data received" << std::endl;

                if (!renderer_init_) {
                    auto *renderer =
                        new open3d::visualization::rendering::FilamentRenderer(
                            open3d::visualization::rendering::EngineInstance::GetInstance(), intrinsics_(0, 2), intrinsics_(1, 2),
                            open3d::visualization::rendering::EngineInstance::GetResourceManager()
                    );
                    renderer_init_ = true;
                    std::cout << "Renderer created" << std::endl;
                }

                // Get the current robot pose
                getOdom(translation, rotation);

                // Transform the path
                MatrixXf transformed_path = TransformPoints(translation, rotation, path_mat_);

                // create open3d scene
                open3d::visualization::rendering::Open3DScene *scene = new open3d::visualization::rendering::Open3DScene(*renderer);

                std::cout << "Scene created" << std::endl;

                // Translate the points and add them to the scene
                for (int i = 0; i < n_waypoints; ++i) {
                    small_spheres[i]->Translate(transformed_path.row(i).cast<double>());
                    scene->AddGeometry("small_sphere" + std::to_string(i), small_spheres[i].get(), mtl);
                }

                std::cout << "Waypoint added" << std::endl;

                // orientate camera
                Vector3f cam_translation = Vector3f(pose_.position.x, pose_.position.y, pose_.position.z);
                Quaternionf cam_rotation = Quaternionf(pose_.orientation.w, pose_.orientation.x, pose_.orientation.y, pose_.orientation.z);
                Matrix3f rotation_matrix = cam_rotation.toRotationMatrix();
                Vector3f target_vec = cam_translation + cam_rotation * CAM_TO_ROBOT_FRAME * Vector3f(0, 0, -1);

                scene->GetCamera()->SetProjection(60.0f, float(intrinsics_(0, 2)) / float(intrinsics_(1, 2)), 0.1f,
                                                10.0f, open3d::visualization::rendering::Camera::FovType::Vertical);
                scene->GetCamera()->LookAt(target_vec, cam_translation,
                                        Vector3f(1, 0, 0));

                std::cout << "Camera set" << std::endl;

                auto o3dImage = app_.RenderToImage(*renderer, scene->GetView(), scene->GetScene(),
                                            intrinsics_(0, 2), intrinsics_(1, 2));

                if (intrinsics_(0, 2) != image_.size[0] || intrinsics_(1, 2) != image_.size[1]) {
                    throw std::runtime_error("Image sizes do not match");
                }

                std::cout << "Image rendered" << std::endl;

                // Convert Open3D image to OpenCV format
                cv::Mat o3dMat((*o3dImage).height_, (*o3dImage).width_, CV_8UC3, (*o3dImage).data_.data());
                cv::cvtColor(o3dMat, o3dMat, cv::COLOR_RGB2BGR);

                // Create mask where Open3D image is not white
                cv::Mat mask;
                cv::cvtColor(o3dMat, mask, cv::COLOR_BGR2GRAY);
                cv::threshold(mask, mask, 1, 255, cv::THRESH_BINARY);

                // Blend images together
                cv::Mat blended = image_.clone();
                float alpha = 0.0;
                cv::addWeighted(blended, 1-alpha, o3dMat, alpha, 0, blended, CV_8UC3);
                o3dMat.copyTo(blended, mask);

                std::cout << "Image blended" << std::endl;

                // Publish as ROS image
                cv_bridge::CvImage cv_image;
                cv_image.header.stamp = current_image_time_;
                cv_image.encoding = sensor_msgs::image_encodings::BGR8;
                cv_image.image = blended;
                pubImage_.publish(cv_image.toImageMsg());

                // Show resulting image
                cv::imshow("Overlay", blended);
                cv::waitKey(1);

                delete scene;

            }

            ros::spinOnce();
            loop_rate_.sleep();
        }

        delete renderer;
        app_.OnTerminate();

    }

private:
    // input Argument
    open3d::visualization::gui::Application &app_;

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
    Eigen::Matrix<float, n_waypoints, 3> path_mat_;
    Eigen::Matrix<float, 3, 3> intrinsics_;
    geometry_msgs::Pose pose_;

    // INIT OPEN3d objects
    open3d::visualization::rendering::MaterialRecord mtl;

    std::vector<std::shared_ptr<open3d::geometry::TriangleMesh>> small_spheres;
    std::vector<std::shared_ptr<open3d::geometry::TriangleMesh>> small_spheres_fear;
    std::vector<std::shared_ptr<open3d::geometry::TriangleMesh>> mesh_sphere;
    std::vector<std::shared_ptr<open3d::geometry::TriangleMesh>> mesh_sphere_fear;
    std::shared_ptr<open3d::geometry::TriangleMesh> mesh_box;
    open3d::visualization::rendering::FilamentRenderer *renderer;
};

int main(int argc, char** argv) {
    ros::init(argc, argv, "VIPlannerViz");

    open3d::visualization::gui::Application &app = open3d::visualization::gui::Application::GetInstance();
    app.Initialize("/usr/local/include/open3d/include/open3d/resources");

    VIPlannerViz node(app);
    node.run();
    return 0;
}
