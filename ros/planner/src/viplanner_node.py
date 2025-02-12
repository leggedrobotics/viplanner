# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# python
import os
import sys
import time
import warnings
from typing import Optional, Tuple

import cv2
import cv_bridge
import numpy as np
import PIL

# ROS
import ros_numpy
import rospkg
import rospy
import scipy.spatial.transform as stf
import tf2_geometry_msgs
import tf2_ros
import torch
from geometry_msgs.msg import Point, PointStamped, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import CameraInfo, CompressedImage, Image, Joy
from std_msgs.msg import Float32, Header, Int16
from visualization_msgs.msg import Marker

# init ros node
rospack = rospkg.RosPack()
pack_path = rospack.get_path("viplanner_node")
sys.path.append(pack_path)

# VIPlanner
from src.m2f_inference import Mask2FormerInference
from src.vip_inference import VIPlannerInference
from utils.rosutil import ROSArgparse

warnings.filterwarnings("ignore")

# conversion matrix from ROS camera convention (z-forward, y-down, x-right)
# to robotics convention (x-forward, y-left, z-up)
ROS_TO_ROBOTICS_MAT = stf.Rotation.from_euler("XYZ", [-90, 0, -90], degrees=True).as_matrix()
CAMERA_FLIP_MAT = stf.Rotation.from_euler("XYZ", [180, 0, 0], degrees=True).as_matrix()


class VIPlannerNode:
    """VIPlanner ROS Node Class"""

    debug: bool = False

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

        # init planner algo class
        self.vip_algo = VIPlannerInference(self.cfg)

        if self.vip_algo.train_cfg.sem:
            # init semantic network
            self.m2f_inference = Mask2FormerInference(
                config_file=args.m2f_config_path,
                checkpoint_file=args.m2f_model_path,
            )
            self.m2f_timer_data = Float32()
            self.m2f_timer_pub = rospy.Publisher(self.cfg.m2f_timer_topic, Float32, queue_size=10)

        # init transforms
        self.tf_buffer = tf2_ros.Buffer(rospy.Duration(100.0))  # tf buffer length
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer)

        # init bridge
        self.bridge = cv_bridge.CvBridge()

        # init flags
        self.is_goal_init = False
        self.ready_for_planning_depth = False
        self.ready_for_planning_rgb_sem = False
        self.is_goal_processed = False
        self.is_smartjoy = False
        self.goal_cam_frame_set = False
        self.init_goal_trans = True

        # planner status
        self.planner_status = Int16()
        self.planner_status.data = 0

        # fear reaction
        self.fear_buffter = 0
        self.is_fear_reaction = False

        # process time
        self.vip_timer_data = Float32()
        self.vip_timer_pub = rospy.Publisher("/viplanner/timer", Float32, queue_size=10)

        # depth and rgb image message
        self.depth_header: Header = Header()
        self.rgb_header: Header = Header()
        self.depth_img: np.ndarray = None
        self.depth_pose: np.ndarray = None
        self.sem_rgb_img: np.ndarray = None
        self.sem_rgb_odom: np.ndarray = None
        self.pix_depth_cam_frame: np.ndarray = None
        rospy.Subscriber(
            self.cfg.depth_topic,
            Image,
            callback=self.depthCallback,
            queue_size=1,
            buff_size=2**24,
        )

        if self.vip_algo.train_cfg.sem or self.vip_algo.train_cfg.rgb:
            if self.cfg.compressed:
                rospy.Subscriber(
                    self.cfg.rgb_topic,
                    CompressedImage,
                    callback=self.imageCallbackCompressed,
                    queue_size=1,
                    buff_size=2**24,
                )
            else:
                rospy.Subscriber(
                    self.cfg.rgb_topic,
                    Image,
                    callback=self.imageCallback,
                    queue_size=1,
                    buff_size=2**24,
                )
        else:
            self.ready_for_planning_rgb_sem = True

        # subscribe to further topics
        self.goal_world_frame: PointStamped = None
        self.goal_robot_frame: PointStamped = None
        rospy.Subscriber(self.cfg.goal_topic, PointStamped, self.goalCallback)
        rospy.Subscriber("/joy", Joy, self.joyCallback, queue_size=10)

        # camera info subscribers
        self.K_depth: np.ndarray = np.zeros((3, 3))
        self.K_rgb: np.ndarray = np.zeros((3, 3))
        self.depth_intrinsics_init: bool = False
        self.rgb_intrinsics_init: bool = False
        rospy.Subscriber(
            self.cfg.depth_info_topic,
            CameraInfo,
            callback=self.depthCamInfoCallback,
        )
        rospy.Subscriber(
            self.cfg.rgb_info_topic,
            CameraInfo,
            callback=self.rgbCamInfoCallback,
        )

        # publish effective goal pose and marker with max distance (circle)
        # and direct line (line)
        self.crop_goal_pub = rospy.Publisher("viplanner/visualization/crop_goal", PointStamped, queue_size=1)
        self.marker_circ_pub = rospy.Publisher("viplanner/visualization/odom_circle", Marker, queue_size=1)
        self.marker_line_pub = rospy.Publisher("viplanner/visualization/goal_line", Marker, queue_size=1)
        self.marker_circ: Marker = None
        self.marker_line: Marker = None
        self.max_goal_distance: float = 10.0
        self._init_markers()

        # planning status topics
        self.status_pub = rospy.Publisher("/viplanner/status", Int16, queue_size=10)

        # path topics
        self.path_pub = rospy.Publisher(self.cfg.path_topic, Path, queue_size=10)
        self.path_viz_pub = rospy.Publisher(self.cfg.path_topic + "_viz", Path, queue_size=10)
        self.fear_path_pub = rospy.Publisher(self.cfg.path_topic + "_fear", Path, queue_size=10)

        # viz semantic image
        self.m2f_pub = rospy.Publisher("/viplanner/sem_image/compressed", CompressedImage, queue_size=3)

        rospy.loginfo("VIPlanner Ready.")
        return

    def spin(self):
        r = rospy.Rate(self.cfg.main_freq)
        while not rospy.is_shutdown():
            if all(
                (
                    self.ready_for_planning_rgb_sem,
                    self.ready_for_planning_depth,
                    self.is_goal_init,
                    self.goal_cam_frame_set,
                )
            ):
                # copy current data
                cur_depth_image = self.depth_img.copy()
                cur_depth_pose = self.depth_pose.copy()

                if self.vip_algo.train_cfg.sem or self.vip_algo.train_cfg.rgb:
                    cur_rgb_pose = self.sem_rgb_odom.copy()
                    cur_rgb_image = self.sem_rgb_img.copy()

                    # warp rgb image
                    if False:
                        start = time.time()
                        if self.pix_depth_cam_frame is None:
                            self.initPixArray(cur_depth_image.shape)
                        (
                            cur_rgb_image,
                            overlap_ratio,
                            depth_zero_ratio,
                        ) = self.imageWarp(
                            cur_rgb_image,
                            cur_depth_image,
                            cur_rgb_pose,
                            cur_depth_pose,
                        )
                        time_warp = time.time() - start

                        if overlap_ratio < self.cfg.overlap_ratio_thres:
                            rospy.logwarn_throttle(
                                2.0,
                                (
                                    "Waiting for new semantic image since"
                                    f" overlap ratio is {overlap_ratio} <"
                                    f" {self.cfg.overlap_ratio_thres}, with"
                                    f" depth zero ratio {depth_zero_ratio}"
                                ),
                            )
                            self.pubPath(np.zeros((51, 3)), self.is_goal_init)
                            continue

                        if depth_zero_ratio > self.cfg.depth_zero_ratio_thres:
                            rospy.logwarn_throttle(
                                2.0,
                                (
                                    "Waiting for new depth image since depth"
                                    f" zero ratio is {depth_zero_ratio} >"
                                    f" {self.cfg.depth_zero_ratio_thres}, with"
                                    f" overlap ratio {overlap_ratio}"
                                ),
                            )
                            self.pubPath(np.zeros((51, 3)), self.is_goal_init)
                            continue
                    else:
                        time_warp = 0.0
                else:
                    time_warp = 0.0
                    self.time_sem = 0.0

                # project goal
                goal_cam_frame = self.goalProjection(cur_depth_pose=cur_depth_pose)

                # Network Planning
                start = time.time()
                if self.vip_algo.train_cfg.sem or self.vip_algo.train_cfg.rgb:
                    waypoints, fear = self.vip_algo.plan(cur_depth_image, cur_rgb_image, goal_cam_frame)
                else:
                    waypoints, fear = self.vip_algo.plan_depth(cur_depth_image, goal_cam_frame)
                time_planner = time.time() - start

                start = time.time()

                # transform waypoint to robot frame (prev in depth cam frame
                # with robotics convention)
                waypoints = (self.cam_rot @ waypoints.T).T + self.cam_offset

                # publish time
                self.vip_timer_data.data = time_planner * 1000
                self.vip_timer_pub.publish(self.vip_timer_data)

                # check goal less than converage range
                if (
                    (np.sqrt(goal_cam_frame[0][0] ** 2 + goal_cam_frame[0][1] ** 2) < self.cfg.conv_dist)
                    and self.is_goal_processed
                    and (not self.is_smartjoy)
                ):
                    self.ready_for_planning = False
                    self.is_goal_init = False
                    # planner status -> Success
                    if self.planner_status.data == 0:
                        self.planner_status.data = 1
                        self.status_pub.publish(self.planner_status)
                    rospy.loginfo("Goal Arrived")

                # check for path with high risk (=fear) path
                if fear > 0.7:
                    self.is_fear_reaction = True
                    is_track_ahead = self.isForwardTraking(waypoints)
                    self.fearPathDetection(fear, is_track_ahead)
                    if self.is_fear_reaction:
                        rospy.logwarn_throttle(2.0, "current path prediction is invalid.")
                        # planner status -> Fails
                        if self.planner_status.data == 0:
                            self.planner_status.data = -1
                            self.status_pub.publish(self.planner_status)

                # publish path
                self.pubPath(waypoints, self.is_goal_init)

                time_other = time.time() - start
                if self.vip_algo.train_cfg.pre_train_sem:
                    total_time = round(time_warp + self.time_sem + time_planner + time_other, 4)
                    print(
                        "Path predicted in"
                        f" {total_time}s"
                        f" \t warp: {round(time_warp, 4)}s \t sem:"
                        f" {round(self.time_sem, 4)}s \t planner:"
                        f" {round(time_planner, 4)}s \t other:"
                        f" {round(time_other, 4)}s"
                    )
                    self.time_sem = 0
                else:
                    print(
                        "Path predicted in"
                        f" {round(time_warp + time_planner + time_other, 4)}s"
                        f" \t warp: {round(time_warp, 4)}s \t planner:"
                        f" {round(time_planner, 4)}s \t other:"
                        f" {round(time_other, 4)}s"
                    )

            r.sleep()
        rospy.spin()

    """GOAL PROJECTION"""

    def goalProjection(self, cur_depth_pose: np.ndarray):
        cur_goal_robot_frame = np.array(
            [
                self.goal_robot_frame.point.x,
                self.goal_robot_frame.point.y,
                self.goal_robot_frame.point.z,
            ]
        )
        cur_goal_world_frame = np.array(
            [
                self.goal_world_frame.point.x,
                self.goal_world_frame.point.y,
                self.goal_world_frame.point.z,
            ]
        )

        if np.linalg.norm(cur_goal_robot_frame[:2]) > self.max_goal_distance:
            # crop goal position
            cur_goal_robot_frame[:2] = (
                cur_goal_robot_frame[:2] / np.linalg.norm(cur_goal_robot_frame[:2]) * (self.max_goal_distance / 2)
            )
            crop_goal = PointStamped()
            crop_goal.header.stamp = self.depth_header.stamp
            crop_goal.header.frame_id = self.cfg.robot_id
            crop_goal.point.x = cur_goal_robot_frame[0]
            crop_goal.point.y = cur_goal_robot_frame[1]
            crop_goal.point.z = cur_goal_robot_frame[2]
            self.crop_goal_pub.publish(crop_goal)

            # update markers
            self.marker_circ.color.a = 0.1
            self.marker_circ.pose.position = Point(cur_depth_pose[0], cur_depth_pose[1], cur_depth_pose[2])
            self.marker_circ_pub.publish(self.marker_circ)
            self.marker_line.points = []
            self.marker_line.points.append(
                Point(cur_depth_pose[0], cur_depth_pose[1], cur_depth_pose[2])
            )  # world frame
            self.marker_line.points.append(
                Point(
                    cur_goal_world_frame[0],
                    cur_goal_world_frame[1],
                    cur_goal_world_frame[2],
                )
            )  # world frame
            self.marker_line_pub.publish(self.marker_line)
        else:
            self.marker_circ.color.a = 0
            self.marker_circ_pub.publish(self.marker_circ)
            self.marker_line.points = []
            self.marker_line_pub.publish(self.marker_line)

        goal_cam_frame = self.cam_rot.T @ (cur_goal_robot_frame - self.cam_offset).T
        return torch.tensor(goal_cam_frame, dtype=torch.float32)[None, ...]

    def _init_markers(self):
        if isinstance(self.vip_algo.train_cfg.data_cfg, list):
            self.max_goal_distance = self.vip_algo.train_cfg.data_cfg[0].max_goal_distance
        else:
            self.max_goal_distance = self.vip_algo.train_cfg.data_cfg.max_goal_distance

        # setup circle marker
        self.marker_circ = Marker()
        self.marker_circ.header.frame_id = self.cfg.world_id
        self.marker_circ.type = Marker.SPHERE
        self.marker_circ.action = Marker.ADD
        self.marker_circ.scale.x = self.max_goal_distance  # only half of the distance
        self.marker_circ.scale.y = self.max_goal_distance  # only half of the distance
        self.marker_circ.scale.z = 0.01
        self.marker_circ.color.a = 0.1
        self.marker_circ.color.r = 0.0
        self.marker_circ.color.g = 0.0
        self.marker_circ.color.b = 1.0
        self.marker_circ.pose.orientation.w = 1.0

        # setip line marker
        self.marker_line = Marker()
        self.marker_line.header.frame_id = self.cfg.world_id
        self.marker_line.type = Marker.LINE_STRIP
        self.marker_line.action = Marker.ADD
        self.marker_line.scale.x = 0.1
        self.marker_line.color.a = 1.0
        self.marker_line.color.r = 0.0
        self.marker_line.color.g = 0.0
        self.marker_line.color.b = 1.0
        self.marker_line.pose.orientation.w = 1.0
        return

    """RGB/ SEM IMAGE WARP"""

    def initPixArray(self, img_shape: tuple):
        # get image plane mesh grid
        pix_u = np.arange(0, img_shape[1])
        pix_v = np.arange(0, img_shape[0])
        grid = np.meshgrid(pix_u, pix_v)
        pixels = np.vstack(list(map(np.ravel, grid))).T
        pixels = np.hstack([pixels, np.ones((len(pixels), 1))])  # add ones for 3D coordinates

        # transform to camera frame
        k_inv = np.linalg.inv(self.K_depth)
        pix_cam_frame = np.matmul(k_inv, pixels.T)  # pixels in ROS camera convention (z forward, x right, y down)

        # reorder to be in "robotics" axis order (x forward, y left, z up)
        self.pix_depth_cam_frame = pix_cam_frame[[2, 0, 1], :].T * np.array([1, -1, -1])
        return

    def imageWarp(
        self,
        rgb_img: np.ndarray,
        depth_img: np.ndarray,
        pose_rgb: np.ndarray,
        pose_depth: np.ndarray,
    ) -> np.ndarray:
        # get 3D points of depth image
        depth_rot = (
            stf.Rotation.from_quat(pose_depth[3:]).as_matrix() @ ROS_TO_ROBOTICS_MAT
        )  # convert orientation from ROS camera to robotics=world frame
        if not self.cfg.image_flip:
            # rotation is included in ROS_TO_ROBOTICS_MAT and has to be
            # removed when not fliped
            depth_rot = depth_rot @ CAMERA_FLIP_MAT
        dep_im_reshaped = depth_img.reshape(-1, 1)
        depth_zero_ratio = np.sum(np.round(dep_im_reshaped, 5) == 0) / len(dep_im_reshaped)
        points = dep_im_reshaped * (depth_rot @ self.pix_depth_cam_frame.T).T + pose_depth[:3]

        # transform points to semantic camera frame
        points_sem_cam_frame = (
            (stf.Rotation.from_quat(pose_rgb[3:]).as_matrix() @ ROS_TO_ROBOTICS_MAT @ CAMERA_FLIP_MAT).T
            @ (points - pose_rgb[:3]).T
        ).T

        # normalize points
        points_sem_cam_frame_norm = points_sem_cam_frame / points_sem_cam_frame[:, 0][:, np.newaxis]

        # reorder points be camera convention (z-forward)
        points_sem_cam_frame_norm = points_sem_cam_frame_norm[:, [1, 2, 0]] * np.array([-1, -1, 1])
        # transform points to pixel coordinates
        pixels = (self.K_rgb @ points_sem_cam_frame_norm.T).T
        # filter points outside of image
        filter_idx = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < rgb_img.shape[1])
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < rgb_img.shape[0])
        )
        # get semantic annotation
        rgb_pixels = np.zeros((pixels.shape[0], 3))
        rgb_pixels[filter_idx] = rgb_img[
            pixels[filter_idx, 1].astype(int) - 1,
            pixels[filter_idx, 0].astype(int) - 1,
        ]
        rgb_warped = rgb_pixels.reshape(depth_img.shape[0], depth_img.shape[1], 3)
        # overlap ratio
        overlap_ratio = np.sum(filter_idx) / pixels.shape[0]

        # DEBUG
        if self.debug:
            print(
                "depth_rot",
                stf.Rotation.from_matrix(depth_rot).as_euler("xyz", degrees=True),
            )
            rgb_rot = stf.Rotation.from_quat(pose_rgb[3:]).as_matrix() @ ROS_TO_ROBOTICS_MAT @ CAMERA_FLIP_MAT
            print(
                "rgb_rot",
                stf.Rotation.from_matrix(rgb_rot).as_euler("xyz", degrees=True),
            )

            import matplotlib.pyplot as plt

            f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4)
            ax1.imshow(depth_img)
            ax2.imshow(rgb_img)
            ax3.imshow(rgb_warped / 255)
            ax4.imshow(depth_img)
            ax4.imshow(rgb_warped / 255, alpha=0.5)
            plt.savefig(os.path.join(os.getcwd(), "depth_sem_warp.png"))
            # plt.show()
            plt.close()

        # reshape to image
        return rgb_warped, overlap_ratio, depth_zero_ratio

    """PATH PUB, GOAL and ODOM SUB and FEAR DETECTION"""

    def pubPath(self, waypoints, is_goal_init=True):
        # create path
        poses = []
        if is_goal_init:
            for p in waypoints:
                # gte individual pose in depth frame
                pose = PoseStamped()
                pose.pose.position.x = p[0]
                pose.pose.position.y = p[1]
                poses.append(pose)

        # Wait for the transform from base frame to odom frame
        trans = None
        while trans is None:
            try:
                trans = self.tf_buffer.lookup_transform(
                    self.cfg.world_id,
                    self.cfg.robot_id,
                    self.depth_header.stamp,
                    rospy.Duration(1.0),
                )
            except (
                tf2_ros.LookupException,
                tf2_ros.ConnectivityException,
                tf2_ros.ExtrapolationException,
            ):
                rospy.logerr(f"Failed to lookup transform from {self.cfg.robot_id} to" f" {self.cfg.world_id}")
                continue

        # Transform each pose from base to odom frame
        transformed_poses = []
        transformed_poses_np = np.zeros(waypoints.shape)
        for idx, pose in enumerate(poses):
            transformed_pose = tf2_geometry_msgs.do_transform_pose(pose, trans)
            transformed_poses.append(transformed_pose)
            transformed_poses_np[idx] = np.array(
                [
                    transformed_pose.pose.position.x,
                    transformed_pose.pose.position.y,
                    transformed_pose.pose.position.z,
                ]
            )

        success, curr_depth_cam_odom_pose = self.poseCallback(self.depth_header.frame_id, rospy.Time(0))

        # remove all waypoints already passed
        front_poses = np.linalg.norm(transformed_poses_np - transformed_poses_np[0], axis=1) > np.linalg.norm(
            curr_depth_cam_odom_pose[:3] - transformed_poses_np[0]
        )
        poses = [pose for idx, pose in enumerate(poses) if front_poses[idx]]
        transformed_poses = [pose for idx, pose in enumerate(transformed_poses) if front_poses[idx]]

        # add header
        base_path = Path()
        base_fear_path = Path()
        odom_path = Path()

        # assign header
        base_path.header.frame_id = base_fear_path.header.frame_id = self.cfg.robot_id
        odom_path.header.frame_id = self.cfg.world_id
        base_path.header.stamp = base_fear_path.header.stamp = odom_path.header.stamp = self.depth_header.stamp

        # assign poses
        if self.is_fear_reaction:
            base_fear_path.poses = poses
            base_path.poses = poses[:1]
        else:
            base_path.poses = poses
        odom_path.poses = transformed_poses

        # publish path
        self.fear_path_pub.publish(base_fear_path)
        self.path_pub.publish(base_path)
        self.path_viz_pub.publish(odom_path)

        return

    def fearPathDetection(self, fear, is_forward):
        if fear > 0.5 and is_forward:
            if not self.is_fear_reaction:
                self.fear_buffter = self.fear_buffter + 1
        elif self.is_fear_reaction:
            self.fear_buffter = self.fear_buffter - 1
        if self.fear_buffter > self.cfg.buffer_size:
            self.is_fear_reaction = True
        elif self.fear_buffter <= 0:
            self.is_fear_reaction = False
        return

    def isForwardTraking(self, waypoints):
        xhead = np.array([1.0, 0])
        phead = None
        for p in waypoints:
            if np.linalg.norm(p[0:2]) > self.cfg.track_dist:
                phead = p[0:2] / np.linalg.norm(p[0:2])
                break
        if np.all(phead is not None) and phead.dot(xhead) > 1.0 - self.cfg.angular_thread:
            return True
        return False

    def joyCallback(self, joy_msg):
        if joy_msg.buttons[4] > 0.9:
            rospy.loginfo("Switch to Smart Joystick mode ...")
            self.is_smartjoy = True
            # reset fear reaction
            self.fear_buffter = 0
            self.is_fear_reaction = False
        if self.is_smartjoy:
            if np.sqrt(joy_msg.axes[3] ** 2 + joy_msg.axes[4] ** 2) < 1e-3:
                # reset fear reaction
                self.fear_buffter = 0
                self.is_fear_reaction = False
                self.ready_for_planning = False
                self.is_goal_init = False
            else:
                joy_goal = PointStamped()
                joy_goal.header.frame_id = self.cfg.robot_id
                joy_goal.point.x = joy_msg.axes[4] * self.cfg.joyGoal_scale
                joy_goal.point.y = joy_msg.axes[3] * self.cfg.joyGoal_scale
                joy_goal.point.z = 0.0
                joy_goal.header.stamp = rospy.Time.now()
                self.goal_pose = joy_goal
                self.is_goal_init = True
                self.is_goal_processed = False
        return

    def goalCallback(self, msg):
        rospy.loginfo("Received a new goal")
        self.goal_pose = msg
        self.is_smartjoy = False
        self.is_goal_init = True
        self.is_goal_processed = False
        # reset fear reaction
        self.fear_buffter = 0
        self.is_fear_reaction = False
        # reste planner status
        self.planner_status.data = 0
        return

    """RGB IMAGE AND DEPTH CALLBACKS"""

    def poseCallback(self, frame_id: str, img_stamp, target_frame_id: Optional[str] = None) -> Tuple[bool, np.ndarray]:
        target_frame_id = target_frame_id if target_frame_id else self.cfg.world_id
        try:
            if self.cfg.mount_cam_frame is None:
                # Wait for the transform to become available
                transform = self.tf_buffer.lookup_transform(target_frame_id, frame_id, img_stamp, rospy.Duration(4.0))
            else:
                frame_id = self.cfg.mount_cam_frame
                transform = self.tf_buffer.lookup_transform(
                    target_frame_id,
                    self.cfg.mount_cam_frame,
                    img_stamp,
                    rospy.Duration(4.0),
                )
            # Extract the translation and rotation from the transform
            translation = transform.transform.translation
            rotation = transform.transform.rotation
            pose = np.array(
                [
                    translation.x,
                    translation.y,
                    translation.z,
                    rotation.x,
                    rotation.y,
                    rotation.z,
                    rotation.w,
                ]
            )
            return True, pose
        except (
            tf2_ros.LookupException,
            tf2_ros.ConnectivityException,
            tf2_ros.ExtrapolationException,
        ):
            rospy.logerr(f"Pose Fail to transfer {frame_id} into" f" {target_frame_id} frame.")
            return False, np.zeros(7)

    def imageCallback(self, rgb_msg: Image):
        rospy.logdebug("Received rgb image %s: %d" % (rgb_msg.header.frame_id, rgb_msg.header.seq))

        # image pose
        success, pose = self.poseCallback(rgb_msg.header.frame_id, rgb_msg.header.stamp)
        if not success:
            return

        # RGB image
        try:
            image = self.bridge.imgmsg_to_cv2(rgb_msg, "bgr8")
        except cv_bridge.CvBridgeError as e:
            print(e)

        if self.vip_algo.train_cfg.sem:
            image = self.semPrediction(image)

        self.sem_rgb_odom = pose
        self.sem_rgb_img = image
        return

    def imageCallbackCompressed(self, rgb_msg: CompressedImage):
        rospy.logdebug(f"Received rgb   image {rgb_msg.header.frame_id}:" f" {rgb_msg.header.stamp.to_sec()}")

        self.rgb_header = rgb_msg.header

        # image pose
        success, pose = self.poseCallback(rgb_msg.header.frame_id, rgb_msg.header.stamp)
        if not success:
            return

        # RGB Image
        try:
            rgb_arr = np.frombuffer(rgb_msg.data, np.uint8)
            image = cv2.imdecode(rgb_arr, cv2.IMREAD_COLOR)
        except cv_bridge.CvBridgeError as e:
            print(e)

        if self.vip_algo.train_cfg.sem:
            image = self.semPrediction(image)

        self.sem_rgb_img = image
        self.sem_rgb_odom = pose
        self.sem_rgb_new = True
        self.ready_for_planning_rgb_sem = True

        # publish the image
        if self.vip_algo.train_cfg.sem:
            image = cv2.resize(image, (480, 360))
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # Convert the image to JPEG format
            success, compressed_image = cv2.imencode(".jpg", image)
            if not success:
                rospy.logerr("Failed to compress semantic image")
                return

            # create compressed image and publish it
            sem_msg = CompressedImage()
            sem_msg.header = rgb_msg.header
            sem_msg.format = "jpeg"
            sem_msg.data = np.array(compressed_image).tostring()
            self.m2f_pub.publish(sem_msg)
        return

    def semPrediction(self, image):
        # semantic estimation with image in BGR format
        start = time.time()
        image = self.m2f_inference.predict(image)
        self.time_sem = time.time() - start
        # publish prediction time
        self.m2f_timer_data.data = self.time_sem * 1000
        self.m2f_timer_pub.publish(self.m2f_timer_data)
        return image

    def depthCallback(self, depth_msg: Image):
        rospy.logdebug(f"Received depth image {depth_msg.header.frame_id}:" f" {depth_msg.header.stamp.to_sec()}")

        # image time and pose
        self.depth_header = depth_msg.header
        success, self.depth_pose = self.poseCallback(depth_msg.header.frame_id, depth_msg.header.stamp)
        if not success:
            return

        # DEPTH Image
        image = ros_numpy.numpify(depth_msg)
        image[~np.isfinite(image)] = 0
        if self.cfg.depth_uint_type:
            image = image / 1000.0
        image[image > self.cfg.max_depth] = 0.0
        if self.cfg.image_flip:
            image = PIL.Image.fromarray(image)
            self.depth_img = np.array(image.transpose(PIL.Image.Transpose.ROTATE_180))
        else:
            self.depth_img = image

        # transform goal into robot frame and world frame
        if self.is_goal_init:
            if self.goal_pose.header.frame_id != self.cfg.robot_id:
                try:
                    trans = self.tf_buffer.lookup_transform(
                        self.cfg.robot_id,
                        self.goal_pose.header.frame_id,
                        self.depth_header.stamp,
                        rospy.Duration(1.0),
                    )
                    self.goal_robot_frame = tf2_geometry_msgs.do_transform_point(self.goal_pose, trans)
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ):
                    rospy.logerr(
                        "Goal: Fail to transfer" f" {self.goal_pose.header.frame_id} into" f" {self.cfg.robot_id}"
                    )
                    return
            else:
                self.goal_robot_frame = self.goal_pose

            if self.goal_pose.header.frame_id != self.cfg.world_id:
                try:
                    trans = self.tf_buffer.lookup_transform(
                        self.cfg.world_id,
                        self.goal_pose.header.frame_id,
                        self.depth_header.stamp,
                        rospy.Duration(1.0),
                    )
                    self.goal_world_frame = tf2_geometry_msgs.do_transform_point(self.goal_pose, trans)
                except (
                    tf2_ros.LookupException,
                    tf2_ros.ConnectivityException,
                    tf2_ros.ExtrapolationException,
                ):
                    rospy.logerr(
                        "Goal: Fail to transfer" f" {self.goal_pose.header.frame_id} into" f" {self.cfg.world_id}"
                    )
                    return
            else:
                self.goal_world_frame = self.goal_pose

            # get static transform and cam offset
            if self.init_goal_trans:
                if self.cfg.mount_cam_frame is None:
                    # get transform from robot frame to depth camera frame
                    success, tf_robot_depth = self.poseCallback(
                        self.depth_header.frame_id,
                        self.depth_header.stamp,
                        self.cfg.robot_id,
                    )
                else:
                    success, tf_robot_depth = self.poseCallback(
                        self.cfg.mount_cam_frame,
                        self.depth_header.stamp,
                        self.cfg.robot_id,
                    )

                if not success:
                    return
                self.cam_offset = tf_robot_depth[0:3]
                self.cam_rot = stf.Rotation.from_quat(tf_robot_depth[3:7]).as_matrix() @ ROS_TO_ROBOTICS_MAT
                if not self.cfg.image_flip:
                    # rotation is included in ROS_TO_ROBOTICS_MAT and has to
                    # be removed when not fliped
                    self.cam_rot = self.cam_rot @ CAMERA_FLIP_MAT
                if self.debug:
                    print(
                        "CAM ROT",
                        stf.Rotation.from_matrix(self.cam_rot).as_euler("xyz", degrees=True),
                    )
                self.init_goal_trans = False

            self.goal_cam_frame_set = True

        # declare ready for planning
        self.ready_for_planning_depth = True
        self.is_goal_processed = True
        return

    """ Camera Info Callbacks"""

    def depthCamInfoCallback(self, cam_info_msg: CameraInfo):
        if not self.depth_intrinsics_init:
            rospy.loginfo("Received depth camera info")
            self.K_depth = cam_info_msg.K
            self.K_depth = np.array(self.K_depth).reshape(3, 3)
            self.depth_intrinsics_init = True
        return

    def rgbCamInfoCallback(self, cam_info_msg: CameraInfo):
        if not self.rgb_intrinsics_init:
            rospy.loginfo("Received rgb camera info")
            self.K_rgb = cam_info_msg.K
            self.K_rgb = np.array(self.K_rgb).reshape(3, 3)
            self.rgb_intrinsics_init = True
        return


if __name__ == "__main__":
    node_name = "viplanner_node"
    rospy.init_node(node_name, anonymous=False)

    parser = ROSArgparse(relative=node_name)
    # planning
    parser.add_argument("main_freq", type=int, default=5, help="frequency of path planner")
    parser.add_argument("image_flip", type=bool, default=True, help="is the image fliped")
    parser.add_argument("conv_dist", type=float, default=0.5, help="converge range to the goal")
    parser.add_argument(
        "max_depth",
        type=float,
        default=10.0,
        help="max depth distance in image",
    )
    parser.add_argument(
        "overlap_ratio_thres",
        type=float,
        default=0.7,
        help="overlap threshold betweens sem/rgb and depth image",
    )
    parser.add_argument(
        "depth_zero_ratio_thres",
        type=float,
        default=0.7,
        help="ratio of depth image that is non-zero",
    )
    # networks
    parser.add_argument(
        "model_save",
        type=str,
        default="models/vip_models/plannernet_env2azQ1b91cZZ_ep100_inputDepSem_costSem_optimSGD",
        help=("model directory (within should be a file called model.pt and" " model.yaml)"),
    )
    parser.add_argument(
        "m2f_cfg_file",
        type=str,
        default=("models/coco_panoptic/swin/maskformer2_swin_tiny_bs16_50ep.yaml"),
        help=("config file for m2f model (or pre-trained backbone for direct RGB" " input)"),
    )
    parser.add_argument(
        "m2f_model_path",
        type=str,
        default="models/coco_panoptic/swin/model_final_9fd0ae.pkl",
        help=("read model for m2f model (or pre-trained backbone for direct RGB" " input)"),
    )
    # ROS topics
    parser.add_argument(
        "depth_topic",
        type=str,
        default="/rgbd_camera/depth/image",
        help="depth image ros topic",
    )
    parser.add_argument(
        "depth_info_topic",
        type=str,
        default="/depth_camera_front_upper/depth/camera_info",
        help="depth image info topic (get intrinsic matrix)",
    )
    parser.add_argument(
        "rgb_topic",
        type=str,
        default="/wide_angle_camera_front/image_raw/compressed",
        help="rgb camera topic",
    )
    parser.add_argument(
        "rgb_info_topic",
        type=str,
        default="/wide_angle_camera_front/camera_info",
        help="rgb camera info topic (get intrinsic matrix)",
    )
    parser.add_argument(
        "goal_topic",
        type=str,
        default="/mp_waypoint",
        help="goal waypoint ros topic",
    )
    parser.add_argument(
        "path_topic",
        type=str,
        default="/viplanner/path",
        help="VIP Path topic",
    )
    parser.add_argument(
        "m2f_timer_topic",
        type=str,
        default="/viplanner/m2f_timer",
        help="Time needed for semantic segmentation",
    )
    parser.add_argument(
        "depth_uint_type",
        type=bool,
        default=False,
        help="image in uint type or not",
    )
    parser.add_argument(
        "compressed",
        type=bool,
        default=True,
        help="If compressed rgb topic is used",
    )
    parser.add_argument(
        "mount_cam_frame",
        type=str,
        default=None,
        help="When cam is mounted, which frame to take for pose compute",
    )

    # frame_ids
    parser.add_argument("robot_id", type=str, default="base", help="robot TF frame id")
    parser.add_argument("world_id", type=str, default="odom", help="world TF frame id")

    # fear reaction
    parser.add_argument(
        "is_fear_act",
        type=bool,
        default=True,
        help="is open fear action or not",
    )
    parser.add_argument(
        "buffer_size",
        type=int,
        default=10,
        help="buffer size for fear reaction",
    )
    parser.add_argument(
        "angular_thread",
        type=float,
        default=0.3,
        help="angular thread for turning",
    )
    parser.add_argument(
        "track_dist",
        type=float,
        default=0.5,
        help="look ahead distance for path tracking",
    )
    # smart joystick
    parser.add_argument(
        "joyGoal_scale",
        type=float,
        default=0.5,
        help="distance for joystick goal",
    )

    args = parser.parse_args()

    # model save path
    args.model_save = os.path.join(pack_path, args.model_save)
    args.m2f_config_path = os.path.join(pack_path, args.m2f_cfg_file)
    args.m2f_model_path = os.path.join(pack_path, args.m2f_model_path)

    node = VIPlannerNode(args)

    node.spin()
