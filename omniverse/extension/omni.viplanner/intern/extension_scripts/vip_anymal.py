# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import time
from typing import Dict, Optional

# omni
import carb

# python
import numpy as np

# omni-isaac-core
import omni.isaac.core.utils.prims as prim_utils
import open3d as o3d

# ROS
import rospy
import scipy.spatial.transform as tf
import torch
import torchvision.transforms as transforms
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path

# omni-isaac-anymal
from omni.isaac.anymal.config import ANYmalCfg, VIPlannerCfg
from omni.isaac.anymal.utils import get_cam_pose
from omni.isaac.core.simulation_context import SimulationContext
from omni.isaac.debug_draw import _debug_draw
from omni.isaac.orbit.robots.legged_robot import LeggedRobot

# omni-isaac-orbit
from omni.isaac.orbit.sensors.camera import Camera
from PIL import Image
from std_msgs.msg import Int16

# viplanner
from viplanner.config import TrainCfg, VIPlannerSemMetaHandler

from .vip_algo import VIPlannerAlgo


class VIPlanner:
    """
    Visual Imperative Planner for Anymal
    """

    debug: bool = False

    def __init__(
        self,
        anymal_cfg: ANYmalCfg,
        vip_cfg: VIPlannerCfg,
        camera_sensors: Dict[str, Camera],
    ) -> None:
        self._cfg_anymal: ANYmalCfg = anymal_cfg
        self._cfg_vip: VIPlannerCfg = vip_cfg
        self._camera_sensors = camera_sensors

        # Simulation context
        self.sim: SimulationContext = SimulationContext.instance()

        # ANYmal model and camera paths
        if self._cfg_vip.use_mount_cam:
            self.cam_path: dict = self._cfg_vip.cam_path["mount"]
        elif self._cfg_anymal.anymal_type == 0:  # ANYmal C
            self.cam_path: dict = self._cfg_vip.cam_path["ANYmal_C"]
        elif self._cfg_anymal.anymal_type == 1:  # ANYmal D
            self.cam_path: dict = self._cfg_vip.cam_path["ANYmal_D"]
        else:
            raise ValueError(
                f"ANYmal type {self._cfg_anymal.anymal_type} not supported!\n"
                "Either select '0' for ANYmal_C and '1' for ANYmal_D"
            )
        if "rgb" in self.cam_path and "depth" in self.cam_path:
            self.same_cam: bool = False if self.cam_path["rgb"] != self.cam_path["depth"] else True

        # planner status
        if self._cfg_vip.ros_pub:
            self.planner_status = Int16()
            self.planner_status.data = 0

        # additional variables
        self.fear: float = 0.0
        self.traj_waypoints_np: np.ndarray = np.zeros(0)
        self.traj_waypoints_odom: np.ndarray = np.zeros(0)
        self._step = 0  # number of times the waypoint have been generated, used together with the frequency
        self.distance_to_goal: float = 0.0
        self.max_goal_distance: float = 0.0 + 1.0e-9  # to avoid division by zero
        self.start_time: float = 0.0

        ##
        # SETUP
        ##

        # check for cuda
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        # setup planner
        self.planner = VIPlannerAlgo(self._cfg_vip.model_dir, self._cfg_vip.m2f_model_dir, self._cfg_vip.viplanner)

        # check camera sensor
        self._check_camera()

        # setup goal transforms
        self.goal_pos = prim_utils.get_prim_at_path(self._cfg_vip.goal_prim).GetAttribute("xformOp:translate")
        self.goal_pos_prev = np.zeros(3)  # previous goal position to check if goal has changed

        # get field of view
        self.alpha_fov: float = 0.0
        self.get_fov()

        # setup pixel array for warping of the semantic image (only if semantics activated)
        self.pix_depth_cam_frame: np.ndarray = np.zeros(
            (
                self._camera_sensors[self.cam_path["depth"]].data.image_shape[0]
                * self._camera_sensors[self.cam_path["depth"]].data.image_shape[1],
                3,
            )
        )
        if self.planner.train_config.sem or self.planner.train_config.rgb:
            self._compute_pixel_tensor()

        # get transforms for images
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(self.planner.train_config.img_input_size),
            ]
        )

        # setup waypoint display in Isaac
        self.draw = _debug_draw.acquire_debug_draw_interface()
        self.point_list = [(1, 0, 0.5)] * self._cfg_vip.num_points_network_return
        self.color = [(0.4, 1.0, 0.1, 1.0)]  # green
        self.color_fear = [(1.0, 0.4, 0.1, 1.0)]  # red
        self.color_path = [(1.0, 0.5, 0.0, 1.0)]  # orange
        self.size = [5.0]

        # setup semantic meta data if carla is used
        if self._cfg_vip.sem_origin == "isaac":
            self.viplanner_sem_meta = VIPlannerSemMetaHandler()

        # setup ROS
        if self._cfg_vip.ros_pub:
            self.path_pub = rospy.Publisher(self._cfg_vip.path_topic, Path, queue_size=10)
            self.fear_path_pub = rospy.Publisher(self._cfg_vip.path_topic + "_fear", Path, queue_size=10)
            self.status_pub = rospy.Publisher(self._cfg_vip.status_topic, Int16, queue_size=10)

        # save image
        if self._cfg_vip.save_images:
            # Create annotator output directory
            file_path = os.path.join(os.getcwd(), "_out_annot", "")
            self.dir = os.path.dirname(file_path)
            os.makedirs(self.dir, exist_ok=True)

        # time
        self.time_measurement: bool = False
        self.time_collect: float = 0.0
        self.time_save: float = 0.0

        # flags
        self.goal_outside_fov: bool = False
        return

    ##
    # Public Functions
    ##

    def set_planner_callback(self, val: bool = True) -> None:
        ##
        # Setup callbacks
        ##
        if val:
            self.sim.add_physics_callback("vip_callback", callback_fn=self._planner_callback)
        else:
            self.sim.remove_physics_callback("vip_callback")
        return

    def switch_model(self, model_dir: str, m2f_model_dir: Optional[str] = None) -> None:
        if m2f_model_dir is None and self._cfg_vip.m2f_model_dir is not None:
            m2f_model_dir = self._cfg_vip.m2f_model_dir
        # delete previous model from GPU
        if self.planner.cuda_avail:
            del self.planner.net
        # load new model
        self.planner.load_model(model_dir, m2f_model_dir)
        return

    ##
    # Internal Functions
    ##

    def _check_camera(self) -> None:
        assert self._camera_sensors[self.cam_path["depth"]]._is_spawned, "Front Depth Camera not spawned!"

        assert (
            "distance_to_image_plane" in self._camera_sensors[self.cam_path["depth"]].cfg.data_types
        ), "Missing data_type 'distance_to_image_plane' for front depth camera"

        if self.planner.train_config.sem or self.planner.train_config.rgb:
            assert self._camera_sensors[self.cam_path["rgb"]]._is_spawned, "Front RGB Camera not spawned!"
            assert (
                "semantic_segmentation" in self._camera_sensors[self.cam_path["rgb"]].cfg.data_types
            ), "Missing data_type 'semantic_segmentation' for front camera"
        if self._cfg_vip.rgb_debug:
            assert (
                "rgb" in self._camera_sensors[self.cam_path["rgb"]].cfg.data_types
            ), "Missing data_type 'rgb' for front RGB camera"
        return

    def _planner_callback(self, dt) -> None:
        # only plan with given frequency
        if self._step % self._cfg_vip.planner_freq == 0:
            # reset step counter
            self._step = 0
            # compute
            self._camera_sensors[self.cam_path["depth"]].update(dt)
            if not self.same_cam and self.planner.train_config.sem:
                if self._cfg_vip.sem_origin == "isaac":
                    # run complete update if carla
                    self._camera_sensors[self.cam_path["rgb"]].update(dt)
                else:
                    # for matterport data will be written in camera by matterport callback, only update pose
                    (
                        self._camera_sensors[self.cam_path["rgb"]].data.position,
                        self._camera_sensors[self.cam_path["rgb"]].data.orientation,
                    ) = self._camera_sensors[self.cam_path["rgb"]]._compute_ros_pose()
            elif not self.same_cam and self.planner.train_config.rgb:
                self._camera_sensors[self.cam_path["rgb"]].update(dt)
            self._compute()

        # increment step counter
        self._step += 1
        return

    def _compute(self) -> None:
        # get goal pos
        goal = np.asarray(self.goal_pos.Get())

        cam_pos, cam_rot_quat = get_cam_pose(self._camera_sensors[self.cam_path["depth"]]._sensor_prim)
        cam_rot = tf.Rotation.from_quat(cam_rot_quat).as_matrix()

        # check if goal already reached --> exit here
        self.distance_to_goal = np.sqrt((goal[0] - cam_pos[0]) ** 2 + (goal[1] - cam_pos[1]) ** 2)
        if self.distance_to_goal < self._cfg_vip.conv_dist:
            carb.log_info("GOAL REACHED!")
            # planner status -> Success
            if self._cfg_vip.ros_pub and self.planner_status.data == 0:
                self.planner_status.data = 1
                self.status_pub.publish(self.planner_status)
            return
        elif self._cfg_vip.ros_pub:
            self.planner_status.data = 0
            self.status_pub.publish(self.planner_status)
        carb.log_verbose(f"DISTANCE TO GOAL: {self.distance_to_goal}")

        # if goal is too far away --> project on max_goal_distance circle around robot
        if self.distance_to_goal > self.planner.max_goal_distance:
            goal[:2] = cam_pos[:2] + (goal[:2] - cam_pos[:2]) / self.distance_to_goal * self.planner.max_goal_distance

        # apply rotation to goal  --> transform goal into camera frame
        goal_cam_frame = goal - cam_pos
        goal_cam_frame[2] = 0  # trained with z difference of 0
        goal_cam_frame = goal_cam_frame @ cam_rot
        goal_cam_frame = torch.tensor(goal_cam_frame, dtype=torch.float32, device=self.device).unsqueeze(0)

        # check if goal pos has changed
        if not np.all(goal == self.goal_pos_prev):
            self.goal_pos_prev = goal
            self.max_goal_distance = self.distance_to_goal
            self.start_time = self.sim.current_time
            self.is_reset = False

            # check if goal is in fov
            if abs(torch.atan2(goal_cam_frame[0, 1], goal_cam_frame[0, 0])) > self.alpha_fov / 2:
                self.goal_outside_fov = True
            else:
                self.goal_outside_fov = False

            carb.log_info(
                f"New goal position: {goal} received in distance {self.distance_to_goal} (out FOV: {self.goal_outside_fov})"
            )
            print(
                f"[VIPlanner INFO] New goal position: {goal} received in distance {self.distance_to_goal} (out FOV: {self.goal_outside_fov})"
            )

        start = time.time()
        # Collect Groundtruth
        depth_image = self._camera_sensors[self.cam_path["depth"]].data.output["distance_to_image_plane"]
        depth_image[~np.isfinite(depth_image)] = 0  # set all inf or nan values to 0
        depth_image[depth_image > self.planner.max_depth] = 0.0
        depth_image_torch = self.transform(depth_image)  # declare as new variable since reused in semantic warp
        depth_image_torch = depth_image_torch.unsqueeze(0).to(self.device)

        # time for collecting data
        self.time_collect = time.time() - start

        if self.planner.train_config.sem:
            # check if semantics available
            if self._cfg_vip.sem_origin not in ["isaac", "callback"]:
                carb.log_error(
                    f"Unknown data source '{self._cfg_vip.sem_origin}'! Select either 'isaac' or 'callback'!"
                )
                return

            if self._camera_sensors[self.cam_path["rgb"]].data.output["semantic_segmentation"] is None:
                carb.log_warn("No semantic segmentation data available! No waypoint generated in this step!")
                return
            elif isinstance(self._camera_sensors[self.cam_path["rgb"]].data.output["semantic_segmentation"], dict) and [
                label_class_dict["class"]
                for label_class_dict in self._camera_sensors[self.cam_path["rgb"]]
                .data.output["semantic_segmentation"]["info"]["idToLabels"]
                .values()
            ] == ["BACKGROUND", "UNLABELLED"]:
                carb.log_warn(
                    "Semantic data only of type BACKGROUND and UNLABELLED! No waypoint generated in this step!"
                )
                return

            # handling for carla using orbit camera class to generate the data
            sem_image: np.ndarray = np.zeros(
                (
                    self._camera_sensors[self.cam_path["rgb"]].data.image_shape[1],
                    self._camera_sensors[self.cam_path["rgb"]].data.image_shape[0],
                )
            )
            if self._cfg_vip.sem_origin == "isaac":
                # collect image
                sem_image = self._camera_sensors[self.cam_path["rgb"]].data.output["semantic_segmentation"]["data"]
                sem_idToLabels = self._camera_sensors[self.cam_path["rgb"]].data.output["semantic_segmentation"][
                    "info"
                ]["idToLabels"]
                sem_image = self.sem_color_transfer(sem_image, sem_idToLabels)
            else:
                sem_image = self._camera_sensors[self.cam_path["rgb"]].data.output["semantic_segmentation"]

            # overlay semantic image on depth image
            sem_image = self._get_overlay_semantics(sem_image, depth_image, depth_rot=cam_rot)
            # move to tensor
            sem_image = self.transform(sem_image.astype(np.uint8))
            sem_image = sem_image.unsqueeze(0).to(self.device)
            # update time
            self.time_collect = time.time() - start

            # run network
            _, traj_waypoints, self.fear = self.planner.plan_dual(depth_image_torch, sem_image, goal_cam_frame)
        elif self.planner.train_config.rgb:
            if self._camera_sensors[self.cam_path["rgb"]].data.output["rgb"] is None:
                carb.log_warn("No rgb data available! No waypoint generated in this step!")
                return
            rgb_image = self._camera_sensors[self.cam_path["rgb"]].data.output["rgb"]

            # overlay semantic image on depth image
            rgb_image = self._get_overlay_semantics(rgb_image, depth_image, depth_rot=cam_rot)

            # apply mean and std normalization
            rgb_image = (rgb_image - self.planner.pixel_mean) / self.planner.pixel_std

            # move to tensor
            rgb_image = self.transform(rgb_image)
            rgb_image = rgb_image.unsqueeze(0).to(self.device)
            # update time
            self.time_collect = time.time() - start

            # run network
            _, traj_waypoints, self.fear = self.planner.plan_dual(depth_image_torch, rgb_image, goal_cam_frame)
        else:
            # run network
            _, traj_waypoints, self.fear = self.planner.plan(depth_image_torch, goal_cam_frame)

        self.traj_waypoints_np = traj_waypoints.cpu().detach().numpy().squeeze(0)
        self.traj_waypoints_np = self.traj_waypoints_np[1:, :]  # first twist command is zero --> remove it

        # plot trajectory
        self.traj_waypoints_odom = self.traj_waypoints_np @ cam_rot.T + cam_pos  # get waypoints in world frame
        self.draw.clear_lines()
        if self.fear > self._cfg_vip.fear_threshold:
            self.draw.draw_lines(
                self.traj_waypoints_odom.tolist()[:-1],
                self.traj_waypoints_odom.tolist()[1:],
                self.color_fear * len(self.traj_waypoints_odom.tolist()[1:]),
                self.size * len(self.traj_waypoints_odom.tolist()[1:]),
            )
            self.draw.draw_lines(
                [cam_pos.tolist()],
                [goal.tolist()],
                self.color_fear,
                [2.5],
            )
        else:
            self.draw.draw_lines(
                self.traj_waypoints_odom.tolist()[:-1],
                self.traj_waypoints_odom.tolist()[1:],
                self.color * len(self.traj_waypoints_odom.tolist()[1:]),
                self.size * len(self.traj_waypoints_odom.tolist()[1:]),
            )
            self.draw.draw_lines(
                [cam_pos.tolist()],
                [goal.tolist()],
                self.color_path,
                [2.5],
            )

        if self._cfg_vip.ros_pub:
            self._pub_path(waypoints=self.traj_waypoints_np)
        else:
            carb.log_info(f"New waypoints generated! \n {self.traj_waypoints_np}")

        if self._cfg_vip.save_images:
            start = time.time()
            self._save_depth(depth_image, self.dir + "/depth_front_step_" + str(self._step))
            self._save_rgb() if self._cfg_vip.rgb_debug else None
            self.time_save = time.time() - start

        if self.time_measurement:
            print(f"Time collect: {self.time_collect} \t Time save: {self.time_save}")
        return

    def reset(self) -> None:
        """Reset the planner variables."""
        self.fear: float = 0.0
        self.traj_waypoints_np: np.ndarray = np.zeros(0)
        self.traj_waypoints_odom: np.ndarray = np.zeros(0)
        self._step = 0
        self.goal_outside_fov: bool = False
        self.goal_pos_prev: np.ndarray = np.zeros(3)
        self.distance_to_goal: float = 0.0
        self.max_goal_distance: float = 0.0 + 1.0e-9
        self.start_time: float = 0.0
        self.is_reset: bool = True
        return

    def _pub_path(self, waypoints: torch.Tensor) -> None:
        path = Path()
        fear_path = Path()
        curr_time = rospy.Time.from_sec(self.sim.current_time)
        for p in waypoints:
            pose = PoseStamped()
            pose.header.stamp = curr_time
            pose.header.frame_id = "odom"
            pose.pose.position.x = p[0]
            pose.pose.position.y = p[1]
            pose.pose.position.z = p[2]
            path.poses.append(pose)
        # add header
        path.header.frame_id = fear_path.header.frame_id = "odom"
        path.header.stamp = fear_path.header.stamp = curr_time
        # publish fear path
        # if self.is_fear_reaction:
        #     fear_path.poses = copy.deepcopy(path.poses)
        #     path.poses = path.poses[:1]
        # publish path
        # self.fear_path_pub.publish(fear_path)
        self.path_pub.publish(path)
        return

    def get_fov(self) -> None:
        # load intrinsics --> used to calculate fov
        intrinsics = self._camera_sensors[self.cam_path["depth"]].data.intrinsic_matrix
        self.alpha_fov = 2 * np.arctan(intrinsics[0, 0] / intrinsics[0, 2])
        return

    """ Helper to warp semantic image to depth image """

    def _get_overlay_semantics(self, sem_img: np.ndarray, depth_img: np.ndarray, depth_rot: np.ndarray) -> np.ndarray:
        # get semantic rotation matrix
        sem_pos, sem_rot_quat = get_cam_pose(self._camera_sensors[self.cam_path["rgb"]]._sensor_prim)
        sem_rot = tf.Rotation.from_quat(sem_rot_quat).as_matrix()
        sem_rot = sem_rot.astype(np.float64)
        depth_rot = depth_rot.astype(np.float64)

        # project depth pixels into 3d space
        # dep_im_reshaped = depth_img.reshape(-1, 1)
        dep_im_reshaped = depth_img.reshape(-1, 1)
        points = (
            dep_im_reshaped * (depth_rot @ self.pix_depth_cam_frame.T).T
            + self._camera_sensors[self.cam_path["depth"]].data.position
        )

        # transform points to semantic camera frame
        points_sem_cam_frame = (sem_rot.T @ (points - sem_pos).T).T
        # normalize points
        points_sem_cam_frame_norm = points_sem_cam_frame / points_sem_cam_frame[:, 0][:, np.newaxis]
        # reorder points be camera convention (z-forward)
        points_sem_cam_frame_norm = points_sem_cam_frame_norm[:, [1, 2, 0]] * np.array([-1, -1, 1])
        # transform points to pixel coordinates
        pixels = (self._camera_sensors[self.cam_path["rgb"]].data.intrinsic_matrix @ points_sem_cam_frame_norm.T).T
        # filter points outside of image
        filter_idx = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < sem_img.shape[1])
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < sem_img.shape[0])
        )
        # get semantic annotation
        sem_annotation = np.zeros((pixels.shape[0], 3), dtype=np.uint16)
        sem_annotation[filter_idx] = sem_img[pixels[filter_idx, 1].astype(int), pixels[filter_idx, 0].astype(int)]
        # reshape to image
        sem_img_warped = sem_annotation.reshape(depth_img.shape[0], depth_img.shape[1], 3)

        # DEBUG
        if self.debug:
            import matplotlib.pyplot as plt

            f, (ax1, ax2, ax3) = plt.subplots(1, 3)
            ax1.imshow(depth_img)
            ax2.imshow(sem_img_warped / 255)
            ax3.imshow(depth_img)
            ax3.imshow(sem_img_warped / 255, alpha=0.5)
            plt.show()

            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            o3d.visualization.draw_geometries([pcd])

        return sem_img_warped

    """Semantic Image Color Transfer"""

    def sem_color_transfer(self, sem_image: np.ndarray, sem_idToLabels: dict) -> np.ndarray:
        """Convert semantic segmentation image to viplanner color space

        Args:
            sem_image (np.ndarray): sem_image as received by the simulation
            sem_idToLabels (dict): information about which class is which index in sem_image

        Returns:
            np.ndarray: sem_image in viplanner color space
        """
        if not sem_idToLabels:
            carb.log_warn("No semantic segmentation data available! No waypoint generated in this step!")
            return

        for k, v in sem_idToLabels.items():
            if not dict(v):
                sem_idToLabels[k] = {"class": "static"}
            elif "BACKGROUND" == v["class"]:
                sem_idToLabels[k] = {"class": "static"}
            elif "UNLABELLED" == v["class"]:
                sem_idToLabels[k] = {"class": "static"}

        # color mapping
        sem_idToColor = np.array(
            [
                [
                    int(k),
                    self.viplanner_sem_meta.class_color[v["class"]][0],
                    self.viplanner_sem_meta.class_color[v["class"]][1],
                    self.viplanner_sem_meta.class_color[v["class"]][2],
                ]
                for k, v in sem_idToLabels.items()
            ]
        )

        # order colors by their id and necessary to account for missing indices (not guaranteed to be consecutive)
        sem_idToColorMap = np.zeros((max(sem_idToColor[:, 0]) + 1, 3), dtype=np.uint8)
        for cls_color in sem_idToColor:
            sem_idToColorMap[cls_color[0]] = cls_color[1:]
        # colorize semantic image
        try:
            sem_image = sem_idToColorMap[sem_image.reshape(-1)].reshape(sem_image.shape + (3,))
        except IndexError:
            print("IndexError: Semantic image contains unknown labels")
            return

        return sem_image


# EoF
