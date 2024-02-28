# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import random
import time
from typing import Tuple

# omniverse
import carb
import cv2

# python
import numpy as np
import omni.isaac.debug_draw._debug_draw as omni_debug_draw
import scipy.spatial.transform as tf
import yaml

# isaac-carla
from omni.isaac.carla.configs import CarlaExplorerConfig, CarlaLoaderConfig
from omni.isaac.core.objects import VisualCuboid

# isaac-core
from omni.isaac.core.simulation_context import SimulationContext

# isaac-orbit
from omni.isaac.orbit.sensors.camera import Camera
from omni.isaac.orbit.utils.math import convert_quat
from omni.physx import get_physx_scene_query_interface
from pxr import Gf, Usd, UsdGeom
from scipy.spatial import KDTree
from scipy.stats import qmc

# isaac-anymal
from viplanner.config.viplanner_sem_meta import VIPlannerSemMetaHandler

from .loader import CarlaLoader


class CarlaExplorer:
    debug: bool = False

    def __init__(self, cfg: CarlaExplorerConfig, cfg_load: CarlaLoaderConfig) -> None:
        self._cfg = cfg
        self._cfg_load = cfg_load

        # check simulation context
        if SimulationContext.instance():
            self.sim: SimulationContext = SimulationContext.instance()
        else:
            carb.log_error("CarlaExplorer can only be loaded in a running simulationcontext!\nRun CarlaLoader!")

        # Acquire draw interface
        self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()

        # VIPlanner Semantic Meta Handler and mesh to sem class mapping
        if self._cfg_load.sem_mesh_to_class_map is not None:
            self.vip_sem_meta: VIPlannerSemMetaHandler = VIPlannerSemMetaHandler()
            with open(self._cfg_load.sem_mesh_to_class_map) as f:
                self.class_keywords = yaml.safe_load(f)

        # init buffers
        self.camera_positions: np.ndarray = np.array([])
        self.cam_angles: np.ndarray = np.array([])
        self.nbr_points: int = 0

        # get camera
        self.camera_depth: Camera = None
        self.camera_semantic: Camera = None

        return

    def explore(self) -> None:
        # init camera
        self._camera_init()

        # define camera positions and targets
        self._get_cam_position()
        self._get_cam_target()

        # record rgb, depth, semantic segmentation at the camera posiitions
        self._domain_recorder()

        return

    """ Exploration Helper Functions """

    def _raycast_check(self, ray_origins: np.ndarray, ray_directions: np.ndarray, max_distance: float):
        """
        Check which object is hit by the raycast and give back the position, loss and class name of the hit object
        """

        start = time.time()
        hits = [
            get_physx_scene_query_interface().raycast_closest(
                carb.Float3(ray_single), carb.Float3(ray_dir), max_distance
            )
            for ray_single, ray_dir in zip(ray_origins, ray_directions)
        ]
        end = time.time()
        print("took ", end - start, "s for raycast the possible camera points")

        # if point achieved a hit, get the hit point and the hit object
        hit_pt_obj = [
            (np.array(single_hit["position"]), single_hit["collision"].lower())
            for single_hit in hits
            if single_hit["hit"]
        ]
        hit_idx = [idx for idx, single_hit in enumerate(hits) if single_hit["hit"]]

        # get offset
        offset = np.array([0.0, 0.0, self._cfg.robot_height])

        # get semantic class for each points and the corresponding cost
        hit_class_name = np.zeros(len(hit_pt_obj), dtype=str)
        hit_loss = np.zeros(len(hit_pt_obj))
        hit_position = np.zeros((len(hit_pt_obj), 3))

        if self._cfg_load.sem_mesh_to_class_map is not None:
            for idx, single_hit in enumerate(hit_pt_obj):
                success = False
                for class_name, keywords in self.class_keywords.items():
                    if any([keyword.lower() in single_hit[1] for keyword in keywords]):
                        hit_class_name[idx] = class_name
                        hit_loss[idx] = self.vip_sem_meta.class_loss[class_name]
                        hit_position[idx] = single_hit[0] + offset  # add offset to get the center of the point
                        success = True
                        break
                assert success, f"No class found for hit object: {single_hit}"
        else:
            hit_position = np.array([single_hit[0] + offset for single_hit in hit_pt_obj])

        return hit_position, hit_loss, hit_class_name, hit_idx

    def _get_cam_position(self) -> None:
        """
        Get suitable robot positions for exploration of the map. Robot positions are are dense cover of the map


        Args:
            points_per_m2 (float, optional): points per m^2. Defaults to 0.1.
            obs_loss_threshold (float, optional): loss threshold for point to be suitable as robot position. Defaults to 0.6.   # choose s.t. no points on the terrain  TODO: change at some point
            debug (bool, optional): debug mode. Defaults to True.
        """
        # get x-y-z coordinates limits where the explortion of all the mesh should take place
        # for Carla, Town01_Opt is the explored map equal to the city surrounded by the road
        # --> get min und max over the maximum extent of the Road_Sidewalk meshes
        # IMPORTANT: y-up!!!
        mesh_prims, mesh_prims_name = CarlaLoader.get_mesh_prims(self._cfg_load.prim_path + self._cfg_load.suffix)

        if self._cfg.space_limiter:
            # if space limiter is given, only consider the meshes with the space limiter in the name
            mesh_idx = [
                idx
                for idx, prim_name in enumerate(mesh_prims_name)
                if self._cfg.space_limiter.lower() in prim_name.lower()
            ]
        else:
            # remove ground plane since has infinite extent
            mesh_idx = [idx for idx, prim_name in enumerate(mesh_prims_name) if "groundplane" not in prim_name.lower()]
        mesh_prims = [mesh_prims[idx] for idx in mesh_idx]

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        bbox = [self.compute_bbox_with_cache(bbox_cache, curr_prim) for curr_prim in mesh_prims]
        prim_max = np.vstack([list(prim_range.GetMax()) for prim_range in bbox])
        prim_min = np.vstack([list(prim_range.GetMin()) for prim_range in bbox])
        x_min, y_min, z_min = np.min(prim_min, axis=0)
        x_max, y_max, z_max = np.max(prim_max, axis=0)

        max_area = (x_max - x_min) * (y_max - y_min)
        max_distance = (z_max - z_min) + 10  # 10m extra
        print("Exploration area: ", round(max_area / (1000) ** 2, 3), "km^2 or ", max_area, "m^2")

        # init sampler as qmc
        sampler = qmc.Halton(d=2, scramble=False)
        # determine number of samples to dram
        nbr_points = int(max_area * self._cfg.points_per_m2)
        # get raw samples origins
        points = sampler.random(nbr_points)
        if self._cfg.nomoko_model:
            points = qmc.scale(points, [y_min, x_min], [y_max, x_max])
        else:
            points = qmc.scale(points, [x_min, y_min], [x_max, y_max])

        if self._cfg.indoor_filter:
            heights = np.ones((nbr_points, 1)) * (z_max + 2 * self._cfg.robot_height)  # above the map highest point
        else:
            heights = np.ones((nbr_points, 1)) * (z_min + 2 * self._cfg.robot_height)  # above the map lowest point
        ray_origins = np.hstack((points, heights))

        # get ray directions in negative z direction
        ray_directions = np.zeros((nbr_points, 3))
        ray_directions[:, 2] = -1.0

        # perform raycast check
        hit_position, hit_loss, _, _ = self._raycast_check(ray_origins, ray_directions, max_distance)

        # filter all indexes which are not in traversable terrain
        camera_positions = hit_position[hit_loss < self._cfg.obs_loss_threshold]

        # indoor filter
        if self._cfg.indoor_filter:
            # check on all 4 sites can only be performed with semantics
            if self._cfg_load.sem_mesh_to_class_map is not None:
                # filter all points within buildings by checking if hit above the point and if yes, if hit on all 4 sites of it
                # rays always send from both sides since mesh only one-sided
                # check if hit above the point
                camera_positions_elevated = camera_positions + np.array([0.0, 0.0, 100])
                _, hit_loss_low, _, hit_idx_low = self._raycast_check(
                    camera_positions_elevated, ray_directions, max_distance=200
                )
                ray_directions[:, 2] = 1.0
                _, hit_loss_high, _, hit_idx_high = self._raycast_check(
                    camera_positions, ray_directions, max_distance=200
                )

                hit_idx_low = np.array(hit_idx_low)[hit_loss_low >= self._cfg.obs_loss_threshold]
                hit_idx_high = np.array(hit_idx_high)[hit_loss_high >= self._cfg.obs_loss_threshold]
                hit_idx = np.unique(np.hstack([hit_idx_low, hit_idx_high]))

                if len(hit_idx) > 0:
                    # check hit on sites of the point
                    ray_directions[:, 2] = 0.0  # reset ray direction

                    ray_directions[:, 0] = 1.0
                    _, hit_loss, _, hit_idx_front = self._raycast_check(
                        camera_positions[hit_idx], ray_directions[hit_idx], max_distance=10
                    )
                    traversable_front_pos = np.ones((len(hit_idx), 1), dtype=bool)
                    traversable_front_pos[hit_idx_front, 0] = hit_loss < self._cfg.obs_loss_threshold
                    ray_directions[:, 0] = -1.0
                    _, hit_loss, _, hit_idx_front = self._raycast_check(
                        camera_positions[hit_idx] + np.array([10, 0.0, 0.0]), ray_directions[hit_idx], max_distance=10
                    )
                    traversable_front_neg = np.ones((len(hit_idx), 1), dtype=bool)
                    traversable_front_neg[hit_idx_front, 0] = hit_loss < self._cfg.obs_loss_threshold
                    traversable_front = np.all(np.hstack([traversable_front_pos, traversable_front_neg]), axis=1)

                    ray_directions[:, 0] = -1.0
                    _, hit_loss, _, hit_idx_back = self._raycast_check(
                        camera_positions[hit_idx], ray_directions[hit_idx], max_distance=10
                    )
                    traversable_back_neg = np.ones((len(hit_idx), 1), dtype=bool)
                    traversable_back_neg[hit_idx_back, 0] = hit_loss < self._cfg.obs_loss_threshold
                    ray_directions[:, 0] = 1.0
                    _, hit_loss, _, hit_idx_back = self._raycast_check(
                        camera_positions[hit_idx] - np.array([10, 0.0, 0.0]), ray_directions[hit_idx], max_distance=10
                    )
                    traversable_back_pos = np.ones((len(hit_idx), 1), dtype=bool)
                    traversable_back_pos[hit_idx_back, 0] = hit_loss < self._cfg.obs_loss_threshold
                    traversable_back = np.all(np.hstack([traversable_back_pos, traversable_back_neg]), axis=1)

                    ray_directions[:, 0] = 0.0  # reset ray direction

                    ray_directions[:, 1] = 1.0
                    _, hit_loss, _, hit_idx_right = self._raycast_check(
                        camera_positions[hit_idx], ray_directions[hit_idx], max_distance=10
                    )
                    traversable_right_pos = np.ones((len(hit_idx), 1), dtype=bool)
                    traversable_right_pos[hit_idx_right, 0] = hit_loss < self._cfg.obs_loss_threshold
                    ray_directions[:, 1] = -1.0
                    _, hit_loss, _, hit_idx_right = self._raycast_check(
                        camera_positions[hit_idx] + np.array([0.0, 10, 0.0]), ray_directions[hit_idx], max_distance=10
                    )
                    traversable_right_neg = np.ones((len(hit_idx), 1), dtype=bool)
                    traversable_right_neg[hit_idx_right, 0] = hit_loss < self._cfg.obs_loss_threshold
                    traversable_right = np.all(np.hstack([traversable_right_pos, traversable_right_neg]), axis=1)

                    ray_directions[:, 1] = -1.0
                    _, hit_loss, _, hit_idx_left = self._raycast_check(
                        camera_positions[hit_idx], ray_directions[hit_idx], max_distance=10
                    )
                    traversable_left_neg = np.ones((len(hit_idx), 1), dtype=bool)
                    traversable_left_neg[hit_idx_left, 0] = hit_loss < self._cfg.obs_loss_threshold
                    ray_directions[:, 1] = -1.0
                    _, hit_loss, _, hit_idx_left = self._raycast_check(
                        camera_positions[hit_idx] - np.array([0.0, 10, 0.0]), ray_directions[hit_idx], max_distance=10
                    )
                    traversable_left_pos = np.ones((len(hit_idx), 1), dtype=bool)
                    traversable_left_pos[hit_idx_left, 0] = hit_loss < self._cfg.obs_loss_threshold
                    traversable_left = np.all(np.hstack([traversable_left_neg, traversable_left_pos]), axis=1)

                    # filter points
                    traversable_all = np.vstack(
                        [traversable_front, traversable_back, traversable_right, traversable_left]
                    ).all(axis=0)
                    hit_idx_non_traverable = np.array(hit_idx)[~traversable_all]
                else:
                    hit_idx_non_traverable = []
            else:
                # semantics not available -> check compared to mean height
                hit_idx_non_traverable = np.where(camera_positions[:, 2] > np.mean(camera_positions[:, 2]))[0]
        else:
            hit_idx_non_traverable = []

        # update camera positions and nbr of points
        if len(hit_idx_non_traverable) > 0:
            self.camera_positions = np.delete(camera_positions, hit_idx_non_traverable, axis=0)
        else:
            self.camera_positions = camera_positions

        # add more people in the scene
        if self._cfg.nb_more_people is not None:
            random.seed(self._cfg.random_seed)
            pts_idx = random.sample(range(len(self.camera_positions)), self._cfg.nb_more_people)

            if self._cfg_load.scale == 1.0:
                scale_people = 100
            else:
                scale_people = 1

            # add people and remove previous added offset
            offset = np.array([0.0, 0.0, self._cfg.robot_height])
            for idx in pts_idx:
                CarlaLoader.insert_single_person(f"random_{idx}", self.camera_positions[idx] - offset, scale_people)

            self.camera_positions = np.delete(self.camera_positions, pts_idx, axis=0)

        if self._cfg.carla_filter:
            # for CARLA filter large open spaces
            # Extract the x and y coordinates from the odom poses
            x_coords = self.camera_positions[:, 0]
            y_coords = self.camera_positions[:, 1]

            # load file

            # Filter the point cloud based on the square coordinates
            mask_area_1 = (y_coords >= 100.5) & (y_coords <= 325.5) & (x_coords >= 208.9) & (x_coords <= 317.8)
            mask_area_2 = (y_coords >= 12.7) & (y_coords <= 80.6) & (x_coords >= 190.3) & (x_coords <= 315.8)
            mask_area_3 = (y_coords >= 10.0) & (y_coords <= 80.0) & (x_coords >= 123.56) & (x_coords <= 139.37)

            combined_mask = mask_area_1 | mask_area_2 | mask_area_3
            points_free_space = ~combined_mask
            self.camera_positions = self.camera_positions[points_free_space]

        self.nbr_points = len(self.camera_positions)

        # plot dense cover of the mesh
        if self.debug:
            self.sim.play()
            self.draw_interface.draw_points(
                self.camera_positions, [(1, 1, 1, 1)] * len(self.camera_positions), [5] * len(self.camera_positions)
            )
            self.draw_interface.draw_points(
                camera_positions[hit_idx_non_traverable],
                [(1.0, 0.5, 0, 1)] * len(camera_positions[hit_idx_non_traverable]),
                [5] * len(camera_positions[hit_idx_non_traverable]),
            )
            for count in range(100000):
                self.sim.step()
            self.sim.pause()

        return

    def _construct_kdtree(self, num_neighbors: int = 50) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # construct kdtree to find nearest neighbors of points
        camera_position_unified_height = np.copy(self.camera_positions)
        camera_position_unified_height[:, 2] = np.max(self.camera_positions[:, 2])

        kdtree = KDTree(camera_position_unified_height)
        _, nearest_neighbors_idx = kdtree.query(camera_position_unified_height, k=num_neighbors + 1, workers=-1)
        # remove first neighbor as it is the point itself
        nearest_neighbors_idx = nearest_neighbors_idx[:, 1:]

        # define origin and neighbor points
        origin_point = np.repeat(camera_position_unified_height, repeats=num_neighbors, axis=0)
        neighbor_points = camera_position_unified_height[nearest_neighbors_idx, :].reshape(-1, 3)
        distance = np.linalg.norm(origin_point - neighbor_points, axis=1)

        # check for collision with raycasting
        hit_position, _, _, hit_idx = self._raycast_check(
            origin_point, neighbor_points - origin_point, np.max(distance)
        )

        # filter connections that collide with the environment
        collision = np.zeros(len(origin_point), dtype=bool)
        collision[hit_idx] = np.linalg.norm(hit_position - origin_point[hit_idx], axis=1) < distance[hit_idx]
        collision = collision.reshape(-1, num_neighbors)
        return nearest_neighbors_idx, collision, distance

    def _get_cam_target(self) -> None:
        """Camera orientation variation (similar to matterport variation)"""
        # the variation around the up axis (z-axis) has to be picked in order to avoid that the camera faces a wall
        # done by construction a graph of all sample points and pick the z angle in order to point at one of the neighbors of the node

        # get nearest neighbors and check for collision
        nearest_neighbors_idx, collision, _ = self._construct_kdtree()

        # get nodes where all neighbors are in collision
        all_collision_idx = np.all(collision, axis=1)

        # select random neighbor that is not in collision
        direction_neighbor_idx = np.hstack(
            [
                (collision_single_node is False).nonzero()[0][-1]
                for collision_single_node in collision[~all_collision_idx, :]
            ]
        )
        direction_neighbor_idx = np.vstack(
            (np.arange(nearest_neighbors_idx.shape[0])[~all_collision_idx], direction_neighbor_idx)
        ).T
        selected_neighbor_idx = nearest_neighbors_idx[direction_neighbor_idx[:, 0], direction_neighbor_idx[:, 1]]

        # get the z angle of the neighbor that is closest to the origin point
        neighbor_direction = (
            self.camera_positions[~all_collision_idx, :] - self.camera_positions[selected_neighbor_idx, :]
        )
        z_angles = np.rad2deg(np.arctan2(neighbor_direction[:, 1], neighbor_direction[:, 0]))

        # filter points that have no neighbors that are not in collision and update number of points
        self.camera_positions = self.camera_positions[~all_collision_idx, :]
        self.nbr_points = self.camera_positions.shape[0]

        # vary the rotation of the forward and horizontal axis (in camera frame) as a uniform distribution within the limits
        x_angles = np.random.uniform(self._cfg.x_angle_range[0], self._cfg.x_angle_range[1], self.nbr_points)
        y_angles = np.random.uniform(self._cfg.y_angle_range[0], self._cfg.y_angle_range[1], self.nbr_points)

        self.cam_angles = np.hstack((x_angles.reshape(-1, 1), y_angles.reshape(-1, 1), z_angles.reshape(-1, 1)))
        return

    """ Camera and Image Creator """

    def _camera_init(self) -> None:
        # Setup camera sensor
        self.camera_depth = Camera(cfg=self._cfg.camera_cfg_depth, device="cpu")
        self.camera_depth.spawn(self._cfg.camera_prim_depth)
        if self._cfg.camera_intrinsics_depth:
            intrinsic_matrix = np.array(self._cfg.camera_intrinsics_depth).reshape(3, 3)
            self.camera_depth.set_intrinsic_matrix(intrinsic_matrix)
        self.camera_depth.initialize()

        if self._cfg.high_res_depth:
            self._cfg.camera_cfg_sem.data_types += ["distance_to_image_plane"]

        self.camera_semantic = Camera(cfg=self._cfg.camera_cfg_sem, device="cpu")
        self.camera_semantic.spawn(self._cfg.camera_prim_sem)
        if self._cfg.camera_intrinsics_sem:
            intrinsic_matrix = np.array(self._cfg.camera_intrinsics_sem).reshape(3, 3)
            self.camera_semantic.set_intrinsic_matrix(intrinsic_matrix)
        self.camera_semantic.initialize()
        return

    def _domain_recorder(self) -> None:
        """
        Will iterate over all camera positions and orientations while recording the resulting images in the different
        domains (rgb, depth, semantic). The resulting images will be saved in the following folder structure:
        NOTE: depth images are saved as png and the corresponding depth arrays are saved as npy files because the large
        depths in CARLA exceed the int16 range of png images and lead to wrong depth values -> png only for visualization

        - self._cfg.output_dir
            - camera_extrinsic{depth_suffix}.txt  (format: x y z qx qy qz qw for depth camera)
            - camera_extrinsic{sem_suffix}.txt  (format: x y z qx qy qz qw for semantic camera)
            - intrinsics.txt (expects ROS CameraInfo format --> P-Matrix, both cameras)
            - rgb
                - xxxx{sem_suffix}.png  (images should be named with 4 digits, e.g. 0000.png, 0001.png, etc.)
            - depth
                - xxxx{depth_suffix}.png  (images should be named with 4 digits, e.g. 0000.png, 0001.png, etc.)
                - xxxx{depth_suffix}.npy  (arrays should be named with 4 digits, e.g. 0000.npy, 0001.npy, etc.)
            - semantics
                - xxxx{sem_suffix}.png  (images should be named with 4 digits, e.g. 0000.png, 0001.png, etc.)

        The depth and semantic suffix are for example "_depth" and "_sem" and can be set in the config file. They are
        necessary to differentiate between the two cameras and their extrinsics. The suffix in the image naming is for
        for compatibility with the matterport3d explorer.

        If high resolution depth images are enabled, the following additional folder is added:
            - depth_high_res
                - xxxx{depth_suffix}.png  (images should be named with 4 digits, e.g. 0000.png, 0001.png, etc.)
                - xxxx{depth_suffix}.npy  (arrays should be named with 4 digits, e.g. 0000.npy, 0001.npy, etc.)

        """

        # create save dirs for domains
        os.makedirs(os.path.join(self._cfg.output_dir, "rgb"), exist_ok=True)
        os.makedirs(os.path.join(self._cfg.output_dir, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self._cfg.output_dir, "semantics"), exist_ok=True)

        if self._cfg.high_res_depth:
            os.makedirs(os.path.join(self._cfg.output_dir, "depth_high_res"), exist_ok=True)

        # save intrinsics
        intrinsics = []
        depth_intrinsics = (
            np.array(self._cfg.camera_intrinsics_depth).reshape(3, 3)
            if self._cfg.camera_intrinsics_depth
            else self.camera_depth.data.intrinsic_matrix
        )
        sem_intrinsics = (
            np.array(self._cfg.camera_intrinsics_sem).reshape(3, 3)
            if self._cfg.camera_intrinsics_sem
            else self.camera_semantic.data.intrinsic_matrix
        )
        for intrinsics_single in [depth_intrinsics, sem_intrinsics]:
            if self._cfg.ros_p_mat:
                p_mat = np.zeros((3, 4))
                p_mat[:3, :3] = intrinsics_single
                intrinsics.append(p_mat.flatten())
            else:
                intrinsics.append(intrinsics_single.flatten())
        np.savetxt(os.path.join(self._cfg.output_dir, "intrinsics.txt"), np.vstack(intrinsics), delimiter=",")

        # init pose buffers
        sem_poses = np.zeros((self._cfg.max_cam_recordings, 7))
        depth_poses = np.zeros((self._cfg.max_cam_recordings, 7))

        # Play simulator
        self.sim.play()

        # Simulate for a few steps
        # FIXME: This is a workaround to ensure that the textures are loaded.
        #   Check "Known Issues" section in the documentation for more details.
        for _ in range(14):
            self.sim.render()

        # matrix to transform opengl to isaac coordinate system
        isaac_to_opengl_mat = tf.Rotation.from_euler("XYZ", [90, -90, 0], degrees=True).as_matrix()

        for idx, sem_pos in enumerate(self.camera_positions):
            start = time.time()

            # Set semantic camera pose
            sem_rot = self.cam_angles[idx].copy()
            sem_rot = sem_rot.astype(np.float64)  # convert to double precision
            sem_rot_mat = tf.Rotation.from_euler("xyz", sem_rot, degrees=True).as_matrix()
            rot = sem_rot_mat @ isaac_to_opengl_mat
            rot_quad = tf.Rotation.from_matrix(rot).as_quat()
            self.camera_semantic._sensor_xform.set_world_pose(sem_pos, convert_quat(rot_quad, "wxyz"))

            # get correct rotation from depth camera to semantic camera
            if self._cfg.tf_quat_convention == "isaac":
                cam_rot_sem_from_depth = tf.Rotation.from_quat(self._cfg.tf_quat).as_matrix()
            elif self._cfg.tf_quat_convention == "roll-pitch-yaw":
                cam_rot_sem_from_depth = tf.Rotation.from_quat(self._cfg.tf_quat).as_euler("XYZ", degrees=True)
                cam_rot_sem_from_depth[[1, 2]] *= -1
                cam_rot_sem_from_depth = tf.Rotation.from_euler("XYZ", cam_rot_sem_from_depth, degrees=True).as_matrix()
            else:
                raise ValueError(f"tf_quat_convention {self._cfg.tf_quat_convention} not supported")

            # set depth camera pose
            cam_rot_sem_from_depth = cam_rot_sem_from_depth.astype(np.float64)  # convert to double precision
            depth_rot_mat = sem_rot_mat @ cam_rot_sem_from_depth.T  # get depth rotation in odom frame
            depth_pos = sem_pos - depth_rot_mat @ self._cfg.tf_pos
            rot = depth_rot_mat @ isaac_to_opengl_mat
            rot_quad = tf.Rotation.from_matrix(rot).as_quat()
            # set depth camera pose
            self.camera_depth._sensor_xform.set_world_pose(depth_pos, convert_quat(rot_quad, "wxyz"))

            # FIXME: This is a workaround to ensure that the textures are loaded.
            #   Check "Known Issues" section in the documentation for more details.
            for _ in range(5):
                self.sim.render()

            # Update camera data
            self.camera_depth.update(dt=0.0)
            self.camera_semantic.update(dt=0.0)

            # save poses in Isaac convention and extrinsic format (xyz)
            sem_poses[idx, :3] = sem_pos
            sem_poses[idx, 3:] = tf.Rotation.from_matrix(sem_rot_mat).as_quat()
            depth_poses[idx, :3] = depth_pos
            depth_poses[idx, 3:] = tf.Rotation.from_matrix(depth_rot_mat).as_quat()

            # Save images
            # RGB
            if "rgb" in self.camera_semantic.data.output:
                cv2.imwrite(
                    os.path.join(self._cfg.output_dir, "rgb", f"{idx}".zfill(4) + self._cfg.sem_suffix + ".png"),
                    cv2.cvtColor(self.camera_semantic.data.output["rgb"], cv2.COLOR_RGB2BGR),
                )
            # DEPTH
            np.save(
                os.path.join(self._cfg.output_dir, "depth", f"{idx}".zfill(4) + self._cfg.depth_suffix + ".npy"),
                self.camera_depth.data.output["distance_to_image_plane"] * self._cfg.depth_scale,
            )
            cv2.imwrite(
                os.path.join(self._cfg.output_dir, "depth", f"{idx}".zfill(4) + self._cfg.depth_suffix + ".png"),
                (self.camera_depth.data.output["distance_to_image_plane"] * self._cfg.depth_scale).astype(
                    np.uint16
                ),  # convert to meters
            )
            # High Resolution Depth
            if self._cfg.high_res_depth:
                np.save(
                    os.path.join(
                        self._cfg.output_dir, "depth_high_res", f"{idx}".zfill(4) + self._cfg.depth_suffix + ".npy"
                    ),
                    self.camera_semantic.data.output["distance_to_image_plane"] * self._cfg.depth_scale,
                )
                cv2.imwrite(
                    os.path.join(
                        self._cfg.output_dir, "depth_high_res", f"{idx}".zfill(4) + self._cfg.depth_suffix + ".png"
                    ),
                    (self.camera_semantic.data.output["distance_to_image_plane"] * self._cfg.depth_scale).astype(
                        np.uint16
                    ),  # convert to meters
                )

            # SEMANTICS
            if self._cfg_load.sem_mesh_to_class_map:
                class_color_with_unlabelled = self.vip_sem_meta.class_color
                class_color_with_unlabelled["unlabelled"] = [0, 0, 0]

                idToColor = np.array(
                    [
                        [
                            int(k),
                            self.vip_sem_meta.class_color[v["class"].lower()][0],
                            self.vip_sem_meta.class_color[v["class"].lower()][1],
                            self.vip_sem_meta.class_color[v["class"].lower()][2],
                        ]
                        for k, v in self.camera_semantic.data.output["semantic_segmentation"]["info"][
                            "idToLabels"
                        ].items()
                    ]
                )
                idToColorArray = np.zeros((idToColor.max(axis=0)[0] + 1, 3))
                idToColorArray[idToColor[:, 0]] = idToColor[:, 1:]
                sem_img = idToColorArray[
                    self.camera_semantic.data.output["semantic_segmentation"]["data"].reshape(-1)
                ].reshape(self.camera_semantic.data.output["semantic_segmentation"]["data"].shape + (3,))
                cv2.imwrite(
                    os.path.join(self._cfg.output_dir, "semantics", f"{idx}".zfill(4) + self._cfg.sem_suffix + ".png"),
                    cv2.cvtColor(sem_img.astype(np.uint8), cv2.COLOR_RGB2BGR),
                )

            # Print Info
            duration = time.time() - start
            print(f"Recording {idx + 1}/{self.nbr_points} ({(idx + 1) / self.nbr_points * 100:.2f}%) in {duration:.4f}")

            # stop condition
            if self._cfg.max_cam_recordings is not None and idx >= self._cfg.max_cam_recordings - 1:
                break

            if self.debug:
                VisualCuboid(
                    prim_path="/cube_example",  # The prim path of the cube in the USD stage
                    name="waypoint",  # The unique name used to retrieve the object from the scene later on
                    position=sem_pos
                    + (
                        tf.Rotation.from_euler("xyz", sem_rot, degrees=True).as_matrix()
                        @ np.array([100, 0, 0]).reshape(-1, 1)
                    ).reshape(
                        -1
                    ),  # Using the current stage units which is in meters by default.
                    scale=np.array([15, 15, 15]),  # most arguments accept mainly numpy arrays.
                    size=1.0,
                    color=np.array([255, 0, 0]),  # RGB channels, going from 0-1
                )

                import matplotlib.pyplot as plt

                _, axs = plt.subplots(1, 3, figsize=(15, 5))
                axs[0].imshow(self.camera_semantic.data.output["rgb"])
                axs[1].imshow(self.camera_depth.data.output["distance_to_image_plane"])
                axs[2].imshow(self.camera_semantic.data.output["semantic_segmentation"]["data"])
                plt.show()

        np.savetxt(
            os.path.join(self._cfg.output_dir, f"camera_extrinsic{self._cfg.sem_suffix}.txt"),
            sem_poses[:idx],
            delimiter=",",
        )
        np.savetxt(
            os.path.join(self._cfg.output_dir, f"camera_extrinsic{self._cfg.depth_suffix}.txt"),
            depth_poses[:idx],
            delimiter=",",
        )

        return

    @staticmethod
    def compute_bbox_with_cache(cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> Gf.Range3d:
        """
        Compute Bounding Box using ComputeWorldBound at UsdGeom.BBoxCache. More efficient if used multiple times.
        See https://graphics.pixar.com/usd/dev/api/class_usd_geom_b_box_cache.html

        Args:
            cache: A cached, i.e. `UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])`
            prim: A prim to compute the bounding box.
        Returns:
            A range (i.e. bounding box), see more at: https://graphics.pixar.com/usd/release/api/class_gf_range3d.html

        """
        bound = cache.ComputeWorldBound(prim)
        bound_range = bound.ComputeAlignedBox()
        return bound_range


# EoF
