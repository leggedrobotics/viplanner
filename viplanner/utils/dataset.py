# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import copy
import math

# python
import os
import random
import shutil
from pathlib import Path
from random import sample
from typing import Dict, List, Optional, Tuple

import cv2
import networkx as nx
import numpy as np
import open3d as o3d
import PIL
import pypose as pp
import scipy.spatial.transform as tf
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from scipy.spatial.kdtree import KDTree
from skimage.util import random_noise
from torch.utils.data import Dataset
from tqdm import tqdm

# implerative-planner-learning
from viplanner.config import DataCfg
from viplanner.cost_maps import CostMapPCD

# set default dtype to float32
torch.set_default_dtype(torch.float32)


class PlannerData(Dataset):
    def __init__(
        self,
        cfg: DataCfg,
        transform,
        semantics: bool = False,
        rgb: bool = False,
        pixel_mean: Optional[np.ndarray] = None,
        pixel_std: Optional[np.ndarray] = None,
    ) -> None:
        """_summary_

        Args:
            cfg (DataCfg): Dataset COnfiguration
            transform (_type_): Compose torchvision transforms (resize and to tensor)
            semantics (bool, optional): If semantics are used in the network input. Defaults to False.
        """

        self._cfg = cfg
        self.transform = transform
        self.semantics = semantics
        self.rgb = rgb
        assert not (semantics and rgb), "Semantics and RGB cannot be used at the same time"
        self.pixel_mean = pixel_mean
        self.pixel_std = pixel_std

        # vertical flip transform
        self.flip_transform = transforms.RandomHorizontalFlip(p=1.0)

        # init buffers
        self.depth_filename: List[str] = []
        self.sem_rgb_filename: List[str] = []
        self.depth_imgs: List[torch.Tensor] = []
        self.sem_imgs: List[torch.Tensor] = []
        self.odom: torch.Tensor = None
        self.goal: torch.Tensor = None
        self.pair_augment: np.ndarray = None
        self.fov_angle: float = 0.0
        self.load_ram: bool = False
        return

    def update_buffers(
        self,
        depth_filename: List[str],
        sem_rgb_filename: List[str],
        odom: torch.Tensor,
        goal: torch.Tensor,
        pair_augment: np.ndarray,
    ) -> None:
        self.depth_filename = depth_filename
        self.sem_rgb_filename = sem_rgb_filename
        self.odom = odom
        self.goal = goal
        self.pair_augment = pair_augment
        return

    def set_fov(self, fov_angle):
        self.fov_angle = fov_angle
        return

    """Augment Images with black polygons"""

    def _add_random_polygons(self, image, nb_polygons, max_size):
        for i in range(nb_polygons):
            num_corners = random.randint(10, 20)
            polygon_points = np.random.randint(0, max_size, size=(num_corners, 2))
            x_offset = np.random.randint(0, image.shape[0])
            y_offset = np.random.randint(0, image.shape[1])
            polygon_points[:, 0] += x_offset
            polygon_points[:, 1] += y_offset

            # Create a convex hull from the points
            hull = cv2.convexHull(polygon_points)

            # Draw the hull on the image
            cv2.fillPoly(image, [hull], (0, 0, 0))
        return image

    """Load images"""

    def load_data_in_memory(self) -> None:
        """Load data into RAM to speed up training"""
        for idx in tqdm(range(len(self.depth_filename)), desc="Load images into RAM"):
            self.depth_imgs.append(self._load_depth_img(idx))
            if self.semantics or self.rgb:
                self.sem_imgs.append(self._load_sem_rgb_img(idx))
        self.load_ram = True
        return

    def _load_depth_img(self, idx) -> torch.Tensor:
        if self.depth_filename[idx].endswith(".png"):
            depth_image = Image.open(self.depth_filename[idx])
            if self._cfg.real_world_data:
                depth_image = np.array(depth_image.transpose(PIL.Image.ROTATE_180))
            else:
                depth_image = np.array(depth_image)
        else:
            depth_image = np.load(self.depth_filename[idx])
        depth_image[~np.isfinite(depth_image)] = 0.0
        depth_image = (depth_image / 1000.0).astype("float32")
        depth_image[depth_image > self._cfg.max_depth] = 0.0

        # add noise to depth image
        if self._cfg.depth_salt_pepper or self._cfg.depth_gaussian:
            depth_norm = (depth_image - np.min(depth_image)) / (np.max(depth_image) - np.min(depth_image))
            if self._cfg.depth_salt_pepper:
                depth_norm = random_noise(
                    depth_norm,
                    mode="s&p",
                    amount=self._cfg.depth_salt_pepper,
                    clip=False,
                )
            if self._cfg.depth_gaussian:
                depth_norm = random_noise(
                    depth_norm,
                    mode="gaussian",
                    mean=0,
                    var=self._cfg.depth_gaussian,
                    clip=False,
                )
            depth_image = depth_norm * (np.max(depth_image) - np.min(depth_image)) + np.min(depth_image)
        if self._cfg.depth_random_polygons_nb and self._cfg.depth_random_polygons_nb > 0:
            depth_image = self._add_random_polygons(
                depth_image,
                self._cfg.depth_random_polygons_nb,
                self._cfg.depth_random_polygon_size,
            )

        # transform depth image
        depth_image = self.transform(depth_image).type(torch.float32)
        if self.pair_augment[idx]:
            depth_image = self.flip_transform.forward(depth_image)

        return depth_image

    def _load_sem_rgb_img(self, idx) -> torch.Tensor:
        image = Image.open(self.sem_rgb_filename[idx])
        if self._cfg.real_world_data:
            image = np.array(image.transpose(PIL.Image.ROTATE_180))
        else:
            image = np.array(image)
        # normalize image
        if self.pixel_mean is not None and self.pixel_std is not None:
            image = (image - self.pixel_mean) / self.pixel_std

        # add noise to semantic image
        if self._cfg.sem_rgb_black_img:
            if random.randint(0, 99) < self._cfg.sem_rgb_black_img * 100:
                image = np.zeros_like(image)
        if self._cfg.sem_rgb_pepper:
            image = random_noise(
                image,
                mode="pepper",
                amount=self._cfg.depth_salt_pepper,
                clip=False,
            )
        if self._cfg.sem_rgb_random_polygons_nb and self._cfg.sem_rgb_random_polygons_nb > 0:
            image = self._add_random_polygons(
                image,
                self._cfg.sem_rgb_random_polygons_nb,
                self._cfg.sem_rgb_random_polygon_size,
            )

        # transform semantic image
        image = self.transform(image).type(torch.float32)
        assert image.round(decimals=1).max() <= 1.0, (
            f"Image '{self.sem_rgb_filename[idx]}' is not normalized with max" f" value {image.max().item()}"
        )

        if self.pair_augment[idx]:
            image = self.flip_transform.forward(image)

        return image

    """Get image in training"""

    def __len__(self):
        return len(self.depth_filename)

    def __getitem__(self, idx):
        """
        Get batch items

        Returns:
            - depth_image: depth image
            - sem_rgb_image: semantic image
            - odom: odometry of the start pose (point and rotation)
            - goal: goal point in the camera frame
            - pair_augment: bool if the pair is augmented (flipped at the y-axis of the image)
        """

        # get depth image
        if self.load_ram:
            depth_image = self.depth_imgs[idx]
            if self.semantics or self.rgb:
                sem_rgb_image = self.sem_imgs[idx]
            else:
                sem_rgb_image = 0
        else:
            depth_image = self._load_depth_img(idx)
            if self.semantics or self.rgb:
                sem_rgb_image = self._load_sem_rgb_img(idx)
            else:
                sem_rgb_image = 0

        return (
            depth_image,
            sem_rgb_image,
            self.odom[idx],
            self.goal[idx],
            self.pair_augment[idx],
        )


class DistanceSchemeIdx:
    def __init__(self, distance: float) -> None:
        self.distance: float = distance

        self.odom_list: List[pp.LieTensor] = []
        self.goal_list: List[pp.LieTensor] = []
        self.pair_within_fov: List[bool] = []
        self.pair_front_of_robot: List[bool] = []
        self.pair_behind_robot: List[bool] = []
        self.depth_img_list: List[str] = []
        self.sem_rgb_img_list: List[str] = []

        # flags
        self.has_data: bool = False
        return

    def update_buffers(
        self,
        odom: pp.LieTensor,
        goal: pp.LieTensor,
        within_fov: bool = False,
        front_of_robot: bool = False,
        behind_robot: bool = False,
        depth_filename: str = None,
        sem_rgb_filename: str = None,
    ) -> None:
        self.odom_list.append(odom)
        self.goal_list.append(goal)
        self.pair_within_fov.append(within_fov)
        self.pair_front_of_robot.append(front_of_robot)
        self.pair_behind_robot.append(behind_robot)
        self.depth_img_list.append(depth_filename)
        self.sem_rgb_img_list.append(sem_rgb_filename)

        self.has_data = len(self.odom_list) > 0
        return

    def get_data(
        self,
        nb_fov: int,
        nb_front: int,
        nb_back: int,
        augment: bool = True,
    ) -> Tuple[List[pp.LieTensor], List[pp.LieTensor], List[str], List[str], np.ndarray,]:
        assert self.has_data, f"DistanceSchemeIdx for distance {self.distance} has no data"

        # get all pairs that are within the fov
        idx_fov = np.where(self.pair_within_fov)[0]
        idx_front = np.where(self.pair_front_of_robot)[0]
        idx_back = np.where(self.pair_behind_robot)[0]
        idx_augment = []

        # augment pairs if not enough
        if len(idx_fov) == 0:
            print(f"[WARNING] for distance {self.distance} no 'within_fov'" " samples")
            idx_fov = np.array([], dtype=np.int64)
        elif len(idx_fov) < nb_fov:
            print(
                f"[INFO] for distance {self.distance} not enough 'within_fov'"
                f" samples ({len(idx_fov)} instead of {nb_fov})"
            )
            if augment:
                idx_augment.append(
                    np.random.choice(
                        idx_fov,
                        min(len(idx_fov), nb_fov - len(idx_fov)),
                        replace=(nb_fov - len(idx_fov) > len(idx_fov)),
                    )
                )
            else:
                idx_fov = np.random.choice(idx_fov, len(idx_fov), replace=False)
        else:
            idx_fov = np.random.choice(idx_fov, nb_fov, replace=False)

        if len(idx_front) == 0:
            print(f"[WARNING] for distance {self.distance} no 'front_of_robot'" " samples")
            idx_front = np.array([], dtype=np.int64)
        elif len(idx_front) < nb_front:
            print(
                f"[INFO] for distance {self.distance} not enough"
                f" 'front_of_robot' samples ({len(idx_front)} instead of"
                f" {nb_front})"
            )
            if augment:
                idx_augment.append(
                    np.random.choice(
                        idx_front,
                        min(len(idx_front), nb_front - len(idx_front)),
                        replace=(nb_front - len(idx_front) > len(idx_front)),
                    )
                )
            else:
                idx_front = np.random.choice(idx_front, len(idx_front), replace=False)
        else:
            idx_front = np.random.choice(idx_front, nb_front, replace=False)

        if len(idx_back) == 0:
            print(f"[WARNING] for distance {self.distance} no 'behind_robot'" " samples")
            idx_back = np.array([], dtype=np.int64)
        elif len(idx_back) < nb_back:
            print(
                f"[INFO] for distance {self.distance} not enough"
                f" 'behind_robot' samples ({len(idx_back)} instead of"
                f" {nb_back})"
            )
            if augment:
                idx_augment.append(
                    np.random.choice(
                        idx_back,
                        min(len(idx_back), nb_back - len(idx_back)),
                        replace=(nb_back - len(idx_back) > len(idx_back)),
                    )
                )
            else:
                idx_back = np.random.choice(idx_back, len(idx_back), replace=False)
        else:
            idx_back = np.random.choice(idx_back, nb_back, replace=False)

        idx = np.hstack([idx_fov, idx_front, idx_back])

        # stack buffers
        odom = torch.stack(self.odom_list)
        goal = torch.stack(self.goal_list)

        # get pairs
        if idx_augment:
            idx_augment = np.hstack(idx_augment)
            odom = torch.vstack([odom[idx], odom[idx_augment]])
            goal = torch.vstack(
                [
                    goal[idx],
                    goal[idx_augment].tensor() * torch.tensor([[1, -1, 1, 1, 1, 1, 1]]),
                ]
            )
            depth_img_list = [self.depth_img_list[j] for j in idx.tolist()] + [
                self.depth_img_list[i] for i in idx_augment.tolist()
            ]
            sem_rgb_img_list = [self.sem_rgb_img_list[j] for j in idx.tolist()] + [
                self.sem_rgb_img_list[i] for i in idx_augment.tolist()
            ]
            augment = np.hstack([np.zeros(len(idx)), np.ones(len(idx_augment))])
            return odom, goal, depth_img_list, sem_rgb_img_list, augment
        else:
            return (
                odom[idx],
                goal[idx],
                [self.depth_img_list[j] for j in idx.tolist()],
                [self.sem_rgb_img_list[j] for j in idx.tolist()],
                np.zeros(len(idx)),
            )


class PlannerDataGenerator(Dataset):
    debug = False
    mesh_size = 0.5

    def __init__(
        self,
        cfg: DataCfg,
        root: str,
        semantics: bool = False,
        rgb: bool = False,
        cost_map: CostMapPCD = None,
    ) -> None:
        print(
            f"[INFO] PlannerDataGenerator init with semantics={semantics},"
            f" rgb={rgb} for ENV {os.path.split(root)[-1]}"
        )
        # super().__init__()
        # set parameters
        self._cfg = cfg
        self.root = root
        self.cost_map = cost_map
        self.semantics = semantics
        self.rgb = rgb
        assert not (self.semantics and self.rgb), "semantics and rgb cannot be true at the same time"

        # init list for final odom, goal and img mapping
        self.depth_filename_list = []
        self.sem_rgb_filename_list = []
        self.odom_depth: torch.Tensor = None
        self.goal: torch.Tensor = None
        self.pair_outside: np.ndarray = None
        self.pair_difficult: np.ndarray = None
        self.pair_augment: np.ndarray = None
        self.pair_within_fov: np.ndarray = None
        self.pair_front_of_robot: np.ndarray = None
        self.odom_array_sem_rgb: pp.LieTensor = None
        self.odom_array_depth: pp.LieTensor = None

        self.odom_used: int = 0
        self.odom_no_suitable_goals: int = 0

        # set parameters
        self._device = "cuda:0" if torch.cuda.is_available() else "cpu"

        # get odom data and filter
        self.load_odom()
        self.filter_obs_inflation()

        # noise edges in depth image --> real world Realsense difficulties along edges
        if self._cfg.noise_edges:
            self.noise_edges()

        # find odom-goal pairs
        self.get_odom_goal_pairs()
        return

    """LOAD HELPER FUNCTIONS"""

    def load_odom(self) -> None:
        print("[INFO] Loading odom data...", end=" ")
        # load odom of every image
        odom_path = os.path.join(self.root, f"camera_extrinsic{self._cfg.depth_suffix}.txt")
        odom_np = np.loadtxt(odom_path, delimiter=",")
        self.odom_array_depth = pp.SE3(odom_np)

        if self.semantics or self.rgb:
            odom_path = os.path.join(self.root, f"camera_extrinsic{self._cfg.sem_suffix}.txt")
            odom_np = np.loadtxt(odom_path, delimiter=",")
            self.odom_array_sem_rgb = pp.SE3(odom_np)

        if self.debug:
            # plot odom
            small_sphere = o3d.geometry.TriangleMesh.create_sphere(self.mesh_size / 3.0)  # successful trajectory points
            small_sphere.paint_uniform_color([0.4, 1.0, 0.1])
            odom_vis_list = []

            for i in range(len(self.odom_array_depth)):
                odom_vis_list.append(
                    copy.deepcopy(small_sphere).translate(
                        (
                            self.odom_array_depth[i, 0],
                            self.odom_array_depth[i, 1],
                            self.odom_array_depth[i, 2],
                        )
                    )
                )
            odom_vis_list.append(self.cost_map.pcd_tsdf)

            o3d.visualization.draw_geometries(odom_vis_list)
        print("DONE!")
        return

    def load_images(self, root_path, domain: str = "depth"):
        img_path = os.path.join(root_path, domain)
        assert os.path.isdir(img_path), f"Image directory path '{img_path}' does not exist for domain" f" {domain}"
        assert len(os.listdir(img_path)) > 0, f"Image directory '{img_path}' is empty for domain {domain}"

        # use the more precise npy files if available
        img_filename_list = [str(s) for s in Path(img_path).rglob("*.npy")]
        if len(img_filename_list) == 0:
            img_filename_list = [str(s) for s in Path(img_path).rglob("*.png")]

        if domain == "depth":
            img_filename_list.sort(key=lambda x: int(x.split("/")[-1][: -(4 + len(self._cfg.depth_suffix))]))
        else:
            img_filename_list.sort(key=lambda x: int(x.split("/")[-1][: -(4 + len(self._cfg.sem_suffix))]))
        return img_filename_list

    """FILTER HELPER FUNCTIONS"""

    def filter_obs_inflation(self) -> None:
        """
        Filter odom points within the inflation range of the obstacles in the cost map.

        Filtering only performed according to the position of the depth camera, due to the close position of depth and semantic camera.
        """
        print(
            ("[INFO] Filter odom points within the inflation range of the" " obstacles in the cost map..."),
            end="",
        )

        norm_inds, _ = self.cost_map.Pos2Ind(self.odom_array_depth[:, None, :3])
        cost_grid = self.cost_map.cost_array.T.expand(self.odom_array_depth.shape[0], 1, -1, -1)
        norm_inds = norm_inds.to(cost_grid.device)
        oloss_M = (
            F.grid_sample(
                cost_grid,
                norm_inds[:, None, :, :],
                mode="bicubic",
                padding_mode="border",
                align_corners=False,
            )
            .squeeze(1)
            .squeeze(1)
        )
        oloss_M = oloss_M.to(torch.float32).to("cpu")
        if self.semantics or self.rgb:
            points_free_space = oloss_M < self._cfg.obs_cost_height + abs(
                self.cost_map.cfg.sem_cost_map.negative_reward
            )
        else:
            points_free_space = oloss_M < self._cfg.obs_cost_height

        if self._cfg.carla:
            # for CARLA filter large open spaces
            # Extract the x and y coordinates from the odom poses
            x_coords = self.odom_array_depth.tensor()[:, 0]
            y_coords = self.odom_array_depth.tensor()[:, 1]

            # Filter the point cloud based on the square coordinates
            mask_area_1 = (y_coords >= 100.5) & (y_coords <= 325.5) & (x_coords >= 208.9) & (x_coords <= 317.8)
            mask_area_2 = (y_coords >= 12.7) & (y_coords <= 80.6) & (x_coords >= 190.3) & (x_coords <= 315.8)
            mask_area_3 = (y_coords >= 10.0) & (y_coords <= 80.0) & (x_coords >= 123.56) & (x_coords <= 139.37)

            combined_mask = mask_area_1 | mask_area_2 | mask_area_3 | ~points_free_space.squeeze(1)
            points_free_space = (~combined_mask).unsqueeze(1)

        if self.debug:
            # plot odom
            odom_vis_list = []
            small_sphere = o3d.geometry.TriangleMesh.create_sphere(self.mesh_size / 3.0)  # successful trajectory points

            for i in range(len(self.odom_array_depth)):
                if round(oloss_M[i].item(), 3) == 0.0:
                    small_sphere.paint_uniform_color([0.4, 0.1, 1.0])  # violette
                elif points_free_space[i]:
                    small_sphere.paint_uniform_color([0.4, 1.0, 0.1])  # green
                else:
                    small_sphere.paint_uniform_color([1.0, 0.4, 0.1])  # red
                if self.semantics or self.rgb:
                    z_height = self.odom_array_depth.tensor()[i, 2] + abs(
                        self.cost_map.cfg.sem_cost_map.negative_reward
                    )
                else:
                    z_height = self.odom_array_depth.tensor()[i, 2]

                odom_vis_list.append(
                    copy.deepcopy(small_sphere).translate(
                        (
                            self.odom_array_depth.tensor()[i, 0],
                            self.odom_array_depth.tensor()[i, 1],
                            z_height,
                        )
                    )
                )

            odom_vis_list.append(self.cost_map.pcd_tsdf)
            o3d.visualization.draw_geometries(odom_vis_list)

        nb_odom_point_prev = len(self.odom_array_depth)
        self.odom_array_depth = self.odom_array_depth[points_free_space.squeeze()]
        self.nb_odom_points = self.odom_array_depth.shape[0]

        # load depth image files as name list
        depth_filename_list = self.load_images(self.root, "depth")
        self.depth_filename_list = [
            depth_filename_list[i] for i in range(len(depth_filename_list)) if points_free_space[i]
        ]

        if self.semantics:
            self.odom_array_sem_rgb = self.odom_array_sem_rgb[points_free_space.squeeze()]
            sem_rgb_filename_list = self.load_images(self.root, "semantics")
            self.sem_rgb_filename_list = [
                sem_rgb_filename_list[i] for i in range(len(sem_rgb_filename_list)) if points_free_space[i]
            ]
        elif self.rgb:
            self.odom_array_sem_rgb = self.odom_array_sem_rgb[points_free_space.squeeze()]
            sem_rgb_filename_list = self.load_images(self.root, "rgb")
            self.sem_rgb_filename_list = [
                sem_rgb_filename_list[i] for i in range(len(sem_rgb_filename_list)) if points_free_space[i]
            ]

        assert len(self.depth_filename_list) != 0, "No depth images left after filtering"
        print("DONE!")
        print(
            "[INFO] odom points outside obs inflation :"
            f" \t{self.nb_odom_points} ({round(self.nb_odom_points/nb_odom_point_prev*100, 2)} %)"
        )

        return

    """GENERATE SAMPLES"""

    def get_odom_goal_pairs(self) -> None:
        # get fov
        self.get_intrinscs_and_fov()
        # construct graph
        self.get_graph()
        # get pairs
        self.get_pairs()

        # free up memory
        self.odom_array_depth = self.odom_array_sem_rgb = None
        return

    def compute_ratios(self) -> Tuple[float, float, float]:
        # ratio of general samples distribution
        num_within_fov = self.odom_depth[self.pair_within_fov].shape[0]
        ratio_fov = num_within_fov / self.odom_depth.shape[0]
        ratio_front = np.sum(self.pair_front_of_robot) / self.odom_depth.shape[0]
        ratio_back = 1 - ratio_front - ratio_fov

        # samples ratios within fov samples
        num_easy = (
            num_within_fov
            - self.pair_difficult[self.pair_within_fov].sum().item()
            - self.pair_outside[self.pair_within_fov].sum().item()
        )
        ratio_easy = num_easy / num_within_fov
        ratio_hard = self.pair_difficult[self.pair_within_fov].sum().item() / num_within_fov
        ratio_outside = self.pair_outside[self.pair_within_fov].sum().item() / num_within_fov
        return (
            ratio_fov,
            ratio_front,
            ratio_back,
            ratio_easy,
            ratio_hard,
            ratio_outside,
        )

    def get_intrinscs_and_fov(self) -> None:
        # load intrinsics
        intrinsic_path = os.path.join(self.root, "intrinsics.txt")
        P = np.loadtxt(intrinsic_path, delimiter=",")  # assumes ROS P matrix
        self.K_depth = P[0].reshape(3, 4)[:3, :3]
        self.K_sem_rgb = P[1].reshape(3, 4)[:3, :3]

        self.alpha_fov = 2 * math.atan(self.K_depth[0, 0] / self.K_depth[0, 2])
        return

    def get_graph(self) -> None:
        num_connections = 3
        num_intermediate = 3

        # get occpuancy map from tsdf map
        cost_array = self.cost_map.tsdf_array.cpu().numpy()
        if self.semantics or self.rgb:
            occupancy_map = (
                cost_array > self._cfg.obs_cost_height + abs(self.cost_map.cfg.sem_cost_map.negative_reward)
            ).astype(np.uint8)
        else:
            occupancy_map = (cost_array > self._cfg.obs_cost_height).astype(np.uint8)
        # construct kdtree to find nearest neighbors of points
        odom_points = self.odom_array_depth.data[:, :2].data.cpu().numpy()
        kdtree = KDTree(odom_points)
        _, nearest_neighbors_idx = kdtree.query(odom_points, k=num_connections + 1, workers=-1)
        # remove first neighbor as it is the point itself
        nearest_neighbors_idx = nearest_neighbors_idx[:, 1:]

        # define origin and neighbor points
        origin_point = np.repeat(odom_points, repeats=num_connections, axis=0)
        neighbor_points = odom_points[nearest_neighbors_idx, :].reshape(-1, 2)
        # interpolate points between origin and neighbor points
        x_interp = (
            origin_point[:, None, 0]
            + (neighbor_points[:, 0] - origin_point[:, 0])[:, None]
            * np.linspace(0, 1, num=num_intermediate + 1, endpoint=False)[1:]
        )
        y_interp = (
            origin_point[:, None, 1]
            + (neighbor_points[:, 1] - origin_point[:, 1])[:, None]
            * np.linspace(0, 1, num=num_intermediate + 1, endpoint=False)[1:]
        )
        inter_points = np.stack((x_interp.reshape(-1), y_interp.reshape(-1)), axis=1)
        # get the indices of the interpolated points in the occupancy map
        occupancy_idx = (
            inter_points - np.array([self.cost_map.cfg.x_start, self.cost_map.cfg.y_start])
        ) / self.cost_map.cfg.general.resolution

        # check occupancy for collisions at the interpolated points
        collision = occupancy_map[
            occupancy_idx[:, 0].astype(np.int64),
            occupancy_idx[:, 1].astype(np.int64),
        ]
        collision = np.any(collision.reshape(-1, num_intermediate), axis=1)

        # get edge indices
        idx_edge_start = np.repeat(np.arange(odom_points.shape[0]), repeats=num_connections, axis=0)
        idx_edge_end = nearest_neighbors_idx.reshape(-1)

        # filter collision edges
        idx_edge_end = idx_edge_end[~collision]
        idx_edge_start = idx_edge_start[~collision]

        # init graph
        self.graph = nx.Graph()
        # add nodes with position attributes
        self.graph.add_nodes_from(list(range(odom_points.shape[0])))
        pos_attr = {i: {"pos": odom_points[i]} for i in range(odom_points.shape[0])}
        nx.set_node_attributes(self.graph, pos_attr)
        # add edges with distance attributes
        self.graph.add_edges_from(list(map(tuple, np.stack((idx_edge_start, idx_edge_end), axis=1))))
        distance_attr = {
            (i, j): {"distance": np.linalg.norm(odom_points[i] - odom_points[j])}
            for i, j in zip(idx_edge_start, idx_edge_end)
        }
        nx.set_edge_attributes(self.graph, distance_attr)

        # DEBUG
        if self.debug:
            import matplotlib.pyplot as plt

            nx.draw_networkx(
                self.graph,
                nx.get_node_attributes(self.graph, "pos"),
                node_size=10,
                with_labels=False,
                node_color=[0.0, 1.0, 0.0],
            )
            plt.show()
        return

    def get_pairs(self):
        # iterate over all odom points and find goal points
        self.odom_no_suitable_goals = 0
        self.odom_used = 0

        # init semantic warp parameters
        if self.semantics or self.rgb:
            # compute pixel tensor
            depth_filename = self.depth_filename_list[0]
            depth_img = self._load_depth_image(depth_filename)
            x_nums, y_nums = depth_img.shape
            self.pix_depth_cam_frame = self.compute_pixel_tensor(x_nums, y_nums, self.K_depth)
            # make dir
            os.makedirs(os.path.join(self.root, "img_warp"), exist_ok=True)

        # get distances between odom and goal points
        odom_goal_distances = dict(
            nx.all_pairs_dijkstra_path_length(
                self.graph,
                cutoff=self._cfg.max_goal_distance,
                weight="distance",
            )
        )

        # init dataclass for each entry in the distance scheme
        self.category_scheme_pairs: Dict[float, DistanceSchemeIdx] = {
            distance: DistanceSchemeIdx(distance=distance) for distance in self._cfg.distance_scheme.keys()
        }

        # iterate over all odom points
        for odom_idx in tqdm(range(self.nb_odom_points), desc="Start-End Pairs Generation"):
            odom = self.odom_array_depth[odom_idx]

            # transform all odom points to current odom frame
            goals = pp.Inv(odom) @ self.odom_array_depth
            # categorize goals
            (
                within_fov,
                front_of_robot,
                behind_robot,
            ) = self.get_goal_categories(
                goals
            )  # returns goals in odom frame

            # filter odom if no suitable goals within the fov are found
            if within_fov.sum() == 0:
                self.odom_no_suitable_goals += 1
                continue
            self.odom_used += 1

            if self.semantics or self.rgb:
                # semantic warp
                img_new_path = self._get_overlay_img(odom_idx)
            else:
                img_new_path = None

            # get pair according to distance scheme for each category
            self.reduce_pairs(
                odom_idx,
                goals,
                within_fov,
                odom_goal_distances[odom_idx],
                img_new_path,
                within_fov=True,
            )
            self.reduce_pairs(
                odom_idx,
                goals,
                behind_robot,
                odom_goal_distances[odom_idx],
                img_new_path,
                behind_robot=True,
            )
            self.reduce_pairs(
                odom_idx,
                goals,
                front_of_robot,
                odom_goal_distances[odom_idx],
                img_new_path,
                front_of_robot=True,
            )

            # DEBUG
            if self.debug:
                # plot odom
                small_sphere = o3d.geometry.TriangleMesh.create_sphere(
                    self.mesh_size / 3.0
                )  # successful trajectory points
                odom_vis_list = []
                goal_odom = odom @ goals
                hit_pcd = (goal_odom).cpu().numpy()[:, :3]
                for idx, pts in enumerate(hit_pcd):
                    if within_fov[idx]:
                        small_sphere.paint_uniform_color([0.4, 1.0, 0.1])
                    elif front_of_robot[idx]:
                        small_sphere.paint_uniform_color([0.0, 0.5, 0.5])
                    else:
                        small_sphere.paint_uniform_color([0.0, 0.1, 1.0])
                    odom_vis_list.append(copy.deepcopy(small_sphere).translate((pts[0], pts[1], pts[2])))

                # viz cost map
                odom_vis_list.append(self.cost_map.pcd_tsdf)

                # field of view visualization
                fov_vis_length = 0.75  # length of the fov visualization plane in meters
                fov_vis_pt_right = odom @ pp.SE3(
                    [
                        fov_vis_length * np.cos(self.alpha_fov / 2),
                        fov_vis_length * np.sin(self.alpha_fov / 2),
                        0,
                        0,
                        0,
                        0,
                        1,
                    ]
                )
                fov_vis_pt_left = odom @ pp.SE3(
                    [
                        fov_vis_length * np.cos(self.alpha_fov / 2),
                        -fov_vis_length * np.sin(self.alpha_fov / 2),
                        0,
                        0,
                        0,
                        0,
                        1,
                    ]
                )
                fov_vis_pt_right = fov_vis_pt_right.numpy()[:3]
                fov_vis_pt_left = fov_vis_pt_left.numpy()[:3]
                fov_mesh = o3d.geometry.TriangleMesh(
                    vertices=o3d.utility.Vector3dVector(
                        np.array(
                            [
                                odom.data.cpu().numpy()[:3],
                                fov_vis_pt_right,
                                fov_vis_pt_left,
                            ]
                        )
                    ),
                    triangles=o3d.utility.Vector3iVector(np.array([[2, 1, 0]])),
                )
                fov_mesh.paint_uniform_color([1.0, 0.5, 0.0])
                odom_vis_list.append(fov_mesh)

                # odom viz
                small_sphere.paint_uniform_color([1.0, 0.0, 0.0])
                odom_vis_list.append(
                    copy.deepcopy(small_sphere).translate(
                        (
                            odom.data[0].item(),
                            odom.data[1].item(),
                            odom.data[2].item(),
                        )
                    )
                )

                # plot goal
                o3d.visualization.draw_geometries(odom_vis_list)

        if self.debug:
            small_sphere = o3d.geometry.TriangleMesh.create_sphere(self.mesh_size / 3.0)  # successful trajectory points
            odom_vis_list = []

            for distance in self._cfg.distance_scheme.keys():
                odoms = torch.vstack(self.category_scheme_pairs[distance].odom_list)
                odoms = odoms.tensor().cpu().numpy()[:, :3]
                for idx, odom in enumerate(odoms):
                    odom_vis_list.append(copy.deepcopy(small_sphere).translate((odom[0], odom[1], odom[2])))
                    if idx > 10:
                        break
            # viz cost map
            odom_vis_list.append(self.cost_map.pcd_tsdf)

            # plot goal
            o3d.visualization.draw_geometries(odom_vis_list)

        return

    def reduce_pairs(
        self,
        odom_idx: int,
        goals: pp.LieTensor,
        decision_tensor: torch.Tensor,
        odom_distances: dict,
        warp_img_path: Optional[str],
        within_fov: bool = False,
        behind_robot: bool = False,
        front_of_robot: bool = False,
    ):
        # remove all goals depending on the decision tensor from the odom_distances dict
        keep_distance_entries = decision_tensor[list(odom_distances.keys())]
        distances = np.array(list(odom_distances.values()))[keep_distance_entries.numpy()]
        goal_idx = np.array(list(odom_distances.keys()))[keep_distance_entries.numpy()]

        # max distance enforced odom_distances, here enforce min distance
        within_distance_idx = distances > self._cfg.min_goal_distance
        goal_idx = goal_idx[within_distance_idx]
        distances = distances[within_distance_idx]

        # check if there are any goals left
        if len(goal_idx) == 0:
            return

        # select the goal according to the distance_scheme
        for distance in self._cfg.distance_scheme.keys():
            # select nbr_samples from goals within distance
            within_curr_distance_idx = distances < distance
            if sum(within_curr_distance_idx) == 0:
                continue
            selected_idx = np.random.choice(
                goal_idx[within_curr_distance_idx],
                min(3, sum(within_curr_distance_idx)),
                replace=False,
            )
            # remove the selected goals from the list for further selection
            distances = distances[~within_curr_distance_idx]
            goal_idx = goal_idx[~within_curr_distance_idx]

            for idx in selected_idx:
                self.category_scheme_pairs[distance].update_buffers(
                    odom=self.odom_array_depth[odom_idx],
                    goal=goals[idx],
                    within_fov=within_fov,
                    front_of_robot=front_of_robot,
                    behind_robot=behind_robot,
                    depth_filename=self.depth_filename_list[odom_idx],
                    sem_rgb_filename=warp_img_path,
                )

    def get_goal_categories(self, goal_odom_frame: pp.LieTensor):
        """
        Decide which of the samples are within the fov, in front of the robot or behind the robot.
        """
        # get if odom-goal is within fov or outside the fov but still in front of the robot
        goal_angle = abs(torch.atan2(goal_odom_frame.data[:, 1], goal_odom_frame.data[:, 0]))
        within_fov = goal_angle < self.alpha_fov / 2 * self._cfg.fov_scale
        front_of_robot = goal_angle < torch.pi / 2
        front_of_robot[within_fov] = False

        behind_robot = ~front_of_robot.clone()
        behind_robot[within_fov] = False

        return within_fov, front_of_robot, behind_robot

    """SPLIT HELPER FUNCTIONS"""

    def split_samples(
        self,
        test_dataset: PlannerData,
        train_dataset: Optional[PlannerData] = None,
        generate_split: bool = False,
        ratio_fov_samples: Optional[float] = None,
        ratio_front_samples: Optional[float] = None,
        ratio_back_samples: Optional[float] = None,
        allow_augmentation: bool = True,
    ) -> None:
        # check if ratios are given or defaults are used
        ratio_fov_samples = ratio_fov_samples if ratio_fov_samples is not None else self._cfg.ratio_fov_samples
        ratio_front_samples = ratio_front_samples if ratio_front_samples is not None else self._cfg.ratio_front_samples
        ratio_back_samples = ratio_back_samples if ratio_back_samples is not None else self._cfg.ratio_back_samples
        assert round(ratio_fov_samples + ratio_front_samples + ratio_back_samples, 2) == 1.0, (
            "Sample ratios must sum up to 1.0, currently"
            f" {ratio_back_samples + ratio_front_samples + ratio_fov_samples}"
        )

        # max sample number
        if self._cfg.max_train_pairs:
            max_sample_number = min(
                int(self._cfg.max_train_pairs / self._cfg.ratio),
                int(self.odom_used * self._cfg.pairs_per_image),
            )
        else:
            max_sample_number = int(self.odom_used * self._cfg.pairs_per_image)

        # init buffers
        odom = torch.zeros((max_sample_number, 7), dtype=torch.float32)
        goal = torch.zeros((max_sample_number, 7), dtype=torch.float32)
        augment_samples = np.zeros((max_sample_number), dtype=bool)
        depth_filename = []
        sem_rgb_filename = []

        current_idx = 0
        for distance, distance_percentage in self._cfg.distance_scheme.items():
            if not self.category_scheme_pairs[distance].has_data:
                print(f"[WARN] No samples for distance {distance} in ENV" f" {os.path.split(self.root)[-1]}")
                continue

            # get number of samples
            buffer_data = self.category_scheme_pairs[distance].get_data(
                nb_fov=int(ratio_fov_samples * distance_percentage * max_sample_number),
                nb_front=int(ratio_front_samples * distance_percentage * max_sample_number),
                nb_back=int(ratio_back_samples * distance_percentage * max_sample_number),
                augment=allow_augmentation,
            )
            nb_samples = buffer_data[0].shape[0]

            # add to buffers
            odom[current_idx : current_idx + nb_samples] = buffer_data[0]
            goal[current_idx : current_idx + nb_samples] = buffer_data[1]
            depth_filename += buffer_data[2]
            sem_rgb_filename += buffer_data[3]
            augment_samples[current_idx : current_idx + nb_samples] = buffer_data[4]

            current_idx += nb_samples

        # cut off unused space
        odom = odom[:current_idx]
        goal = goal[:current_idx]
        augment_samples = augment_samples[:current_idx]

        # print data mix
        print(
            f"[INFO] datamix containing {odom.shape[0]} suitable odom-goal"
            " pairs: \n"
            "\t fov               :"
            f" \t{int(odom.shape[0] * ratio_fov_samples)  } ({round(ratio_fov_samples*100, 2)} %) \n"
            "\t front of robot    :"
            f" \t{int(odom.shape[0] * ratio_front_samples)} ({round(ratio_front_samples*100, 2)} %) \n"
            "\t back of robot     :"
            f" \t{int(odom.shape[0] * ratio_back_samples) } ({round(ratio_back_samples*100, 2)} %) \n"
            "from"
            f" {self.odom_used} ({round(self.odom_used/self.nb_odom_points*100, 2)} %)"
            " different starting points where \n"
            "\t non-suitable filter:"
            f" {self.odom_no_suitable_goals} ({round(self.odom_no_suitable_goals/self.nb_odom_points*100, 2)} %)"
        )

        # generate split
        idx = np.arange(odom.shape[0])
        if generate_split:
            train_index = sample(idx.tolist(), int(len(idx) * self._cfg.ratio))
            idx = np.delete(idx, train_index)

            train_dataset.update_buffers(
                depth_filename=[depth_filename[i] for i in train_index],
                sem_rgb_filename=([sem_rgb_filename[i] for i in train_index] if (self.semantics or self.rgb) else None),
                odom=odom[train_index],
                goal=goal[train_index],
                pair_augment=augment_samples[train_index],
            )
            train_dataset.set_fov(self.alpha_fov)

        test_dataset.update_buffers(
            depth_filename=[depth_filename[i] for i in idx],
            sem_rgb_filename=([sem_rgb_filename[i] for i in idx] if (self.semantics or self.rgb) else None),
            odom=odom[idx],
            goal=goal[idx],
            pair_augment=augment_samples[idx],
        )
        test_dataset.set_fov(self.alpha_fov)

        return

    """ Warp semantic on depth image helper functions"""

    @staticmethod
    def compute_pixel_tensor(x_nums: int, y_nums: int, K_depth: np.ndarray) -> None:
        # get image plane mesh grid
        pix_u = np.arange(0, y_nums)
        pix_v = np.arange(0, x_nums)
        grid = np.meshgrid(pix_u, pix_v)
        pixels = np.vstack(list(map(np.ravel, grid))).T
        pixels = np.hstack([pixels, np.ones((len(pixels), 1))])  # add ones for 3D coordinates

        # transform to camera frame
        k_inv = np.linalg.inv(K_depth)
        pix_cam_frame = np.matmul(k_inv, pixels.T)
        # reorder to be in "robotics" axis order (x forward, y left, z up)
        return pix_cam_frame[[2, 0, 1], :].T * np.array([1, -1, -1])

    def _load_depth_image(self, depth_filename):
        if depth_filename.endswith(".png"):
            depth_image = Image.open(depth_filename)
            if self._cfg.real_world_data:
                depth_image = np.array(depth_image.transpose(PIL.Image.ROTATE_180))
            else:
                depth_image = np.array(depth_image)
        else:
            depth_image = np.load(depth_filename)

        depth_image[~np.isfinite(depth_image)] = 0.0
        depth_image = (depth_image / self._cfg.depth_scale).astype("float32")
        depth_image[depth_image > self._cfg.max_depth] = 0.0
        return depth_image

    @staticmethod
    def compute_overlay(
        pose_dep,
        pose_sem,
        depth_img,
        sem_rgb_image,
        pix_depth_cam_frame,
        K_sem_rgb,
    ):
        # get 3D points of depth image
        rot = tf.Rotation.from_quat(pose_dep[3:]).as_matrix()
        dep_im_reshaped = depth_img.reshape(
            -1, 1
        )  # flip s.t. start in lower left corner of image as (0,0) -> has to fit to the pixel tensor
        points = dep_im_reshaped * (rot @ pix_depth_cam_frame.T).T + pose_dep[:3]

        # transform points to semantic camera frame
        points_sem_cam_frame = (tf.Rotation.from_quat(pose_sem[3:]).as_matrix().T @ (points - pose_sem[:3]).T).T
        # normalize points
        points_sem_cam_frame_norm = points_sem_cam_frame / points_sem_cam_frame[:, 0][:, np.newaxis]
        # reorder points be camera convention (z-forward)
        points_sem_cam_frame_norm = points_sem_cam_frame_norm[:, [1, 2, 0]] * np.array([-1, -1, 1])
        # transform points to pixel coordinates
        pixels = (K_sem_rgb @ points_sem_cam_frame_norm.T).T
        # filter points outside of image
        filter_idx = (
            (pixels[:, 0] >= 0)
            & (pixels[:, 0] < sem_rgb_image.shape[1])
            & (pixels[:, 1] >= 0)
            & (pixels[:, 1] < sem_rgb_image.shape[0])
        )
        # get semantic annotation
        sem_annotation = np.zeros((pixels.shape[0], 3), dtype=np.uint8)
        sem_annotation[filter_idx] = sem_rgb_image[
            pixels[filter_idx, 1].astype(int),
            pixels[filter_idx, 0].astype(int),
        ]
        # reshape to image

        return sem_annotation.reshape(depth_img.shape[0], depth_img.shape[1], 3)

    def _get_overlay_img(self, odom_idx):
        # get corresponding filenames
        depth_filename = self.depth_filename_list[odom_idx]
        sem_rgb_filename = self.sem_rgb_filename_list[odom_idx]

        # load semantic and depth image and get their poses
        depth_img = self._load_depth_image(depth_filename)
        sem_rgb_image = Image.open(sem_rgb_filename)
        if self._cfg.real_world_data:
            sem_rgb_image = np.array(sem_rgb_image.transpose(PIL.Image.ROTATE_180))
        else:
            sem_rgb_image = np.array(sem_rgb_image)
        pose_dep = self.odom_array_depth[odom_idx].data.cpu().numpy()
        pose_sem = self.odom_array_sem_rgb[odom_idx].data.cpu().numpy()

        sem_rgb_image_warped = self.compute_overlay(
            pose_dep,
            pose_sem,
            depth_img,
            sem_rgb_image,
            self.pix_depth_cam_frame,
            self.K_sem_rgb,
        )
        assert sem_rgb_image_warped.dtype == np.uint8, "sem_rgb_image_warped has to be uint8"

        # DEBUG
        if self.debug:
            import matplotlib.pyplot as plt

            f, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 5))
            ax1.imshow(depth_img)
            ax2.imshow(sem_rgb_image_warped / 255)
            ax3.imshow(sem_rgb_image)
            # ax3.imshow(depth_img)
            # ax3.imshow(sem_rgb_image_warped / 255, alpha=0.5)
            ax1.axis("off")
            ax2.axis("off")
            ax3.axis("off")
            plt.show()

        # save semantic image under the new path
        sem_rgb_filename = os.path.split(sem_rgb_filename)[1]
        sem_rgb_image_path = os.path.join(self.root, "img_warp", sem_rgb_filename)
        sem_rgb_image_warped = cv2.cvtColor(sem_rgb_image_warped, cv2.COLOR_RGB2BGR)  # convert to BGR for cv2
        assert cv2.imwrite(sem_rgb_image_path, sem_rgb_image_warped)

        return sem_rgb_image_path

    """Noise Edges helper functions"""

    def noise_edges(self):
        """
        Along the edges in the depth image, set the values to 0.
        Mimics the real-world behavior where RealSense depth cameras have difficulties along edges.
        """
        print("[INFO] Adding noise to edges in depth images ...", end=" ")
        new_depth_filename_list = []
        # create new directory
        depth_noise_edge_dir = os.path.join(self.root, "depth_noise_edges")
        os.makedirs(depth_noise_edge_dir, exist_ok=True)

        for depth_filename in self.depth_filename_list:
            depth_img = self._load_depth_image(depth_filename)
            # Perform Canny edge detection
            image = ((depth_img / depth_img.max()) * 255).astype(np.uint8)  # convert to CV_U8 format
            edges = cv2.Canny(image, self._cfg.edge_threshold, self._cfg.edge_threshold * 3)
            # Dilate the edges to extend their space
            kernel = np.ones(self._cfg.extend_kernel_size, np.uint8)
            dilated_edges = cv2.dilate(edges, kernel, iterations=1)
            # Erode the edges to refine their shape
            eroded_edges = cv2.erode(dilated_edges, kernel, iterations=1)
            # modify depth image
            depth_img[eroded_edges == 255] = 0.0
            # save depth image
            depth_img = (depth_img * self._cfg.depth_scale).astype("uint16")
            if depth_filename.endswith(".png"):
                assert cv2.imwrite(
                    os.path.join(depth_noise_edge_dir, os.path.split(depth_filename)[1]),
                    depth_img,
                )
            else:
                np.save(
                    os.path.join(depth_noise_edge_dir, os.path.split(depth_filename)[1]),
                    depth_img,
                )
            new_depth_filename_list.append(os.path.join(depth_noise_edge_dir, os.path.split(depth_filename)[1]))

        self.depth_filename_list = new_depth_filename_list
        print("Done!")
        return

    """ Cleanup Script for files generated by this class"""

    def cleanup(self):
        print(
            ("[INFO] Cleaning up for environment" f" {os.path.split(self.root)[1]} ..."),
            end=" ",
        )
        # remove semantic_warp directory
        if os.path.isdir(os.path.join(self.root, "img_warp")):
            shutil.rmtree(os.path.join(self.root, "img_warp"))
        # remove depth_noise_edges directory
        if os.path.isdir(os.path.join(self.root, "depth_noise_edges")):
            shutil.rmtree(os.path.join(self.root, "depth_noise_edges"))
        print("Done!")
        return


# EoF
