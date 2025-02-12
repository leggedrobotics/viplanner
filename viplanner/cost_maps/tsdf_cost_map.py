# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

# python
import os

import numpy as np
import open3d as o3d
from scipy import ndimage
from scipy.ndimage import gaussian_filter

# imperative-cost-map
from viplanner.config import GeneralCostMapConfig, TsdfCostMapConfig


class TsdfCostMap:
    """
    Cost Map based on geometric information
    """

    def __init__(self, cfg_general: GeneralCostMapConfig, cfg_tsdf: TsdfCostMapConfig):
        self._cfg_general = cfg_general
        self._cfg_tsdf = cfg_tsdf
        # set init flag
        self.is_map_ready = False
        # init point clouds
        self.obs_pcd = o3d.geometry.PointCloud()
        self.free_pcd = o3d.geometry.PointCloud()
        return

    def UpdatePCDwithPs(self, P_obs, P_free, is_downsample=False):
        self.obs_pcd.points = o3d.utility.Vector3dVector(P_obs)
        self.free_pcd.points = o3d.utility.Vector3dVector(P_free)
        if is_downsample:
            self.obs_pcd = self.obs_pcd.voxel_down_sample(self._cfg_general.resolution)
            self.free_pcd = self.free_pcd.voxel_down_sample(self._cfg_general.resolution * 0.85)

        self.obs_points = np.asarray(self.obs_pcd.points)
        self.free_points = np.asarray(self.free_pcd.points)
        print("number of obs points: %d, free points: %d" % (self.obs_points.shape[0], self.free_points.shape[0]))

    def ReadPointFromFile(self):
        pcd_load = o3d.io.read_point_cloud(os.path.join(self._cfg_general.root_path, self._cfg_general.ply_file))
        obs_p, free_p = self.TerrainAnalysis(np.asarray(pcd_load.points))
        self.UpdatePCDwithPs(obs_p, free_p, is_downsample=True)
        if self._cfg_tsdf.filter_outliers:
            obs_p = self.FilterCloud(self.obs_points)
            free_p = self.FilterCloud(self.free_points, outlier_filter=False)
            self.UpdatePCDwithPs(obs_p, free_p)
        self.UpdateMapParams()
        return

    def TerrainAnalysis(self, input_points):
        obs_points = np.zeros(input_points.shape)
        free_poins = np.zeros(input_points.shape)
        obs_idx = 0
        free_idx = 0
        # naive approach with z values
        for p in input_points:
            p_height = p[2] + self._cfg_tsdf.offset_z
            if (p_height > self._cfg_tsdf.ground_height * 1.2) and (
                p_height < self._cfg_tsdf.robot_height * self._cfg_tsdf.robot_height_factor
            ):  # remove ground and ceiling
                obs_points[obs_idx, :] = p
                obs_idx = obs_idx + 1
            elif p_height < self._cfg_tsdf.ground_height and p_height > -self._cfg_tsdf.ground_height:
                free_poins[free_idx, :] = p
                free_idx = free_idx + 1
        return obs_points[:obs_idx, :], free_poins[:free_idx, :]

    def UpdateMapParams(self):
        if self.obs_points.shape[0] == 0:
            print("No points received.")
            return
        max_x, max_y, _ = np.amax(self.obs_points, axis=0) + self._cfg_general.clear_dist
        min_x, min_y, _ = np.amin(self.obs_points, axis=0) - self._cfg_general.clear_dist

        self.num_x = np.ceil((max_x - min_x) / self._cfg_general.resolution / 10).astype(int) * 10
        self.num_y = np.ceil((max_y - min_y) / self._cfg_general.resolution / 10).astype(int) * 10
        self.start_x = (max_x + min_x) / 2.0 - self.num_x / 2.0 * self._cfg_general.resolution
        self.start_y = (max_y + min_y) / 2.0 - self.num_y / 2.0 * self._cfg_general.resolution

        print("tsdf map initialized, with size: %d, %d" % (self.num_x, self.num_y))
        self.is_map_ready = True

    def CreateTSDFMap(self):
        if not self.is_map_ready:
            raise ValueError("create tsdf map fails, no points received.")
        free_map = np.ones([self.num_x, self.num_y])
        obs_map = np.zeros([self.num_x, self.num_y])
        free_I = self.IndexArrayOfPs(self.free_points)
        obs_I = self.IndexArrayOfPs(self.obs_points)
        # create free place map
        for i in obs_I:
            obs_map[i[0], i[1]] = 1.0
        obs_map = gaussian_filter(obs_map, sigma=self._cfg_tsdf.sigma_expand)
        for i in free_I:
            if i[0] < self.num_x and i[1] < self.num_y:
                free_map[i[0], i[1]] = 0
        free_map = gaussian_filter(free_map, sigma=self._cfg_tsdf.sigma_expand)
        free_map[free_map < self._cfg_tsdf.free_space_threshold] = 0
        # assign obstacles
        free_map[obs_map > self._cfg_tsdf.obstacle_threshold] = 1.0

        print("occupancy map generation completed.")
        # Distance Transform
        tsdf_array = ndimage.distance_transform_edt(free_map)
        tsdf_array[tsdf_array > 0.0] = np.log(tsdf_array[tsdf_array > 0.0] + math.e)
        tsdf_array = gaussian_filter(tsdf_array, sigma=self._cfg_general.sigma_smooth)

        viz_points = np.concatenate((self.obs_points, self.free_points), axis=0)

        # TODO: Using true terrain analysis module
        ground_array = np.ones([self.num_x, self.num_y]) * 0.0
        return [tsdf_array, viz_points, ground_array], [
            float(self.start_x),
            float(self.start_y),
        ]

    def IndexArrayOfPs(self, points):
        indexes = points[:, :2] - np.array([self.start_x, self.start_y])
        indexes = (np.round(indexes / self._cfg_general.resolution)).astype(int)
        return indexes

    def FilterCloud(self, points, outlier_filter=True):
        # crop points
        if any(
            [
                self._cfg_general.x_max,
                self._cfg_general.x_min,
                self._cfg_general.y_max,
                self._cfg_general.y_min,
            ]
        ):
            points_x_idx_upper = (
                (points[:, 0] < self._cfg_general.x_max)
                if self._cfg_general.x_max is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points_x_idx_lower = (
                (points[:, 0] > self._cfg_general.x_min)
                if self._cfg_general.x_min is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points_y_idx_upper = (
                (points[:, 1] < self._cfg_general.y_max)
                if self._cfg_general.y_max is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points_y_idx_lower = (
                (points[:, 1] > self._cfg_general.y_min)
                if self._cfg_general.y_min is not None
                else np.ones(points.shape[0], dtype=bool)
            )
            points = points[
                np.vstack(
                    (
                        points_x_idx_lower,
                        points_x_idx_upper,
                        points_y_idx_upper,
                        points_y_idx_lower,
                    )
                ).all(axis=0)
            ]

        if outlier_filter:
            # Filter outlier in points
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            cl, _ = pcd.remove_statistical_outlier(
                nb_neighbors=self._cfg_tsdf.nb_neighbors,
                std_ratio=self._cfg_tsdf.std_ratio,
            )
            points = np.asarray(cl.points)

        return points

    def VizCloud(self, pcd):
        o3d.visualization.draw_geometries([pcd])  # visualize point cloud


# EoF
