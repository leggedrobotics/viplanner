# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math
import multiprocessing as mp

# python
import os
from functools import partial
from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import scipy

# imperative-cost-map
from viplanner.config import (
    OBSTACLE_LOSS,
    GeneralCostMapConfig,
    SemCostMapConfig,
    VIPlannerSemMetaHandler,
)


class SemCostMap:
    """
    Cost Map based on semantic information
    """

    def __init__(
        self,
        cfg_general: GeneralCostMapConfig,
        cfg: SemCostMapConfig,
        visualize: bool = True,
    ):
        self._cfg_general = cfg_general
        self._cfg_sem = cfg
        self.visualize = visualize

        # init VIPlanner Semantic Class Meta Handler
        self.sem_meta = VIPlannerSemMetaHandler()

        # cost map init parameters
        self.pcd: o3d.geometry.PointCloud = None
        self.pcd_filtered: o3d.geometry.PointCloud = None
        self.height_map: np.ndarray = None
        self._num_x: int = 0.0
        self._num_y: int = 0.0
        self._start_x: float = 0.0
        self._start_y: float = 0.0
        self._init_done: bool = False

        # cost map
        self.grid_cell_loss: np.ndarray = None
        return

    def pcd_init(self) -> None:
        # load pcd and filter it
        print("COST-MAP INIT START")
        print("start loading and filtering point cloud from:" f" {self._cfg_general.ply_file}")
        pc_path = os.path.join(self._cfg_general.root_path, self._cfg_general.ply_file)
        assert os.path.exists(pc_path), f"point cloud file does not exist: {pc_path}"
        self.pcd = o3d.io.read_point_cloud(pc_path)

        # filter for x and y coordinates
        if any(
            [
                self._cfg_general.x_max,
                self._cfg_general.x_min,
                self._cfg_general.y_max,
                self._cfg_general.y_min,
            ]
        ):
            pts = np.asarray(self.pcd.points)
            pts_x_idx_upper = (
                (pts[:, 0] < self._cfg_general.x_max)
                if self._cfg_general.x_max is not None
                else np.ones(pts.shape[0], dtype=bool)
            )
            pts_x_idx_lower = (
                (pts[:, 0] > self._cfg_general.x_min)
                if self._cfg_general.x_min is not None
                else np.ones(pts.shape[0], dtype=bool)
            )
            pts_y_idx_upper = (
                (pts[:, 1] < self._cfg_general.y_max)
                if self._cfg_general.y_max is not None
                else np.ones(pts.shape[0], dtype=bool)
            )
            pts_y_idx_lower = (
                (pts[:, 1] > self._cfg_general.y_min)
                if self._cfg_general.y_min is not None
                else np.ones(pts.shape[0], dtype=bool)
            )
            self.pcd = self.pcd.select_by_index(
                np.where(
                    np.vstack(
                        (
                            pts_x_idx_lower,
                            pts_x_idx_upper,
                            pts_y_idx_upper,
                            pts_y_idx_lower,
                        )
                    ).all(axis=0)
                )[0]
            )

        # set parameters
        self._set_map_parameters(self.pcd)

        # get ground height map
        if self._cfg_sem.compute_height_map:
            self.height_map = self._pcd_ground_height_map(self.pcd)
        else:
            self.height_map = np.zeros((self._num_x, self._num_y))

        # filter point cloud depending on height
        self.pcd_filtered = self._pcd_filter()

        # update init flag
        self._init_done = True
        print("COST-MAP INIT DONE")
        return

    def create_costmap(self) -> Tuple[list, list]:
        assert self._init_done, "cost map not initialized, call pcd_init() first"
        print("COST-MAP CREATION START")

        # get the loss for each grid cell
        grid_loss = self._get_grid_loss()

        # make grid loss differentiable
        grid_loss = self._dense_grid_loss(grid_loss)

        print("COST-MAP CREATION DONE")
        return [grid_loss, self.pcd_filtered.points, self.height_map], [
            float(self._start_x),
            float(self._start_y),
        ]

    """Helper functions"""

    def _pcd_ground_height_map(self, pcd: o3d.geometry.PointCloud) -> np.ndarray:
        "Start building height map"
        # for each grid cell, get the point with the highest z value
        pts = np.asarray(pcd.points)
        pts_grid_idx_red, pts_idx = self._get_unqiue_grid_idx(pts)

        # ground height of human constructed things (buildings, bench, etc.) should be equal to the ground height of the surrounding terrain/ street
        # --> classify the selected points and change depending on the class
        # get colors
        color = np.asarray(pcd.colors)[pts_idx] * 255.0
        # pts to class idx array
        pts_ground = np.zeros(color.shape[0], dtype=bool)
        # assign each point to a class
        color = color.astype(int)
        for class_name, class_color in self.sem_meta.class_color.items():
            pts_idx_of_class = (color == class_color).all(axis=1).nonzero()[0]
            pts_ground[pts_idx_of_class] = self.sem_meta.class_ground[class_name]

        # filter outliers
        pts_ground_idx = pts_idx[pts_ground]
        if False:
            pcd_ground = pcd.select_by_index(pts_ground_idx)
            _, ind = pcd_ground.remove_radius_outlier(nb_points=5, radius=5 * self._cfg_general.resolution)
            pts_ground_idx = pts_ground_idx[ind]
            pts_ground_red = np.zeros(pts_grid_idx_red.shape[0], dtype=bool)
            pts_ground_red[np.where(pts_ground)[0][ind]] = True
            pts_ground = pts_ground_red

        # fit kdtree to the points on the ground and assign ground height to all other points based on the nearest neighbor
        pts_ground_location = pts[pts_ground_idx]
        ground_kdtree = scipy.spatial.KDTree(pts_ground_location)
        _, non_ground_neighbor_idx = ground_kdtree.query(pts[pts_idx[~pts_ground]], workers=-1)

        # init height map and assign ground height to all points on the ground
        height_pts_ground = np.zeros(pts_grid_idx_red.shape[0])
        height_pts_ground[pts_ground] = pts_ground_location[:, 2]
        height_pts_ground[~pts_ground] = pts_ground_location[non_ground_neighbor_idx, 2]

        # fill the holes
        height_map = np.full((self._num_x, self._num_y), np.nan)
        height_map[pts_grid_idx_red[:, 0], pts_grid_idx_red[:, 1]] = height_pts_ground
        hole_idx = np.vstack(np.where(np.isnan(height_map))).T

        kdtree_grid = scipy.spatial.KDTree(pts_grid_idx_red)
        distance, neighbor_idx = kdtree_grid.query(hole_idx, k=3, workers=-1)
        weights = distance / np.sum(distance, axis=1)[:, None]
        height_map[hole_idx[:, 0], hole_idx[:, 1]] = np.sum(height_pts_ground[neighbor_idx] * weights, axis=1)

        if self.visualize:
            # visualize the height map
            plt.imshow(height_map)
            plt.colorbar()
            plt.show()

        print("Done building height map")
        return height_map

    def _pcd_filter(self) -> o3d.geometry.PointCloud:
        """remove points above the robot height, under the ground and filter for outliers"""
        pts = np.asarray(self.pcd.points)

        if self.height_map is not None:
            pts_grid_idx = (
                np.round((pts[:, :2] - np.array([self._start_x, self._start_y])) / self._cfg_general.resolution)
            ).astype(int)
            pts[:, 2] -= self.height_map[pts_grid_idx[:, 0], pts_grid_idx[:, 1]]

        pts_ceil_idx = pts[:, 2] < self._cfg_sem.robot_height * self._cfg_sem.robot_height_factor
        pts_ground_idx = (
            pts[:, 2] > self._cfg_sem.ground_height
            if self._cfg_sem.ground_height is not None
            else np.ones(pts.shape[0], dtype=bool)
        )
        pcd_height_filtered = self.pcd.select_by_index(
            np.where(np.vstack((pts_ceil_idx, pts_ground_idx)).all(axis=0))[0]
        )

        # downsampling
        if self._cfg_sem.downsample:
            pcd_height_filtered = pcd_height_filtered.voxel_down_sample(self._cfg_general.resolution)
            print("Voxel Downsampling applied")

        # remove statistical outliers
        pcd_filtered, _ = pcd_height_filtered.remove_statistical_outlier(
            nb_neighbors=self._cfg_sem.nb_neighbors,
            std_ratio=self._cfg_sem.std_ratio,
        )

        return pcd_filtered

    def _set_map_parameters(self, pcd: o3d.geometry.PointCloud) -> None:
        """Define the size and start position of the cost map"""
        pts = np.asarray(pcd.points)
        assert pts.shape[0] > 0, "No points received."

        # get max and minimum of cost map
        max_x, max_y, _ = np.amax(pts, axis=0) + self._cfg_general.clear_dist
        min_x, min_y, _ = np.amin(pts, axis=0) - self._cfg_general.clear_dist

        prev_param = (
            self._num_x,
            self._num_y,
            round(self._start_x, 3),
            round(self._start_y, 3),
        )
        self._num_x = np.ceil((max_x - min_x) / self._cfg_general.resolution / 10).astype(int) * 10
        self._num_y = np.ceil((max_y - min_y) / self._cfg_general.resolution / 10).astype(int) * 10
        self._start_x = (max_x + min_x) / 2.0 - self._num_x / 2.0 * self._cfg_general.resolution
        self._start_y = (max_y + min_y) / 2.0 - self._num_y / 2.0 * self._cfg_general.resolution

        print(f"cost map size set to: {self._num_x} x {self._num_y}")
        if prev_param != (
            self._num_x,
            self._num_y,
            round(self._start_x, 3),
            round(self._start_y, 3),
        ):
            print("Map parameters changed!")
            return True

        return False

    def _class_mapping(self) -> np.ndarray:
        # get colors
        color = np.asarray(self.pcd_filtered.colors) * 255.0

        # pts to class idx array
        pts_class_idx = np.ones(color.shape[0], dtype=int) * -1

        # assign each point to a class
        color = color.astype(int)
        for class_idx, class_color in enumerate(self.sem_meta.colors):
            pts_idx_of_class = (color == class_color).all(axis=1).nonzero()[0]
            pts_class_idx[pts_idx_of_class] = class_idx

        # identify points with unknown classes --> remove from point cloud
        known_idx = np.where(pts_class_idx != -1)[0]
        self.pcd_filtered = self.pcd_filtered.select_by_index(known_idx)
        print(f"Class of {len(known_idx)} points identified" f" ({len(known_idx) / len(color)} %).")

        return pts_class_idx[known_idx]

    @staticmethod
    def _smoother(
        pts_idx: np.ndarray,
        pts_grid: np.ndarray,
        pts_loss: np.ndarray,
        conv_crit: float,
        nb_neigh: int,
        change_decimal: int,
        max_iterations: int,
    ) -> np.ndarray:
        # get grid idx for each point
        print(f"Process {mp.current_process().name} started")

        lock.acquire()  # do not access the same memort twice
        pts_loss_local = pts_loss[pts_idx].copy()
        pts_grid_local = pts_grid[pts_idx].copy()
        lock.release()

        print(f"Process {mp.current_process().name} data loaded")

        # fit kd-tree to available points
        kd_tree = scipy.spatial.KDTree(pts_grid_local)
        pt_dist, pt_neigh_idx = kd_tree.query(pts_grid_local, k=nb_neigh + 1)
        pt_dist = pt_dist[:, 1:]  # filter the point itself
        pt_neigh_idx = pt_neigh_idx[:, 1:]  # filter the point itself

        # turn distance into weight
        # pt_dist_weighted = pt_dist * np.linspace(1, 0.01, nb_neigh)
        pt_dist_inv = 1.0 / pt_dist
        pt_dist_inv[
            ~np.isfinite(pt_dist_inv)
        ] = 0.0  # set inf to 0 (inf or nan values when closest point at the same position)
        pt_weights = scipy.special.softmax(pt_dist_inv, axis=1)

        # smooth losses
        counter = 0
        pts_loss_smooth = pts_loss_local.copy()
        while counter < max_iterations:
            counter += 1
            pts_loss_smooth = np.sum(pts_loss_smooth[pt_neigh_idx] * pt_weights, axis=1)

            conv_rate = (
                np.sum(np.round(pts_loss_smooth, change_decimal) != np.round(pts_loss_local, change_decimal))
                / pts_loss_local.shape[0]
            )

            if conv_rate > conv_crit:
                print(
                    f"Process {mp.current_process().name} converged with"
                    f" {np.round(conv_rate * 100, decimals=2)} % of changed"
                    f" points after {counter} iterations."
                )
                break

        return pts_loss_smooth

    @staticmethod
    def _smoother_init(l_local: mp.Lock) -> None:
        global lock
        lock = l_local
        return

    def _get_grid_loss(self) -> np.ndarray:
        """convert points to grid"""
        # get class mapping --> execute first because pcd are filtered
        class_idx = self._class_mapping()

        # update map parameters --> has to be done after mapping because last step where points are removed
        changed = self._set_map_parameters(self.pcd_filtered)
        if changed and self._cfg_sem.compute_height_map:
            print("Recompute heightmap map due to changed parameters")
            self.height_map = self._pcd_ground_height_map(self.pcd_filtered)
        elif changed:
            self.height_map = np.zeros((self._num_x, self._num_y))

        # get points
        pts = np.asarray(self.pcd_filtered.points)
        pts_grid = (pts[:, :2] - np.array([self._start_x, self._start_y])) / self._cfg_general.resolution

        # get loss for each point
        pts_loss = np.zeros(class_idx.shape[0])
        for sem_class in range(len(self.sem_meta.losses)):
            pts_loss[class_idx == sem_class] = self.sem_meta.losses[sem_class]

        # split task index
        num_tasks = self._cfg_sem.nb_tasks if self._cfg_sem.nb_tasks else mp.cpu_count()
        pts_task_idx = np.array_split(np.random.permutation(pts_loss.shape[0]), num_tasks)

        # create pool with lock
        lock_local = mp.Lock()
        pool = mp.pool.Pool(processes=num_tasks, initializer=self._smoother_init, initargs=(lock_local,))
        loss_array = pool.map(
            partial(
                self._smoother,
                pts_grid=pts_grid,
                pts_loss=pts_loss,
                conv_crit=self._cfg_sem.conv_crit,
                nb_neigh=self._cfg_sem.nb_neigh,
                change_decimal=self._cfg_sem.change_decimal,
                max_iterations=self._cfg_sem.max_iterations,
            ),
            pts_task_idx,
        )
        pool.close()
        pool.join()

        # reassemble loss array
        smooth_loss = np.zeros_like(pts_loss)
        for process_idx in range(num_tasks):
            smooth_loss[pts_task_idx[process_idx]] = loss_array[process_idx]

        if False:  # self.visualize:
            plt.scatter(pts[:, 0], pts[:, 1], c=smooth_loss, cmap="jet")
            plt.show()

        return smooth_loss

    def _distance_based_gradient(
        self,
        loss_level_idx: np.ndarray,
        loss_min: float,
        loss_max: float,
        log_scaling: bool,
    ) -> np.ndarray:
        grid = np.zeros((self._num_x, self._num_y))

        # distance transform
        grid[loss_level_idx] = 1
        grid = scipy.ndimage.distance_transform_edt(grid)

        # loss scaling
        if log_scaling:
            grid[grid > 0.0] = np.log(grid[grid > 0.0] + math.e)
        else:
            grid = (grid - np.min(grid)) / (np.max(grid) - np.min(grid))
            grid = grid * (loss_max - loss_min) + loss_min

        return grid[loss_level_idx]

    def _dense_grid_loss(self, smooth_loss: np.ndarray) -> None:
        # get grid idx of all classified points
        pts = np.asarray(self.pcd_filtered.points)
        pts_grid_idx_red, pts_idx = self._get_unqiue_grid_idx(pts)

        grid_loss = np.ones((self._num_x, self._num_y)) * -10
        grid_loss[pts_grid_idx_red[:, 0], pts_grid_idx_red[:, 1]] = smooth_loss[pts_idx]

        # get grid idx of all (non-) classified points
        non_classified_idx = np.where(grid_loss == -10)
        non_classified_idx = np.vstack((non_classified_idx[0], non_classified_idx[1])).T

        kdtree = scipy.spatial.KDTree(pts_grid_idx_red)
        distances, idx = kdtree.query(non_classified_idx, k=1)

        # only use points within the mesh, i.e. distance to nearest neighbor smaller than 10 cells
        within_mesh = distances < 10

        # assign each point its neighbor loss
        grid_loss[
            non_classified_idx[within_mesh, 0],
            non_classified_idx[within_mesh, 1],
        ] = grid_loss[
            pts_grid_idx_red[idx[within_mesh], 0],
            pts_grid_idx_red[idx[within_mesh], 1],
        ]

        # apply smoothing for filter missclassified points
        grid_loss[
            non_classified_idx[~within_mesh, 0],
            non_classified_idx[~within_mesh, 1],
        ] = OBSTACLE_LOSS
        grid_loss = scipy.ndimage.gaussian_filter(grid_loss, sigma=self._cfg_sem.sigma_smooth)

        # get different loss levels
        loss_levels = np.unique(self.sem_meta.losses)
        assert round(loss_levels[0], 3) == 0.0, f"Lowest loss level should be 0.0, instead found {loss_levels[0]}."
        if round(loss_levels[-1], 3) == 1.0:
            print("WARNING: Highest loss level should be 1.0, instead found" f" {loss_levels[-1]}.")

        # intended traversable area is best traversed with maximum distance to any area with higher cost
        # apply distance transform to nearest obstacle to enforce smallest loss when distance is max
        traversable_idx = np.where(
            np.round(grid_loss, decimals=self._cfg_sem.round_decimal_traversable) == loss_levels[0]
        )
        grid_loss[traversable_idx] = (
            self._distance_based_gradient(
                traversable_idx,
                loss_levels[0],
                abs(self._cfg_sem.negative_reward),
                False,
            )
            * -1
        )

        # outside of the mesh is an obstacle and all points over obstacle threshold of grid loss are obstacles
        obs_within_mesh_idx = np.where(grid_loss > self._cfg_sem.obstacle_threshold * loss_levels[-1])
        obs_idx = (
            np.hstack((obs_within_mesh_idx[0], non_classified_idx[~within_mesh, 0])),
            np.hstack((obs_within_mesh_idx[1], non_classified_idx[~within_mesh, 1])),
        )
        grid_loss[obs_idx] = self._distance_based_gradient(obs_idx, None, None, True)

        # repeat distance transform for intermediate loss levels
        for i in range(1, len(loss_levels) - 1):
            loss_level_idx = np.where(
                np.round(grid_loss, decimals=self._cfg_sem.round_decimal_traversable) == loss_levels[i]
            )
            grid_loss[loss_level_idx] = self._distance_based_gradient(
                loss_level_idx, loss_levels[i], loss_levels[i + 1], False
            )

        assert not (grid_loss == -10).any(), "There are still grid cells without a loss value."

        # elevate grid_loss to avoid negative values due to negative reward in area with smallest loss level
        if np.min(grid_loss) < 0:
            grid_loss = grid_loss + np.abs(np.min(grid_loss))

        # smooth loss again
        loss_smooth = scipy.ndimage.gaussian_filter(grid_loss, sigma=self._cfg_general.sigma_smooth)

        # plot grid classes and losses
        if self.visualize:
            fig, axs = plt.subplots(2, 2)
            axs[0, 0].set_title("grid loss")
            axs[0, 0].imshow(grid_loss, cmap="jet")
            axs[0, 1].set_title("loss smooth")
            axs[0, 1].imshow(loss_smooth, cmap="jet")
            axs[1, 0].set_title("grid loss x-grad")
            axs[1, 0].imshow(
                np.log(np.abs(scipy.ndimage.sobel(grid_loss, axis=0, mode="constant")) + math.e) - 1,
                cmap="jet",
            )
            axs[1, 1].set_title("grid loss y-grad")
            axs[1, 1].imshow(
                np.log(np.abs(scipy.ndimage.sobel(grid_loss, axis=1, mode="constant")) + math.e) - 1,
                cmap="jet",
            )
            plt.show()

        return loss_smooth

    def _get_unqiue_grid_idx(self, pts):
        """
        Will select the points that are unique in their grid position and have the highest z location
        """
        pts_grid_idx = (
            np.round((pts[:, :2] - np.array([self._start_x, self._start_y])) / self._cfg_general.resolution)
        ).astype(int)

        # convert pts_grid_idx to 1d array
        pts_grid_idx_1d = pts_grid_idx[:, 0] * self._num_x + pts_grid_idx[:, 1]

        # get index of all points mapped to the same grid location --> take highest value to avoid local minima in e.g. cars
        # following solution given at: https://stackoverflow.com/questions/30003068/how-to-get-a-list-of-all-indices-of-repeated-elements-in-a-numpy-array
        # creates an array of indices, sorted by unique element
        idx_sort = np.argsort(pts_grid_idx_1d)
        # sorts pts_grid_idx_1d so all unique elements are together
        pts_grid_idx_1d_sorted = pts_grid_idx_1d[idx_sort]
        # returns the unique values, the index of the first occurrence of a value, and the count for each element
        vals, idx_start, count = np.unique(pts_grid_idx_1d_sorted, return_counts=True, return_index=True)
        # splits the indices into separate arrays
        pts_grid_location_map = np.split(idx_sort, idx_start[1:])

        # filter for points with more than one occurrence
        pts_grid_location_map = np.array(pts_grid_location_map, dtype=object)
        pts_grid_location_map_multiple = pts_grid_location_map[count > 1]

        # get index with maximum z value for all points mapped to the same grid location
        pts_grid_location_map_multiple_idx = np.array(
            [
                pts_grid_location_map_multiple[idx][np.argmax(pts[pts_idx, 2])]
                for idx, pts_idx in enumerate(pts_grid_location_map_multiple)
            ]
        )

        # combine indices to get for every grid location the index of the point with the highest z value
        grid_idx = np.zeros(len(pts_grid_location_map), dtype=int)
        grid_idx[count > 1] = pts_grid_location_map_multiple_idx
        grid_idx[count == 1] = pts_grid_location_map[count == 1]
        pts_grid_idx_red = pts_grid_idx[grid_idx]

        return pts_grid_idx_red, grid_idx


# EoF
