# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# python
import os
from dataclasses import dataclass
from typing import Optional

import yaml


class Loader(yaml.SafeLoader):
    pass


def construct_GeneralCostMapConfig(loader, node):
    return GeneralCostMapConfig(**loader.construct_mapping(node))


Loader.add_constructor(
    "tag:yaml.org,2002:python/object:viplanner.config.costmap_cfg.GeneralCostMapConfig",
    construct_GeneralCostMapConfig,
)


def construct_ReconstructionCfg(loader, node):
    return ReconstructionCfg(**loader.construct_mapping(node))


Loader.add_constructor(
    "tag:yaml.org,2002:python/object:viplanner.config.costmap_cfg.ReconstructionCfg",
    construct_ReconstructionCfg,
)


def construct_SemCostMapConfig(loader, node):
    return SemCostMapConfig(**loader.construct_mapping(node))


Loader.add_constructor(
    "tag:yaml.org,2002:python/object:viplanner.config.costmap_cfg.SemCostMapConfig",
    construct_SemCostMapConfig,
)


def construct_TsdfCostMapConfig(loader, node):
    return TsdfCostMapConfig(**loader.construct_mapping(node))


Loader.add_constructor(
    "tag:yaml.org,2002:python/object:viplanner.config.costmap_cfg.TsdfCostMapConfig",
    construct_TsdfCostMapConfig,
)


@dataclass
class ReconstructionCfg:
    """
    Arguments for 3D reconstruction using depth maps
    """

    # directory where the environment with the depth (and semantic) images is located
    data_dir: str = "${USER_PATH_TO_DATA}"  # e.g. "<path-to-repo>/omniverse/extension/omni.viplanner/data/warehouse"
    # environment name
    env: str = "warehouse_new"  # has to be adjusted
    # image suffix
    depth_suffix = ""
    sem_suffix = ""
    # higher resolution depth images available for reconstruction  (meaning that the depth images are also taked by the semantic camera)
    high_res_depth: bool = False

    # reconstruction parameters
    voxel_size: float = 0.05  # [m] 0.05 for matterport 0.1 for carla
    start_idx: int = 0  # start index for reconstruction
    max_images: Optional[int] = 1000  # maximum number of images to reconstruct, if None, all images are used
    depth_scale: float = 1000  # depth scale factor
    # semantic reconstruction
    semantics: bool = True

    # speed vs. memory trade-off parameters
    point_cloud_batch_size: int = (
        200  # 3d points of nbr images added to point cloud at once (higher values use more memory but faster)
    )

    """ Internal functions """

    def get_data_path(self) -> str:
        return os.path.join(self.data_dir, self.env)

    def get_out_path(self) -> str:
        return os.path.join(self.out_dir, self.env)


@dataclass
class SemCostMapConfig:
    """Configuration for the semantic cost map"""

    # point-cloud filter parameters
    ground_height: Optional[float] = -0.5  # None for matterport  -0.5 for carla  -1.0 for nomoko
    robot_height: float = 0.70
    robot_height_factor: float = 3.0
    nb_neighbors: int = 100
    std_ratio: float = 2.0  # keep high, otherwise ground will be removed
    downsample: bool = False
    # smoothing
    nb_neigh: int = 15
    change_decimal: int = 3
    conv_crit: float = (
        0.45  # ration of points that have to change by at least the #change_decimal decimal value to converge
    )
    nb_tasks: Optional[int] = 10  # number of tasks for parallel processing, if None, all available cores are used
    sigma_smooth: float = 2.5
    max_iterations: int = 1
    # obstacle threshold  (multiplied with highest loss value defined for a semantic class)
    obstacle_threshold: float = 0.5  # 0.5/ 0.6 for matterport, 0.8 for carla
    # negative reward for space with smallest cost (introduces a gradient in area with smallest loss value, steering towards center)
    # NOTE: at the end cost map is elevated by that amount to ensure that the smallest cost is 0
    negative_reward: float = 0.5
    # loss values rounded up to decimal #round_decimal_traversable equal to 0.0 are selected and the traversable gradient is determined based on them
    round_decimal_traversable: int = 2
    # compute height map
    compute_height_map: bool = False  # false for matterport, true for carla and nomoko


@dataclass
class TsdfCostMapConfig:
    """Configuration for the tsdf cost map"""

    # offset of the point cloud
    offset_z: float = 0.0
    # filter parameters
    ground_height: float = 0.35
    robot_height: float = 0.70
    robot_height_factor: float = 2.0
    nb_neighbors: int = 50
    std_ratio: float = 0.2
    filter_outliers: bool = True
    # dilation parameters
    sigma_expand: float = 2.0
    obstacle_threshold: float = 0.01
    free_space_threshold: float = 0.5


@dataclass
class GeneralCostMapConfig:
    """General Cost Map Configuration"""

    # path to point cloud
    root_path: str = "<path-to-data>/<env-name>"
    ply_file: str = "cloud.ply"
    # resolution of the cost map
    resolution: float = 0.04  # [m]  (0.04 for matterport, 0.1 for carla)
    # map parameters
    clear_dist: float = 1.0  # cost map expansion over the point cloud space (prevent paths to go out of the map)
    # smoothing parameters
    sigma_smooth: float = 3.0
    # cost map expansion
    x_min: Optional[float] = None
    # [m] if None, the minimum of the point cloud is used None (carla town01:  -8.05   matterport: None)
    y_min: Optional[float] = None
    # [m] if None, the minimum of the point cloud is used None (carla town01:  -8.05   matterport: None)
    x_max: Optional[float] = None
    # [m] if None, the maximum of the point cloud is used None (carla town01:  346.22  matterport: None)
    y_max: Optional[float] = None
    # [m] if None, the maximum of the point cloud is used None (carla town01:  336.65  matterport: None)


@dataclass
class CostMapConfig:
    """General Cost Map Configuration"""

    # cost map domains
    semantics: bool = True
    geometry: bool = False

    # name
    map_name: str = "cost_map_sem"

    # general cost map configuration
    general: GeneralCostMapConfig = GeneralCostMapConfig()

    # individual cost map configurations
    sem_cost_map: SemCostMapConfig = SemCostMapConfig()
    tsdf_cost_map: TsdfCostMapConfig = TsdfCostMapConfig()

    # visualize cost map
    visualize: bool = True

    # FILLED BY CODE -> DO NOT CHANGE ###
    x_start: float = None
    y_start: float = None


# EoF
