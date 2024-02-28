# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# python
import os
from dataclasses import dataclass
from typing import Optional, Tuple

from omni.isaac.carla.config import DATA_DIR

# isaac-orbit
from omni.isaac.orbit.sensors.camera import PinholeCameraCfg


@dataclass
class CarlaExplorerConfig:
    """Configuration for the CarlaMap class."""

    # coverage parameters
    points_per_m2: float = 0.5
    obs_loss_threshold: float = 0.8
    max_cam_recordings: Optional[int] = 10000  # if None, not limitation is applied
    # indoor filter (for outdoor maps filter inside of buildings as traversable, for inside maps set to False)
    indoor_filter: bool = True
    carla_filter: Optional[str] = os.path.join(DATA_DIR, "town01", "area_filter_cfg.yml")
    # nomoko model
    nomoko_model: bool = False
    # are limiter --> only select area within the defined prim names (e.g. "Road_SideWalk")
    space_limiter: Optional[str] = "Road_Sidewalk"  # carla: "Road_Sidewalk"  nomoko None  park: MergedRoad05
    # robot height
    robot_height = 0.7  # m
    # depth camera
    camera_cfg_depth: PinholeCameraCfg = PinholeCameraCfg(
        sensor_tick=0,
        height=480,
        width=848,
        data_types=["distance_to_image_plane"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=1.93, clipping_range=(0.01, 1.0e5), horizontal_aperture=3.8
        ),
    )
    camera_intrinsics_depth: Optional[Tuple[float]] = None
    # ANYmal D/C realsense:         (423.54608, 0.0, 427.69815, 0.0, 423.54608, 240.17773, 0.0, 0.0, 1.0)
    # RealSense D455:               (430.31607, 0.0, 428.28408, 0.0, 430.31607, 244.00695, 0.0, 0.0, 1.0)
    # ANYmal D wide_angle_camera: 1.0 <-> ANYmal C realsense: 1.93 <-> RealSense D455: 1.93
    camera_prim_depth: str = "/World/CameraSensor_depth"
    # semantic camera
    camera_cfg_sem: PinholeCameraCfg = PinholeCameraCfg(
        sensor_tick=0,
        height=720,  # 480,  # 1080
        width=1280,  # 848,  # 1440
        data_types=["rgb", "semantic_segmentation"],
        usd_params=PinholeCameraCfg.UsdCameraCfg(
            focal_length=1.93, clipping_range=(0.01, 1.0e5), horizontal_aperture=3.8
        ),
    )
    # ANYmal D wide_angle_camera: (1440, 1080)  <-> ANYmal C realsense (848, 480) <-> RealSense D455 (1280, 720)
    # ANYmal D wide_angle_camera: 1.93 <-> ANYmal C realsense: 1.93 <-> RealSense D455: 1.93
    camera_intrinsics_sem: Optional[Tuple[float]] = None
    # ANYmal D wide_angle_camera:   (575.60504, 0.0, 745.73121, 0.0, 578.56484, 519.52070, 0.0, 0.0, 1.0)
    # ANYmal C realsense:           (423.54608, 0.0, 427.69815, 0.0, 423.54608, 240.17773, 0.0, 0.0, 1.0)
    # RealSense D455:               (644.15496, 0.0, 639.53125, 0.0, 643.49212, 366.30880, 0.0, 0.0, 1.0)
    camera_prim_sem: str = "/World/CameraSensor_sem"
    x_angle_range: Tuple[float, float] = (-5, 5)  # downtilt angle of the camera in degree
    y_angle_range: Tuple[float, float] = (
        -2,
        5,
    )  # downtilt angle of the camera in degree  --> isaac convention, positive is downwards
    # image suffix
    depth_suffix = "_cam0"
    sem_suffix = "_cam1"
    # transformation from depth (src) to semantic camera (target)
    tf_pos: tuple = (0.0, 0.0, 0.0)  # (translation in depth frame)
    # ANYmal D: (-0.002, 0.025, 0.042)  <-> ANYmal C and RealSense D455: (0.0, 0.0, 0.0)
    tf_quat: tuple = (0.0, 0.0, 0.0, 1.0)  # xyzw quaternion format (rotation in depth frame)
    # ANYmal D: (0.001, 0.137, -0.000, 0.991)  <-> ANYmal C and RealSense D455: (0.0, 0.0, 0.0, 1.0)
    tf_quat_convention: str = "roll-pitch-yaw"  # or "isaac"
    # NOTE: if the quat follows the roll-pitch-yaw convention, i.e. x-forward, y-right, z-down, will be converted to the isaac convention
    # high resolution depth for reconstruction (in city environment can otherwise lead to artifacts)
    # will now also take the depth image of the rgb camera and use its depth images for reconstruction
    high_res_depth: bool = False
    # output_dir
    output_root: Optional[str] = None  # if None, output dir is stored under root_dir
    output_dir_name: str = "town01"
    ros_p_mat: bool = True  # save intrinsic matrix in ros P-matrix format
    depth_scale: float = 1000.0  # scale depth values before saving s.t. mm resolution can be achieved

    # add more people to the scene
    nb_more_people: Optional[int] = 1200  # if None, no people are added
    random_seed: Optional[int] = 42  # if None, no seed is set

    @property
    def output_dir(self) -> str:
        if self.output_root is not None:
            return os.path.join(self.output_root, self.output_dir_name)
        else:
            return os.path.join(self.root_path, self.output_dir_name)
