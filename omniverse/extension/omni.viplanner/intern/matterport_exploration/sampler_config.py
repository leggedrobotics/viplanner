# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Config for Exploration/ Data Sampling in Matterport3D Dataset
"""

# python
from dataclasses import dataclass
from typing import Optional, Tuple

import torch


@dataclass
class SamplerCfg:
    points_per_m2: int = 20
    """Number of random points per m2 of the mesh surface area."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    """Device to use for computations."""
    height: float = 0.5
    """Height to use for the random points."""
    min_height: float = 0.2
    """Maximum height to be considered an accessible point for the robot"""
    ground_height: float = -0.1
    """Height of the ground plane"""
    min_wall_distance: float = 0.5
    """Minimum distance to a wall to be considered an accessible point for the robot"""
    x_angle_range: Tuple[float, float] = (-2.5, 2.5)
    y_angle_range: Tuple[float, float] = (-2, 5)  # negative angle means in isaac convention: look down
    # ANYmal D: (-2, 5)  <-> ANYmal C: (25, 35)
    # NOTE: the angles should follow the isaac convention, i.e. x-forward, y-left, z-up
    """Range of the x and y angle of the camera (in degrees), will be randomly selected according to a uniform distribution"""
    min_hit_rate: float = 0.8
    """Don't use a point if the hit rate is below this value"""
    min_avg_hit_distance: float = 0.5
    """Don't use a point if the max hit distance is below this value"""
    min_std_hit_distance: float = 0.5
    """Don't use a point if the std hit distance is below this value"""
    conv_rate: float = 0.9
    """Rate of faces that are covered by three different images, used to terminate the exploration"""

    # DEPTH CAMERA
    cam_depth_prim: str = "/cam_depth"
    cam_depth_resolution: Tuple[int, int] = (848, 480)  # (width, height)
    cam_depth_focal_length: float = 1.93  # in mm
    # ANYmal D wide_angle_camera: 1.0 <-> ANYmal C realsense: 1.93 <-> RealSense D455: 1.93
    cam_depth_clipping_range: Tuple[float, float] = (0.01, 1000.0)
    cam_depth_aperture: float = 3.8  # in mm
    cam_depth_intrinsics: Optional[Tuple[float]] = (430.31607, 0.0, 428.28408, 0.0, 430.31607, 244.00695, 0.0, 0.0, 1.0)
    # ANYmal C/D: (423.54608, 0.0, 427.69815, 0.0, 423.54608, 240.17773, 0.0, 0.0, 1.0)
    # RealSense D455: (430.31607, 0.0, 428.28408, 0.0, 430.31607, 244.00695, 0.0, 0.0, 1.0)
    # NOTE: either provide the aperture or the camera matrix (if both, the camera matrix will be used)
    """Depth camera configuration"""
    tf_pos: tuple = (0.0, 0.0, 0.0)  # (translation in depth frame)
    # ANYmal D: (-0.002, 0.025, 0.042)  <-> ANYmal C and RealSense D455: (0.0, 0.0, 0.0)
    tf_quat: tuple = (0.0, 0.0, 0.0, 1.0)  # xyzw quaternion format (rotation in depth frame)
    # ANYmal D: (0.001, 0.137, -0.000, 0.991)  <-> ANYmal C and RealSense D455: (0.0, 0.0, 0.0, 1.0)
    tf_quat_convention: str = "roll-pitch-yaw"  # or "isaac"
    # NOTE: if the quat follows the roll-pitch-yaw convention, i.e. x-forward, y-right, z-down, will be converted to the isaac convention
    """transformation from depth (src) to semantic camera (target)"""

    # SEMANTIC CAMERA
    cam_sem_prim: str = "/cam_sem"
    cam_sem_resolution: Tuple[int, int] = (1280, 720)
    # ANYmal D wide_angle_camera: (1440, 1080)  <-> ANYmal C realsense (848, 480) <-> RealSense D455 (1280, 720)
    cam_sem_focal_length: float = 1.93  # in mm (for ANYmal C100 - https://anymal-research.docs.anymal.com/user_manual/anymal_c100/release-23.02/documents/anymal_c_hardware_guide/main.html?highlight=wide+angle#achg-sssec-wide-angle-cameras)
    # ANYmal D wide_angle_camera: 1.93  <-> ANYmal C realsense: 1.0 <-> RealSense D455: 1.93
    cam_sem_clipping_range: Tuple[float, float] = (0.01, 1000.0)
    cam_sem_aperture: float = 3.8  # in mm
    cam_sem_intrinsics: Optional[Tuple[float]] = (644.15496, 0.0, 639.53125, 0.0, 643.49212, 366.30880, 0.0, 0.0, 1.0)
    # ANYmal D wide_angle_camera:   (575.60504, 0.0, 745.73121, 0.0, 578.56484, 519.52070, 0.0, 0.0, 1.0)
    # ANYmal C realsense:           (423.54608, 0.0, 427.69815, 0.0, 423.54608, 240.17773, 0.0, 0.0, 1.0)
    # RealSense D455:               (644.15496, 0.0, 639.53125, 0.0, 643.49212, 366.30880, 0.0, 0.0, 1.0)
    # NOTE: either provide the aperture or the camera matrix (if both, the camera matrix will be used)
    """Semantic camera configuration"""
    cam_sem_rgb: bool = True
    """Whether to record rgb images or not"""

    # SAVING
    max_images: int = 2000
    """Maximum number of images recorded"""
    save_path: str = "/home/pascal/viplanner/imperative_learning/data"
    suffix: Optional[str] = "cam_mount"
    """Path to save the data to (directly with env name will be created)"""


# EoF
