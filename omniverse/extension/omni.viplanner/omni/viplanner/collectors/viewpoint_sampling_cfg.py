# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

from .terrain_analysis_cfg import TerrainAnalysisCfg


@configclass
class ViewpointSamplingCfg:
    """Configuration for the viewpoint sampling."""

    terrain_analysis: TerrainAnalysisCfg = TerrainAnalysisCfg(raycaster_sensor="camera_0")
    """Name of the camera object in the scene definition used for the terrain analysis."""

    # dict of cameras and corresponding annotators
    cameras: dict[str, str] = {
        "camera_0": "distance_to_image_plane",
        "camera_1": "semantic_segmentation",
    }
    """Dict of cameras and corresponding annotators to use for the viewpoint sampling."""
    depth_scale: float = 1000.0
    """Scaling factor for the depth values."""

    # sampling
    sample_points: int = 10000
    """Number of random points to sample."""
    x_angle_range: tuple[float, float] = (-2.5, 2.5)
    y_angle_range: tuple[float, float] = (-2, 5)  # negative angle means in isaac convention: look down
    """Range of the x and y angle of the camera (in degrees), will be randomly selected according to a uniform distribution"""
    height: float = 0.5
    """Height to use for the random points."""

    # SAVING
    save_path: str | None = None
    """Directory to save the viewpoint samples, camera intrinsics and rendered images to.

    If None, the directory is the same as the one of the obj file. Default is None."""

    # debug
    debug_viz: bool = True
    """Whether to visualize the sampled points and orientations."""
