# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# python
import os
from dataclasses import MISSING

import numpy as np

# orbit-assets
from omni.isaac.assets import ASSETS_RESOURCES_DIR

# omni-isaac-orbit
from omni.isaac.orbit.utils.configclass import configclass


@configclass
class TwistControllerCfg:
    lookAheadDistance: float = 0.5
    minPointsWithinLookAhead: int = 3
    two_way_drive: bool = False
    switch_time_threshold: float = 1.0
    maxSpeed: float = 0.5
    maxAccel: float = 2.5 / 100.0  # 2.5 / 100
    joyYaw: float = 1.0
    yawRateGain: float = 7.0  # 3.5
    stopYawRateGain: float = 7.0  # 3.5
    maxYawRate: float = 90.0 * np.pi / 360
    dirDiffThre: float = 0.7
    stopDisThre: float = 0.4
    slowDwnDisThre: float = 0.3
    slowRate1: float = 0.25
    slowRate2: float = 0.5
    noRotAtGoal: bool = True
    autonomyMode: bool = False

    # extra functionality
    stuck_time_threshold: float = 2.0
    stuck_avoidance_duration: int = 30  # number of steps stuck avoidance twist will be executed


@configclass
class VIPlannerCfg:
    """Configuration for the ROS publishing for Waypoint Follower and VIPlanner (ROS)."""

    viplanner: bool = True
    """Use VIPlanner or iPlanner"""
    model_dir: str = os.path.join(
        ASSETS_RESOURCES_DIR,
        "vip_models/plannernet_env2azQ1b91cZZ_new_colorspace_ep100_inputDepSem_costSem_optimSGD_new_colorspace_sharpend_indoor",
    )
    """Path to the model directory (expects a model.pt and model.yaml file in the directory)."""
    sem_origin: str = (
        "isaac"  # "isaac" or "callback (in case the semantics cannot be generated in isaac e.g. matterport)"
    )
    """Data source of the environment --> important for color mapping of the semantic segmentation"""
    m2f_model_dir: str = os.path.join(ASSETS_RESOURCES_DIR, "vip_models", "m2f_models")
    """Path to mask2former model for direct RGB input (directly including config file and model weights)"""
    planner_freq: int = 20
    """Frequency of the planner in Hz."""
    goal_prim: str = "/World/waypoint"
    """The prim path of the cube in the USD stage"""
    cam_path: dict = {
        "ANYmal_C": {"rgb": "front_depth", "depth": "front_depth"},
        "ANYmal_D": {"rgb": "front_rgb", "depth": "front_depth"},
        "mount": {"rgb": "viplanner_rgb", "depth": "viplanner_depth"},
    }
    use_mount_cam: bool = False
    """Camera Path names as defined in config.sensor_cfg that should be used to render the network inputs"""
    rgb_debug: bool = False
    """Save RGB images together with depth (mainly for debug reasons)."""
    num_points_network_return: int = 51
    """Number of points the network returns."""
    conv_dist: float = 0.5
    """Distance to the goal to save that it has been reached successfully"""
    obs_loss_threshold: float = 0.3
    """Obstacle threshold to consider a path as successful"""
    path_topic: str = "/path"
    """Topic to publish the path."""
    status_topic: str = "/status"
    """Topic to publish the planner status."""
    save_images: bool = False
    """Save depth images to disk."""
    ros_pub: bool = False
    """Publish the path and status to ROS (only needed for VIPlanner ROS)."""
    look_back_factor: int = 15
    """Look back factor for the path."""
    fear_threshold: float = 0.5

    # twist controller config
    twist_controller_cfg: TwistControllerCfg = TwistControllerCfg()


# EoF
