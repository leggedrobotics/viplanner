# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""

import math
from dataclasses import MISSING

from omni.isaac.orbit.managers import CommandTermCfg
from omni.isaac.orbit.utils.configclass import configclass
from typing_extensions import Literal

from .path_follower_command_generator import PathFollowerCommandGenerator


@configclass
class PathFollowerCommandGeneratorCfg(CommandTermCfg):
    class_type: PathFollowerCommandGenerator = PathFollowerCommandGenerator
    """Name of the command generator class."""

    robot_attr: str = MISSING
    """Name of the robot attribute from the environment."""

    path_frame: Literal["world", "robot"] = "world"
    """Frame in which the path is defined.
    - ``world``: the path is defined in the world frame. Also called ``odom``.
    - ``robot``: the path is defined in the robot frame. Also called ``base``.
    """

    lookAheadDistance: float = MISSING
    """The lookahead distance for the path follower."""
    two_way_drive: bool = False
    """Allow robot to use reverse gear."""
    switch_time_threshold: float = 1.0
    """Time threshold to switch between the forward and backward drive."""
    maxSpeed: float = 0.5
    """Maximum speed of the robot."""
    maxAccel: float = 2.5 / 100.0  # 2.5 / 100
    """Maximum acceleration of the robot."""
    joyYaw: float = 1.0
    """TODO: add description"""
    yawRateGain: float = 7.0  # 3.5
    """Gain for the yaw rate."""
    stopYawRateGain: float = 7.0  # 3.5
    """"""
    maxYawRate: float = 90.0 * math.pi / 360
    dirDiffThre: float = 0.7
    stopDisThre: float = 0.2
    slowDwnDisThre: float = 0.3
    slowRate1: float = 0.25
    slowRate2: float = 0.5
    noRotAtGoal: bool = True
    autonomyMode: bool = False

    dynamic_lookahead: bool = False
    min_points_within_lookahead: int = 3
