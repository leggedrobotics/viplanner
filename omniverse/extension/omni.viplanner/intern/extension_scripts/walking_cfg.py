# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Simulation configuration for the robot.

Note:
    These are originally taken from the locomotion/velocity.py environment in Orbit.
"""

# python
import os
from dataclasses import dataclass, field
from typing import List

# orbit-assets
from omni.isaac.assets import ASSETS_DATA_DIR, ASSETS_RESOURCES_DIR

# orbit
from omni.isaac.orbit.robots.config.anymal import ANYMAL_B_CFG, ANYMAL_C_CFG
from omni.isaac.orbit.robots.legged_robot import LeggedRobotCfg
from omni.isaac.orbit.sensors.height_scanner import HeightScannerCfg
from omni.isaac.orbit.sensors.height_scanner.utils import create_points_from_grid
from omni.isaac.orbit.utils.configclass import configclass

# omni-isaac-anymal
from .controller_cfg import LocomotionRlControllerCfg


@dataclass
class ANYmalCfg:
    """Configuration for the walking extension."""

    # simulator
    sim: SimCfg = SimCfg()
    viewer: ViewerCfg = ViewerCfg()
    # scene
    terrain: TerrainCfg = TerrainCfg()

    # controller
    rl_policy: LocomotionRlControllerCfg = LocomotionRlControllerCfg(
        checkpoint_path=os.path.join(ASSETS_RESOURCES_DIR, "policy", "policy_obs_to_action_exp.pt"),
    )
    # robot
    robot: List[LeggedRobotCfg] = field(default_factory=lambda: [ANYMAL_C_CFG, ANYMAL_C_CFG])  # ANYmal D not available
    sensor: SensorCfg = SensorCfg()
    height_scanner: HeightScannerCfg = HeightScannerCfg(
        sensor_tick=0.0,
        points=create_points_from_grid(size=(1.6, 1.0), resolution=0.1),
        offset=(0.0, 0.0, 0.0),
        direction=(0.0, 0.0, -1.0),
        max_distance=1.0,
    )
    # translation and rotation
    translation_x: float = 0.0
    translation_y: float = 0.0
    translation_z: float = 0.7
    quat: tuple = (1.0, 0.0, 0.0, 0.0)  # w,x,y,z

    # prim path
    prim_path: str = "/World/Anymal_c/Robot"
    # ANYmal type
    anymal_type: int = 0  # 0: ANYmal C, 1: ANYmal D

    # record data for evaluation
    follow_camera: bool = True
    rec_frequency: int = 1  # nbr of sim.steps between two camera updates
    rec_path: bool = True
    rec_sensor: bool = True

    # set functions
    def _set_translation_x(self, value: list):
        self.translation_x = value

    def _set_translation_y(self, value: list):
        self.translation_y = value

    def _set_translation_z(self, value: list):
        self.translation_z = value

    def _set_prim_path(self, value: str):
        self.prim_path = value

    def _set_anymal_type(self, value: int):
        self.anymal_type = value

    # get functions
    def get_translation(self):
        return (self.translation_x, self.translation_y, self.translation_z)


# EoF
