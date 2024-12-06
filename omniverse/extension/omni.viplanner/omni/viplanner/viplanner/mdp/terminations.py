# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import TYPE_CHECKING

import omni.isaac.core.utils.prims as prim_utils
import torch
from omni.isaac.lab.assets import Articulation
from omni.isaac.lab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from omni.isaac.lab.envs import ManagerBasedRLEnv


def at_goal(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    distance_threshold: float = 0.2,
) -> torch.Tensor:
    """Terminate the planner when the goal is reached.

    Args:
        env: The learning environment.
        asset_cfg: The name of the robot asset.
        distance_threshold: The distance threshold to the goal.

    Returns:
        Boolean tensor indicating whether the goal is reached.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # extract goal position
    goal_pos = prim_utils.get_prim_at_path("/World/goal").GetAttribute("xformOp:translate")
    goals = torch.tensor(goal_pos.Get(), device=env.device).repeat(env.num_envs, 1)

    # Check conditions for termination
    distance_goal = torch.norm(asset.data.root_pos_w[:, :2] - goals[:, :2], dim=1, p=2)
    return distance_goal < distance_threshold
