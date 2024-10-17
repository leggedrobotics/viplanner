# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""This sub-module contains observation terms specific for viplanner.

The functions can be passed to the :class:`omni.isaac.lab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from omni.isaac.lab.managers import SceneEntityCfg
from omni.isaac.lab.sensors.camera import CameraData

from viplanner.config import VIPlannerSemMetaHandler

from .actions import NavigationAction

if TYPE_CHECKING:
    from omni.isaac.lab.envs.base_env import ManagerBasedEnv


# initialize viplanner config
VIPLANNER_SEM_META = VIPlannerSemMetaHandler()


def matterport_raycast_camera_data(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, data_type: str) -> torch.Tensor:
    """Images generated by the raycast camera."""
    # extract the used quantities (to enable type-hinting)
    sensor: CameraData = env.scene.sensors[sensor_cfg.name].data

    # return the data
    if data_type == "distance_to_image_plane":
        output = sensor.output[data_type].clone().unsqueeze(1)
        output[torch.isnan(output)] = 0.0
        output[torch.isinf(output)] = 0.0
        return output
    else:
        return sensor.output[data_type].clone().permute(0, 3, 1, 2)


def isaac_camera_data(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg, data_type: str) -> torch.Tensor:
    """Images generated by the usd camera."""
    # extract the used quantities (to enable type-hinting)
    sensor: CameraData = env.scene.sensors[sensor_cfg.name].data

    # return the data
    if data_type == "distance_to_image_plane":
        output = sensor.output[data_type].clone()
        output[torch.isnan(output)] = 0.0
        output[torch.isinf(output)] = 0.0
        return output.permute(0, 3, 1, 2)
    elif data_type == "semantic_segmentation":
        # retrieve data
        info = [sensor.info[env_id][data_type]["idToLabels"] for env_id in range(env.num_envs)]
        data = sensor.output[data_type].clone()

        # assign each key a color from the VIPlanner color space
        info = [
            {
                int(k): VIPLANNER_SEM_META.class_color["static"]
                if v["class"] in ("BACKGROUND", "UNLABELLED")
                else VIPLANNER_SEM_META.class_color[v["class"]]
                for k, v in d.items()
            }
            for d in info
        ]

        # create recolored images
        output = torch.zeros((*data.shape[:3], 3), device=env.device, dtype=torch.uint8)

        for env_id in range(env.num_envs):
            mapping = torch.zeros((max(info[env_id].keys()) + 1, 3), dtype=torch.uint8, device=env.device)
            mapping[list(info[env_id].keys())] = torch.tensor(
                list(info[env_id].values()), dtype=torch.uint8, device=env.device
            )
            output[env_id] = mapping[data[env_id].long().squeeze(-1)]

        return output.permute(0, 3, 1, 2)
    else:
        return sensor.output[data_type].clone()


def cam_position(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Position of the camera."""
    # extract the used quantities (to enable type-hinting)
    sensor: CameraData = env.scene.sensors[sensor_cfg.name].data

    return sensor.pos_w.clone()


def cam_orientation(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Orientation of the camera."""
    # extract the used quantities (to enable type-hinting)
    sensor: CameraData = env.scene.sensors[sensor_cfg.name].data

    return sensor.quat_w_world.clone()


def low_level_actions(env: ManagerBasedEnv) -> torch.Tensor:
    """Low-level actions."""
    # extract the used quantities (to enable type-hinting)
    action_term: NavigationAction = env.action_manager._terms["paths"]

    return action_term.low_level_actions.clone()
