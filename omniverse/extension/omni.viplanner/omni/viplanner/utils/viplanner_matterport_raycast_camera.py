# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import torch
import yaml
from omni.isaac.lab.sensors.ray_caster import RayCasterCameraCfg
from omni.isaac.lab.utils.configclass import configclass
from omni.isaac.matterport.domains import MatterportRayCasterCamera
from omni.viplanner.viplanner import DATA_DIR

from viplanner.config.viplanner_sem_meta import VIPlannerSemMetaHandler


class VIPlannerMatterportRayCasterCamera(MatterportRayCasterCamera):
    def __init__(self, cfg: object):
        super().__init__(cfg)

    def _color_mapping(self):
        viplanner_sem = VIPlannerSemMetaHandler()
        with open(DATA_DIR + "/mpcat40_to_vip_sem.yml") as file:
            map_mpcat40_to_vip_sem = yaml.safe_load(file)
        color = viplanner_sem.get_colors_for_names(list(map_mpcat40_to_vip_sem.values()))
        self.color = torch.tensor(color, device=self._device, dtype=torch.uint8)


@configclass
class VIPlannerMatterportRayCasterCameraCfg(RayCasterCameraCfg):
    class_type = VIPlannerMatterportRayCasterCamera
