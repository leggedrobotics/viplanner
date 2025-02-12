# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

OBSTACLE_COST = 1.0
TRAVERSABLE_COST = 0.0


@configclass
class MatterportSemanticCostMapping:
    # Mapping from Matterport3D mpcat40 categories to some example cost values
    void: float = OBSTACLE_COST
    wall: float = OBSTACLE_COST
    floor: float = TRAVERSABLE_COST
    chair: float = OBSTACLE_COST
    door: float = OBSTACLE_COST
    table: float = OBSTACLE_COST
    picture: float = OBSTACLE_COST
    cabinet: float = OBSTACLE_COST
    cushion: float = OBSTACLE_COST
    window: float = OBSTACLE_COST
    sofa: float = OBSTACLE_COST
    bed: float = OBSTACLE_COST
    curtain: float = OBSTACLE_COST
    chest_of_drawers: float = OBSTACLE_COST
    plant: float = OBSTACLE_COST
    sink: float = OBSTACLE_COST
    stairs: float = OBSTACLE_COST
    ceiling: float = OBSTACLE_COST
    toilet: float = OBSTACLE_COST
    stool: float = OBSTACLE_COST
    towel: float = OBSTACLE_COST
    mirror: float = OBSTACLE_COST
    tv_monitor: float = OBSTACLE_COST
    shower: float = OBSTACLE_COST
    column: float = OBSTACLE_COST
    bathtub: float = OBSTACLE_COST
    counter: float = OBSTACLE_COST
    fireplace: float = OBSTACLE_COST
    lighting: float = OBSTACLE_COST
    beam: float = OBSTACLE_COST
    shelving: float = OBSTACLE_COST
    blinds: float = OBSTACLE_COST
    gym_equipment: float = OBSTACLE_COST
    seating: float = OBSTACLE_COST
    board_panel: float = OBSTACLE_COST
    furniture: float = OBSTACLE_COST
    appliances: float = OBSTACLE_COST
    clothes: float = OBSTACLE_COST
    objects: float = OBSTACLE_COST
    misc: float = OBSTACLE_COST
    unlabeled: float = OBSTACLE_COST
