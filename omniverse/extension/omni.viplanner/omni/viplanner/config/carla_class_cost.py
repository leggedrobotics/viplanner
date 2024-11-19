# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from omni.isaac.lab.utils import configclass

OBSTACLE_COST = 2.0
ROAD_LOSS = 1.5
TRAVERSABLE_COST = 0.0
TRAVERSABLE_UNINTENDED_LOSS = 0.5
TERRAIN_LOSS = 1.0


@configclass
class CarlaSemanticCostMapping:
    # Mapping from Carla categories to some example cost values
    void: float = OBSTACLE_COST
    road: float = ROAD_LOSS
    sidewalk: float = TRAVERSABLE_COST
    crosswalk: float = TRAVERSABLE_COST
    floor: float = TRAVERSABLE_COST
    vehicle: float = OBSTACLE_COST
    building: float = OBSTACLE_COST
    wall: float = OBSTACLE_COST
    fence: float = OBSTACLE_COST
    pole: float = OBSTACLE_COST
    traffic_sign: float = OBSTACLE_COST
    traffic_light: float = OBSTACLE_COST
    bench: float = OBSTACLE_COST
    vegetation: float = OBSTACLE_COST
    terrain: float = TERRAIN_LOSS
    water_surface: float = OBSTACLE_COST
    sky: float = OBSTACLE_COST
    dynamic: float = OBSTACLE_COST
    static: float = OBSTACLE_COST
    furniture: float = OBSTACLE_COST
