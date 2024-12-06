# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from omni.isaac.lab.utils import configclass
from omni.viplanner.config import MatterportSemanticCostMapping


@configclass
class TerrainAnalysisCfg:
    robot_height: float = 0.6
    """Height of the robot"""

    wall_height: float = 1.0
    """Height of the walls.

    Wall filtering will start rays from that height and filter all that hit the mesh within 0.3m."""

    robot_buffer_spawn: float = 0.7
    """Robot buffer for spawn location"""

    sample_points: int = 1000
    """Number of nodes in the tree"""

    max_path_length: float = 10.0
    """Maximum distance from the start location to the goal location"""

    height_diff_edge_filter: bool = False
    """Filter investigated edges in the height difference filter is both are on the same height. Default is False.

    If True, the height difference filter will only be applied if the two points are on different heights.
    This can lead to a speed up if the graph is large. If False, the height difference filter will be applied to all."""

    door_filtering: bool = False
    """Account for doors when doing the height difference based edge filtering. Default is False.

    Normally, the height of the terrain is just determined by top-down raycasting. If True, there will be an additional
    raycasting 0.1m above the ground. If a upward pointing ray does not yield the same height as the top-down ray, the
    algorithms assumes that there is a door and a new height is determined."""

    door_height_threshold: float = 1.5
    """Threshold of the door height for the door detection.

    As some objects are composed out of multiple layers of meshes (e.g. stairs as combination of boxes), a door will be
    identified as a height difference of the top-down ray and the upward ray of at least this threshold."""

    num_connections: int = 5
    """Number of connections to make in the graph"""

    raycaster_sensor: str | None = None
    """Name of the raycaster sensor to use for terrain analysis.

    If None, the terrain analysis will be done on the USD stage. For matterport environments,
    the IsaacLab raycaster sensor can be used as the ply mesh is a single mesh. On the contrary,
    for unreal engine meshes (as they consists out of multiple meshes), raycasting should be
    performed over the USD stage. Default is None."""

    grid_resolution: float = 0.1
    """Resolution of the grid to check for not traversable edges"""

    height_diff_threshold: float = 0.3
    """Threshold for height difference between two points"""

    viz_graph: bool = True
    """Visualize the graph after the construction for a short amount of time."""

    viz_height_map: bool = True
    """Visualize the height map after the construction for a short amount of time."""

    semantic_cost_mapping: object | None = MatterportSemanticCostMapping()
    """Mapping of semantic categories to costs for filtering edges and nodes"""

    semantic_cost_threshold: float = 0.5
    """Threshold for semantic cost filtering"""

    # dimension limiting
    dim_limiter_prim: str | None = None
    """Prim name that should be used to limit the dimensions of the mesh.

    All meshes including this prim string are used to set the range in which the graph is constructed and samples are
    generated. If None, all meshes are considered.

    .. note::
        Only used if not a raycaster sensor is passed to the terrain analysis.
    """

    max_terrain_size: float | None = None
    """Maximum size of the terrain in meters.

    This can be useful when e.g. a ground plan is given and the entire anlaysis would run out of memory. If None, the
    entire terrain is considered.
    """
