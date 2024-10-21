# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import numpy as np
import trimesh
import warp as wp
from omni.isaac.matterport.domains import DATA_DIR
from omni.isaac.lab.sensors.ray_caster import RayCaster

if TYPE_CHECKING:
    from .raycaster_cfg import MatterportRayCasterCfg


class MatterportRayCaster(RayCaster):
    """A ray-casting sensor for matterport meshes.

    The ray-caster uses a set of rays to detect collisions with meshes in the scene. The rays are
    defined in the sensor's local coordinate frame. The sensor can be configured to ray-cast against
    a set of meshes with a given ray pattern.

    The meshes are parsed from the list of primitive paths provided in the configuration. These are then
    converted to warp meshes and stored in the `warp_meshes` list. The ray-caster then ray-casts against
    these warp meshes using the ray pattern provided in the configuration.

    .. note::
        Currently, only static meshes are supported. Extending the warp mesh to support dynamic meshes
        is a work in progress.
    """

    cfg: MatterportRayCasterCfg
    """The configuration parameters."""

    def __init__(self, cfg: MatterportRayCasterCfg):
        """Initializes the ray-caster object.

        Args:
            cfg (MatterportRayCasterCfg): The configuration parameters.
        """
        # initialize base class
        super().__init__(cfg)

    def _initialize_warp_meshes(self):
        # check if mesh is already loaded
        for mesh_prim_path in self.cfg.mesh_prim_paths:
            if mesh_prim_path in MatterportRayCaster.meshes:
                continue

            # find ply
            if os.path.isabs(mesh_prim_path):
                file_path = mesh_prim_path
                assert os.path.isfile(mesh_prim_path), f"No .ply file found under absolute path: {mesh_prim_path}"
            else:
                file_path = os.path.join(DATA_DIR, mesh_prim_path)
                assert os.path.isfile(
                    file_path
                ), f"No .ply file found under relative path to extension data: {file_path}"

            # load ply
            curr_trimesh = trimesh.load(file_path)

            # Convert trimesh into wp mesh
            mesh_wp = wp.Mesh(
                points=wp.array(curr_trimesh.vertices.astype(np.float32), dtype=wp.vec3, device=self._device),
                indices=wp.array(curr_trimesh.faces.astype(np.int32).flatten(), dtype=int, device=self._device),
            )
            # save mesh
            MatterportRayCaster.meshes[mesh_prim_path] = mesh_wp
