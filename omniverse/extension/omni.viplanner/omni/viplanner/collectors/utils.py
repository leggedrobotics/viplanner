# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.core.utils.prims as prim_utils
from pxr import Usd


def get_all_meshes(env_prim: str) -> tuple[list[Usd.Prim], list[str]]:
    def recursive_mesh_search(start_prim: str, mesh_prims: list):
        for curr_prim in prim_utils.get_prim_at_path(start_prim).GetChildren():
            if curr_prim.GetTypeName() == "Mesh":
                mesh_prims.append(curr_prim)
            else:
                mesh_prims = recursive_mesh_search(start_prim=curr_prim.GetPath().pathString, mesh_prims=mesh_prims)

        return mesh_prims

    assert prim_utils.is_prim_path_valid(env_prim), f"Prim path '{env_prim}' is not valid"

    mesh_prims = recursive_mesh_search(env_prim, [])

    # mesh_prims: dict = prim_utils.get_prim_at_path(self.cfg.prim_path + "/" + self.cfg.usd_name.split(".")[0]).GetChildren()
    mesh_prims_name = [mesh_prim_single.GetName() for mesh_prim_single in mesh_prims]

    return mesh_prims, mesh_prims_name
