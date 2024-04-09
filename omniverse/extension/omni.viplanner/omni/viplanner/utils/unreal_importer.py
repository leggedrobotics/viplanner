# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import carb
import numpy as np
import omni
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.sim as sim_utils
import trimesh
import yaml
from omni.isaac.core.utils.semantics import add_update_semantics, remove_all_semantics
from omni.isaac.orbit.terrains import TerrainImporter
from omni.isaac.orbit.utils.assets import ISAAC_NUCLEUS_DIR
from omni.isaac.orbit.utils.warp import convert_to_warp_mesh
from pxr import Gf, Usd, UsdGeom

if TYPE_CHECKING:
    from .unreal_importer_cfg import UnRealImporterCfg


class UnRealImporter(TerrainImporter):
    """
    Default stairs environment for testing
    """

    cfg: UnRealImporterCfg

    def __init__(self, cfg: UnRealImporterCfg) -> None:
        """
        :param
        """
        super().__init__(cfg)

        # modify mesh
        if self.cfg.cw_config_file:
            self._multiply_crosswalks()

        if self.cfg.people_config_file:
            self._insert_people()

        if self.cfg.vehicle_config_file:
            self._insert_vehicles()

        # assign semantic labels
        if self.cfg.sem_mesh_to_class_map:
            self._add_semantics()

    """
    Import Functions
    """

    def import_usd(self, key: str, usd_path: str):
        """Import a mesh from a USD file.

        USD file can contain arbitrary many meshes.

        Note:
            We do not apply any material properties to the mesh. The material properties should
            be defined in the USD file.

        Args:
            key: The key to store the mesh.
            usd_path: The path to the USD file.

        Raises:
            ValueError: If a terrain with the same key already exists.
        """
        # add mesh to the dict
        if key in self.meshes:
            raise ValueError(f"Mesh with key {key} already exists. Existing keys: {self.meshes.keys()}.")
        # add the prim path
        cfg = sim_utils.UsdFileCfg(usd_path=usd_path)

        if self.cfg.axis_up == "Y" or self.cfg.axis_up == "y":
            cfg.func(self.cfg.prim_path + f"/{key}", cfg, orientation=(0.2759, 0.4469, 0.4469, 0.7240))
        else:
            cfg.func(self.cfg.prim_path + f"/{key}", cfg)

        # assign each submesh it's own geometry prim --> important for raytracing to be able to identify the submesh
        submeshes = self.get_mesh_prims(self.cfg.prim_path + f"/{key}")

        # get material
        # physics material
        # material = PhysicsMaterial(
        #     "/World/PhysicsMaterial", static_friction=0.7, dynamic_friction=0.7, restitution=0
        # )
        for submesh, submesh_name in zip(submeshes[0], submeshes[1]):
            #     # create geometry prim
            #     GeometryPrim(
            #         prim_path=submesh.GetPath().pathString,
            #         name="collision",
            #         position=None,
            #         orientation=None,
            #         collision=True,
            #     ).apply_physics_material(material)
            # physx_utils.setCollider(submesh, approximationShape="None")
            # "None" will use the base triangle mesh if available

            # cast into UsdGeomMesh
            mesh_prim = UsdGeom.Mesh(submesh)
            # store the mesh
            vertices = np.asarray(mesh_prim.GetPointsAttr().Get())
            faces = np.asarray(mesh_prim.GetFaceVertexIndicesAttr().Get())
            # check if both faces and vertices are valid
            if not vertices or not faces:
                carb.log_warn(f"Mesh {submesh_name} has no faces or vertices.")
                continue
            faces = faces.reshape(-1, 3)
            self.meshes[submesh_name] = trimesh.Trimesh(vertices=vertices, faces=faces)
            # create a warp mesh
            device = "cuda" if "cuda" in self.device else "cpu"
            self.warp_meshes[submesh_name] = convert_to_warp_mesh(vertices, faces, device=device)

        # add colliders and physics material
        if self.cfg.groundplane:
            ground_plane_cfg = sim_utils.GroundPlaneCfg(
                physics_material=self.cfg.physics_material, size=(500, 500), visible=False
            )
            ground_plane = ground_plane_cfg.func("/World/GroundPlane", ground_plane_cfg, translation=(0, 0, -0.1))
            ground_plane.visible = False

    """ Assign Semantic Labels """

    def _add_semantics(self):
        # remove all previous semantic labels
        remove_all_semantics(prim_utils.get_prim_at_path(self.cfg.prim_path + "/terrain"), recursive=True)

        # get mesh prims
        mesh_prims, mesh_prims_name = self.get_mesh_prims(self.cfg.prim_path + "/terrain")

        carb.log_info(f"Total of {len(mesh_prims)} meshes in the scene, start assigning semantic class ...")

        # mapping from prim name to class
        with open(self.cfg.sem_mesh_to_class_map) as stream:
            class_keywords = yaml.safe_load(stream)

        # make all the string lower case
        mesh_prims_name = [mesh_prim_single.lower() for mesh_prim_single in mesh_prims_name]
        keywords_class_mapping_lower = {
            key: [value_single.lower() for value_single in value] for key, value in class_keywords.items()
        }

        # assign class to mesh in ISAAC
        def recursive_semUpdate(prim, sem_class_name: str, update_submesh: bool) -> bool:
            # Necessary for Park Mesh
            if (
                prim.GetName() == "HierarchicalInstancedStaticMesh"
            ):  # or "FoliageInstancedStaticMeshComponent" in prim.GetName():
                add_update_semantics(prim, sem_class_name)
                update_submesh = True
            children = prim.GetChildren()
            if len(children) > 0:
                for child in children:
                    update_submesh = recursive_semUpdate(child, sem_class_name, update_submesh)
            return update_submesh

        def recursive_meshInvestigator(mesh_idx, mesh_name, mesh_prim_list) -> bool:
            success = False
            for class_name, keywords in keywords_class_mapping_lower.items():
                if any([keyword in mesh_name for keyword in keywords]):
                    update_submesh = recursive_semUpdate(mesh_prim_list[mesh_idx], class_name, False)
                    if not update_submesh:
                        add_update_semantics(mesh_prim_list[mesh_idx], class_name)
                    success = True
                    break

            if not success:
                success_child = []
                mesh_prims_children, mesh_prims_name_children = self.get_mesh_prims(
                    mesh_prim_list[mesh_idx].GetPrimPath().pathString
                )
                mesh_prims_name_children = [mesh_prim_single.lower() for mesh_prim_single in mesh_prims_name_children]
                for mesh_idx_child, mesh_name_child in enumerate(mesh_prims_name_children):
                    success_child.append(
                        recursive_meshInvestigator(mesh_idx_child, mesh_name_child, mesh_prims_children)
                    )
                success = any(success_child)

            return success

        mesh_list = []
        for mesh_idx, mesh_name in enumerate(mesh_prims_name):
            success = recursive_meshInvestigator(mesh_idx=mesh_idx, mesh_name=mesh_name, mesh_prim_list=mesh_prims)
            if success:
                mesh_list.append(mesh_idx)

        missing = [i for x, y in zip(mesh_list, mesh_list[1:]) for i in range(x + 1, y) if y - x > 1]
        assert len(mesh_list) > 0, "No mesh is assigned a semantic class!"
        assert len(mesh_list) == len(
            mesh_prims_name
        ), f"Not all meshes are assigned a semantic class! Following mesh names are included yet: {[mesh_prims_name[miss_idx] for miss_idx in missing]}"
        carb.log_info("Semantic mapping done.")

        return

    """ Modify Mesh """

    def _multiply_crosswalks(self) -> None:
        """Increase number of crosswalks in the scene."""

        with open(self.cfg.cw_config_file) as stream:
            multipy_cfg: dict = yaml.safe_load(stream)

        # get the stage
        stage = omni.usd.get_context().get_stage()

        # get town prim
        town_prim = multipy_cfg.pop("town_prim")

        # init counter
        crosswalk_add_counter = 0

        for key, value in multipy_cfg.items():
            print(f"Execute crosswalk multiplication '{key}'")

            # iterate over the number of crosswalks to be created
            for copy_idx in range(value["factor"]):
                success = omni.usd.duplicate_prim(
                    stage=stage,
                    prim_path=os.path.join(self.cfg.prim_path + "/terrain", town_prim, value["cw_prim"]),
                    path_to=os.path.join(
                        self.cfg.prim_path + "/terrain",
                        town_prim,
                        value["cw_prim"] + f"_cp{copy_idx}" + value.get("suffix", ""),
                    ),
                    duplicate_layers=True,
                )
                assert success, f"Failed to duplicate crosswalk '{key}'"

                # get crosswalk prim
                prim = prim_utils.get_prim_at_path(
                    os.path.join(
                        self.cfg.prim_path + "/terrain",
                        town_prim,
                        value["cw_prim"] + f"_cp{copy_idx}" + value.get("suffix", ""),
                    )
                )
                xform = UsdGeom.Mesh(prim).AddTranslateOp()
                xform.Set(
                    Gf.Vec3d(value["translation"][0], value["translation"][1], value["translation"][2]) * (copy_idx + 1)
                )

                # update counter
                crosswalk_add_counter += 1

        carb.log_info(f"Number of crosswalks added: {crosswalk_add_counter}")
        print(f"Number of crosswalks added: {crosswalk_add_counter}")

        return

    def _insert_vehicles(self):
        # load vehicle config file
        with open(self.cfg.vehicle_config_file) as stream:
            vehicle_cfg: dict = yaml.safe_load(stream)

        # get the stage
        stage = omni.usd.get_context().get_stage()

        # get town prim and all its meshes
        town_prim = vehicle_cfg.pop("town_prim")
        mesh_prims: dict = prim_utils.get_prim_at_path(f"{self.cfg.prim_path}/terrain/{town_prim}").GetChildren()
        mesh_prims_name = [mesh_prim_single.GetName() for mesh_prim_single in mesh_prims]

        # car counter
        car_add_counter = 0

        for key, vehicle in vehicle_cfg.items():
            print(f"Execute vehicle multiplication '{key}'")

            # get all meshs that include the keystring
            meshs = [
                mesh_prim_single for mesh_prim_single in mesh_prims_name if vehicle["prim_part"] in mesh_prim_single
            ]

            # iterate over the number of vehicles to be created
            for idx, translation in enumerate(vehicle["translation"]):
                for single_mesh in meshs:
                    success = omni.usd.duplicate_prim(
                        stage=stage,
                        prim_path=os.path.join(self.cfg.prim_path + "/terrain", town_prim, single_mesh),
                        path_to=os.path.join(
                            self.cfg.prim_path + "/terrain", town_prim, single_mesh + key + f"_cp{idx}"
                        ),
                        duplicate_layers=True,
                    )
                    assert success, f"Failed to duplicate vehicle '{key}'"

                    prim = prim_utils.get_prim_at_path(
                        os.path.join(self.cfg.prim_path + "/terrain", town_prim, single_mesh + key + f"_cp{idx}")
                    )
                    xform = UsdGeom.Mesh(prim).AddTranslateOp()
                    xform.Set(Gf.Vec3d(translation[0], translation[1], translation[2]))

                car_add_counter += 1

        carb.log_info(f"Number of vehicles added: {car_add_counter}")
        print(f"Number of vehicles added: {car_add_counter}")

        return

    def _insert_people(self):
        # load people config file
        with open(self.cfg.people_config_file) as stream:
            people_cfg: dict = yaml.safe_load(stream)

        # if self.cfg.scale == 1.0:
        #     scale_people = 100
        # else:
        #     scale_people = 1

        for key, person_cfg in people_cfg.items():
            carb.log_verbose(f"Insert person '{key}'")

            self.insert_single_person(
                person_cfg["prim_name"],
                person_cfg["translation"],
                scale_people=1,
                usd_path=person_cfg.get("usd_path", "People/Characters/F_Business_02/F_Business_02.usd"),
            )
            # TODO: movement of the people

        carb.log_info(f"Number of people added: {len(people_cfg)}")
        print(f"Number of people added: {len(people_cfg)}")

        return

    @staticmethod
    def insert_single_person(
        prim_name: str,
        translation: list,
        scale_people: float = 1.0,
        usd_path: str = "People/Characters/F_Business_02/F_Business_02.usd",
    ) -> None:
        person_prim = prim_utils.create_prim(
            prim_path=os.path.join("/World/People", prim_name),
            translation=tuple(translation),
            usd_path=os.path.join(ISAAC_NUCLEUS_DIR, usd_path),
            scale=(scale_people, scale_people, scale_people),
        )

        if isinstance(person_prim.GetAttribute("xformOp:orient").Get(), Gf.Quatd):
            person_prim.GetAttribute("xformOp:orient").Set(Gf.Quatd(1.0, 0.0, 0.0, 0.0))
        else:
            person_prim.GetAttribute("xformOp:orient").Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))

        add_update_semantics(person_prim, "person")

        # add collision body
        UsdGeom.Mesh(person_prim)

        return

    @staticmethod
    def get_mesh_prims(env_prim: str) -> tuple[list[Usd.Prim], list[str]]:
        def recursive_search(start_prim: str, mesh_prims: list):
            for curr_prim in prim_utils.get_prim_at_path(start_prim).GetChildren():
                if curr_prim.GetTypeName() == "Xform" or curr_prim.GetTypeName() == "Mesh":
                    mesh_prims.append(curr_prim)
                elif curr_prim.GetTypeName() == "Scope":
                    mesh_prims = recursive_search(start_prim=curr_prim.GetPath().pathString, mesh_prims=mesh_prims)

            return mesh_prims

        assert prim_utils.is_prim_path_valid(env_prim), f"Prim path '{env_prim}' is not valid"

        mesh_prims = []
        mesh_prims = recursive_search(env_prim, mesh_prims)

        # mesh_prims: dict = prim_utils.get_prim_at_path(self.cfg.prim_path + "/" + self.cfg.usd_name.split(".")[0]).GetChildren()
        mesh_prims_name = [mesh_prim_single.GetName() for mesh_prim_single in mesh_prims]

        return mesh_prims, mesh_prims_name
