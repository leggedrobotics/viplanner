# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

from omni.isaac.core.utils import extensions
from omni.isaac.matterport.domains import MatterportImporter
from omni.isaac.lab.terrains import TerrainImporterCfg
from omni.isaac.lab.utils import configclass
from typing_extensions import Literal

extensions.enable_extension("omni.kit.asset_converter")
from omni.kit.asset_converter.impl import AssetConverterContext

# NOTE: hopefully will be soon changed to dataclass, then initialization can be improved
asset_converter_cfg: AssetConverterContext = AssetConverterContext()
asset_converter_cfg.ignore_materials = False
# Don't import/export materials
asset_converter_cfg.ignore_animations = False
# Don't import/export animations
asset_converter_cfg.ignore_camera = False
# Don't import/export cameras
asset_converter_cfg.ignore_light = False
# Don't import/export lights
asset_converter_cfg.single_mesh = False
# By default, instanced props will be export as single USD for reference. If
# this flag is true, it will export all props into the same USD without instancing.
asset_converter_cfg.smooth_normals = True
# Smoothing normals, which is only for assimp backend.
asset_converter_cfg.export_preview_surface = False
# Imports material as UsdPreviewSurface instead of MDL for USD export
asset_converter_cfg.use_meter_as_world_unit = True
# Sets world units to meters, this will also scale asset if it's centimeters model.
asset_converter_cfg.create_world_as_default_root_prim = True
# Creates /World as the root prim for Kit needs.
asset_converter_cfg.embed_textures = True
# Embedding textures into output. This is only enabled for FBX and glTF export.
asset_converter_cfg.convert_fbx_to_y_up = False
# Always use Y-up for fbx import.
asset_converter_cfg.convert_fbx_to_z_up = True
# Always use Z-up for fbx import.
asset_converter_cfg.keep_all_materials = False
# If it's to remove non-referenced materials.
asset_converter_cfg.merge_all_meshes = False
# Merges all meshes to single one if it can.
asset_converter_cfg.use_double_precision_to_usd_transform_op = False
# Uses double precision for all transform ops.
asset_converter_cfg.ignore_pivots = False
# Don't export pivots if assets support that.
asset_converter_cfg.disabling_instancing = False
# Don't export instancing assets with instanceable flag.
asset_converter_cfg.export_hidden_props = False
# By default, only visible props will be exported from USD exporter.
asset_converter_cfg.baking_scales = False
# Only for FBX. It's to bake scales into meshes.


@configclass
class MatterportImporterCfg(TerrainImporterCfg):
    class_type: type = MatterportImporter
    """The class name of the terrain importer."""

    terrain_type: Literal["matterport"] = "matterport"
    """The type of terrain to generate. Defaults to "matterport".

    """

    prim_path: str = "/World/Matterport"
    """The absolute path of the Matterport Environment prim.

    All sub-terrains are imported relative to this prim path.
    """

    obj_filepath: str = ""

    asset_converter: AssetConverterContext = asset_converter_cfg

    groundplane: bool = True
