# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import omni.isaac.orbit.sim as sim_utils
import omni.viplanner.viplanner.mdp as mdp
from omni.isaac.matterport.config import MatterportImporterCfg
from omni.isaac.matterport.domains import MatterportRayCasterCfg
from omni.isaac.orbit.assets import AssetBaseCfg
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.scene import InteractiveSceneCfg
from omni.isaac.orbit.sensors import ContactSensorCfg, patterns
from omni.isaac.orbit.utils import configclass
from omni.viplanner.utils import VIPlannerMatterportRayCasterCameraCfg

from .base_cfg import ObservationsCfg, ViPlannerBaseCfg

##
# Pre-defined configs
##
# isort: off
from omni.isaac.orbit_assets.anymal import ANYMAL_C_CFG

##
# Scene definition
##


@configclass
class TerrainSceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = MatterportImporterCfg(
        prim_path="/World/matterport",
        terrain_type="matterport",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        obj_filepath="${USER_PATH_TO_USD}/matterport.usd",
        groundplane=True,
    )
    # robots
    robot = ANYMAL_C_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (8.0, -0.5, 0.6)
    robot.init_state.rot = (0.6126, 0.0327, 0.0136, -0.7896)

    # sensors
    height_scanner = MatterportRayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=MatterportRayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.5)),
        attach_yaw_only=True,
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=True,
        mesh_prim_paths=["${USER_PATH_TO_USD}/matterport.ply"],
    )
    contact_forces = ContactSensorCfg(prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, debug_vis=False)
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=1000.0,
        ),
    )
    sphere_1 = AssetBaseCfg(
        prim_path="/World/sphere_1",
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=3000.0,
        ),
    )
    sphere_1.init_state.pos = (8, 1, 2.0)
    sphere_2 = AssetBaseCfg(
        prim_path="/World/sphere_2",
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=30000.0,
        ),
    )
    sphere_2.init_state.pos = (10.5, -5.5, 2.0)
    sphere_3 = AssetBaseCfg(
        prim_path="/World/sphere_3",
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=30000.0,
        ),
    )
    sphere_3.init_state.pos = (6.0, -5.5, 2.0)
    sphere_4 = AssetBaseCfg(
        prim_path="/World/sphere_4",
        spawn=sim_utils.SphereLightCfg(
            color=(1.0, 1.0, 1.0),
            intensity=30000.0,
        ),
    )
    sphere_4.init_state.pos = (8.0, -12, 2.0)
    # camera
    depth_camera = VIPlannerMatterportRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=VIPlannerMatterportRayCasterCameraCfg.OffsetCfg(
            pos=(0.510, 0.0, 0.015), rot=(-0.5, 0.5, -0.5, 0.5)
        ),  # FIXME: currently in ROS convention
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            width=848,
            height=480,
            # intrinsics=(430.31607, 0.0, 428.28408, 0.0, 430.31607, 244.00695, 0.0, 0.0, 1.0),  # FIXME: intrinsics not supported yet
        ),
        debug_vis=False,
        max_distance=10,
        mesh_prim_paths=["${USER_PATH_TO_USD}/matterport.ply"],
        data_types=["distance_to_image_plane"],
    )
    semantic_camera = VIPlannerMatterportRayCasterCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=VIPlannerMatterportRayCasterCameraCfg.OffsetCfg(
            pos=(0.510, 0.0, 0.015), rot=(-0.5, 0.5, -0.5, 0.5)
        ),  # FIXME: currently in ROS convention
        pattern_cfg=patterns.PinholeCameraPatternCfg(
            width=1280,
            height=720,
            # intrinsics=(644.15496, 0.0, 639.53125, 0.0, 643.49212, 366.30880, 0.0, 0.0, 1.0),  # FIXME: intrinsics not supported yet
        ),
        data_types=["semantic_segmentation"],
        debug_vis=False,
        mesh_prim_paths=["${USER_PATH_TO_USD}/matterport.ply"],
    )


@configclass
class MatterportObservationsCfg(ObservationsCfg):
    """Observations for locomotion and planner with adjustments for Matterport Environments"""

    @configclass
    class MatterportPlannerImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func=mdp.matterport_raycast_camera_data,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "data_type": "distance_to_image_plane"},
        )
        semantic_measurement = ObsTerm(
            func=mdp.matterport_raycast_camera_data,
            params={"sensor_cfg": SceneEntityCfg("semantic_camera"), "data_type": "semantic_segmentation"},
        )

        def __post_init__(self):
            self.concatenate_terms = False
            self.enable_corruption = False

    planner_image: MatterportPlannerImageCfg = MatterportPlannerImageCfg()


##
# Environment configuration
##


@configclass
class ViPlannerMatterportCfg(ViPlannerBaseCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Scene settings
    scene: TerrainSceneCfg = TerrainSceneCfg(num_envs=1, env_spacing=1.0)
    # adjust image observations
    observations: MatterportObservationsCfg = MatterportObservationsCfg()

    def __post_init__(self):
        """Post initialization."""
        super().__post_init__()
        # adapt viewer
        self.viewer.eye = (8.5, 3.0, 2.5)
        self.viewer.lookat = (8.5, -4.0, 0.0)
