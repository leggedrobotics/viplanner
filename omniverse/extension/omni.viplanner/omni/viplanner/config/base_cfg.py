# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import omni.viplanner.viplanner.mdp as mdp
from omni.isaac.orbit.envs import RLTaskEnvCfg
from omni.isaac.orbit.managers import ObservationGroupCfg as ObsGroup
from omni.isaac.orbit.managers import ObservationTermCfg as ObsTerm
from omni.isaac.orbit.managers import RandomizationTermCfg as RandTerm
from omni.isaac.orbit.managers import SceneEntityCfg
from omni.isaac.orbit.managers import TerminationTermCfg as DoneTerm
from omni.isaac.orbit.utils import configclass
from omni.isaac.orbit.utils.assets import ISAAC_ORBIT_NUCLEUS_DIR

##
# MDP settings
##


@configclass
class RewardsCfg:
    pass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    paths = mdp.NavigationActionCfg(
        asset_name="robot",
        low_level_decimation=4,
        low_level_action=mdp.JointPositionActionCfg(
            asset_name="robot", joint_names=[".*"], scale=0.5, use_default_offset=True
        ),
        low_level_policy_file=os.path.join(ISAAC_ORBIT_NUCLEUS_DIR, "Policies", "ANYmal-C", "policy.pt"),
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        velocity_commands = ObsTerm(func=mdp.generated_commands, params={"command_name": "vel_command"})
        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.low_level_actions)
        height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class PlannerImageCfg(ObsGroup):
        depth_measurement = ObsTerm(
            func=mdp.isaac_camera_data,
            params={"sensor_cfg": SceneEntityCfg("depth_camera"), "data_type": "distance_to_image_plane"},
        )
        semantic_measurement = ObsTerm(
            func=mdp.isaac_camera_data,
            params={"sensor_cfg": SceneEntityCfg("semantic_camera"), "data_type": "semantic_segmentation"},
        )

        def __post_init__(self):
            self.concatenate_terms = False
            self.enable_corruption = False

    @configclass
    class PlannerTransformCfg(ObsGroup):
        cam_position = ObsTerm(
            func=mdp.cam_position,
            params={"sensor_cfg": SceneEntityCfg("depth_camera")},
        )
        cam_orientation = ObsTerm(
            func=mdp.cam_orientation,
            params={"sensor_cfg": SceneEntityCfg("depth_camera")},
        )

        def __post_init__(self):
            self.concatenate_terms = False
            self.enable_corruption = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    planner_image: PlannerImageCfg = PlannerImageCfg()
    planner_transform: PlannerTransformCfg = PlannerTransformCfg()


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    base_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names="base"), "threshold": 1.0},
    )


@configclass
class RandomizationCfg:
    """Configuration for randomization."""

    reset_base = RandTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {"x": (0, 0), "y": (0, 0), "yaw": (0, 0)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    reset_robot_joints = RandTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "position_range": (0.0, 0.0),
            "velocity_range": (0.0, 0.0),
        },
    )


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    vel_command: mdp.PathFollowerCommandGeneratorCfg = mdp.PathFollowerCommandGeneratorCfg(
        robot_attr="robot",
        lookAheadDistance=1.0,
        debug_vis=True,
    )


##
# Environment configuration
##


@configclass
class ViPlannerBaseCfg(RLTaskEnvCfg):
    """Configuration for the locomotion velocity-tracking environment."""

    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # managers
    terminations: TerminationsCfg = TerminationsCfg()
    randomization: RandomizationCfg = RandomizationCfg()
    rewards: RewardsCfg = RewardsCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 20  # 10 Hz
        self.episode_length_s = 60.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.disable_contact_processing = True
        self.sim.physics_material.static_friction = 1.0
        self.sim.physics_material.dynamic_friction = 1.0
        self.sim.physics_material.friction_combine_mode = "max"
        self.sim.physics_material.restitution_combine_mode = "max"
        # update sensor update periods
        # we tick all the sensors based on the smallest update period (physics update period)
        self.scene.height_scanner.update_period = 4 * self.sim.dt  # should we low-level decimation
        self.scene.contact_forces.update_period = self.sim.dt
