# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022-2024, The ORBIT Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Sub-module containing command generators for the velocity-based locomotion task."""
from __future__ import annotations

import math
from typing import TYPE_CHECKING, Sequence

import omni.isaac.orbit.utils.math as math_utils
import torch
from omni.isaac.orbit.assets.articulation import Articulation
from omni.isaac.orbit.envs import RLTaskEnv
from omni.isaac.orbit.managers import CommandTerm
from omni.isaac.orbit.markers import VisualizationMarkers
from omni.isaac.orbit.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
)
from omni.isaac.orbit.sim import SimulationContext

if TYPE_CHECKING:
    from .path_follower_command_generator_cfg import PathFollowerCommandGeneratorCfg


class PathFollowerCommandGenerator(CommandTerm):
    r"""Command generator that generates a velocity command in SE(2) from a path given by a local planner.

    The command comprises of a linear velocity in x and y direction and an angular velocity around
    the z-axis. It is given in the robot's base frame.

    The path follower acts as a PD-controller that checks for the last point on the path within a lookahead distance
    and uses it to compute the steering angle and the linear velocity.
    """

    cfg: PathFollowerCommandGeneratorCfg
    """The configuration of the command generator."""

    def __init__(self, cfg: PathFollowerCommandGeneratorCfg, env: RLTaskEnv):
        """Initialize the command generator.

        Args:
            cfg (PathFollowerCommandGeneratorCfg): The configuration of the command generator.
            env (object): The environment.
        """
        super().__init__(cfg, env)
        # -- robot
        self.robot: Articulation = env.scene[cfg.robot_attr]
        # -- Simulation Context
        self.sim: SimulationContext = SimulationContext.instance()
        # -- buffers
        self.vehicleSpeed: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.switch_time: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.vehicleYawRate: torch.Tensor = torch.zeros(self.num_envs, device=self.device)
        self.navigation_forward: torch.Tensor = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.twist: torch.Tensor = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_reached: torch.Tensor = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        # -- debug vis
        self._base_vel_goal_markers = None
        self._base_vel_markers = None

    def __str__(self) -> str:
        """Return a string representation of the command generator."""
        msg = "PathFollowerCommandGenerator:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tLookahead distance: {self.cfg.lookAheadDistance}\n"
        return msg

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """The desired base velocity command in the base frame. Shape is (num_envs, 3)."""
        return self.twist

    """
    Operations.
    """

    def reset(self, env_ids: Sequence[int] | None = None) -> dict:
        """Reset the command generator.

        This function resets the command generator. It should be called whenever the environment is reset.

        Args:
            env_ids (Optional[Sequence[int]], optional): The list of environment IDs to reset. Defaults to None.
        """
        if env_ids is None:
            env_ids = ...

        self.vehicleSpeed = torch.zeros(self.num_envs, device=self.device)
        self.switch_time = torch.zeros(self.num_envs, device=self.device)
        self.vehicleYawRate = torch.zeros(self.num_envs, device=self.device)
        self.navigation_forward = torch.ones(self.num_envs, device=self.device, dtype=torch.bool)
        self.twist = torch.zeros((self.num_envs, 3), device=self.device)
        self.goal_reached = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        return {}

    def compute(self, dt: float):
        """Compute the command.

        Paths as a tensor of shape (num_envs, N, 3) where N is number of poses on the path. The paths
        should be given in the base frame of the robot. Num_envs is equal to the number of robots spawned in all
        environments.
        """
        # get paths
        paths = self._env.action_manager._terms[0]._processed_navigation_velocity_actions.clone()
        # get number of pases of the paths
        num_envs, N, _ = paths.shape
        assert N > 0, "PathFollowerCommandGenerator: paths must have at least one poses."
        # get the current simulation time
        curr_time = self.sim.current_time
        # define current maxSpeed for the velocities
        max_speed = torch.ones(num_envs, device=self.device) * self.cfg.maxSpeed

        # transform path in base/ robot frame if given in world frame
        if self.cfg.path_frame == "world":
            paths = math_utils.quat_apply(
                math_utils.quat_inv(self.robot.data.root_quat_w[:, None, :].repeat(1, N, 1)),
                paths - self.robot.data.root_pos_w[:, None, :],
            )

        # get distance that robot has to travel until last set waypoint
        distance_end_point = torch.linalg.norm(paths[:, -1, :2], axis=1)

        # get point that is still within the lookAheadDis, compute its distance and the z-axis rotation
        dis_all_poses = torch.linalg.norm(paths[:, :, :2], axis=2)
        sorted_dis, sorted_dis_idx = torch.sort(dis_all_poses, dim=1)
        if self.cfg.dynamic_lookahead:
            dis_max = sorted_dis[:, self.cfg.min_points_within_lookahead - 1]
            dis_max_poses = paths[:, sorted_dis_idx[:, self.cfg.min_points_within_lookahead - 1], :2]
        else:
            sorted_dis[sorted_dis > self.cfg.lookAheadDistance] = 0.0
            dis_max, dis_max_idx = sorted_dis.max(dim=1)
            dis_max_poses = paths[
                torch.arange(self.num_envs), sorted_dis_idx[torch.arange(self.num_envs), dis_max_idx], :2
            ]
        direction_diff = -torch.atan2(dis_max_poses[:, 1], dis_max_poses[:, 0])

        # decide whether to drive forward or backward
        if self.cfg.two_way_drive:
            switch_time_threshold_exceeded = curr_time - self.switch_time > self.cfg.switch_time_threshold
            # get index of robots that should switch direction
            switch_to_backward_idx = torch.all(
                torch.vstack(
                    (abs(direction_diff) > math.pi / 2, switch_time_threshold_exceeded, self.navigation_forward)
                ),
                dim=0,
            )
            switch_to_forward_idx = torch.all(
                torch.vstack(
                    (abs(direction_diff) < math.pi / 2, switch_time_threshold_exceeded, ~self.navigation_forward)
                ),
                dim=0,
            )
            # update buffers
            self.navigation_forward[switch_to_backward_idx] = False
            self.navigation_forward[switch_to_forward_idx] = True
            self.switch_time[switch_to_backward_idx] = curr_time
            self.switch_time[switch_to_forward_idx] = curr_time

        # adapt direction difference and maxSpeed depending on driving direction
        direction_diff[~self.navigation_forward] += math.pi
        limit_radians = torch.all(torch.vstack((direction_diff > math.pi, ~self.navigation_forward)), dim=0)
        direction_diff[limit_radians] -= 2 * math.pi
        max_speed[~self.navigation_forward] *= -1

        # determine yaw rate of robot
        vehicleYawRate = torch.zeros(num_envs, device=self.device)
        stop_yaw_rate_bool = abs(direction_diff) < 2.0 * self.cfg.maxAccel
        vehicleYawRate[stop_yaw_rate_bool] = -self.cfg.stopYawRateGain * direction_diff[stop_yaw_rate_bool]
        vehicleYawRate[~stop_yaw_rate_bool] = -self.cfg.yawRateGain * direction_diff[~stop_yaw_rate_bool]

        # limit yaw rate of robot
        vehicleYawRate[vehicleYawRate > self.cfg.maxYawRate] = self.cfg.maxYawRate
        vehicleYawRate[vehicleYawRate < -self.cfg.maxYawRate] = -self.cfg.maxYawRate

        # catch special cases
        if not self.cfg.autonomyMode:
            vehicleYawRate[max_speed == 0.0] = self.cfg.maxYawRate * self.cfg.joyYaw
        if N <= 1:
            vehicleYawRate *= 0
            max_speed *= 0
        elif self.cfg.noRotAtGoal:
            vehicleYawRate[dis_max < self.cfg.stopDisThre] = 0.0

        # determine joyspeed at the end of the path
        slow_down_bool = distance_end_point / self.cfg.slowDwnDisThre < max_speed
        max_speed[slow_down_bool] *= distance_end_point[slow_down_bool] / self.cfg.slowDwnDisThre

        # update vehicle speed
        drive_at_max_speed = torch.all(
            torch.vstack((abs(direction_diff) < self.cfg.dirDiffThre, dis_max > self.cfg.stopDisThre)), dim=0
        )
        increase_speed = torch.all(torch.vstack((self.vehicleSpeed < max_speed, drive_at_max_speed)), dim=0)
        decrease_speed = torch.all(torch.vstack((self.vehicleSpeed > max_speed, drive_at_max_speed)), dim=0)
        self.vehicleSpeed[increase_speed] += self.cfg.maxAccel
        self.vehicleSpeed[decrease_speed] -= self.cfg.maxAccel
        increase_speed = torch.all(torch.vstack((self.vehicleSpeed <= 0, ~drive_at_max_speed)), dim=0)
        decrease_speed = torch.all(torch.vstack((self.vehicleSpeed > 0, ~drive_at_max_speed)), dim=0)
        self.vehicleSpeed[increase_speed] += self.cfg.maxAccel
        self.vehicleSpeed[decrease_speed] -= self.cfg.maxAccel

        # update twist command
        self.twist[:, 0] = self.vehicleSpeed
        self.twist[abs(self.vehicleSpeed) < self.cfg.maxAccel * dt, 0] = 0.0
        self.twist[abs(self.vehicleSpeed) > self.cfg.maxSpeed, 0] = self.cfg.maxSpeed
        self.twist[:, 2] = vehicleYawRate

        return self.twist

    """
    Implementation specific functions.
    """

    def _update_command(self):
        pass

    def _update_metrics(self):
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        pass

    def _set_debug_vis_impl(self, debug_vis: bool):
        # set visibility of markers
        # note: parent only deals with callbacks. not their visibility
        if debug_vis:
            # create markers if necessary for the first tome
            if not hasattr(self, "base_vel_goal_visualizer"):
                # -- goal
                marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/velocity_goal"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_goal_visualizer = VisualizationMarkers(marker_cfg)
                # -- current
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.prim_path = "/Visuals/Command/velocity_current"
                marker_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
                self.base_vel_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.base_vel_goal_visualizer.set_visibility(True)
            self.base_vel_visualizer.set_visibility(True)
        else:
            if hasattr(self, "base_vel_goal_visualizer"):
                self.base_vel_goal_visualizer.set_visibility(False)
                self.base_vel_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # get marker location
        # -- base state
        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.5
        # -- resolve the scales and quaternions
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
        # display markers
        self.base_vel_goal_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.base_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    """
    Internal helpers.
    """

    def _resolve_xy_velocity_to_arrow(self, xy_velocity: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Converts the XY base velocity command to arrow direction rotation."""
        # obtain default scale of the marker
        default_scale = self.base_vel_goal_visualizer.cfg.markers["arrow"].scale
        # arrow-scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1)
        # arrow-direction
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        return arrow_scale, arrow_quat
