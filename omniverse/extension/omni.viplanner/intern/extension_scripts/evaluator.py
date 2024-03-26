# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# python
import asyncio
import datetime
import json
import os
import pickle
import random
import shutil
from abc import abstractmethod
from typing import List, Tuple

# omni
import carb
import cv2
import networkx as nx
import numpy as np

# isaac-debug
import omni.isaac.debug_draw._debug_draw as omni_debug_draw
import scipy.spatial.transform as tf
import torch

# isaac-anymal
from omni.isaac.anymal.config import (
    ANYMAL_FOLLOW,
    ANYmalCfg,
    ANYmalEvaluatorConfig,
    VIPlannerCfg,
)
from omni.isaac.anymal.robot import ANYmal
from omni.isaac.anymal.tasks import VIPlannerANYmal
from omni.isaac.anymal.utils.camera_utils import get_cam_pose
from omni.isaac.anymal.utils.gif_utils import create_gif

# isaac-core
from omni.isaac.core.simulation_context import SimulationContext

# isaac-orbit
from omni.isaac.orbit.utils.math import convert_quat, quat_mul
from pxr import Usd

# viplanner
from viplanner.utils.eval_utils import BaseEvaluator


class ANYmalOrbitEvaluator(BaseEvaluator):
    def __init__(
        self,
        cfg: ANYmalEvaluatorConfig,
        cfg_anymal: ANYmalCfg,
        cfg_planner: VIPlannerCfg,
    ) -> None:
        # get args
        self._cfg = cfg
        self._cfg_anymal = cfg_anymal
        self._cfg_planner = cfg_planner
        # change flag
        if self._cfg_anymal.viewer.debug_vis:
            print(
                "WARNING: Debug visualization will be switched off since markers do not have semantic label and lead to errors."
            )
            self._cfg_anymal.viewer.debug_vis = False

        # super init
        super().__init__(
            distance_tolerance=self._cfg_planner.conv_dist,
            obs_loss_threshold=self._cfg_planner.obs_loss_threshold,
            cost_map_dir=self._cfg.cost_map_dir,
            cost_map_name=self._cfg.cost_map_name,
        )

        # Acquire draw interface
        self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()

        # init ANYmal with corresponding agent
        self._anymal: ANYmal = None
        self._agent: VIPlannerANYmal = None

        # get simulation context
        self.sim: SimulationContext = None

        # flags
        self.use_waypoint_file: bool = True if self._cfg.waypoint_dir and self._cfg.handcrafted_waypoint_file else False

        return

    @abstractmethod
    def load_scene(self) -> None:
        """Load scene."""
        raise NotImplementedError

    @abstractmethod
    def explore_env(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Setup explorer."""
        raise NotImplementedError

    @abstractmethod
    def post_setup(self) -> None:
        """Post step."""
        pass

    @abstractmethod
    def get_env_name(self) -> str:
        """Get environment name."""
        raise NotImplementedError

    ##
    # Buffers
    ##

    def create_buffers(self) -> None:
        # create standard buffers
        super().create_buffers()
        # add additional buffers
        self.goal_reached: np.ndarray = np.zeros(self._nbr_paths, dtype=bool)
        self.goal_within_fov: np.ndarray = np.ones(self._nbr_paths, dtype=bool)
        self.base_collision: np.ndarray = np.zeros(self._nbr_paths, dtype=bool)
        self.knee_collision: np.ndarray = np.zeros(self._nbr_paths, dtype=bool)
        self.walking_time: np.ndarray = np.ones(self._nbr_paths) * self._cfg.max_time
        self.skip_waypoint: np.ndarray = np.zeros(self._nbr_paths, dtype=bool)

    ##
    # Run Simulation
    ##

    def run_single(self, show_plot: bool = True, repeat_idx: str = "") -> None:
        """RUN SINGLE MODEL"""
        eval_dir = os.path.join(self._cfg_planner.model_dir, f"eval_{self.get_env_name()}", repeat_idx)
        os.makedirs(eval_dir, exist_ok=True)

        # check if results already exist
        if self._cfg.use_prev_results:
            print(f"[INFO] Using previous results from {eval_dir}!")
            success_use_prev_results, start_idx = self.load_eval_arrays(eval_dir=eval_dir)
        else:
            success_use_prev_results = False
            start_idx = 0

        if not success_use_prev_results:
            self.run(eval_dir, start_idx)
            # create eval stats and plots
            self.save_eval_arrays(eval_dir=eval_dir)
            self._filter_statistics()
            self.eval_statistics()
            self.save_eval_results()

            self.plt_single_model(eval_dir=eval_dir, show=show_plot)
        else:
            self.eval_statistics()

        return

    def run_repeat(self) -> None:
        # adjust configs
        self._cfg_anymal.rec_path = True
        self._cfg_anymal.rec_sensor = False
        self._cfg_anymal.follow_camera = True

        start_point = self.waypoints["start"]

        assert (
            self._cfg.repeat_waypoints
        ), "To repeat waypoints, please specify the repeat_waypoints flag in the config!"

        if self._cfg.use_prev_results:
            repeat_indexes = []
            for file in os.listdir(os.path.join(self._cfg_planner.model_dir, f"eval_{self.get_env_name()}")):
                if file.startswith("repeat_") and file[len("repeat_") :].isdigit():
                    index = int(file[len("repeat_") :])
                    repeat_indexes.append(index)
            start_idx = max(repeat_indexes) + 1
        else:
            start_idx = 0

        for repeat_idx in range(start_idx, self._cfg.repeat_waypoints):
            # reset anymal to start position
            self._cfg_anymal.translation_x = start_point[0]
            self._cfg_anymal.translation_y = start_point[1]
            self._cfg_anymal.translation_z = 1.0  # start_point[2]

            self.sim.play()
            # self._anymal.robot._process_info_cfg()
            # reset robot
            self._anymal._reset_robot()
            # reset planner
            # self._agent.reset()
            self.sim.pause()

            self.run_single(show_plot=False, repeat_idx=f"repeat_{repeat_idx}")
            self.reset()
        return

    def run_multi(self, show_single_plot: bool = False) -> None:
        """RUN MULTI MODEL"""
        print(
            f"[INFO] Running multi model evaluation with the models defined in ANYmalEvaluatorConfig! \n"
            f"The model defined under VIPlannerCfg will not be used! The used models are: \n"
            f"{self._cfg.models}"
        )

        length_path_list = []
        length_goal_list = []
        goal_distances_list = []
        obs_loss_list = []

        for model_dir in self._cfg.models:
            print(f"[INFO] Running model {model_dir}!")
            # switch model and update it in the config
            self._agent.planner.switch_model(model_dir)
            self._cfg_planner.model_dir = model_dir
            # run for new model
            self.run_single(show_plot=show_single_plot)
            length_path_list.append(self.length_path)
            length_goal_list.append(self.length_goal)
            goal_distances_list.append(self.goal_distances)
            obs_loss_list.append(self.loss_obstacles)

            self.reset()

        self.plt_comparison(
            length_goal_list,
            length_path_list,
            goal_distances_list,
            self._cfg.models,
            self._cfg.save_dir,
            obs_loss_list,
            model_names=["VIPlanner", "iPlanner"],
        )
        return

    def run(self, eval_dir: str, start_idx: int = 0) -> None:
        # init camera buffers by rendering a first time (otherwise will stop simulation)
        self.sim.play()
        self._anymal.sensors_camera[self._agent.planner.cam_path["rgb"]].update(dt=0.0)
        self._anymal.sensors_camera[self._agent.planner.cam_path["depth"]].update(dt=0.0)
        self.sim.pause()

        # iterate over all waypoints
        for idx in range(self._nbr_paths - start_idx):
            idx_curr = idx + start_idx

            if self.use_waypoint_file:
                next_goalpoint = self.waypoints["waypoints"][idx_curr]
            else:
                next_goalpoint = list(self.waypoints[idx_curr].values())[0]

            # set new goal
            self._agent.cube.set_world_pose(next_goalpoint)
            # reset counter and flags
            counter = 0
            start_time = self.sim.current_time
            past_robot_position = self._anymal.robot.data.root_pos_w.numpy()[0, :2].copy()
            robot_position_time = self.sim.current_time
            self.length_goal[idx_curr] = (
                np.linalg.norm(past_robot_position - next_goalpoint[:2]) - self._agent.planner._cfg_vip.conv_dist
            )

            if self._cfg_anymal.follow_camera:
                cam_follow_save_path = os.path.join(
                    eval_dir,
                    "eval_video",
                    self._cfg.handcrafted_waypoint_file + f"_waypoint{idx_curr}_of_{self.nbr_paths}"
                    if self.use_waypoint_file
                    else f"random_seed{self._cfg.seed}_pairs{self._cfg.num_pairs}_waypoint{idx_curr}_of_{self._nbr_paths}",
                )
                os.makedirs(cam_follow_save_path, exist_ok=True)
            if self._cfg_anymal.rec_sensor:
                sensor_save_paths = []
                for sensor in self._anymal.sensors_camera.keys():
                    sensor_save_path = os.path.join(
                        eval_dir,
                        sensor,
                        self._cfg.handcrafted_waypoint_file + f"_waypoint{idx_curr}_of_{self.nbr_paths}"
                        if self.use_waypoint_file
                        else f"random_seed{self._cfg.seed}_pairs{self._cfg.num_pairs}_waypoint{idx_curr}_of_{self._nbr_paths}",
                    )
                    os.makedirs(sensor_save_path, exist_ok=True)
                    sensor_save_paths.append(sensor_save_path)

            self.sim.play()

            base_net_contact_force = self._anymal.base_contact.get_net_contact_forces(
                clone=False, dt=self.sim.get_physics_dt()
            )
            if (base_net_contact_force > 0.0).any():
                print(f"Waypoint {idx_curr}:\t Start Position Base collides, will discard waypoint!")
                self.base_collision[idx_curr] = True
                self.skip_waypoint[idx_curr] = True
                self.sim_reset(idx_curr, next_goalpoint=next_goalpoint, eval_dir=eval_dir)
                continue
            knee_net_contact_force = self._anymal.knee_contact.get_net_contact_forces(
                clone=False, dt=self.sim.get_physics_dt()
            )
            knee_net_contact_force = knee_net_contact_force.view(-1, 4, 3)
            if (knee_net_contact_force > 0.0).any():
                print(f"Waypoint {idx_curr}:\t Start Position Knee collides, will discard waypoint!")
                self.knee_collision[idx_curr] = True
                self.skip_waypoint[idx_curr] = True
                self.sim_reset(idx_curr, next_goalpoint=next_goalpoint, eval_dir=eval_dir)
                continue

            # collect path
            path = []

            while True:
                self.sim.step()
                counter += 1

                if self._agent.twist.goal_reached:
                    self.goal_reached[idx_curr] = True
                    self.walking_time[idx_curr] = self.sim.current_time - start_time
                    print(
                        f"Waypoint {idx_curr}:\t Goal reached within {self.walking_time[idx_curr]}s ({counter} steps)."
                    )
                    break

                # check if robot is stuck and get path length
                if (
                    self._anymal.robot.data.root_pos_w.numpy()[0, :2].round(decimals=1)
                    == past_robot_position.round(decimals=1)
                ).all():
                    if self.sim.current_time - robot_position_time > self._cfg.max_remain_time:
                        print(f"Waypoint {idx_curr}:\t Robot is stuck!")
                        break
                else:
                    self.length_path[idx_curr] += np.linalg.norm(
                        self._anymal.robot.data.root_pos_w.numpy()[0, :2] - past_robot_position
                    )
                    past_robot_position = self._anymal.robot.data.root_pos_w.numpy()[0, :2].copy()
                    path.append(self._anymal.sensors_camera[self._agent.planner.cam_path["rgb"]]._compute_ros_pose()[0])
                    robot_position_time = self.sim.current_time

                # contact forces
                base_net_contact_force = self._anymal.base_contact.get_net_contact_forces(
                    clone=False, dt=self.sim.get_physics_dt()
                )
                if (base_net_contact_force > 0.0).any() and not self.base_collision[idx_curr]:
                    self.base_collision[idx_curr] = True
                knee_net_contact_force = self._anymal.knee_contact.get_net_contact_forces(
                    clone=False, dt=self.sim.get_physics_dt()
                )
                knee_net_contact_force = knee_net_contact_force.view(-1, 4, 3)
                if (knee_net_contact_force > 0.0).any() and not self.knee_collision[idx_curr]:
                    self.knee_collision[idx_curr] = True
                # feet_net_contact_force = self._anymal.foot_contact.get_net_contact_forces(clone=False, dt=self.sim.get_physics_dt())
                # feet_net_contact_force = feet_net_contact_force.view(-1, 4, 3)

                # check for max time
                if (self.sim.current_time - start_time) >= self._cfg.max_time:
                    print(f"Waypoint {idx_curr}:\t Goal NOT reached.")
                    break

                # eval video
                if self._cfg_anymal.follow_camera and counter % self._cfg_anymal.rec_frequency == 0:
                    # set to constant height and orientation
                    pos = (
                        tf.Rotation.from_quat(
                            convert_quat(self._anymal.robot.data.root_quat_w.clone().numpy()[0], "xyzw")
                        ).as_matrix()
                        @ np.asarray(ANYMAL_FOLLOW.pos)
                        + self._anymal.robot.data.root_pos_w.numpy()[0]
                    )
                    pos[2] = 1.7  # constant height
                    target = self._anymal.robot.data.root_pos_w.clone().numpy()[0]
                    extra_world_frame = tf.Rotation.from_quat(
                        convert_quat(self._anymal.robot.data.root_quat_w.clone().numpy()[0], "xyzw")
                    ).as_matrix() @ np.array([1, 0, 0])
                    target += extra_world_frame
                    target[2] = 0.7  # constant height
                    self._anymal.follow_cam.set_world_pose_from_view(
                        pos,
                        target,
                    )
                    self._anymal.follow_cam.update(self._cfg_anymal.sim.dt)
                    # write image
                    cv2.imwrite(
                        os.path.join(cam_follow_save_path, "step" + f"{counter}".zfill(5) + ".png"),
                        cv2.cvtColor(self._anymal.follow_cam.data.output["rgb"], cv2.COLOR_BGR2RGB),
                    )
                if self._cfg_anymal.rec_sensor and counter % self._cfg_anymal.rec_frequency == 0:
                    for idx, sensor in enumerate(self._anymal.sensors_camera.values()):
                        for data_type, data_array in sensor.data.output.items():
                            if data_array is None:
                                continue

                            if data_type == "rgb" or data_type == "semantic_segmentation":
                                if isinstance(data_array, dict):
                                    # collect image and transfer it to viplanner color space
                                    sem_image = data_array["data"]
                                    sem_idToLabels = data_array["info"]["idToLabels"]
                                    data_array = self._agent.planner.sem_color_transfer(sem_image, sem_idToLabels)

                                cv2.imwrite(
                                    os.path.join(
                                        sensor_save_paths[idx], data_type + "_step" + f"{counter}".zfill(5) + ".png"
                                    ),
                                    cv2.cvtColor(data_array.astype(np.uint8), cv2.COLOR_BGR2RGB),
                                )
                            elif data_type == "distance_to_image_plane":
                                if isinstance(self._agent.planner.planner.train_config.data_cfg, list):
                                    depth_scale = self._agent.planner.planner.train_config.data_cfg[0].depth_scale
                                else:
                                    depth_scale = self._agent.planner.planner.train_config.data_cfg.depth_scale

                                cv2.imwrite(
                                    os.path.join(
                                        sensor_save_paths[idx], data_type + "_step" + f"{counter}".zfill(5) + ".png"
                                    ),
                                    (data_array * depth_scale).astype(np.uint16),
                                )

                # add current position to draw interface to show robot path
                if counter % 100 == 0:
                    self.draw_interface.draw_points(
                        self._anymal.robot.data.root_pos_w.tolist(), [(1, 1, 1, 1)], [5]  # white
                    )

            # pause and reset anymal, planner, ...
            self.sim.pause()
            self.draw_interface.clear_points()
            self.sim_reset(idx_curr, next_goalpoint, eval_dir, path)

            # save intermediate results
            if idx_curr % self._cfg.save_period == 0:
                os.makedirs(os.path.join(eval_dir, f"pre_{idx_curr}"), exist_ok=True)
                self.save_eval_arrays(eval_dir=os.path.join(eval_dir, f"pre_{idx_curr}"), suffix=f"_{idx_curr}")
                if os.path.exists(os.path.join(eval_dir, f"pre_{int(idx_curr-self._cfg.save_period)}")):
                    shutil.rmtree(os.path.join(eval_dir, f"pre_{int(idx_curr-self._cfg.save_period)}"))

            # save git if cam follower is activated
            # if self._cfg_anymal.follow_camera and counter > self._cfg_anymal.rec_frequency:
            #     try:
            #         create_gif(
            #             cam_follow_save_path,
            #             gif_name=f"waypoint{idx_curr}",
            #             # speedup by factor of self._cfg_anymal.follow_camera_frequency
            #             duration=(self.sim.current_time - self._agent.planner.start_time) / counter,
            #         )
            #     except:
            #         carb.log_warn("Could not create gif!")
        return

    ##
    # Sim Setup and Reset
    ##

    def setup(self) -> None:
        # load scene to init simulation context
        self.load_scene()
        # get the simulationContext
        self.sim: SimulationContext = SimulationContext().instance()
        # load waypoints
        self.setup_waypoints()
        # create buffers
        self.create_buffers()
        # setup anymal
        self.anymal_setup()
        # post setup script
        self.post_setup()
        return

    def anymal_setup(self) -> None:
        print("Initializing ANYmal and setup callbacks ...")
        # init anymal
        self._anymal = ANYmal(self._cfg_anymal)
        self._anymal.setup_sync()
        # init anymal agent
        self._agent_setup()
        print("ANYmal initialized.")
        return

    def _agent_setup(self) -> None:
        self._agent = VIPlannerANYmal(
            cfg=self._cfg_anymal,
            camera_sensors=self._anymal.sensors_camera,
            robot=self._anymal.robot,
            height_scanner=self._anymal.height_scanner,
            ros_controller=False,
            planner_cfg=self._cfg_planner,
        )

        self._agent.planner.set_planner_callback()
        asyncio.ensure_future(self._agent.set_walk_callback())

        # prevent local goal to be visible --> messes up semantic and depth images
        self._agent.cube.set_visibility(False)
        return

    def _get_rot_to_point(self, start: list, end: list) -> tuple:
        # set the initial rotation to point to the first waypoint
        angle = np.arctan2(end[1] - start[1], end[0] - start[0])
        rot_quat = tf.Rotation.from_euler("z", angle, degrees=False).as_quat()
        return tuple(convert_quat(rot_quat, "wxyz").tolist())

    def sim_reset(self, idx: int, next_goalpoint: np.array, eval_dir: str, path: List[torch.Tensor] = []) -> None:
        # save distance depth camera to goal and if goal was within fov at the starting position
        # NOTE: base position cannot be taken since the path is determined from the camera position which has an offset
        cam_pos, _ = get_cam_pose(self._anymal.sensors_camera[self._agent.planner.cam_path["depth"]]._sensor_prim)
        self.goal_within_fov[idx] = not self._agent.planner.goal_outside_fov
        self.goal_distances[idx] = max(
            [np.linalg.norm(next_goalpoint[:2] - cam_pos[:2]) - self._agent.planner._cfg_vip.conv_dist, 0.0]
        )

        if len(path) > 0:
            straight_distance = np.linalg.norm(path[-1][:2] - path[0][:2])
            self.path_extension[idx] = (self.length_path[idx] - straight_distance) / straight_distance

            if self._use_cost_map:
                self.loss_obstacles[idx] = self._get_cost_map_loss(np.vstack(path))

            if self._cfg_anymal.rec_path:
                np.save(os.path.join(eval_dir, f"waypoint{idx}_path.npy"), np.vstack(path))

        # reset the robot to new start position if necessary
        if not (self.goal_reached[idx] and self.use_waypoint_file) and idx + 1 < self._nbr_paths:
            # move anymal to new start position
            if self.use_waypoint_file:
                next_goalpoint[2] = 1.0
                self._anymal.robot.cfg.init_state.pos = tuple(next_goalpoint)
                self._anymal.robot.cfg.init_state.rot = self._get_rot_to_point(
                    next_goalpoint, self.waypoints["waypoints"][idx + 1]
                )
            else:
                self._anymal.robot.cfg.init_state.pos = list(self.waypoints[idx + 1].keys())[0]
                self._anymal.robot.cfg.init_state.rot = self._get_rot_to_point(
                    np.array(list(self.waypoints[idx + 1].keys())[0]), list(self.waypoints[idx + 1].values())[0]
                )
            self._anymal.robot._process_info_cfg()
            # reset robot
            self._anymal._reset_robot()
            # reset planner
            self._agent.reset()

        # reset pbar
        self._agent.pbar.close()
        self._agent._setup_pbar()
        return

    ##
    # Eval Stats
    ##
    def _filter_statistics(self) -> None:
        # remove skipped waypoints
        print(f"Waypoint skipped {sum(self.skip_waypoint)} due to knee or base collision in start position.")
        self.goal_reached = self.goal_reached[self.skip_waypoint == False]
        self.goal_within_fov = self.goal_within_fov[self.skip_waypoint == False]
        self.base_collision = self.base_collision[self.skip_waypoint == False]
        self.knee_collision = self.knee_collision[self.skip_waypoint == False]
        self.walking_time = self.walking_time[self.skip_waypoint == False]
        self.goal_distances = self.goal_distances[self.skip_waypoint == False]
        self.length_goal = self.length_goal[self.skip_waypoint == False]
        self.length_path = self.length_path[self.skip_waypoint == False]
        self.loss_obstacles = self.loss_obstacles[self.skip_waypoint == False]
        self.path_extension = self.path_extension[self.skip_waypoint == False]
        return

    def eval_statistics(self) -> None:
        # perform general eval stats
        super().eval_statistics()

        # further eval stats
        within_fov_rate = sum(self.goal_within_fov) / len(self.goal_within_fov)
        avg_time = (
            sum(self.walking_time[self.goal_reached]) / len(self.walking_time[self.goal_reached])
            if len(self.walking_time[self.goal_reached]) > 0
            else np.inf
        )
        base_collision_rate = sum(self.base_collision) / len(self.base_collision)
        knee_collision_rate = sum(self.knee_collision) / len(self.knee_collision)

        print(
            f"Avg time (success):           {avg_time} \n"
            f"Goal within FOV:              {within_fov_rate} \n"
            f"Base collision rate:          {base_collision_rate} \n"
            f"Knee collision rate:          {knee_collision_rate}"
        )

        # extend eval stats
        self.eval_stats["within_fov_rate"] = within_fov_rate
        self.eval_stats["avg_time"] = avg_time
        self.eval_stats["base_collision_rate"] = base_collision_rate
        self.eval_stats["knee_collision_rate"] = knee_collision_rate

        return

    def save_eval_results(self) -> None:
        save_name = self._cfg.handcrafted_waypoint_file if self.use_waypoint_file else self.get_env_name()
        return super().save_eval_results(self._agent._planner_cfg.model_dir, save_name)

    def get_save_prefix(self) -> str:
        return (
            self._cfg.handcrafted_waypoint_file
            if self.use_waypoint_file
            else self.get_env_name() + f"_seed{self._cfg.seed}_pairs{self._cfg.num_pairs}"
        )

    def save_eval_arrays(self, eval_dir: str, suffix: str = "") -> None:
        subdirectories = [name for name in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, name))]
        pre_directories = [subdir for subdir in subdirectories if "pre" in subdir] if len(subdirectories) > 0 else []
        if len(pre_directories) > 0:
            [shutil.rmtree(os.path.join(eval_dir, pre)) for pre in pre_directories]

        prefix: str = self.get_save_prefix()
        np.save(os.path.join(eval_dir, prefix + f"_goal_reached{suffix}.npy"), self.goal_reached)
        np.save(os.path.join(eval_dir, prefix + f"_goal_within_fov{suffix}.npy"), self.goal_within_fov)
        np.save(os.path.join(eval_dir, prefix + f"_base_collision{suffix}.npy"), self.base_collision)
        np.save(os.path.join(eval_dir, prefix + f"_knee_collision{suffix}.npy"), self.knee_collision)
        np.save(os.path.join(eval_dir, prefix + f"_walking_time{suffix}.npy"), self.walking_time)
        np.save(os.path.join(eval_dir, prefix + f"_goal_distances{suffix}.npy"), self.goal_distances)
        np.save(os.path.join(eval_dir, prefix + f"_length_goal{suffix}.npy"), self.length_goal)
        np.save(os.path.join(eval_dir, prefix + f"_length_path{suffix}.npy"), self.length_path)
        np.save(os.path.join(eval_dir, prefix + f"_loss_obstacles{suffix}.npy"), self.loss_obstacles)
        np.save(os.path.join(eval_dir, prefix + f"_skip_waypoint{suffix}.npy"), self.skip_waypoint)
        np.save(os.path.join(eval_dir, prefix + f"_path_extension{suffix}.npy"), self.path_extension)
        return

    def load_eval_arrays(self, eval_dir: str, suffix: str = "") -> Tuple[bool, int]:
        try:
            self._load_eval_arrays(eval_dir, suffix)
            self._filter_statistics()
            return True, 0
        except FileNotFoundError:
            print(f"[INFO] No previous results found in {eval_dir}, search for preliminary results!")

        subdirectories = [name for name in os.listdir(eval_dir) if os.path.isdir(os.path.join(eval_dir, name))]
        pre_directories = [subdir for subdir in subdirectories if "pre" in subdir] if len(subdirectories) > 0 else []

        if len(pre_directories) > 1:
            raise ValueError(f"Multiple pre directories found {pre_directories}, please only keep the most recent one")
        elif len(pre_directories) == 1:
            try:
                eval_dir = os.path.join(eval_dir, pre_directories[0])
                idx = pre_directories[0][3:]
                self._load_eval_arrays(eval_dir, idx)
                print(f"[INFO] Found preliminary results in {eval_dir}, continue from {idx} waypoint!")
                return False, int(idx[1:])
            except FileNotFoundError:
                print(f"[INFO] No preliminary results found in {eval_dir}, start from scratch!")
                return False, 0
        else:
            print(f"[INFO] No preliminary results found in {eval_dir}, start from scratch!")
            return False, 0

    def _load_eval_arrays(self, eval_dir: str, suffix: str = "") -> None:
        prefix: str = self.get_save_prefix()
        self.goal_reached = np.load(os.path.join(eval_dir, prefix + f"_goal_reached{suffix}.npy"))
        self.goal_within_fov = np.load(os.path.join(eval_dir, prefix + f"_goal_within_fov{suffix}.npy"))
        self.base_collision = np.load(os.path.join(eval_dir, prefix + f"_base_collision{suffix}.npy"))
        self.knee_collision = np.load(os.path.join(eval_dir, prefix + f"_knee_collision{suffix}.npy"))
        self.walking_time = np.load(os.path.join(eval_dir, prefix + f"_walking_time{suffix}.npy"))
        self.goal_distances = np.load(os.path.join(eval_dir, prefix + f"_goal_distances{suffix}.npy"))
        self.length_goal = np.load(os.path.join(eval_dir, prefix + f"_length_goal{suffix}.npy"))
        self.length_path = np.load(os.path.join(eval_dir, prefix + f"_length_path{suffix}.npy"))
        self.loss_obstacles = np.load(os.path.join(eval_dir, prefix + f"_loss_obstacles{suffix}.npy"))
        self.skip_waypoint = np.load(os.path.join(eval_dir, prefix + f"_skip_waypoint{suffix}.npy"))
        self.path_extension = np.load(os.path.join(eval_dir, prefix + f"_path_extension{suffix}.npy"))
        return

    ##
    # Waypoint functions
    ##

    def setup_waypoints(self) -> np.ndarray:
        if self.use_waypoint_file:
            print(f"Loading waypoints from {self._cfg.waypoint_file} ...", end=" ")
            # load waypoints
            self._load_waypoints()
            # define start-points
            start_point = self.waypoints["start"]
            # set number of waypoint pairs
            self.set_nbr_paths(len(self.waypoints["waypoints"]))
            print("Waypoints loaded.")
        else:
            save_waypoint_path = os.path.join(
                self._cfg.waypoint_dir,
                f"explored_{self.get_env_name()}_seed{self._cfg.seed}_pairs{self._cfg.num_pairs}",
            )

            if self._cfg.use_existing_explored_waypoints and os.path.isfile(save_waypoint_path + ".pkl"):
                print(f"[INFO] Loading explored waypoints from {save_waypoint_path} ...", end=" ")
                with open(save_waypoint_path + ".pkl", "rb") as f:
                    self.waypoints = pickle.load(f)
                print("Waypoints loaded.")
            else:
                print(
                    "[INFO] No waypoints specified. Using random exploration to select start-goals. Generating now ..."
                )

                sample_points, nn_idx, collision, distance = self.explore_env()
                nbr_points = len(sample_points)

                # get edge indices
                idx_edge_start = np.repeat(np.arange(nbr_points), repeats=self._cfg.num_connections, axis=0)
                idx_edge_end = nn_idx.reshape(-1)

                # filter collision edges and distances
                idx_edge_end = idx_edge_end[~collision.reshape(-1)]
                idx_edge_start = idx_edge_start[~collision.reshape(-1)]
                distance = distance[~collision.reshape(-1)]

                # init graph
                graph = nx.Graph()
                # add nodes with position attributes
                graph.add_nodes_from(list(range(nbr_points)))
                pos_attr = {i: {"pos": sample_points[i]} for i in range(nbr_points)}
                nx.set_node_attributes(graph, pos_attr)
                # add edges with distance attributes
                graph.add_edges_from(list(map(tuple, np.stack((idx_edge_start, idx_edge_end), axis=1))))
                distance_attr = {
                    (i, j): {"distance": distance[idx]} for idx, (i, j) in enumerate(zip(idx_edge_start, idx_edge_end))
                }
                nx.set_edge_attributes(graph, distance_attr)

                # get all shortest paths
                odom_goal_distances = dict(
                    nx.all_pairs_dijkstra_path_length(graph, cutoff=self._cfg.max_goal_dist * 5, weight="distance")
                )

                # map distance to idx pairs
                random.seed(self._cfg.seed)
                distance_map = {}
                for curr_distance in range(self._cfg.min_goal_dist, int(self._cfg.max_goal_dist)):
                    # get all nodes with a distance to the goal of curr_distance
                    pairs = []
                    for key, value in odom_goal_distances.items():
                        norm_distance = np.linalg.norm(sample_points[key] - sample_points[list(value.keys())], axis=1)
                        decisions = np.where(
                            np.logical_and(norm_distance >= curr_distance, norm_distance <= curr_distance + 1)
                        )[0]
                        if len(decisions) > 0:
                            entries = np.array(list(value.keys()))[decisions]
                            [pairs.append({key: entry}) for entry in entries]

                    # randomly select certain pairs
                    distance_map[curr_distance + 1] = random.sample(
                        pairs,
                        min(len(pairs), int(self._cfg.num_pairs / (self._cfg.max_goal_dist - self._cfg.min_goal_dist))),
                    )

                waypoints_idx = []
                for values in distance_map.values():
                    waypoints_idx.extend(values)

                self.waypoints = []
                for idxs in waypoints_idx:
                    self.waypoints.append(
                        {tuple(graph.nodes[list(idxs.keys())[0]]["pos"]): graph.nodes[list(idxs.values())[0]]["pos"]}
                    )

                # save waypoints
                os.makedirs(self._cfg.waypoint_dir, exist_ok=True)
                if os.path.isfile(save_waypoint_path + ".pkl"):
                    print(f"[INFO] File already exists {save_waypoint_path}, will save new one with time!")
                    now = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                    with open(save_waypoint_path + now + ".pkl", "wb") as fp:
                        pickle.dump(self.waypoints, fp)
                else:
                    with open(save_waypoint_path + ".pkl", "wb") as fp:
                        pickle.dump(self.waypoints, fp)

            # define start points
            start_point = list(self.waypoints[0].keys())[0]
            # define number of waypoints / paths
            self.set_nbr_paths(len(self.waypoints))

            print("Done.")

        # set start position and spawn position for anymal
        self._cfg_anymal.translation_x = start_point[0]
        self._cfg_anymal.translation_y = start_point[1]
        self._cfg_anymal.translation_z = 1.0  # start_point[2]

        return start_point

    def _load_waypoints(self) -> None:
        """
        Expected that the waypoints have been recorded with the omni.isaac.waypoint extension and saved in .json format.
        Structure of the json file:
        {
            start: [x, y, z],
            end: [x, y, z],
            waypoints: [[x, y, z], [x, y, z], ...]
        }
        """

        if self._cfg.waypoint_file.endswith(".json"):
            self.waypoints = json.load(open(self._cfg.waypoint_file))
        else:
            self.waypoints = json.load(open(self._cfg.waypoint_file + ".json"))

        # apply scale
        self.waypoints["start"] = [x for x in self.waypoints["start"]]
        self.waypoints["end"] = [x for x in self.waypoints["end"]]
        self.waypoints["waypoints"] = [[x for x in waypoint] for waypoint in self.waypoints["waypoints"]]

        # draw waypoints
        self.draw_interface.draw_points([self.waypoints["start"]], [(1.0, 0.4, 0.0, 1.0)], [(10)])  # orange
        self.draw_interface.draw_points([self.waypoints["end"]], [(0.0, 1.0, 0.0, 1.0)], [(10)])  # green
        self.draw_interface.draw_points(
            self.waypoints["waypoints"],
            [(0.0, 0.0, 1.0, 1.0)] * len(self.waypoints["waypoints"]),  # blue
            [(10)] * len(self.waypoints["waypoints"]),
        )

        # attach end as further goal-point
        self.waypoints["waypoints"].append(self.waypoints["end"])

        return


# EoF
