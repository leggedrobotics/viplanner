# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins
import os
import pickle
import random
import time

import cv2
import numpy as np
import omni.isaac.lab.utils.math as math_utils
import torch
from omni.isaac.lab.markers import VisualizationMarkers
from omni.isaac.lab.markers.config import GREEN_ARROW_X_MARKER_CFG
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.sensors import Camera
from omni.isaac.lab.sim import SimulationContext

from viplanner.config import VIPlannerSemMetaHandler

from .terrain_analysis import TerrainAnalysis
from .viewpoint_sampling_cfg import ViewpointSamplingCfg


class ViewpointSampling:
    def __init__(self, cfg: ViewpointSamplingCfg, scene: InteractiveScene):
        # save cfg and env
        self.cfg = cfg
        self.scene = scene

        # get sim context
        self.sim = SimulationContext.instance()

        # analyse terrains
        self.terrain_analyser = TerrainAnalysis(self.cfg.terrain_analysis, scene=self.scene)

        # initialize viplanner color mapping
        self.viplanner_sem_meta = VIPlannerSemMetaHandler()

    def sample_viewpoints(self, nbr_viewpoints: int, seed: int = 1) -> torch.Tensor:
        """Sample viewpoints for the given number of viewpoints and seed."""
        # the samples are stored in a torch tensor with the structure
        # [x, y, z, qw, qx, qv, qz]

        # load viewpoint samples if the exists
        filename = f"viewpoints_seed{seed}_samples{nbr_viewpoints}.pkl"
        filedir = self.cfg.save_path if self.cfg.save_path else self._get_save_filedir()
        filename = os.path.join(filedir, filename)
        if os.path.isfile(filename):
            with open(filename, "rb") as f:
                data = pickle.load(f)
            # add loaded path dict to data dict
            print(f"[INFO] Loaded {nbr_viewpoints} with seed {seed}.")
            return data
        else:
            print(f"[INFO] No viewpoint samples found for seed {seed} and {nbr_viewpoints} samples.")

        # analyse terrain if not done yet
        if not self.terrain_analyser.complete:
            self.terrain_analyser.analyse()

        # set seed
        random.seed(seed)
        print(f"[INFO] Start sampling {nbr_viewpoints} viewpoints.")

        # samples are organized in [point_idx, neighbor_idx, distance]
        # sample from each point the neighbor with the largest distance
        nbr_samples_per_point = int(np.ceil(nbr_viewpoints / self.terrain_analyser.points.shape[0]).item())
        sample_locations = torch.zeros((nbr_samples_per_point * self.terrain_analyser.points.shape[0], 2))
        sample_locations_count = 0
        curr_point_idx = 0
        while sample_locations_count < nbr_viewpoints:
            # get samples
            sample_idx = self.terrain_analyser.samples[:, 0] == curr_point_idx
            sample_idx_select = torch.randperm(sample_idx.sum())[
                : min(nbr_samples_per_point, nbr_viewpoints - sample_locations_count)
            ]
            sample_locations[sample_locations_count : sample_locations_count + sample_idx_select.shape[0]] = (
                self.terrain_analyser.samples[sample_idx][sample_idx_select, :2]
            )
            sample_locations_count += sample_idx_select.shape[0]
            curr_point_idx += 1
            # reset point index if all points are sampled
            if curr_point_idx >= self.terrain_analyser.points.shape[0]:
                curr_point_idx = 0

        sample_locations = sample_locations[:sample_locations_count].type(torch.int64)

        # get the z angle of the neighbor that is closest to the origin point
        neighbor_direction = (
            self.terrain_analyser.points[sample_locations[:, 0]] - self.terrain_analyser.points[sample_locations[:, 1]]
        )
        z_angles = torch.atan2(neighbor_direction[:, 1], neighbor_direction[:, 0]).to("cpu")

        # vary the rotation of the forward and horizontal axis (in camera frame) as a uniform distribution within the limits
        x_angles = math_utils.sample_uniform(
            self.cfg.x_angle_range[0], self.cfg.x_angle_range[1], sample_locations_count, device="cpu"
        )
        y_angles = math_utils.sample_uniform(
            self.cfg.y_angle_range[0], self.cfg.y_angle_range[1], sample_locations_count, device="cpu"
        )
        x_angles = torch.deg2rad(x_angles)
        y_angles = torch.deg2rad(y_angles)

        samples = torch.zeros((sample_locations_count, 7))
        samples[:, :3] = self.terrain_analyser.points[sample_locations[:, 0]]
        samples[:, 3:] = math_utils.quat_from_euler_xyz(x_angles, y_angles, z_angles)

        print(f"[INFO] Sampled {sample_locations_count} viewpoints.")

        # save samples
        os.makedirs(filedir, exist_ok=True)
        with open(filename, "wb") as f:
            pickle.dump(samples, f)

        print(f"[INFO] Saved {sample_locations_count} viewpoints with seed {seed} to {filename}.")

        # debug points and orientation
        if self.cfg.debug_viz:
            env_render_steps = 1000
            marker_cfg = GREEN_ARROW_X_MARKER_CFG.copy()
            marker_cfg.prim_path = "/Visuals/viewpoints"
            marker_cfg.markers["arrow"].scale = (0.1, 0.1, 0.1)
            self.visualizer = VisualizationMarkers(marker_cfg)
            self.visualizer.visualize(samples[:, :3], samples[:, 3:])

            # check if launched from terminal or in extension workflow
            if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
                print(f"[INFO] Visualizing {sample_locations_count} samples for {env_render_steps} render steps...")
                for _ in range(env_render_steps):
                    self.sim.render()

                self.visualizer.set_visibility(False)
                print("[INFO] Done visualizing.")

        return samples

    def render_viewpoints(self, samples: torch.Tensor):
        """Render the images at the given viewpoints and save them to the drive."""
        print(f"[INFO] Start rendering {samples.shape[0]} images.")

        # get number of environments (are the number of cameras)
        num_envs = self.scene.num_envs
        # define how many rounds are necessary to render all viewpoints
        num_rounds = int(np.ceil(samples.shape[0] / num_envs))
        # image_idx
        image_idx = [0] * len(self.cfg.cameras)

        # save poses
        filedir = self.cfg.save_path if self.cfg.save_path else self._get_save_filedir()
        # create directories
        os.makedirs(os.path.join(filedir, "semantics"), exist_ok=True)
        os.makedirs(os.path.join(filedir, "depth"), exist_ok=True)
        if "rgb" in self.cfg.cameras.values():
            os.makedirs(os.path.join(filedir, "rgb"), exist_ok=True)

        # save camera configurations
        print(f"[INFO] Saving camera configurations to {filedir}.")
        # NOTE: the intrinsics of the depth camera have to come first
        intrinsics = np.zeros((len(self.cfg.cameras), 3, 4))  # saved as ROS Projection matrix
        for cam_idx, cam in enumerate(self.cfg.cameras.keys()):
            intrinsics[cam_idx][:3, :3] = self.scene.sensors[cam].data.intrinsic_matrices[0].cpu().numpy()
        np.savetxt(os.path.join(filedir, "intrinsics.txt"), intrinsics.reshape(-1, 12), delimiter=",")

        # save camera poses (format: x y z qx qy qz qw instead of x y z qw qx qy qz)
        np.savetxt(
            os.path.join(filedir, "camera_extrinsic.txt"),
            samples[:, [0, 1, 2, 4, 5, 6, 3]].cpu().numpy(),
            delimiter=",",
        )

        # save images
        samples = samples.to(self.scene.device)
        start_time = time.time()
        for i in range(num_rounds):
            # get samples idx
            samples_idx = torch.arange(i * num_envs, min((i + 1) * num_envs, samples.shape[0]))
            # set camera positions
            for cam in self.cfg.cameras.keys():
                self.scene.sensors[cam].set_world_poses(
                    positions=samples[samples_idx, :3],
                    orientations=samples[samples_idx, 3:],
                    env_ids=torch.arange(samples_idx.shape[0]),
                    convention="world",
                )
            # update simulation
            self.scene.write_data_to_sim()
            # perform render steps to fill buffers if usd cameras are used
            if any([isinstance(self.scene.sensors[cam], Camera) for cam in self.cfg.cameras.keys()]):
                for _ in range(10):
                    self.sim.render()
            # update scene buffers
            self.scene.update(self.sim.get_physics_dt())
            # render
            for cam_idx, (cam, annotator) in enumerate(self.cfg.cameras.items()):
                image_data_np = self.scene.sensors[cam].data.output[annotator].clone().cpu().numpy()
                # filter nan
                image_data_np[np.isnan(image_data_np)] = 0
                # filter inf
                image_data_np[np.isinf(image_data_np)] = 0

                # save images
                for idx in range(samples_idx.shape[0]):
                    # semantic segmentation
                    if annotator == "semantic_segmentation" or annotator == "rgb":
                        if image_data_np.shape[-1] == 1:
                            # get info data
                            info = self.scene.sensors[cam].data.info[idx][annotator]["idToLabels"]

                            # assign each key a color from the VIPlanner color space
                            info = {
                                int(k): (
                                    self.viplanner_sem_meta.class_color["static"]
                                    if v["class"] in ("BACKGROUND", "UNLABELLED")
                                    else self.viplanner_sem_meta.class_color[v["class"]]
                                )
                                for k, v in info.items()
                            }

                            # NOTE: the label_ids and the ids in the data might not be the same, label ids might not be continuous and
                            #       might not start from 0 as well as some data ids might not be present in the label ids
                            unique_data_ids = np.unique(image_data_np)
                            unique_data_ids.sort()
                            mapping = np.zeros(
                                (max(unique_data_ids.max() + 1, max(info.keys()) + 1), 3), dtype=np.uint8
                            )
                            mapping[list(info.keys())] = np.array(list(info.values()), dtype=np.uint8)
                            output = mapping[image_data_np[idx].squeeze(-1)]
                        else:
                            output = image_data_np[idx]

                        assert cv2.imwrite(
                            os.path.join(
                                filedir,
                                "semantics" if annotator == "semantic_segmentation" else "rgb",
                                f"{image_idx[cam_idx]}".zfill(4) + ".png"
                            ),
                            cv2.cvtColor(output.astype(np.uint8), cv2.COLOR_RGB2BGR),
                        )
                    # depth
                    else:
                        assert cv2.imwrite(
                            os.path.join(filedir, "depth", f"{image_idx[cam_idx]}".zfill(4) + ".png"),
                            np.uint16(image_data_np[idx] * self.cfg.depth_scale),
                        )
                        # save as npy
                        np.save(
                            os.path.join(filedir, "depth", f"{image_idx[cam_idx]}".zfill(4) + ".npy"),
                            image_data_np[idx] * self.cfg.depth_scale,
                        )

                    image_idx[cam_idx] += 1

                    if sum(image_idx) % 100 == 0:
                        print(f"[INFO] Rendered {sum(image_idx)} images in {(time.time() - start_time):.4f}s.")

    ###
    # Safe paths
    ###

    def _get_save_filedir(self) -> str:
        # get env name
        if hasattr(self.scene.terrain.cfg, "obj_filepath"):
            terrain_file_path = self.scene.terrain.cfg.obj_filepath
        elif hasattr(self.scene.terrain.cfg, "usd_path") and isinstance(self.scene.terrain.cfg.usd_path, str):
            terrain_file_path = self.scene.terrain.cfg.usd_path
        else:
            raise KeyError("Only implemented for terrains loaded from usd and matterport")
        env_name = os.path.splitext(terrain_file_path)[0]
        # create directory if necessary
        filedir = os.path.join(terrain_file_path, env_name)
        os.makedirs(filedir, exist_ok=True)
        return filedir
