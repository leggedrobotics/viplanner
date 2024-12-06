# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import carb
import omni.isaac.lab.utils.math as math_utils
import torch
import torchvision.transforms as transforms

from viplanner.config import TrainCfg

# viplanner
from viplanner.plannernet import AutoEncoder, DualAutoEncoder
from viplanner.traj_cost_opt.traj_opt import TrajOpt

"""
VIPlanner Helpers
"""


class VIPlannerAlgo:
    def __init__(self, model_dir: str, fear_threshold: float = 0.5, device: str = "cuda"):
        """Apply VIPlanner Algorithm

        Args:
            model_dir (str): Directory that include model.pt and model.yaml
        """
        super().__init__()

        assert os.path.exists(model_dir), "Model directory does not exist"
        assert os.path.isfile(os.path.join(model_dir, "model.pt")), "Model file does not exist"
        assert os.path.isfile(os.path.join(model_dir, "model.yaml")), "Model config file does not exist"

        # params
        self.fear_threshold = fear_threshold
        self.device = device

        # load model
        self.train_config: TrainCfg = None
        self.load_model(model_dir)

        # get transforms for images
        self.transform = transforms.Resize(self.train_config.img_input_size, antialias=None)

        # init trajectory optimizer
        self.traj_generate = TrajOpt()

        # setup waypoint display in Isaac
        # in headless mode, we cannot visualize the graph and omni.debug.draw is not available
        try:
            import omni.isaac.debug_draw._debug_draw as omni_debug_draw

            self.draw = omni_debug_draw.acquire_debug_draw_interface()
        except ImportError:
            print("[WARNING] Graph Visualization is not available in headless mode.")
        self.color_fear = [(1.0, 0.4, 0.1, 1.0)]  # red
        self.color_path = [(0.4, 1.0, 0.1, 1.0)]  # green
        self.size = [5.0]

    def load_model(self, model_dir: str):
        # load train config
        self.train_config: TrainCfg = TrainCfg.from_yaml(os.path.join(model_dir, "model.yaml"))
        carb.log_info(
            f"Model loaded using sem: {self.train_config.sem}, rgb: {self.train_config.rgb}, knodes: {self.train_config.knodes}, in_channel: {self.train_config.in_channel}"
        )

        if isinstance(self.train_config.data_cfg, list):
            self.max_goal_distance = self.train_config.data_cfg[0].max_goal_distance
            self.max_depth = self.train_config.data_cfg[0].max_depth
        else:
            self.max_goal_distance = self.train_config.data_cfg.max_goal_distance
            self.max_depth = self.train_config.data_cfg.max_depth

        if self.train_config.sem:
            self.net = DualAutoEncoder(self.train_config)
        else:
            self.net = AutoEncoder(self.train_config.in_channel, self.train_config.knodes)

        # get model and load weights
        try:
            model_state_dict, _ = torch.load(os.path.join(model_dir, "model.pt"), weights_only=True)
        except ValueError:
            model_state_dict = torch.load(os.path.join(model_dir, "model.pt"), weights_only=True)
        self.net.load_state_dict(model_state_dict)

        # inference script = no grad for model
        self.net.eval()

        # move to GPU if available
        if self.device.lower() == "cpu":
            carb.log_warn("CUDA not available, VIPlanner will run on CPU")
            self.cuda_avail = False
        else:
            self.net = self.net.cuda()
            self.cuda_avail = True
        return

    ###
    # Transformations
    ###

    def goal_transformer(self, goal: torch.Tensor, cam_pos: torch.Tensor, cam_quat: torch.Tensor) -> torch.Tensor:
        """transform goal into camera frame"""
        goal_cam_frame = goal - cam_pos
        goal_cam_frame[:, 2] = 0  # trained with z difference of 0
        goal_cam_frame = math_utils.quat_apply(math_utils.quat_inv(cam_quat), goal_cam_frame)
        return goal_cam_frame

    def path_transformer(
        self, path_cam_frame: torch.Tensor, cam_pos: torch.Tensor, cam_quat: torch.Tensor
    ) -> torch.Tensor:
        """transform path from camera frame to world frame"""
        return math_utils.quat_apply(
            cam_quat.unsqueeze(1).repeat(1, path_cam_frame.shape[1], 1), path_cam_frame
        ) + cam_pos.unsqueeze(1)

    def input_transformer(self, image: torch.Tensor) -> torch.Tensor:
        # transform images
        image = self.transform(image)
        image[image > self.max_depth] = 0.0
        image[~torch.isfinite(image)] = 0  # set all inf or nan values to 0
        return image

    ###
    # Planning
    ###

    def plan(self, image: torch.Tensor, goal_robot_frame: torch.Tensor) -> tuple:
        with torch.no_grad():
            keypoints, fear = self.net(self.input_transformer(image), goal_robot_frame)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return keypoints, traj, fear

    def plan_dual(self, dep_image: torch.Tensor, sem_image: torch.Tensor, goal_robot_frame: torch.Tensor) -> tuple:
        # transform input
        sem_image = self.transform(sem_image) / 255
        with torch.no_grad():
            keypoints, fear = self.net(self.input_transformer(dep_image), sem_image, goal_robot_frame)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return keypoints, traj, fear

    ###
    # Debug Draw
    ###

    def debug_draw(self, paths: torch.Tensor, fear: torch.Tensor, goal: torch.Tensor):
        self.draw.clear_lines()
        self.draw.clear_points()

        def draw_single_traj(traj, color, size):
            traj[:, 2] = torch.mean(traj[:, 2])
            self.draw.draw_lines(traj[:-1].tolist(), traj[1:].tolist(), color * len(traj[1:]), size * len(traj[1:]))

        for idx, curr_path in enumerate(paths):
            if fear[idx] > self.fear_threshold:
                draw_single_traj(curr_path, self.color_fear, self.size)
                self.draw.draw_points(goal.tolist(), self.color_fear * len(goal), self.size * len(goal))
            else:
                draw_single_traj(curr_path, self.color_path, self.size)
                self.draw.draw_points(goal.tolist(), self.color_path * len(goal), self.size * len(goal))
