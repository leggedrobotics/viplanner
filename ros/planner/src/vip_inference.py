# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import numpy as np
import torch
import torchvision.transforms as transforms

from viplanner.config.learning_cfg import TrainCfg
from viplanner.plannernet import AutoEncoder, DualAutoEncoder, get_m2f_cfg
from viplanner.traj_cost_opt.traj_opt import TrajOpt

torch.set_default_dtype(torch.float32)


class VIPlannerInference:
    def __init__(
        self,
        cfg,
    ) -> None:
        """VIPlanner Inference Script

        Args:
            cfg (Namespace): Config Namespace
        """
        # get configs
        model_path = os.path.join(cfg.model_save, "model.pt")
        config_path = os.path.join(cfg.model_save, "model.yaml")

        # get train config
        self.train_cfg: TrainCfg = TrainCfg.from_yaml(config_path)

        # get model
        if self.train_cfg.rgb:
            m2f_cfg = get_m2f_cfg(cfg.m2f_config_path)
            self.pixel_mean = m2f_cfg.MODEL.PIXEL_MEAN
            self.pixel_std = m2f_cfg.MODEL.PIXEL_STD
        else:
            m2f_cfg = None
            self.pixel_mean = [0, 0, 0]
            self.pixel_std = [1, 1, 1]

        if self.train_cfg.rgb or self.train_cfg.sem:
            self.net = DualAutoEncoder(train_cfg=self.train_cfg, m2f_cfg=m2f_cfg)
        else:
            self.net = AutoEncoder(
                encoder_channel=self.train_cfg.in_channel,
                k=self.train_cfg.knodes,
            )
        try:
            model_state_dict, _ = torch.load(model_path)
        except ValueError:
            model_state_dict = torch.load(model_path)
        self.net.load_state_dict(model_state_dict)

        # inference script = no grad for model
        self.net.eval()

        # move to GPU if available
        if torch.cuda.is_available():
            self.net = self.net.cuda()
            self._device = "cuda"
        else:
            self._device = "cpu"

        # transforms
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize(tuple(self.train_cfg.img_input_size)),
            ]
        )

        # get trajectory generator
        self.traj_generate = TrajOpt()
        return

    def img_converter(self, img: np.ndarray) -> torch.Tensor:
        # crop image and convert to tensor
        img = self.transforms(img)
        return img.unsqueeze(0).to(self._device)

    def plan(
        self,
        depth_image: np.ndarray,
        sem_rgb_image: np.ndarray,
        goal_robot_frame: torch.Tensor,
    ) -> tuple:
        """Plan to path towards the goal given depth and semantic image

        Args:
            depth_image (np.ndarray): Depth image from the robot
            goal_robot_frame (torch.Tensor): Goal in robot frame
            sem_rgb_image (np.ndarray): Semantic/ RGB Image from the robot.

        Returns:
            tuple: _description_
        """

        with torch.no_grad():
            depth_image = self.img_converter(depth_image).float()
            if self.train_cfg.rgb:
                sem_rgb_image = (sem_rgb_image - self.pixel_mean) / self.pixel_std
            sem_rgb_image = self.img_converter(sem_rgb_image.astype(np.uint8)).float()
            keypoints, fear = self.net(depth_image, sem_rgb_image, goal_robot_frame.to(self._device))

        # generate trajectory
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return traj.cpu().squeeze(0).numpy(), fear.cpu().numpy()

    def plan_depth(
        self,
        depth_image: np.ndarray,
        goal_robot_frame: torch.Tensor,
    ) -> tuple:
        with torch.no_grad():
            depth_image = self.img_converter(depth_image).float()
            keypoints, fear = self.net(depth_image, goal_robot_frame.to(self._device))

        # generate trajectory
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return traj.cpu().squeeze(0).numpy(), fear.cpu().numpy()


# EoF
