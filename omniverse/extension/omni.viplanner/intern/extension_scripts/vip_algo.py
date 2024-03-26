# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
from typing import Optional

# omni
import carb
import numpy as np

# python
import torch

from viplanner.config import TrainCfg

# viplanner src
from viplanner.plannernet import (
    PRE_TRAIN_POSSIBLE,
    AutoEncoder,
    DualAutoEncoder,
    get_m2f_cfg,
)
from viplanner.traj_cost_opt.traj_opt import TrajOpt

torch.set_default_dtype(torch.float32)


class VIPlannerAlgo:
    def __init__(self, model_dir: str, m2f_model_dir: Optional[str] = None, viplanner: bool = True) -> None:
        """Apply VIPlanner Algorithm

        Args:
            model_dir (str): Directory that include model.pt and model.yaml
        """
        super().__init__()

        assert os.path.exists(model_dir), "Model directory does not exist"
        if viplanner:
            assert os.path.isfile(os.path.join(model_dir, "model.pt")), "Model file does not exist"
            assert os.path.isfile(os.path.join(model_dir, "model.yaml")), "Model config file does not exist"
        else:
            assert os.path.isfile(os.path.join(model_dir, "plannernet_scripted.pt")), "Model file does not exist"

        # load model
        self.train_config: TrainCfg = None
        self.pixel_mean = None
        self.pixel_std = None
        self.load_model(model_dir, m2f_model_dir, viplanner)

        self.traj_generate = TrajOpt()
        return None

    def load_model(self, model_dir: str, m2f_model_dir: Optional[str] = None, viplanner: bool = True) -> None:
        if viplanner:
            # load train config
            self.train_config: TrainCfg = TrainCfg.from_yaml(os.path.join(model_dir, "model.yaml"))
            carb.log_info(
                f"Model loaded using sem: {self.train_config.sem}, rgb: {self.train_config.rgb}, knodes: {self.train_config.knodes}, in_channel: {self.train_config.in_channel}"
            )

            if isinstance(self.train_config.data_cfg, list):
                self.max_goal_distance = self.train_config.data_cfg[0].max_goal_distance
                self.max_depth = self.train_config.data_cfg[0].max_depth
                self.depth_scale = self.train_config.data_cfg[0].depth_scale
            else:
                self.max_goal_distance = self.train_config.data_cfg.max_goal_distance
                self.max_depth = self.train_config.data_cfg.max_depth
                self.depth_scale = self.train_config.data_cfg.depth_scale

            if self.train_config.rgb or self.train_config.sem:
                if self.train_config.rgb and self.train_config.pre_train_sem:
                    assert (
                        PRE_TRAIN_POSSIBLE
                    ), "Pretrained model not available since either detectron2 or mask2former not correctly setup"
                    pre_train_cfg = os.path.join(m2f_model_dir, self.train_config.pre_train_cfg)
                    pre_train_weights = (
                        os.path.join(m2f_model_dir, self.train_config.pre_train_weights)
                        if self.train_config.pre_train_weights
                        else None
                    )
                    m2f_cfg = get_m2f_cfg(pre_train_cfg)
                    self.pixel_mean = m2f_cfg.MODEL.PIXEL_MEAN
                    self.pixel_std = m2f_cfg.MODEL.PIXEL_STD
                else:
                    m2f_cfg = None
                    pre_train_weights = None

                self.net = DualAutoEncoder(self.train_config, m2f_cfg=m2f_cfg, weight_path=pre_train_weights)
            else:
                self.net = AutoEncoder(self.train_config.in_channel, self.train_config.knodes)

            # get model and load weights
            try:
                model_state_dict, _ = torch.load(os.path.join(model_dir, "model.pt"))
            except ValueError:
                model_state_dict = torch.load(os.path.join(model_dir, "model.pt"))
            self.net.load_state_dict(model_state_dict)
        else:
            self.train_config: TrainCfg = TrainCfg(rgb=False, sem=False)
            self.max_goal_distance = self.train_config.data_cfg.max_goal_distance
            self.max_depth = self.train_config.data_cfg.max_depth
            self.depth_scale = self.train_config.data_cfg.depth_scale
            self.net = torch.jit.load(os.path.join(model_dir, "plannernet_scripted.pt"))

        # inference script = no grad for model
        self.net.eval()

        # move to GPU if available
        if torch.cuda.is_available():
            self.net = self.net.cuda()
            self.cuda_avail = True
        else:
            carb.log_warn("CUDA not available, VIPlanner will run on CPU")
            self.cuda_avail = False
        return

    def plan(self, image: torch.Tensor, goal_robot_frame: torch.Tensor) -> tuple:
        image = image.expand(-1, 3, -1, -1)
        keypoints, fear = self.net(image, goal_robot_frame)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return keypoints, traj, fear

    def plan_dual(self, dep_image: torch.Tensor, sem_image: torch.Tensor, goal_robot_frame: torch.Tensor) -> tuple:
        keypoints, fear = self.net(dep_image, sem_image, goal_robot_frame)
        traj = self.traj_generate.TrajGeneratorFromPFreeRot(keypoints, step=0.1)

        return keypoints, traj, fear


# EoF
