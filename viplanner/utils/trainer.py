# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import contextlib

# python
import os
from typing import List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as Data
import torchvision.transforms as transforms
import tqdm
import wandb  # logging
import yaml

# imperative-planning-learning
from viplanner.config import TrainCfg
from viplanner.plannernet import (
    PRE_TRAIN_POSSIBLE,
    AutoEncoder,
    DualAutoEncoder,
    get_m2f_cfg,
)
from viplanner.traj_cost_opt import TrajCost, TrajViz
from viplanner.utils.torchutil import EarlyStopScheduler, count_parameters

from .dataset import PlannerData, PlannerDataGenerator

torch.set_default_dtype(torch.float32)


class Trainer:
    """
    VIPlanner Trainer
    """

    def __init__(self, cfg: TrainCfg) -> None:
        self._cfg = cfg

        # set model save/load path
        os.makedirs(self._cfg.curr_model_dir, exist_ok=True)
        self.model_path = os.path.join(self._cfg.curr_model_dir, "model.pt")
        if self._cfg.hierarchical:
            self.model_dir_hierarch = os.path.join(self._cfg.curr_model_dir, "hierarchical")
            os.makedirs(self.model_dir_hierarch, exist_ok=True)
            self.hierach_losses = {}

        # image transforms
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((self._cfg.img_input_size), antialias=True),
            ]
        )

        # init buffers DATA
        self.data_generators: List[PlannerDataGenerator] = []
        self.data_traj_cost: List[TrajCost] = []
        self.data_traj_viz: List[TrajViz] = []
        self.fov_ratio: float = None
        self.front_ratio: float = None
        self.back_ratio: float = None
        self.pixel_mean: np.ndarray = None
        self.pixel_std: np.ndarray = None

        # inti buffers MODEL
        self.best_loss = float("inf")
        self.test_loss = float("inf")
        self.net: nn.Module = None
        self.optimizer: optim.Optimizer = None
        self.scheduler: EarlyStopScheduler = None

        print("[INFO] Trainer initialized")
        return

    """PUBLIC METHODS"""

    def train(self) -> None:
        print("[INFO] Start Training")
        # init logging
        self._init_logging()
        # load model and prepare model for training
        self._load_model(self._cfg.resume)
        self._configure_optimizer()

        # get dataloader for training
        self._load_data(train=True)
        if self._cfg.hierarchical:
            step_counter = 0
            train_loader_list, val_loader_list = self._get_dataloader(step=step_counter)
        else:
            train_loader_list, val_loader_list = self._get_dataloader()

        try:
            wandb.watch(self.net)
        except:  # noqa: E722
            print("[WARNING] Wandb model watch failed")

        for epoch in range(self._cfg.epochs):
            train_loss = 0
            val_loss = 0
            for i in range(len(train_loader_list)):
                train_loss += self._train_epoch(train_loader_list[i], epoch, env_id=i)
                val_loss += self._test_epoch(val_loader_list[i], env_id=i, epoch=epoch)

            train_loss /= len(train_loader_list)
            val_loss /= len(train_loader_list)

            try:
                wandb.log(
                    {
                        "train_loss": train_loss,
                        "val_loss": val_loss,
                        "epoch": epoch,
                    }
                )
            except:  # noqa: E722
                print("[WARNING] Wandb logging failed")

            # if val_loss < best_loss:
            if val_loss < self.best_loss:
                print("[INFO] Save model of epoch %d" % (epoch))
                torch.save((self.net.state_dict(), val_loss), self.model_path)
                self.best_loss = val_loss
                print("[INFO] Current val loss: %.4f" % (self.best_loss))

            if self.scheduler.step(val_loss):
                print("[INFO] Early Stopping!")
                break

            if self._cfg.hierarchical and (epoch + 1) % self._cfg.hierarchical_step == 0:
                torch.save(
                    (self.net.state_dict(), self.best_loss),
                    os.path.join(
                        self.model_dir_hierarch,
                        (
                            f"model_ep{epoch}_fov{round(self.fov_ratio, 3)}_"
                            f"front{round(self.front_ratio, 3)}_"
                            f"back{round(self.back_ratio, 3)}.pt"
                        ),
                    ),
                )
                step_counter += 1
                train_loader_list, val_loader_list = self._get_dataloader(step=step_counter)
                self.hierach_losses[epoch] = self.best_loss

        torch.cuda.empty_cache()

        # cleanup data
        for generator in self.data_generators:
            generator.cleanup()

        # empty buffers
        self.data_generators = []
        self.data_traj_cost = []
        self.data_traj_viz = []
        return

    def test(self, step: Optional[int] = None) -> None:
        print("[INFO] Start Training")
        # set random seed for reproducibility
        torch.manual_seed(self._cfg.seed)

        # define step
        if step is None and self._cfg.hierarchical:
            step = int(self._cfg.epochs / self._cfg.hierarchical_step)

        # load model
        self._load_model(resume=True)
        # get dataloader for training
        self._load_data(train=False)
        _, test_loader = self._get_dataloader(train=False, step=step)

        self.test_loss = self._test_epoch(
            test_loader[0],
            env_id=0,
            is_visual=not os.getenv("EXPERIMENT_DIRECTORY"),
            fov_angle=self.data_generators[0].alpha_fov,
            dataset="test",
        )

        # cleanup data
        for generator in self.data_generators:
            generator.cleanup()

    def save_config(self) -> None:
        print(f"[INFO] val_loss: {self.best_loss:.2f}, test_loss," f"{self.test_loss:.4f}")
        """ Save config and loss to file"""
        path, _ = os.path.splitext(self.model_path)
        yaml_path = path + ".yaml"
        print(f"[INFO] Save config and loss to {yaml_path} file")

        loss_dict = {"val_loss": self.best_loss, "test_loss": self.test_loss}
        save_dict = {"config": vars(self._cfg), "loss": loss_dict}

        # dump yaml
        with open(yaml_path, "w+") as file:
            yaml.dump(save_dict, file, allow_unicode=True, default_flow_style=False)

        # logging
        with contextlib.suppress(Exception):
            wandb.finish()

        # plot hierarchical losses
        if self._cfg.hierarchical:
            plt.figure(figsize=(10, 10))
            plt.plot(
                list(self.hierach_losses.keys()),
                list(self.hierach_losses.values()),
            )
            plt.xlabel("Epoch")
            plt.ylabel("Validation Loss")
            plt.title("Hierarchical Losses")
            plt.savefig(os.path.join(self.model_dir_hierarch, "hierarchical_losses.png"))
            plt.close()

        return

    """PRIVATE METHODS"""

    # Helper function DATA
    def _load_data(self, train: bool = True) -> None:
        if not isinstance(self._cfg.data_cfg, list):
            self._cfg.data_cfg = [self._cfg.data_cfg] * len(self._cfg.env_list)
        assert len(self._cfg.data_cfg) == len(self._cfg.env_list), (
            "Either single DataCfg or number matching number of environments" "must be provided"
        )

        for idx, env_name in enumerate(self._cfg.env_list):
            if (train and idx == self._cfg.test_env_id) or (not train and idx != self._cfg.test_env_id):
                continue

            data_path = os.path.join(self._cfg.data_dir, env_name)

            # get trajectory cost map
            traj_cost = TrajCost(
                self._cfg.gpu_id,
                log_data=train,
                w_obs=self._cfg.w_obs,
                w_height=self._cfg.w_height,
                w_goal=self._cfg.w_goal,
                w_motion=self._cfg.w_motion,
                obstalce_thread=self._cfg.obstacle_thread,
            )
            traj_cost.SetMap(
                data_path,
                self._cfg.cost_map_name,
            )

            generator = PlannerDataGenerator(
                cfg=self._cfg.data_cfg[idx],
                root=data_path,
                semantics=self._cfg.sem,
                rgb=self._cfg.rgb,
                cost_map=traj_cost.cost_map,  # trajectory cost class
            )

            traj_viz = TrajViz(
                intrinsics=generator.K_depth,
                cam_resolution=self._cfg.img_input_size,
                camera_tilt=self._cfg.camera_tilt,
                cost_map=traj_cost.cost_map,
            )

            self.data_generators.append(generator)
            self.data_traj_cost.append(traj_cost)
            self.data_traj_viz.append(traj_viz)
            print(f"LOADED DATA FOR ENVIRONMENT: {env_name}")

        print("[INFO] LOADED ALL DATA")
        return

    # Helper function TRAINING
    def _init_logging(self) -> None:
        # logging
        os.environ["WANDB_API_KEY"] = self._cfg.wb_api_key
        os.environ["WANDB_MODE"] = "online"
        os.makedirs(self._cfg.log_dir, exist_ok=True)

        try:
            wandb.init(
                project=self._cfg.wb_project,
                entity=self._cfg.wb_entity,
                name=self._cfg.get_model_save(),
                config=self._cfg.__dict__,
                dir=self._cfg.log_dir,
            )
        except:  # noqa: E722
            print("[WARNING: Wandb not available")
        return

    def _load_model(self, resume: bool = False) -> None:
        if self._cfg.sem or self._cfg.rgb:
            if self._cfg.rgb and self._cfg.pre_train_sem:
                assert PRE_TRAIN_POSSIBLE, (
                    "Pretrained model not available since either detectron2"
                    " not installed or mask2former not found in thrid_party"
                    " folder"
                )
                pre_train_cfg = os.path.join(self._cfg.all_model_dir, self._cfg.pre_train_cfg)
                pre_train_weights = (
                    os.path.join(self._cfg.all_model_dir, self._cfg.pre_train_weights)
                    if self._cfg.pre_train_weights
                    else None
                )
                m2f_cfg = get_m2f_cfg(pre_train_cfg)
                self.pixel_mean = m2f_cfg.MODEL.PIXEL_MEAN
                self.pixel_std = m2f_cfg.MODEL.PIXEL_STD
            else:
                m2f_cfg = None
                pre_train_weights = None

            self.net = DualAutoEncoder(self._cfg, m2f_cfg=m2f_cfg, weight_path=pre_train_weights)
        else:
            self.net = AutoEncoder(self._cfg.in_channel, self._cfg.knodes)

        assert torch.cuda.is_available(), "Code requires GPU"
        print(f"Available GPU list: {list(range(torch.cuda.device_count()))}")
        print(f"Running on GPU: {self._cfg.gpu_id}")
        self.net = self.net.cuda(self._cfg.gpu_id)
        print(f"[INFO] MODEL LOADED ({count_parameters(self.net)} parameters)")

        if resume:
            model_state_dict, self.best_loss = torch.load(self.model_path)
            self.net.load_state_dict(model_state_dict)
            print(f"Resume train from {self.model_path} with loss " f"{self.best_loss}")

        return

    def _configure_optimizer(self) -> None:
        if self._cfg.optimizer == "adam":
            self.optimizer = optim.Adam(
                self.net.parameters(),
                lr=self._cfg.lr,
                weight_decay=self._cfg.w_decay,
            )
        elif self._cfg.optimizer == "sgd":
            self.optimizer = optim.SGD(
                self.net.parameters(),
                lr=self._cfg.lr,
                momentum=self._cfg.momentum,
                weight_decay=self._cfg.w_decay,
            )
        else:
            raise KeyError(f"Optimizer {self._cfg.optimizer} not supported")
        self.scheduler = EarlyStopScheduler(
            self.optimizer,
            factor=self._cfg.factor,
            verbose=True,
            min_lr=self._cfg.min_lr,
            patience=self._cfg.patience,
        )
        print("[INFO] OPTIMIZER AND SCHEDULER CONFIGURED")
        return

    def _get_dataloader(
        self,
        train: bool = True,
        step: Optional[int] = None,
        allow_augmentation: bool = True,
    ) -> None:
        train_loader_list: List[Data.DataLoader] = []
        val_loader_list: List[Data.DataLoader] = []

        if step is not None:
            self.fov_ratio = (
                1.0 - (self._cfg.hierarchical_front_step_ratio + self._cfg.hierarchical_back_step_ratio) * step
            )
            self.front_ratio = self._cfg.hierarchical_front_step_ratio * step
            self.back_ratio = self._cfg.hierarchical_back_step_ratio * step

        for generator in self.data_generators:
            # init data classes

            val_data = PlannerData(
                cfg=generator._cfg,
                transform=self.transform,
                semantics=self._cfg.sem,
                rgb=self._cfg.rgb,
                pixel_mean=self.pixel_mean,
                pixel_std=self.pixel_std,
            )

            if train:
                train_data = PlannerData(
                    cfg=generator._cfg,
                    transform=self.transform,
                    semantics=self._cfg.sem,
                    rgb=self._cfg.rgb,
                    pixel_mean=self.pixel_mean,
                    pixel_std=self.pixel_std,
                )
            else:
                train_data = None

            # split data in train and validation with given sample ratios
            if train:
                generator.split_samples(
                    train_dataset=train_data,
                    test_dataset=val_data,
                    generate_split=train,
                    ratio_back_samples=self.back_ratio,
                    ratio_front_samples=self.front_ratio,
                    ratio_fov_samples=self.fov_ratio,
                    allow_augmentation=allow_augmentation,
                )
            else:
                generator.split_samples(
                    train_dataset=train_data,
                    test_dataset=val_data,
                    generate_split=train,
                    ratio_back_samples=self.back_ratio,
                    ratio_front_samples=self.front_ratio,
                    ratio_fov_samples=self.fov_ratio,
                    allow_augmentation=allow_augmentation,
                )

            if self._cfg.load_in_ram:
                if train:
                    train_data.load_data_in_memory()
                val_data.load_data_in_memory()

            if train:
                train_loader = Data.DataLoader(
                    dataset=train_data,
                    batch_size=self._cfg.batch_size,
                    shuffle=True,
                    pin_memory=True,
                    num_workers=self._cfg.num_workers,
                )
            val_loader = Data.DataLoader(
                dataset=val_data,
                batch_size=self._cfg.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self._cfg.num_workers,
            )

            if train:
                train_loader_list.append(train_loader)
            val_loader_list.append(val_loader)

        return train_loader_list, val_loader_list

    def _train_epoch(
        self,
        loader: Data.DataLoader,
        epoch: int,
        env_id: int,
    ) -> float:
        train_loss, batches = 0, len(loader)
        enumerater = tqdm.tqdm(enumerate(loader))

        for batch_idx, inputs in enumerater:
            odom = inputs[2].cuda(self._cfg.gpu_id)
            goal = inputs[3].cuda(self._cfg.gpu_id)
            self.optimizer.zero_grad()

            if self._cfg.sem or self._cfg.rgb:
                depth_image = inputs[0].cuda(self._cfg.gpu_id)
                sem_rgb_image = inputs[1].cuda(self._cfg.gpu_id)
                preds, fear = self.net(depth_image, sem_rgb_image, goal)
            else:
                image = inputs[0].cuda(self._cfg.gpu_id)
                preds, fear = self.net(image, goal)

            # flip y axis for augmented samples  (clone necessary due to
            # inplace operation that otherwise leads to error in backprop)
            preds_flip = torch.clone(preds)
            preds_flip[inputs[4], :, 1] = preds_flip[inputs[4], :, 1] * -1
            goal_flip = torch.clone(goal)
            goal_flip[inputs[4], 1] = goal_flip[inputs[4], 1] * -1

            log_step = batch_idx + epoch * batches
            loss, _ = self._loss(
                preds_flip,
                fear,
                self.data_traj_cost[env_id],
                odom,
                goal_flip,
                log_step=log_step,
            )
            wandb.log({"train_loss_step": loss}, step=log_step)

            loss.backward()
            self.optimizer.step()
            train_loss += loss.item()
            enumerater.set_description(
                f"Epoch: {epoch} in Env: "
                f"({env_id+1}/{len(self._cfg.env_list)-1}) "
                f"- train loss:{round(train_loss/(batch_idx+1), 4)} on"
                f" {batch_idx}/{batches}"
            )
        return train_loss / (batch_idx + 1)

    def _test_epoch(
        self,
        loader,
        env_id: int,
        epoch: int = 0,
        is_visual=False,
        fov_angle: float = 90.0,
        dataset: str = "val",
    ) -> float:
        test_loss = 0
        num_batches = len(loader)
        preds_viz = []
        wp_viz = []
        image_viz = []

        with torch.no_grad():
            for batch_idx, inputs in enumerate(loader):
                odom = inputs[2].cuda(self._cfg.gpu_id)
                goal = inputs[3].cuda(self._cfg.gpu_id)

                if self._cfg.sem or self._cfg.rgb:
                    image = inputs[0].cuda(self._cfg.gpu_id)  # depth
                    sem_rgb_image = inputs[1].cuda(self._cfg.gpu_id)  # sem
                    preds, fear = self.net(image, sem_rgb_image, goal)
                else:
                    image = inputs[0].cuda(self._cfg.gpu_id)
                    preds, fear = self.net(image, goal)

                # flip y axis for augmented samples
                preds[inputs[4], :, 1] = preds[inputs[4], :, 1] * -1
                goal[inputs[4], 1] = goal[inputs[4], 1] * -1

                log_step = epoch * num_batches + batch_idx
                loss, waypoints = self._loss(
                    preds,
                    fear,
                    self.data_traj_cost[env_id],
                    odom,
                    goal,
                    log_step=log_step,
                    dataset=dataset,
                )

                if dataset == "val":
                    wandb.log({f"{dataset}_loss_step": loss}, step=log_step)

                test_loss += loss.item()

                if is_visual and len(preds_viz) * batch_idx < self._cfg.n_visualize:
                    if batch_idx == 0:
                        odom_viz = odom.cpu()
                        goal_viz = goal.cpu()
                        fear_viz = fear.cpu()
                        augment_viz = inputs[4].cpu()
                    else:
                        odom_viz = torch.cat((odom_viz, odom.cpu()), dim=0)
                        goal_viz = torch.cat((goal_viz, goal.cpu()), dim=0)
                        fear_viz = torch.cat((fear_viz, fear.cpu()), dim=0)
                        augment_viz = torch.cat((augment_viz, inputs[4].cpu()), dim=0)
                    preds_viz.append(preds.cpu())
                    wp_viz.append(waypoints.cpu())
                    image_viz.append(image.cpu())

            if is_visual:
                preds_viz = torch.vstack(preds_viz)
                wp_viz = torch.vstack(wp_viz)
                image_viz = torch.vstack(image_viz)

                # limit again to number of visualizations since before
                # added as multiple of batch size
                preds_viz = preds_viz[: self._cfg.n_visualize]
                wp_viz = wp_viz[: self._cfg.n_visualize]
                image_viz = image_viz[: self._cfg.n_visualize]
                odom_viz = odom_viz[: self._cfg.n_visualize]
                goal_viz = goal_viz[: self._cfg.n_visualize]
                fear_viz = fear_viz[: self._cfg.n_visualize]
                augment_viz = augment_viz[: self._cfg.n_visualize]

                # visual trajectory and images
                self.data_traj_viz[env_id].VizTrajectory(
                    preds_viz,
                    wp_viz,
                    odom_viz,
                    goal_viz,
                    fear_viz,
                    fov_angle=fov_angle,
                    augment_viz=augment_viz,
                )
                self.data_traj_viz[env_id].VizImages(preds_viz, wp_viz, odom_viz, goal_viz, fear_viz, image_viz)
        return test_loss / (batch_idx + 1)

    def _loss(
        self,
        preds: torch.Tensor,
        fear: torch.Tensor,
        traj_cost: TrajCost,
        odom: torch.Tensor,
        goal: torch.Tensor,
        log_step: int,
        step: float = 0.1,
        dataset: str = "train",
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        waypoints = traj_cost.opt.TrajGeneratorFromPFreeRot(preds, step=step)
        loss = traj_cost.CostofTraj(
            waypoints,
            odom,
            goal,
            fear,
            log_step,
            ahead_dist=self._cfg.fear_ahead_dist,
            dataset=dataset,
        )

        return loss, waypoints


# EoF
