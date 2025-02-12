# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# python
import torch

torch.set_default_dtype(torch.float32)

# imperative-planning-learning
from viplanner.config import DataCfg, TrainCfg
from viplanner.utils.trainer import Trainer

if __name__ == "__main__":
    env_list_combi = [
        "2azQ1b91cZZ",  # matterport mesh
        "JeFG25nYj2p",  # matterport mesh
        "Vvot9Ly1tCj",  # matterport mesh
        "town01",  # carla mesh
        "ur6pFq6Qu1A",  # matterport mesh
        "B6ByNegPMKs",  # matterport mesh
        "8WUmhLawc2A",  # matterport mesh
        "town01",  # carla mesh
        "2n8kARJN3HM",  # matterport mesh
    ]
    carla: TrainCfg = TrainCfg(
        sem=True,
        cost_map_name="cost_map_sem",
        env_list=env_list_combi,
        test_env_id=8,
        file_name="combi_more_data",
        data_cfg=DataCfg(
            max_goal_distance=10.0,
        ),
        n_visualize=128,
        wb_project="viplanner",
    )
    trainer = Trainer(carla)
    trainer.train()
    trainer.test()
    trainer.save_config()
    torch.cuda.empty_cache()
