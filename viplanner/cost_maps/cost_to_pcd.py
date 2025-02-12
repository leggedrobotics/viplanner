# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import argparse
import os
from typing import Optional, Union

import numpy as np
import open3d as o3d
import pypose as pp
import torch
import yaml

# viplanner
from viplanner.config.costmap_cfg import CostMapConfig, Loader

torch.set_default_dtype(torch.float32)


class CostMapPCD:
    def __init__(
        self,
        cfg: CostMapConfig,
        tsdf_array: np.ndarray,
        viz_points: np.ndarray,
        ground_array: np.ndarray,
        gpu_id: Optional[int] = 0,
        load_from_file: Optional[bool] = False,
    ):
        # determine device
        if torch.cuda.is_available() and gpu_id is not None:
            self.device = torch.device("cuda:" + str(gpu_id))
        else:
            self.device = torch.device("cpu")

        # args
        self.cfg: CostMapConfig = cfg
        self.load_from_file: bool = load_from_file
        self.tsdf_array: torch.Tensor = torch.tensor(tsdf_array, device=self.device)
        self.viz_points: np.ndarray = viz_points
        self.ground_array: torch.Tensor = torch.tensor(ground_array, device=self.device)

        # init flag
        self.map_init = False

        # init pointclouds
        self.pcd_tsdf = o3d.geometry.PointCloud()
        self.pcd_viz = o3d.geometry.PointCloud()

        # execute setup
        self.num_x: int = 0
        self.num_y: int = 0
        self.setup()
        return

    def setup(self):
        # expand of cost map
        self.num_x, self.num_y = self.tsdf_array.shape
        # visualization points
        self.pcd_viz.points = o3d.utility.Vector3dVector(self.viz_points)
        # set cost map
        self.SetUpCostArray()
        # update pcd instance
        xv, yv = np.meshgrid(
            np.linspace(0, self.num_x * self.cfg.general.resolution, self.num_x),
            np.linspace(0, self.num_y * self.cfg.general.resolution, self.num_y),
            indexing="ij",
        )
        T = np.concatenate((np.expand_dims(xv, axis=0), np.expand_dims(yv, axis=0)), axis=0)
        T = np.concatenate(
            (
                T,
                np.expand_dims(self.cost_array.cpu().detach().numpy(), axis=0),
            ),
            axis=0,
        )
        if self.load_from_file:
            wps = T.reshape(3, -1).T + np.array([self.cfg.x_start, self.cfg.y_start, 0.0])
            self.pcd_tsdf.points = o3d.utility.Vector3dVector(wps)
        else:
            self.pcd_tsdf.points = o3d.utility.Vector3dVector(T.reshape(3, -1).T)

        self.map_init = True
        return

    def ShowTSDFMap(self, cost_map=True):  # not run with cuda
        if not self.map_init:
            print("Error: cannot show map, map has not been init yet!")
            return
        if cost_map:
            o3d.visualization.draw_geometries([self.pcd_tsdf])
        else:
            o3d.visualization.draw_geometries([self.pcd_viz])
        return

    def Pos2Ind(self, points: Union[torch.Tensor, pp.LieTensor]):
        # points [torch shapes [num_p, 3]]
        start_xy = torch.tensor(
            [self.cfg.x_start, self.cfg.y_start],
            dtype=torch.float64,
            device=points.device,
        ).expand(1, 1, -1)
        if isinstance(points, pp.LieTensor):
            H = (points.tensor()[:, :, 0:2] - start_xy) / self.cfg.general.resolution
        else:
            H = (points[:, :, 0:2] - start_xy) / self.cfg.general.resolution
        mask = torch.logical_and(
            (H > 0).all(axis=2),
            (H < torch.tensor([self.num_x, self.num_y], device=points.device)[None, None, :]).all(axis=2),
        )
        return self.NormInds(H), H[mask, :]

    def NormInds(self, H):
        norm_matrix = torch.tensor(
            [self.num_x / 2.0, self.num_y / 2.0],
            dtype=torch.float64,
            device=H.device,
        )
        H = (H - norm_matrix) / norm_matrix
        return H

    def DeNormInds(self, NH):
        norm_matrix = torch.tensor(
            [self.num_x / 2.0, self.num_y / 2.0],
            dtype=torch.float64,
            device=NH.device,
        )
        NH = NH * norm_matrix + norm_matrix
        return NH

    def SaveTSDFMap(self):
        if not self.map_init:
            print("Error: map has not been init yet!")
            return

        # make directories
        os.makedirs(
            os.path.join(self.cfg.general.root_path, "maps", "data"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(self.cfg.general.root_path, "maps", "cloud"),
            exist_ok=True,
        )
        os.makedirs(
            os.path.join(self.cfg.general.root_path, "maps", "params"),
            exist_ok=True,
        )

        map_path = os.path.join(
            self.cfg.general.root_path,
            "maps",
            "data",
            self.cfg.map_name + "_map.txt",
        )
        ground_path = os.path.join(
            self.cfg.general.root_path,
            "maps",
            "data",
            self.cfg.map_name + "_ground.txt",
        )
        cloud_path = os.path.join(
            self.cfg.general.root_path,
            "maps",
            "cloud",
            self.cfg.map_name + "_cloud.txt",
        )
        # save data
        np.savetxt(map_path, self.tsdf_array.cpu())
        np.savetxt(ground_path, self.ground_array.cpu())
        np.savetxt(cloud_path, self.viz_points)
        # save config parameters
        yaml_path = os.path.join(
            self.cfg.general.root_path,
            "maps",
            "params",
            f"config_{self.cfg.map_name}.yaml",
        )
        with open(yaml_path, "w+") as file:
            yaml.dump(
                vars(self.cfg),
                file,
                allow_unicode=True,
                default_flow_style=False,
            )

        print("TSDF Map saved.")
        return

    def SetUpCostArray(self):
        self.cost_array = self.tsdf_array
        return

    @classmethod
    def ReadTSDFMap(cls, root_path: str, map_name: str, gpu_id: Optional[int] = None):
        # read config
        with open(os.path.join(root_path, "maps", "params", f"config_{map_name}.yaml")) as f:
            cfg: CostMapConfig = CostMapConfig(**yaml.load(f, Loader))

        # load data
        tsdf_array = np.loadtxt(os.path.join(root_path, "maps", "data", map_name + "_map.txt"))
        viz_points = np.loadtxt(os.path.join(root_path, "maps", "cloud", map_name + "_cloud.txt"))
        ground_array = np.loadtxt(os.path.join(root_path, "maps", "data", map_name + "_ground.txt"))

        return cls(
            cfg=cfg,
            tsdf_array=tsdf_array,
            viz_points=viz_points,
            ground_array=ground_array,
            gpu_id=gpu_id,
            load_from_file=True,
        )


if __name__ == "__main__":
    # parse environment directory and cost_map name
    parser = argparse.ArgumentParser(prog="Show Costmap", description="Show Costmap")
    parser.add_argument(
        "-e",
        "--env",
        type=str,
        help="path to the environment directory",
        required=True,
    )
    parser.add_argument("-m", "--map", type=str, help="name of the cost_map", required=True)
    args = parser.parse_args()

    # show costmap
    map = CostMapPCD.ReadTSDFMap(args.env, args.map)
    map.ShowTSDFMap()

# EoF
