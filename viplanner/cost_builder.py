# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# imperative-cost-map
from viplanner.config import CostMapConfig
from viplanner.cost_maps import CostMapPCD, SemCostMap, TsdfCostMap


def main(cfg: CostMapConfig, final_viz: bool = True):
    assert any([cfg.semantics, cfg.geometry]), "no cost map type selected"

    # create semantic cost map
    if cfg.semantics:
        print("============ Creating Semantic Map from cloud ===============")
        sem_cost_map = SemCostMap(cfg.general, cfg.sem_cost_map, visualize=cfg.visualize)
        sem_cost_map.pcd_init()
        data, coord = sem_cost_map.create_costmap()
    # create tsdf cost map
    elif cfg.geometry:
        print("============== Creating tsdf Map from cloud =================")
        tsdf_cost_map = TsdfCostMap(cfg.general, cfg.tsdf_cost_map)
        tsdf_cost_map.ReadPointFromFile()
        data, coord = tsdf_cost_map.CreateTSDFMap()
        (tsdf_cost_map.VizCloud(tsdf_cost_map.obs_pcd) if cfg.visualize else None)
    else:
        raise ValueError("no cost map type selected")

    # set coords in costmap config
    cfg.x_start, cfg.y_start = coord

    # construct final cost map as pcd and save parameters
    print("======== Generate and Save costmap as Point-Cloud ===========")
    cost_mapper = CostMapPCD(
        cfg=cfg,
        tsdf_array=data[0],
        viz_points=data[1],
        ground_array=data[2],
        load_from_file=False,
    )
    cost_mapper.SaveTSDFMap()
    if final_viz:
        cost_mapper.ShowTSDFMap(cost_map=True)
    return


if __name__ == "__main__":
    cfg = CostMapConfig()
    main(cfg)

# EoF
