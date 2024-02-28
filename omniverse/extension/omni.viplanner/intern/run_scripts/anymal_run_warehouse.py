# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Launch Omniverse Toolkit first.
"""

# python
import argparse
import json

# omni-isaac-orbit
from omni.isaac.kit import SimulationApp

# add argparse arguments
parser = argparse.ArgumentParser("Welcome to Orbit: Omniverse Robotics Environments!")
parser.add_argument("--headless", action="store_false", default=True, help="Force display off at all times.")
args_cli = parser.parse_args()

# launch omniverse app
config = {"headless": args_cli.headless}
launcher = SimulationApp(config)


"""
Rest everything follows.
"""
import os

# python
from typing import Tuple

import numpy as np

# isaac-core
from omni.isaac.core.objects.ground_plane import GroundPlane
from omni.isaac.core.utils import extensions

# enable ROS bridge extension  --> otherwise rospy cannot be imported
extensions.enable_extension("omni.isaac.ros_bridge")
extensions.enable_extension("omni.kit.manipulator.viewport")

# isaac-anymal
from omni.isaac.anymal.config import (
    ANYmalCfg,
    ANYmalEvaluatorConfig,
    SensorCfg,
    TwistControllerCfg,
    VIPlannerCfg,
)
from omni.isaac.anymal.viplanner.evaluator import ANYmalOrbitEvaluator

# orbit-assets
from omni.isaac.assets import ASSETS_RESOURCES_DIR

# isaac-carla
from omni.isaac.carla.configs import DATA_DIR, CarlaExplorerConfig, CarlaLoaderConfig
from omni.isaac.carla.scripts import CarlaExplorer, CarlaLoader


class ANYmalRunCarla(ANYmalOrbitEvaluator):
    def __init__(
        self,
        cfg: ANYmalEvaluatorConfig,
        cfg_carla: CarlaLoaderConfig = CarlaLoaderConfig(),
        cfg_explore: CarlaExplorerConfig = CarlaExplorerConfig(),
        cfg_anymal: ANYmalCfg = ANYmalCfg(),
        cfg_planner: VIPlannerCfg = VIPlannerCfg(),
    ) -> None:
        # configs
        self._cfg_carla = cfg_carla
        self._cfg_explore = cfg_explore
        # run init
        super().__init__(cfg, cfg_anymal, cfg_planner)
        return

    def load_scene(self) -> None:
        print("Loading scene...")
        if self._cfg_carla.groundplane:
            self._cfg_carla.groundplane = False
            self._groundplane = True
        else:
            self._groundplane = False
        self._loader = CarlaLoader(self._cfg_carla)
        self._loader.load()
        print("DONE")
        return

    def explore_env(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        explorer: CarlaExplorer = CarlaExplorer(self._cfg_explore, self._cfg_carla)
        explorer._get_cam_position()
        nearest_neighor_idx, collision, distance = explorer._construct_kdtree(num_neighbors=self._cfg.num_connections)

        if self._groundplane:
            # add groundplane back
            _ = GroundPlane(
                "/World/GroundPlane", z_position=0.25, physics_material=self._loader.material, visible=False
            )

        return explorer.camera_positions, nearest_neighor_idx, collision, distance

    def get_env_name(self) -> str:
        return os.path.splitext(self._cfg_carla.usd_name)[0]

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
        self.waypoints["start"] = [x * self._cfg_carla.scale for x in self.waypoints["start"]]
        self.waypoints["end"] = [x * self._cfg_carla.scale for x in self.waypoints["end"]]
        self.waypoints["waypoints"] = [
            [x * self._cfg_carla.scale for x in waypoint] for waypoint in self.waypoints["waypoints"]
        ]

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


if __name__ == "__main__":
    # configs
    cfg = ANYmalEvaluatorConfig(
        handcrafted_waypoint_file=None,  # "warehouse_paper",  #
        cost_map_dir="/home/pascal/viplanner/imperative_learning/data/warehouse_multiple_shelves_without_ppl_ext_sem_space",  #
        cost_map_name="cost_map_sem",
        models=[
            os.path.join(
                ASSETS_RESOURCES_DIR,
                "vip_models/plannernet_env2azQ1b91cZZ_ep100_inputDepSem_costSem_optimSGD_combi_more_data_neg05",
            ),
            os.path.join(
                ASSETS_RESOURCES_DIR, "vip_models/plannernet_env2azQ1b91cZZ_ep100_inputDep_costSem_optimSGD_depth_carla"
            ),
        ],
        multi_model=False,
        num_pairs=500,
        use_prev_results=True,
        repeat_waypoints=None,
    )
    cfg_carla = CarlaLoaderConfig(
        root_path="/home/pascal/viplanner/env/warehouse",
        usd_name="warehouse_multiple_shelves_without_ppl_ext_sem_space.usd",  #
        suffix="",
        prim_path="/World/Warehouse",
        scale=1.0,
        axis_up="Z",
        cw_config_file=None,
        sem_mesh_to_class_map=os.path.join(DATA_DIR, "warehouse", "keyword_mapping.yml"),
        groundplane=False,
        people_config_file=os.path.join(DATA_DIR, "warehouse", "people_cfg.yml"),
        vehicle_config_file=None,
    )
    cfg_carla_explore = CarlaExplorerConfig(
        nb_more_people=None,
        max_cam_recordings=100,
        points_per_m2=0.3,
        space_limiter="SM_WallA",
        carla_filter=None,
        indoor_filter=False,
    )
    cfg_planner = VIPlannerCfg(
        model_dir="/home/pascal/viplanner/imperative_learning/code/iPlanner/iplanner/models",
        # model_dir=os.path.join(
        #     ASSETS_RESOURCES_DIR,
        #     "vip_models/plannernet_env2azQ1b91cZZ_cam_mount_ep100_inputDepSem_costSem_optimSGD_new_cam_mount_combi_lossWidthMod_wgoal4.0_warehouse",
        # ),
        sem_origin="isaac",
        twist_controller_cfg=TwistControllerCfg(
            lookAheadDistance=1.2,
            stopDisThre=0.2,
        ),
        use_mount_cam=True,
        conv_dist=0.8,
        viplanner=False,
    )
    cfg_anymal = ANYmalCfg(
        anymal_type=1,  # 0: ANYmal C, 1: ANYmal D
        sensor=SensorCfg(
            cam_front_rgb=False,
            cam_front_depth=False,
            cam_viplanner_rgb=True,
            cam_viplanner_depth=True,
        ),
        rec_path=True,
        rec_sensor=False,
        follow_camera=True,
    )

    # init class
    run = ANYmalRunCarla(
        cfg=cfg,
        cfg_carla=cfg_carla,
        cfg_explore=cfg_carla_explore,
        cfg_anymal=cfg_anymal,
        cfg_planner=cfg_planner,
    )
    run.setup()

    if not cfg.multi_model and cfg.repeat_waypoints is None:
        run.run_single()
    elif not cfg.multi_model:
        run.run_repeat()
    else:
        run.run_multi()

    # Close the simulator
    launcher.close()

# EoF
