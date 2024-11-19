# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Collect Training Data for ViPlanner
"""

"""Launch Isaac Sim Simulator first."""

import argparse

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Data collection for ViPlanner.")
parser.add_argument("--num_envs", type=int, default=2, help="Number of environments to spawn.")
parser.add_argument(
    "--scene", default="warehouse", choices=["matterport", "carla", "warehouse"], type=str, help="Scene to load."
)
parser.add_argument("--num_samples", type=int, default=10000, help="Number of samples to generate")

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()
args_cli.enable_cameras = True
# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""
import omni.isaac.lab.sim as sim_utils
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils.timer import Timer
from omni.viplanner.collectors import ViewpointSampling, ViewpointSamplingCfg
from omni.viplanner.config import (
    CarlaSemanticCostMapping,
    MatterportSemanticCostMapping,
)
from omni.viplanner.config.carla_cfg import TerrainSceneCfg as CarlaTerrainSceneCfg
from omni.viplanner.config.matterport_cfg import (
    TerrainSceneCfg as MatterportTerrainSceneCfg,
)
from omni.viplanner.config.warehouse_cfg import (
    TerrainSceneCfg as WarehouseTerrainSceneCfg,
)


def main():
    """Main function to start the data collection in different environments."""
    # setup sampling config
    cfg = ViewpointSamplingCfg()
    cfg.terrain_analysis.raycaster_sensor = "depth_camera"

    # create environment cfg and modify the collector config depending on the environment
    if args_cli.scene == "matterport":
        # NOTE: only one env possible as the prims for the cameras cannot be initialized with the env regex
        scene_cfg = MatterportTerrainSceneCfg(1, env_spacing=1.0)
        # overwrite semantic cost mapping and adjust parameters based on larger map
        cfg.terrain_analysis.semantic_cost_mapping = MatterportSemanticCostMapping()
    elif args_cli.scene == "carla":
        scene_cfg = CarlaTerrainSceneCfg(args_cli.num_envs, env_spacing=1.0)
        scene_cfg.terrain.groundplane = False
        # overwrite semantic cost mapping and adjust parameters based on larger map
        cfg.terrain_analysis.semantic_cost_mapping = CarlaSemanticCostMapping()
        cfg.terrain_analysis.grid_resolution = 1.0
        cfg.terrain_analysis.sample_points = 10000
        # limit space to be within the road network
        cfg.terrain_analysis.dim_limiter_prim = "Road_Sidewalk"
    elif args_cli.scene == "warehouse":
        scene_cfg = WarehouseTerrainSceneCfg(args_cli.num_envs, env_spacing=1.0)
        # overwrite semantic cost mapping
        cfg.terrain_analysis.semantic_cost_mapping = CarlaSemanticCostMapping()
        # limit space to be within the road network
        cfg.terrain_analysis.dim_limiter_prim = "Section"  # name of the meshes of the walls
    else:
        raise NotImplementedError(f"Scene {args_cli.scene} not yet supported!")

    # remove elements not necessary for the data collection
    scene_cfg.robot = None
    scene_cfg.height_scanner = None
    scene_cfg.contact_forces = None

    # change the path to the semantic cameras as the robot base frame does not exist anymore
    if args_cli.scene == "warehouse" or args_cli.scene == "carla":
        scene_cfg.depth_camera.prim_path = "{ENV_REGEX_NS}/depth_cam"
        scene_cfg.semantic_camera.prim_path = "{ENV_REGEX_NS}/sem_cam"
    else:
        scene_cfg.depth_camera.prim_path = "/World/matterport"
        scene_cfg.semantic_camera.prim_path = "/World/matterport"
    cfg.cameras = {
        "depth_camera": "distance_to_image_plane",
        "semantic_camera": "semantic_segmentation",
    }

    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg()
    sim = SimulationContext(sim_cfg)

    # generate scene
    with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
        scene = InteractiveScene(scene_cfg)
    print("[INFO]: Scene manager: ", scene)
    with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
        sim.reset()

    explorer = ViewpointSampling(cfg, scene)
    # Now we are ready!
    print("[INFO]: Setup complete...")

    # sample and render viewpoints
    samples = explorer.sample_viewpoints(args_cli.num_samples)
    explorer.render_viewpoints(samples)
    print("[INFO]: Viewpoints sampled.")

    if not args_cli.headless:
        print("Rendering will continue to render the environment and visualize the last camera positions.")
        # Define simulation stepping
        sim_dt = sim.get_physics_dt()
        # Simulation loop
        while simulation_app.is_running():
            # Perform step
            sim.render()
            # Update buffers
            explorer.scene.update(sim_dt)


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
