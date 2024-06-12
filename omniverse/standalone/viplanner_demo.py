# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
This script demonstrates how to use the rigid objects class.
"""

"""Launch Isaac Sim Simulator first."""

import argparse

# omni-isaac-orbit
from omni.isaac.orbit.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="This script demonstrates how to use the camera sensor.")
parser.add_argument("--headless", action="store_true", default=False, help="Force display off at all times.")
parser.add_argument("--conv_distance", default=0.2, type=float, help="Distance for a goal considered to be reached.")
parser.add_argument(
    "--scene", default="warehouse", choices=["matterport", "carla", "warehouse"], type=str, help="Scene to load."
)
parser.add_argument("--beat_the_planner", default=True, action="store_true", help="Beat the planner Demo.")
parser.add_argument("--model_dir", default="/home/pascal/Downloads", type=str, help="Path to model directory.")
args_cli = parser.parse_args()

# launch omniverse app
if args_cli.beat_the_planner:
    app_launcher = AppLauncher(headless=args_cli.headless, ros=1)
else:
    app_launcher = AppLauncher(headless=args_cli.headless)
simulation_app = app_launcher.app

"""Rest everything follows."""
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.orbit.utils.math as math_utils
import torch
from omni.isaac.core.objects import VisualCuboid
from omni.isaac.orbit.envs import RLTaskEnv
from omni.viplanner.config import (
    ViPlannerCarlaCfg,
    ViPlannerMatterportCfg,
    ViPlannerWarehouseCfg,
)
from omni.viplanner.config.beat_the_planner_cfg import (
    BeatThePlannerCarlaCfg,
    BeatThePlannerMatterportCfg,
    BeatThePlannerWarehouseCfg,
)
from omni.viplanner.viplanner import VIPlannerAlgo
from pxr import UsdGeom

if args_cli.beat_the_planner:
    import time

    import cv2
    import numpy as np
    import rospy
    from cv_bridge import CvBridge
    from pyfiglet import Figlet
    from sensor_msgs.msg import CompressedImage, Image

"""
Main
"""


def main():
    """Imports all legged robots supported in Orbit and applies zero actions."""

    # create environment cfg
    if args_cli.scene == "matterport":
        env_cfg = ViPlannerMatterportCfg() if not args_cli.beat_the_planner else BeatThePlannerMatterportCfg()
        goal_pos = torch.tensor([7.0, -12.2, 1.0])
    elif args_cli.scene == "carla":
        env_cfg = ViPlannerCarlaCfg() if not args_cli.beat_the_planner else BeatThePlannerCarlaCfg()
        goal_pos = torch.tensor([137, 111.0, 1.0])
    elif args_cli.scene == "warehouse":
        env_cfg = ViPlannerWarehouseCfg() if not args_cli.beat_the_planner else BeatThePlannerWarehouseCfg()
        goal_pos = torch.tensor([3, -4.5, 1.0])
    else:
        raise NotImplementedError(f"Scene {args_cli.scene} not yet supported!")

    # create environment
    env = RLTaskEnv(env_cfg)

    # adjust the intrinsics of the camera
    depth_intrinsic = torch.tensor([[430.31607, 0.0, 428.28408], [0.0, 430.31607, 244.00695], [0.0, 0.0, 1.0]])
    env.scene.sensors["depth_camera"].set_intrinsic_matrices(matrices=depth_intrinsic.repeat(env.num_envs, 1, 1))
    semantic_intrinsic = torch.tensor([[644.15496, 0.0, 639.53125], [0.0, 643.49212, 366.30880], [0.0, 0.0, 1.0]])
    env.scene.sensors["semantic_camera"].set_intrinsic_matrices(matrices=semantic_intrinsic.repeat(env.num_envs, 1, 1))

    # Make sure that groundplane is invisible
    if args_cli.scene == "carla":
        assert (
            prim_utils.get_prim_at_path("/World/GroundPlane").GetAttribute("visibility").Set(UsdGeom.Tokens.invisible)
        )

    # reset the environment
    with torch.inference_mode():
        obs = env.reset()[0]

    # set goal cube
    VisualCuboid(
        prim_path="/World/goal",  # The prim path of the cube in the USD stage
        name="waypoint",  # The unique name used to retrieve the object from the scene later on
        position=goal_pos,  # Using the current stage units which is in meters by default.
        scale=torch.tensor([0.15, 0.15, 0.15]),  # most arguments accept mainly numpy arrays.
        size=1.0,
        color=torch.tensor([1, 0, 0]),  # RGB channels, going from 0-1
    )
    goal_pos = prim_utils.get_prim_at_path("/World/goal").GetAttribute("xformOp:translate")

    # pause the simulator
    env.sim.pause()

    # load viplanner
    viplanner = VIPlannerAlgo(model_dir=args_cli.model_dir)

    goals = torch.tensor(goal_pos.Get(), device=env.device).repeat(env.num_envs, 1)

    # initial paths
    _, paths, fear = viplanner.plan_dual(
        obs["planner_image"]["depth_measurement"], obs["planner_image"]["semantic_measurement"], goals
    )

    if args_cli.beat_the_planner:
        # rosnode to publish images
        rospy.init_node("beat_the_planner", anonymous=False)
        # viz semantic image
        sem_pub = rospy.Publisher("/beat_the_planner/sem_image/compressed", CompressedImage, queue_size=1)
        depth_pub = rospy.Publisher("/beat_the_planner/depth_image", Image, queue_size=1)
        # cvBridge for depth image
        cv_bridge = CvBridge()

        # init figlet for printing
        f = Figlet(font="slant")

    # Simulate physics
    while simulation_app.is_running():
        with torch.inference_mode():
            # If simulation is paused, then skip.
            if not env.sim.is_playing():
                env.sim.step(render=~args_cli.headless)
                continue

            obs = env.step(action=paths.view(paths.shape[0], -1))[0]

        if args_cli.beat_the_planner:
            # check if goal is reached
            env_idx = torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
            env_idx[env_cfg.actions.paths.gamepad_controlled_robot_id] = False
            distance = torch.norm((obs["planner_transform"]["cam_position"] - goals)[:, :2], dim=1)
            if distance[env_idx] < args_cli.conv_distance:
                print(f.renderText("PLANNER WON"))
                print("PLANNER has reached the Goal before you! Try AGAIN!")
                env.reset()
                env.sim.pause()
                continue
            if distance[~env_idx] < args_cli.conv_distance:
                print(f.renderText("YOU WON"))
                print("YOU have reached the Goal before the planner! Lets try with a more difficult goal!")
                env.reset()
                env.sim.pause()
                continue

        # apply planner
        goals = torch.tensor(goal_pos.Get(), device=env.device).repeat(env.num_envs, 1)
        # if torch.any(
        #     torch.norm(obs["planner_transform"]["cam_position"] - goals)
        #     > viplanner.train_config.data_cfg[0].max_goal_distance
        # ):
        #     print(
        #         f"[WARNING]: Max goal distance is {viplanner.train_config.data_cfg[0].max_goal_distance} but goal is {torch.norm(obs['planner_transform']['cam_position'] - goals)} away from camera position! Please select new goal!"
        #     )
        #     env.sim.pause()
        #     continue

        goal_cam_frame = viplanner.goal_transformer(
            goals, obs["planner_transform"]["cam_position"], obs["planner_transform"]["cam_orientation"]
        )
        _, paths, fear = viplanner.plan_dual(
            obs["planner_image"]["depth_measurement"], obs["planner_image"]["semantic_measurement"], goal_cam_frame
        )
        paths = viplanner.path_transformer(
            paths, obs["planner_transform"]["cam_position"], obs["planner_transform"]["cam_orientation"]
        )

        if args_cli.beat_the_planner:
            # set gamepad controlled path to 0
            paths[env_cfg.actions.paths.gamepad_controlled_robot_id] *= 0.0

        # draw path
        viplanner.debug_draw(paths, fear, goals)

        if args_cli.beat_the_planner:
            start_time = time.time()

            # overlay the goal on the image
            goal_cam_frame = (
                goals[env_cfg.actions.paths.gamepad_controlled_robot_id]
                - obs["planner_transform"]["cam_position"][env_cfg.actions.paths.gamepad_controlled_robot_id]
            ) @ math_utils.matrix_from_quat(
                obs["planner_transform"]["cam_orientation"][env_cfg.actions.paths.gamepad_controlled_robot_id]
            )

            mat = torch.tensor([[1.0, 0, 0], [0, -1, 0], [0, 0, -1]])
            rotm = mat @ math_utils.matrix_from_euler(torch.tensor([torch.pi / 2, -torch.pi / 2, 0]), "XYZ").T
            goal_cam_frame_conv = goal_cam_frame @ rotm.to(goal_cam_frame.device).T

            goal_depth_pixel_frame = depth_intrinsic.to(goal_cam_frame.device) @ goal_cam_frame_conv.T
            goal_depth_pixel_frame = goal_depth_pixel_frame / goal_depth_pixel_frame[2]
            goal_sem_pixel_frame = semantic_intrinsic.to(goal_cam_frame.device) @ goal_cam_frame_conv.T
            goal_sem_pixel_frame = goal_sem_pixel_frame / goal_sem_pixel_frame[2]

            # mark the goal pixel
            depth_mark_pixel_range: int = 10
            sem_mark_pixel_range: int = 15
            sem_image = (
                obs["planner_image"]["semantic_measurement"][env_cfg.actions.paths.gamepad_controlled_robot_id]
                .permute(1, 2, 0)
                .cpu()
                .numpy()
            )
            sem_image[
                goal_sem_pixel_frame[1].long()
                - sem_mark_pixel_range : goal_sem_pixel_frame[1].long()
                + sem_mark_pixel_range,
                goal_sem_pixel_frame[0].long()
                - sem_mark_pixel_range : goal_sem_pixel_frame[0].long()
                + sem_mark_pixel_range,
            ] = [255, 0, 0]
            depth_image = (
                obs["planner_image"]["depth_measurement"][env_cfg.actions.paths.gamepad_controlled_robot_id, 0]
                .cpu()
                .numpy()
            )
            depth_image[
                goal_depth_pixel_frame[1].long()
                - depth_mark_pixel_range : goal_depth_pixel_frame[1].long()
                + depth_mark_pixel_range,
                goal_depth_pixel_frame[0].long()
                - depth_mark_pixel_range : goal_depth_pixel_frame[0].long()
                + depth_mark_pixel_range,
            ] = 0

            # resize the images
            sem_image = cv2.resize(sem_image, (480, 360))
            sem_image = cv2.cvtColor(sem_image, cv2.COLOR_RGB2BGR)
            depth_image = cv2.resize(depth_image, (480, 360))

            # Convert the image to JPEG format
            success, compressed_sem_image = cv2.imencode(".jpg", sem_image)
            if not success:
                rospy.logerr("Failed to compress semantic image")
                return

            # create compressed image and publish it
            sem_msg = CompressedImage()
            sem_msg.format = "jpeg"
            sem_msg.data = np.array(compressed_sem_image).tobytes()
            sem_pub.publish(sem_msg)

            depth_image = np.uint16(depth_image * 1000)
            depth_msg = cv_bridge.cv2_to_imgmsg(depth_image, "16UC1")
            depth_pub.publish(depth_msg)
            print(f"Publishing images took {time.time() - start_time} seconds")


if __name__ == "__main__":
    # Run the main function
    main()
    # Close the simulator
    simulation_app.close()
