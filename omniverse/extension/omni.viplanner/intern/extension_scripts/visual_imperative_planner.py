# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict, Optional

# python
import numpy as np
import torch

# omni-isaac-anymal
from omni.isaac.anymal.config import ANYmalCfg, ROSPublisherCfg, VIPlannerCfg
from omni.isaac.anymal.policy import Agent
from omni.isaac.anymal.utils import (
    AnymalROSPublisher,
    AnymalROSSubscriber,
    TwistController,
)
from omni.isaac.anymal.utils.ros_utils import check_roscore_running, init_rosnode
from omni.isaac.anymal.utils.twist_controller_new import TwistControllerNew
from omni.isaac.anymal.viplanner import VIPlanner

# omni-isaac-core
from omni.isaac.core.objects import VisualCuboid

# omni-isaac-orbit
from omni.isaac.orbit.robots.legged_robot import LeggedRobot
from omni.isaac.orbit.sensors.camera import Camera
from omni.isaac.orbit.sensors.height_scanner import HeightScanner
from tqdm import tqdm


class VIPlannerANYmal(Agent):
    """
    Visual Imperative Planner to guide ANYway to a waypoint defined by a cube in the world.

    Two versions available:
    - Isaac Twist Controller (default), Twist Controller is implemented in Python, no ROS exchange has to be done
    - ROS Twist Controller (old), Twist Controller is implemented in C++, path has to be published to ROS and twist command are received

    """

    def __init__(
        self,
        cfg: ANYmalCfg,
        camera_sensors: Dict[str, Camera],
        robot: LeggedRobot,
        height_scanner: HeightScanner,
        ros_controller: bool = False,
        planner_cfg: Optional[VIPlannerCfg] = None,
    ) -> None:
        # init agent
        super().__init__(cfg.rl_policy, robot, height_scanner)
        self._anymal_cfg = cfg
        self._camera_sensors = camera_sensors
        self._ros_controller = ros_controller

        # viplanner
        self.planner: VIPlanner = None
        # waypoint cube
        self.cube: VisualCuboid = None
        # planner cfg
        self._planner_cfg = planner_cfg if planner_cfg else VIPlannerCfg()

        if self._ros_controller:
            # init planner config
            self._planner_cfg.ros_pub = True
            # init ROS publisher config
            self._ros_publisher_cfg = ROSPublisherCfg(sensor_pub=False)
            # setup cube as waypoint and ros connection
            self.ros_publisher: AnymalROSPublisher = None
            self.ros_subscriber: AnymalROSSubscriber = None
        else:
            self._planner_cfg.ros_pub = False
            self.twist: TwistController = None

        self._setup()

        # reset once at initialization
        self.reset()

        # get message
        self.title += "with Visual Imperative Planner \n"
        self.msg += "\n\n"
        self.msg += f""  # TODO: add more info
        return

    def compute_command_ros(self, step_size: float) -> None:
        """Compute the command for the robot using the ROS Twist Controller"""
        # get command from joystick planner
        last_command, command_time = self.twist.get_command()
        # check if last command is not too long ago (would happen if goal is reached)
        if command_time > (self.sim.current_time - self._planner_cfg.look_back_factor * step_size):
            return torch.tensor(last_command, device=self.robot.device)
        else:
            return torch.zeros(3, device=self.robot.device)

    def compute_command_isaac(self, step_size: float) -> None:
        """Compute the command for the robot using the Python Twist Controller"""
        # get command from twist controller
        last_command = self.twist.compute(self.planner.traj_waypoints_odom, self.planner.fear)
        try:
            return torch.tensor(last_command, device=self.robot.device)
        except TypeError:
            return torch.zeros(3, device=self.robot.device)

    def reset(self) -> None:
        super().reset()
        self.planner.reset()
        if not self._ros_controller:
            self.twist.reset()
        # reset pbar
        self.pbar.reset()
        return

    ##
    # Helper Functions
    ##

    def _setup(self) -> None:
        """Setup cube and the ros connection to the smart joystick"""
        # cube
        self._setup_cube()
        # viplanner
        self.planner = VIPlanner(
            anymal_cfg=self._anymal_cfg, vip_cfg=self._planner_cfg, camera_sensors=self._camera_sensors
        )
        # for ROS based controller
        if self._ros_controller:
            # init rosnode
            check_roscore_running()
            init_rosnode("anymal_node")
            # init publisher and subscriber
            self.ros_publisher = AnymalROSPublisher(
                anymal_cfg=self._anymal_cfg,
                ros_cfg=self._ros_publisher_cfg,
                camera_sensors=self._camera_sensors,
                lidar_sensors=self._lidar_sensors,
            )
            self.twist = AnymalROSSubscriber()
            # define function to compute command
            self.compute_command = self.compute_command_ros
        else:
            # self.twist = TwistController(
            #     cfg=self._planner_cfg.twist_controller_cfg,
            #     cfg_vip=self._planner_cfg,
            #     cfg_anymal=self._anymal_cfg,
            #     camera_sensors=self._camera_sensors,
            # )
            self.twist = TwistControllerNew(
                cfg=self._planner_cfg.twist_controller_cfg,
                cfg_vip=self._planner_cfg,
                cfg_anymal=self._anymal_cfg,
                robot=self.robot,
            )
            # define function to compute command
            self.compute_command = self.compute_command_isaac
        # setup pbar
        self._setup_pbar()
        return

    def _setup_cube(self) -> None:
        """cube as the definition of a goalpoint"""
        self.cube = VisualCuboid(
            prim_path=self._planner_cfg.goal_prim,  # The prim path of the cube in the USD stage
            name="waypoint",  # The unique name used to retrieve the object from the scene later on
            position=np.array([5, 0, 1.0]),  # Using the current stage units which is in meters by default.
            scale=np.array([0.15, 0.15, 0.15]),  # most arguments accept mainly numpy arrays.
            size=1.0,
            color=np.random.uniform((1, 0, 0)),  # RGB channels, going from 0-1
        )
        return

    # progress bar
    def _setup_pbar(self):
        """Setup progress bar"""
        self.pbar = tqdm(total=100, position=0, leave=False, bar_format="{desc}{percentage:.0f}%|{bar}|")
        return

    def _update_pbar(self):
        """Update progress bar"""
        if self.planner.is_reset:
            return

        desc = (
            f"Time Elapsed: {self.sim.current_time - self.planner.start_time:.2f}s | "
            f"Walked Distance: {self.planner.max_goal_distance-self.planner.distance_to_goal:.2f}/{self.planner.max_goal_distance:.2f}m | "
            f"Twist: {self.twist.twist}"
        )
        self.pbar.set_description(desc)

        percentage_completed_path = (1 - (self.planner.distance_to_goal / self.planner.max_goal_distance)) * 100
        update_percentage = percentage_completed_path - self.pbar.n
        if update_percentage > 0:
            self.pbar.update(update_percentage)
        else:
            self.pbar.update(0)
        return


# EoF
