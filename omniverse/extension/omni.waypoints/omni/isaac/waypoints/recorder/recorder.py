# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import json

# python
import os
from typing import List

import numpy as np

# omni
import omni

# isaac-debug
import omni.isaac.debug_draw._debug_draw as omni_debug_draw
import scipy.spatial.transform as tf

# isaac-core
from omni.isaac.core.objects import VisualCuboid
from pxr import UsdGeom


class Recorder:
    """
    Record arbitrary number of waypoints and save them as .json file
    """

    cube_scale = 100  # convert from meters to cm

    def __init__(self) -> None:
        # init buffers
        self.start_point: List[float] = [0.0] * 3
        self.end_point: List[float] = [0.0] * 3
        self.way_points: List[List[float]] = []

        # init params
        self.save_path: str = None
        self.file_name: str = "waypoints"

        # Acquire draw interface
        self.draw_interface = omni_debug_draw.acquire_debug_draw_interface()

        # cube
        self.cube = VisualCuboid(
            prim_path="/Waypoint",  # The prim path of the cube in the USD stage
            name="waypoint",  # The unique name used to retrieve the object from the scene later on
            position=np.array([0, 0, 1.0]),  # Using the current stage units which is in meters by default.
            scale=np.array([0.25, 0.25, 0.25]) * self.cube_scale,  # most arguments accept mainly numpy arrays.
            size=1.0,
            color=np.array([1, 0.4, 0]),  # RGB channels, going from 0-1
        )

        # identfy up axis of the stage (y or z)
        stage = omni.usd.get_context().get_stage()
        if UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.y:
            self.rot_mat = tf.Rotation.from_euler("XYZ", [90, 90, 0], degrees=True).as_matrix()
        elif UsdGeom.GetStageUpAxis(stage) == UsdGeom.Tokens.z:
            self.rot_mat = np.eye(3)
        else:
            raise ValueError("Stage Up Axis not supported")
        return

    def set_save_path(self, path: str) -> None:
        self.save_path = path
        return

    def set_filename(self, name) -> None:
        self.file_name = name.get_value_as_string()
        return

    def set_start_point(self) -> None:
        # get coordinates of the start
        start_point = self._get_cube_coords()
        # save start point with z-up axis
        self.start_point = np.matmul(self.rot_mat, start_point).tolist()
        # draw start point
        self.draw_interface.draw_points([start_point], [(0, 1, 0, 1)], [10])  # green
        return

    def add_way_point(self) -> None:
        # get coordinates of the cube
        way_point = self._get_cube_coords()
        # save way point with z-up axis
        self.way_points.append(np.matmul(self.rot_mat, way_point).tolist())
        # draw start point
        self.draw_interface.draw_points([way_point], [(0, 0, 1, 1)], [10])  # blue
        return

    def set_end_point(self) -> None:
        """
        Set the end point of the path and save all waypoints as .json file with the following structure:
        {
            start: [x, y, z],
            end: [x, y, z],
            waypoints: [[x, y, z], [x, y, z], ...]
        }
        All points are saved in the z-up axis convention.
        """

        # get coordinates of the end
        end_point = self._get_cube_coords()
        # save end point with z-up axis
        self.end_point = np.matmul(self.rot_mat, end_point).tolist()
        # draw start point
        self.draw_interface.draw_points([end_point], [(1, 0, 0, 1)], [10])  # red
        # save points
        if self.file_name.endswith(".json"):
            file_path = os.path.join(self.save_path, self.file_name)
        else:
            file_path = os.path.join(self.save_path, self.file_name + ".json")

        data = {"start": self.start_point, "end": self.end_point, "waypoints": self.way_points}
        with open(file_path, "w") as file:
            json.dump(data, file)
        return

    def reset(self) -> None:
        self.start_point = [0.0] * 3
        self.end_point = [0.0] * 3
        self.way_points = []
        self.draw_interface.clear_points()
        return

    """ Helper functions """

    def _get_cube_coords(self) -> np.ndarray:
        pose = omni.usd.utils.get_world_transform_matrix(self.cube.prim)
        pose = np.array(pose).T
        return pose[:3, 3]


# EoF
