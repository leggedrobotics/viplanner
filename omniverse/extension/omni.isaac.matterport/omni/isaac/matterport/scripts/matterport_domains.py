# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from typing import Dict

import carb
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import omni
import torch
from omni.isaac.matterport.domains.matterport_raycast_camera import (
    MatterportRayCasterCamera,
)
from omni.isaac.lab.sensors.camera import CameraData
from omni.isaac.lab.sensors.ray_caster import RayCasterCfg
from omni.isaac.lab.sim import SimulationContext

from .ext_cfg import MatterportExtConfig

mpl.use("Qt5Agg")


class MatterportDomains:
    """
    Load Matterport3D Semantics and make them available to Isaac Sim
    """

    def __init__(self, cfg: MatterportExtConfig):
        """
        Initialize MatterportSemWarp

        Args:
            path (str): path to Matterport3D Semantics
        """
        self._cfg: MatterportExtConfig = cfg

        # setup camera list
        self.cameras: Dict[str, MatterportRayCasterCamera] = {}

        # setup camera visualization
        self.figures = {}

        # internal parameters
        self.callback_set = False
        self.vis_init = False
        self.prev_position = torch.zeros(3)
        self.prev_orientation = torch.zeros(4)

        # add callbacks for stage play/stop
        physx_interface = omni.physx.acquire_physx_interface()
        self._initialize_handle = physx_interface.get_simulation_event_stream_v2().create_subscription_to_pop_by_type(
            int(omni.physx.bindings._physx.SimulationEvent.RESUMED), self._initialize_callback
        )
        self._invalidate_initialize_handle = (
            physx_interface.get_simulation_event_stream_v2().create_subscription_to_pop_by_type(
                int(omni.physx.bindings._physx.SimulationEvent.STOPPED), self._invalidate_initialize_callback
            )
        )
        return

    ##
    # Public Methods
    ##

    def register_camera(self, cfg: RayCasterCfg):
        """
        Register a camera to the MatterportSemWarp
        """
        # append to camera list
        self.cameras[cfg.prim_path] = MatterportRayCasterCamera(cfg)

    ##
    # Callback Setup
    ##
    def _invalidate_initialize_callback(self, val):
        if self.callback_set:
            self._sim.remove_render_callback("matterport_update")
            self.callback_set = False

    def _initialize_callback(self, val):
        if self.callback_set:
            return

        #  check for camera
        if len(self.cameras) == 0:
            carb.log_warn("No cameras added! Add cameras first, then enable the callback!")
            return

        # get SimulationContext
        if SimulationContext.instance():
            self._sim: SimulationContext = SimulationContext.instance()
        else:
            carb.log_error("No Simulation Context found! Matterport Callback not attached!")

        # add callback
        self._sim.add_render_callback("matterport_update", callback_fn=self._update)
        self.callback_set = True

    ##
    # Callback Function
    ##

    def _update(self, dt: float):
        for camera in self.cameras.values():
            camera.update(dt.payload["dt"])

        if self._cfg.visualize:
            vis_prim = self._cfg.visualize_prim if self._cfg.visualize_prim else list(self.cameras.keys())[0]
            if torch.all(self.cameras[vis_prim].data.pos_w.cpu() == self.prev_position) and torch.all(
                self.cameras[vis_prim].data.quat_w_world.cpu() == self.prev_orientation
            ):
                return
            self._update_visualization(self.cameras[vis_prim].data)
            self.prev_position = self.cameras[vis_prim].data.pos_w.clone().cpu()
            self.prev_orientation = self.cameras[vis_prim].data.quat_w_world.clone().cpu()

    ##
    # Private Methods (Helper Functions)
    ##

    # Visualization helpers ###

    def _init_visualization(self, data: CameraData):
        """Initializes the visualization plane."""
        # init depth figure
        self.n_bins = 500  # Number of bins in the colormap
        self.color_array = mpl.colormaps["gist_rainbow"](np.linspace(0, 1, self.n_bins))  # Colormap

        if "semantic_segmentation" in data.output.keys():  # noqa: SIM118
            # init semantics figure
            fg_sem = plt.figure()
            ax_sem = fg_sem.gca()
            ax_sem.set_title("Semantic Segmentation")
            img_sem = ax_sem.imshow(data.output["semantic_segmentation"][0].cpu().numpy())
            self.figures["semantics"] = {"fig": fg_sem, "axis": ax_sem, "img": img_sem}

        if "distance_to_image_plane" in data.output.keys():  # noqa: SIM118
            # init semantics figure
            fg_depth = plt.figure()
            ax_depth = fg_depth.gca()
            ax_depth.set_title("Distance To Image Plane")
            img_depth = ax_depth.imshow(self.convert_depth_to_color(data.output["distance_to_image_plane"][0]))
            self.figures["depth"] = {"fig": fg_depth, "axis": ax_depth, "img": img_depth}

        if len(self.figures) > 0:
            plt.ion()
            # update flag
            self.vis_init = True

    def _update_visualization(self, data: CameraData) -> None:
        """
        Updates the visualization plane.
        """
        if self.vis_init is False:
            self._init_visualization(data)
        else:
            # SEMANTICS
            if "semantic_segmentation" in data.output.keys():  # noqa: SIM118
                self.figures["semantics"]["img"].set_array(data.output["semantic_segmentation"][0].cpu().numpy())
                self.figures["semantics"]["fig"].canvas.draw()
                self.figures["semantics"]["fig"].canvas.flush_events()

            # DEPTH
            if "distance_to_image_plane" in data.output.keys():  # noqa: SIM118
                # cam_data.img_depth.set_array(cam_data.render_depth)
                self.figures["depth"]["img"].set_array(
                    self.convert_depth_to_color(data.output["distance_to_image_plane"][0])
                )
                self.figures["depth"]["fig"].canvas.draw()
                self.figures["depth"]["fig"].canvas.flush_events()

        plt.pause(1e-6)

    def convert_depth_to_color(self, depth_img):
        depth_img = depth_img.cpu().numpy()
        depth_img[~np.isfinite(depth_img)] = depth_img.max()
        depth_img_flattend = np.clip(depth_img.flatten(), a_min=0, a_max=depth_img.max())
        depth_img_flattend = np.round(depth_img_flattend / depth_img.max() * (self.n_bins - 1)).astype(np.int32)
        depth_colors = self.color_array[depth_img_flattend]
        depth_colors = depth_colors.reshape(depth_img.shape[0], depth_img.shape[1], 4)
        return depth_colors
