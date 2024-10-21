# Copyright (c) 2024 ETH Zurich (Robotic Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import asyncio
import gc

# python
import os

import carb

# omni
import omni
import omni.client
import omni.ext
import omni.isaac.core.utils.prims as prim_utils
import omni.isaac.core.utils.stage as stage_utils

# isaac-core
import omni.ui as ui
from omni.isaac.matterport.domains import MatterportImporter
from omni.isaac.lab.sensors.ray_caster import RayCasterCfg, patterns
from omni.isaac.lab.sim import SimulationCfg, SimulationContext

# omni-isaac-ui
from omni.isaac.ui.ui_utils import (
    btn_builder,
    cb_builder,
    dropdown_builder,
    float_builder,
    get_style,
    int_builder,
    setup_ui_headers,
    str_builder,
)

# omni-isaac-matterport
from .ext_cfg import MatterportExtConfig
from .matterport_domains import MatterportDomains

EXTENSION_NAME = "Matterport Importer"


def is_mesh_file(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in [".obj", ".usd"]


def is_ply_file(path: str) -> bool:
    _, ext = os.path.splitext(path.lower())
    return ext in [".ply"]


def on_filter_obj_item(item) -> bool:
    if not item or item.is_folder:
        return not (item.name == "Omniverse" or item.path.startswith("omniverse:"))
    return is_mesh_file(item.path)


def on_filter_ply_item(item) -> bool:
    if not item or item.is_folder:
        return not (item.name == "Omniverse" or item.path.startswith("omniverse:"))
    return is_ply_file(item.path)


class MatterPortExtension(omni.ext.IExt):
    """Extension to load Matterport 3D Environments into Isaac Sim"""

    def on_startup(self, ext_id):
        self._ext_id = ext_id
        self._usd_context = omni.usd.get_context()
        self._window = omni.ui.Window(
            EXTENSION_NAME, width=400, height=500, visible=True, dockPreference=ui.DockPreference.LEFT_BOTTOM
        )

        # init config class and get path to extension
        self._config = MatterportExtConfig()
        self._extension_path = omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id)

        # set additional parameters
        self._input_fields: dict = {}  # dictionary to store values of buttion, float fields, etc.
        self.domains: MatterportDomains = None  # callback class for semantic rendering
        self.ply_proposal: str = ""
        # build ui
        self.build_ui()
        return

    ##
    # UI Build functions
    ##

    def build_ui(self, build_cam: bool = False, build_viz: bool = False):
        with self._window.frame:
            with ui.VStack(spacing=5, height=0):
                self._build_info_ui()

                self._build_import_ui()

                if build_cam:
                    self._build_camera_ui()

                if build_viz:
                    self._build_viz_ui()

        async def dock_window():
            await omni.kit.app.get_app().next_update_async()

            def dock(space, name, location, pos=0.5):
                window = omni.ui.Workspace.get_window(name)
                if window and space:
                    window.dock_in(space, location, pos)
                return window

            tgt = ui.Workspace.get_window("Viewport")
            dock(tgt, EXTENSION_NAME, omni.ui.DockPosition.LEFT, 0.33)
            await omni.kit.app.get_app().next_update_async()

        self._task = asyncio.ensure_future(dock_window())

    def _build_info_ui(self):
        title = EXTENSION_NAME
        doc_link = "https://github.com/leggedrobotics/omni_isaac_orbit"

        overview = "This utility is used to import Matterport3D Environments into Isaac Sim. "
        overview += "The environment and additional information are available at https://github.com/niessner/Matterport"
        overview += "\n\nPress the 'Open in IDE' button to view the source code."

        setup_ui_headers(self._ext_id, __file__, title, doc_link, overview)
        return

    def _build_import_ui(self):
        frame = ui.CollapsableFrame(
            title="Import Dataset",
            height=0,
            collapsed=False,
            style=get_style(),
            style_type_name_override="CollapsableFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # PhysicsMaterial
                self._input_fields["friction_dynamic"] = float_builder(
                    "Dynamic Friction",
                    default_val=self._config.importer.physics_material.dynamic_friction,
                    tooltip=f"Sets the dynamic friction of the physics material (default: {self._config.importer.physics_material.dynamic_friction})",
                )
                self._input_fields["friction_dynamic"].add_value_changed_fn(
                    lambda m, config=self._config: config.set_friction_dynamic(m.get_value_as_float())
                )
                self._input_fields["friction_static"] = float_builder(
                    "Static Friction",
                    default_val=self._config.importer.physics_material.static_friction,
                    tooltip=f"Sets the static friction of the physics material (default: {self._config.importer.physics_material.static_friction})",
                )
                self._input_fields["friction_static"].add_value_changed_fn(
                    lambda m, config=self._config: config.set_friction_static(m.get_value_as_float())
                )
                self._input_fields["restitution"] = float_builder(
                    "Restitution",
                    default_val=self._config.importer.physics_material.restitution,
                    tooltip=f"Sets the restitution of the physics material (default: {self._config.importer.physics_material.restitution})",
                )
                self._input_fields["restitution"].add_value_changed_fn(
                    lambda m, config=self._config: config.set_restitution(m.get_value_as_float())
                )
                friction_restitution_options = ["average", "min", "multiply", "max"]
                dropdown_builder(
                    "Friction Combine Mode",
                    items=friction_restitution_options,
                    default_val=friction_restitution_options.index(
                        self._config.importer.physics_material.friction_combine_mode
                    ),
                    on_clicked_fn=lambda mode_str, config=self._config: config.set_friction_combine_mode(mode_str),
                    tooltip=f"Sets the friction combine mode of the physics material (default: {self._config.importer.physics_material.friction_combine_mode})",
                )
                dropdown_builder(
                    "Restitution Combine Mode",
                    items=friction_restitution_options,
                    default_val=friction_restitution_options.index(
                        self._config.importer.physics_material.restitution_combine_mode
                    ),
                    on_clicked_fn=lambda mode_str, config=self._config: config.set_restitution_combine_mode(mode_str),
                    tooltip=f"Sets the friction combine mode of the physics material (default: {self._config.importer.physics_material.restitution_combine_mode})",
                )
                cb_builder(
                    label="Improved Patch Friction",
                    tooltip=f"Sets the improved patch friction of the physics material (default: {self._config.importer.physics_material.improve_patch_friction})",
                    on_clicked_fn=lambda m, config=self._config: config.set_improved_patch_friction(m),
                    default_val=self._config.importer.physics_material.improve_patch_friction,
                )

                # Set prim path for environment
                self._input_fields["prim_path"] = str_builder(
                    "Prim Path of the Environment",
                    tooltip="Prim path of the environment",
                    default_val=self._config.importer.prim_path,
                )
                self._input_fields["prim_path"].add_value_changed_fn(
                    lambda m, config=self._config: config.set_prim_path(m.get_value_as_string())
                )

                # read import location
                def check_file_type(model=None):
                    path = model.get_value_as_string()
                    if is_mesh_file(path):
                        self._input_fields["import_btn"].enabled = True
                        self._make_ply_proposal(path)
                        self._config.set_obj_filepath(path)
                    else:
                        self._input_fields["import_btn"].enabled = False
                        carb.log_warn(f"Invalid path to .obj file: {path}")

                kwargs = {
                    "label": "Input File",
                    "default_val": self._config.importer.obj_filepath,
                    "tooltip": "Click the Folder Icon to Set Filepath",
                    "use_folder_picker": True,
                    "item_filter_fn": on_filter_obj_item,
                    "bookmark_label": "Included Matterport3D meshs",
                    "bookmark_path": f"{self._extension_path}/data/mesh",
                    "folder_dialog_title": "Select .obj File",
                    "folder_button_title": "*.obj, *.usd",
                }
                self._input_fields["input_file"] = str_builder(**kwargs)
                self._input_fields["input_file"].add_value_changed_fn(check_file_type)

                self._input_fields["import_btn"] = btn_builder(
                    "Import", text="Import", on_clicked_fn=self._start_loading
                )
                self._input_fields["import_btn"].enabled = False

        return

    def _build_camera_ui(self):
        frame = ui.CollapsableFrame(
            title="Add Camera",
            height=0,
            collapsed=False,
            style=get_style(),
            style_type_name_override="CollapsableFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # get import location and save directory
                kwargs = {
                    "label": "Input ply File",
                    "default_val": self.ply_proposal,
                    "tooltip": "Click the Folder Icon to Set Filepath",
                    "use_folder_picker": True,
                    "item_filter_fn": on_filter_ply_item,
                    "bookmark_label": "Included Matterport3D Point-Cloud with semantic labels",
                    "bookmark_path": f"{self._extension_path}/data/mesh",
                    "folder_dialog_title": "Select .ply Point-Cloud File",
                    "folder_button_title": "Select .ply Point-Cloud",
                }
                self._input_fields["input_ply_file"] = str_builder(**kwargs)

                # data fields parameters
                self._input_fields["camera_semantics"] = cb_builder(
                    label="Enable Semantics",
                    tooltip="Enable access to the semantics information of the mesh (default: True)",
                    default_val=True,
                )
                self._input_fields["camera_depth"] = cb_builder(
                    label="Enable Distance to Camera Frame",
                    tooltip="Enable access to the depth information of the mesh - no additional compute effort (default: True)",
                    default_val=True,
                )

                # add camera sensor for which semantics and depth should be rendered
                kwargs = {
                    "label": "Camera Prim Path",
                    "type": "stringfield",
                    "default_val": "",
                    "tooltip": "Enter Camera Prim Path",
                    "use_folder_picker": False,
                }
                self._input_fields["camera_prim"] = str_builder(**kwargs)
                self._input_fields["camera_prim"].add_value_changed_fn(self.activate_load_camera)

                self._input_fields["cam_height"] = int_builder(
                    "Camera Height in Pixels",
                    default_val=480,
                    tooltip="Set the height of the camera image plane in pixels (default: 480)",
                )

                self._input_fields["cam_width"] = int_builder(
                    "Camera Width in Pixels",
                    default_val=640,
                    tooltip="Set the width of the camera image plane in pixels (default: 640)",
                )

                self._input_fields["load_camera"] = btn_builder(
                    "Add Camera", text="Add Camera", on_clicked_fn=self._register_camera
                )
                self._input_fields["load_camera"].enabled = False
        return

    def _build_viz_ui(self):
        frame = ui.CollapsableFrame(
            title="Visualization",
            height=0,
            collapsed=False,
            style=get_style(),
            style_type_name_override="CollapsableFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                cb_builder(
                    label="Visualization",
                    tooltip=f"Visualize Semantics and/or Depth (default: {self._config.visualize})",
                    on_clicked_fn=lambda m, config=self._config: config.set_visualize(m),
                    default_val=self._config.visualize,
                )
                dropdown_builder(
                    "Shown Camera Prim",
                    items=list(self.domains.cameras.keys()),
                    default_val=0,
                    on_clicked_fn=lambda mode_str, config=self._config: config.set_visualization_prim(mode_str),
                    tooltip="Select the camera prim shown in the visualization window",
                )

    ##
    # Shutdown Helpers
    ##

    def on_shutdown(self):
        if self._window:
            self._window = None
        gc.collect()
        stage_utils.clear_stage()

        if self.domains is not None and self.domains.callback_set:
            self.domains.set_domain_callback(True)

    ##
    # Path Helpers
    ##
    def _make_ply_proposal(self, path: str) -> None:
        """use default matterport datastructure to make proposal about point-cloud file

        - "env_id"
            - matterport_mesh
                - "id_nbr"
                    - "id_nbr".obj
            - house_segmentations
                - "env_id".ply

        """
        file_dir, file_name = os.path.split(path)
        ply_dir = os.path.join(file_dir, "../..", "house_segmentations")
        env_id = file_dir.split("/")[-3]
        try:
            ply_file = os.path.join(ply_dir, f"{env_id}.ply")
            os.path.isfile(ply_file)
            carb.log_verbose(f"Found ply file: {ply_file}")
            self.ply_proposal = ply_file
        except FileNotFoundError:
            carb.log_verbose("No ply file found in default matterport datastructure")

    ##
    # Load Mesh and Point-Cloud
    ##

    async def load_matterport(self):
        # simulation settings
        # check if simulation context was created earlier or not.
        if SimulationContext.instance():
            SimulationContext.clear_instance()
            carb.log_warn("SimulationContext already loaded. Will clear now and init default SimulationContext")

        # create new simulation context
        self.sim = SimulationContext(SimulationCfg())
        # initialize simulation
        await self.sim.initialize_simulation_context_async()
        # load matterport
        self._matterport = MatterportImporter(self._config.importer)
        await self._matterport.load_world_async()

        # reset the simulator
        # note: this plays the simulator which allows setting up all the physics handles.
        await self.sim.reset_async()
        await self.sim.pause_async()

    def _start_loading(self):
        path = self._config.importer.obj_filepath
        if not path:
            return

        # find obj, usd file
        if os.path.isabs(path):
            file_path = path
            assert os.path.isfile(file_path), f"No .obj or .usd file found under absolute path: {file_path}"
        else:
            file_path = os.path.join(self._extension_path, "data", path)
            assert os.path.isfile(
                file_path
            ), f"No .obj or .usd file found under relative path to extension data: {file_path}"
            self._config.set_obj_filepath(file_path)  # update config
        carb.log_verbose("MatterPort 3D Mesh found, start loading...")

        asyncio.ensure_future(self.load_matterport())

        carb.log_info("MatterPort 3D Mesh loaded")
        self.build_ui(build_cam=True)
        self._input_fields["import_btn"].enabled = False

    ##
    # Register Cameras
    ##

    def activate_load_camera(self, val):
        self._input_fields["load_camera"].enabled = True

    def _register_camera(self):
        ply_filepath = self._input_fields["input_ply_file"].get_value_as_string()
        if not is_ply_file(ply_filepath):
            carb.log_error("Given ply path is not valid! No camera created!")

        camera_path = self._input_fields["camera_prim"].get_value_as_string()
        if not prim_utils.is_prim_path_valid(camera_path):  # create prim if no prim found
            prim_utils.create_prim(camera_path, "Xform")

        camera_semantics = self._input_fields["camera_semantics"].get_value_as_bool()
        camera_depth = self._input_fields["camera_depth"].get_value_as_bool()
        camera_width = self._input_fields["cam_width"].get_value_as_int()
        camera_height = self._input_fields["cam_height"].get_value_as_int()

        # Setup camera sensor
        data_types = []
        if camera_semantics:
            data_types += ["semantic_segmentation"]
        if camera_depth:
            data_types += ["distance_to_image_plane"]

        camera_pattern_cfg = patterns.PinholeCameraPatternCfg(
            focal_length=24.0,
            horizontal_aperture=20.955,
            height=camera_height,
            width=camera_width,
            data_types=data_types,
        )
        camera_cfg = RayCasterCfg(
            prim_path=camera_path,
            mesh_prim_paths=[ply_filepath],
            update_period=0,
            offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(1.0, 0.0, 0.0, 0.0)),
            debug_vis=True,
            pattern_cfg=camera_pattern_cfg,
        )

        if self.domains is None:
            self.domains = MatterportDomains(self._config)
        # register camera
        self.domains.register_camera(camera_cfg)

        # initialize physics handles
        self.sim.reset()

        # allow for tasks
        self.build_ui(build_cam=True, build_viz=True)
        return
