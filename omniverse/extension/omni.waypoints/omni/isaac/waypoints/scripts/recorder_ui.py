# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
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

# isaac-core
import omni.ui as ui

# omni-isaac-ui
from omni.isaac.ui.ui_utils import btn_builder, get_style, setup_ui_headers, str_builder

# isaac-waypoints
from omni.isaac.waypoints.recorder import Recorder

EXTENSION_NAME = "Waypoint Recorder"


class WaypointExtension(omni.ext.IExt):
    """Extension to record Waypoints in Isaac Sim"""

    def on_startup(self, ext_id):
        self._ext_id = ext_id
        self._usd_context = omni.usd.get_context()
        self._window = omni.ui.Window(
            EXTENSION_NAME, width=400, height=500, visible=True, dockPreference=ui.DockPreference.LEFT_BOTTOM
        )

        # init recorder class and get path to extension
        self._extension_path = omni.kit.app.get_app().get_extension_manager().get_extension_path(ext_id)
        self.recorder = Recorder()

        # set additional parameters
        self._input_fields: dict = {}  # dictionary to store values of buttion, float fields, etc.

        # build ui
        self.build_ui()
        return

    ##
    # UI Build functions
    ##

    def build_ui(self):
        with self._window.frame:
            with ui.VStack(spacing=5, height=0):
                self._build_info_ui()

                self._build_recorder_ui()

                self._build_display_ui()

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

        overview = "Extension to record waypoints in any Environment and export them to a .json file."

        setup_ui_headers(self._ext_id, __file__, title, doc_link, overview)
        return

    def _build_recorder_ui(self):
        frame = ui.CollapsableFrame(
            title="Record Waypoints",
            height=0,
            collapsed=False,
            style=get_style(),
            style_type_name_override="CollapsableFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # get save directory
                kwargs = {
                    "label": "Save Directory",
                    "type": "stringfield",
                    "default_val": "",
                    "tooltip": "Click the Folder Icon to Set Filepath",
                    "use_folder_picker": True,
                }
                self._input_fields["save_path"] = str_builder(**kwargs)
                self._input_fields["save_path"].add_value_changed_fn(self._check_save_path)

                kwargs = {
                    "label": "Save Filename",
                    "type": "stringfield",
                    "default_val": "waypoints",
                }
                self._input_fields["file_name"] = str_builder(**kwargs)
                self._input_fields["file_name"].add_value_changed_fn(self.recorder.set_filename)

                self._input_fields["start_point"] = btn_builder(
                    "Start-Point", text="Record", on_clicked_fn=self._set_start_point
                )
                self._input_fields["start_point"].enabled = False

                self._input_fields["way_point"] = btn_builder(
                    "Intermediate-Point", text="Record", on_clicked_fn=self._set_way_point
                )
                self._input_fields["way_point"].enabled = False

                self._input_fields["end_point"] = btn_builder(
                    "End-Point", text="Record", on_clicked_fn=self._set_end_point
                )
                self._input_fields["end_point"].enabled = False

                self._input_fields["reset"] = btn_builder("Reset", text="Reset", on_clicked_fn=self.recorder.reset)
                self._input_fields["reset"].enabled = True
        return

    def _build_display_ui(self):
        frame = ui.CollapsableFrame(
            title="Waypoint Information",
            height=0,
            collapsed=False,
            style=get_style(),
            style_type_name_override="CollapsableFrame",
            horizontal_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_AS_NEEDED,
            vertical_scrollbar_policy=ui.ScrollBarPolicy.SCROLLBAR_ALWAYS_ON,
        )
        with frame:
            with ui.VStack(style=get_style(), spacing=5, height=0):
                # control parameters
                pass
        return

    ##
    # Shutdown Helpers
    ##

    def on_shutdown(self):
        if self._window:
            self._window = None
        gc.collect()

    ##
    # Recorder Helper
    ##

    def _check_save_path(self, path):
        path = path.get_value_as_string()

        if not os.path.isfile(path):
            self._input_fields["start_point"].enabled = True
            self.recorder.set_save_path(path=path)
        else:
            self._input_fields["start_point"].enabled = False
            carb.log_warn(f"Directory at save path {path} does not exist!")

        return

    def _set_start_point(self) -> None:
        # set start point
        self.recorder.set_start_point()

        # enable intermediate waypoints
        self._input_fields["start_point"].enabled = False
        self._input_fields["way_point"].enabled = True
        return

    def _set_way_point(self) -> None:
        # add intermediate waypoint to list
        self.recorder.add_way_point()

        # enable end point
        self._input_fields["end_point"].enabled = True
        return

    def _set_end_point(self) -> None:
        # set end point
        self.recorder.set_end_point()

        # enable / disable buttons
        self._input_fields["way_point"].enabled = False
        self._input_fields["end_point"].enabled = False
        self._input_fields["start_point"].enabled = True
        return


# EoF
