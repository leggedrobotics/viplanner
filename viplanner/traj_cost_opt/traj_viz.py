# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# python
import copy
from typing import Optional

import cv2
import matplotlib.pyplot as plt
import numpy as np
import open3d as o3d
import open3d.visualization.rendering as rendering
import pypose as pp
import scipy.spatial.transform as tf
import torch

# visual-planning-learning
from viplanner.cost_maps import CostMapPCD

from .traj_cost import TrajCost


class TrajViz:
    def __init__(
        self,
        intrinsics: np.ndarray,
        cam_resolution: tuple = (360, 640),
        camera_tilt: float = 0.0,
        cost_map: Optional[CostMapPCD] = None,
    ):
        # get parameters
        self._cam_resolution = cam_resolution
        self._intrinsics = intrinsics
        self._cost_map = cost_map
        self._camera_tilt = camera_tilt

        # init camera
        self.set_camera()

    def set_camera(self):
        self.camera = o3d.camera.PinholeCameraIntrinsic(
            self._cam_resolution[1],  # width
            self._cam_resolution[0],  # height
            self._intrinsics[0, 0],  # fx
            self._intrinsics[1, 1],  # fy
            self._intrinsics[0, 2],  # cx  (width/2)
            self._intrinsics[1, 2],  # cy  (height/2)
        )
        return

    def VizTrajectory(
        self,
        preds: torch.Tensor,
        waypoints: torch.Tensor,
        odom: torch.Tensor,
        goal: torch.Tensor,
        fear: torch.Tensor,
        augment_viz: torch.Tensor,
        cost_map: bool = True,
        visual_height: float = 0.5,
        mesh_size: float = 0.5,
        fov_angle: float = 0.0,
    ) -> None:
        """Visualize the trajectory within the costmap

        Args:
            preds (torch.Tensor): predicted keypoints
            waypoints (torch.Tensor): waypoints
            odom (torch.Tensor): odom tensor
            goal (torch.Tensor): goal tensor
            fear (torch.Tensor): if trajectory is risky
            augment_viz (torch.Tensor): if input has been augmented
            cost_map (bool, optional): visualize costmap. Defaults to True.
            visual_height (float, optional): visual height of the keypoints. Defaults to 0.5.
            mesh_size (float, optional): size of the mesh. Defaults to 0.5.
            fov_angle (float, optional): field of view angle. Defaults to 0.0.
        """
        # transform to map frame
        if not isinstance(self._cost_map, CostMapPCD):
            print("Cost map is missing.")
            return
        batch_size = len(waypoints)
        # transform to world frame
        preds_ws = TrajCost.TransformPoints(odom, preds).tensor().cpu().detach().numpy()
        wp_ws = TrajCost.TransformPoints(odom, waypoints).tensor().cpu().detach().numpy()
        goal_ws = pp.SE3(odom) @ pp.SE3(goal)
        # convert to positions
        goal_ws = goal_ws.tensor()[:, 0:3].numpy()
        visual_list = []
        if cost_map:
            visual_list.append(self._cost_map.pcd_tsdf)
        else:
            visual_list.append(self._cost_map.pcd_viz)
            visual_height = visual_height / 5.0

        # visualize and trajs
        traj_pcd = o3d.geometry.PointCloud()
        wp_ws = np.concatenate(wp_ws, axis=0)
        wp_ws[:, 2] = wp_ws[:, 2] + visual_height
        traj_pcd.points = o3d.utility.Vector3dVector(wp_ws[:, 0:3])
        traj_pcd.paint_uniform_color([0.99, 0.1, 0.1])
        visual_list.append(traj_pcd)
        # start and goal marks
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(mesh_size / 1.5)  # start points
        mesh_sphere_augment = o3d.geometry.TriangleMesh.create_sphere(mesh_size / 1.5)  # start points
        small_sphere = o3d.geometry.TriangleMesh.create_sphere(mesh_size / 3.0)  # successful trajectory points
        small_sphere_fear = o3d.geometry.TriangleMesh.create_sphere(mesh_size / 3.0)  # unsuccessful trajectory points
        mesh_box = o3d.geometry.TriangleMesh.create_box(mesh_size, mesh_size, mesh_size)  # end points
        # set mesh colors
        mesh_box.paint_uniform_color([1.0, 0.64, 0.0])
        small_sphere.paint_uniform_color([0.4, 1.0, 0.1])
        small_sphere_fear.paint_uniform_color([1.0, 0.4, 0.1])
        mesh_sphere_augment.paint_uniform_color([0.0, 0.0, 1.0])
        # field of view visualization
        fov_vis_length = 0.75  # length of the fov visualization plane in meters
        fov_vis_pt_right = pp.SE3(odom) @ pp.SE3(
            [
                fov_vis_length * np.cos(fov_angle / 2),
                fov_vis_length * np.sin(fov_angle / 2),
                0,
                0,
                0,
                0,
                1,
            ]
        )
        fov_vis_pt_left = pp.SE3(odom) @ pp.SE3(
            [
                fov_vis_length * np.cos(fov_angle / 2),
                -fov_vis_length * np.sin(fov_angle / 2),
                0,
                0,
                0,
                0,
                1,
            ]
        )
        fov_vis_pt_right = fov_vis_pt_right.numpy()[:, 0:3]
        fov_vis_pt_right[:, 2] += visual_height
        fov_vis_pt_left = fov_vis_pt_left.numpy()[:, 0:3]
        fov_vis_pt_left[:, 2] += visual_height

        lines = []
        points = []
        for i in range(batch_size):
            lines.append([2 * i, 2 * i + 1])
            gp = goal_ws[i, :]
            op = odom.numpy()[i, :]
            op[2] = op[2] + visual_height
            gp[2] = gp[2] + visual_height
            points.append(gp[:3].tolist())
            points.append(op[:3].tolist())
            # add fov visualization
            fov_mesh = o3d.geometry.TriangleMesh(
                vertices=o3d.utility.Vector3dVector(np.array([op[:3], fov_vis_pt_right[i], fov_vis_pt_left[i]])),
                triangles=o3d.utility.Vector3iVector(np.array([[2, 1, 0]])),
            )
            fov_mesh.paint_uniform_color([1.0, 0.5, 0.0])
            visual_list.append(fov_mesh)
            # add visualization
            if augment_viz[i]:
                visual_list.append(copy.deepcopy(mesh_sphere_augment).translate((op[0], op[1], op[2])))
            else:
                visual_list.append(copy.deepcopy(mesh_sphere).translate((op[0], op[1], op[2])))
            visual_list.append(
                copy.deepcopy(mesh_box).translate(
                    (
                        gp[0] - mesh_size / 2.0,
                        gp[1] - mesh_size / 2.0,
                        gp[2] - mesh_size / 2.0,
                    )
                )
            )
            for j in range(preds_ws[i].shape[0]):
                kp = preds_ws[i][j, :]
                if fear[i, :] > 0.5:
                    visual_list.append(
                        copy.deepcopy(small_sphere_fear).translate((kp[0], kp[1], kp[2] + visual_height))
                    )
                else:
                    visual_list.append(copy.deepcopy(small_sphere).translate((kp[0], kp[1], kp[2] + visual_height)))
        # set line from odom to goal
        colors = [[0.99, 0.99, 0.1] for i in range(len(lines))]
        line_set = o3d.geometry.LineSet(
            o3d.utility.Vector3dVector(points),
            o3d.utility.Vector2iVector(lines),
        )
        line_set.colors = o3d.utility.Vector3dVector(colors)
        visual_list.append(line_set)
        o3d.visualization.draw_geometries(visual_list)
        return

    def VizImages(
        self,
        preds: torch.Tensor,
        waypoints: torch.Tensor,
        odom: torch.Tensor,
        goal: torch.Tensor,
        fear,
        images: torch.Tensor,
        visual_offset=0.35,
        mesh_size=0.3,
        is_shown=True,
        iplanner: bool = False,
        transform: bool = True,
    ):
        batch_size = len(waypoints)
        if transform:
            preds_ws = TrajCost.TransformPoints(odom, preds).tensor().cpu().detach().numpy()
            wp_ws = TrajCost.TransformPoints(odom, waypoints).tensor().cpu().detach().numpy()
            if goal.shape[-1] != 7:
                pp_goal = pp.identity_SE3(batch_size, device=goal.device)
                pp_goal.tensor()[:, 0:3] = goal
                goal = pp_goal.tensor()
            goal_ws = pp.SE3(odom) @ pp.SE3(goal)
            # convert to positions
            goal_ws = goal_ws.tensor()[:, 0:3].cpu().detach().numpy()
        else:
            preds_ws = preds.cpu().detach().numpy()
            wp_ws = waypoints.cpu().detach().numpy()
            goal_ws = goal.cpu().detach().numpy()

        # adjust height
        goal_ws[:, 2] = goal_ws[:, 2] - visual_offset

        # set materia shader
        mtl = o3d.visualization.rendering.MaterialRecord()
        mtl.base_color = [1.0, 1.0, 1.0, 0.3]
        mtl.shader = "defaultUnlit"
        # set meshes
        small_sphere = o3d.geometry.TriangleMesh.create_sphere(mesh_size / 10.0)  # trajectory points
        small_sphere_fear = o3d.geometry.TriangleMesh.create_sphere(mesh_size / 10.0)  # trajectory points
        mesh_sphere = o3d.geometry.TriangleMesh.create_sphere(mesh_size / 2.0)  # successful predict points
        mesh_sphere_fear = o3d.geometry.TriangleMesh.create_sphere(mesh_size / 2.0)  # unsuccessful predict points
        mesh_box = o3d.geometry.TriangleMesh.create_box(mesh_size, mesh_size, mesh_size * 2)  # end points
        # set colors
        if iplanner:
            small_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # blue
            mesh_sphere.paint_uniform_color([1.0, 1.0, 0.0])
        else:
            small_sphere.paint_uniform_color([0.99, 0.2, 0.1])  # green
            mesh_sphere.paint_uniform_color([0.4, 1.0, 0.1])

        small_sphere_fear.paint_uniform_color([1.0, 0.4, 0.2])
        mesh_sphere_fear.paint_uniform_color([1.0, 0.2, 0.1])

        mesh_box.paint_uniform_color([1.0, 0.64, 0.1])

        # init open3D render
        render = rendering.OffscreenRenderer(self.camera.width, self.camera.height)
        render.scene.set_background([0.0, 0.0, 0.0, 1.0])  # RGBA

        # wp_start_idx = int(waypoints.shape[1] / preds.shape[1])
        wp_start_idx = 1
        cv_img_list = []

        if is_shown:
            fig, ax = plt.subplots()

        for i in range(batch_size):
            # add geometries
            gp = goal_ws[i, :]
            # add goal marker
            goal_mesh = copy.deepcopy(mesh_box).translate(
                (
                    gp[0] - mesh_size / 2.0,
                    gp[1] - mesh_size / 2.0,
                    gp[2] - mesh_size / 2.0,
                )
            )
            render.scene.add_geometry("goal_mesh", goal_mesh, mtl)
            # add predictions
            for j, kp in enumerate(preds_ws[i]):
                if fear[i, :] > 0.5:
                    kp_mesh = copy.deepcopy(mesh_sphere_fear).translate((kp[0], kp[1], kp[2] - visual_offset))
                else:
                    kp_mesh = copy.deepcopy(mesh_sphere).translate((kp[0], kp[1], kp[2] - visual_offset))
                render.scene.add_geometry("keypose" + str(j), kp_mesh, mtl)
            # add trajectory
            for k, wp in enumerate(wp_ws[i]):
                if k < wp_start_idx:
                    continue
                if fear[i, :] > 0.5:
                    wp_mesh = copy.deepcopy(small_sphere_fear).translate((wp[0], wp[1], wp[2] - visual_offset))
                else:
                    wp_mesh = copy.deepcopy(small_sphere).translate((wp[0], wp[1], wp[2] - visual_offset))
                render.scene.add_geometry("waypoint" + str(k), wp_mesh, mtl)
            # set cameras
            self.CameraLookAtPose(odom[i, :], render)
            # project to image
            img_o3d = np.asarray(render.render_to_image())
            mask = (img_o3d < 10).all(axis=2)
            # Attach image
            c_img = images[i, :, :].expand(3, -1, -1)
            c_img = c_img.cpu().detach().numpy()
            c_img = np.moveaxis(c_img, 0, 2)
            c_img = (c_img * 255 / np.max(c_img)).astype("uint8")
            img_o3d[mask, :] = c_img[mask, :]
            img_cv2 = cv2.cvtColor(img_o3d, cv2.COLOR_RGBA2BGRA)
            cv_img_list.append(img_cv2)
            if is_shown:
                plt.imshow(img_cv2)
                plt.draw()
                plt.waitforbuttonpress(0)  # this will wait for indefinite time
                plt.close(fig)
            # clear render geometry
            render.scene.clear_geometry()

        return cv_img_list

    def CameraLookAtPose(self, odom, render):
        unit_vec = pp.identity_SE3(device=odom.device)
        unit_vec.tensor()[0] = 1.0
        tilt_vec = [0, 0, 0]
        tilt_vec.extend(list(tf.Rotation.from_euler("y", self._camera_tilt, degrees=False).as_quat()))
        tilt_vec = torch.tensor(tilt_vec, device=odom.device, dtype=odom.dtype)
        target_pose = pp.SE3(odom) @ pp.SE3(tilt_vec) @ unit_vec
        camera_up = [0, 0, 1]  # camera orientation
        eye = pp.SE3(odom)
        eye = eye.tensor()[0:3].cpu().detach().numpy()
        target = target_pose.tensor()[0:3].cpu().detach().numpy()
        render.scene.camera.look_at(target, eye, camera_up)
        return


# EoF
