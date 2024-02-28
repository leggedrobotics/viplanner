# Copyright (c) 2023-2024, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import random
import time
from typing import Tuple

import carb
import numpy as np

# omni
import omni
import scipy.spatial.transform as tf
import torch
from omni.isaac.matterport.config import SamplerCfg

# omni-isaac-matterport
from omni.isaac.matterport.semantics import MatterportWarp

# omni-isaac-orbit
from omni.isaac.orbit.sensors.camera import Camera, PinholeCameraCfg
from scipy.spatial import KDTree

# python
from scipy.stats import qmc


class RandomExplorer:
    debug = False
    time_measurement: bool = False

    def __init__(self, domains: MatterportWarp, cfg: SamplerCfg = None):
        # config
        self._cfg_explorer = SamplerCfg() if cfg is None else cfg

        # domains
        self.domains: MatterportWarp = domains

        # setup cameras and writer
        ply_file = os.path.split(self.domains._cfg.import_file_ply)[1]
        if self._cfg_explorer.suffix is not None and isinstance(self._cfg_explorer.suffix, str):
            suffix = "_" + self._cfg_explorer.suffix
        else:
            suffix = ""
        self.save_path = os.path.join(self._cfg_explorer.save_path, ply_file[:-4] + suffix)

        # get camera centers
        self.nbr_points: int = 0
        self.camera_centers: torch.Tensor = torch.empty((0, 3), dtype=torch.float32)

        # get_variations
        self.x_angles: np.ndarray = np.zeros(self.nbr_points)
        self.y_angles: np.ndarray = np.zeros(self.nbr_points)
        self.z_angles: np.ndarray = np.zeros(self.nbr_points)

        # setup conv
        self.nbr_faces: int = 0
        self.face_idx: torch.Tensor = torch.zeros((0,), dtype=torch.int64)
        self.conv_crit = True

        self.pt_idx = 0
        self.save_idx = 0

        # time measurements
        self.time_raycast: float = 0.0
        self.time_render: float = 0.0
        self.time_save: float = 0.0
        self.time_total: float = 0.0

        return

    ##
    # Public Function
    ##

    def setup(self) -> None:
        self._setup_cameras()
        self.domains.init_save(save_path=self.save_path)
        self._get_sample_points()
        self._get_view_variations()
        self._setup_conv()
        return

    def explore(self) -> None:
        # get cam_data
        cam_data_depth = self.domains.cameras[0]
        cam_data_sem = self.domains.cameras[1]

        # start sim if RGB images should be rendered
        if self._cfg_explorer.cam_sem_rgb:
            self.domains.sim.play()

        while self.conv_crit:
            total_start = time.time()

            # get current variation in camera position and rotation of the semantic camera
            # rotations follow the Isaac convention: x-forward, y-left, z-up
            cam_data_sem.pos = self.camera_centers[self.pt_idx]
            cam_rot_sem_from_odom = np.array(
                [self.x_angles[self.pt_idx], self.y_angles[self.pt_idx], int(self.z_angles[self.pt_idx])]
            )
            cam_rot_sem_from_odom_mat = tf.Rotation.from_euler("xyz", cam_rot_sem_from_odom, degrees=True).as_matrix()
            cam_data_sem.rot = torch.tensor(cam_rot_sem_from_odom_mat, dtype=torch.float32)
            carb.log_verbose(f"Point: {self.pt_idx} \tsem camera pose: {cam_data_sem.pos} {cam_rot_sem_from_odom}")

            # get depth camera rotation relative to the semantic camera rotation and convert it to Isaac convention
            # Isaac Convention: x-forward, y-left, z-up
            if self._cfg_explorer.tf_quat_convention == "isaac":
                cam_rot_sem_from_depth = tf.Rotation.from_quat(self._cfg_explorer.tf_quat).as_matrix()
            elif self._cfg_explorer.tf_quat_convention == "roll-pitch-yaw":
                cam_rot_sem_from_depth = tf.Rotation.from_quat(self._cfg_explorer.tf_quat).as_euler("XYZ", degrees=True)
                cam_rot_sem_from_depth[[1, 2]] *= -1
                cam_rot_sem_from_depth = tf.Rotation.from_euler("XYZ", cam_rot_sem_from_depth, degrees=True).as_matrix()
            else:
                raise ValueError(f"tf_quat_convention {self._cfg_explorer.tf_quat_convention} not supported")

            # get depth camera pose and rotation from the semantic camera pose and rotation
            cam_rot_depth_from_odom = np.matmul(cam_rot_sem_from_odom_mat, cam_rot_sem_from_depth.T)
            cam_data_depth.rot = torch.tensor(cam_rot_depth_from_odom, dtype=torch.float32)
            vec_depth_to_sem_odom_frame = np.matmul(cam_rot_depth_from_odom, self._cfg_explorer.tf_pos)
            cam_data_depth.pos = cam_data_sem.pos - torch.tensor(vec_depth_to_sem_odom_frame, dtype=torch.float32)

            # do raycasting
            start = time.time()
            hit_rate = 1.0
            for cam_data in self.domains.cameras:
                # get new ray directions in world frame
                cam_data.ray_directions = self.domains._get_ray_directions(
                    cam_data.pos, cam_data.rot, cam_data.pixel_coords
                )

                # raycast
                cam_data.ray_hit_coords, cam_data.ray_face_indices, cam_data.ray_distances = self.domains._raycast(
                    cam_data.pos.repeat(len(cam_data.ray_directions)),
                    cam_data.ray_directions,
                    cam_rot=cam_data.rot,
                    pix_offset=cam_data.pixel_offset,
                )

                # filter inf values
                hit_rate_single_cam = torch.isfinite(cam_data.ray_distances).sum() / len(cam_data.ray_distances)
                hit_rate = min(hit_rate, hit_rate_single_cam)
                carb.log_verbose(f"Point: {self.pt_idx} \tRate of rays hitting the mesh: {hit_rate_single_cam}")
                cam_data.ray_hit_coords[~torch.isfinite(cam_data.ray_hit_coords)] = 0
                cam_data.ray_distances[~torch.isfinite(cam_data.ray_distances)] = 0

            self.time_raycast = time.time() - start

            # filter points with insufficient hit rate and too small min distance (use the semantic camera)
            if hit_rate < self._cfg_explorer.min_hit_rate:
                print(f"Point: {self.pt_idx} \trejected due to insufficient hit rate")
                self.pt_idx += 1
                continue
            elif torch.mean(cam_data_sem.ray_distances) < self._cfg_explorer.min_avg_hit_distance:
                print(f"Point: {self.pt_idx} \trejected due to too small average hit distance")
                self.pt_idx += 1
                continue
            elif torch.std(cam_data_sem.ray_distances) < self._cfg_explorer.min_std_hit_distance:
                print(f"Point: {self.pt_idx} \trejected due to too small standard deviation of hit distance")
                self.pt_idx += 1
                continue

            # DEBUG
            if self.debug:
                # set camera to the random selected pose
                self.domains.draw.clear_points()
                self.domains.draw.draw_points(
                    random.choices(cam_data_sem.ray_hit_coords.cpu().tolist(), k=5000),
                    self.domains.colors_2,
                    self.domains.sizes,
                )
                self.domains.draw.draw_points(
                    random.choices(cam_data_sem.pixel_coords.cpu().tolist(), k=5000),
                    self.domains.colors_3,
                    self.domains.sizes,
                )

            # render and save data
            for idx, cam_data in enumerate(self.domains.cameras):
                start = time.time()
                self.domains._render(cam_data)
                self.time_render = time.time() - start

                if cam_data.visualize:
                    start = time.time()
                    self.domains._update_visualization(cam_data)
                    self.time_visualize = time.time() - start

                start = time.time()
                self.domains._save_data(cam_data, self.save_idx, cam_idx=idx)
                self.time_save = time.time() - start

            # DEBUG
            if self.debug:
                import matplotlib.pyplot as plt

                _, axs = plt.subplots(1, 2, figsize=(15, 5))
                axs[0].imshow(cam_data_depth.render_depth)
                axs[1].imshow(cam_data_sem.render_sem)
                plt.show()

            # check convergence according to semantic camera
            ray_face_filtered = cam_data_sem.ray_face_indices[cam_data_sem.ray_face_indices != -1]
            self.face_idx[ray_face_filtered.long().cpu()] += 1

            conv_face = torch.sum(self.face_idx > 2)
            conv_rate = conv_face / self.nbr_faces
            if conv_rate > self._cfg_explorer.conv_rate or self.save_idx > self._cfg_explorer.max_images:
                self.conv_crit = False

            self.time_total = time.time() - total_start

            # Update messages
            face1_count = torch.sum(self.face_idx >= 1).item()
            print(
                f"Point: {self.pt_idx} \t Save Idx: {self.save_idx} \t Faces 1: {face1_count} <=> {(round(float(face1_count / self.nbr_faces * 100), 6))} (%)"
                f"\t Faces 3: {conv_face} <=> {(round(float(conv_rate*100), 6))} (%) \t in {self.time_total}s"
            )

            if self.time_measurement:
                print(
                    f"Raycast: {self.time_raycast} \t Render: {self.time_render} \t Visualize: {self.time_visualize}"
                    f"\t Save: {self.time_save} \n Overall: {self.time_total}"
                )

            # update index
            self.pt_idx += 1
            self.save_idx += 1

            if self.pt_idx >= self.nbr_points - 1:
                self.conv_crit = False
                print(
                    f"All points have been sampled, currently {self.save_idx} points saved. If more points are "
                    f"needed, increase the number of points per m2"
                )

        self.domains._end_save()
        return

    ##
    # Helper Sample Points
    ##

    def _setup_cameras(self) -> None:
        """Setup the cameras for the exploration."""
        stage = omni.usd.get_context().get_stage()

        # depth camera
        if self._cfg_explorer.cam_depth_intrinsics is not None:
            intrinscis = np.array(self._cfg_explorer.cam_depth_intrinsics).reshape(3, 3)
            horizontalAperture = (
                self._cfg_explorer.cam_depth_resolution[0]
                * self._cfg_explorer.cam_depth_focal_length
                / intrinscis[0, 0]
            )
        else:
            horizontalAperture = self._cfg_explorer.cam_depth_aperture

        depth_cam_prim = stage.DefinePrim(self._cfg_explorer.cam_depth_prim, "Camera")
        depth_cam_prim.GetAttribute("focalLength").Set(self._cfg_explorer.cam_depth_focal_length)  # set focal length
        depth_cam_prim.GetAttribute("clippingRange").Set(
            self._cfg_explorer.cam_depth_clipping_range
        )  # set clipping range
        depth_cam_prim.GetAttribute("horizontalAperture").Set(horizontalAperture)  # set aperture

        self.domains.register_camera(
            depth_cam_prim,
            self._cfg_explorer.cam_depth_resolution[0],
            self._cfg_explorer.cam_depth_resolution[1],
            depth=True,
            visualization=self.debug,
        )

        # semantic and rgb camera
        if self._cfg_explorer.cam_sem_intrinsics is not None:
            intrinscis = np.array(self._cfg_explorer.cam_sem_intrinsics).reshape(3, 3)
            horizontalAperture = (
                self._cfg_explorer.cam_sem_resolution[0] * self._cfg_explorer.cam_sem_focal_length / intrinscis[0, 0]
            )
        else:
            horizontalAperture = self._cfg_explorer.cam_sem_aperture

        sem_cam_prim = stage.DefinePrim(self._cfg_explorer.cam_sem_prim, "Camera")
        sem_cam_prim.GetAttribute("focalLength").Set(self._cfg_explorer.cam_sem_focal_length)  # set focal length
        sem_cam_prim.GetAttribute("horizontalAperture").Set(horizontalAperture)  # set aperture
        sem_cam_prim.GetAttribute("clippingRange").Set(self._cfg_explorer.cam_sem_clipping_range)  # set clipping range

        if self._cfg_explorer.cam_sem_rgb:
            orbit_cam_cfg = PinholeCameraCfg(
                width=self._cfg_explorer.cam_sem_resolution[0],
                height=self._cfg_explorer.cam_sem_resolution[1],
            )
            orbit_cam_cfg.usd_params.clipping_range = self._cfg_explorer.cam_sem_clipping_range
            orbit_cam_cfg.usd_params.focal_length = self._cfg_explorer.cam_sem_focal_length
            orbit_cam_cfg.usd_params.horizontal_aperture = horizontalAperture
            orbit_cam = Camera(orbit_cam_cfg)
            orbit_cam.spawn(self._cfg_explorer.cam_sem_prim + "_rgb")
            orbit_cam.initialize()
        else:
            orbit_cam = None

        self.domains.register_camera(
            sem_cam_prim,
            self._cfg_explorer.cam_sem_resolution[0],
            self._cfg_explorer.cam_sem_resolution[1],
            semantics=True,
            rgb=self._cfg_explorer.cam_sem_rgb,
            visualization=self.debug,
            omni_cam=orbit_cam,
        )
        return

    def _get_sample_points(self) -> None:
        # get min, max of the mesh in the xy plane
        x_min = self.domains.mesh.bounds[0][0]
        x_max = self.domains.mesh.bounds[1][0]
        y_min = self.domains.mesh.bounds[0][1]
        y_max = self.domains.mesh.bounds[1][1]
        max_area = (x_max - x_min) * (y_max - y_min)

        # init sampler as qmc
        sampler = qmc.Halton(d=2, scramble=False)
        # determine number of samples to dram
        nbr_points = int(max_area * self._cfg_explorer.points_per_m2)
        # get raw samples origins
        points = sampler.random(nbr_points)
        points = qmc.scale(points, [x_min, y_min], [x_max, y_max])
        heights = np.ones((nbr_points, 1)) * self._cfg_explorer.height
        ray_origins = torch.from_numpy(np.hstack((points, heights)))
        ray_origins = ray_origins.type(torch.float32)

        # get ray directions in negative z direction
        ray_directions = torch.zeros((nbr_points, 3), dtype=torch.float32)
        ray_directions[:, 2] = -1.0

        # raycast
        ray_hits_world_down, _, _ = self.domains._raycast(
            ray_origins * torch.tensor([1, 1, 2]),  # include objects above the robot
            ray_directions,
            cam_rot=torch.eye(3),
            pix_offset=torch.zeros_like(ray_origins),
        )

        z_depth = torch.abs(ray_hits_world_down[:, 2] - ray_origins[:, 2] * 2)
        # filter points outside the mesh and within walls
        filter_inside_mesh = torch.isfinite(z_depth)  # outside mesh
        filter_inside_mesh[
            ray_hits_world_down[:, 2] < self._cfg_explorer.ground_height
        ] = False  # above holes in the ground
        print(f"filtered {nbr_points - filter_inside_mesh.sum()} points outside of mesh")
        filter_outside_wall = z_depth > (self._cfg_explorer.min_height + ray_origins[:, 2])
        print(f"filtered {nbr_points - filter_outside_wall.sum()} points inside wall")
        filter_combined = torch.all(torch.stack((filter_inside_mesh, filter_outside_wall), dim=1), dim=1)
        print(f"filtered total of {round(float((1 - filter_combined.sum() / nbr_points) * 100), 4)} % of points")

        if self.debug:
            import copy

            import open3d as o3d

            o3d_mesh = self.domains.mesh.as_open3d
            o3d_mesh.compute_vertex_normals()
            odom_vis_list = [o3d_mesh]

            small_sphere = o3d.geometry.TriangleMesh.create_sphere(0.05)  # successful trajectory points

            camera_centers = ray_origins.cpu().numpy()

            for idx, camera_center in enumerate(camera_centers):
                if filter_combined[idx]:
                    small_sphere.paint_uniform_color([0.4, 1.0, 0.1])  # green
                else:
                    small_sphere.paint_uniform_color([1.0, 0.1, 0.1])  # red

                odom_vis_list.append(
                    copy.deepcopy(small_sphere).translate((camera_center[0], camera_center[1], camera_center[2]))
                )

            o3d.visualization.draw_geometries(odom_vis_list)

        self.camera_centers = ray_origins[filter_combined].type(torch.float32)

        # free gpu memory
        ray_origins = filter_combined = filter_inside_mesh = filter_outside_wall = z_depth = ray_hits_world_down = None

        # enforce a minimum distance to the walls
        angles = np.linspace(-np.pi, np.pi, 20)
        ray_directions = tf.Rotation.from_euler("z", angles, degrees=False).as_matrix() @ np.array([1, 0, 0])
        ray_hit = []

        for ray_direction in ray_directions:
            ray_direction_torch = (
                torch.from_numpy(ray_direction).repeat(self.camera_centers.shape[0], 1).type(torch.float32)
            )
            ray_hits_world, _, _ = self.domains._raycast(
                self.camera_centers,
                ray_direction_torch,
                cam_rot=torch.eye(3),
                pix_offset=torch.zeros_like(ray_direction_torch),
            )
            ray_hit.append(
                torch.norm(ray_hits_world - self.camera_centers, dim=1) > self._cfg_explorer.min_wall_distance
            )

        # check if every point has the minimum distance in every direction
        without_wall = torch.all(torch.vstack(ray_hit), dim=0)

        if self.debug:
            import copy

            import open3d as o3d

            o3d_mesh = self.domains.mesh.as_open3d
            o3d_mesh.compute_vertex_normals()
            odom_vis_list = [o3d_mesh]

            small_sphere = o3d.geometry.TriangleMesh.create_sphere(0.05)  # successful trajectory points

            camera_centers = self.camera_centers.cpu().numpy()

            for idx, camera_center in enumerate(camera_centers):
                if without_wall[idx]:
                    small_sphere.paint_uniform_color([0.4, 1.0, 0.1])  # green
                else:
                    small_sphere.paint_uniform_color([1.0, 0.1, 0.1])  # red

                odom_vis_list.append(
                    copy.deepcopy(small_sphere).translate((camera_center[0], camera_center[1], camera_center[2]))
                )

            o3d.visualization.draw_geometries(odom_vis_list)

        print(f"filtered {self.camera_centers.shape[0] - without_wall.sum()} points too close to walls")
        self.camera_centers = self.camera_centers[without_wall].type(torch.float32)
        self.nbr_points = self.camera_centers.shape[0]
        return

    def _construct_kdtree(self, num_neighbors: int = 50) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # construct kdtree to find nearest neighbors of points
        kdtree = KDTree(self.camera_centers.cpu().numpy())
        _, nearest_neighbors_idx = kdtree.query(self.camera_centers.cpu().numpy(), k=num_neighbors + 1, workers=-1)

        # remove first neighbor as it is the point itself
        nearest_neighbors_idx = torch.tensor(nearest_neighbors_idx[:, 1:], dtype=torch.int64)

        # define origin and neighbor points
        origin_point = torch.repeat_interleave(self.camera_centers, repeats=num_neighbors, axis=0)
        neighbor_points = self.camera_centers[nearest_neighbors_idx, :].reshape(-1, 3)
        distance = torch.norm(origin_point - neighbor_points, dim=1)

        # check for collision with raycasting
        _, _, hit_depth = self.domains._raycast(
            origin_point,
            origin_point - neighbor_points,
            cam_rot=torch.eye(3),
            pix_offset=torch.zeros_like(origin_point),
        )

        hit_depth[torch.isnan(hit_depth)] = self.domains._cfg.max_depth
        # filter connections that collide with the environment
        collision = (hit_depth < distance).reshape(-1, num_neighbors)
        return nearest_neighbors_idx, collision, distance

    def _get_view_variations(self):
        # the variation around the up axis (z-axis) has to be picked in order to avoid that the camera faces a wall
        # done by construction a graph of all sample points and pick the z angle in order to point at one of the neighbors of the node

        # get nearest neighbors and check for collision
        nearest_neighbors_idx, collision, _ = self._construct_kdtree()

        # remove nodes
        all_collision_idx = torch.all(collision, dim=1)

        # select neighbor with the largest distance that is not in collision
        direction_neighbor_idx = torch.hstack(
            [
                (collision_single_node == False).nonzero().reshape(-1)[-1]
                for collision_single_node in collision[~all_collision_idx, :]
            ]
        )
        direction_neighbor_idx = torch.vstack(
            (torch.arange(nearest_neighbors_idx.shape[0])[~all_collision_idx], direction_neighbor_idx)
        ).T
        selected_neighbor_idx = nearest_neighbors_idx[direction_neighbor_idx[:, 0], direction_neighbor_idx[:, 1]]

        if self.debug:
            import copy

            import open3d as o3d

            o3d_mesh = self.domains.mesh.as_open3d
            o3d_mesh.compute_vertex_normals()
            odom_vis_list = [o3d_mesh]

            small_sphere = o3d.geometry.TriangleMesh.create_sphere(0.05)  # successful trajectory points

            camera_centers = self.camera_centers[nearest_neighbors_idx[0]].cpu().numpy()

            for idx, camera_center in enumerate(camera_centers):
                if collision[0][idx]:  # in collision or nan
                    small_sphere.paint_uniform_color([1.0, 0.4, 0.0])  # orange
                elif idx == direction_neighbor_idx[0][1]:  # selected neighbor
                    small_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # blue
                else:
                    small_sphere.paint_uniform_color([0.1, 1.0, 0.1])  # green
                odom_vis_list.append(
                    copy.deepcopy(small_sphere).translate((camera_center[0], camera_center[1], camera_center[2]))
                )

            small_sphere.paint_uniform_color([1.0, 0.1, 0.1])  # red
            odom_vis_list.append(
                copy.deepcopy(small_sphere).translate(
                    (
                        self.camera_centers[0][0].cpu().numpy(),
                        self.camera_centers[0][1].cpu().numpy(),
                        self.camera_centers[0][2].cpu().numpy(),
                    )
                )
            )

            # check if selected neighbor idx is correct by plotting the neighbor again
            small_sphere.paint_uniform_color([0.0, 0.0, 1.0])  # blue
            neighbor = self.camera_centers[selected_neighbor_idx[0]].cpu().numpy()
            odom_vis_list.append(copy.deepcopy(small_sphere).translate((neighbor[0], neighbor[1], neighbor[2])))

            # draw line
            line_set = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(self.camera_centers.cpu().numpy()),
                o3d.utility.Vector2iVector(np.array([[0, selected_neighbor_idx[0].cpu().numpy()]])),
            )
            line_set.colors = o3d.utility.Vector3dVector([[0.99, 0.99, 0.1]])
            odom_vis_list.append(line_set)

            o3d.visualization.draw_geometries(odom_vis_list)

        # get the z angle of the neighbor that is closest to the origin point
        neighbor_direction = self.camera_centers[~all_collision_idx, :] - self.camera_centers[selected_neighbor_idx, :]
        self.z_angles = np.rad2deg(torch.atan2(neighbor_direction[:, 1], neighbor_direction[:, 0]).cpu().numpy())

        if self.debug:
            import copy

            import open3d as o3d

            o3d_mesh = self.domains.mesh.as_open3d
            o3d_mesh.compute_vertex_normals()
            odom_vis_list = [o3d_mesh]

            small_sphere = o3d.geometry.TriangleMesh.create_sphere(0.05)  # successful trajectory points
            small_sphere.paint_uniform_color([0.4, 1.0, 0.1])  # green

            camera_centers = self.camera_centers.cpu().numpy()

            for camera_center in camera_centers:
                odom_vis_list.append(
                    copy.deepcopy(small_sphere).translate((camera_center[0], camera_center[1], camera_center[2]))
                )

            colors = [[0.99, 0.99, 0.1] for i in range(len(camera_centers))]
            neighbor_map = []
            selected_neighbor_idx_counter = 0
            for curr_center in range(self.camera_centers.shape[0]):
                if not all_collision_idx[curr_center]:
                    neighbor_map.append(
                        [curr_center, selected_neighbor_idx[selected_neighbor_idx_counter].cpu().numpy()]
                    )
                    selected_neighbor_idx_counter += 1
            line_set = o3d.geometry.LineSet(
                o3d.utility.Vector3dVector(camera_centers), o3d.utility.Vector2iVector(np.array(neighbor_map))
            )
            line_set.colors = o3d.utility.Vector3dVector(colors)
            odom_vis_list.append(line_set)
            o3d.visualization.draw_geometries(odom_vis_list)

        # filter points that have no neighbors that are not in collision and update number of points
        self.camera_centers = self.camera_centers[~all_collision_idx, :]
        self.nbr_points = self.camera_centers.shape[0]

        # vary the rotation of the forward and horizontal axis (in camera frame) as a uniform distribution within the limits
        self.x_angles = np.random.uniform(
            self._cfg_explorer.x_angle_range[0], self._cfg_explorer.y_angle_range[1], self.nbr_points
        )
        self.y_angles = np.random.uniform(
            self._cfg_explorer.y_angle_range[0], self._cfg_explorer.y_angle_range[1], self.nbr_points
        )
        return

    def _setup_conv(self):
        # index array
        self.nbr_faces = len(self.domains.mesh.metadata["_ply_raw"]["face"]["data"])
        self.face_idx = torch.zeros(self.nbr_faces, dtype=torch.int64)
        return

    # # save data helpers ###

    # def _init_save(self, save_path: Optional[str] = None) -> None:
    #     if save_path is not None:
    #         self._cfg.save_path = save_path

    #     # create directories
    #     os.makedirs(self._cfg.save_path, exist_ok=True)
    #     os.makedirs(os.path.join(self._cfg.save_path, "semantics"), exist_ok=True)
    #     os.makedirs(os.path.join(self._cfg.save_path, "depth"), exist_ok=True)

    #     # save camera configurations
    #     intrinsics = np.zeros((len(self.cameras), 9))
    #     for idx, curr_cam in enumerate(self.cameras):
    #         intrinsics[idx] = curr_cam.data.intrinsic_matrices[0].cpu().numpy().flatten()
    #     np.savetxt(os.path.join(self._cfg.save_path, "intrinsics.txt"), intrinsics, delimiter=",")

    # def _save_data(self) -> None:
    #     # TODO: use replicator writer, currently too slow
    #     for camera in self.cameras:
    #         suffix = f"_{camera.cfg.prim_path}"
    #     cam_suffix = f"_cam{cam_idx}" if len(self.cameras) > 1 else ""

    #     # SEMANTICS
    #     if cam_data.semantic:
    #         cv2.imwrite(
    #             os.path.join(self._cfg.save_path, "semantics", f"{idx}".zfill(4) + cam_suffix + ".png"),
    #             cv2.cvtColor(cam_data.render_sem.astype(np.uint8), cv2.COLOR_RGB2BGR),
    #         )

    #     # DEPTH
    #     if cam_data.depth:
    #         cv2.imwrite(
    #             os.path.join(self._cfg.save_path, "depth", f"{idx}".zfill(4) + cam_suffix + ".png"),
    #             cam_data.render_depth,
    #         )

    #     # camera pose in robotics frame (x forward, y left, z up)
    #     rot_quat = tf.Rotation.from_matrix(cam_data.rot.cpu().numpy()).as_quat()  # get quat as (x, y, z, w) format
    #     pose = np.hstack((cam_data.pos.cpu().numpy(), rot_quat))
    #     cam_data.poses = np.append(cam_data.poses, pose.reshape(1, -1), axis=0)
    #     return

    # def _end_save(self) -> None:
    #     # save camera poses
    #     for idx, cam_data in enumerate(self.cameras):
    #         np.savetxt(
    #             os.path.join(self._cfg.save_path, f"camera_extrinsic_cam{idx}.txt"),
    #             cam_data.poses[1:],
    #             delimiter=",",
    #         )
    #     return
