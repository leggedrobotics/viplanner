# Copyright (c) 2023-2025, ETH Zurich (Robotics Systems Lab)
# Author: Pascal Roth
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import builtins

import carb
import networkx as nx
import numpy as np
import omni.isaac.core.utils.prims as prims_utils
import scipy.spatial.transform as tf
import torch
from omni.isaac.core.utils.semantics import get_semantics
from omni.isaac.lab.scene import InteractiveScene
from omni.isaac.lab.sensors import RayCaster, RayCasterCamera
from omni.isaac.lab.sim import SimulationContext
from omni.isaac.lab.utils.warp import raycast_mesh
from omni.isaac.matterport.domains import MatterportRayCaster, MatterportRayCasterCamera
from omni.physx import get_physx_scene_query_interface
from pxr import Gf, Usd, UsdGeom
from scipy.spatial import KDTree
from scipy.stats import qmc
from skimage.draw import line

from .terrain_analysis_cfg import TerrainAnalysisCfg
from .utils import get_all_meshes


class TerrainAnalysis:
    def __init__(self, cfg: TerrainAnalysisCfg, scene: InteractiveScene):
        # save cfg and env
        self.cfg = cfg
        self.scene = scene

    def __del__(self):
        if hasattr(self, "_raycaster"):
            del self._raycaster
            del self._raycaster_mesh_param

    @property
    def device(self) -> str:
        return self.scene.device

    @property
    def complete(self) -> bool:
        return hasattr(self, "graph") and hasattr(self, "samples")

    @property
    def height_grid(self) -> torch.Tensor:
        if not hasattr(self, "_height_grid"):
            self._setup_raycaster()
            self.construct_height_map()
        return self._height_grid

    @property
    def mesh_dimensions(self) -> tuple[float, float, float, float]:
        if not hasattr(self, "_mesh_dimensions"):
            self._setup_raycaster()
        return self._mesh_dimensions

    ###
    # Operations
    ###

    def analyse(self):
        print("[INFO] Starting terrain analysis...")
        # get raycaster and mesh dimensions
        self._setup_raycaster()
        # build height grid of the environment
        self.construct_height_map()
        # get the points and sample the graph
        self._sample_points()
        self._construct_graph()

    def get_height(self, positions: torch.Tensor) -> torch.Tensor:
        """Given position coordinates will return their respective height in the height map

        Args:
            positions: Coordinates of positions (Shape: [N, 2])

        Returns:
            The height of the positions (Shape: [N])
        """
        # get the indexes of the positions
        pos_idx = (
            (positions.cpu() - torch.tensor([self.mesh_dimensions[2], self.mesh_dimensions[3]]))
            / self.cfg.grid_resolution
        ).int()
        # clamp the indexes to the grid
        pos_idx[:, 0] = torch.clamp(pos_idx[:, 0], 0, self.height_grid.shape[0] - 1)
        pos_idx[:, 1] = torch.clamp(pos_idx[:, 1], 0, self.height_grid.shape[1] - 1)
        # get the height of the positions
        return self.height_grid[pos_idx[:, 0], pos_idx[:, 1]]

    ###
    # Helper functions
    ###

    def _sample_points(self):
        # init sampler as qmc
        sampler = qmc.Halton(d=2, scramble=False)
        sampled_nb_points = 0
        sampled_points = []

        print(f"[INFO] Sampling {self.cfg.sample_points} points...")
        while sampled_nb_points < self.cfg.sample_points:
            # get raw samples origins
            points = sampler.random(self.cfg.sample_points)
            points = qmc.scale(
                points,
                [self._mesh_dimensions[2], self._mesh_dimensions[3]],
                [self._mesh_dimensions[0], self._mesh_dimensions[1]],
            )
            heights = np.ones((self.cfg.sample_points, 1)) * self.cfg.wall_height

            ray_origins = torch.from_numpy(np.hstack((points, heights))).type(torch.float32).to(self.device)

            # filter points that are outside the mesh or inside walls
            ray_origins, heights = self._point_filter_wall(ray_origins)

            # filter points that are too close to walls
            ray_origins, heights = self._point_filter_wall_closeness(ray_origins, heights)

            # filter points based on semantic cost
            if self.cfg.semantic_cost_mapping is not None:
                ray_origins, heights = self._point_filter_semantic_cost(ray_origins, heights)

            # set z height of samples to be at the robot's height above the terrain.
            ray_origins[:, 2] = heights + self.cfg.robot_height

            sampled_points.append(torch.clone(ray_origins))
            sampled_nb_points += ray_origins.shape[0]

        self.points = torch.vstack(sampled_points)
        self.points = self.points[: self.cfg.sample_points]
        return

    def _construct_graph(self):
        # construct kdtree to find nearest neighbors of points
        kdtree = KDTree(self.points.cpu().numpy())
        _, nearest_neighbors_idx = kdtree.query(self.points.cpu().numpy(), k=self.cfg.num_connections + 1, workers=-1)
        # remove first neighbor as it is the point itself
        nearest_neighbors_idx = torch.tensor(nearest_neighbors_idx[:, 1:], dtype=torch.int64, device=self.device)

        # filter connections that collide with the environment
        idx_edge_start, idx_edge_end, distance = self._edge_filter_mesh_collisions(nearest_neighbors_idx)

        (
            idx_edge_start,
            idx_edge_end,
            distance,
            idx_edge_start_filtered,
            idx_edge_end_filtered,
        ) = self._edge_filter_height_diff(idx_edge_start, idx_edge_end, distance)

        # filter edges based on semantic cost
        if self.cfg.semantic_cost_mapping is not None:
            (
                idx_edge_start,
                idx_edge_end,
                distance,
                idx_edge_start_filtered_sem,
                idx_edge_end_filtered_sem,
            ) = self._edge_filter_semantic_cost(idx_edge_start, idx_edge_end, distance)

        # init graph
        print(f"[INFO] Constructing graph with {idx_edge_start.shape[0]} edges")
        self.graph = nx.Graph()
        # add nodes with position attributes
        self.graph.add_nodes_from(list(range(self.cfg.sample_points)))
        pos_attr = {i: {"pos": self.points[i].cpu().numpy()} for i in range(self.cfg.sample_points)}
        nx.set_node_attributes(self.graph, pos_attr)
        # add edges with distance attributes
        # NOTE: as the shortest path searching algorithm only stores integers
        self.graph.add_edges_from(list(map(tuple, np.stack((idx_edge_start, idx_edge_end), axis=1))))
        distance_attr = {
            (i, j): {"distance": distance[idx]} for idx, (i, j) in enumerate(zip(idx_edge_start, idx_edge_end))
        }
        nx.set_edge_attributes(self.graph, distance_attr)

        # remove nodes with no edges
        # NOTE: while the nodes are removed, the node ids and the self.point id are still the same, thus we don't prune
        #   the self.points tensor
        self.isolated_points_ids = list(nx.isolates(self.graph))
        self.graph.remove_nodes_from(self.isolated_points_ids)
        print(f"[INFO] Removed {len(self.isolated_points_ids)} isolated nodes")

        # get all shortest paths
        odom_goal_distances = dict(
            nx.all_pairs_dijkstra_path_length(self.graph, cutoff=self.cfg.max_path_length, weight="distance")
        )

        # summarize to samples
        # samples are in the format (node, connected neighbor, distance)
        samples = []
        for key, value in odom_goal_distances.items():
            curr_samples = torch.zeros((len(value), 3))
            curr_samples[:, 0] = key
            curr_samples[:, 1] = torch.tensor(list(value.keys()))
            curr_samples[:, 2] = torch.tensor(list(value.values()))
            samples.append(curr_samples)
        self.samples = torch.vstack(samples).to(self.device)

        # debug visualization
        if self.cfg.viz_graph:
            env_render_steps = 1000
            if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
                print(f"[INFO] Visualizing graph. Will do {env_render_steps} render steps...")
            else:
                print("[INFO] Visualizing graph.")

            # in headless mode, we cannot visualize the graph and omni.debug.draw is not available
            try:
                import omni.isaac.debug_draw._debug_draw as omni_debug_draw

                draw_interface = omni_debug_draw.acquire_debug_draw_interface()
                draw_interface.draw_points(
                    self.points.tolist(),
                    [(1.0, 0.5, 0, 1)] * self.cfg.sample_points,
                    [5] * self.cfg.sample_points,
                )
                for start_idx, goal_idx in zip(idx_edge_start, idx_edge_end):
                    draw_interface.draw_lines(
                        [self.points[start_idx].tolist()],
                        [self.points[goal_idx].tolist()],
                        [(0, 1, 0, 1)],
                        [1],
                    )
                for start_idx, goal_idx in zip(idx_edge_start_filtered, idx_edge_end_filtered):
                    draw_interface.draw_lines(
                        [self.points[start_idx].tolist()],
                        [self.points[goal_idx].tolist()],
                        [(1, 0, 0, 1)],
                        [1],
                    )
                if self.cfg.semantic_cost_mapping is not None:
                    for start_idx, goal_idx in zip(idx_edge_start_filtered_sem, idx_edge_end_filtered_sem):
                        draw_interface.draw_lines(
                            [self.points[start_idx].tolist()],
                            [self.points[goal_idx].tolist()],
                            [(1, 0, 0, 1)],
                            [1],
                        )

                if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
                    sim = SimulationContext.instance()
                    for _ in range(env_render_steps):
                        sim.render()

                    # clear the drawn points and lines
                    draw_interface.clear_points()
                    draw_interface.clear_lines()

                    print("[INFO] Finished visualizing graph.")

            except ImportError:
                print("[WARNING] Graph Visualization is not available in headless mode.")

    ###
    # Mesh dimensions
    ###

    def _get_mesh_dimensions(self) -> tuple[float, float, float, float]:
        # get min, max of the mesh in the xy plane
        # Get bounds of the terrain
        bounds = []
        for mesh in self._raycaster.meshes.values():
            curr_bounds = torch.zeros((2, 3))
            # for new RSL implementation of raycaster
            # FIXME: @pascal-roth: this is a temporary fix until the new raycaster is merged into the public main branch
            if isinstance(mesh, list):
                curr_bounds[0] = torch.tensor(mesh[0][0].points).max(dim=0)[0]
                curr_bounds[1] = torch.tensor(mesh[0][0].points).min(dim=0)[0]
            else:
                curr_bounds[0] = torch.tensor(mesh.points).max(dim=0)[0]
                curr_bounds[1] = torch.tensor(mesh.points).min(dim=0)[0]
            bounds.append(curr_bounds)
        bounds = torch.vstack(bounds)
        x_min, y_min = bounds[:, 0].min().item(), bounds[:, 1].min().item()
        x_max, y_max = bounds[:, 0].max().item(), bounds[:, 1].max().item()
        return x_max, y_max, x_min, y_min

    def _get_usd_stage_dimensions(self) -> tuple[float, float, float, float]:
        # get all mesh prims
        mesh_prims, mesh_prims_name = get_all_meshes(self.scene.terrain.cfg.prim_path)

        # if space limiter is given, only consider the meshes with the space limiter in the name
        if self.cfg.dim_limiter_prim:
            mesh_idx = [
                idx
                for idx, prim_name in enumerate(mesh_prims_name)
                if self.cfg.dim_limiter_prim.lower() in prim_name.lower()
            ]
        else:
            # remove ground plane since has infinite extent
            mesh_idx = [idx for idx, prim_name in enumerate(mesh_prims_name) if "groundplane" not in prim_name.lower()]

        mesh_prims = [mesh_prims[idx] for idx in mesh_idx]

        bbox_cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default", "render"])
        bbox = [self.compute_bbox_with_cache(bbox_cache, curr_prim) for curr_prim in mesh_prims]
        prim_max = np.vstack([list(prim_range.GetMax()) for prim_range in bbox])
        prim_min = np.vstack([list(prim_range.GetMin()) for prim_range in bbox])
        x_min, y_min, z_min = np.min(prim_min, axis=0)
        x_max, y_max, z_max = np.max(prim_max, axis=0)

        return x_max, y_max, x_min, y_min

    @staticmethod
    def compute_bbox_with_cache(cache: UsdGeom.BBoxCache, prim: Usd.Prim) -> Gf.Range3d:
        """
        Compute Bounding Box using ComputeWorldBound at UsdGeom.BBoxCache. More efficient if used multiple times.
        See https://graphics.pixar.com/usd/dev/api/class_usd_geom_b_box_cache.html

        Args:
            cache: A cached, i.e. `UsdGeom.BBoxCache(Usd.TimeCode.Default(), ['default', 'render'])`
            prim: A prim to compute the bounding box.
        Returns:
            A range (i.e. bounding box), see more at: https://graphics.pixar.com/usd/release/api/class_gf_range3d.html

        """
        bound = cache.ComputeWorldBound(prim)
        bound_range = bound.ComputeAlignedBox()
        return bound_range

    ###
    # Point filter functions
    ###

    def _point_filter_wall(self, ray_origins: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # get ray directions in negative z direction
        ray_directions = torch.zeros((self.cfg.sample_points, 3), dtype=torch.float32, device=self.device)
        ray_directions[:, 2] = -1.0
        # elevate the ray origins to be above the height of the walls
        ray_origins[:, 2] += 1.0

        if self._raycaster is not None:
            hit_point = raycast_mesh(
                ray_starts=ray_origins.unsqueeze(0),
                ray_directions=ray_directions.unsqueeze(0),
                **self._raycaster_mesh_param,
            )[0].squeeze(0)
        else:
            hit_point = self._raycast_usd_stage(
                ray_starts=ray_origins,
                ray_directions=ray_directions,
            )[0]

        # filter points outside the mesh and within walls
        filter_inside_mesh = torch.isfinite(hit_point[..., 2])  # outside mesh
        filter_outside_wall = hit_point[..., 2] < self.cfg.wall_height  # inside wall
        filter_combined = torch.all(torch.stack((filter_inside_mesh, filter_outside_wall), dim=1), dim=1)
        print(
            f"[DEBUG] filtered {round(float((1 - filter_combined.sum() / self.cfg.sample_points) * 100), 4)} % of"
            f" points ({self.cfg.sample_points - filter_inside_mesh.sum()} outside of the mesh and"
            f" {self.cfg.sample_points - filter_outside_wall.sum()} points inside wall)"
        )

        return ray_origins[filter_combined].type(torch.float32), hit_point[filter_combined, 2]

    def _point_filter_wall_closeness(
        self, ray_origins: torch.Tensor, heights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # reduce ground height to check for closeness to walls and other objects
        ray_origins[:, 2] = heights + self.cfg.robot_height
        # enforce a minimum distance to the walls
        angles = np.linspace(-np.pi, np.pi, 20)
        ray_directions = tf.Rotation.from_euler("z", angles, degrees=False).as_matrix() @ np.array([1, 0, 0])
        ray_hit = []

        for ray_direction in ray_directions:
            ray_direction_torch = (
                torch.from_numpy(ray_direction).repeat(ray_origins.shape[0], 1).type(torch.float32).to(self.device)
            )
            if self._raycaster is not None:
                distance = raycast_mesh(
                    ray_starts=ray_origins.unsqueeze(0),
                    ray_directions=ray_direction_torch.unsqueeze(0),
                    max_dist=self.cfg.robot_buffer_spawn,
                    return_distance=True,
                    **self._raycaster_mesh_param,
                )[1].squeeze(0)
            else:
                distance = self._raycast_usd_stage(
                    ray_starts=ray_origins,
                    ray_directions=ray_direction_torch,
                    max_dist=self.cfg.robot_buffer_spawn,
                    return_distance=True,
                )[1]
            ray_hit.append(torch.isinf(distance))

        # check if every point has the minimum distance in every direction
        without_wall = torch.all(torch.vstack(ray_hit), dim=0)

        print(f"[DEBUG] filtered {ray_origins.shape[0] - without_wall.sum().item()} points too close to walls")
        ray_origins = ray_origins[without_wall].type(torch.float32)
        heights = heights[without_wall]
        return ray_origins, heights

    def _point_filter_semantic_cost(
        self, ray_origins: torch.Tensor, heights: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # raycast vertically down and get the corresponding face id
        ray_directions = torch.zeros((ray_origins.shape[0], 3), dtype=torch.float32, device=self.device)
        ray_directions[:, 2] = -1.0

        if isinstance(self._raycaster, MatterportRayCaster | MatterportRayCasterCamera):
            ray_face_ids = raycast_mesh(
                ray_starts=ray_origins.unsqueeze(0),
                ray_directions=ray_directions.unsqueeze(0),
                max_dist=self.cfg.wall_height * 2,
                return_face_id=True,
                **self._raycaster_mesh_param,
            )[3]

            # assign each hit the semantic class
            class_id = self._raycaster.face_id_category_mapping[self._raycaster.cfg.mesh_prim_paths[0]][
                ray_face_ids.flatten().type(torch.long)
            ]
            # map category index to reduced set
            class_id = self._raycaster.mapping_mpcat40[class_id.type(torch.long) - 1]

            # get class_id to cost mapping
            assert self.cfg.semantic_cost_mapping is not None, "Semantic cost mapping is not available"
            class_id_to_cost = torch.ones(len(self._raycaster.classes_mpcat40), device=self.device) * max(
                list(self.cfg.semantic_cost_mapping.to_dict().values())
            )
            for class_name, class_cost in self.cfg.semantic_cost_mapping.to_dict().items():
                class_id_to_cost[self._raycaster.classes_mpcat40 == class_name] = class_cost

            # get cost
            cost = class_id_to_cost[class_id.cpu()]
        else:
            ray_classes = self._raycast_usd_stage(
                ray_starts=ray_origins,
                ray_directions=ray_directions,
                max_dist=self.cfg.wall_height * 2,
                return_class=True,
            )[3]

            # get class to cost mapping
            assert self.cfg.semantic_cost_mapping is not None, "Semantic cost mapping is not available"
            max_cost = max(list(self.cfg.semantic_cost_mapping.to_dict().values()))
            cost = torch.tensor(
                [
                    self.cfg.semantic_cost_mapping.to_dict()[ray_class] if ray_class is not None else max_cost
                    for ray_class in ray_classes
                ],
                device=self.device,
            )

        # filter points based on cost
        filter_cost = cost < self.cfg.semantic_cost_threshold
        print(f"[DEBUG] filtered {ray_origins.shape[0] - filter_cost.sum().item()} points based on semantic cost")
        return ray_origins[filter_cost].type(torch.float32), heights[filter_cost]

    ###
    # Edge filtering functions
    ###

    def _edge_filter_height_diff(
        self, idx_edge_start: np.ndarray, idx_edge_end: np.ndarray, distance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Filter edges based on height difference between points."""
        # compute height difference
        height_diff = torch.diff(
            self._height_grid, dim=0, append=torch.zeros(1, self._height_grid.shape[1], device=self.device)
        ) + torch.diff(self._height_grid, dim=1, append=torch.zeros(self._height_grid.shape[0], 1, device=self.device))
        height_diff = np.abs(height_diff.cpu().numpy()) > self.cfg.height_diff_threshold

        # identify which edges are on different heights
        if self.cfg.height_diff_edge_filter:
            edge_idx = torch.abs(self.points[idx_edge_start, 2] - self.points[idx_edge_end, 2]) > 0.1
        else:
            edge_idx = torch.ones(self.points[idx_edge_start, 2].shape[0], dtype=bool, device=self.device)

        # filter edges that are on different heights
        check_idx_edge_start = idx_edge_start[edge_idx.cpu().numpy()]
        check_idx_edge_end = idx_edge_end[edge_idx.cpu().numpy()]

        check_grid_idx_start = (
            (
                (
                    self.points[check_idx_edge_start, :2]
                    - torch.tensor([self._mesh_dimensions[2], self._mesh_dimensions[3]], device=self.device)
                )
                / self.cfg.grid_resolution
            )
            .int()
            .cpu()
            .numpy()
        )
        check_grid_idx_end = (
            (
                (
                    self.points[check_idx_edge_end, :2]
                    - torch.tensor([self._mesh_dimensions[2], self._mesh_dimensions[3]], device=self.device)
                )
                / self.cfg.grid_resolution
            )
            .int()
            .cpu()
            .numpy()
        )

        filter_idx = np.zeros(check_idx_edge_start.shape[0], dtype=bool)

        for idx, (edge_start_idx, edge_end_idx) in enumerate(zip(check_grid_idx_start, check_grid_idx_end)):
            grid_idx_x, grid_idx_y = line(edge_start_idx[0], edge_start_idx[1], edge_end_idx[0], edge_end_idx[1])

            filter_idx[idx] = np.any(height_diff[grid_idx_x, grid_idx_y])

        # set the indexes that should be removed in edge_idx to true
        edge_idx[edge_idx.clone()] = torch.tensor(filter_idx, device=self.device)
        edge_idx = edge_idx.cpu().numpy()
        # filter edges
        idx_edge_start_filtered = idx_edge_start[edge_idx]
        idx_edge_end_filtered = idx_edge_end[edge_idx]

        idx_edge_start = idx_edge_start[~edge_idx]
        idx_edge_end = idx_edge_end[~edge_idx]
        distance = distance[~edge_idx]

        return idx_edge_start, idx_edge_end, distance, idx_edge_start_filtered, idx_edge_end_filtered

    def _edge_filter_mesh_collisions(
        self, nearest_neighbors_idx: torch.Tensor
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Filter connections that collide with the environment."""
        # define origin and neighbor points
        origin_point = torch.repeat_interleave(self.points, repeats=self.cfg.num_connections, axis=0)
        neighbor_points = self.points[nearest_neighbors_idx, :].reshape(-1, 3)
        min_distance = torch.norm(origin_point - neighbor_points, dim=1)

        # check for collision with raycasting
        if self._raycaster is not None:
            distance = raycast_mesh(
                ray_starts=origin_point.unsqueeze(0),
                ray_directions=(origin_point - neighbor_points).unsqueeze(0),
                max_dist=self.cfg.max_path_length,
                return_distance=True,
                **self._raycaster_mesh_param,
            )[1]
        else:
            distance = self._raycast_usd_stage(
                ray_starts=origin_point,
                ray_directions=(origin_point - neighbor_points),
                max_dist=self.cfg.max_path_length,
                return_distance=True,
            )[1]

        distance[torch.isinf(distance)] = self.cfg.max_path_length
        # filter connections that collide with the environment
        collision = (distance < min_distance).reshape(-1, self.cfg.num_connections)

        # get edge indices
        idx_edge_start = np.repeat(np.arange(self.cfg.sample_points), repeats=self.cfg.num_connections, axis=0)
        idx_edge_end = nearest_neighbors_idx.reshape(-1).cpu().numpy()

        # filter collision edges and distances
        idx_edge_end = idx_edge_end[~collision.reshape(-1).cpu().numpy()]
        idx_edge_start = idx_edge_start[~collision.reshape(-1).cpu().numpy()]
        distance = min_distance[~collision.reshape(-1)].cpu().numpy()

        return idx_edge_start, idx_edge_end, distance

    def _edge_filter_semantic_cost(
        self, idx_edge_start: np.ndarray, idx_edge_end: np.ndarray, distance: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Filter edges based on height difference between points."""
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(
                self._mesh_dimensions[2],
                self._mesh_dimensions[0],
                int(np.ceil((self._mesh_dimensions[0] - self._mesh_dimensions[2]) / self.cfg.grid_resolution)),
                device=self.device,
            ),
            torch.linspace(
                self._mesh_dimensions[3],
                self._mesh_dimensions[1],
                int(np.ceil((self._mesh_dimensions[1] - self._mesh_dimensions[3]) / self.cfg.grid_resolution)),
                device=self.device,
            ),
        )
        grid_z = torch.ones_like(grid_x, device=self.device) * max(
            list(self.cfg.semantic_cost_mapping.to_dict().values())
        )
        grid_points = torch.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T
        direction = torch.zeros_like(grid_points, device=self.device)
        direction[:, 2] = -1.0

        if isinstance(self._raycaster, MatterportRayCaster | MatterportRayCasterCamera):
            # check for collision with raycasting
            ray_face_ids = raycast_mesh(
                ray_starts=grid_points.unsqueeze(0),
                ray_directions=direction.unsqueeze(0),
                max_dist=self.cfg.wall_height * 2,
                return_face_id=True,
                **self._raycaster_mesh_param,
            )[3].squeeze(0)

            # assign each hit the semantic class
            class_id = self._raycaster.face_id_category_mapping[self._raycaster.cfg.mesh_prim_paths[0]][
                ray_face_ids.flatten().type(torch.long)
            ]
            # map category index to reduced set
            class_id = self._raycaster.mapping_mpcat40[class_id.type(torch.long) - 1]

            # get class_id to cost mapping
            assert self.cfg.semantic_cost_mapping is not None, "Semantic cost mapping is not available"
            class_id_to_cost = torch.ones(len(self._raycaster.classes_mpcat40)) * max(
                list(self.cfg.semantic_cost_mapping.to_dict().values())
            )
            for class_name, class_cost in self.cfg.semantic_cost_mapping.to_dict().items():
                class_id_to_cost[self._raycaster.classes_mpcat40 == class_name] = class_cost

            cost = class_id_to_cost[class_id.cpu()]
        else:
            ray_classes = self._raycast_usd_stage(
                ray_starts=grid_points,
                ray_directions=direction,
                max_dist=self.cfg.wall_height * 2,
                return_class=True,
            )[3]

            # get class to cost mapping
            assert self.cfg.semantic_cost_mapping is not None, "Semantic cost mapping is not available"
            max_cost = max(list(self.cfg.semantic_cost_mapping.to_dict().values()))
            cost = torch.tensor(
                [
                    self.cfg.semantic_cost_mapping.to_dict()[ray_class] if ray_class is not None else max_cost
                    for ray_class in ray_classes
                ],
                device=self.device,
            )

        # get cost grid
        cost_grid = (
            cost.reshape(
                int(np.ceil((self._mesh_dimensions[0] - self._mesh_dimensions[2]) / self.cfg.grid_resolution)),
                int(np.ceil((self._mesh_dimensions[1] - self._mesh_dimensions[3]) / self.cfg.grid_resolution)),
            )
            .cpu()
            .numpy()
        )

        # get grid indexes of edges
        check_grid_idx_start = (
            (
                (
                    self.points[idx_edge_start, :2]
                    - torch.tensor([self._mesh_dimensions[2], self._mesh_dimensions[3]], device=self.points.device)
                )
                / self.cfg.grid_resolution
            )
            .int()
            .cpu()
            .numpy()
        )
        check_grid_idx_end = (
            (
                (
                    self.points[idx_edge_end, :2]
                    - torch.tensor([self._mesh_dimensions[2], self._mesh_dimensions[3]], device=self.points.device)
                )
                / self.cfg.grid_resolution
            )
            .int()
            .cpu()
            .numpy()
        )

        filter_idx = np.zeros(check_grid_idx_start.shape[0], dtype=bool)

        for idx, (edge_start_idx, edge_end_idx) in enumerate(zip(check_grid_idx_start, check_grid_idx_end)):
            grid_idx_x, grid_idx_y = line(edge_start_idx[0], edge_start_idx[1], edge_end_idx[0], edge_end_idx[1])

            filter_idx[idx] = np.any(cost_grid[grid_idx_x, grid_idx_y] > self.cfg.semantic_cost_threshold)

        # filter edges
        idx_edge_start_filtered = idx_edge_start[filter_idx]
        idx_edge_end_filtered = idx_edge_end[filter_idx]

        idx_edge_start = idx_edge_start[~filter_idx]
        idx_edge_end = idx_edge_end[~filter_idx]
        distance = distance[~filter_idx]

        return idx_edge_start, idx_edge_end, distance, idx_edge_start_filtered, idx_edge_end_filtered

    def _setup_raycaster(self):
        # get the raycaster sensor that should be used to raycast against all the ground meshes
        if isinstance(
            self.scene.sensors[self.cfg.raycaster_sensor],
            MatterportRayCaster | MatterportRayCasterCamera | RayCaster | RayCasterCamera,
        ):
            self._raycaster: MatterportRayCaster | MatterportRayCasterCamera | RayCaster | RayCasterCamera = (
                self.scene.sensors[self.cfg.raycaster_sensor]
            )

            if isinstance(self._raycaster.meshes[self._raycaster.cfg.mesh_prim_paths[0]], list):
                # for new RSL implementation of raycaster
                # FIXME: @pascal-roth: this is a temporary fix until the new raycaster is merged into the public main branch
                self._raycaster_mesh_param = {"mesh_id": self._raycaster._mesh_ids_wp.numpy()[0][0]}
            else:
                self._raycaster_mesh_param = {"mesh": self._raycaster.meshes[self._raycaster.cfg.mesh_prim_paths[0]]}

            # get mesh dimensions [x_max, y_max, x_min, y_min]
            self._mesh_dimensions = self._get_mesh_dimensions()
        else:
            # raycaster is not available in multi-mesh scenes (i.e. unreal meshes) as it only works with a single mesh
            # TODO (@pascal-roth) change when raycaster can handle multiple meshes
            self._raycaster = None

            # get mesh dimensions [x_max, y_max, x_min, y_min]
            self._mesh_dimensions = self._get_usd_stage_dimensions()

        self._mesh_dimensions = list(self._mesh_dimensions)

        # limit the size of the mesh if required (otherwise run out of memory)
        if self.cfg.max_terrain_size is not None:
            if self._mesh_dimensions[0] - self._mesh_dimensions[2] > self.cfg.max_terrain_size:
                print(f"[WARNING] Mesh is too large in the x dimension, limiting to {self.cfg.max_terrain_size} max")
                mesh_over_limit = (self._mesh_dimensions[0] - self._mesh_dimensions[2] - self.cfg.max_terrain_size) / 2
                self._mesh_dimensions[0] -= mesh_over_limit
                self._mesh_dimensions[2] += mesh_over_limit
            if self._mesh_dimensions[1] - self._mesh_dimensions[3] > self.cfg.max_terrain_size:
                print(f"[WARNING] Mesh is too large in the y dimension, limiting to {self.cfg.max_terrain_size} max")
                mesh_over_limit = (self._mesh_dimensions[1] - self._mesh_dimensions[3] - self.cfg.max_terrain_size) / 2
                self._mesh_dimensions[1] -= mesh_over_limit
                self._mesh_dimensions[3] += mesh_over_limit

    ###
    # Construct height map of the environment
    ###

    def construct_height_map(self):
        # get dimensions and construct height grid with raycasting
        grid_x, grid_y = torch.meshgrid(
            torch.linspace(
                self._mesh_dimensions[2],
                self._mesh_dimensions[0],
                int(np.ceil((self._mesh_dimensions[0] - self._mesh_dimensions[2]) / self.cfg.grid_resolution)),
                device=self.device,
            ),
            torch.linspace(
                self._mesh_dimensions[3],
                self._mesh_dimensions[1],
                int(np.ceil((self._mesh_dimensions[1] - self._mesh_dimensions[3]) / self.cfg.grid_resolution)),
                device=self.device,
            ),
        )
        grid_z = torch.ones_like(grid_x) * (self.cfg.wall_height * 2)
        grid_points = torch.vstack((grid_x.flatten(), grid_y.flatten(), grid_z.flatten())).T
        direction = torch.zeros_like(grid_points)
        direction[:, 2] = -1.0

        # check for collision with raycasting from the top
        if self._raycaster is not None:
            hit_point = raycast_mesh(
                ray_starts=grid_points.unsqueeze(0),
                ray_directions=direction.unsqueeze(0),
                max_dist=15,
                **self._raycaster_mesh_param,
            )[0].squeeze(0)
        else:
            hit_point = self._raycast_usd_stage(
                ray_starts=grid_points,
                ray_directions=direction,
                max_dist=15,
            )[0]

        # detection of doors inside walls
        # we raycast one more time shortly above the ground up and down, if the up raycast hits and is lower than the
        # initial raycast, the height of the down raycast will be used
        if self.cfg.door_filtering:
            # adopt the height
            grid_points[..., 2] = 0.5

            # check for potential hit downwards
            if self._raycaster is not None:
                hit_point_down = raycast_mesh(
                    ray_starts=grid_points.unsqueeze(0),
                    ray_directions=direction.unsqueeze(0),
                    max_dist=15,
                    **self._raycaster_mesh_param,
                )[0].squeeze(0)
            else:
                hit_point_down = self._raycast_usd_stage(
                    ray_starts=grid_points,
                    ray_directions=direction,
                    max_dist=15,
                )[0]

            # change the direction of the raycaster to the top
            direction[:, 2] = 1.0

            # check for potential hit upwards
            if self._raycaster is not None:
                hit_point_up = raycast_mesh(
                    ray_starts=grid_points.unsqueeze(0),
                    ray_directions=direction.unsqueeze(0),
                    max_dist=15,
                    **self._raycaster_mesh_param,
                )[0].squeeze(0)
            else:
                hit_point_up = self._raycast_usd_stage(
                    ray_starts=grid_points,
                    ray_directions=direction,
                    max_dist=15,
                )[0]

            # check if up height scan receives a hit and is lower than the initial height scan (from above the wall height)
            # and where the difference is larger than the height difference threshold
            lower_height = (
                (hit_point_up[..., 2] < (hit_point[..., 2] - 1e-3))
                & torch.isfinite(hit_point_up[..., 2])
                & ((hit_point_up[..., 2] - hit_point_down[..., 2]) > self.cfg.door_height_threshold)
                & torch.isfinite(hit_point_down[..., 2])
            )
            # override height with the lower height
            hit_point[lower_height] = hit_point_down[lower_height]

        # get the height grid
        self._height_grid = hit_point[:, 2].reshape(
            int(np.ceil((self._mesh_dimensions[0] - self._mesh_dimensions[2]) / self.cfg.grid_resolution)),
            int(np.ceil((self._mesh_dimensions[1] - self._mesh_dimensions[3]) / self.cfg.grid_resolution)),
        )

        if self.cfg.viz_height_map:
            env_render_steps = 1000
            if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
                print(f"[INFO] Visualizing height map. Will do {env_render_steps} render steps...")
            else:
                print("[INFO] Visualizing height map.")

            # in headless mode, we cannot visualize the graph and omni.debug.draw is not available
            try:
                import omni.isaac.debug_draw._debug_draw as omni_debug_draw

                # add small offset to height grid to visualize it
                hit_point[:, 2] += 0.1

                draw_interface = omni_debug_draw.acquire_debug_draw_interface()
                if self.cfg.door_filtering:
                    draw_interface.draw_points(
                        hit_point[lower_height].cpu().tolist(),
                        [(0.0, 0.7, 0.0, 1)] * hit_point[lower_height].shape[0],
                        [5] * hit_point[lower_height].shape[0],
                    )
                    draw_interface.draw_points(
                        hit_point[~lower_height].cpu().tolist(),
                        [(0.0, 0.0, 0.7, 1)] * hit_point[~lower_height].shape[0],
                        [5] * hit_point[~lower_height].shape[0],
                    )
                else:
                    draw_interface.draw_points(
                        hit_point.cpu().tolist(),
                        [(0.0, 0.0, 0.7, 1)] * hit_point.shape[0],
                        [5] * hit_point.shape[0],
                    )

                if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:
                    sim = SimulationContext.instance()
                    for _ in range(env_render_steps):
                        sim.render()

                    # clear the drawn points and lines
                    draw_interface.clear_points()

                    print("[INFO] Finished visualizing height map.")

            except ImportError:
                print("[WARNING] Height Map Visualization is not available in headless mode.")

    ###
    # Helper function when isaaclab raycaster is not available
    ###

    def _raycast_usd_stage(
        self,
        ray_starts: torch.Tensor,
        ray_directions: torch.Tensor,
        max_dist: float = 1e6,
        return_distance: bool = False,
        return_normal: bool = False,
        return_class: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None, list | None]:
        """
        Perform raycasting over the entire loaded stage.

        Interface is the same as the normal raycast_mesh function without the option to provide specific meshes.
        """

        hits = [
            get_physx_scene_query_interface().raycast_closest(carb.Float3(ray_single), carb.Float3(ray_dir), max_dist)
            for ray_single, ray_dir in zip(ray_starts.cpu().numpy(), ray_directions.cpu().numpy())
        ]

        # get all hit idx
        hit_idx = [idx for idx, single_hit in enumerate(hits) if single_hit["hit"]]

        # hit positions
        hit_positions = torch.zeros_like(ray_starts).fill_(float("inf"))
        hit_positions[hit_idx] = torch.tensor([single_hit["position"] for single_hit in hits if single_hit["hit"]]).to(
            ray_starts.device
        )

        # get distance
        if return_distance:
            ray_distance = torch.zeros(ray_starts.shape[0], device=ray_starts.device).fill_(float("inf"))
            ray_distance[hit_idx] = torch.tensor(
                [single_hit["distance"] for single_hit in hits if single_hit["hit"]]
            ).to(ray_starts.device)
        else:
            ray_distance = None

        # get normal
        if return_normal:
            ray_normal = torch.zeros_like(ray_starts).fill_(float("inf"))
            ray_normal[hit_idx] = torch.tensor([single_hit["normal"] for single_hit in hits if single_hit["hit"]])
        else:
            ray_normal = None

        # get class
        if return_class:
            ray_class = [
                (
                    get_semantics(prims_utils.get_prim_at_path(single_hit["collision"]))["Semantics"][1]
                    if single_hit["hit"]
                    else None
                )
                for single_hit in hits
            ]
        else:
            ray_class = None

        return hit_positions, ray_distance, ray_normal, ray_class
