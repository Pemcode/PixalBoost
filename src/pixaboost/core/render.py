"""Offscreen rasterisation of silhouettes and depth, in pure numpy.

Why not a real renderer: the gate must run on CPU, offline, in under 60 s, with
no GPU and no headless GL context (docs/testing.md). Silhouette IoU is the P1
metric for the catalogue use case, so this path is on the critical measurement
route and has to be dependency-free and deterministic.

Cost is O(faces) in Python. That is fine for the metric loop; if F05 makes it a
bottleneck, optimise then, with a measurement in hand.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt

from pixaboost.core.geometry import BlenderCamera, FloatArray, Sim3, project

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]

#: Below this magnitude a projected triangle is a line and covers nothing.
_DEGENERATE_AREA_PX = 1e-12


def _validate(vertices: FloatArray, faces: IntArray) -> tuple[FloatArray, IntArray]:
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if vertices.ndim != 2 or vertices.shape[1] != 3:
        raise ValueError(f"vertices must have shape (V, 3), got {vertices.shape}")
    if faces.ndim != 2 or faces.shape[1] != 3:
        raise ValueError(f"faces must have shape (F, 3), got {faces.shape}")
    if faces.size and (faces.min() < 0 or faces.max() >= vertices.shape[0]):
        raise ValueError("faces index vertices out of range")
    return vertices, faces


def rasterise_depth(
    vertices: FloatArray,
    faces: IntArray,
    camera_to_world: Sim3,
    camera: BlenderCamera,
) -> FloatArray:
    """Z-buffer the mesh, returning depth per pixel and ``inf`` where nothing was hit.

    Triangles are drawn regardless of winding: we measure occupancy, not shading,
    and a back-facing triangle still occludes. Depth is interpolated
    perspective-correctly, i.e. linearly in ``1 / depth``.
    """
    vertices, faces = _validate(vertices, faces)
    resolution = camera.resolution
    depth_buffer = np.full((resolution, resolution), np.inf, dtype=np.float64)
    if faces.shape[0] == 0:
        return depth_buffer

    pixels, vertex_depth, _ = project(vertices, camera_to_world, camera)

    for face in faces:
        corner_depth = vertex_depth[face]
        # Triangles crossing the camera plane need clipping to be drawn correctly;
        # nothing in this pipeline relies on it, so skip them rather than approximate.
        if np.any(corner_depth <= 0.0):
            continue

        corner = pixels[face]
        min_col = max(int(np.floor(corner[:, 0].min() - 0.5)), 0)
        max_col = min(int(np.ceil(corner[:, 0].max() + 0.5)), resolution - 1)
        min_row = max(int(np.floor(corner[:, 1].min() - 0.5)), 0)
        max_row = min(int(np.ceil(corner[:, 1].max() + 0.5)), resolution - 1)
        if min_col > max_col or min_row > max_row:
            continue

        edge_a = corner[1] - corner[0]
        edge_b = corner[2] - corner[0]
        signed_area = edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]
        if abs(signed_area) < _DEGENERATE_AREA_PX:
            continue

        cols = np.arange(min_col, max_col + 1, dtype=np.float64) + 0.5
        rows = np.arange(min_row, max_row + 1, dtype=np.float64) + 0.5
        grid_x, grid_y = np.meshgrid(cols, rows)

        # Edge functions, normalised by the signed area so that the barycentric
        # coordinates come out positive inside the triangle for either winding.
        weight_0 = (
            (corner[1, 0] - grid_x) * (corner[2, 1] - grid_y)
            - (corner[1, 1] - grid_y) * (corner[2, 0] - grid_x)
        ) / signed_area
        weight_1 = (
            (corner[2, 0] - grid_x) * (corner[0, 1] - grid_y)
            - (corner[2, 1] - grid_y) * (corner[0, 0] - grid_x)
        ) / signed_area
        weight_2 = 1.0 - weight_0 - weight_1

        inside = (weight_0 >= 0.0) & (weight_1 >= 0.0) & (weight_2 >= 0.0)
        if not inside.any():
            continue

        inverse_depth = (
            weight_0 / corner_depth[0] + weight_1 / corner_depth[1] + weight_2 / corner_depth[2]
        )
        candidate = np.where(inside & (inverse_depth > 0.0), 1.0 / inverse_depth, np.inf)

        window = depth_buffer[min_row : max_row + 1, min_col : max_col + 1]
        np.minimum(window, candidate, out=window)

    return depth_buffer


def rasterise_silhouette(
    vertices: FloatArray,
    faces: IntArray,
    camera_to_world: Sim3,
    camera: BlenderCamera,
) -> BoolArray:
    """Boolean coverage mask: exactly the pixels where the depth buffer is finite."""
    return np.isfinite(rasterise_depth(vertices, faces, camera_to_world, camera))
