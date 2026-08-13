"""TDD for core.render (F03).

A pure-numpy rasteriser. It exists so silhouette and depth comparisons run on
CPU, offline, inside the 60 s gate -- no GPU renderer, no headless GL context.

Expected values are computed from the projected geometry itself rather than
hardcoded, so the assertions stay analytic rather than golden.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixaboost.core.geometry import BlenderCamera, front_view_camera, project
from pixaboost.core.render import rasterise_depth, rasterise_silhouette

CAMERA = BlenderCamera(camera_angle_x=0.857556, resolution=128)
VIEW = front_view_camera(2.0)


def facing_triangle(size: float = 0.3, y: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """A triangle in the world plane y = const, i.e. square-on to the front view."""
    vertices = np.array([[0.0, y, 0.0], [size, y, 0.0], [0.0, y, size]], dtype=np.float64)
    return vertices, np.array([[0, 1, 2]], dtype=np.int64)


def projected_area_px(vertices: np.ndarray) -> float:
    pixels, _, _ = project(vertices, VIEW, CAMERA)
    edge_a, edge_b = pixels[1] - pixels[0], pixels[2] - pixels[0]
    return float(abs(edge_a[0] * edge_b[1] - edge_a[1] * edge_b[0]) / 2.0)


# ---------------------------------------------------------------------------
# Silhouette
# ---------------------------------------------------------------------------


def test_silhouette_has_the_camera_resolution() -> None:
    vertices, faces = facing_triangle()
    silhouette = rasterise_silhouette(vertices, faces, VIEW, CAMERA)
    assert silhouette.shape == (CAMERA.resolution, CAMERA.resolution)
    assert silhouette.dtype == np.bool_


def test_covered_pixel_count_matches_the_projected_triangle_area() -> None:
    # Large enough that the O(perimeter) discretisation error stays a few percent.
    vertices, faces = facing_triangle(size=0.8)
    covered = int(rasterise_silhouette(vertices, faces, VIEW, CAMERA).sum())
    assert covered == pytest.approx(projected_area_px(vertices), rel=0.08)


def test_an_empty_mesh_produces_an_empty_silhouette() -> None:
    empty_v = np.zeros((0, 3), dtype=np.float64)
    empty_f = np.zeros((0, 3), dtype=np.int64)
    assert not rasterise_silhouette(empty_v, empty_f, VIEW, CAMERA).any()


def test_geometry_behind_the_camera_is_not_drawn() -> None:
    vertices, faces = facing_triangle(y=-5.0)  # further along -Y than the camera itself
    assert not rasterise_silhouette(vertices, faces, VIEW, CAMERA).any()


def test_geometry_outside_the_frame_is_not_drawn() -> None:
    vertices, faces = facing_triangle(size=0.2)
    assert not rasterise_silhouette(
        vertices + np.array([50.0, 0.0, 0.0]), faces, VIEW, CAMERA
    ).any()


def test_a_triangle_edge_on_to_the_camera_covers_almost_nothing() -> None:
    """Degenerate projection must not fill the frame or crash."""
    vertices = np.array([[0.0, -0.3, 0.0], [0.0, 0.3, 0.0], [0.0, 0.0, 0.3]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    assert int(rasterise_silhouette(vertices, faces, VIEW, CAMERA).sum()) < CAMERA.resolution


def test_winding_order_does_not_cull_geometry() -> None:
    """We rasterise silhouettes, not shaded surfaces: back-facing triangles count."""
    vertices, faces = facing_triangle()
    flipped = faces[:, ::-1].copy()
    np.testing.assert_array_equal(
        rasterise_silhouette(vertices, faces, VIEW, CAMERA),
        rasterise_silhouette(vertices, flipped, VIEW, CAMERA),
    )


# ---------------------------------------------------------------------------
# Depth
# ---------------------------------------------------------------------------


def test_a_plane_square_on_to_the_camera_has_constant_depth() -> None:
    vertices, faces = facing_triangle(size=0.3, y=0.25)
    depth = rasterise_depth(vertices, faces, VIEW, CAMERA)
    hit = np.isfinite(depth)
    assert hit.any()
    # depth = distance + y for the canonical front view.
    np.testing.assert_allclose(depth[hit], 2.25, atol=1e-9)


def test_empty_pixels_are_infinite() -> None:
    vertices, faces = facing_triangle(size=0.1)
    depth = rasterise_depth(vertices, faces, VIEW, CAMERA)
    assert np.isinf(depth).any()
    assert np.isfinite(depth).any()


def test_the_nearer_surface_wins_the_depth_buffer() -> None:
    near_v, _ = facing_triangle(size=0.3, y=-0.5)
    far_v, _ = facing_triangle(size=0.3, y=0.5)
    vertices = np.vstack([near_v, far_v])
    faces = np.array([[0, 1, 2], [3, 4, 5]], dtype=np.int64)
    depth = rasterise_depth(vertices, faces, VIEW, CAMERA)
    overlap = np.isfinite(depth)
    assert overlap.any()
    np.testing.assert_allclose(depth[overlap].min(), 1.5, atol=1e-9)
    assert depth[overlap].max() == pytest.approx(1.5, abs=1e-9), "the far triangle must be hidden"


def test_silhouette_is_exactly_where_depth_is_finite() -> None:
    vertices, faces = facing_triangle(size=0.4)
    depth = rasterise_depth(vertices, faces, VIEW, CAMERA)
    np.testing.assert_array_equal(
        rasterise_silhouette(vertices, faces, VIEW, CAMERA), np.isfinite(depth)
    )


def test_depth_interpolation_is_perspective_correct() -> None:
    """A triangle slanted in depth: screen-space linear interpolation of depth
    itself would be wrong; only 1/depth is linear in screen space."""
    vertices = np.array([[-0.4, -0.4, 0.0], [0.4, 0.4, 0.0], [0.0, 0.4, 0.4]], dtype=np.float64)
    faces = np.array([[0, 1, 2]], dtype=np.int64)
    camera = BlenderCamera(camera_angle_x=1.2, resolution=96)
    depth = rasterise_depth(vertices, faces, VIEW, camera)
    hit = np.isfinite(depth)
    assert hit.any()

    pixels, vertex_depth, _ = project(vertices, VIEW, camera)
    rows, cols = np.nonzero(hit)
    query = np.stack([cols + 0.5, rows + 0.5], axis=-1)
    matrix = np.array(
        [
            [pixels[0, 0] - pixels[2, 0], pixels[1, 0] - pixels[2, 0]],
            [pixels[0, 1] - pixels[2, 1], pixels[1, 1] - pixels[2, 1]],
        ]
    )
    weights = np.linalg.solve(matrix, (query - pixels[2]).T)
    bary = np.vstack([weights, 1.0 - weights.sum(axis=0)])
    expected = 1.0 / (bary / vertex_depth[:, None]).sum(axis=0)
    np.testing.assert_allclose(depth[hit], expected, rtol=1e-6)


def test_a_degenerate_face_is_background_without_warning() -> None:
    """An edge-on triangle divides by zero internally; it must stay silent.

    Surfaced by the F15 pose search, which sweeps orientations and therefore
    hits exactly-edge-on faces routinely. The pixels were already correct --
    the warning was not.
    """
    import warnings

    from pixaboost.core.geometry import BlenderCamera, front_view_camera
    from pixaboost.core.render import rasterise_silhouette

    # A triangle lying in the plane that contains the viewing direction.
    vertices = np.array([[0.0, -0.2, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]])
    faces = np.array([[0, 1, 2]])
    camera = BlenderCamera(camera_angle_x=0.857556, resolution=32)

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        mask = rasterise_silhouette(vertices, faces, front_view_camera(2.0), camera)

    assert mask.shape == (32, 32)
