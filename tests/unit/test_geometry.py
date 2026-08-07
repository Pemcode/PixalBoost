"""TDD for core.geometry (F02).

Every assertion here is analytic: closed-form expected values, no fixtures, no
recorded model output. See docs/testing.md.

The camera conventions under test are Pixal3D's, mapped in F01
(docs/pixal3d-internals.md): Blender frame, camera looking down -Z, 32 mm
sensor, focal = 16 / tan(fov / 2).
"""

from __future__ import annotations

import numpy as np
import pytest

from pixaboost.core.geometry import (
    BLENDER_GRID_ROTATION,
    BlenderCamera,
    Sim3,
    backproject,
    canonical_grid_points,
    front_view_camera,
    project,
)


def rotation(axis: tuple[float, float, float], angle: float) -> np.ndarray:
    """Rodrigues rotation, so the tests do not depend on the implementation."""
    k = np.asarray(axis, dtype=np.float64)
    k = k / np.linalg.norm(k)
    kx = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]], dtype=np.float64)
    return np.eye(3) + np.sin(angle) * kx + (1 - np.cos(angle)) * (kx @ kx)


def sample_transform() -> Sim3:
    return Sim3(
        rotation=rotation((0.3, -0.7, 0.5), 0.9),
        translation=np.array([1.5, -0.25, 4.0]),
        scale=2.5,
    )


def sample_points() -> np.ndarray:
    rng = np.random.default_rng(20260806)
    return rng.uniform(-0.5, 0.5, size=(32, 3))


# ---------------------------------------------------------------------------
# Sim3
# ---------------------------------------------------------------------------


def test_identity_leaves_points_untouched() -> None:
    points = sample_points()
    np.testing.assert_allclose(Sim3.identity().apply(points), points)


def test_transform_composed_with_its_inverse_is_the_identity() -> None:
    composed = sample_transform().compose(sample_transform().inverse())
    np.testing.assert_allclose(composed.rotation, np.eye(3), atol=1e-12)
    np.testing.assert_allclose(composed.translation, np.zeros(3), atol=1e-12)
    assert composed.scale == pytest.approx(1.0)


def test_inverse_undoes_apply() -> None:
    transform = sample_transform()
    points = sample_points()
    np.testing.assert_allclose(transform.inverse().apply(transform.apply(points)), points)


def test_matrix_round_trip_preserves_the_transform() -> None:
    transform = sample_transform()
    restored = Sim3.from_matrix(transform.as_matrix())
    np.testing.assert_allclose(restored.rotation, transform.rotation)
    np.testing.assert_allclose(restored.translation, transform.translation)
    assert restored.scale == pytest.approx(transform.scale)


def test_compose_agrees_with_matrix_multiplication() -> None:
    a = sample_transform()
    b = Sim3(
        rotation=rotation((1.0, 0.0, 0.0), -0.4), translation=np.array([0.0, 2.0, -1.0]), scale=0.5
    )
    np.testing.assert_allclose(a.compose(b).as_matrix(), a.as_matrix() @ b.as_matrix(), atol=1e-12)


def test_scale_multiplies_pairwise_distances() -> None:
    """The whole reason poses are Sim3 and not SE3 -- see core/ARCHITECTURE.md."""
    points = sample_points()
    scaled = Sim3(rotation=np.eye(3), translation=np.zeros(3), scale=3.0).apply(points)
    original = np.linalg.norm(points[1:] - points[:-1], axis=1)
    after = np.linalg.norm(scaled[1:] - scaled[:-1], axis=1)
    np.testing.assert_allclose(after, 3.0 * original)


def test_non_orthonormal_rotation_is_rejected() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        Sim3(rotation=np.diag([1.0, 1.0, 2.0]), translation=np.zeros(3), scale=1.0)


def test_non_positive_scale_is_rejected() -> None:
    with pytest.raises(ValueError, match="scale"):
        Sim3(rotation=np.eye(3), translation=np.zeros(3), scale=0.0)


# ---------------------------------------------------------------------------
# Camera intrinsics -- Pixal3D's 32 mm sensor model
# ---------------------------------------------------------------------------


def test_focal_length_has_the_expected_closed_form_at_ninety_degrees() -> None:
    """fov = 90 deg -> focal = 16 / tan(45 deg) = 16 mm -> f_px = 16 * res / 32 = res / 2."""
    camera = BlenderCamera(camera_angle_x=np.pi / 2, resolution=512)
    assert camera.focal_px == pytest.approx(256.0)


def test_focal_length_matches_the_vendored_formula() -> None:
    """Parity with vendor/pixal3d inference.py:118-121 and image_conditioned_proj.py:84-86."""
    fov, resolution = 0.8575560450553894, 518
    expected = (16.0 / np.tan(fov / 2.0)) * resolution / 32.0
    assert BlenderCamera(camera_angle_x=fov, resolution=resolution).focal_px == pytest.approx(
        expected
    )


# ---------------------------------------------------------------------------
# Projection -- Pixal3D's canonical front view
# ---------------------------------------------------------------------------


def test_front_view_places_the_camera_at_minus_y_looking_at_the_origin() -> None:
    """Reproduces vendor front_view_transform_matrix (image_conditioned_proj.py:172-178)."""
    camera_to_world = front_view_camera(distance=2.0)
    np.testing.assert_allclose(camera_to_world.translation, [0.0, -2.0, 0.0], atol=1e-12)
    # Blender cameras look down their own -Z; that axis must point at the origin, i.e. +Y.
    view_direction = camera_to_world.rotation @ np.array([0.0, 0.0, -1.0])
    np.testing.assert_allclose(view_direction, [0.0, 1.0, 0.0], atol=1e-12)


def test_origin_projects_to_the_image_centre_at_the_camera_distance() -> None:
    camera = BlenderCamera(camera_angle_x=0.857556, resolution=512)
    pixels, depth, valid = project(np.zeros((1, 3)), front_view_camera(2.0), camera)
    np.testing.assert_allclose(pixels[0], [256.0, 256.0], atol=1e-9)
    assert depth[0] == pytest.approx(2.0)
    assert bool(valid[0])


def test_projection_round_trips_through_backprojection() -> None:
    camera = BlenderCamera(camera_angle_x=0.857556, resolution=512)
    camera_to_world = front_view_camera(2.0)
    points = sample_points()
    pixels, depth, valid = project(points, camera_to_world, camera)
    assert bool(valid.all())
    np.testing.assert_allclose(
        backproject(pixels, depth, camera_to_world, camera), points, atol=1e-9
    )


def test_points_behind_the_camera_are_invalid() -> None:
    camera = BlenderCamera(camera_angle_x=0.857556, resolution=512)
    behind = np.array([[0.0, -5.0, 0.0]])  # further along -Y than the camera itself
    _, depth, valid = project(behind, front_view_camera(2.0), camera)
    assert depth[0] < 0
    assert not bool(valid[0])


def test_points_outside_the_frame_are_invalid() -> None:
    camera = BlenderCamera(camera_angle_x=0.2, resolution=512)  # narrow fov
    _, _, valid = project(np.array([[4.0, 0.0, 0.0]]), front_view_camera(2.0), camera)
    assert not bool(valid[0])


def test_y_axis_is_flipped_so_that_world_up_is_image_up() -> None:
    camera = BlenderCamera(camera_angle_x=0.857556, resolution=512)
    pixels, _, _ = project(np.array([[0.0, 0.0, 0.1]]), front_view_camera(2.0), camera)
    assert pixels[0, 1] < 256.0, "a point above the origin must land above the image centre"


# ---------------------------------------------------------------------------
# The conditioning grid
# ---------------------------------------------------------------------------


def test_blender_grid_rotation_matches_the_vendored_constant() -> None:
    """image_conditioned_proj.py:162-166, repeated at inference.py:125."""
    np.testing.assert_array_equal(
        BLENDER_GRID_ROTATION, [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]
    )


def test_canonical_grid_has_one_point_per_voxel_inside_the_export_aabb() -> None:
    """Grid spans [-1, 1], divided by mesh_scale then by 2 -> the [-0.5, 0.5] export AABB."""
    grid = canonical_grid_points(resolution=16)
    assert grid.shape == (16**3, 3)
    assert grid.min() == pytest.approx(-0.5)
    assert grid.max() == pytest.approx(0.5)


def test_canonical_grid_scales_inversely_with_mesh_scale() -> None:
    dense = canonical_grid_points(resolution=8, mesh_scale=2.0)
    assert dense.max() == pytest.approx(0.25)


def test_canonical_grid_is_ordered_like_the_vendored_meshgrid() -> None:
    """indexing='ij' over (x, y, z), then rotated. The order is what maps a flat
    conditioning vector back onto voxel coordinates, so it cannot drift."""
    grid = canonical_grid_points(resolution=2, mesh_scale=1.0)
    axis = np.array([-0.5, 0.5])
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    expected = np.stack((x, y, z), axis=-1).reshape(-1, 3) @ BLENDER_GRID_ROTATION.T
    np.testing.assert_allclose(grid, expected)
