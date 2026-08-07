"""TDD for bench.rig (F05).

The rig mirrors the real capture protocol: 6 azimuths x 3 elevations
(+45, 0, -45 degrees), 18 views. Keeping the synthetic and real protocols
identical is what stops a synthetic-to-real comparison from confounding the
domain gap with a capture-geometry difference.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixaboost.bench.rig import AZIMUTH_COUNT, ELEVATIONS_DEG, ViewPose, capture_rig
from pixaboost.core.geometry import front_view_camera


def test_the_rig_has_one_view_per_azimuth_and_elevation() -> None:
    rig = capture_rig(distance=2.0)
    assert len(rig) == AZIMUTH_COUNT * len(ELEVATIONS_DEG) == 18
    assert all(isinstance(view, ViewPose) for view in rig)


def test_the_rig_reproduces_the_real_capture_protocol() -> None:
    assert AZIMUTH_COUNT == 6
    assert sorted(ELEVATIONS_DEG) == [-45.0, 0.0, 45.0]


def test_view_names_are_unique_and_stable() -> None:
    names = [view.name for view in capture_rig(distance=2.0)]
    assert len(set(names)) == len(names)
    assert names == [view.name for view in capture_rig(distance=2.0)]


def test_every_camera_sits_at_the_requested_distance_from_the_origin() -> None:
    for view in capture_rig(distance=3.5):
        assert np.linalg.norm(view.camera_to_world.translation) == pytest.approx(3.5)


def test_every_camera_looks_at_the_origin() -> None:
    """Blender cameras look down their own -Z; that ray must hit the origin."""
    for view in capture_rig(distance=2.0):
        pose = view.camera_to_world
        direction = pose.rotation @ np.array([0.0, 0.0, -1.0])
        np.testing.assert_allclose(
            direction, -pose.translation / np.linalg.norm(pose.translation), atol=1e-9
        )


def test_the_reference_view_matches_pixal3d_canonical_front_camera() -> None:
    """Azimuth 0, elevation 0 must be exactly the conditioning camera Pixal3D
    was trained on (docs/pixal3d-internals.md), or every pose we hand the model
    is silently offset."""
    reference = next(
        v for v in capture_rig(distance=2.0) if v.azimuth_deg == 0.0 and v.elevation_deg == 0.0
    )
    expected = front_view_camera(2.0)
    np.testing.assert_allclose(reference.camera_to_world.rotation, expected.rotation, atol=1e-12)
    np.testing.assert_allclose(
        reference.camera_to_world.translation, expected.translation, atol=1e-12
    )


def test_positive_elevation_places_the_camera_above_the_object() -> None:
    rig = capture_rig(distance=2.0)
    for view in rig:
        height = view.camera_to_world.translation[2]
        if view.elevation_deg > 0:
            assert height > 0, f"{view.name} should look down at the part"
        elif view.elevation_deg < 0:
            assert height < 0
        else:
            assert height == pytest.approx(0.0, abs=1e-12)


def test_azimuths_are_evenly_spaced_over_a_full_turn() -> None:
    azimuths = sorted({view.azimuth_deg for view in capture_rig(distance=2.0)})
    np.testing.assert_allclose(azimuths, [0.0, 60.0, 120.0, 180.0, 240.0, 300.0])


def test_poses_are_rigid() -> None:
    """A capture rig has no scale ambiguity: only the reconstruction does."""
    for view in capture_rig(distance=2.0):
        assert view.camera_to_world.scale == pytest.approx(1.0)


def test_a_non_positive_distance_is_rejected() -> None:
    with pytest.raises(ValueError, match="distance"):
        capture_rig(distance=0.0)
