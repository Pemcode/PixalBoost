"""Analytic contract for render-and-compare pose search (F15).

The whole point: recover the relative pose between two views *without any
calibration*, using the object as its own target. Every fixture here is
closed-form -- a mesh from the benchmark catalogue, rendered at a rotation we
chose, then handed back to the search with that rotation withheld.

The interesting case is the axisymmetric one. ADR-0007 treats symmetry as a
reason to refuse a pose, and it is right when the pose is *estimated from
geometry alone*. Here the conclusion inverts: many azimuths tie, and that is
harmless, because a rotation about a true symmetry axis does not change the
shape. The search must therefore report the ambiguity and still return a pose
that renders correctly.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixaboost.bench.shapes import flange_ring, l_bracket, stepped_shaft
from pixaboost.core.geometry import BlenderCamera, front_view_camera
from pixaboost.core.metrics import silhouette_iou
from pixaboost.core.pose_search import (
    OPPOSITE_FACES,
    PoseSearchResult,
    crop_to_canonical_framing,
    object_rotation,
    rotation_angle_between,
    search_object_pose,
)
from pixaboost.core.render import rasterise_silhouette

DISTANCE = 2.0
CAMERA = BlenderCamera(camera_angle_x=0.857556, resolution=48)


def render_at(mesh: tuple[np.ndarray, np.ndarray], rotation: np.ndarray) -> np.ndarray:
    vertices, faces = mesh
    return rasterise_silhouette(
        vertices @ rotation.T, faces, front_view_camera(DISTANCE), CAMERA
    )


def search(
    mesh: tuple[np.ndarray, np.ndarray], target: np.ndarray, **kw: object
) -> PoseSearchResult:
    """Deliberately coarser than the production default, to keep the gate under 60 s."""
    settings: dict[str, object] = {
        "azimuth_steps": 12,
        "elevation_steps": 3,
        "refine_rounds": 3,
        **kw,
    }
    vertices, faces = mesh
    return search_object_pose(
        vertices, faces, target, camera=CAMERA, distance=DISTANCE, **settings  # type: ignore[arg-type]
    )


# --------------------------------------------------------------------------
# the basic contract: recover a rotation we hid
# --------------------------------------------------------------------------


def test_the_identity_pose_is_recovered_from_its_own_render() -> None:
    mesh = l_bracket()
    result = search(mesh, render_at(mesh, np.eye(3)))
    assert result.iou > 0.95
    assert rotation_angle_between(result.rotation, np.eye(3)) < np.deg2rad(20)


@pytest.mark.parametrize("azimuth_deg", [45.0, 195.0, 300.0])
def test_a_hidden_azimuth_is_recovered_on_an_asymmetric_part(azimuth_deg: float) -> None:
    """This is the no-calibration claim, reduced to its smallest testable form."""
    mesh = l_bracket()
    truth = object_rotation(np.deg2rad(azimuth_deg), 0.0, 0.0)

    result = search(mesh, render_at(mesh, truth))

    assert result.iou > 0.90, f"silhouette did not match: {result.iou:.3f}"
    assert rotation_angle_between(result.rotation, truth) < np.deg2rad(20)


def test_an_elevated_view_is_recovered_too() -> None:
    """Needs a finer elevation grid than the others: the default 3 steps span
    120 degrees, and local refinement cannot climb out of the wrong bracket."""
    mesh = l_bracket()
    truth = object_rotation(np.deg2rad(60.0), np.deg2rad(30.0), 0.0)
    result = search(mesh, render_at(mesh, truth), elevation_steps=5)
    assert result.iou > 0.90
    assert rotation_angle_between(result.rotation, truth) < np.deg2rad(25)


def test_on_a_revolved_part_the_elevation_is_recovered_but_the_azimuth_is_not() -> None:
    """Exactly what a symmetry axis costs, and exactly what it does not.

    `stepped_shaft` is revolved about Z, so azimuth is unobservable by
    construction -- no algorithm can recover it, and none needs to. Elevation
    tilts the axis itself and is therefore fully determined.
    """
    mesh = stepped_shaft()
    truth = object_rotation(np.deg2rad(60.0), np.deg2rad(30.0), 0.0)
    target = render_at(mesh, truth)

    result = search(mesh, target)

    assert silhouette_iou(render_at(mesh, result.rotation), target) > 0.90
    # The shape is invariant under `p -> Rz(d) p` in its own frame, so an
    # equally good answer is `truth @ Rz(d)`. The residual to inspect is
    # therefore `truth.T @ recovered`, which must be a rotation about Z.
    residual = truth.T @ result.rotation
    assert float(abs(residual[2, 2])) > 0.97, "the residual must be a spin about the axis"
    assert rotation_angle_between(result.rotation, truth) > np.deg2rad(20), (
        "and the azimuth genuinely differs: this is the unobservable degree of freedom"
    )


def test_the_returned_rotation_actually_reproduces_the_target() -> None:
    """The score must be a property of the pose, not a number carried alongside it.

    Both sides are compared through the canonical framing, because that is what
    the search itself optimises -- a raw comparison would measure scale and
    centring too.
    """
    mesh = l_bracket()
    target = render_at(mesh, object_rotation(np.deg2rad(150.0), np.deg2rad(-20.0), 0.0))

    result = search(mesh, target)

    reproduced = crop_to_canonical_framing(render_at(mesh, result.rotation), CAMERA.resolution)
    assert silhouette_iou(
        reproduced, crop_to_canonical_framing(target, CAMERA.resolution)
    ) == pytest.approx(result.iou, abs=1e-9)


# --------------------------------------------------------------------------
# symmetry: ADR-0007 inverted
# --------------------------------------------------------------------------


def test_an_axisymmetric_part_is_reported_ambiguous_yet_still_renders_correctly() -> None:
    """The ambiguity is real and harmless: a spin about the axis changes nothing.

    ADR-0007 refuses such a pose because it estimates orientation from geometry
    alone. Here the pose is only ever used to *render*, so a tie between
    equivalent azimuths costs nothing.
    """
    mesh = flange_ring()
    truth = object_rotation(np.deg2rad(70.0), 0.0, 0.0)
    target = render_at(mesh, truth)

    result = search(mesh, target)

    assert result.is_ambiguous, "a revolved part must not claim a unique azimuth"
    assert result.iou > 0.95, "yet the mask it produces must still match"
    assert silhouette_iou(render_at(mesh, result.rotation), target) > 0.95


def test_an_asymmetric_part_is_not_reported_ambiguous() -> None:
    mesh = l_bracket()
    result = search(mesh, render_at(mesh, object_rotation(np.deg2rad(35.0), 0.0, 0.0)))
    assert not result.is_ambiguous


# --------------------------------------------------------------------------
# candidates, determinism, refusals
# --------------------------------------------------------------------------


def test_candidates_are_reported_best_first() -> None:
    mesh = l_bracket()
    result = search(mesh, render_at(mesh, np.eye(3)))
    scores = [c.iou for c in result.candidates]
    assert scores == sorted(scores, reverse=True)
    assert result.iou == pytest.approx(scores[0])


def test_the_search_is_deterministic() -> None:
    mesh = stepped_shaft()
    target = render_at(mesh, object_rotation(np.deg2rad(80.0), 0.0, 0.0))
    first, second = search(mesh, target), search(mesh, target)
    assert first.iou == second.iou
    assert np.array_equal(first.rotation, second.rotation)


def test_an_empty_target_mask_is_refused() -> None:
    mesh = l_bracket()
    with pytest.raises(ValueError, match="empty"):
        search(mesh, np.zeros((CAMERA.resolution, CAMERA.resolution), dtype=bool))


def test_a_target_at_the_photographs_own_resolution_is_accepted() -> None:
    """A SAM mask arrives at 2048 px, never at the search resolution.

    Requiring the caller to pre-resize was the earlier contract and it invited
    exactly the mistake this module exists to prevent: comparing a mask and a
    render that do not share a framing.
    """
    mesh = l_bracket()
    truth = object_rotation(np.deg2rad(40.0), 0.0, 0.0)
    big = np.zeros((300, 400), dtype=bool)
    small = render_at(mesh, truth)
    big[100:100 + small.shape[0], 200:200 + small.shape[1]] = small

    result = search(mesh, big)

    assert result.iou > 0.85
    assert rotation_angle_between(result.rotation, truth) < np.deg2rad(20)


def test_a_non_two_dimensional_target_is_refused() -> None:
    mesh = l_bracket()
    with pytest.raises(ValueError, match="2-D"):
        search(mesh, np.ones((8, 9, 3), dtype=bool))


def test_a_finer_grid_never_scores_worse_than_a_coarse_one() -> None:
    """Monotonicity is the only guarantee a grid search can honestly offer."""
    mesh = l_bracket()
    target = render_at(mesh, object_rotation(np.deg2rad(23.0), np.deg2rad(11.0), 0.0))

    coarse = search(mesh, target, azimuth_steps=6, elevation_steps=3, refine_rounds=0)
    fine = search(mesh, target, azimuth_steps=24, elevation_steps=7, refine_rounds=2)

    assert fine.iou >= coarse.iou - 1e-9


# --------------------------------------------------------------------------
# the rotation helpers themselves
# --------------------------------------------------------------------------


def test_object_rotation_is_a_proper_rotation() -> None:
    rotation = object_rotation(0.7, -0.3, 1.1)
    assert np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(rotation) == pytest.approx(1.0)


def test_rotation_angle_between_matches_a_known_offset() -> None:
    a = object_rotation(0.0, 0.0, 0.0)
    b = object_rotation(np.deg2rad(37.0), 0.0, 0.0)
    assert rotation_angle_between(a, b) == pytest.approx(np.deg2rad(37.0), abs=1e-9)


def test_rotation_angle_between_is_zero_for_identical_rotations() -> None:
    rotation = object_rotation(1.0, 0.4, -0.2)
    assert rotation_angle_between(rotation, rotation) == pytest.approx(0.0, abs=1e-9)


# --------------------------------------------------------------------------
# the canonical crop -- the silent killer identified in the vendor pipeline
# --------------------------------------------------------------------------


def test_the_target_mask_is_cropped_the_way_pixal3d_crops() -> None:
    """`preprocess_image` recentres each view on its own alpha bbox, square, x1.1.

    Comparing a raw SAM mask against a canonical render would compare two
    different framings, and the pose search would quietly converge on a wrong
    scale. The mask must be put through the same rule.
    """
    mask = np.zeros((400, 600), dtype=bool)
    mask[150:250, 250:350] = True  # 100x100 blob, centred at (200, 300)

    cropped = crop_to_canonical_framing(mask, resolution=64)

    assert cropped.shape == (64, 64)
    assert cropped.any()
    # A square blob occupying 1/1.1 of a square crop covers (1/1.1)^2 ~ 83 %.
    assert 0.78 < cropped.mean() < 0.88


def test_the_crop_is_scale_invariant() -> None:
    """The same part filmed closer must produce the same canonical mask."""
    small = np.zeros((400, 400), dtype=bool)
    small[190:210, 190:210] = True
    large = np.zeros((400, 400), dtype=bool)
    large[100:300, 100:300] = True

    near, far = (
        crop_to_canonical_framing(small, resolution=32),
        crop_to_canonical_framing(large, resolution=32),
    )
    assert near.shape == far.shape
    # Coverage is the invariant that matters; nearest-neighbour resampling of a
    # 22 px crop and a 220 px one cannot agree pixel for pixel.
    assert abs(near.mean() - far.mean()) < 0.02


def test_a_blob_touching_the_border_is_still_centred() -> None:
    """The 1.1 margin runs off the image; the crop must pad rather than shift."""
    mask = np.zeros((200, 200), dtype=bool)
    mask[0:60, 0:60] = True

    cropped = crop_to_canonical_framing(mask, resolution=48)

    assert cropped.shape == (48, 48)
    rows, cols = np.nonzero(cropped)
    assert abs(rows.mean() - 24) < 4 and abs(cols.mean() - 24) < 4


def test_an_empty_mask_cannot_be_cropped() -> None:
    with pytest.raises(ValueError, match="empty"):
        crop_to_canonical_framing(np.zeros((10, 10), dtype=bool), resolution=16)


# --------------------------------------------------------------------------
# the prior: one bit the photographer always has and the silhouette never does
# --------------------------------------------------------------------------


def test_without_a_prior_two_opposite_faces_of_a_disc_are_indistinguishable() -> None:
    """The measured degeneracy, stated before it is fixed.

    A revolved part seen front and back has the *same* outline, so the search
    scores a 180 degree flip no better than the identity. It reports the tie
    honestly -- and that is all it can do from silhouettes alone.
    """
    mesh = flange_ring()
    flipped = object_rotation(np.pi, 0.0, 0.0)
    target = render_at(mesh, flipped)

    result = search(mesh, target)

    assert result.is_ambiguous
    identity_scores_as_well = silhouette_iou(
        crop_to_canonical_framing(render_at(mesh, np.eye(3)), CAMERA.resolution),
        crop_to_canonical_framing(target, CAMERA.resolution),
    )
    assert identity_scores_as_well > result.iou - 0.05, (
        "if the identity ever stops matching, this fixture is no longer degenerate"
    )


def test_a_prior_confines_the_search_to_orientations_near_it() -> None:
    mesh = l_bracket()
    prior = object_rotation(np.pi, 0.0, 0.0)

    result = search(
        mesh,
        render_at(mesh, object_rotation(np.deg2rad(170.0), 0.0, 0.0)),
        prior_rotation=prior,
        max_deviation=np.deg2rad(45.0),
    )

    assert rotation_angle_between(result.rotation, prior) <= np.deg2rad(45.0) + 1e-9
    assert all(
        rotation_angle_between(c.rotation, prior) <= np.deg2rad(45.0) + 1e-9
        for c in result.candidates
    )


def test_the_prior_makes_the_flip_recoverable_on_a_symmetric_part() -> None:
    """The point of the whole thing: the photographer knows it is the back."""
    mesh = flange_ring()
    flipped = object_rotation(np.pi, 0.0, 0.0)

    result = search(
        mesh,
        render_at(mesh, flipped),
        prior_rotation=flipped,
        max_deviation=np.deg2rad(60.0),
    )

    assert result.iou > 0.90
    assert rotation_angle_between(result.rotation, np.eye(3)) > np.deg2rad(90.0), (
        "the answer must be a flip, not the identity the silhouette also allows"
    )


def test_a_prior_that_excludes_every_grid_orientation_is_refused() -> None:
    """Silence here would mean searching an empty set and returning nothing.

    The prior sits at 15 degrees, deliberately between two grid points of the
    30-degree sweep, with a bound too tight to reach either.
    """
    mesh = l_bracket()
    with pytest.raises(ValueError, match="prior"):
        search(
            mesh,
            render_at(mesh, np.eye(3)),
            prior_rotation=object_rotation(np.deg2rad(15.0), 0.0, 0.0),
            max_deviation=np.deg2rad(0.5),
        )


def test_a_prior_without_a_deviation_is_refused_rather_than_silently_ignored() -> None:
    mesh = l_bracket()
    with pytest.raises(ValueError, match="max_deviation"):
        search(mesh, render_at(mesh, np.eye(3)), prior_rotation=np.eye(3))


def test_opposite_faces_is_a_half_turn_about_the_vertical_axis() -> None:
    """The named relation the GUI offers, pinned so it cannot drift."""
    assert np.allclose(OPPOSITE_FACES, object_rotation(np.pi, 0.0, 0.0), atol=1e-12)
