"""TDD for core.registration (F11).

Pixal3D emits no extrinsics -- `camera_params` is only
`{camera_angle_x, distance, mesh_scale}` (docs/pixal3d-internals.md) -- so every
pair of single-view reconstructions has to be aligned by geometry alone, at
7 degrees of freedom because MoGe-2's depth scale differs per view.

The load-bearing requirement is not accuracy, it is **refusal**. A wrong pose
makes multi-view fusion worse than single view, and it does so silently. So the
interesting tests here are the ones where registration must decline: symmetric
parts, unrelated clouds, degenerate inputs.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixaboost.core.geometry import Sim3
from pixaboost.core.registration import (
    DEFAULT_RESTARTS,
    ICP_BASIN_DEG,
    IcpFit,
    RegistrationRejected,
    azimuthal_initialisations,
    coarse_alignment,
    icp_sim3,
    register,
    umeyama_sim3,
)


def rotation_about_z(angle: float) -> np.ndarray:
    cos, sin = np.cos(angle), np.sin(angle)
    return np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]])


def asymmetric_cloud(count: int = 240, seed: int = 3) -> np.ndarray:
    """A blob with no rotational symmetry, so its pose is genuinely determined."""
    rng = np.random.default_rng(seed)
    points = rng.normal(size=(count, 3))
    points[:, 0] *= 1.0
    points[:, 1] *= 0.45
    points[:, 2] *= 0.2
    points[: count // 3] += np.array([1.4, 0.9, 0.0])  # a lug, breaking symmetry
    return points


def symmetric_ring(fold: int = 12, per_fold: int = 20, seed: int = 5) -> np.ndarray:
    """A flange-like ring with exact `fold` rotational symmetry about Z."""
    rng = np.random.default_rng(seed)
    base = np.column_stack(
        [
            rng.uniform(0.9, 1.0, per_fold),
            np.zeros(per_fold),
            rng.uniform(-0.1, 0.1, per_fold),
        ]
    )
    angles = np.linspace(0.0, 2.0 * np.pi, fold, endpoint=False)
    sector = 2.0 * np.pi / fold
    spread = rng.uniform(0.0, sector, per_fold)
    wedge = np.column_stack(
        [
            base[:, 0] * np.cos(spread),
            base[:, 0] * np.sin(spread),
            base[:, 2],
        ]
    )
    return np.vstack([wedge @ rotation_about_z(a).T for a in angles])


KNOWN = Sim3(rotation=rotation_about_z(0.4), translation=np.array([0.3, -0.2, 0.1]), scale=1.7)


# ---------------------------------------------------------------------------
# Umeyama: the closed-form core
# ---------------------------------------------------------------------------


def test_umeyama_recovers_a_known_transform_exactly() -> None:
    source = asymmetric_cloud()
    recovered = umeyama_sim3(source, KNOWN.apply(source))
    np.testing.assert_allclose(recovered.rotation, KNOWN.rotation, atol=1e-9)
    np.testing.assert_allclose(recovered.translation, KNOWN.translation, atol=1e-9)
    assert recovered.scale == pytest.approx(KNOWN.scale, rel=1e-9)


def test_umeyama_recovers_scale_alone() -> None:
    source = asymmetric_cloud()
    pure_scale = Sim3(rotation=np.eye(3), translation=np.zeros(3), scale=0.25)
    assert umeyama_sim3(source, pure_scale.apply(source)).scale == pytest.approx(0.25)


def test_umeyama_never_returns_a_reflection() -> None:
    """A mirrored cloud must be fitted with a proper rotation, not a flip: a
    reflected reconstruction is not a valid rigid alignment."""
    source = asymmetric_cloud()
    mirrored = source * np.array([1.0, 1.0, -1.0])
    assert np.linalg.det(umeyama_sim3(source, mirrored).rotation) == pytest.approx(1.0)


def test_umeyama_needs_at_least_three_points() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        umeyama_sim3(np.zeros((2, 3)), np.zeros((2, 3)))


def test_umeyama_rejects_mismatched_counts() -> None:
    with pytest.raises(ValueError, match="same number"):
        umeyama_sim3(np.zeros((5, 3)), np.zeros((4, 3)))


def test_umeyama_rejects_a_degenerate_source() -> None:
    with pytest.raises(ValueError, match="degenerate"):
        umeyama_sim3(np.ones((8, 3)), asymmetric_cloud(8))


# ---------------------------------------------------------------------------
# ICP: no correspondences given
# ---------------------------------------------------------------------------


def test_icp_converges_from_a_perturbed_initialisation() -> None:
    source = asymmetric_cloud()
    target = KNOWN.apply(source)
    nudged = Sim3(
        rotation=rotation_about_z(0.4 + 0.25) @ KNOWN.rotation @ rotation_about_z(-0.4),
        translation=KNOWN.translation + 0.05,
        scale=KNOWN.scale * 1.1,
    )
    fit = icp_sim3(source, target, initial=nudged)
    assert fit.converged
    assert fit.rmse < 1e-6
    np.testing.assert_allclose(fit.transform.apply(source), target, atol=1e-6)


def test_icp_reports_a_scale_free_rmse() -> None:
    """Normalised by the target bounding box, so a threshold means the same
    thing whatever units a reconstruction happens to arrive in."""
    source = asymmetric_cloud()
    small = icp_sim3(source, source, initial=Sim3.identity())
    blown_up = Sim3(rotation=np.eye(3), translation=np.zeros(3), scale=1000.0)
    large = icp_sim3(blown_up.apply(source), blown_up.apply(source), initial=Sim3.identity())
    assert small.rmse == pytest.approx(large.rmse, abs=1e-9)


def test_icp_tolerates_partial_overlap() -> None:
    """Views only ever share part of a surface, so trimming is not optional."""
    source = asymmetric_cloud(count=300, seed=11)
    target = KNOWN.apply(source)[: int(0.7 * 300)]
    fit = icp_sim3(source, target, initial=KNOWN, trim_ratio=0.6)
    np.testing.assert_allclose(fit.transform.rotation, KNOWN.rotation, atol=1e-3)


def test_icp_stops_at_the_iteration_budget() -> None:
    source = asymmetric_cloud()
    fit = icp_sim3(source, asymmetric_cloud(seed=99), initial=Sim3.identity(), max_iterations=3)
    assert fit.iterations <= 3
    assert isinstance(fit, IcpFit)


# ---------------------------------------------------------------------------
# Confidence and refusal -- the point of the feature
# ---------------------------------------------------------------------------


def test_a_clean_asymmetric_registration_is_accepted() -> None:
    source = asymmetric_cloud()
    result = register(source, KNOWN.apply(source))
    assert result.confidence > 0.8
    np.testing.assert_allclose(result.transform.rotation, KNOWN.rotation, atol=1e-4)


def test_a_rotationally_symmetric_part_is_refused() -> None:
    """The failure this feature exists to prevent.

    A 12-fold ring fits its own rotation by any multiple of 30 degrees equally
    well. Residual alone would call that an excellent registration; the pose is
    in fact undetermined, and fusing on it would smear the part.
    """
    ring = symmetric_ring()
    with pytest.raises(RegistrationRejected, match="ambiguous"):
        register(ring, ring.copy())


def test_the_symmetric_part_is_refused_despite_a_perfect_residual() -> None:
    """Why the confidence score cannot be residual-only.

    The ring aligns to itself *exactly* -- zero error, every point an inlier. A
    fit-quality score would rate it 1.0 and hand back a pose picked essentially
    at random from twelve equally valid ones. Only the ambiguity term separates
    "this is right" from "this is one of many".
    """
    ring = symmetric_ring()
    inspected = register(ring, ring.copy(), min_confidence=0.0)
    assert inspected.rmse < 1e-9, "the fit itself is perfect"
    assert inspected.inlier_ratio == pytest.approx(1.0), "every point is explained"
    assert inspected.distinctness < 0.1, "yet the pose is not determined"
    assert inspected.confidence < 0.1


def test_the_refusal_names_the_rival_pose() -> None:
    ring = symmetric_ring()
    try:
        register(ring, ring.copy())
    except RegistrationRejected as error:
        message = str(error)
        assert "WHAT" in message and "WHY" in message and "FIX" in message
        assert "distinctness" in message


def test_incompatible_shapes_are_refused() -> None:
    """Two clouds of different intrinsic shape cannot be explained by any pose.

    Note this is stated as *incompatible*, not merely *unrelated*: once centroids
    and extents are matched, two similar blobs really do align, and accepting
    that is correct. What must be refused is geometry no similarity can reconcile
    -- here a one-dimensional curve against a three-dimensional volume.
    """
    turns = np.linspace(0.0, 6.0 * np.pi, 300)
    helix = np.column_stack([np.cos(turns), np.sin(turns), turns / 6.0])
    with pytest.raises(RegistrationRejected):
        register(helix, asymmetric_cloud(count=300, seed=2))


def test_the_threshold_is_the_caller_s_to_set() -> None:
    ring = symmetric_ring()
    permissive = register(ring, ring.copy(), min_confidence=0.0)
    assert permissive.distinctness < 0.2, "the ambiguity must still be reported, not hidden"


def test_confidence_degrades_as_noise_grows() -> None:
    source = asymmetric_cloud(count=400, seed=21)
    rng = np.random.default_rng(7)
    scores = []
    for sigma in (0.0, 0.02, 0.08):
        target = KNOWN.apply(source) + rng.normal(scale=sigma, size=source.shape)
        scores.append(register(source, target, min_confidence=0.0).confidence)
    assert scores[0] > scores[1] > scores[2]


def test_confidence_is_bounded() -> None:
    source = asymmetric_cloud()
    result = register(source, KNOWN.apply(source), min_confidence=0.0)
    assert 0.0 <= result.confidence <= 1.0
    assert 0.0 <= result.distinctness <= 1.0
    assert 0.0 <= result.inlier_ratio <= 1.0


# ---------------------------------------------------------------------------
# Initialisations
# ---------------------------------------------------------------------------


def test_azimuthal_initialisations_span_a_full_turn() -> None:
    starts = azimuthal_initialisations(6)
    assert len(starts) == 6
    assert all(start.scale == pytest.approx(1.0) for start in starts)
    angles = sorted(round(float(np.arctan2(s.rotation[1, 0], s.rotation[0, 0])), 6) for s in starts)
    assert len(set(angles)) == 6


def test_the_default_restart_spacing_stays_inside_the_icp_basin() -> None:
    """Encodes a measurement so it is never silently re-derived.

    ICP converges from about 20 degrees of initial rotation error and sticks
    beyond that. If restarts were spaced wider than the basin, orientations in
    the middle of a gap would be unreachable no matter how good the data -- and
    the bad fit would be misread as low confidence rather than a bad search.
    """
    assert 360.0 / DEFAULT_RESTARTS < ICP_BASIN_DEG


def test_icp_converges_from_anywhere_inside_the_basin() -> None:
    source = asymmetric_cloud()
    target = KNOWN.apply(source)
    truth_angle = np.degrees(np.arccos((np.trace(KNOWN.rotation) - 1.0) / 2.0))
    centre = target.mean(axis=0)
    coarse = coarse_alignment(source, target)

    for offset in (truth_angle - 15.0, truth_angle, truth_angle + 15.0):
        spin = Sim3(
            rotation=rotation_about_z(np.radians(offset)), translation=np.zeros(3), scale=1.0
        )
        start = Sim3(
            rotation=spin.rotation,
            translation=centre - spin.rotation @ centre,
            scale=1.0,
        ).compose(coarse)
        fit = icp_sim3(source, target, initial=start)
        assert fit.rmse < 1e-9, f"failed from {offset - truth_angle:+.0f} deg of error"


def test_ambiguity_cannot_be_assessed_from_a_single_start() -> None:
    """Refusing here is deliberate: one start gives no rival to compare against,
    so a symmetric part would sail through undetected."""
    with pytest.raises(ValueError, match="at least two"):
        register(asymmetric_cloud(), asymmetric_cloud(), initialisations=[Sim3.identity()])
