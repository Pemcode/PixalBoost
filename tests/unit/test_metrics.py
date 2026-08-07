"""TDD for core.metrics (F03).

Geometry-only metrics. LPIPS is deliberately absent: it needs torch and
downloads weights, which would break the offline 60 s gate. It lives in
backends/perceptual.py instead -- see ADR-0003.

Ranking of these metrics for the catalogue use case is set in
docs/methodology.md: silhouette IoU is P1, F-score is P2, Chamfer is diagnostic.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixaboost.core.metrics import (
    bbox_diagonal,
    chamfer_distance,
    f_score,
    sample_surface,
    silhouette_iou,
)


def unit_square_mesh() -> tuple[np.ndarray, np.ndarray]:
    vertices = np.array(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float64
    )
    return vertices, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)


# ---------------------------------------------------------------------------
# Silhouette IoU -- the P1 metric
# ---------------------------------------------------------------------------


def test_a_mask_against_itself_scores_one() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[2:6, 2:6] = True
    assert silhouette_iou(mask, mask) == pytest.approx(1.0)


def test_complementary_masks_score_zero() -> None:
    mask = np.zeros((8, 8), dtype=bool)
    mask[:4] = True
    assert silhouette_iou(mask, ~mask) == pytest.approx(0.0)


def test_half_overlap_scores_one_third() -> None:
    """Two 4x8 bands sharing a 2x8 strip: intersection 16, union 48."""
    a = np.zeros((8, 8), dtype=bool)
    b = np.zeros((8, 8), dtype=bool)
    a[0:4] = True
    b[2:6] = True
    assert silhouette_iou(a, b) == pytest.approx(16 / 48)


def test_two_empty_masks_score_one() -> None:
    """Both agree there is nothing there. Returning 0 would punish agreement."""
    empty = np.zeros((4, 4), dtype=bool)
    assert silhouette_iou(empty, empty) == pytest.approx(1.0)


def test_iou_is_symmetric() -> None:
    rng = np.random.default_rng(7)
    a = rng.random((16, 16)) > 0.5
    b = rng.random((16, 16)) > 0.3
    assert silhouette_iou(a, b) == pytest.approx(silhouette_iou(b, a))


def test_mismatched_mask_shapes_are_rejected() -> None:
    with pytest.raises(ValueError, match="shape"):
        silhouette_iou(np.zeros((4, 4), dtype=bool), np.zeros((4, 5), dtype=bool))


# ---------------------------------------------------------------------------
# Chamfer -- diagnostic only
# ---------------------------------------------------------------------------


def test_chamfer_of_a_cloud_against_itself_is_zero() -> None:
    rng = np.random.default_rng(11)
    points = rng.random((64, 3))
    assert chamfer_distance(points, points) == pytest.approx(0.0)


def test_chamfer_of_two_single_points_is_their_distance() -> None:
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 3.0]])
    assert chamfer_distance(a, b) == pytest.approx(3.0)


def test_chamfer_is_symmetric() -> None:
    rng = np.random.default_rng(13)
    a, b = rng.random((32, 3)), rng.random((48, 3))
    assert chamfer_distance(a, b) == pytest.approx(chamfer_distance(b, a))


def test_chamfer_averages_both_directions() -> None:
    """One reference point sits far from the two prediction points.

    a -> b nearest distances: 0 and 1, mean 0.5. b -> a: 0 and 1, mean 0.5.
    """
    a = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    assert chamfer_distance(a, b) == pytest.approx(0.5)


def test_chamfer_rejects_an_empty_cloud() -> None:
    with pytest.raises(ValueError, match="empty"):
        chamfer_distance(np.zeros((0, 3)), np.ones((3, 3)))


# ---------------------------------------------------------------------------
# F-score -- the P2 metric
# ---------------------------------------------------------------------------


def test_identical_clouds_score_a_perfect_f_score() -> None:
    rng = np.random.default_rng(17)
    points = rng.random((32, 3))
    result = f_score(points, points, threshold=1e-6)
    assert result.precision == pytest.approx(1.0)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(1.0)


def test_clouds_further_apart_than_the_threshold_score_zero() -> None:
    a = np.array([[0.0, 0.0, 0.0]])
    b = np.array([[10.0, 0.0, 0.0]])
    assert f_score(a, b, threshold=1.0).f1 == pytest.approx(0.0)


def test_precision_and_recall_are_counted_independently() -> None:
    """Two of four predictions land on the single reference point.

    precision = 2/4, recall = 1/1, f1 = 2 * .5 * 1 / 1.5.
    """
    a = np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [5.0, 0.0, 0.0], [6.0, 0.0, 0.0]])
    b = np.array([[0.0, 0.0, 0.0]])
    result = f_score(a, b, threshold=0.1)
    assert result.precision == pytest.approx(0.5)
    assert result.recall == pytest.approx(1.0)
    assert result.f1 == pytest.approx(2 / 3)


def test_a_non_positive_threshold_is_rejected() -> None:
    with pytest.raises(ValueError, match="threshold"):
        f_score(np.zeros((1, 3)), np.zeros((1, 3)), threshold=0.0)


# ---------------------------------------------------------------------------
# Surface sampling and bbox
# ---------------------------------------------------------------------------


def test_bbox_diagonal_of_the_unit_cube_is_root_three() -> None:
    corners = np.array(np.meshgrid([0.0, 1.0], [0.0, 1.0], [0.0, 1.0])).reshape(3, -1).T
    assert bbox_diagonal(corners) == pytest.approx(np.sqrt(3.0))


def test_sampling_returns_the_requested_count() -> None:
    vertices, faces = unit_square_mesh()
    assert sample_surface(vertices, faces, count=500, seed=0).shape == (500, 3)


def test_sampling_is_reproducible_for_a_given_seed() -> None:
    vertices, faces = unit_square_mesh()
    np.testing.assert_array_equal(
        sample_surface(vertices, faces, count=64, seed=42),
        sample_surface(vertices, faces, count=64, seed=42),
    )


def test_different_seeds_give_different_samples() -> None:
    vertices, faces = unit_square_mesh()
    assert not np.array_equal(
        sample_surface(vertices, faces, count=64, seed=1),
        sample_surface(vertices, faces, count=64, seed=2),
    )


def test_samples_lie_on_the_surface() -> None:
    vertices, faces = unit_square_mesh()
    points = sample_surface(vertices, faces, count=256, seed=3)
    np.testing.assert_allclose(points[:, 1], 0.0, atol=1e-12)
    assert points[:, 0].min() >= -1e-12
    assert points[:, 0].max() <= 1.0 + 1e-12


def test_sampling_is_area_weighted_not_face_weighted() -> None:
    """One triangle carries 99 % of the area; face-uniform sampling would give 50 %."""
    vertices = np.array(
        [[0.0, 0.0, 0.0], [10.0, 0.0, 0.0], [0.0, 0.0, 10.0], [0.0, 0.0, -0.1], [0.1, 0.0, -0.1]],
        dtype=np.float64,
    )
    faces = np.array([[0, 1, 2], [0, 3, 4]], dtype=np.int64)
    points = sample_surface(vertices, faces, count=4000, seed=5)
    on_big_triangle = float((points[:, 2] >= 0.0).mean())
    assert on_big_triangle > 0.95


def test_sampling_rejects_a_mesh_with_no_faces() -> None:
    with pytest.raises(ValueError, match="faces"):
        sample_surface(np.zeros((3, 3)), np.zeros((0, 3), dtype=np.int64), count=8, seed=0)
