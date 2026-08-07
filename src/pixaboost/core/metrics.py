"""Geometry metrics for comparing a reconstruction against a reference.

Ranked for the catalogue use case in docs/methodology.md: silhouette IoU is P1
(it is what a viewer actually sees, and it exposes a hallucinated back face),
F-score is P2, Chamfer is diagnostic only.

LPIPS is deliberately not here. It is a neural network: it needs torch and
downloads weights, which would break the offline gate. See ADR-0003.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.spatial import cKDTree

from pixaboost.core.geometry import FloatArray

IntArray = npt.NDArray[np.int64]
BoolArray = npt.NDArray[np.bool_]


@dataclass(frozen=True)
class FScore:
    """Precision and recall at a distance threshold, plus their harmonic mean."""

    precision: float
    recall: float
    f1: float


def silhouette_iou(predicted: BoolArray, reference: BoolArray) -> float:
    """Intersection over union of two coverage masks.

    Two empty masks score 1.0: they agree that nothing is there, and scoring
    that as 0 would punish agreement on an empty view.
    """
    predicted = np.asarray(predicted, dtype=bool)
    reference = np.asarray(reference, dtype=bool)
    if predicted.shape != reference.shape:
        raise ValueError(f"masks must share a shape, got {predicted.shape} and {reference.shape}")
    union = int(np.count_nonzero(predicted | reference))
    if union == 0:
        return 1.0
    return int(np.count_nonzero(predicted & reference)) / union


def _as_cloud(points: FloatArray, name: str) -> FloatArray:
    points = np.asarray(points, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3), got {points.shape}")
    if points.shape[0] == 0:
        raise ValueError(f"{name} is empty; a distance to an empty cloud is undefined")
    return points


def _nearest_distances(source: FloatArray, target: FloatArray) -> FloatArray:
    distances: FloatArray = cKDTree(target).query(source, k=1)[0]
    return distances


def chamfer_distance(predicted: FloatArray, reference: FloatArray) -> float:
    """Symmetric mean nearest-neighbour distance, averaged over both directions.

    Diagnostic only. It is dominated by outliers and has no scale-free reading,
    so it never gates a decision -- see docs/methodology.md.
    """
    predicted = _as_cloud(predicted, "predicted")
    reference = _as_cloud(reference, "reference")
    forward = float(_nearest_distances(predicted, reference).mean())
    backward = float(_nearest_distances(reference, predicted).mean())
    return (forward + backward) / 2.0


def f_score(predicted: FloatArray, reference: FloatArray, threshold: float) -> FScore:
    """Fraction of points with a counterpart within ``threshold``.

    Precision looks from the reconstruction outwards (how much of what we built
    is real), recall from the reference inwards (how much of the object we
    recovered). The threshold is conventionally 1 % of the reference bounding
    box diagonal -- see `bbox_diagonal`.
    """
    if not np.isfinite(threshold) or threshold <= 0.0:
        raise ValueError(f"threshold must be finite and strictly positive, got {threshold}")
    predicted = _as_cloud(predicted, "predicted")
    reference = _as_cloud(reference, "reference")

    precision = float((_nearest_distances(predicted, reference) <= threshold).mean())
    recall = float((_nearest_distances(reference, predicted) <= threshold).mean())
    total = precision + recall
    return FScore(
        precision=precision,
        recall=recall,
        f1=0.0 if total == 0.0 else 2.0 * precision * recall / total,
    )


def bbox_diagonal(points: FloatArray) -> float:
    """Diagonal of the axis-aligned bounding box; the natural unit for thresholds."""
    points = _as_cloud(points, "points")
    return float(np.linalg.norm(points.max(axis=0) - points.min(axis=0)))


def sample_surface(
    vertices: FloatArray,
    faces: IntArray,
    count: int,
    seed: int,
) -> FloatArray:
    """Draw ``count`` points uniformly over the mesh surface.

    Area-weighted, not face-weighted: a tessellation that splits one region into
    many small triangles must not bias the sample towards it. `seed` is required
    rather than optional -- `core` forbids implicit global randomness, so that a
    metric is reproducible bit for bit (core/ARCHITECTURE.md).
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    faces = np.asarray(faces, dtype=np.int64)
    if faces.ndim != 2 or faces.shape[1] != 3 or faces.shape[0] == 0:
        raise ValueError(f"faces must have shape (F, 3) with F > 0, got {faces.shape}")
    if count <= 0:
        raise ValueError(f"count must be positive, got {count}")

    corner_a = vertices[faces[:, 0]]
    edge_a = vertices[faces[:, 1]] - corner_a
    edge_b = vertices[faces[:, 2]] - corner_a
    areas = 0.5 * np.linalg.norm(np.cross(edge_a, edge_b), axis=1)
    total_area = float(areas.sum())
    if total_area <= 0.0:
        raise ValueError("mesh has zero total surface area; cannot sample it")

    generator = np.random.default_rng(seed)
    chosen = generator.choice(faces.shape[0], size=count, p=areas / total_area)

    # Uniform barycentric sampling: fold the unit square onto its lower triangle.
    u = generator.random(count)
    v = generator.random(count)
    folded = u + v > 1.0
    u[folded], v[folded] = 1.0 - u[folded], 1.0 - v[folded]

    return corner_a[chosen] + u[:, None] * edge_a[chosen] + v[:, None] * edge_b[chosen]
