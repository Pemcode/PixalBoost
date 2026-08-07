"""Aligning two reconstructions, and knowing when not to trust the result.

Pixal3D emits no extrinsics: `camera_params` carries only
`{camera_angle_x, distance, mesh_scale}` (docs/pixal3d-internals.md). Every
single-view reconstruction therefore arrives in its own frame, at MoGe-2's own
depth scale, and aligning two of them is a 7-DoF problem -- which is why this
module speaks `Sim3` and not `SE3`.

**Refusal is the feature.** A wrong pose does not make multi-view fusion
slightly worse than single view; it smears the part, and it does so without
raising anything. So the confidence score deliberately measures two different
things:

- *fit*: how small the residual is, and how much of the cloud is explained;
- *distinctness*: whether a materially different pose fits about as well.

The second matters because this project targets machined parts, which are
routinely symmetric. A flange fits its own rotation at every bolt-hole spacing,
and a residual-only score would call that an excellent registration while the
pose is in fact undetermined. Confidence is the product of the two, so a
perfect fit with a rival pose still scores near zero.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.spatial import cKDTree

from pixaboost.core.geometry import FloatArray, Sim3
from pixaboost.core.metrics import bbox_diagonal

#: Residual, as a fraction of the target bounding box, at which fit quality hits zero.
RMSE_TOLERANCE = 0.05
#: A point counts as explained when it lands within this fraction of the bounding box.
INLIER_TOLERANCE = 0.02
#: Two poses are "the same answer" below this rotational separation, in radians.
DISTINCT_ANGLE = np.radians(10.0)
#: Measured basin of convergence of point-to-point ICP on an unstructured cloud.
#: Beyond roughly this much initial rotation error it settles into a local
#: minimum instead of the true pose. See ADR-0006.
ICP_BASIN_DEG = 20.0
#: Restart spacing must stay inside the basin, or some orientations are
#: unreachable however many restarts are used.
DEFAULT_RESTARTS = 24
#: Below this, `register` refuses rather than returning a pose nobody should use.
MIN_CONFIDENCE = 0.5


class RegistrationRejected(RuntimeError):
    """The alignment was not trustworthy enough to build on."""


@dataclass(frozen=True)
class IcpFit:
    """One converged alignment attempt."""

    transform: Sim3
    rmse: float
    inlier_ratio: float
    iterations: int
    converged: bool


@dataclass(frozen=True)
class RegistrationResult:
    """An alignment, with the evidence for and against believing it."""

    transform: Sim3
    rmse: float
    inlier_ratio: float
    distinctness: float
    confidence: float
    iterations: int
    converged: bool


def umeyama_sim3(source: FloatArray, target: FloatArray) -> Sim3:
    """Closed-form optimal similarity between *corresponding* point sets.

    Umeyama (1991). The reflection guard matters: without it a noisy or mirrored
    cloud can be fitted with `det(R) = -1`, which is not a rotation and would
    hand a mirrored part to everything downstream.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape:
        raise ValueError(
            f"source and target must have the same number of points, "
            f"got {source.shape} and {target.shape}"
        )
    if source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3), got {source.shape}")
    if source.shape[0] < 3:
        raise ValueError(f"need at least 3 points to fit a similarity, got {source.shape[0]}")

    count = source.shape[0]
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)
    centred_source = source - source_mean
    centred_target = target - target_mean

    source_variance = float((centred_source**2).sum() / count)
    if source_variance <= 1e-15:
        raise ValueError("degenerate source: all points coincide, no orientation to recover")

    covariance = centred_target.T @ centred_source / count
    left, singular_values, right = np.linalg.svd(covariance)

    correction = np.eye(3)
    if np.linalg.det(left) * np.linalg.det(right) < 0.0:
        correction[2, 2] = -1.0  # forbid the reflection

    rotation = left @ correction @ right
    scale = float(np.trace(np.diag(singular_values) @ correction) / source_variance)
    if scale <= 0.0:
        raise ValueError("degenerate fit: recovered a non-positive scale")

    return Sim3(
        rotation=rotation,
        translation=target_mean - scale * (rotation @ source_mean),
        scale=scale,
    )


def icp_sim3(
    source: FloatArray,
    target: FloatArray,
    *,
    initial: Sim3 | None = None,
    max_iterations: int = 60,
    tolerance: float = 1e-10,
    trim_ratio: float = 0.8,
) -> IcpFit:
    """Iterative closest point over a similarity, with trimming.

    Trimming is not an optional refinement here: two views only ever share part
    of a surface, so the worst-matching fraction is expected to be genuinely
    unmatched rather than merely noisy. Fitting it would drag the pose.

    `rmse` is normalised by the target bounding box diagonal, so a threshold
    means the same thing whatever scale a reconstruction arrives in.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if not 0.0 < trim_ratio <= 1.0:
        raise ValueError(f"trim_ratio must lie in (0, 1], got {trim_ratio}")

    tree = cKDTree(target)
    extent = bbox_diagonal(target)
    keep_count = max(3, round(trim_ratio * source.shape[0]))

    transform = initial or Sim3.identity()
    previous = np.inf
    normalised = np.inf
    converged = False
    iterations = 0

    while iterations < max_iterations:
        iterations += 1
        distances, indices = tree.query(transform.apply(source), k=1)
        order = np.argsort(distances)[:keep_count]
        transform = umeyama_sim3(source[order], target[indices[order]])

        trimmed_rmse = float(np.sqrt(np.mean(distances[order] ** 2)))
        normalised = trimmed_rmse / extent
        if abs(previous - normalised) < tolerance:
            converged = True
            break
        previous = normalised

    distances, _ = tree.query(transform.apply(source), k=1)
    return IcpFit(
        transform=transform,
        rmse=float(np.sqrt(np.mean((distances / extent) ** 2))),
        inlier_ratio=float((distances / extent <= INLIER_TOLERANCE).mean()),
        iterations=iterations,
        converged=converged,
    )


def azimuthal_initialisations(count: int = DEFAULT_RESTARTS) -> list[Sim3]:
    """Evenly spaced rotations about the world up axis.

    Restarting does two jobs at once: it escapes the local minimum a single
    start settles into, and it is what makes the ambiguity check possible at all
    -- with one start there is no rival to compare against.

    **The count is not arbitrary.** ICP refines, it does not search: measured on
    an unstructured cloud, it converges from about 20 degrees of initial
    rotation error and gets stuck beyond that. Six restarts leave 60-degree gaps,
    so orientations near the middle of a gap are simply unreachable -- and the
    resulting bad fits then look like low confidence rather than a bad search.
    Spacing has to stay inside the basin. See ADR-0006.
    """
    if count < 2:
        raise ValueError(f"need at least two initialisations to assess ambiguity, got {count}")
    starts = []
    for angle in np.linspace(0.0, 2.0 * np.pi, count, endpoint=False):
        cos, sin = np.cos(angle), np.sin(angle)
        starts.append(
            Sim3(
                rotation=np.array([[cos, -sin, 0.0], [sin, cos, 0.0], [0.0, 0.0, 1.0]]),
                translation=np.zeros(3),
                scale=1.0,
            )
        )
    return starts


def coarse_alignment(source: FloatArray, target: FloatArray) -> Sim3:
    """Match centroids and overall extent, ignoring orientation.

    ICP only refines; it does not search. Starting it from a pure rotation while
    the true similarity carries a 1.7x scale and an offset leaves it in a local
    minimum, which then looks like a low-confidence result rather than a bad
    initialisation. Absorbing translation and scale first means every restart
    differs only in the one degree of freedom worth searching: orientation.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    source_mean = source.mean(axis=0)
    target_mean = target.mean(axis=0)

    source_spread = float(np.sqrt(np.mean(((source - source_mean) ** 2).sum(axis=1))))
    target_spread = float(np.sqrt(np.mean(((target - target_mean) ** 2).sum(axis=1))))
    if source_spread <= 1e-15:
        raise ValueError("degenerate source: all points coincide, no extent to match")

    scale = target_spread / source_spread if target_spread > 0.0 else 1.0
    return Sim3(rotation=np.eye(3), translation=target_mean - scale * source_mean, scale=scale)


def _spin_about(centre: FloatArray, rotation: FloatArray) -> Sim3:
    """A rotation applied around `centre` rather than the origin."""
    return Sim3(rotation=rotation, translation=centre - rotation @ centre, scale=1.0)


def _angle_between(first: Sim3, second: Sim3) -> float:
    relative = first.rotation.T @ second.rotation
    cosine = (float(np.trace(relative)) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def register(
    source: FloatArray,
    target: FloatArray,
    *,
    initialisations: list[Sim3] | None = None,
    min_confidence: float = MIN_CONFIDENCE,
) -> RegistrationResult:
    """Align `source` onto `target`, refusing when the pose is not trustworthy.

    Raises `RegistrationRejected` below `min_confidence`. Pass
    ``min_confidence=0.0`` to inspect a rejected alignment rather than act on it
    -- the ambiguity is still reported, never hidden.
    """
    if initialisations is None:
        coarse = coarse_alignment(source, target)
        centre = np.asarray(target, dtype=np.float64).mean(axis=0)
        starts = [
            _spin_about(centre, spin.rotation).compose(coarse)
            for spin in azimuthal_initialisations()
        ]
    else:
        starts = initialisations
    if len(starts) < 2:
        raise ValueError(
            "need at least two initialisations: with one start there is no rival pose, "
            "so a symmetric part would pass the ambiguity check undetected"
        )

    fits = sorted(
        (icp_sim3(source, target, initial=start) for start in starts),
        key=lambda fit: fit.rmse,
    )
    best = fits[0]

    # The best *materially different* pose. If one fits about as well, the
    # answer is not determined by the geometry, however small the residual is.
    rival = next(
        (fit for fit in fits[1:] if _angle_between(best.transform, fit.transform) > DISTINCT_ANGLE),
        None,
    )
    if rival is None or rival.rmse <= 0.0:
        distinctness = 1.0
    else:
        distinctness = float(np.clip(1.0 - best.rmse / rival.rmse, 0.0, 1.0))

    quality = float(np.clip(1.0 - best.rmse / RMSE_TOLERANCE, 0.0, 1.0))
    confidence = float(best.inlier_ratio * quality * distinctness)

    result = RegistrationResult(
        transform=best.transform,
        rmse=best.rmse,
        inlier_ratio=best.inlier_ratio,
        distinctness=distinctness,
        confidence=confidence,
        iterations=best.iterations,
        converged=best.converged,
    )

    if confidence < min_confidence:
        reason = (
            "the pose is ambiguous: a materially different alignment fits about as well"
            if distinctness < 0.5
            else "the clouds do not overlap well enough"
        )
        raise RegistrationRejected(
            f"WHAT: registration confidence {confidence:.3f} is below {min_confidence:.3f} "
            f"(fit quality {quality:.3f}, inliers {best.inlier_ratio:.3f}, "
            f"distinctness {distinctness:.3f}).\n"
            f"WHY: {reason}. Fusing on a wrong pose makes the reconstruction worse than a "
            f"single view, silently -- so this refuses instead.\n"
            f"FIX: add views that break the symmetry, seed the alignment with a known "
            f"capture pose, or accept the single-view result for this object."
        )
    return result
