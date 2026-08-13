"""Render-and-compare pose search: the object is its own calibration target (F15).

Pixal3D is pixel-aligned and reports no extrinsics, so N photographs give N
reconstructions in N unrelated frames. Recovering the relation between them
normally means calibrating -- a fiducial in the scene, or an indexed turntable.

This module does it without either. Given a reconstruction and the silhouette
of another photograph, it searches for the object rotation whose render best
matches that silhouette. The object supplies the reference frame that the
capture never recorded.

Two properties make it work where feature matching would not:

**Silhouettes survive metal.** A matte cast part under workshop lighting has
almost no repeatable photometric detail, and its specular highlights move with
the camera. Its outline does not.

**Symmetry stops being a problem.** ADR-0007 refuses an ambiguous pose because
a wrong one silently smears a fusion. Here the pose is only ever used to
render, and a spin about a true symmetry axis leaves the render unchanged, so a
tie is reported and costs nothing. That inversion is pinned by
`test_an_axisymmetric_part_is_reported_ambiguous_yet_still_renders_correctly`.

Everything is pure CPU numpy: the rasteriser is `core/render.py` (F03).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from pixaboost.core.geometry import BlenderCamera, front_view_camera
from pixaboost.core.metrics import silhouette_iou
from pixaboost.core.render import rasterise_silhouette

FloatArray = np.ndarray
IntArray = np.ndarray
BoolArray = np.ndarray

#: Two rotations closer than this are the same answer, not rival ones. Matches
#: the 10 deg used by `core/registration.py` for the same purpose (ADR-0007).
MATERIALLY_DIFFERENT_RAD = np.deg2rad(10.0)
#: A runner-up within this fraction of the winner makes the answer a tie.
AMBIGUITY_MARGIN = 0.02


#: Margin Pixal3D leaves around the silhouette before cropping
#: (`pixal3d_image_to_3d.py`, `size = int(size * 1.1)`).
CANONICAL_MARGIN = 1.1


def crop_to_canonical_framing(mask: BoolArray, resolution: int) -> BoolArray:
    """Put a silhouette through the framing `preprocess_image` applies.

    Pixal3D recentres every view on its own alpha bounding box and crops a
    square of `max(extent) * 1.1` before resizing. A mask compared against a
    canonical render *without* that step is compared at a different scale and
    centring, and the pose search converges on a confidently wrong answer --
    silently, because the silhouettes still overlap a lot.

    One deliberate deviation: the extent is measured as a pixel *count*
    (`max - min + 1`) rather than as the index difference upstream uses. That
    makes the framing exactly scale-invariant, which the index version is not,
    at a cost of one pixel on a 4000 px photograph.
    """
    binary = np.asarray(mask, dtype=bool)
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {binary.shape}")
    if not binary.any():
        raise ValueError("cannot frame an empty mask")
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")

    rows, cols = np.nonzero(binary)
    top, bottom = int(rows.min()), int(rows.max())
    left, right = int(cols.min()), int(cols.max())
    side = int(max(bottom - top + 1, right - left + 1) * CANONICAL_MARGIN)
    half = max(1, side // 2)
    centre_row = (top + bottom) // 2
    centre_col = (left + right) // 2

    # Pad rather than shift: a part touching the border must stay centred, or
    # the recovered pose absorbs the offset as a spurious rotation.
    padded = np.pad(binary, half + 1, mode="constant", constant_values=False)
    origin_row = centre_row + half + 1 - half
    origin_col = centre_col + half + 1 - half
    canvas = padded[origin_row : origin_row + 2 * half, origin_col : origin_col + 2 * half]

    return _resize_nearest_bool(canvas, resolution)


def _resize_nearest_bool(mask: BoolArray, resolution: int) -> BoolArray:
    if mask.shape == (resolution, resolution):
        return mask
    rows = np.clip((np.arange(resolution) * mask.shape[0]) // resolution, 0, mask.shape[0] - 1)
    cols = np.clip((np.arange(resolution) * mask.shape[1]) // resolution, 0, mask.shape[1] - 1)
    resized: BoolArray = mask[np.ix_(rows, cols)]
    return resized


def object_rotation(azimuth: float, elevation: float, roll: float) -> FloatArray:
    """Rotation applied to the *object*, the camera staying canonical.

    `front_view_camera` sits at -Y looking towards +Y with +Z up, so azimuth
    turns about Z (the up axis), elevation about X, and roll about Y (the
    viewing axis). Composed as `Ry(roll) @ Rx(elevation) @ Rz(azimuth)`.
    """
    ca, sa = np.cos(azimuth), np.sin(azimuth)
    ce, se = np.cos(elevation), np.sin(elevation)
    cr, sr = np.cos(roll), np.sin(roll)
    about_z = np.array([[ca, -sa, 0.0], [sa, ca, 0.0], [0.0, 0.0, 1.0]])
    about_x = np.array([[1.0, 0.0, 0.0], [0.0, ce, -se], [0.0, se, ce]])
    about_y = np.array([[cr, 0.0, sr], [0.0, 1.0, 0.0], [-sr, 0.0, cr]])
    rotation: FloatArray = about_y @ about_x @ about_z
    return rotation


def rotation_angle_between(a: FloatArray, b: FloatArray) -> float:
    """Geodesic angle in radians between two rotations."""
    cosine = (float(np.trace(np.asarray(a).T @ np.asarray(b))) - 1.0) / 2.0
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


@dataclass(frozen=True)
class PoseCandidate:
    """One evaluated orientation and how well it reproduced the target."""

    rotation: FloatArray
    iou: float


@dataclass(frozen=True)
class PoseSearchResult:
    """The winning orientation, with the rivals kept for inspection."""

    rotation: FloatArray
    iou: float
    #: Best first, and pruned so that no two are within `MATERIALLY_DIFFERENT_RAD`.
    candidates: tuple[PoseCandidate, ...] = field(default=())

    @property
    def is_ambiguous(self) -> bool:
        """True when a materially different orientation scores just as well.

        Reported, never acted on. On a revolved part this is the correct
        answer, not a failure: every azimuth about the symmetry axis renders
        identically, so any of them may be used.
        """
        if len(self.candidates) < 2:
            return False
        best, rival = self.candidates[0].iou, self.candidates[1].iou
        return best > 0.0 and (best - rival) / best < AMBIGUITY_MARGIN


def _validate(target: BoolArray) -> BoolArray:
    mask = np.asarray(target, dtype=bool)
    if mask.ndim != 2:
        raise ValueError(f"target mask must be 2-D, got shape {mask.shape}")
    if not mask.any():
        raise ValueError("cannot search a pose against an empty target silhouette")
    return mask


def _grid(azimuth_steps: int, elevation_steps: int, roll_steps: int) -> list[FloatArray]:
    """Orientations to try, spaced to cover the sphere without duplicating poles."""
    if azimuth_steps < 1 or elevation_steps < 1 or roll_steps < 1:
        raise ValueError("grid steps must all be positive")
    azimuths = np.linspace(0.0, 2.0 * np.pi, azimuth_steps, endpoint=False)
    elevations = (
        np.array([0.0])
        if elevation_steps == 1
        else np.linspace(-np.pi / 3.0, np.pi / 3.0, elevation_steps)
    )
    rolls = (
        np.array([0.0])
        if roll_steps == 1
        else np.linspace(0.0, 2.0 * np.pi, roll_steps, endpoint=False)
    )
    return [
        np.array([azimuth, elevation, roll])
        for elevation in elevations
        for azimuth in azimuths
        for roll in rolls
    ]


def search_object_pose(
    vertices: FloatArray,
    faces: IntArray,
    target: BoolArray,
    *,
    camera: BlenderCamera,
    distance: float = 2.0,
    azimuth_steps: int = 24,
    elevation_steps: int = 7,
    roll_steps: int = 1,
    refine_rounds: int = 2,
) -> PoseSearchResult:
    """Find the object rotation whose silhouette best matches `target`.

    A coarse sweep first, then `refine_rounds` local passes that halve the
    spacing around the leader. Deterministic: no random restarts, and ties are
    broken by the order the grid is generated in.
    """
    mask = crop_to_canonical_framing(_validate(target), camera.resolution)
    camera_to_world = front_view_camera(distance)
    points = np.asarray(vertices, dtype=np.float64)

    def score(angles: FloatArray) -> PoseCandidate:
        rotation = object_rotation(*angles)
        rendered = rasterise_silhouette(points @ rotation.T, faces, camera_to_world, camera)
        # Both sides go through the same framing, so the search is left with
        # rotation alone: scale and centring are absorbed rather than fitted.
        # A photograph's mask and a render never share a framing otherwise.
        framed = (
            crop_to_canonical_framing(rendered, camera.resolution)
            if rendered.any()
            else rendered
        )
        return PoseCandidate(rotation=rotation, iou=silhouette_iou(framed, mask))

    evaluated: list[tuple[FloatArray, PoseCandidate]] = [
        (angles, score(angles)) for angles in _grid(azimuth_steps, elevation_steps, roll_steps)
    ]
    best_angles = max(evaluated, key=lambda item: item[1].iou)[0]

    spacing = np.array(
        [
            2.0 * np.pi / max(1, azimuth_steps),
            (2.0 * np.pi / 3.0) / max(1, elevation_steps - 1) if elevation_steps > 1 else 0.0,
            2.0 * np.pi / max(1, roll_steps),
        ]
    )
    for _ in range(max(0, refine_rounds)):
        spacing = spacing / 2.0
        neighbours = [
            best_angles + spacing * np.array(offset)
            for offset in ((-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1))
        ]
        evaluated.extend((angles, score(angles)) for angles in neighbours)
        best_angles = max(evaluated, key=lambda item: item[1].iou)[0]

    ordered = sorted((candidate for _, candidate in evaluated), key=lambda c: -c.iou)
    return PoseSearchResult(
        rotation=ordered[0].rotation,
        iou=ordered[0].iou,
        candidates=_prune_near_duplicates(ordered),
    )


def _prune_near_duplicates(ordered: list[PoseCandidate]) -> tuple[PoseCandidate, ...]:
    """Keep only orientations that are rivals rather than neighbours.

    Without this, the runner-up is always the grid cell next door, the margin
    is always tiny, and everything looks ambiguous -- which would make the flag
    useless exactly where it matters.
    """
    kept: list[PoseCandidate] = []
    for candidate in ordered:
        if all(
            rotation_angle_between(candidate.rotation, other.rotation) >= MATERIALLY_DIFFERENT_RAD
            for other in kept
        ):
            kept.append(candidate)
        if len(kept) == 8:
            break
    return tuple(kept)
