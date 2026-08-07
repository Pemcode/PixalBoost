"""Poses, camera intrinsics, projection and back-projection.

Conventions are Pixal3D's, mapped in docs/pixal3d-internals.md: a Blender frame
where the camera looks down its own -Z, a 32 mm sensor, and a focal length of
16 / tan(fov / 2) millimetres. Anything arriving from another source is
converted at the `backends/` boundary, never here -- see core/ARCHITECTURE.md.

Poses are `Sim3`, not `SE3`. Pixal3D emits pixel-aligned geometry at the scale
of MoGe-2's depth estimate, which differs from view to view, so registering two
views is a 7-DoF problem. Carrying scale explicitly is also what keeps the
metric extension of Sprint 4 a localised change.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]

#: Aligns the conditioning grid with the Blender frame.
#: Mirrors vendor/pixal3d image_conditioned_proj.py:162-166 and inference.py:125.
BLENDER_GRID_ROTATION: FloatArray = np.array(
    [[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]],
    dtype=np.float64,
)

#: Pixal3D models a 32 mm sensor with a 16 mm reference focal length.
SENSOR_WIDTH_MM = 32.0
REFERENCE_FOCAL_MM = 16.0


@dataclass(frozen=True, eq=False)
class Sim3:
    """A similarity transform: ``p -> scale * rotation @ p + translation``."""

    rotation: FloatArray
    translation: FloatArray
    scale: float

    def __post_init__(self) -> None:
        rotation = np.asarray(self.rotation, dtype=np.float64)
        translation = np.asarray(self.translation, dtype=np.float64)

        if rotation.shape != (3, 3):
            raise ValueError(f"rotation must have shape (3, 3), got {rotation.shape}")
        if translation.shape != (3,):
            raise ValueError(f"translation must have shape (3,), got {translation.shape}")
        if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-9):
            raise ValueError("rotation must be orthonormal; got a matrix with R @ R.T != I")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=1e-9):
            raise ValueError(
                "rotation must be orthonormal and right-handed; det(R) must be +1, "
                f"got {float(np.linalg.det(rotation)):.6f}"
            )
        if not np.isfinite(self.scale) or self.scale <= 0.0:
            raise ValueError(f"scale must be finite and strictly positive, got {self.scale}")

        object.__setattr__(self, "rotation", rotation)
        object.__setattr__(self, "translation", translation)
        object.__setattr__(self, "scale", float(self.scale))

    @classmethod
    def identity(cls) -> Sim3:
        return cls(rotation=np.eye(3), translation=np.zeros(3), scale=1.0)

    @classmethod
    def from_matrix(cls, matrix: FloatArray) -> Sim3:
        """Decompose a 4x4 homogeneous matrix whose linear part is ``scale * rotation``."""
        matrix = np.asarray(matrix, dtype=np.float64)
        if matrix.shape != (4, 4):
            raise ValueError(f"matrix must have shape (4, 4), got {matrix.shape}")
        linear = matrix[:3, :3]
        # Every column of scale * rotation has norm `scale`; averaging the three
        # is more stable than reading a single one.
        scale = float(np.linalg.norm(linear, axis=0).mean())
        if scale <= 0.0:
            raise ValueError("matrix has a degenerate linear part; scale must be positive")
        return cls(rotation=linear / scale, translation=matrix[:3, 3], scale=scale)

    def as_matrix(self) -> FloatArray:
        matrix = np.eye(4, dtype=np.float64)
        matrix[:3, :3] = self.scale * self.rotation
        matrix[:3, 3] = self.translation
        return matrix

    def inverse(self) -> Sim3:
        inverse_rotation = self.rotation.T
        inverse_scale = 1.0 / self.scale
        return Sim3(
            rotation=inverse_rotation,
            translation=-inverse_scale * (inverse_rotation @ self.translation),
            scale=inverse_scale,
        )

    def compose(self, other: Sim3) -> Sim3:
        """Return the transform applying ``other`` first, then ``self``."""
        return Sim3(
            rotation=self.rotation @ other.rotation,
            translation=self.scale * (self.rotation @ other.translation) + self.translation,
            scale=self.scale * other.scale,
        )

    def apply(self, points: FloatArray) -> FloatArray:
        """Transform an ``(N, 3)`` array of points."""
        points = np.asarray(points, dtype=np.float64)
        if points.ndim != 2 or points.shape[1] != 3:
            raise ValueError(f"points must have shape (N, 3), got {points.shape}")
        return points @ (self.scale * self.rotation).T + self.translation


@dataclass(frozen=True)
class BlenderCamera:
    """A square pinhole camera in Pixal3D's sensor model."""

    camera_angle_x: float
    resolution: int

    def __post_init__(self) -> None:
        if not 0.0 < self.camera_angle_x < np.pi:
            raise ValueError(f"camera_angle_x must lie in (0, pi), got {self.camera_angle_x}")
        if self.resolution <= 0:
            raise ValueError(f"resolution must be positive, got {self.resolution}")

    @property
    def focal_px(self) -> float:
        focal_mm = REFERENCE_FOCAL_MM / np.tan(self.camera_angle_x / 2.0)
        return float(focal_mm * self.resolution / SENSOR_WIDTH_MM)

    @property
    def principal_point(self) -> float:
        return self.resolution / 2.0


def front_view_camera(distance: float) -> Sim3:
    """Pixal3D's canonical conditioning camera, sitting at ``-Y`` and facing the origin.

    Reproduces `front_view_transform_matrix` with its distance override
    (vendor/pixal3d image_conditioned_proj.py:172-178 and :215). This is the
    *only* camera the released checkpoint was conditioned on -- see ADR-0005.
    """
    return Sim3(
        rotation=np.array([[1.0, 0.0, 0.0], [0.0, 0.0, -1.0], [0.0, 1.0, 0.0]]),
        translation=np.array([0.0, -distance, 0.0]),
        scale=1.0,
    )


def project(
    points_world: FloatArray,
    camera_to_world: Sim3,
    camera: BlenderCamera,
) -> tuple[FloatArray, FloatArray, npt.NDArray[np.bool_]]:
    """Project world points into pixel coordinates.

    Returns ``(pixels, depth, valid)`` where `pixels` is ``(N, 2)``, `depth` is
    ``(N,)` measured along the viewing axis, and `valid` marks points that are
    both in front of the camera and inside the frame.

    Unlike the vendored implementation this divides exactly rather than adding a
    1e-8 guard to the denominator: the epsilon is a numerical crutch, not a
    convention, and it would break invertibility against `backproject`.
    """
    points_camera = camera_to_world.inverse().apply(points_world)
    depth = -points_camera[:, 2]

    # Points at or behind the camera plane have no meaningful projection; give
    # them a placeholder denominator and let `valid` reject them.
    safe_depth = np.where(depth > 0.0, depth, 1.0)
    focal = camera.focal_px
    centre = camera.principal_point

    x_pixel = focal * points_camera[:, 0] / safe_depth + centre
    y_pixel = -focal * points_camera[:, 1] / safe_depth + centre

    valid = (
        (depth > 0.0)
        & (x_pixel >= 0.0)
        & (x_pixel < camera.resolution)
        & (y_pixel >= 0.0)
        & (y_pixel < camera.resolution)
    )
    return np.stack([x_pixel, y_pixel], axis=-1), depth, valid


def backproject(
    pixels: FloatArray,
    depth: FloatArray,
    camera_to_world: Sim3,
    camera: BlenderCamera,
) -> FloatArray:
    """Lift pixel coordinates at a known depth back into world space."""
    pixels = np.asarray(pixels, dtype=np.float64)
    depth = np.asarray(depth, dtype=np.float64)
    if pixels.ndim != 2 or pixels.shape[1] != 2:
        raise ValueError(f"pixels must have shape (N, 2), got {pixels.shape}")
    if depth.shape != (pixels.shape[0],):
        raise ValueError(f"depth must have shape ({pixels.shape[0]},), got {depth.shape}")

    focal = camera.focal_px
    centre = camera.principal_point
    x_camera = (pixels[:, 0] - centre) * depth / focal
    y_camera = -(pixels[:, 1] - centre) * depth / focal
    points_camera = np.stack([x_camera, y_camera, -depth], axis=-1)
    return camera_to_world.apply(points_camera)


def canonical_grid_points(resolution: int, mesh_scale: float = 1.0) -> FloatArray:
    """The conditioning grid Pixal3D back-projects image features onto.

    ``resolution**3`` points spanning ``[-1, 1]``, rotated into the Blender
    frame, then divided by ``mesh_scale`` and by two -- which lands them in the
    ``[-0.5, 0.5]`` axis-aligned box the GLB is finally exported in
    (vendor/pixal3d image_conditioned_proj.py:156-169, :210; inference.py:266).

    The ordering is ``indexing="ij"`` over ``(x, y, z)``. It is what maps a flat
    conditioning vector back onto voxel coordinates, so it must not drift.
    """
    if resolution <= 0:
        raise ValueError(f"resolution must be positive, got {resolution}")
    if mesh_scale <= 0.0:
        raise ValueError(f"mesh_scale must be positive, got {mesh_scale}")

    axis = np.linspace(-1.0, 1.0, resolution, dtype=np.float64)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    grid = np.stack((x, y, z), axis=-1).reshape(-1, 3) @ BLENDER_GRID_ROTATION.T
    return grid / mesh_scale / 2.0
