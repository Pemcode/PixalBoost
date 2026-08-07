"""The capture rig: 6 azimuths x 3 elevations, matching the real photo protocol.

The synthetic and real protocols are deliberately identical. If they differed, a
synthetic-to-real comparison would confound the domain gap with a difference in
capture geometry, and stop measuring anything interpretable.

Azimuth 0 / elevation 0 reproduces Pixal3D's canonical conditioning camera
exactly (docs/pixal3d-internals.md), so poses handed to the model are not
silently offset from the frame it was trained in.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from pixaboost.core.geometry import FloatArray, Sim3

AZIMUTH_COUNT = 6
ELEVATIONS_DEG: tuple[float, ...] = (45.0, 0.0, -45.0)

_WORLD_UP = np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class ViewPose:
    """One capture station: where the camera sits and how it is named."""

    name: str
    azimuth_deg: float
    elevation_deg: float
    camera_to_world: Sim3


def _look_at_origin(position: FloatArray) -> Sim3:
    """Build a Blender-convention pose: the camera's own -Z points at the origin."""
    z_axis = position / np.linalg.norm(position)
    x_axis = np.cross(_WORLD_UP, z_axis)
    x_axis = x_axis / np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    return Sim3(
        rotation=np.column_stack([x_axis, y_axis, z_axis]),
        translation=position,
        scale=1.0,
    )


def capture_rig(distance: float) -> list[ViewPose]:
    """The 18 stations, ordered elevation-major then azimuth."""
    if not np.isfinite(distance) or distance <= 0.0:
        raise ValueError(f"distance must be finite and strictly positive, got {distance}")

    views: list[ViewPose] = []
    for elevation_deg in ELEVATIONS_DEG:
        elevation = np.radians(elevation_deg)
        for index in range(AZIMUTH_COUNT):
            azimuth_deg = 360.0 * index / AZIMUTH_COUNT
            azimuth = np.radians(azimuth_deg)
            position = distance * np.array(
                [
                    np.sin(azimuth) * np.cos(elevation),
                    -np.cos(azimuth) * np.cos(elevation),
                    np.sin(elevation),
                ]
            )
            views.append(
                ViewPose(
                    name=f"az{round(azimuth_deg):03d}_el{round(elevation_deg):+03d}",
                    azimuth_deg=azimuth_deg,
                    elevation_deg=elevation_deg,
                    camera_to_world=_look_at_origin(position),
                )
            )
    return views
