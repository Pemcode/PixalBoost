"""Procedural CAD-like reference parts.

Procedural rather than a downloaded dataset so the benchmark builds offline, in
CI, and reproduces bit for bit. Real CAD files can be added later behind the
same `(vertices, faces)` interface.

Every part must be **watertight**. A mesh with internal faces -- what you get by
naively concatenating two overlapping boxes -- puts ground-truth surface samples
*inside* the solid and silently corrupts every Chamfer and F-score downstream.
Both builders here are closed by construction: extrusion caps its profile, and
revolution closes on itself.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import numpy.typing as npt

FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
Mesh = tuple[FloatArray, IntArray]

#: Pixal3D exports into this axis-aligned box (inference.py:266); references match it.
EXPORT_HALF_EXTENT = 0.5


def _signed_area(polygon: FloatArray) -> float:
    x, y = polygon[:, 0], polygon[:, 1]
    return float(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0


def _point_in_triangle(point: FloatArray, a: FloatArray, b: FloatArray, c: FloatArray) -> bool:
    def side(p: FloatArray, q: FloatArray, r: FloatArray) -> float:
        return float((q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0]))

    d1, d2, d3 = side(point, a, b), side(point, b, c), side(point, c, a)
    has_negative = min(d1, d2, d3) < 0.0
    has_positive = max(d1, d2, d3) > 0.0
    return not (has_negative and has_positive)


def triangulate_polygon(polygon: FloatArray) -> IntArray:
    """Ear-clipping triangulation, returning indices into `polygon`.

    Ear clipping rather than a triangle fan because the profiles are concave: an
    L-shaped bracket fanned from one vertex would emit triangles lying outside
    the profile.
    """
    polygon = np.asarray(polygon, dtype=np.float64)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError(f"polygon must have shape (N, 2), got {polygon.shape}")
    count = polygon.shape[0]
    if count < 3:
        raise ValueError(f"polygon needs at least 3 vertices, got {count}")

    remaining = list(range(count))
    if _signed_area(polygon) < 0.0:
        remaining.reverse()

    triangles: list[tuple[int, int, int]] = []
    guard = 0
    while len(remaining) > 3:
        guard += 1
        if guard > count * count:
            raise ValueError("polygon could not be triangulated; is it self-intersecting?")
        for position in range(len(remaining)):
            previous = remaining[position - 1]
            current = remaining[position]
            following = remaining[(position + 1) % len(remaining)]
            a, b, c = polygon[previous], polygon[current], polygon[following]
            if (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]) <= 0.0:
                continue  # reflex vertex, not an ear
            others = [i for i in remaining if i not in (previous, current, following)]
            if any(_point_in_triangle(polygon[i], a, b, c) for i in others):
                continue  # another vertex sits inside the candidate ear
            triangles.append((previous, current, following))
            remaining.pop(position)
            break
        else:
            raise ValueError("polygon could not be triangulated; is it self-intersecting?")

    triangles.append((remaining[0], remaining[1], remaining[2]))
    return np.array(triangles, dtype=np.int64)


def extrude_polygon(profile: FloatArray, thickness: float) -> Mesh:
    """Extrude a closed 2D profile along Z into a watertight prism."""
    profile = np.asarray(profile, dtype=np.float64)
    if thickness <= 0.0:
        raise ValueError(f"thickness must be positive, got {thickness}")
    cap = triangulate_polygon(profile)
    count = profile.shape[0]

    half = thickness / 2.0
    bottom = np.column_stack([profile, np.full(count, -half)])
    top = np.column_stack([profile, np.full(count, half)])
    vertices = np.vstack([bottom, top])

    faces: list[tuple[int, int, int]] = []
    for a, b, c in cap:
        faces.append((int(a), int(c), int(b)))  # bottom cap, normal down
        faces.append((count + int(a), count + int(b), count + int(c)))  # top cap, normal up

    order = list(range(count))
    if _signed_area(profile) < 0.0:
        order.reverse()
    for position, current in enumerate(order):
        following = order[(position + 1) % count]
        faces.append((current, following, count + following))
        faces.append((current, count + following, count + current))

    return vertices, np.array(faces, dtype=np.int64)


def revolve_profile(profile: FloatArray, segments: int) -> Mesh:
    """Revolve a closed ``(radius, height)`` profile around the Z axis.

    Offsetting the whole profile from the axis is what produces a through hole,
    so a washer or a flange needs no CSG. Profile vertices sitting exactly on
    the axis collapse to a single pole vertex, which is what keeps the caps
    manifold instead of emitting a ring of degenerate slivers.
    """
    profile = np.asarray(profile, dtype=np.float64)
    if profile.ndim != 2 or profile.shape[1] != 2:
        raise ValueError(f"profile must have shape (N, 2), got {profile.shape}")
    if profile.shape[0] < 3:
        raise ValueError(f"profile needs at least 3 vertices, got {profile.shape[0]}")
    if segments < 3:
        raise ValueError(f"segments must be at least 3, got {segments}")
    if profile[:, 0].min() < 0.0:
        raise ValueError("profile radii must be non-negative")

    if _signed_area(profile) < 0.0:
        profile = profile[::-1].copy()

    angles = np.linspace(0.0, 2.0 * np.pi, segments, endpoint=False)
    vertices: list[tuple[float, float, float]] = []
    index_of: list[list[int]] = []
    for radius, height in profile:
        if radius == 0.0:
            index_of.append([len(vertices)] * segments)
            vertices.append((0.0, 0.0, float(height)))
        else:
            index_of.append([len(vertices) + s for s in range(segments)])
            vertices.extend(
                (float(radius * np.cos(a)), float(radius * np.sin(a)), float(height))
                for a in angles
            )

    faces: list[tuple[int, int, int]] = []
    for i in range(profile.shape[0]):
        j = (i + 1) % profile.shape[0]
        on_axis_i = profile[i, 0] == 0.0
        on_axis_j = profile[j, 0] == 0.0
        if on_axis_i and on_axis_j:
            continue  # the segment lies on the axis and sweeps nothing
        for s in range(segments):
            t = (s + 1) % segments
            # Wound so the normal is (d/dtheta) x (d/dprofile), i.e. outward.
            if on_axis_i:
                faces.append((index_of[i][s], index_of[j][t], index_of[j][s]))
            elif on_axis_j:
                faces.append((index_of[i][s], index_of[i][t], index_of[j][s]))
            else:
                faces.append((index_of[i][s], index_of[i][t], index_of[j][t]))
                faces.append((index_of[i][s], index_of[j][t], index_of[j][s]))

    return np.array(vertices, dtype=np.float64), np.array(faces, dtype=np.int64)


def fit_to_export_box(vertices: FloatArray) -> FloatArray:
    """Centre a part and scale it so its largest half-extent is exactly 0.5."""
    centred = vertices - (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    scaled: FloatArray = centred * (EXPORT_HALF_EXTENT / np.abs(centred).max())
    return scaled


def l_bracket() -> Mesh:
    """A prismatic part: concave profile, sharp edges, flat faces."""
    profile = np.array(
        [[0.0, 0.0], [3.0, 0.0], [3.0, 0.8], [0.8, 0.8], [0.8, 2.6], [0.0, 2.6]],
        dtype=np.float64,
    )
    vertices, faces = extrude_polygon(profile, thickness=1.4)
    return fit_to_export_box(vertices), faces


def stepped_shaft() -> Mesh:
    """A turned part: axisymmetric, three diameters, no hole."""
    profile = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [1.0, 0.9],
            [0.66, 0.9],
            [0.66, 2.1],
            [0.4, 2.1],
            [0.4, 3.0],
            [0.0, 3.0],
        ],
        dtype=np.float64,
    )
    vertices, faces = revolve_profile(profile, segments=64)
    return fit_to_export_box(vertices), faces


def flange_ring() -> Mesh:
    """A washer: genus 1, so the silhouette carries a hole from most angles."""
    profile = np.array([[0.45, 0.0], [1.2, 0.0], [1.2, 0.34], [0.45, 0.34]], dtype=np.float64)
    vertices, faces = revolve_profile(profile, segments=64)
    return fit_to_export_box(vertices), faces


#: Three distinct topologies: prismatic-concave, axisymmetric-solid, and holed.
CATALOGUE: dict[str, Callable[[], Mesh]] = {
    "l_bracket": l_bracket,
    "stepped_shaft": stepped_shaft,
    "flange_ring": flange_ring,
}
