"""TDD for bench.shapes (F05).

Procedural CAD-like parts rather than a downloaded dataset: the benchmark has to
be buildable offline, in CI, and be bit-for-bit reproducible. Real CAD files can
be added later behind the same interface.

Watertightness is the property that matters. A mesh with internal faces would
put ground-truth surface samples *inside* the solid and quietly corrupt every
Chamfer and F-score number downstream.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixaboost.bench.shapes import (
    CATALOGUE,
    extrude_polygon,
    flange_ring,
    l_bracket,
    revolve_profile,
    stepped_shaft,
)


def edge_counts(faces: np.ndarray) -> dict[tuple[int, int], int]:
    counts: dict[tuple[int, int], int] = {}
    for face in faces:
        for i in range(3):
            edge = (int(face[i]), int(face[(i + 1) % 3]))
            key = (min(edge), max(edge))
            counts[key] = counts.get(key, 0) + 1
    return counts


def signed_volume(vertices: np.ndarray, faces: np.ndarray) -> float:
    a, b, c = vertices[faces[:, 0]], vertices[faces[:, 1]], vertices[faces[:, 2]]
    return float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)


def assert_watertight(vertices: np.ndarray, faces: np.ndarray) -> None:
    assert faces.min() >= 0 and faces.max() < len(vertices)
    shared = edge_counts(faces)
    bad = {edge: n for edge, n in shared.items() if n != 2}
    assert not bad, f"{len(bad)} edges are not shared by exactly two faces, e.g. {list(bad)[:3]}"


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------


def test_extruded_square_is_a_watertight_box_of_the_expected_volume() -> None:
    square = np.array([[0.0, 0.0], [2.0, 0.0], [2.0, 3.0], [0.0, 3.0]])
    vertices, faces = extrude_polygon(square, thickness=5.0)
    assert_watertight(vertices, faces)
    assert abs(signed_volume(vertices, faces)) == pytest.approx(2.0 * 3.0 * 5.0)


def test_extrusion_triangulates_a_concave_profile_correctly() -> None:
    """An L is concave: a naive fan triangulation would spill outside the profile."""
    profile = np.array([[0.0, 0.0], [3.0, 0.0], [3.0, 1.0], [1.0, 1.0], [1.0, 3.0], [0.0, 3.0]])
    vertices, faces = extrude_polygon(profile, thickness=2.0)
    assert_watertight(vertices, faces)
    assert abs(signed_volume(vertices, faces)) == pytest.approx(5.0 * 2.0)


def test_revolved_rectangle_is_a_watertight_cylinder() -> None:
    profile = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 4.0], [0.0, 4.0]])
    vertices, faces = revolve_profile(profile, segments=128)
    assert_watertight(vertices, faces)
    # A 128-gon slightly under-fills the circle it approximates.
    assert abs(signed_volume(vertices, faces)) == pytest.approx(np.pi * 1.0**2 * 4.0, rel=2e-3)


def test_revolving_an_offset_rectangle_produces_a_through_hole() -> None:
    """Offsetting the profile from the axis is what creates the bore -- no CSG needed."""
    profile = np.array([[1.0, 0.0], [2.0, 0.0], [2.0, 1.0], [1.0, 1.0]])
    vertices, faces = revolve_profile(profile, segments=128)
    assert_watertight(vertices, faces)
    expected = np.pi * (2.0**2 - 1.0**2) * 1.0
    assert abs(signed_volume(vertices, faces)) == pytest.approx(expected, rel=2e-3)
    assert np.linalg.norm(vertices[:, :2], axis=1).min() == pytest.approx(1.0, rel=1e-6)


def test_builders_reject_a_degenerate_profile() -> None:
    with pytest.raises(ValueError, match="at least 3"):
        extrude_polygon(np.array([[0.0, 0.0], [1.0, 0.0]]), thickness=1.0)
    with pytest.raises(ValueError, match="segments"):
        revolve_profile(np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]]), segments=2)


# ---------------------------------------------------------------------------
# The catalogue
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("name", sorted(CATALOGUE))
def test_catalogue_parts_are_watertight_and_centred(name: str) -> None:
    vertices, faces = CATALOGUE[name]()
    assert_watertight(vertices, faces)
    assert abs(signed_volume(vertices, faces)) > 0.0
    centre = (vertices.max(axis=0) + vertices.min(axis=0)) / 2.0
    np.testing.assert_allclose(centre, 0.0, atol=1e-9)


@pytest.mark.parametrize("name", sorted(CATALOGUE))
def test_catalogue_parts_fit_the_pixal3d_export_box(name: str) -> None:
    """Pixal3D exports into a [-0.5, 0.5] AABB; the reference must live there too."""
    vertices, _ = CATALOGUE[name]()
    assert np.abs(vertices).max() == pytest.approx(0.5, rel=1e-9)


@pytest.mark.parametrize("name", sorted(CATALOGUE))
def test_catalogue_parts_are_reproducible(name: str) -> None:
    first_v, first_f = CATALOGUE[name]()
    second_v, second_f = CATALOGUE[name]()
    np.testing.assert_array_equal(first_v, second_v)
    np.testing.assert_array_equal(first_f, second_f)


def test_catalogue_covers_distinct_topologies() -> None:
    assert set(CATALOGUE) == {"l_bracket", "stepped_shaft", "flange_ring"}
    assert CATALOGUE["l_bracket"] is l_bracket
    assert CATALOGUE["stepped_shaft"] is stepped_shaft
    assert CATALOGUE["flange_ring"] is flange_ring
