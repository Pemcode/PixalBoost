"""Analytic contract for the mask helpers that feed SAM a point prompt (F14).

Every fixture here is closed-form: a disc, an annulus, a rectangle. No model
output, no binary file, no download -- see docs/testing.md, deterministic
regime.

The annulus is the fixture that matters. The test piece is a wheel with a
central bore, and the obvious way to derive a prompt from a coarse mask -- its
centroid -- lands *in the hole*, which is background. A prompt in the
background makes SAM segment the workbench. These tests pin the failure and
the fix.
"""

from __future__ import annotations

import numpy as np
import pytest

from pixaboost.core.segmentation import (
    compose_rgba,
    deepest_interior_point,
    largest_connected_component,
    mask_centroid,
    mask_from_rgba,
)


def disc(size: int, centre: tuple[int, int], radius: float) -> np.ndarray:
    rows, cols = np.ogrid[:size, :size]
    return ((rows - centre[0]) ** 2 + (cols - centre[1]) ** 2) <= radius**2


def annulus(size: int, centre: tuple[int, int], inner: float, outer: float) -> np.ndarray:
    return disc(size, centre, outer) & ~disc(size, centre, inner)


# --------------------------------------------------------------------------
# deepest_interior_point
# --------------------------------------------------------------------------


def test_the_deepest_point_of_a_disc_is_its_centre() -> None:
    mask = disc(129, (64, 64), 40.0)
    row, col = deepest_interior_point(mask)
    assert abs(row - 64) <= 1 and abs(col - 64) <= 1


def test_the_deepest_point_of_a_rectangle_lies_on_its_medial_segment() -> None:
    """A 60x100 rectangle has no single deepest pixel: the argmax is a segment.

    The inscribed circle has radius min(60, 100) / 2 = 30, and every point on
    the horizontal mid-line far enough from the short edges achieves it. The
    contract is therefore "on the medial segment, 30 px clear of every edge",
    not "at the centroid" -- asserting the latter would pin an arbitrary
    tie-break rather than the geometry.
    """
    mask = np.zeros((100, 200), dtype=bool)
    mask[20:80, 50:150] = True  # rows 20..79, cols 50..149

    row, col = deepest_interior_point(mask)

    assert mask[row, col]
    half_height = 60 / 2
    assert abs(row - 49.5) <= 1, "must sit on the horizontal mid-line"
    assert min(row - 20, 79 - row) >= half_height - 1.5
    assert min(col - 50, 149 - col) >= half_height - 1.5


def test_the_centroid_of_an_annulus_falls_in_the_hole_which_is_why_it_cannot_be_the_prompt() -> (
    None
):
    """The measured trap: the test piece is a wheel, and its centroid is background."""
    mask = annulus(201, (100, 100), inner=30.0, outer=90.0)

    centroid = mask_centroid(mask)
    assert abs(centroid[0] - 100) <= 1 and abs(centroid[1] - 100) <= 1
    assert not mask[centroid], "the annulus centroid must land in the bore, i.e. outside the mask"


def test_the_deepest_point_of_an_annulus_sits_on_the_mid_radius_ring() -> None:
    """Closed form: the widest inscribed circle is centred at (inner+outer)/2."""
    inner, outer = 30.0, 90.0
    mask = annulus(201, (100, 100), inner=inner, outer=outer)

    row, col = deepest_interior_point(mask)

    assert mask[row, col], "the prompt must be inside the mask"
    radius = float(np.hypot(row - 100, col - 100))
    assert abs(radius - (inner + outer) / 2) <= 1.5


def test_the_deepest_point_lands_in_the_thickest_blob_not_the_first_one() -> None:
    mask = disc(200, (50, 50), 12.0) | disc(200, (140, 140), 35.0)
    row, col = deepest_interior_point(mask)
    assert np.hypot(row - 140, col - 140) < 35.0


@pytest.mark.parametrize(
    "mask",
    [
        disc(101, (50, 50), 30.0),
        annulus(151, (75, 75), 20.0, 60.0),
        annulus(151, (75, 75), 55.0, 60.0),  # thin ring
    ],
)
def test_the_prompt_is_always_inside_the_mask(mask: np.ndarray) -> None:
    assert mask[deepest_interior_point(mask)]


def test_an_empty_mask_has_no_prompt_and_says_so() -> None:
    with pytest.raises(ValueError, match="empty"):
        deepest_interior_point(np.zeros((32, 32), dtype=bool))


def test_a_non_two_dimensional_mask_is_rejected() -> None:
    with pytest.raises(ValueError, match="2-D"):
        deepest_interior_point(np.ones((4, 4, 3), dtype=bool))


def test_the_prompt_is_deterministic() -> None:
    mask = annulus(201, (100, 100), 30.0, 90.0)
    assert deepest_interior_point(mask) == deepest_interior_point(mask)


# --------------------------------------------------------------------------
# largest_connected_component
# --------------------------------------------------------------------------


def test_specks_are_dropped_and_the_main_blob_survives() -> None:
    mask = disc(200, (100, 100), 40.0)
    mask[5:8, 5:8] = True  # a BiRefNet speck on the workbench

    kept = largest_connected_component(mask)

    assert not kept[5:8, 5:8].any()
    assert kept[100, 100]
    assert kept.sum() == disc(200, (100, 100), 40.0).sum()


def test_the_hole_of_an_annulus_is_not_filled_in() -> None:
    """A wheel keeps its bore: filling it would hand SAM a prompt in the background."""
    mask = annulus(201, (100, 100), 30.0, 90.0)
    kept = largest_connected_component(mask)
    assert not kept[100, 100]
    assert kept.sum() == mask.sum()


def test_an_empty_mask_stays_empty() -> None:
    assert not largest_connected_component(np.zeros((16, 16), dtype=bool)).any()


# --------------------------------------------------------------------------
# compose_rgba
# --------------------------------------------------------------------------


def test_alpha_is_binary_and_matches_the_mask() -> None:
    rgb = np.full((10, 12, 3), 200, dtype=np.uint8)
    mask = np.zeros((10, 12), dtype=bool)
    mask[2:6, 3:9] = True

    rgba = compose_rgba(rgb, mask)

    assert rgba.shape == (10, 12, 4)
    assert rgba.dtype == np.uint8
    assert np.array_equal(rgba[..., 3] == 255, mask)
    assert np.array_equal(rgba[..., 3] == 0, ~mask)


def test_the_colour_channels_are_left_untouched_inside_the_mask() -> None:
    rng = np.random.default_rng(0)
    rgb = rng.integers(0, 256, size=(8, 8, 3), dtype=np.uint8)
    mask = np.zeros((8, 8), dtype=bool)
    mask[1:5, 1:5] = True

    rgba = compose_rgba(rgb, mask)

    assert np.array_equal(rgba[..., :3][mask], rgb[mask])


def test_the_background_colour_is_zeroed_so_no_workbench_leaks_through() -> None:
    rgb = np.full((6, 6, 3), 123, dtype=np.uint8)
    mask = np.zeros((6, 6), dtype=bool)
    mask[2:4, 2:4] = True

    rgba = compose_rgba(rgb, mask)

    assert (rgba[..., :3][~mask] == 0).all()


def test_a_fully_opaque_result_is_refused_because_pixal3d_would_ignore_it() -> None:
    """preprocess_image only honours alpha when `not np.all(alpha == 255)`."""
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="fully opaque"):
        compose_rgba(rgb, np.ones((4, 4), dtype=bool))


def test_an_empty_mask_is_refused() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="empty"):
        compose_rgba(rgb, np.zeros((4, 4), dtype=bool))


def test_mismatched_shapes_are_refused() -> None:
    rgb = np.zeros((4, 4, 3), dtype=np.uint8)
    with pytest.raises(ValueError, match="shape"):
        compose_rgba(rgb, np.ones((5, 5), dtype=bool))


# --------------------------------------------------------------------------
# mask_from_rgba -- the alpha channel read back as the object mask (F15)
# --------------------------------------------------------------------------


def cutout(size: int, mask: np.ndarray) -> np.ndarray:
    return compose_rgba(np.full((size, size, 3), 200, dtype=np.uint8), mask)


def test_the_alpha_channel_comes_back_as_the_mask_it_was_written_from() -> None:
    """Round trip: what SAM approved is what the pose search will see."""
    mask = annulus(40, (20, 20), 6.0, 16.0)

    assert np.array_equal(mask_from_rgba(cutout(40, mask)), mask)


def test_the_threshold_is_the_one_pixal3d_itself_applies() -> None:
    """`preprocess_image` crops to `alpha > 0.8 * 255`.

    Reading a softer edge than Pixal3D does would search for a pose against a
    silhouette Pixal3D never reconstructed. Both sides must cut at the same
    place, so the boundary is pinned rather than left to taste.
    """
    rgba = np.zeros((2, 2, 4), dtype=np.uint8)
    rgba[..., 3] = np.array([[203, 205], [0, 255]], dtype=np.uint8)  # 0.8 * 255 = 204.0

    assert np.array_equal(
        mask_from_rgba(rgba), np.array([[False, True], [False, True]], dtype=bool)
    )


def test_an_image_without_an_alpha_channel_is_refused() -> None:
    """A JPEG carries no mask, and guessing one here would hide that."""
    with pytest.raises(ValueError, match="alpha"):
        mask_from_rgba(np.zeros((4, 4, 3), dtype=np.uint8))


def test_a_fully_opaque_image_is_refused_because_it_carries_no_mask() -> None:
    """Same trap as `compose_rgba`: Pixal3D would rerun its own rembg."""
    rgba = np.full((4, 4, 4), 255, dtype=np.uint8)
    with pytest.raises(ValueError, match="fully opaque"):
        mask_from_rgba(rgba)


def test_a_fully_transparent_image_is_refused() -> None:
    rgba = np.zeros((4, 4, 4), dtype=np.uint8)
    with pytest.raises(ValueError, match="empty"):
        mask_from_rgba(rgba)
