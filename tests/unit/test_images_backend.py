"""Reading a cutout back off disk (F15).

Thin by construction: the only judgement here belongs to
`core.segmentation.mask_from_rgba`, and these tests exist to prove that the
file layer does not quietly add any of its own -- in particular that it does
not invent a mask for an image that has none.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixaboost.backends.images import read_object_mask, read_rgba
from pixaboost.core.segmentation import compose_rgba


def annulus(size: int, inner: float, outer: float) -> np.ndarray:
    rows, cols = np.ogrid[:size, :size]
    radius = (rows - size / 2) ** 2 + (cols - size / 2) ** 2
    return (radius <= outer**2) & (radius > inner**2)


def write_cutout(path: Path, mask: np.ndarray) -> Path:
    rgb = np.full((*mask.shape, 3), 180, dtype=np.uint8)
    Image.fromarray(compose_rgba(rgb, mask), mode="RGBA").save(path, format="PNG")
    return path


def test_a_cutout_written_by_the_segmentation_panel_reads_back_identically(
    tmp_path: Path,
) -> None:
    """The whole point: one file carries both the image and its silhouette."""
    mask = annulus(48, 8.0, 20.0)
    path = write_cutout(tmp_path / "wheel_cutout.png", mask)

    assert np.array_equal(read_object_mask(path), mask)


def test_the_rgb_channels_survive_the_round_trip(tmp_path: Path) -> None:
    mask = annulus(32, 5.0, 14.0)
    path = write_cutout(tmp_path / "c.png", mask)

    rgba = read_rgba(path)

    assert rgba.shape == (32, 32, 4)
    assert (rgba[..., :3][mask] == 180).all()


def test_a_jpeg_is_refused_by_name_rather_than_given_an_invented_mask(
    tmp_path: Path,
) -> None:
    """The failure a user will actually hit: picking the raw photograph."""
    path = tmp_path / "view07.jpg"
    Image.fromarray(np.full((16, 16, 3), 90, dtype=np.uint8), mode="RGB").save(path)

    with pytest.raises(ValueError, match=r"view07\.jpg: this image has no alpha channel"):
        read_object_mask(path)


def test_an_opaque_png_is_refused_too(tmp_path: Path) -> None:
    """A PNG is not automatically a cutout."""
    path = tmp_path / "opaque.png"
    Image.fromarray(np.full((16, 16, 4), 255, dtype=np.uint8), mode="RGBA").save(path)

    with pytest.raises(ValueError, match="opaque"):
        read_object_mask(path)


def test_an_unreadable_file_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "broken.png"
    path.write_bytes(b"not an image")

    with pytest.raises(ValueError, match=r"broken\.png"):
        read_object_mask(path)


def test_the_exif_orientation_is_applied_so_clicks_and_renders_agree(
    tmp_path: Path,
) -> None:
    """All 18 real photographs carry `Orientation=6`; PIL does not apply it.

    The segmentation panel uprights the photograph before the user clicks, so
    reading it back any other way would compare a mask to a render of a part
    lying on its side.
    """
    tall = np.zeros((20, 10, 4), dtype=np.uint8)
    tall[5:15, 2:8, 3] = 255
    path = tmp_path / "rotated.png"
    image = Image.fromarray(tall, mode="RGBA")
    exif = image.getexif()
    exif[274] = 6  # Orientation: rotate 90 degrees
    image.save(path, format="PNG", exif=exif)

    assert read_rgba(path).shape[:2] == (10, 20), "the orientation tag was ignored"
