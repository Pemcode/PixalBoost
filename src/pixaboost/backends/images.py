"""Read an RGBA cutout off disk and hand `core/` the mask it carries (F15).

Deliberately the thinnest module in `backends/`: PIL in, numpy out, and every
decision about what counts as a usable mask delegated to
`core.segmentation.mask_from_rgba`. Nothing here may grow a fallback that
guesses a mask -- the silhouette is what determines the derived pose, so a
guessed one would produce a confident, wrong alignment.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from pixaboost.core.segmentation import mask_from_rgba

BoolArray = np.ndarray
UInt8Array = np.ndarray


def read_rgba(path: Path | str) -> UInt8Array:
    """Load an image as upright RGBA.

    The EXIF orientation is applied, as the segmentation panel does before the
    user clicks: all the reference photographs carry `Orientation=6` and PIL
    honours it nowhere by default. Skipping it would compare a mask against a
    render of the part lying on its side.
    """
    rgba, _had_alpha = _read(Path(path))
    return rgba


def read_object_mask(path: Path | str) -> BoolArray:
    """Return the object mask carried by a cutout's alpha channel.

    Raises `ValueError` naming the file when it carries none -- a raw
    photograph, or an opaque PNG. That refusal is the feature: it sends the
    caller back to the segmentation step instead of aligning against whatever
    Pixal3D's own background removal happens to produce.
    """
    source = Path(path)
    try:
        rgba, had_alpha = _read(source)
    except Exception as error:
        raise ValueError(f"{source.name}: unreadable image ({type(error).__name__})") from None
    if not had_alpha:
        # Converting to RGBA would have filled alpha with 255, and the caller
        # would be told the image is "fully opaque" -- true, but the wrong
        # diagnosis for someone who simply picked the raw photograph.
        raise ValueError(f"{source.name}: this image has no alpha channel, so it carries no mask")
    try:
        return mask_from_rgba(rgba)
    except ValueError as error:
        raise ValueError(f"{source.name}: {error}") from None


def _read(source: Path) -> tuple[UInt8Array, bool]:
    """Return the upright RGBA pixels and whether the file itself had alpha."""
    from PIL import Image, ImageOps

    with Image.open(source) as opened:
        upright = ImageOps.exif_transpose(opened) or opened
        had_alpha = "A" in upright.getbands()
        return np.asarray(upright.convert("RGBA"), dtype=np.uint8), had_alpha
