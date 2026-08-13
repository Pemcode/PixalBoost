"""Mask helpers that turn a coarse saliency mask into a point prompt (F14).

BiRefNet answers "what is in the foreground?". SAM answers "which object, given
this hint?". Only the second can drop a lifting clamp that *touches* the part,
so BiRefNet is demoted here to a prompt generator and SAM's mask is the one
that counts.

Deriving that prompt is pure geometry, which is why it lives in `core/` and not
next to the model. The one thing it must not do is take the centroid: the test
piece is a wheel with a central bore, and its centroid is background.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

BoolArray = np.ndarray
UInt8Array = np.ndarray


def _as_binary_2d(mask: BoolArray) -> BoolArray:
    array = np.asarray(mask)
    if array.ndim != 2:
        raise ValueError(f"mask must be 2-D, got shape {array.shape}")
    return array.astype(bool, copy=False)


def deepest_interior_point(mask: BoolArray) -> tuple[int, int]:
    """Return the `(row, col)` furthest from any background pixel.

    This is the centre of the largest circle inscribed in the mask, obtained as
    the argmax of the Euclidean distance transform. On a disc it coincides with
    the centroid; on an annulus it sits on the mid-radius ring while the
    centroid falls in the hole. It is also the point a human would click.

    Ties are broken by `argmax`'s row-major scan, so the result is
    deterministic for a given mask.
    """
    binary = _as_binary_2d(mask)
    if not binary.any():
        raise ValueError("cannot derive a prompt from an empty mask")

    # Pad by one so a blob touching the border is not treated as infinitely
    # deep in that direction; the offset is removed on the way out.
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    distance = ndimage.distance_transform_edt(padded)
    flat = int(np.argmax(distance))
    row, col = np.unravel_index(flat, distance.shape)
    return int(row) - 1, int(col) - 1


def mask_centroid(mask: BoolArray) -> tuple[int, int]:
    """Return the centre of mass of `mask`, rounded to a pixel.

    Present so that callers -- and `test_segmentation.py` -- can state plainly
    that this point is *not* a usable prompt on a holed part. It is a
    diagnostic, never an input to SAM.
    """
    binary = _as_binary_2d(mask)
    if not binary.any():
        raise ValueError("cannot take the centroid of an empty mask")
    rows, cols = np.nonzero(binary)
    return round(float(rows.mean())), round(float(cols.mean()))


def largest_connected_component(mask: BoolArray) -> BoolArray:
    """Keep only the biggest 8-connected blob.

    Saliency masks come back speckled with bits of rack and pallet. Holes are
    left alone on purpose: filling the bore of a wheel would put the deepest
    interior point back in the middle of the background.
    """
    binary = _as_binary_2d(mask)
    if not binary.any():
        return np.zeros_like(binary)

    structure = np.ones((3, 3), dtype=bool)
    labels, count = ndimage.label(binary, structure=structure)
    if count <= 1:
        return binary.copy()
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0  # background
    biggest: BoolArray = labels == int(np.argmax(sizes))
    return biggest


def compose_rgba(rgb: UInt8Array, mask: BoolArray) -> UInt8Array:
    """Build the RGBA image that short-circuits Pixal3D's own background removal.

    `preprocess_image` (vendor/pixal3d/.../pixal3d_image_to_3d.py) only honours
    a supplied alpha when the image is RGBA *and* `not np.all(alpha == 255)`.
    A fully opaque result would therefore be silently handed to rembg, and the
    clamp we just removed would come straight back -- so it is refused here
    rather than discovered on a billed GPU.

    Background colour is zeroed as well as made transparent: the downstream
    crop reads `alpha > 0.8 * 255`, but any consumer that flattens the image
    would otherwise composite the workbench back in.
    """
    colour = np.asarray(rgb)
    if colour.ndim != 3 or colour.shape[2] != 3:
        raise ValueError(f"rgb must have shape (H, W, 3), got {colour.shape}")
    binary = _as_binary_2d(mask)
    if binary.shape != colour.shape[:2]:
        raise ValueError(f"mask shape {binary.shape} does not match image shape {colour.shape[:2]}")
    if not binary.any():
        raise ValueError("refusing to build a cutout from an empty mask")
    if binary.all():
        raise ValueError(
            "refusing to build a fully opaque cutout: Pixal3D ignores alpha that is "
            "uniformly 255 and would run its own background removal instead"
        )

    rgba = np.zeros((*binary.shape, 4), dtype=np.uint8)
    rgba[..., :3] = np.where(binary[..., None], colour.astype(np.uint8, copy=False), 0)
    rgba[..., 3] = np.where(binary, 255, 0).astype(np.uint8)
    return rgba
