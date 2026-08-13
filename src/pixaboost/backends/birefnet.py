"""BiRefNet saliency, used only to place a point prompt (F14).

This is the *coarse* half of the chain. BiRefNet does salient object detection:
it answers "what is in the foreground?", with no notion of object identity, so
on these photos it keeps the lifting clamp and the sling because they genuinely
are foreground. Its mask therefore never leaves the system -- it exists to pick
a point that is certainly *on* the part, which SAM then segments properly.

`ZhengPeng7/BiRefNet` is MIT and not gated, per ADR-0010. That matters twice
over here: it is the same weight the pod is forced onto instead of the gated
RMBG-2.0, so the local prompt and the remote cutout agree on what "foreground"
means.

Like `sam3.py`, torch and transformers are imported lazily so that importing
this module -- which `poe check` does -- costs nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np

#: MIT, not gated. See ADR-0010: the gated RMBG-2.0 is the same architecture.
DEFAULT_CHECKPOINT = "ZhengPeng7/BiRefNet"
INPUT_SIZE = 1024
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
#: Saliency is a soft map; anything above this is "probably foreground".
DEFAULT_THRESHOLD = 0.5

BoolArray = np.ndarray


class SaliencyError(RuntimeError):
    """BiRefNet could not be loaded or returned nothing usable."""


class SaliencyRunner(Protocol):
    """The seam the tests substitute, so the gate downloads nothing."""

    def coarse_mask(self, image: np.ndarray) -> BoolArray:
        """Return a boolean foreground mask the size of `image`."""
        ...


def threshold_saliency(
    probabilities: np.ndarray, threshold: float = DEFAULT_THRESHOLD
) -> BoolArray:
    """Turn a soft saliency map into a boolean mask.

    Kept separate from the model so the decision -- and its failure mode -- is
    testable without torch. An all-background map is an error rather than an
    empty mask: it would send `deepest_interior_point` an empty array, and the
    caller would see a confusing geometry error instead of "nothing was found".
    """
    array = np.asarray(probabilities, dtype=np.float64)
    if array.ndim != 2:
        raise SaliencyError(f"saliency map must be 2-D, got shape {array.shape}")
    if not 0.0 < threshold < 1.0:
        raise SaliencyError(f"threshold must be strictly between 0 and 1, got {threshold}")
    mask: BoolArray = array >= threshold
    if not mask.any():
        raise SaliencyError(
            f"BiRefNet found no foreground above {threshold:.2f} "
            f"(max score was {array.max():.3f}); click the part manually"
        )
    return mask


@dataclass
class BiRefNetRunner:
    """Loads BiRefNet once and returns coarse foreground masks."""

    checkpoint: str = DEFAULT_CHECKPOINT
    device: str | None = None
    threshold: float = DEFAULT_THRESHOLD
    half_precision: bool = True
    _model: Any = field(default=None, init=False, repr=False)
    _device: str = field(default="cpu", init=False, repr=False)

    def load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForImageSegmentation
        except ImportError as error:  # pragma: no cover - depends on the optional extra
            raise SaliencyError(
                f"BiRefNet needs torch and transformers: {error}. "
                "Install the extra with `uv sync --extra segmentation`."
            ) from None

        self._device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        try:
            model = AutoModelForImageSegmentation.from_pretrained(
                self.checkpoint, trust_remote_code=True
            )
            model = model.to(self._device).eval()
            if self.half_precision and self._device.startswith("cuda"):
                model = model.half()
        except Exception as error:
            # No token is involved here (the repo is public), so the upstream
            # text is kept verbatim -- it is what names a missing dependency.
            raise SaliencyError(
                f"could not load {self.checkpoint} on {self._device}: "
                f"{type(error).__name__}: {error}"
            ) from None
        self._model = model

    def coarse_mask(self, image: np.ndarray) -> BoolArray:
        self.load()
        import torch

        array = np.asarray(image, dtype=np.uint8)
        if array.ndim != 3 or array.shape[2] != 3:
            raise SaliencyError(f"expected an RGB image, got shape {array.shape}")
        height, width = array.shape[:2]

        tensor = _preprocess(array)
        batch = torch.from_numpy(tensor).unsqueeze(0).to(self._device)
        if self.half_precision and self._device.startswith("cuda"):
            batch = batch.half()
        with torch.no_grad():
            # BiRefNet returns a list of supervision maps; the last is the finest.
            prediction = self._model(batch)[-1].sigmoid()
        probabilities = prediction[0].squeeze().float().cpu().numpy()
        return threshold_saliency(_resize_nearest(probabilities, height, width), self.threshold)


def _preprocess(image: np.ndarray) -> np.ndarray:
    """Resize to 1024x1024, scale to [0, 1], apply ImageNet normalisation."""
    from PIL import Image

    with Image.fromarray(image, mode="RGB") as pil:
        resized = pil.resize((INPUT_SIZE, INPUT_SIZE), Image.Resampling.BILINEAR)
        scaled = np.asarray(resized, dtype=np.float32) / 255.0
    normalised = (scaled - np.array(IMAGENET_MEAN, dtype=np.float32)) / np.array(
        IMAGENET_STD, dtype=np.float32
    )
    return np.ascontiguousarray(normalised.transpose(2, 0, 1))


def _resize_nearest(array: np.ndarray, height: int, width: int) -> np.ndarray:
    """Put a 1024x1024 score map back on the original pixel grid."""
    if array.shape == (height, width):
        return array
    rows = np.clip((np.arange(height) * array.shape[0]) // max(1, height), 0, array.shape[0] - 1)
    cols = np.clip((np.arange(width) * array.shape[1]) // max(1, width), 0, array.shape[1] - 1)
    return array[np.ix_(rows, cols)]
