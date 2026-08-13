"""SAM 3 point-prompt segmentation, behind a narrow adapter (F14).

Translation only, no policy: this module turns `(image, points)` into a boolean
mask -- a `core/` type -- and nothing else. Which point to click, what to do
with the mask, and whether the result is acceptable are decided elsewhere. See
backends/CONSTRAINTS.md.

Two things are worth knowing before touching it.

**The click path is `Sam3Tracker*`, not `Sam3Model`.** SAM 3's headline feature
is Promptable *Concept* Segmentation -- text and image exemplars -- and
`Sam3Model` exposes exactly that: `text` and `input_boxes`, no `input_points`.
A user click is Promptable *Visual* Segmentation, which lives in
`Sam3TrackerModel`, described upstream as SAM 2 with the same API and better
weights. Same `facebook/sam3` checkpoint, different head.

**The checkpoint is gated.** `facebook/sam3` is `license: other` and requires
accepting Meta's SAM License on the Hub. See ADR-0014 for why the project took
that on despite ADR-0010, and `docs/segmentation.md` for the operational cost.

`torch` and `transformers` are imported lazily inside the loader so that
importing this module -- which `poe check` does -- stays free.
"""

from __future__ import annotations

import os
import re
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

DEFAULT_CHECKPOINT = "facebook/sam3"
DEFAULT_TOKEN_FILE = Path("huggingface.env")
#: Hub env vars, in the order the Hugging Face libraries themselves consult them.
TOKEN_ENV_VARS = ("HUGGINGFACE_TOKEN", "HF_TOKEN", "HUGGING_FACE_HUB_TOKEN")

BoolArray = np.ndarray


class Sam3Error(RuntimeError):
    """SAM 3 could not be loaded or produced nothing usable.

    Never carries a token: upstream text is passed through `redact_token`.
    """


#: Any Hugging Face-shaped token, whoever emitted it. The explicit token is
#: redacted too, in case the Hub ever echoes it in some other shape.
_TOKEN_PATTERN = re.compile(r"(?i)(?<![A-Za-z0-9_.-])hf_[A-Za-z0-9_.-]{4,}")


def redact_token(text: str, token: str | None = None) -> str:
    """Strip credentials from text that is about to be shown or logged.

    Dropping the message instead was tried and is worse: it turned a missing
    `torchvision` into a bare "ImportError" that looked like an auth failure,
    and cost a debugging session. Keep the diagnosis, remove the secret.
    """
    cleaned = _TOKEN_PATTERN.sub("<redacted>", text)
    if token:
        cleaned = cleaned.replace(token, "<redacted>")
    return cleaned


@contextmanager
def _hub_token_in_environment(token: str) -> Iterator[None]:
    """Expose the token to every Hub call for the duration of a load.

    Passing `token=` to `from_pretrained` only covers the calls we make
    ourselves. `transformers` resolves configs, remote code and companion files
    through its own internal helpers, and those that do not forward the
    argument fall back to the ambient credentials -- which produced an
    intermittent 401 on this gated repo while an explicit request for the very
    same file returned 200.

    Restored on exit, including on failure, so the token does not leak into
    later subprocesses.
    """
    previous = {name: os.environ.get(name) for name in TOKEN_ENV_VARS}
    os.environ.update(dict.fromkeys(TOKEN_ENV_VARS, token))
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


@dataclass(frozen=True)
class PointPrompt:
    """One click, in image pixel coordinates.

    `positive=False` is a click on something to *exclude* -- the clamp, the
    sling, the bench -- which is how a user corrects a mask that swallowed the
    thing touching the part.
    """

    x: int
    y: int
    positive: bool = True


@dataclass(frozen=True)
class SegmentationResult:
    """The chosen mask and the score that chose it."""

    mask: BoolArray
    iou_score: float
    #: All candidate scores, best first, so a caller can see how close the runner-up was.
    candidate_scores: tuple[float, ...] = ()

    @property
    def is_ambiguous(self) -> bool:
        """True when the runner-up mask scored within 5 % of the winner.

        Reported, never acted on: a click on a wheel rim legitimately has a
        near-tie between "the rim" and "the whole wheel".
        """
        if len(self.candidate_scores) < 2:
            return False
        best, second = self.candidate_scores[0], self.candidate_scores[1]
        return best > 0.0 and (best - second) / best < 0.05


class Sam3Runner(Protocol):
    """The seam the tests substitute, so the gate never downloads 3.4 GB."""

    def segment(self, image: np.ndarray, prompts: tuple[PointPrompt, ...]) -> SegmentationResult:
        """Return the best mask for `prompts` over `image` (H, W, 3 uint8)."""
        ...


def load_hf_token(source: Path | str | None = None) -> str:
    """Read the Hugging Face token from the environment or from a file.

    Mirrors `runpod_client.load_api_key`: the file may hold a bare `hf_...`
    token on one line -- which is how `huggingface.env` is written -- or a
    `NAME=value` pair. Comments and blank lines are ignored.

    No exception raised here ever contains the token.
    """
    for variable in TOKEN_ENV_VARS:
        from_env = os.environ.get(variable, "").strip()
        if from_env:
            return from_env

    path = Path(source) if source is not None else DEFAULT_TOKEN_FILE
    if not path.is_file():
        raise Sam3Error(
            f"no Hugging Face credentials: set {TOKEN_ENV_VARS[0]}, or create {path}. "
            f"{DEFAULT_CHECKPOINT} is a gated repository, so a token is required even "
            f"though the weights are public. The file may contain the bare token on one line."
        )

    for raw in path.read_text(encoding="utf-8").splitlines() or [""]:
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        _, separator, value = line.partition("=")
        candidate = (value if separator else line).strip().strip("'\"")
        if candidate:
            return candidate

    raise Sam3Error(f"{path} holds no usable token (it is empty or only comments)")


def _select_best(masks: Any, scores: Any) -> SegmentationResult:
    """Pick the highest-IoU candidate and record the runner-ups.

    `post_process_masks` returns `(objects, candidates, H, W)` and `iou_scores`
    returns `(batch, object, candidates)`; both are indexed here for the single
    object this adapter prompts for.
    """
    mask_array = np.asarray(masks)
    score_array = np.asarray(scores, dtype=np.float64).reshape(-1)
    if mask_array.ndim != 3 or mask_array.shape[0] == 0:
        raise Sam3Error(f"SAM 3 returned no usable mask (masks had shape {mask_array.shape})")
    if score_array.size != mask_array.shape[0]:
        raise Sam3Error(
            f"SAM 3 returned {score_array.size} scores for {mask_array.shape[0]} masks"
        )

    order = np.argsort(-score_array)
    best = int(order[0])
    mask = mask_array[best].astype(bool, copy=False)
    if not mask.any():
        raise Sam3Error("SAM 3 returned an empty mask for this prompt")
    return SegmentationResult(
        mask=mask,
        iou_score=float(score_array[best]),
        candidate_scores=tuple(float(s) for s in score_array[order]),
    )


@dataclass
class Sam3TrackerRunner:
    """Loads `facebook/sam3` once and answers clicks against it.

    Kept out of `poe check` by construction: nothing here runs until `segment`
    or `load` is called, and both need torch.
    """

    checkpoint: str = DEFAULT_CHECKPOINT
    device: str | None = None
    token_file: Path | None = None
    #: fp16 halves the ~3.4 GB of weights; an 8 GB laptop card has no room to spare.
    half_precision: bool = True
    _model: Any = field(default=None, init=False, repr=False)
    _processor: Any = field(default=None, init=False, repr=False)

    def load(self) -> None:
        """Download (once) and place the model. Safe to call repeatedly."""
        if self._model is not None:
            return
        try:
            import torch
            from transformers import Sam3TrackerModel, Sam3TrackerProcessor
        except ImportError as error:  # pragma: no cover - depends on the optional extra
            raise Sam3Error(
                f"SAM 3 needs torch and transformers: {error}. Install the extra with "
                "`uv sync --extra segmentation`."
            ) from None

        token = load_hf_token(self.token_file)
        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        dtype = torch.float16 if (self.half_precision and device.startswith("cuda")) else None
        try:
            with _hub_token_in_environment(token):
                self._processor = Sam3TrackerProcessor.from_pretrained(
                    self.checkpoint, token=token
                )
                self._model = Sam3TrackerModel.from_pretrained(
                    self.checkpoint, token=token, dtype=dtype
                ).to(device)
        except Exception as error:
            # Redact the token, keep the text. An earlier version dropped the
            # message entirely and reported only the exception class, which
            # turned a plain missing-torchvision install into an unactionable
            # "ImportError" that read like an auth failure.
            raise Sam3Error(
                f"could not load {self.checkpoint} on {device}: "
                f"{type(error).__name__}: {redact_token(str(error), token)}"
            ) from None
        self._model.eval()

    def segment(self, image: np.ndarray, prompts: tuple[PointPrompt, ...]) -> SegmentationResult:
        if not prompts:
            raise Sam3Error("at least one point prompt is required")
        self.load()
        import torch
        from PIL import Image

        pil = Image.fromarray(np.asarray(image, dtype=np.uint8), mode="RGB")
        # Shapes are (image, object, point, xy) and (image, object, point).
        input_points = [[[[int(p.x), int(p.y)] for p in prompts]]]
        input_labels = [[[1 if p.positive else 0 for p in prompts]]]

        inputs = self._processor(
            images=pil,
            input_points=input_points,
            input_labels=input_labels,
            return_tensors="pt",
        ).to(self._model.device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        masks = self._processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"]
        )[0]
        return _select_best(np.asarray(masks[0]), np.asarray(outputs.iou_scores.cpu()).reshape(-1))
