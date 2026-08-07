"""RunPod serverless entrypoint for Pixal3D inference.

Deliberately thin. It decodes an image, delegates to upstream's own
`run_inference`, and encodes the resulting GLB back out. It does not reimplement
the camera estimation or the sampler wiring -- that would be business logic in a
backend, which CLAUDE.md forbids, and it would silently drift from the pinned
upstream on the next submodule bump.

Known limitation, accepted for now: `run_inference` reloads the pipeline on every
call, so a warm worker gains nothing. Fine for the smoke test and the batch runs
that fill artifacts/; revisit when Sprint 3 puts this behind a user-facing API.
"""

from __future__ import annotations

import base64
import binascii
import os
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any

sys.path.insert(0, "/opt/pixal3d")

import runpod  # noqa: E402

MAX_IMAGE_BYTES = 32 * 1024 * 1024


def _decode_image(payload: str) -> bytes:
    """Accept a bare base64 string or a data: URI."""
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError(f"image is not valid base64: {error}") from error
    if not raw:
        raise ValueError("image decoded to zero bytes")
    if len(raw) > MAX_IMAGE_BYTES:
        raise ValueError(f"image is {len(raw)} bytes, over the {MAX_IMAGE_BYTES} limit")
    return raw


def handler(job: dict[str, Any]) -> dict[str, Any]:
    job_input = job.get("input") or {}

    # Cheap liveness probe: proves the worker booted and the CUDA extensions
    # import, without paying for a full generation.
    if job_input.get("ping"):
        import torch

        return {
            "pong": True,
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        }

    image_payload = job_input.get("image")
    if not isinstance(image_payload, str) or not image_payload:
        return {"error": "input.image is required and must be a base64-encoded image"}

    try:
        image_bytes = _decode_image(image_payload)
    except ValueError as error:
        return {"error": str(error)}

    from inference import run_inference  # upstream, on PYTHONPATH

    with tempfile.TemporaryDirectory() as workdir:
        image_path = Path(workdir) / "input.png"
        output_path = Path(workdir) / "output.glb"
        image_path.write_bytes(image_bytes)

        try:
            run_inference(
                image_path=str(image_path),
                output_path=str(output_path),
                seed=int(job_input.get("seed", 42)),
                manual_fov=float(job_input.get("fov", -1.0)),
                low_vram=bool(job_input.get("low_vram", True)),
                resolution=int(job_input.get("resolution", -1)),
            )
        except Exception as error:  # surfaced to the caller, not swallowed
            return {"error": f"{type(error).__name__}: {error}", "traceback": traceback.format_exc()}

        if not output_path.is_file():
            return {"error": "inference finished but wrote no GLB"}

        glb = output_path.read_bytes()

    return {
        "glb_base64": base64.b64encode(glb).decode("ascii"),
        "glb_bytes": len(glb),
        "seed": int(job_input.get("seed", 42)),
        "pixal3d_sha": os.environ.get("PIXAL3D_SHA", "unknown"),
    }


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
