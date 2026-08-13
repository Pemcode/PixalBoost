"""Adapter: image bytes in, cached GLB out.

Thin on purpose (backends/CONSTRAINTS.md). Its whole job is to decide between a
cache read and exactly one billed GPU job, and to translate the worker's payload
into a `CachedArtifact`.

Multi-view generation is **not** here yet, and must not be added before the F13
gate has ruled. What it will involve is mapped in docs/pixal3d-internals.md:
subclassing upstream's `ProjGrid` to lift its `assert transform_matrix is None`,
recovering the `valid_mask` it discards, and averaging across views under that
mask.
"""

from __future__ import annotations

import base64
import binascii
import struct
from dataclasses import asdict, dataclass
from typing import Any, Protocol

from pixaboost.backends.cache import ArtifactCache, CachedArtifact, cache_key
from pixaboost.backends.runpod_client import RunPodError


class JobRunner(Protocol):
    """The slice of RunPodClient this module needs, so tests can stand in for it."""

    def run(self, payload: dict[str, Any]) -> dict[str, Any]: ...


@dataclass(frozen=True)
class GenerationParams:
    """Everything that changes the output, and therefore the cache key.

    `low_vram` defaults on: during the research phase we fill an artefact cache
    in batches, where throughput per euro matters more than single-shot latency.
    """

    seed: int = 42
    resolution: int = -1
    low_vram: bool = True
    fov: float = -1.0


def generate_single_view(
    *,
    image: bytes,
    params: GenerationParams,
    client: JobRunner,
    cache: ArtifactCache,
    model_revision: str,
) -> CachedArtifact:
    """Return the GLB for `image`, running a GPU job only if it is not cached."""
    key = cache_key(image=image, params=asdict(params), model_revision=model_revision)

    cached = cache.load(key)
    if cached is not None:
        return cached

    with cache.reserve(key):
        # Another process may have completed the same request while this one
        # waited. This second read is the budget-critical part of the lock.
        cached = cache.load(key)
        if cached is not None:
            return cached

        result = client.run({"image": base64.b64encode(image).decode("ascii"), **asdict(params)})

        payload = result.get("glb_base64")
        if not payload:
            raise RunPodError(f"job returned no GLB (keys present: {sorted(result)})")
        try:
            glb = base64.b64decode(payload, validate=True)
        except (binascii.Error, ValueError) as error:
            raise RunPodError(f"job returned a GLB that is not valid base64: {error}") from None
        if not glb:
            raise RunPodError("job returned an empty GLB")
        _validate_glb(glb, result)

        return cache.store(
            key,
            glb=glb,
            metadata={
                "model_revision": model_revision,
                "params": asdict(params),
                "pixal3d_sha": result.get("pixal3d_sha", "unknown"),
            },
        )


def _validate_glb(glb: bytes, result: dict[str, Any]) -> None:
    if len(glb) < 12 or glb[:4] != b"glTF":
        raise RunPodError("job returned an artifact with invalid GLB magic")
    version, declared_length = struct.unpack("<II", glb[4:12])
    if version != 2:
        raise RunPodError(f"job returned unsupported GLB version {version}")
    if declared_length != len(glb):
        raise RunPodError(
            f"job returned a GLB length mismatch: header {declared_length}, bytes {len(glb)}"
        )
    reported_size = result.get("glb_bytes")
    if reported_size is not None and reported_size != len(glb):
        raise RunPodError(f"job reported {reported_size} GLB bytes but returned {len(glb)}")
