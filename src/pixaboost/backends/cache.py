"""Content-addressed cache for GPU artefacts.

This module is what makes a GPU-less development loop possible. Once a
generation exists on disk, every downstream experiment reads it instead of
paying for it again -- and with no local GPU, that is the difference between
iterating in seconds and iterating in billed minutes.

Two invariants carry the weight:

- **The key covers the model revision.** Bumping the `vendor/pixal3d` submodule
  must invalidate every artefact. Otherwise we would compare outputs from two
  different models and call the difference a result.
- **`meta.json` is written last and is the only completion marker.** A GLB on
  disk without it is a crashed transfer, not a cache hit. A truncated file
  reported as present would poison every metric downstream while looking fine.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

GLB_NAME = "output.glb"
META_NAME = "meta.json"
_KEY_LENGTH = 32


def cache_key(*, image: bytes, params: Mapping[str, Any], model_revision: str) -> str:
    """Derive a stable, filesystem-safe key for one generation request.

    Parameters are serialised with sorted keys so that argument order cannot
    produce two keys for the same request.
    """
    digest = hashlib.sha256()
    digest.update(model_revision.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(json.dumps(dict(params), sort_keys=True, default=str).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(image)
    return digest.hexdigest()[:_KEY_LENGTH]


@dataclass(frozen=True)
class CachedArtifact:
    """A completed cache entry."""

    key: str
    glb_path: Path
    metadata: dict[str, Any]


class ArtifactCache:
    """A directory of completed generations, one subdirectory per key."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def directory_for(self, key: str) -> Path:
        return self.root / key

    def has(self, key: str) -> bool:
        entry = self.directory_for(key)
        return (entry / META_NAME).is_file() and (entry / GLB_NAME).is_file()

    def load(self, key: str) -> CachedArtifact | None:
        if not self.has(key):
            return None
        entry = self.directory_for(key)
        return CachedArtifact(
            key=key,
            glb_path=entry / GLB_NAME,
            metadata=json.loads((entry / META_NAME).read_text(encoding="utf-8")),
        )

    def store(
        self,
        key: str,
        *,
        glb: bytes,
        metadata: Mapping[str, Any],
        overwrite: bool = False,
    ) -> CachedArtifact:
        """Write an artefact, leaving an existing one untouched unless asked.

        Refusing to overwrite by default is the enforcement point for hard
        constraint 9: never regenerate an artefact that already exists.
        """
        if not glb:
            raise ValueError("refusing to cache an empty GLB payload")

        existing = self.load(key)
        if existing is not None and not overwrite:
            return existing

        entry = self.directory_for(key)
        entry.mkdir(parents=True, exist_ok=True)
        self._atomic_write(entry / GLB_NAME, glb)
        self._atomic_write(
            entry / META_NAME,
            json.dumps({**dict(metadata), "key": key, "glb_bytes": len(glb)}, indent=2).encode(
                "utf-8"
            ),
        )
        loaded = self.load(key)
        assert loaded is not None  # just written
        return loaded

    @staticmethod
    def _atomic_write(destination: Path, payload: bytes) -> None:
        """Write via a sibling temp file and rename, so readers never see a partial file."""
        handle, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".partial")
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise
