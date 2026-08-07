"""TDD for backends.cache (F07).

The artefact cache is what makes a GPU-less development loop possible: once a
generation exists, every downstream experiment reads it from disk instead of
paying for it again. Hard constraints 8 and 9 in CLAUDE.md depend on this
module being correct.

The property that matters most is that a *partial* write is never mistaken for
a cached result. A truncated GLB reported as present would poison every
downstream metric while looking like a cache hit.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from pixaboost.backends.cache import ArtifactCache, cache_key

IMAGE = b"\x89PNG\r\n\x1a\n fake image bytes"
PARAMS = {"seed": 42, "resolution": 1024, "low_vram": True}
REVISION = "cdbb2bb"


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def test_the_same_request_always_yields_the_same_key() -> None:
    assert cache_key(image=IMAGE, params=PARAMS, model_revision=REVISION) == cache_key(
        image=IMAGE, params=PARAMS, model_revision=REVISION
    )


def test_key_order_of_parameters_does_not_matter() -> None:
    shuffled = {"low_vram": True, "resolution": 1024, "seed": 42}
    assert cache_key(image=IMAGE, params=shuffled, model_revision=REVISION) == cache_key(
        image=IMAGE, params=PARAMS, model_revision=REVISION
    )


@pytest.mark.parametrize(
    ("image", "params", "revision"),
    [
        (b"different bytes", PARAMS, REVISION),
        (IMAGE, {**PARAMS, "seed": 43}, REVISION),
        (IMAGE, PARAMS, "0000000"),
    ],
)
def test_any_change_to_the_request_changes_the_key(
    image: bytes, params: dict[str, object], revision: str
) -> None:
    """The model revision is part of the key on purpose: bumping the submodule
    must invalidate every artefact, or we would compare outputs from two
    different models and call it a result."""
    assert cache_key(image=image, params=params, model_revision=revision) != cache_key(
        image=IMAGE, params=PARAMS, model_revision=REVISION
    )


def test_keys_are_filesystem_safe() -> None:
    key = cache_key(image=IMAGE, params=PARAMS, model_revision=REVISION)
    assert key.isalnum() and key.islower() and 16 <= len(key) <= 64


# ---------------------------------------------------------------------------
# Store and load
# ---------------------------------------------------------------------------


def test_a_miss_reports_absent_and_loads_to_none(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    assert not cache.has("deadbeef")
    assert cache.load("deadbeef") is None


def test_a_stored_artefact_round_trips(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    entry = cache.store("abc123", glb=b"glb-bytes", metadata={"seed": 42})
    assert cache.has("abc123")
    loaded = cache.load("abc123")
    assert loaded is not None
    assert loaded.glb_path.read_bytes() == b"glb-bytes"
    assert loaded.metadata["seed"] == 42
    assert loaded.glb_path == entry.glb_path


def test_metadata_records_the_payload_size_and_key(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    cache.store("abc123", glb=b"1234567890", metadata={"seed": 1})
    stored = json.loads((tmp_path / "abc123" / "meta.json").read_text(encoding="utf-8"))
    assert stored["glb_bytes"] == 10
    assert stored["key"] == "abc123"
    assert stored["seed"] == 1


def test_a_half_written_entry_is_not_a_hit(tmp_path: Path) -> None:
    """meta.json is written last and is the only completion marker. A GLB on
    disk without it means a crashed transfer, not a cached result."""
    orphan = tmp_path / "abc123"
    orphan.mkdir()
    (orphan / "output.glb").write_bytes(b"truncated")
    cache = ArtifactCache(tmp_path)
    assert not cache.has("abc123")
    assert cache.load("abc123") is None


def test_storing_twice_is_idempotent_and_keeps_the_first_result(tmp_path: Path) -> None:
    """Never regenerate an artefact that already exists -- the GPU is remote and
    billed by the second (CLAUDE.md constraint 9)."""
    cache = ArtifactCache(tmp_path)
    cache.store("abc123", glb=b"first", metadata={"seed": 1})
    cache.store("abc123", glb=b"second", metadata={"seed": 2})
    loaded = cache.load("abc123")
    assert loaded is not None
    assert loaded.glb_path.read_bytes() == b"first"
    assert loaded.metadata["seed"] == 1


def test_overwrite_is_possible_but_must_be_asked_for(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    cache.store("abc123", glb=b"first", metadata={"seed": 1})
    cache.store("abc123", glb=b"second", metadata={"seed": 2}, overwrite=True)
    loaded = cache.load("abc123")
    assert loaded is not None
    assert loaded.glb_path.read_bytes() == b"second"


def test_storing_empty_bytes_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="empty"):
        ArtifactCache(tmp_path).store("abc123", glb=b"", metadata={})


def test_the_cache_root_is_created_on_demand(tmp_path: Path) -> None:
    root = tmp_path / "does" / "not" / "exist"
    ArtifactCache(root).store("abc123", glb=b"x", metadata={})
    assert (root / "abc123" / "output.glb").is_file()
