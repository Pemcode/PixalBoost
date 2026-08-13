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
import multiprocessing
import time
from pathlib import Path
from typing import Any

import pytest

from pixaboost.backends.cache import (
    ArtifactCache,
    CacheCorruptionError,
    CacheReservationTimeoutError,
    cache_key,
)

IMAGE = b"\x89PNG\r\n\x1a\n fake image bytes"
PARAMS = {"seed": 42, "resolution": 1024, "low_vram": True}


def _hold_cache_reservation(root: str, ready: Any) -> None:
    """Acquire a real interprocess reservation, then wait to be terminated by the test."""
    with ArtifactCache(root).reserve(
        "shared-key",
        timeout_seconds=2.0,
        stale_after_seconds=60.0,
        poll_interval_seconds=0.01,
    ):
        ready.set()
        time.sleep(30.0)


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
    assert stored["glb_sha256"]


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


def test_tampered_cached_bytes_are_a_blocking_corruption_not_a_miss(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    cache.store("abc123", glb=b"first", metadata={"seed": 1})
    (tmp_path / "abc123" / "output.glb").write_bytes(b"other")
    with pytest.raises(CacheCorruptionError, match="SHA-256"):
        cache.load("abc123")


def test_completion_marker_without_glb_is_a_blocking_corruption(tmp_path: Path) -> None:
    entry = tmp_path / "abc123"
    entry.mkdir()
    (entry / "meta.json").write_text(
        '{"key":"abc123","glb_bytes":12,"glb_sha256":"' + "0" * 64 + '"}',
        encoding="utf-8",
    )
    with pytest.raises(CacheCorruptionError, match="missing"):
        ArtifactCache(tmp_path).load("abc123")


def test_malformed_or_unverifiable_metadata_is_a_blocking_corruption(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    cache.store("abc123", glb=b"first", metadata={})
    meta = tmp_path / "abc123" / "meta.json"
    meta.write_text('{"key":"wrong","glb_bytes":5}', encoding="utf-8")
    with pytest.raises(CacheCorruptionError, match="metadata"):
        cache.load("abc123")


def test_reservation_left_by_a_dead_process_is_reclaimed(tmp_path: Path) -> None:
    """A killed worker must not block this cache key forever or require a paid retry."""
    context = multiprocessing.get_context("spawn")
    ready = context.Event()
    process = context.Process(target=_hold_cache_reservation, args=(str(tmp_path), ready))
    process.start()
    try:
        assert ready.wait(timeout=10.0)
        process.terminate()
        process.join(timeout=5.0)
        assert not process.is_alive()

        with ArtifactCache(tmp_path).reserve(
            "shared-key",
            timeout_seconds=1.0,
            stale_after_seconds=60.0,
            poll_interval_seconds=0.01,
        ):
            pass
    finally:
        if process.is_alive():
            process.terminate()
            process.join(timeout=5.0)


def test_live_reservation_times_out_without_being_stolen(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    with cache.reserve("shared-key", timeout_seconds=1.0):
        started = time.monotonic()
        with (
            pytest.raises(CacheReservationTimeoutError),
            ArtifactCache(tmp_path).reserve(
                "shared-key",
                timeout_seconds=0.05,
                stale_after_seconds=0.0,
                poll_interval_seconds=0.005,
            ),
        ):
            raise AssertionError("a live owner must not be displaced")
        assert time.monotonic() - started < 0.5


def test_reservation_is_released_when_the_protected_operation_raises(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    with pytest.raises(RuntimeError, match="fake failure"), cache.reserve(
        "shared-key", timeout_seconds=1.0
    ):
        raise RuntimeError("fake failure")

    with ArtifactCache(tmp_path).reserve("shared-key", timeout_seconds=0.0):
        pass
