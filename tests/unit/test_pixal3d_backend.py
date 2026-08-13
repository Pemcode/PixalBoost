"""TDD for backends.pixal3d (F07).

The adapter that turns "I want a GLB for this image" into either a cache read or
exactly one billed GPU job. The assertion that protects the budget is that a
second identical request submits nothing.
"""

from __future__ import annotations

import base64
import multiprocessing
import struct
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event, Lock
from typing import Any

import pytest

from pixaboost.backends.cache import ArtifactCache, CacheCorruptionError
from pixaboost.backends.pixal3d import GenerationParams, generate_single_view
from pixaboost.backends.runpod_client import RunPodError

IMAGE = b"\x89PNG\r\n\x1a\n pretend this is a photo"
_GLB_PAYLOAD = b"pretend this is a mesh"
GLB = b"glTF" + struct.pack("<II", 2, 12 + len(_GLB_PAYLOAD)) + _GLB_PAYLOAD


class CountingClient:
    """Stands in for RunPodClient and records every job it is asked to run."""

    def __init__(self, output: dict[str, Any] | None = None) -> None:
        self.output = (
            output if output is not None else {"glb_base64": base64.b64encode(GLB).decode()}
        )
        self.jobs: list[dict[str, Any]] = []

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.jobs.append(payload)
        return self.output


class BlockingCountingClient(CountingClient):
    """Keep the first fake GPU call open while a concurrent request starts."""

    def __init__(self) -> None:
        super().__init__()
        self.first_call_started = Event()
        self.duplicate_call_started = Event()
        self.release = Event()
        self._calls_lock = Lock()

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        with self._calls_lock:
            self.jobs.append(payload)
            if len(self.jobs) == 1:
                self.first_call_started.set()
            else:
                self.duplicate_call_started.set()
        if not self.release.wait(timeout=5.0):
            raise AssertionError("test did not release the fake GPU call")
        return self.output


def _generate_in_process(
    root: str,
    ready: Any,
    start: Any,
    calls: Any,
    results: Any,
) -> None:
    class SharedCountingClient:
        def run(self, _payload: dict[str, Any]) -> dict[str, Any]:
            with calls.get_lock():
                calls.value += 1
            time.sleep(0.2)
            return {"glb_base64": base64.b64encode(GLB).decode("ascii")}

    ready.set()
    if not start.wait(timeout=10.0):
        raise RuntimeError("parent did not start the concurrent generation")
    artifact = generate_single_view(
        image=IMAGE,
        params=GenerationParams(),
        client=SharedCountingClient(),
        cache=ArtifactCache(root),
        model_revision="cdbb2bb",
    )
    results.put(artifact.key)


def test_a_first_request_runs_one_job_and_caches_the_result(tmp_path: Path) -> None:
    client, cache = CountingClient(), ArtifactCache(tmp_path)
    artifact = generate_single_view(
        image=IMAGE, params=GenerationParams(), client=client, cache=cache, model_revision="cdbb2bb"
    )
    assert len(client.jobs) == 1
    assert artifact.glb_path.read_bytes() == GLB
    assert cache.has(artifact.key)


def test_a_repeated_request_submits_nothing(tmp_path: Path) -> None:
    """The budget guard: the GPU is remote and billed by the second."""
    client, cache = CountingClient(), ArtifactCache(tmp_path)
    common = {"image": IMAGE, "client": client, "cache": cache, "model_revision": "cdbb2bb"}
    first = generate_single_view(params=GenerationParams(), **common)
    second = generate_single_view(params=GenerationParams(), **common)
    assert len(client.jobs) == 1
    assert first.key == second.key


def test_concurrent_identical_requests_submit_exactly_one_job(tmp_path: Path) -> None:
    """Both callers observe the initial miss, but only the reservation owner may submit."""
    client = BlockingCountingClient()
    cache = ArtifactCache(tmp_path)
    common = {
        "image": IMAGE,
        "params": GenerationParams(),
        "client": client,
        "cache": cache,
        "model_revision": "cdbb2bb",
    }
    second_invocation_started = Event()

    def invoke_second() -> Any:
        second_invocation_started.set()
        return generate_single_view(**common)

    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(generate_single_view, **common)
        assert client.first_call_started.wait(timeout=2.0)
        second_future = executor.submit(invoke_second)
        assert second_invocation_started.wait(timeout=2.0)
        duplicate_started = client.duplicate_call_started.wait(timeout=0.25)
        client.release.set()
        first = first_future.result(timeout=2.0)
        second = second_future.result(timeout=2.0)

    assert duplicate_started is False
    assert len(client.jobs) == 1
    assert first.key == second.key


def test_identical_requests_in_two_processes_submit_exactly_one_job(tmp_path: Path) -> None:
    context = multiprocessing.get_context("spawn")
    ready = [context.Event(), context.Event()]
    start = context.Event()
    calls = context.Value("i", 0)
    results = context.Queue()
    processes = [
        context.Process(
            target=_generate_in_process,
            args=(str(tmp_path), ready[index], start, calls, results),
        )
        for index in range(2)
    ]
    for process in processes:
        process.start()
    try:
        assert all(event.wait(timeout=10.0) for event in ready)
        start.set()
        for process in processes:
            process.join(timeout=10.0)
            assert process.exitcode == 0
        keys = [results.get(timeout=2.0), results.get(timeout=2.0)]
        assert calls.value == 1
        assert keys[0] == keys[1]
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join(timeout=5.0)


def test_changing_the_seed_is_a_different_artefact(tmp_path: Path) -> None:
    client, cache = CountingClient(), ArtifactCache(tmp_path)
    common = {"image": IMAGE, "client": client, "cache": cache, "model_revision": "cdbb2bb"}
    generate_single_view(params=GenerationParams(seed=1), **common)
    generate_single_view(params=GenerationParams(seed=2), **common)
    assert len(client.jobs) == 2


def test_a_new_model_revision_invalidates_the_cache(tmp_path: Path) -> None:
    client, cache = CountingClient(), ArtifactCache(tmp_path)
    common = {"image": IMAGE, "params": GenerationParams(), "client": client, "cache": cache}
    generate_single_view(model_revision="cdbb2bb", **common)
    generate_single_view(model_revision="0000000", **common)
    assert len(client.jobs) == 2, "two different models must never share an artefact"


def test_the_job_payload_carries_the_image_and_the_parameters(tmp_path: Path) -> None:
    client, cache = CountingClient(), ArtifactCache(tmp_path)
    generate_single_view(
        image=IMAGE,
        params=GenerationParams(seed=7, resolution=1024, low_vram=False),
        client=client,
        cache=cache,
        model_revision="cdbb2bb",
    )
    payload = client.jobs[0]
    assert base64.b64decode(payload["image"]) == IMAGE
    assert payload["seed"] == 7
    assert payload["resolution"] == 1024
    assert payload["low_vram"] is False


def test_metadata_records_provenance(tmp_path: Path) -> None:
    client, cache = CountingClient(), ArtifactCache(tmp_path)
    artifact = generate_single_view(
        image=IMAGE,
        params=GenerationParams(seed=3),
        client=client,
        cache=cache,
        model_revision="cdbb2bb",
    )
    assert artifact.metadata["model_revision"] == "cdbb2bb"
    assert artifact.metadata["params"]["seed"] == 3


def test_a_response_without_a_glb_is_an_error_not_an_empty_cache_entry(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    client = CountingClient(output={"glb_bytes": 0})
    with pytest.raises(RunPodError, match="no GLB"):
        generate_single_view(
            image=IMAGE,
            params=GenerationParams(),
            client=client,
            cache=cache,
            model_revision="cdbb2bb",
        )
    assert not any(tmp_path.iterdir()), "a failed job must leave no cache entry behind"


def test_an_undecodable_glb_is_rejected(tmp_path: Path) -> None:
    cache = ArtifactCache(tmp_path)
    client = CountingClient(output={"glb_base64": "not!valid!base64!"})
    with pytest.raises(RunPodError, match="base64"):
        generate_single_view(
            image=IMAGE,
            params=GenerationParams(),
            client=client,
            cache=cache,
            model_revision="cdbb2bb",
        )
    assert not any(tmp_path.iterdir())


@pytest.mark.parametrize(
    ("glb", "message"),
    [
        (b"not-a-glb-payload", "magic"),
        (b"glTF" + struct.pack("<II", 1, 12), "version"),
        (b"glTF" + struct.pack("<II", 2, 99), "length"),
    ],
)
def test_a_structurally_invalid_glb_is_rejected_before_cache(
    tmp_path: Path, glb: bytes, message: str
) -> None:
    client = CountingClient(output={"glb_base64": base64.b64encode(glb).decode()})
    with pytest.raises(RunPodError, match=message):
        generate_single_view(
            image=IMAGE,
            params=GenerationParams(),
            client=client,
            cache=ArtifactCache(tmp_path),
            model_revision="cdbb2bb",
        )
    assert not any(tmp_path.iterdir())


def test_corrupted_cache_never_triggers_a_new_gpu_job(tmp_path: Path) -> None:
    client = CountingClient()
    cache = ArtifactCache(tmp_path)
    common = {
        "image": IMAGE,
        "params": GenerationParams(),
        "client": client,
        "cache": cache,
        "model_revision": "cdbb2bb",
    }
    artifact = generate_single_view(**common)
    artifact.glb_path.write_bytes(GLB[:-1] + bytes([GLB[-1] ^ 1]))
    with pytest.raises(CacheCorruptionError):
        generate_single_view(**common)
    assert len(client.jobs) == 1
