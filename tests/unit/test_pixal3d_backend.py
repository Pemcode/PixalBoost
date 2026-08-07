"""TDD for backends.pixal3d (F07).

The adapter that turns "I want a GLB for this image" into either a cache read or
exactly one billed GPU job. The assertion that protects the budget is that a
second identical request submits nothing.
"""

from __future__ import annotations

import base64
from pathlib import Path
from typing import Any

import pytest

from pixaboost.backends.cache import ArtifactCache
from pixaboost.backends.pixal3d import GenerationParams, generate_single_view
from pixaboost.backends.runpod_client import RunPodError

IMAGE = b"\x89PNG\r\n\x1a\n pretend this is a photo"
GLB = b"glTF\x02\x00\x00\x00 pretend this is a mesh"


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
