"""Real single-view inference against a RunPod endpoint (F07).

Never runs in CI and never in `poe check`: it costs GPU seconds. Run it
deliberately:

    RUNPOD_ENDPOINT_ID=<id> uv run pytest tests/e2e -m gpu

Credentials come from RUNPOD_API_KEY or runpod.env. The input is a benchmark
render, so the test needs no photographs and no extra fixtures -- run
`uv run poe bench-build` first.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path

import pytest

from pixaboost.backends.cache import ArtifactCache
from pixaboost.backends.pixal3d import GenerationParams, generate_single_view
from pixaboost.backends.runpod_client import RunPodClient, RunPodEndpoint, load_api_key

pytestmark = [pytest.mark.gpu, pytest.mark.network]

REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_IMAGE = REPO_ROOT / "data" / "bench" / "l_bracket" / "images" / "az000_el+00.png"
GLTF_MAGIC = b"glTF"


def pixal3d_revision() -> str:
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=REPO_ROOT / "vendor" / "pixal3d",
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


@pytest.fixture(scope="module")
def client() -> RunPodClient:
    endpoint_id = os.environ.get("RUNPOD_ENDPOINT_ID", "").strip()
    if not endpoint_id:
        pytest.skip(
            "WHAT: no RunPod endpoint configured.\n"
            "WHY: this test runs real inference and needs a deployed worker.\n"
            "FIX: deploy ghcr.io/pemcode/pixalboost:gpu-latest as a serverless endpoint, "
            "attach a network volume for the ~26 GB of weights, then re-run with "
            "RUNPOD_ENDPOINT_ID=<id>."
        )
    return RunPodClient(
        RunPodEndpoint(endpoint_id=endpoint_id, api_key=load_api_key(REPO_ROOT / "runpod.env")),
        timeout_seconds=3600.0,
    )


def test_the_worker_is_alive_and_sees_a_gpu(client: RunPodClient) -> None:
    """Cheapest possible proof the image boots and its CUDA extensions import."""
    result = client.run({"ping": True})
    assert result["pong"] is True
    assert result["cuda_available"] is True, f"worker has no GPU: {result}"


def test_a_benchmark_render_comes_back_as_a_glb_and_is_cached(
    client: RunPodClient, tmp_path: Path
) -> None:
    if not BENCH_IMAGE.is_file():
        pytest.skip(f"{BENCH_IMAGE} missing; run `uv run poe bench-build` first")

    cache = ArtifactCache(tmp_path)
    revision = pixal3d_revision()
    image = BENCH_IMAGE.read_bytes()

    artifact = generate_single_view(
        image=image,
        params=GenerationParams(seed=42, low_vram=True),
        client=client,
        cache=cache,
        model_revision=revision,
    )

    glb = artifact.glb_path.read_bytes()
    assert glb.startswith(GLTF_MAGIC), f"not a GLB: first bytes were {glb[:8]!r}"
    assert len(glb) > 10_000, f"suspiciously small GLB: {len(glb)} bytes"
    assert artifact.metadata["model_revision"] == revision


def test_the_second_identical_request_costs_nothing(client: RunPodClient, tmp_path: Path) -> None:
    """Hard constraint 9: never regenerate an artefact that already exists."""
    if not BENCH_IMAGE.is_file():
        pytest.skip(f"{BENCH_IMAGE} missing; run `uv run poe bench-build` first")

    cache = ArtifactCache(tmp_path)
    submitted: list[dict[str, object]] = []

    class RecordingClient:
        def run(self, payload: dict[str, object]) -> dict[str, object]:
            submitted.append(payload)
            return client.run(payload)

    recording = RecordingClient()
    common = {
        "image": BENCH_IMAGE.read_bytes(),
        "params": GenerationParams(seed=42, low_vram=True),
        "client": recording,
        "cache": cache,
        "model_revision": pixal3d_revision(),
    }

    first = generate_single_view(**common)  # type: ignore[arg-type]
    second = generate_single_view(**common)  # type: ignore[arg-type]

    assert first.key == second.key
    assert len(submitted) == 1, "the cache did not prevent a second billed job"
