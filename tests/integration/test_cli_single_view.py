"""Qt-free CLI integration checks for the single-view reconstruction path."""

from __future__ import annotations

import base64
import json
import struct
from dataclasses import asdict
from pathlib import Path

from pixaboost.backends.cache import ArtifactCache, cache_key
from pixaboost.backends.pixal3d import GenerationParams
from pixaboost.cli import main
from pixaboost.observability import EVENT_PREFIX, parse_event_line

MODEL_REVISION = "cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af"
PROJECT_SHA = "2" * 40
IMAGE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42Y"
    "AAAAASUVORK5CYII="
)


def _glb() -> bytes:
    payload = b"cached-mesh"
    return b"glTF" + struct.pack("<II", 2, 12 + len(payload)) + payload


def _arguments(tmp_path: Path, image: Path) -> list[str]:
    return [
        "reconstruct",
        "single-view",
        str(image),
        "--backend",
        "ssh-pod",
        "--host",
        "ssh.runpod.io",
        "--user",
        "existing-pod-user",
        "--key",
        str(tmp_path / "pod-key"),
        "--known-hosts",
        str(tmp_path / "known-hosts"),
        "--revision",
        MODEL_REVISION,
        "--project-sha",
        PROJECT_SHA,
        "--cache",
        str(tmp_path / "artifacts"),
        "--runs",
        str(tmp_path / "runs"),
        "--resolution",
        "1024",
    ]


def test_cli_cache_hit_emits_jsonl_and_never_needs_confirmation(
    tmp_path: Path, capsys: object
) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(IMAGE)
    params = GenerationParams(seed=42, resolution=1024, low_vram=True, fov=-1.0)
    cache = ArtifactCache(tmp_path / "artifacts")
    key = cache_key(image=IMAGE, params=asdict(params), model_revision=MODEL_REVISION)
    cache.store(key, glb=_glb(), metadata={"model_revision": MODEL_REVISION})

    assert main(_arguments(tmp_path, image)) == 0

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    lines = [line for line in captured.out.splitlines() if line.startswith(EVENT_PREFIX)]
    events = [parse_event_line(line) for line in lines]
    assert [event.phase for event in events] == ["preflight", "completed"]
    assert events[-1].artifact == cache.directory_for(key) / "output.glb"
    manifests = list((tmp_path / "runs").glob("*/manifest.json"))
    assert len(manifests) == 1
    assert json.loads(manifests[0].read_text())["cache"]["hit"] is True


def test_cli_cache_miss_without_confirmation_has_distinct_exit_and_no_network(
    tmp_path: Path, capsys: object
) -> None:
    image = tmp_path / "input.png"
    image.write_bytes(IMAGE)

    assert main(_arguments(tmp_path, image)) == 3

    captured = capsys.readouterr()  # type: ignore[attr-defined]
    assert "explicit approval" in captured.err
    manifests = list((tmp_path / "runs").glob("*/manifest.json"))
    assert len(manifests) == 1
    manifest = json.loads(manifests[0].read_text())
    assert manifest["error"]["code"] == "approval_required"
    assert not (tmp_path / "artifacts").exists()
