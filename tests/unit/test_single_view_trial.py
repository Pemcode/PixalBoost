"""CPU contract for the observable, cache-first single-view trial service."""

from __future__ import annotations

import base64
import hashlib
import json
import struct
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

from pixaboost.backends.cache import ArtifactCache, cache_key
from pixaboost.backends.pixal3d import GenerationParams
from pixaboost.backends.ssh_pod import ExistingPodUseApproval, SshPodConfig, SshPodError
from pixaboost.observability import TelemetryEvent
from pixaboost.trials.single_view import (
    SingleViewTrialConfig,
    preflight_single_view,
    run_single_view_trial,
)

MODEL_REVISION = "cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af"
PROJECT_SHA = "1" * 40
IMAGE = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42Y"
    "AAAAASUVORK5CYII="
)
IMAGE_2X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAABCAIAAAB7QOjdAAAADElEQVR4nGP8zwACAAYIAQFa"
    "zwZIAAAAAElFTkSuQmCC"
)


def _glb(payload: bytes = b"mesh") -> bytes:
    size = 12 + len(payload)
    return b"glTF" + struct.pack("<II", 2, size) + payload


def _config(tmp_path: Path, *, params: GenerationParams | None = None) -> SingleViewTrialConfig:
    image_path = tmp_path / "input.png"
    image_path.write_bytes(IMAGE)
    ssh = SshPodConfig(
        host="ssh.runpod.io",
        username="existing-pod",
        private_key_path=tmp_path / "pod-key",
        known_hosts_path=tmp_path / "known-hosts",
        expected_pixal3d_sha=MODEL_REVISION,
        project_git_sha=PROJECT_SHA,
        local_runs_root=tmp_path / "runs",
    )
    return SingleViewTrialConfig(
        image_path=image_path,
        params=params or GenerationParams(seed=42, resolution=1024),
        cache=ArtifactCache(tmp_path / "artifacts"),
        ssh=ssh,
        poses=({"view": "canonical", "transform": None},),
    )


def _cache_artifact(config: SingleViewTrialConfig, glb: bytes | None = None) -> str:
    key = cache_key(
        image=IMAGE,
        params=asdict(config.params),
        model_revision=config.ssh.expected_pixal3d_sha,
    )
    config.cache.store(
        key,
        glb=glb or _glb(),
        metadata={
            "model_revision": config.ssh.expected_pixal3d_sha,
            "params": asdict(config.params),
        },
    )
    return key


class FakeClient:
    def __init__(self, tmp_path: Path, events: list[Any]) -> None:
        self.calls = 0
        self.last_manifest_path = tmp_path / "transport" / "manifest.json"
        self.events = events

    def run(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.calls += 1
        self.last_manifest_path.parent.mkdir(parents=True, exist_ok=True)
        self.last_manifest_path.write_text('{"status":"completed"}', encoding="utf-8")
        glb = _glb(b"generated")
        return {
            "glb_base64": base64.b64encode(glb).decode("ascii"),
            "glb_bytes": len(glb),
            "seed": payload["seed"],
            "pixal3d_sha": MODEL_REVISION,
            "manifest_path": str(self.last_manifest_path),
        }


def test_preflight_identifies_the_exact_cache_key_without_network(tmp_path: Path) -> None:
    config = _config(tmp_path)
    key = _cache_artifact(config)

    preflight = preflight_single_view(config)

    assert preflight.cache_hit is True
    assert preflight.cache_key == key
    assert preflight.artifact is not None
    assert preflight.approval_required is False


def test_cache_hit_needs_no_approval_or_client_and_writes_complete_evidence(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    key = _cache_artifact(config)
    constructed = 0
    events: list[TelemetryEvent] = []

    def forbidden_factory(*_args: Any, **_kwargs: Any) -> FakeClient:
        nonlocal constructed
        constructed += 1
        raise AssertionError("a cache hit must not construct or connect an SSH client")

    result = run_single_view_trial(
        config,
        approval=None,
        event_sink=events.append,
        client_factory=forbidden_factory,
        run_id_factory=lambda: "trial-cache-hit",
    )

    assert constructed == 0
    assert result.cache_hit is True
    assert result.artifact.key == key
    assert result.manifest_path.is_file()
    assert result.metrics_path.is_file()
    assert result.logs_path.is_file()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["cache"]["hit"] is True
    assert manifest["git_sha"] == PROJECT_SHA
    assert manifest["model_revision"] == MODEL_REVISION
    assert manifest["seeds"] == [42]
    assert manifest["params"] == asdict(config.params)
    assert manifest["poses"] == list(config.poses)
    assert manifest["artifact"]["sha256"] == hashlib.sha256(_glb()).hexdigest()
    metrics = json.loads(result.metrics_path.read_text(encoding="utf-8"))
    assert metrics["cache_hit"] is True
    assert metrics["status"] == "completed"
    log_records = [json.loads(line) for line in result.logs_path.read_text().splitlines()]
    assert [record["phase"] for record in log_records] == ["preflight", "completed"]
    assert events[-1].artifact == result.artifact.glb_path
    assert not list(result.run_dir.glob("*.partial"))


def test_cache_miss_without_confirmation_refuses_before_client_construction(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    constructed = 0

    def forbidden_factory(*_args: Any, **_kwargs: Any) -> FakeClient:
        nonlocal constructed
        constructed += 1
        raise AssertionError("approval must be checked before constructing the SSH client")

    with pytest.raises(SshPodError, match="explicit approval") as caught:
        run_single_view_trial(
            config,
            approval=None,
            client_factory=forbidden_factory,
            run_id_factory=lambda: "trial-no-approval",
        )

    assert caught.value.code == "approval_required"
    assert constructed == 0
    run_dir = config.ssh.local_runs_root / "trial-no-approval"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]["code"] == "approval_required"
    assert json.loads((run_dir / "metrics.json").read_text())["status"] == "failed"
    assert (run_dir / "logs.jsonl").is_file()


def test_confirmed_cache_miss_uses_generate_single_view_and_caches_result(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    approval = ExistingPodUseApproval.grant(config.ssh)
    fake = FakeClient(tmp_path, [])
    received: list[tuple[SshPodConfig, ExistingPodUseApproval | None]] = []

    def factory(
        ssh: SshPodConfig,
        granted: ExistingPodUseApproval | None,
        _sink: Any,
    ) -> FakeClient:
        received.append((ssh, granted))
        return fake

    result = run_single_view_trial(
        config,
        approval=approval,
        client_factory=factory,
        run_id_factory=lambda: "trial-generated",
    )

    assert received == [(config.ssh, approval)]
    assert fake.calls == 1
    assert result.cache_hit is False
    assert result.artifact.glb_path.read_bytes() == _glb(b"generated")
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["transport_manifest"] == str(fake.last_manifest_path)
    assert manifest["cache"]["hit"] is False
    assert manifest["status"] == "completed"


def test_second_call_is_a_cache_hit_and_does_not_consume_a_new_approval(tmp_path: Path) -> None:
    config = _config(tmp_path)
    first_approval = ExistingPodUseApproval.grant(config.ssh)
    first_client = FakeClient(tmp_path, [])
    run_single_view_trial(
        config,
        approval=first_approval,
        client_factory=lambda *_args: first_client,
        run_id_factory=lambda: "trial-first",
    )

    constructed = 0

    def forbidden_factory(*_args: Any) -> FakeClient:
        nonlocal constructed
        constructed += 1
        raise AssertionError

    second = run_single_view_trial(
        config,
        approval=None,
        client_factory=forbidden_factory,
        run_id_factory=lambda: "trial-second",
    )

    assert second.cache_hit is True
    assert constructed == 0


def test_corrupted_cache_blocks_before_client_and_writes_failure_evidence(
    tmp_path: Path,
) -> None:
    config = _config(tmp_path)
    key = _cache_artifact(config)
    config.cache.directory_for(key).joinpath("output.glb").write_bytes(b"tampered")
    constructed = 0

    def forbidden_factory(*_args: Any) -> FakeClient:
        nonlocal constructed
        constructed += 1
        raise AssertionError("corrupt cache must never fall back to a paid job")

    with pytest.raises(SshPodError) as captured:
        run_single_view_trial(
            config,
            approval=ExistingPodUseApproval.grant(config.ssh),
            client_factory=forbidden_factory,
            run_id_factory=lambda: "trial-corrupt-cache",
        )
    assert captured.value.code == "cache_corruption"
    assert constructed == 0
    manifest = json.loads(
        (config.ssh.local_runs_root / "trial-corrupt-cache" / "manifest.json").read_text()
    )
    assert manifest["status"] == "failed"
    assert manifest["error"]["code"] == "cache_corruption"


def test_remote_cancellation_is_recorded_as_cancelled_not_failed(tmp_path: Path) -> None:
    config = _config(tmp_path)

    class CancelledClient:
        def run(self, _payload: dict[str, Any]) -> dict[str, Any]:
            raise SshPodError(
                "remote worker acknowledged cancellation",
                code="remote_cancelled",
            )

    with pytest.raises(SshPodError) as captured:
        run_single_view_trial(
            config,
            approval=ExistingPodUseApproval.grant(config.ssh),
            client_factory=lambda *_args: CancelledClient(),
            run_id_factory=lambda: "trial-cancelled",
        )

    assert captured.value.code == "remote_cancelled"
    run_dir = config.ssh.local_runs_root / "trial-cancelled"
    manifest = json.loads((run_dir / "manifest.json").read_text(encoding="utf-8"))
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8"))
    records = [json.loads(line) for line in (run_dir / "logs.jsonl").read_text().splitlines()]
    assert manifest["status"] == "cancelled"
    assert metrics["status"] == "cancelled"
    assert records[-1]["phase"] == "cancelled"


def test_preflight_rejects_oversized_image_before_cache_or_network(tmp_path: Path) -> None:
    original = _config(tmp_path)
    original.image_path.write_bytes(b"123456789")
    config = replace(
        original,
        ssh=replace(original.ssh, max_input_bytes=8),
    )

    with pytest.raises(SshPodError) as captured:
        preflight_single_view(config)

    assert captured.value.code == "invalid_request"
    assert not config.cache.root.exists()


def test_invalid_image_is_rejected_before_remote_client_construction(tmp_path: Path) -> None:
    config = _config(tmp_path)
    config.image_path.write_bytes(b"not an image")
    constructed = 0

    def forbidden_factory(*_args: Any, **_kwargs: Any) -> FakeClient:
        nonlocal constructed
        constructed += 1
        raise AssertionError("invalid input must not construct an SSH client")

    with pytest.raises(SshPodError) as captured:
        run_single_view_trial(
            config,
            approval=ExistingPodUseApproval.grant(config.ssh),
            client_factory=forbidden_factory,
            run_id_factory=lambda: "trial-invalid-image",
        )

    assert captured.value.code == "invalid_request"
    assert constructed == 0
    assert not config.cache.root.exists()


@pytest.mark.parametrize("max_pixels", [1, 0])
def test_decompression_bomb_warning_and_error_are_invalid_requests(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    max_pixels: int,
) -> None:
    config = _config(tmp_path)
    config.image_path.write_bytes(IMAGE_2X1)
    monkeypatch.setattr(Image, "MAX_IMAGE_PIXELS", max_pixels)

    with pytest.raises(SshPodError) as captured:
        preflight_single_view(config)

    assert captured.value.code == "invalid_request"
    assert not config.cache.root.exists()
