from __future__ import annotations

import base64
import struct
import threading
from pathlib import Path
from typing import Any

from pixaboost.backends.cache import ArtifactCache, cache_key
from pixaboost.backends.pixal3d import GenerationParams
from pixaboost.backends.ssh_pod import CancelState, SshPodError
from pixaboost.gui.model import RunState
from pixaboost.gui.remote_trial import RemoteTrialRequest
from pixaboost.gui.single_view_adapter import ExistingPodSingleViewRunner

PNG_1X1 = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNk+A8AAQUBAScY42Y"
    "AAAAASUVORK5CYII="
)


def request(tmp_path: Path) -> RemoteTrialRequest:
    image = tmp_path / "part.png"
    key = tmp_path / "id_ed25519"
    known_hosts = tmp_path / "known_hosts"
    image.write_bytes(PNG_1X1)
    key.write_text("key", encoding="utf-8")
    known_hosts.write_text("host key", encoding="utf-8")
    return RemoteTrialRequest(
        image_path=image,
        host="pod.example.test",
        username="researcher",
        private_key_path=key,
        known_hosts_path=known_hosts,
        expected_pixal3d_sha="a" * 40,
        project_git_sha="b" * 40,
    )


def valid_glb() -> bytes:
    return b"glTF" + struct.pack("<II", 2, 12)


class FakeClient:
    def __init__(self, revision: str) -> None:
        self.revision = revision
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.cancel_calls = 0

    def run(self, _payload: dict[str, Any]) -> dict[str, Any]:
        self.started.set()
        if self.block:
            assert self.release.wait(3.0)
        glb = valid_glb()
        return {
            "glb_base64": base64.b64encode(glb).decode("ascii"),
            "glb_bytes": len(glb),
            "pixal3d_sha": self.revision,
        }

    def cancel(self) -> CancelState:
        self.cancel_calls += 1
        return CancelState.UNKNOWN


def test_adapter_cache_hit_never_constructs_a_remote_client(tmp_path: Path) -> None:
    trial_request = request(tmp_path)
    params = GenerationParams(resolution=1024)
    key = cache_key(
        image=trial_request.image_path.read_bytes(),
        params={
            "seed": params.seed,
            "resolution": params.resolution,
            "low_vram": params.low_vram,
            "fov": params.fov,
        },
        model_revision=trial_request.expected_pixal3d_sha,
    )
    ArtifactCache(tmp_path / "artifacts").store(key, glb=valid_glb(), metadata={})
    constructed: list[bool] = []
    runner = ExistingPodSingleViewRunner(
        tmp_path,
        project_git_revision="b" * 40,
        client_factory=lambda *_args: constructed.append(True),  # type: ignore[arg-type,func-returns-value]
    )

    assert runner.preflight(trial_request)
    result = runner.run(
        trial_request,
        approve_existing_pod=False,
        event_sink=lambda _event: None,
    )

    assert result.state is RunState.SUCCEEDED
    assert result.cache_hit
    assert constructed == []
    assert result.artifacts[0].read_bytes() == valid_glb()
    assert all(path.is_file() for path in result.artifacts)


def test_adapter_approval_is_one_shot_and_cancel_delegates_to_the_live_client(
    tmp_path: Path,
) -> None:
    trial_request = request(tmp_path)
    client = FakeClient(trial_request.expected_pixal3d_sha)
    client.block = True
    approvals: list[object] = []

    def client_factory(_config: object, approval: object, _sink: object) -> FakeClient:
        approvals.append(approval)
        return client

    runner = ExistingPodSingleViewRunner(
        tmp_path,
        project_git_revision="b" * 40,
        client_factory=client_factory,
    )
    assert not runner.preflight(trial_request)
    results: list[object] = []
    thread = threading.Thread(
        target=lambda: results.append(
            runner.run(
                trial_request,
                approve_existing_pod=True,
                event_sink=lambda _event: None,
            )
        )
    )
    thread.start()
    assert client.started.wait(2.0)

    assert runner.cancel() is CancelState.UNKNOWN
    assert client.cancel_calls == 1
    assert len(approvals) == 1
    assert approvals[0] is not None

    client.release.set()
    thread.join(3.0)
    assert not thread.is_alive()
    assert len(results) == 1


def test_adapter_latches_cancel_before_client_construction(tmp_path: Path) -> None:
    trial_request = request(tmp_path)
    constructed: list[bool] = []
    runner = ExistingPodSingleViewRunner(
        tmp_path,
        project_git_revision=trial_request.project_git_sha,
        client_factory=lambda *_args: constructed.append(True),  # type: ignore[arg-type,func-returns-value]
    )
    assert not runner.preflight(trial_request)

    assert runner.cancel() is CancelState.ACKNOWLEDGED
    result = runner.run(
        trial_request,
        approve_existing_pod=True,
        event_sink=lambda _event: None,
    )

    assert result.state is RunState.CANCELLED
    assert constructed == []


def test_adapter_maps_remote_cancel_error_to_cancelled_result(tmp_path: Path) -> None:
    trial_request = request(tmp_path)

    class CancelledClient:
        def run(self, _payload: dict[str, Any]) -> dict[str, Any]:
            raise SshPodError(
                "transport closed before the result frame",
                code="remote_cancelled",
                cancel_state=CancelState.UNKNOWN,
                remote_terminal=False,
            )

        def cancel(self) -> CancelState:
            return CancelState.ACKNOWLEDGED

    runner = ExistingPodSingleViewRunner(
        tmp_path,
        project_git_revision=trial_request.project_git_sha,
        client_factory=lambda *_args: CancelledClient(),
    )
    assert not runner.preflight(trial_request)

    result = runner.run(
        trial_request,
        approve_existing_pod=True,
        event_sink=lambda _event: None,
    )

    assert result.state is RunState.CANCELLED
    assert result.cancel_state is CancelState.UNKNOWN
    assert result.remote_terminal is False
    assert "Annulation locale, état Pod inconnu" in result.error
