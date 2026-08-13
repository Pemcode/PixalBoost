"""Contract tests for the existing-pod SSH JobRunner.

No network, GPU, Paramiko installation or RunPod API is involved here.  The
session boundary is injected so the budget and integrity guards are executable
in the free CPU gate.
"""

from __future__ import annotations

import base64
import hashlib
import json
import re
import struct
import threading
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pixaboost.backends.cache import ArtifactCache
from pixaboost.backends.pixal3d import GenerationParams, generate_single_view
from pixaboost.backends.ssh_pod import (
    PROTOCOL_VERSION,
    CancelState,
    ExistingPodUseApproval,
    ParamikoPtySession,
    SshPodClient,
    SshPodConfig,
    SshPodError,
)
from pixaboost.backends.ssh_worker_source import REMOTE_WORKER_SOURCE

REVISION = "cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af"
PROJECT_REVISION = "1" * 40
IMAGE = b"\x89PNG\r\n\x1a\nsmall synthetic render"


def glb(payload: bytes = b"mesh-payload") -> bytes:
    """Return the smallest GLB-shaped byte string needed by this contract."""
    size = 12 + len(payload)
    return b"glTF" + struct.pack("<II", 2, size) + payload


class FakeSession:
    """In-memory implementation of the SSH session boundary."""

    def __init__(
        self,
        *,
        revision: str = REVISION,
        artifact: bytes | None = None,
        acknowledge_cancel: bool = True,
    ) -> None:
        self.revision = revision
        self.artifact = artifact if artifact is not None else glb()
        self.acknowledge_cancel = acknowledge_cancel
        self.connected = False
        self.closed = False
        self.uploads: list[tuple[str, bytes, str, int]] = []
        self.worker_calls: list[dict[str, Any]] = []
        self.cancel_calls: list[str] = []
        self.worker_started = threading.Event()
        self.release_worker = threading.Event()
        self.block_worker = False

    def connect(self, config: SshPodConfig) -> None:
        self.connected = True

    def remote_revision(self) -> str:
        return self.revision

    def upload_bytes(
        self,
        remote_path: str,
        payload: bytes,
        *,
        expected_sha256: str,
        max_bytes: int,
    ) -> None:
        assert len(payload) <= max_bytes
        assert hashlib.sha256(payload).hexdigest() == expected_sha256
        self.uploads.append((remote_path, payload, expected_sha256, max_bytes))

    def run_worker(
        self,
        worker_path: str,
        request: dict[str, Any],
        frame_sink: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        self.worker_calls.append(request)
        self.worker_started.set()
        if self.block_worker:
            assert self.release_worker.wait(timeout=2.0)
        frame_sink(
            {
                "protocol": PROTOCOL_VERSION,
                "kind": "event",
                "run_id": request["run_id"],
                "sequence": 0,
                "phase": "inference",
                "progress": 0.25,
                "message": "Pipeline loaded",
            }
        )
        digest = hashlib.sha256(self.artifact).hexdigest()
        return {
            "protocol": PROTOCOL_VERSION,
            "kind": "result",
            "run_id": request["run_id"],
            "status": "completed",
            "pixal3d_sha": self.revision,
            "seed": request["params"]["seed"],
            "artifact": {
                "path": request["output_path"],
                "bytes": len(self.artifact),
                "sha256": digest,
                "magic": "glTF",
                "version": 2,
            },
        }

    def download_bytes(
        self,
        remote_path: str,
        *,
        expected_sha256: str,
        expected_size: int,
        max_bytes: int,
    ) -> bytes:
        assert len(self.artifact) == expected_size
        assert len(self.artifact) <= max_bytes
        assert hashlib.sha256(self.artifact).hexdigest() == expected_sha256
        return self.artifact

    def cancel(self, run_id: str) -> bool:
        self.cancel_calls.append(run_id)
        return self.acknowledge_cancel

    def close(self) -> None:
        self.closed = True


def config(tmp_path: Path, **overrides: Any) -> SshPodConfig:
    values: dict[str, Any] = {
        "host": "ssh.runpod.io",
        "username": "existing-pod-user",
        "private_key_path": tmp_path / "pod-key",
        "known_hosts_path": tmp_path / "known_hosts",
        "expected_pixal3d_sha": REVISION,
        "project_git_sha": PROJECT_REVISION,
        "local_runs_root": tmp_path / "runs",
    }
    values.update(overrides)
    return SshPodConfig(**values)


def approval_for(settings: SshPodConfig) -> ExistingPodUseApproval:
    return ExistingPodUseApproval.grant(settings, valid_for_seconds=60.0, clock=lambda: 10.0)


def payload() -> dict[str, Any]:
    return {
        "image": base64.b64encode(IMAGE).decode("ascii"),
        "seed": 42,
        "resolution": 1024,
        "low_vram": True,
        "fov": -1.0,
    }


def test_missing_ephemeral_approval_prevents_any_connection(tmp_path: Path) -> None:
    session = FakeSession()
    client = SshPodClient(
        config(tmp_path),
        approval=None,
        session_factory=lambda: session,
        run_id_factory=lambda: "run-no-approval",
        clock=lambda: 10.0,
    )

    with pytest.raises(SshPodError, match="explicit approval"):
        client.run(payload())

    assert session.connected is False
    manifest = json.loads(client.last_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]["code"] == "approval_required"


def test_approval_is_ephemeral_and_cannot_be_reused(tmp_path: Path) -> None:
    settings = config(tmp_path)
    one_shot = approval_for(settings)
    first = FakeSession()
    SshPodClient(
        settings,
        approval=one_shot,
        session_factory=lambda: first,
        run_id_factory=lambda: "run-first",
        clock=lambda: 10.0,
    ).run(payload())

    second = FakeSession()
    client = SshPodClient(
        settings,
        approval=one_shot,
        session_factory=lambda: second,
        run_id_factory=lambda: "run-second",
        clock=lambda: 10.0,
    )
    with pytest.raises(SshPodError, match="already been used"):
        client.run(payload())
    assert second.connected is False


@pytest.mark.parametrize("failure", ["expired", "other-host"])
def test_expired_or_mismatched_approval_prevents_connection(tmp_path: Path, failure: str) -> None:
    settings = config(tmp_path)
    approved_settings = (
        config(tmp_path, host="another-pod.example") if failure == "other-host" else settings
    )
    one_shot = ExistingPodUseApproval.grant(
        approved_settings, valid_for_seconds=1.0, clock=lambda: 10.0
    )
    session = FakeSession()
    client = SshPodClient(
        settings,
        approval=one_shot,
        session_factory=lambda: session,
        run_id_factory=lambda: f"run-{failure}",
        clock=(lambda: 12.0) if failure == "expired" else (lambda: 10.0),
    )

    with pytest.raises(SshPodError, match="approval"):
        client.run(payload())
    assert session.connected is False


def test_revision_mismatch_stops_before_upload_or_inference(tmp_path: Path) -> None:
    settings = config(tmp_path)
    session = FakeSession(revision="0" * 40)
    client = SshPodClient(
        settings,
        approval=approval_for(settings),
        session_factory=lambda: session,
        run_id_factory=lambda: "run-revision-mismatch",
        clock=lambda: 10.0,
    )

    with pytest.raises(SshPodError, match="revision mismatch"):
        client.run(payload())

    assert session.connected is True
    assert session.uploads == []
    assert session.worker_calls == []
    manifest = json.loads(client.last_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["remote"]["pixal3d_sha"] == "0" * 40


def test_success_returns_handler_compatible_output_and_atomic_manifest(tmp_path: Path) -> None:
    settings = config(tmp_path)
    session = FakeSession()
    events = []
    client = SshPodClient(
        settings,
        approval=approval_for(settings),
        session_factory=lambda: session,
        event_sink=events.append,
        run_id_factory=lambda: "run-success",
        clock=lambda: 10.0,
    )

    result = client.run(payload())

    assert base64.b64decode(result["glb_base64"], validate=True) == session.artifact
    assert result["glb_bytes"] == len(session.artifact)
    assert result["seed"] == 42
    assert result["pixal3d_sha"] == REVISION
    assert [event.phase for event in events] == [
        "connecting",
        "preflight",
        "transfer",
        "inference",
        "completed",
    ]
    assert len(session.uploads) == 2, "the versioned worker and input must both be transferred"
    manifest = json.loads(client.last_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "completed"
    assert manifest["git_sha"] == PROJECT_REVISION
    assert manifest["artifact"]["sha256"] == hashlib.sha256(session.artifact).hexdigest()
    assert not list(client.last_manifest_path.parent.glob("*.partial"))


def test_invalid_glb_is_rejected_and_recorded_as_failure(tmp_path: Path) -> None:
    settings = config(tmp_path)
    session = FakeSession(artifact=b"BAD!" + struct.pack("<II", 2, 16) + b"junk")
    client = SshPodClient(
        settings,
        approval=approval_for(settings),
        session_factory=lambda: session,
        run_id_factory=lambda: "run-invalid-glb",
        clock=lambda: 10.0,
    )

    with pytest.raises(SshPodError, match="GLB magic"):
        client.run(payload())

    manifest = json.loads(client.last_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]["code"] == "invalid_artifact"


def test_downloaded_glb_sha_must_match_the_remote_result_frame(tmp_path: Path) -> None:
    settings = config(tmp_path)

    class TamperingSession(FakeSession):
        def download_bytes(
            self,
            remote_path: str,
            *,
            expected_sha256: str,
            expected_size: int,
            max_bytes: int,
        ) -> bytes:
            del remote_path, expected_sha256, expected_size, max_bytes
            return self.artifact[:-1] + bytes([self.artifact[-1] ^ 1])

    session = TamperingSession()
    client = SshPodClient(
        settings,
        approval=approval_for(settings),
        session_factory=lambda: session,
        run_id_factory=lambda: "run-tampered-glb",
        clock=lambda: 10.0,
    )

    with pytest.raises(SshPodError, match="SHA-256 mismatch"):
        client.run(payload())

    manifest = json.loads(client.last_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "failed"
    assert manifest["error"]["code"] == "invalid_artifact"


def test_cancel_is_idempotent_and_unknown_without_remote_ack(tmp_path: Path) -> None:
    settings = config(tmp_path)
    session = FakeSession(acknowledge_cancel=False)
    session.block_worker = True
    client = SshPodClient(
        settings,
        approval=approval_for(settings),
        session_factory=lambda: session,
        run_id_factory=lambda: "run-cancel",
        clock=lambda: 10.0,
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            client.run(payload())
        except BaseException as error:  # pragma: no cover - diagnostic capture
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    assert session.worker_started.wait(timeout=1.0)

    assert client.cancel() is CancelState.UNKNOWN
    assert client.cancel() is CancelState.UNKNOWN
    assert session.cancel_calls == ["run-cancel"]

    session.release_worker.set()
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert len(errors) == 1
    error = errors[0]
    assert isinstance(error, SshPodError)
    assert error.code == "remote_cancelled"
    assert error.cancel_state is CancelState.UNKNOWN
    assert error.remote_terminal is True
    manifest = json.loads(client.last_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
    assert manifest["cancel_state"] == CancelState.UNKNOWN
    assert manifest["remote_terminal_observed"] is True


@pytest.mark.parametrize("transport_kind", ["timeout", "eof"])
def test_cancel_unknown_survives_transport_end_without_result_frame(
    tmp_path: Path,
    transport_kind: str,
) -> None:
    settings = config(tmp_path)

    class InterruptedSession(FakeSession):
        def run_worker(
            self,
            worker_path: str,
            request: dict[str, Any],
            frame_sink: Callable[[dict[str, Any]], None],
        ) -> dict[str, Any]:
            del worker_path, request, frame_sink
            self.worker_started.set()
            assert self.release_worker.wait(timeout=2.0)
            if transport_kind == "timeout":
                raise SshPodError("result frame timed out", code="transport_timeout")
            raise EOFError("transport closed before the result frame")

    session = InterruptedSession(acknowledge_cancel=False)
    client = SshPodClient(
        settings,
        approval=approval_for(settings),
        session_factory=lambda: session,
        run_id_factory=lambda: f"run-cancel-{transport_kind}",
        clock=lambda: 10.0,
    )
    errors: list[BaseException] = []

    def run() -> None:
        try:
            client.run(payload())
        except BaseException as error:  # pragma: no cover - diagnostic capture
            errors.append(error)

    thread = threading.Thread(target=run)
    thread.start()
    assert session.worker_started.wait(timeout=1.0)
    assert client.cancel() is CancelState.UNKNOWN
    session.release_worker.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    error = errors[0]
    assert isinstance(error, SshPodError)
    assert error.code == "remote_cancelled"
    assert error.cancel_state is CancelState.UNKNOWN
    assert error.remote_terminal is False
    manifest = json.loads(client.last_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
    assert manifest["cancel_state"] == CancelState.UNKNOWN
    assert manifest["remote_terminal_observed"] is False


def test_input_limit_is_enforced_before_connection(tmp_path: Path) -> None:
    settings = config(tmp_path, max_input_bytes=3)
    session = FakeSession()
    client = SshPodClient(
        settings,
        approval=approval_for(settings),
        session_factory=lambda: session,
        run_id_factory=lambda: "run-too-large",
        clock=lambda: 10.0,
    )

    with pytest.raises(SshPodError, match="input image"):
        client.run(payload())
    assert session.connected is False


class FakePtyChannel:
    """Echo the command before returning its real framed response."""

    def __init__(self) -> None:
        self.pending = b""
        self.closed = False
        self.sent: list[str] = []
        self.timeout: float | None = None

    def settimeout(self, value: float) -> None:
        self.timeout = value

    def send(self, command: str) -> None:
        self.sent.append(command)
        if command == "\x03":
            return
        begin = re.search(r"__PIXABOOST_BEGIN_[0-9a-f]+__", command)
        end = re.search(r"__PIXABOOST_END_[0-9a-f]+__:", command)
        assert begin is not None and end is not None
        # The echoed source contains both marker strings followed by shell
        # syntax. Only the real marker lines below may delimit the response.
        self.pending = (
            command.encode()
            + b"\r\n"
            + begin.group().encode()
            + b"\r\n"
            + end.group().encode()
            + b"0\r\n"
        )

    def sendall(self, command: str) -> None:
        self.send(command)

    def recv_ready(self) -> bool:
        return bool(self.pending)

    def recv(self, _size: int) -> bytes:
        result, self.pending = self.pending, b""
        return result

    def close(self) -> None:
        self.closed = True


class FakeSshClient:
    def __init__(self) -> None:
        self.channel = FakePtyChannel()
        self.loaded_host_keys: str | None = None
        self.policy: object | None = None
        self.connect_kwargs: dict[str, Any] | None = None
        self.invoke_shell_calls = 0

    def load_host_keys(self, path: str) -> None:
        self.loaded_host_keys = path

    def set_missing_host_key_policy(self, policy: object) -> None:
        self.policy = policy

    def connect(self, **kwargs: Any) -> None:
        self.connect_kwargs = kwargs

    def invoke_shell(self, **_kwargs: Any) -> FakePtyChannel:
        self.invoke_shell_calls += 1
        return self.channel

    def exec_command(self, *_args: Any, **_kwargs: Any) -> None:
        raise AssertionError("exec_command must never be used through the RunPod PTY gateway")

    def close(self) -> None:
        pass


def test_paramiko_transport_rejects_unknown_hosts_and_uses_only_a_pty(tmp_path: Path) -> None:
    settings = config(tmp_path)
    settings.private_key_path.write_text("fake", encoding="utf-8")
    settings.known_hosts_path.write_text("fake", encoding="utf-8")
    ssh_client = FakeSshClient()

    class RejectPolicy:
        pass

    module = SimpleNamespace(SSHClient=lambda: ssh_client, RejectPolicy=RejectPolicy)
    session = ParamikoPtySession(paramiko_loader=lambda: module)
    session.connect(settings)

    assert ssh_client.loaded_host_keys == str(settings.known_hosts_path)
    assert isinstance(ssh_client.policy, RejectPolicy)
    assert ssh_client.connect_kwargs is not None
    assert ssh_client.connect_kwargs["look_for_keys"] is False
    assert ssh_client.connect_kwargs["allow_agent"] is False
    assert ssh_client.invoke_shell_calls == 1
    assert ssh_client.channel.timeout == settings.connect_timeout_seconds
    assert "stty -echo" in ssh_client.channel.sent[0]


def test_paramiko_cancel_latch_prevents_the_next_pty_command(tmp_path: Path) -> None:
    settings = config(tmp_path)
    session = ParamikoPtySession()
    channel = FakePtyChannel()
    session._config = settings
    session._channel = channel

    assert session.cancel("run") is False
    with pytest.raises(SshPodError) as captured:
        session._execute(
            "python3 expensive-worker.py",
            timeout_seconds=1.0,
            max_output_bytes=4096,
        )

    assert captured.value.code == "remote_cancelled"
    assert channel.sent == ["\x03"]


def test_versioned_remote_worker_is_valid_python_310_source() -> None:
    compile(REMOTE_WORKER_SOURCE, "ssh_worker.py", "exec")
    assert "subprocess.run" in REMOTE_WORKER_SOURCE
    assert '["git", "-C", "/opt/pixal3d", "rev-parse", "HEAD"]' in REMOTE_WORKER_SOURCE
    assert '"status", "--porcelain", "--untracked-files=all"' in REMOTE_WORKER_SOURCE


def test_remote_revision_reads_the_checkout_not_an_environment_hint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = config(tmp_path)
    session = ParamikoPtySession()
    session._config = settings
    commands: list[str] = []

    def execute(
        command: str,
        *,
        timeout_seconds: float,
        max_output_bytes: int,
        output_sink: Callable[[bytes], None] | None = None,
    ) -> object:
        del timeout_seconds, max_output_bytes, output_sink
        commands.append(command)
        return SimpleNamespace(output=(REVISION + "\n").encode(), exit_code=0)

    monkeypatch.setattr(session, "_execute", execute)
    assert session.remote_revision() == REVISION
    assert commands == [
        "git -C /opt/pixal3d rev-parse HEAD && "
        'test -z "$(git -C /opt/pixal3d status --porcelain --untracked-files=all)"'
    ]


def test_remote_revision_refuses_a_dirty_checkout(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    session = ParamikoPtySession()
    session._config = config(tmp_path)

    def execute(*_args: Any, **_kwargs: Any) -> object:
        return SimpleNamespace(output=(REVISION + "\n").encode(), exit_code=1)

    monkeypatch.setattr(session, "_execute", execute)
    with pytest.raises(SshPodError, match="clean remote Pixal3D checkout"):
        session.remote_revision()


def test_remote_cancellation_is_recorded_as_cancelled_not_failed(tmp_path: Path) -> None:
    settings = config(tmp_path)

    class CancelledSession(FakeSession):
        def run_worker(
            self,
            worker_path: str,
            request: dict[str, Any],
            frame_sink: Callable[[dict[str, Any]], None],
        ) -> dict[str, Any]:
            del worker_path, frame_sink
            return {
                "protocol": PROTOCOL_VERSION,
                "kind": "result",
                "run_id": request["run_id"],
                "status": "cancelled",
                "pixal3d_sha": self.revision,
                "error": "remote inference interrupted",
            }

    session = CancelledSession()
    client = SshPodClient(
        settings,
        approval=approval_for(settings),
        session_factory=lambda: session,
        run_id_factory=lambda: "run-cancelled-ack",
        clock=lambda: 10.0,
    )
    with pytest.raises(SshPodError) as captured:
        client.run(payload())
    assert captured.value.code == "remote_cancelled"
    assert captured.value.cancel_state is CancelState.ACKNOWLEDGED
    assert captured.value.remote_terminal is True
    manifest = json.loads(client.last_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
    assert manifest["cancel_state"] == CancelState.ACKNOWLEDGED
    assert manifest["remote_terminal_observed"] is True


def test_client_plugs_into_generate_single_view_and_second_call_hits_cache(
    tmp_path: Path,
) -> None:
    settings = config(tmp_path)
    session = FakeSession()
    client = SshPodClient(
        settings,
        approval=approval_for(settings),
        session_factory=lambda: session,
        run_id_factory=lambda: "run-generate-single-view",
        clock=lambda: 10.0,
    )
    cache = ArtifactCache(tmp_path / "artifacts")
    arguments = {
        "image": IMAGE,
        "params": GenerationParams(seed=42, resolution=1024, low_vram=True),
        "client": client,
        "cache": cache,
        "model_revision": REVISION,
    }

    first = generate_single_view(**arguments)
    second = generate_single_view(**arguments)

    assert first.key == second.key
    assert first.glb_path.read_bytes() == session.artifact
    assert len(session.worker_calls) == 1
