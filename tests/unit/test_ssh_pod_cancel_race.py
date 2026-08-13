"""Deterministic cancellation races before SSH session publication."""

from __future__ import annotations

import base64
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any, NoReturn

from pixaboost.backends.ssh_pod import (
    CancelState,
    ExistingPodUseApproval,
    SshPodClient,
    SshPodConfig,
    SshPodError,
)

REVISION = "cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af"


class BlockingConnectSession:
    """Session boundary whose connection can be paused deterministically."""

    def __init__(self, *, block_connect: bool) -> None:
        self.block_connect = block_connect
        self.connect_started = threading.Event()
        self.release_connect = threading.Event()
        self.connected = False
        self.closed = False

    def connect(self, _config: SshPodConfig) -> None:
        self.connect_started.set()
        if self.block_connect:
            assert self.release_connect.wait(timeout=2.0)
        self.connected = True

    def remote_revision(self) -> str:
        raise AssertionError("cancelled connection must not reach remote preflight")

    def upload_bytes(
        self,
        _remote_path: str,
        _payload: bytes,
        *,
        expected_sha256: str,
        max_bytes: int,
    ) -> NoReturn:
        del expected_sha256, max_bytes
        raise AssertionError("cancelled connection must not upload")

    def run_worker(
        self,
        _worker_path: str,
        _request: dict[str, Any],
        _frame_sink: Callable[[dict[str, Any]], None],
    ) -> NoReturn:
        raise AssertionError("cancelled connection must not run inference")

    def download_bytes(
        self,
        _remote_path: str,
        *,
        expected_sha256: str,
        expected_size: int,
        max_bytes: int,
    ) -> NoReturn:
        del expected_sha256, expected_size, max_bytes
        raise AssertionError("cancelled connection must not download")

    def cancel(self, _run_id: str) -> bool:
        raise AssertionError("an unpublished session must not receive cancellation")

    def close(self) -> None:
        self.closed = True


def _settings(tmp_path: Path) -> SshPodConfig:
    return SshPodConfig(
        host="ssh.runpod.io",
        username="existing-pod-user",
        private_key_path=tmp_path / "pod-key",
        known_hosts_path=tmp_path / "known-hosts",
        expected_pixal3d_sha=REVISION,
        project_git_sha="1" * 40,
        local_runs_root=tmp_path / "runs",
    )


def _payload() -> dict[str, Any]:
    return {
        "image": base64.b64encode(b"\x89PNG\r\n\x1a\nimage").decode("ascii"),
        "seed": 42,
        "resolution": 1024,
        "low_vram": True,
        "fov": -1.0,
    }


def _client(
    settings: SshPodConfig,
    factory: Callable[[], BlockingConnectSession],
    run_id: str,
) -> SshPodClient:
    return SshPodClient(
        settings,
        approval=ExistingPodUseApproval.grant(settings, clock=lambda: 10.0),
        session_factory=factory,
        run_id_factory=lambda: run_id,
        clock=lambda: 10.0,
    )


def _run_in_thread(client: SshPodClient) -> tuple[threading.Thread, list[BaseException]]:
    errors: list[BaseException] = []

    def target() -> None:
        try:
            client.run(_payload())
        except BaseException as error:
            errors.append(error)

    thread = threading.Thread(target=target)
    thread.start()
    return thread, errors


def test_cancel_while_session_factory_is_running_stops_before_connect(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session = BlockingConnectSession(block_connect=False)
    factory_started = threading.Event()
    release_factory = threading.Event()

    def factory() -> BlockingConnectSession:
        factory_started.set()
        assert release_factory.wait(timeout=2.0)
        return session

    client = _client(settings, factory, "run-cancel-during-factory")
    thread, errors = _run_in_thread(client)
    assert factory_started.wait(timeout=1.0)
    assert client.cancel() is CancelState.ACKNOWLEDGED
    release_factory.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], SshPodError)
    assert errors[0].code == "remote_cancelled"
    assert session.connected is False
    assert session.closed is True


def test_cancel_while_connecting_aborts_before_session_publication(tmp_path: Path) -> None:
    settings = _settings(tmp_path)
    session = BlockingConnectSession(block_connect=True)
    client = _client(settings, lambda: session, "run-cancel-during-connect")
    thread, errors = _run_in_thread(client)
    assert session.connect_started.wait(timeout=1.0)
    assert client.cancel() is CancelState.ACKNOWLEDGED
    session.release_connect.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], SshPodError)
    assert errors[0].code == "remote_cancelled"
    assert session.connected is True
    assert session.closed is True


def test_cancel_after_publication_stops_before_upload_or_worker(tmp_path: Path) -> None:
    settings = _settings(tmp_path)

    class BlockingRevisionSession(BlockingConnectSession):
        def __init__(self) -> None:
            super().__init__(block_connect=False)
            self.revision_started = threading.Event()
            self.release_revision = threading.Event()

        def remote_revision(self) -> str:
            self.revision_started.set()
            assert self.release_revision.wait(timeout=2.0)
            return REVISION

        def cancel(self, _run_id: str) -> bool:
            return False

    session = BlockingRevisionSession()
    client = _client(settings, lambda: session, "run-cancel-after-publication")
    thread, errors = _run_in_thread(client)
    assert session.revision_started.wait(timeout=1.0)

    assert client.cancel() is CancelState.UNKNOWN
    session.release_revision.set()
    thread.join(timeout=2.0)

    assert not thread.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], SshPodError)
    assert errors[0].code == "remote_cancelled"
    assert session.closed is True
