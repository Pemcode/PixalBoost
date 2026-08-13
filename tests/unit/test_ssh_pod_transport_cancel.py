"""Cancellation ordering and resource ownership at the Paramiko PTY boundary."""

from __future__ import annotations

import re
import threading
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from pixaboost.backends.ssh_pod import ParamikoPtySession, SshPodConfig, SshPodError

REVISION = "cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af"


class BlockingCancelChannel:
    """Hold the transport send lock while Ctrl-C is being written."""

    def __init__(self) -> None:
        self.cancel_send_started = threading.Event()
        self.release_cancel_send = threading.Event()
        self.sent: list[str] = []
        self.closed = False

    def sendall(self, command: str) -> None:
        self.sent.append(command)
        if command == "\x03":
            self.cancel_send_started.set()
            assert self.release_cancel_send.wait(timeout=2.0)

    def recv_ready(self) -> bool:
        return False

    def recv(self, _size: int) -> bytes:
        return b""

    def close(self) -> None:
        self.closed = True


class FramedChannel:
    """Return a deterministic non-zero result for the initial stty command."""

    def __init__(self) -> None:
        self.pending = b""
        self.closed = False
        self.timeout: float | None = None

    def settimeout(self, value: float) -> None:
        self.timeout = value

    def sendall(self, command: str) -> None:
        begin = re.search(r"__PIXABOOST_BEGIN_[0-9a-f]+__", command)
        end = re.search(r"__PIXABOOST_END_[0-9a-f]+__:", command)
        assert begin is not None and end is not None
        self.pending = begin.group().encode() + b"\r\n" + end.group().encode() + b"1\r\n"

    def recv_ready(self) -> bool:
        return bool(self.pending)

    def recv(self, _size: int) -> bytes:
        payload, self.pending = self.pending, b""
        return payload

    def close(self) -> None:
        self.closed = True


class FakeSshClient:
    def __init__(self, *, channel: FramedChannel | None = None, fail_invoke: bool = False) -> None:
        self.channel = channel
        self.fail_invoke = fail_invoke
        self.closed = False

    def load_host_keys(self, _path: str) -> None:
        pass

    def set_missing_host_key_policy(self, _policy: object) -> None:
        pass

    def connect(self, **_kwargs: Any) -> None:
        pass

    def invoke_shell(self, **_kwargs: Any) -> FramedChannel:
        if self.fail_invoke:
            raise RuntimeError("invoke shell failed")
        assert self.channel is not None
        return self.channel

    def close(self) -> None:
        self.closed = True


def _settings(tmp_path: Path) -> SshPodConfig:
    key = tmp_path / "pod-key"
    known_hosts = tmp_path / "known-hosts"
    key.write_text("private key placeholder", encoding="utf-8")
    known_hosts.write_text("host ssh-ed25519 public-key", encoding="utf-8")
    return SshPodConfig(
        host="ssh.runpod.io",
        username="existing-pod-user",
        private_key_path=key,
        known_hosts_path=known_hosts,
        expected_pixal3d_sha=REVISION,
        project_git_sha="1" * 40,
    )


def _paramiko(client: FakeSshClient) -> object:
    return SimpleNamespace(SSHClient=lambda: client, RejectPolicy=lambda: object())


def test_cancel_wins_send_lock_and_prevents_the_next_command() -> None:
    session = ParamikoPtySession()
    channel = BlockingCancelChannel()
    session._channel = channel
    errors: list[BaseException] = []

    cancel_thread = threading.Thread(target=lambda: session.cancel("run"))
    cancel_thread.start()
    assert channel.cancel_send_started.wait(timeout=1.0)

    def execute() -> None:
        try:
            session._execute("echo must-not-run", timeout_seconds=1.0, max_output_bytes=1024)
        except BaseException as error:
            errors.append(error)

    execute_thread = threading.Thread(target=execute)
    execute_thread.start()
    channel.release_cancel_send.set()
    cancel_thread.join(timeout=2.0)
    execute_thread.join(timeout=2.0)

    assert not cancel_thread.is_alive()
    assert not execute_thread.is_alive()
    assert channel.sent == ["\x03"]
    assert len(errors) == 1
    assert isinstance(errors[0], SshPodError)
    assert errors[0].code == "remote_cancelled"


def test_invoke_shell_failure_closes_the_created_ssh_client(tmp_path: Path) -> None:
    client = FakeSshClient(fail_invoke=True)
    session = ParamikoPtySession(paramiko_loader=lambda: _paramiko(client))

    with pytest.raises(RuntimeError, match="invoke shell failed"):
        session.connect(_settings(tmp_path))

    assert client.closed is True
    assert session._client is None
    assert session._channel is None


def test_stty_failure_closes_channel_and_client(tmp_path: Path) -> None:
    channel = FramedChannel()
    client = FakeSshClient(channel=channel)
    session = ParamikoPtySession(paramiko_loader=lambda: _paramiko(client))

    with pytest.raises(SshPodError, match="configure PTY"):
        session.connect(_settings(tmp_path))

    assert channel.closed is True
    assert client.closed is True
    assert session._client is None
    assert session._channel is None


def test_pty_send_timeout_is_classified_instead_of_blocking_forever() -> None:
    class TimingOutChannel(BlockingCancelChannel):
        def sendall(self, _command: str) -> None:
            raise TimeoutError("write stalled")

    session = ParamikoPtySession()
    session._channel = TimingOutChannel()

    with pytest.raises(SshPodError) as captured:
        session._execute(
            "echo bounded",
            timeout_seconds=1.0,
            max_output_bytes=1024,
        )

    assert captured.value.code == "transport_timeout"
