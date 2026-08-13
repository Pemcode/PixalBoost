"""Regression test for cancellation before an SSH session exists."""

from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import NoReturn

import pytest

from pixaboost.backends.ssh_pod import (
    CancelState,
    ExistingPodUseApproval,
    SshPodClient,
    SshPodConfig,
    SshPodError,
)

REVISION = "cdbb2bbffbf4e6f298b5f2af3d1d76a8d823d2af"
PROJECT_REVISION = "1" * 40


def test_cancel_before_run_is_latched_and_prevents_session_construction(tmp_path: Path) -> None:
    settings = SshPodConfig(
        host="ssh.runpod.io",
        username="existing-pod-user",
        private_key_path=tmp_path / "pod-key",
        known_hosts_path=tmp_path / "known-hosts",
        expected_pixal3d_sha=REVISION,
        project_git_sha=PROJECT_REVISION,
        local_runs_root=tmp_path / "runs",
    )
    approval = ExistingPodUseApproval.grant(
        settings, valid_for_seconds=60.0, clock=lambda: 10.0
    )
    factory_calls = 0

    def forbidden_session_factory() -> NoReturn:
        nonlocal factory_calls
        factory_calls += 1
        raise AssertionError("a latched cancellation must stop before session construction")

    client = SshPodClient(
        settings,
        approval=approval,
        session_factory=forbidden_session_factory,
        run_id_factory=lambda: "run-cancelled-before-start",
        clock=lambda: 10.0,
    )

    assert client.cancel() is CancelState.ACKNOWLEDGED
    assert client.cancel() is CancelState.ACKNOWLEDGED
    with pytest.raises(SshPodError) as caught:
        client.run(
            {
                "image": base64.b64encode(b"\x89PNG\r\n\x1a\nimage").decode("ascii"),
                "seed": 42,
                "resolution": 1024,
                "low_vram": True,
                "fov": -1.0,
            }
        )

    assert caught.value.code == "remote_cancelled"
    assert factory_calls == 0
    manifest = json.loads(client.last_manifest_path.read_text(encoding="utf-8"))
    assert manifest["status"] == "cancelled"
    assert manifest["cancel_state"] == CancelState.ACKNOWLEDGED.value
    assert manifest["error"]["code"] == "remote_cancelled"
