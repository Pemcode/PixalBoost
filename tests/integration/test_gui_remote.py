from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Generator
from pathlib import Path

import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from pixaboost.backends.ssh_pod import CancelState, SshPodEvent
from pixaboost.gui.main_window import MainWindow
from pixaboost.gui.model import CommandSpec, RunState
from pixaboost.gui.remote_trial import (
    RemoteTrialController,
    RemoteTrialDefaults,
    RemoteTrialRequest,
    RemoteTrialResult,
)


@pytest.fixture
def app() -> Generator[QApplication, None, None]:
    instance = QApplication.instance() or QApplication([])
    assert isinstance(instance, QApplication)
    yield instance
    instance.processEvents()


def wait_until(
    app: QApplication, predicate: Callable[[], bool], *, timeout_seconds: float = 4.0
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.005)
    raise AssertionError("condition did not become true")


class FakeRemoteRunner:
    def __init__(self, *, cache_hit: bool, result: RemoteTrialResult) -> None:
        self.cache_hit = cache_hit
        self.result = result
        self.preflight_calls = 0
        self.run_calls: list[bool] = []
        self.cancel_calls = 0
        self.started = threading.Event()
        self.release = threading.Event()
        self.block = False
        self.preflight_started = threading.Event()
        self.preflight_release = threading.Event()
        self.block_preflight = False
        self.cancel_state = CancelState.ACKNOWLEDGED

    def preflight(self, request: RemoteTrialRequest) -> bool:
        assert request.image_path.is_file()
        self.preflight_calls += 1
        self.preflight_started.set()
        if self.block_preflight:
            assert self.preflight_release.wait(3.0)
        return self.cache_hit

    def run(
        self,
        request: RemoteTrialRequest,
        *,
        approve_existing_pod: bool,
        event_sink: Callable[[object], None],
    ) -> RemoteTrialResult:
        self.run_calls.append(approve_existing_pod)
        self.started.set()
        event_sink(SshPodEvent("trial", 0, "inference", 0.4, "Inférence distante"))
        if self.block:
            assert self.release.wait(3.0)
        return self.result

    def cancel(self) -> CancelState:
        self.cancel_calls += 1
        return self.cancel_state


def local_command(root: Path) -> CommandSpec:
    return CommandSpec(
        key="noop",
        label="Local",
        description="Local",
        program="python",
        arguments=("-c", "print('ok')"),
        working_directory=root,
    )


def remote_defaults(tmp_path: Path) -> RemoteTrialDefaults:
    image = tmp_path / "part.png"
    key = tmp_path / "id_ed25519"
    known_hosts = tmp_path / "known_hosts"
    image.write_bytes(b"png")
    key.write_text("private key placeholder", encoding="utf-8")
    known_hosts.write_text("host ssh-ed25519 public-key", encoding="utf-8")
    return RemoteTrialDefaults(
        image_path=image,
        host="pod.example.test",
        username="researcher",
        private_key_path=key,
        known_hosts_path=known_hosts,
        expected_pixal3d_sha="a" * 40,
        project_git_sha="b" * 40,
    )


def successful_result(tmp_path: Path, *, cache_hit: bool) -> RemoteTrialResult:
    artifact = tmp_path / "artifacts" / "trial" / "output.glb"
    manifest = tmp_path / "runs" / "trial" / "manifest.json"
    artifact.parent.mkdir(parents=True, exist_ok=True)
    manifest.parent.mkdir(parents=True, exist_ok=True)
    artifact.write_bytes(b"glTF-result")
    manifest.write_text('{"schema_version": 1}', encoding="utf-8")
    return RemoteTrialResult(
        state=RunState.SUCCEEDED,
        duration_seconds=0.1,
        artifacts=(artifact, manifest),
        cache_hit=cache_hit,
    )


def test_image_is_selected_with_the_native_file_dialog(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    defaults = remote_defaults(tmp_path)
    selected = tmp_path / "selected.jpg"
    selected.write_bytes(b"jpeg")
    runner = FakeRemoteRunner(cache_hit=True, result=successful_result(tmp_path, cache_hit=True))
    window = MainWindow(
        commands=(local_command(tmp_path),),
        repo_root=tmp_path,
        remote_runner_factory=lambda: runner,
        remote_defaults=defaults,
    )
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: (str(selected), "Images (*.jpg)"),
    )

    window.gpu_browse_button.click()

    assert window.gpu_image_edit.text() == str(selected)
    window.close()


def test_cache_miss_refusal_starts_no_worker_or_remote_run(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = FakeRemoteRunner(cache_hit=False, result=successful_result(tmp_path, cache_hit=False))
    window = MainWindow(
        commands=(local_command(tmp_path),),
        repo_root=tmp_path,
        remote_runner_factory=lambda: runner,
        remote_defaults=remote_defaults(tmp_path),
    )
    monkeypatch.setattr(window, "_confirm_remote_use", lambda _request: False)

    window.gpu_button.click()
    wait_until(app, lambda: runner.preflight_calls == 1)
    wait_until(app, lambda: not window.remote_controller.owns_single_flight)

    assert runner.preflight_calls == 1
    assert runner.run_calls == []
    assert not window.remote_controller.has_worker
    assert window.remote_controller.state is RunState.IDLE
    window.close()


def test_cache_hit_is_free_and_skips_confirmation_then_selects_artifacts(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    result = successful_result(tmp_path, cache_hit=True)
    runner = FakeRemoteRunner(cache_hit=True, result=result)
    window = MainWindow(
        commands=(local_command(tmp_path),),
        repo_root=tmp_path,
        remote_runner_factory=lambda: runner,
        remote_defaults=remote_defaults(tmp_path),
    )
    monkeypatch.setattr(
        window,
        "_confirm_remote_use",
        lambda _request: pytest.fail("cache hit must not request paid-use confirmation"),
    )

    window.gpu_button.click()
    wait_until(app, lambda: window.remote_controller.state.is_terminal)

    assert runner.run_calls == [False]
    assert "cache local" in window.result_banner.text().lower()
    selected = {item.data(1, 256) for item in window.artifact_list.selectedItems()}
    assert str(result.artifacts[0].resolve()) in selected
    assert str(result.artifacts[1].resolve()) in selected
    window.close()


def test_cache_miss_acceptance_streams_telemetry_without_blocking_the_ui(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = FakeRemoteRunner(
        cache_hit=False,
        result=successful_result(tmp_path, cache_hit=False),
    )
    runner.block = True
    window = MainWindow(
        commands=(local_command(tmp_path),),
        repo_root=tmp_path,
        remote_runner_factory=lambda: runner,
        remote_defaults=remote_defaults(tmp_path),
    )
    monkeypatch.setattr(window, "_confirm_remote_use", lambda _request: True)

    window.gpu_button.click()
    wait_until(app, runner.started.is_set)
    ticked: list[bool] = []
    QTimer.singleShot(0, lambda: ticked.append(True))
    wait_until(app, lambda: bool(ticked))

    assert window.remote_controller.state is RunState.RUNNING
    assert not window.start_button.isEnabled()
    assert not window.gpu_button.isEnabled()
    assert "inférence" in window.phase_value.text().lower()
    runner.release.set()
    wait_until(app, lambda: window.remote_controller.state.is_terminal)
    assert runner.run_calls == [True]
    assert "Inférence distante" in window.log_view.toPlainText()
    window.close()


def test_local_preflight_runs_off_the_gui_thread_before_confirmation(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = FakeRemoteRunner(
        cache_hit=False,
        result=successful_result(tmp_path, cache_hit=False),
    )
    runner.block_preflight = True
    window = MainWindow(
        commands=(local_command(tmp_path),),
        repo_root=tmp_path,
        remote_runner_factory=lambda: runner,
        remote_defaults=remote_defaults(tmp_path),
    )
    confirmations: list[bool] = []
    monkeypatch.setattr(
        window,
        "_confirm_remote_use",
        lambda _request: confirmations.append(True) or False,
    )

    window.gpu_button.click()
    wait_until(app, runner.preflight_started.is_set)
    ticked: list[bool] = []
    QTimer.singleShot(0, lambda: ticked.append(True))
    wait_until(app, lambda: bool(ticked))

    assert confirmations == []
    assert runner.run_calls == []
    assert window.remote_controller.state is RunState.STARTING
    assert not window.start_button.isEnabled()
    assert not window.gpu_button.isEnabled()
    assert not window.gpu_image_edit.isEnabled()
    assert not window.gpu_host_edit.isEnabled()
    assert not window.gpu_revision_edit.isEnabled()
    assert not window.gpu_browse_button.isEnabled()

    runner.preflight_release.set()
    wait_until(app, lambda: confirmations == [True])
    wait_until(app, lambda: not window.remote_controller.owns_single_flight)
    assert runner.run_calls == []
    assert window.gpu_image_edit.isEnabled()
    assert window.gpu_host_edit.isEnabled()
    assert window.gpu_revision_edit.isEnabled()
    assert window.gpu_browse_button.isEnabled()
    window.close()


def test_remote_result_error_is_sanitised_before_controller_and_gui_signals(
    app: QApplication, tmp_path: Path
) -> None:
    secret = "TOPSECRET"
    runner = FakeRemoteRunner(
        cache_hit=True,
        result=RemoteTrialResult(
            state=RunState.FAILED,
            duration_seconds=0.1,
            error=f"Authorization: Bearer {secret}",
            cache_hit=True,
        ),
    )
    window = MainWindow(
        commands=(local_command(tmp_path),),
        repo_root=tmp_path,
        remote_runner_factory=lambda: runner,
        remote_defaults=remote_defaults(tmp_path),
    )
    results: list[RemoteTrialResult] = []
    window.remote_controller.completed.connect(results.append)

    window.gpu_button.click()
    wait_until(app, lambda: bool(results))

    assert secret not in results[0].error
    assert "[REDACTED]" in results[0].error
    assert secret not in window.log_view.toPlainText()
    assert secret not in window.result_banner.text()
    window.close()


def test_remote_cancel_calls_the_backend_and_reports_unknown_without_ack(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = FakeRemoteRunner(
        cache_hit=False,
        result=RemoteTrialResult(
            state=RunState.CANCELLED,
            duration_seconds=0.2,
            error="annulation demandée",
            cancel_state=CancelState.UNKNOWN,
            remote_terminal=False,
        ),
    )
    runner.block = True
    runner.cancel_state = CancelState.UNKNOWN
    window = MainWindow(
        commands=(local_command(tmp_path),),
        repo_root=tmp_path,
        remote_runner_factory=lambda: runner,
        remote_defaults=remote_defaults(tmp_path),
    )
    monkeypatch.setattr(window, "_confirm_remote_use", lambda _request: True)
    window.show()
    window.gpu_button.click()
    wait_until(app, runner.started.is_set)

    window.gpu_cancel_button.click()
    wait_until(app, lambda: runner.cancel_calls == 1)
    wait_until(app, lambda: "unknown" in window.phase_value.text().lower())

    assert "annulation locale, état pod inconnu" in window.phase_value.text().lower()
    runner.release.set()
    wait_until(app, lambda: window.remote_controller.state.is_terminal)
    wait_until(app, lambda: not window.remote_controller.has_pending_workers)
    assert window.isVisible()
    assert "annulation locale, état pod inconnu" in window.result_banner.text().lower()
    window.close()


def test_cancel_button_with_ack_does_not_close_the_window(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = FakeRemoteRunner(
        cache_hit=False,
        result=RemoteTrialResult(
            state=RunState.CANCELLED,
            duration_seconds=0.2,
            cancel_state=CancelState.UNKNOWN,
            remote_terminal=False,
        ),
    )
    runner.block = True
    runner.cancel_state = CancelState.ACKNOWLEDGED
    window = MainWindow(
        commands=(local_command(tmp_path),),
        repo_root=tmp_path,
        remote_runner_factory=lambda: runner,
        remote_defaults=remote_defaults(tmp_path),
    )
    monkeypatch.setattr(window, "_confirm_remote_use", lambda _request: True)
    window.show()
    window.gpu_button.click()
    wait_until(app, runner.started.is_set)

    window.gpu_cancel_button.click()
    wait_until(app, lambda: runner.cancel_calls == 1)
    runner.release.set()
    wait_until(app, lambda: window.remote_controller.state.is_terminal)
    app.processEvents()

    assert window.isVisible()
    window.close()


def test_controller_retains_single_flight_until_worker_finished(
    app: QApplication, tmp_path: Path
) -> None:
    result = successful_result(tmp_path, cache_hit=True)
    runner = FakeRemoteRunner(cache_hit=True, result=result)
    controller = RemoteTrialController(lambda: runner)
    request = RemoteTrialRequest(
        image_path=remote_defaults(tmp_path).image_path,  # type: ignore[arg-type]
        host="pod.example.test",
        username="researcher",
        private_key_path=tmp_path / "id_ed25519",
        known_hosts_path=tmp_path / "known_hosts",
        expected_pixal3d_sha="a" * 40,
        project_git_sha="b" * 40,
    )
    prepared: list[bool] = []
    controller.prepared.connect(prepared.append)
    assert controller.prepare(request)
    wait_until(app, lambda: prepared == [True])
    observed: list[tuple[bool, bool]] = []
    controller.completed.connect(
        lambda _result: observed.append(
            (controller.has_pending_workers, controller.can_start)
        )
    )

    assert controller.start_prepared(approve_existing_pod=False)
    wait_until(app, lambda: bool(observed))

    assert observed == [(True, False)]
    wait_until(app, lambda: not controller.has_pending_workers)


def test_close_requested_after_terminal_result_waits_for_worker_cleanup(
    app: QApplication, tmp_path: Path
) -> None:
    runner = FakeRemoteRunner(
        cache_hit=True,
        result=successful_result(tmp_path, cache_hit=True),
    )
    window = MainWindow(
        commands=(local_command(tmp_path),),
        repo_root=tmp_path,
        remote_runner_factory=lambda: runner,
        remote_defaults=remote_defaults(tmp_path),
    )
    close_intents: list[bool] = []
    window.remote_controller.completed.connect(
        lambda _result: (window.close(), close_intents.append(window._close_requested))
    )
    window.show()

    window.gpu_button.click()
    wait_until(app, lambda: bool(close_intents))
    assert close_intents == [True]
    wait_until(app, lambda: not window.isVisible())


def test_displayed_remote_command_matches_the_public_cli(tmp_path: Path) -> None:
    defaults = remote_defaults(tmp_path)
    request = RemoteTrialRequest(
        image_path=defaults.image_path,  # type: ignore[arg-type]
        host=defaults.host,
        username=defaults.username,
        private_key_path=defaults.private_key_path,  # type: ignore[arg-type]
        known_hosts_path=defaults.known_hosts_path,  # type: ignore[arg-type]
        expected_pixal3d_sha=defaults.expected_pixal3d_sha,
        project_git_sha=defaults.project_git_sha,
    )

    command = request.display_command_for(approve_existing_pod=True)

    assert "pixaboost reconstruct single-view" in command
    assert "--backend ssh-pod" in command
    assert f"--revision {defaults.expected_pixal3d_sha}" in command
    assert f"--project-sha {defaults.project_git_sha}" in command
    assert "--resolution 1024" in command
    assert "--low-vram" in command
    assert "--confirm-existing-pod-use" in command
    assert "--transport" not in command
    assert "--pixal3d-sha" not in command


def test_gpu_action_is_enabled_only_for_a_complete_existing_file_configuration(
    app: QApplication, tmp_path: Path
) -> None:
    runner = FakeRemoteRunner(cache_hit=True, result=successful_result(tmp_path, cache_hit=True))
    window = MainWindow(
        commands=(local_command(tmp_path),),
        repo_root=tmp_path,
        remote_runner_factory=lambda: runner,
        remote_defaults=remote_defaults(tmp_path),
    )
    assert window.gpu_button.isEnabled()

    window.gpu_revision_edit.setText("not-a-sha")
    assert not window.gpu_button.isEnabled()
    assert "révision" in window.gpu_reason.text().lower()

    window.gpu_revision_edit.setText("b" * 40)
    assert window.gpu_button.isEnabled()
    window.gpu_key_edit.setText(str(tmp_path / "missing-key"))
    assert not window.gpu_button.isEnabled()
    window.close()


def test_close_during_remote_run_never_silently_detaches(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = FakeRemoteRunner(
        cache_hit=False,
        result=RemoteTrialResult(state=RunState.CANCELLED, duration_seconds=0.2),
    )
    runner.block = True
    runner.cancel_state = CancelState.UNKNOWN
    window = MainWindow(
        commands=(local_command(tmp_path),),
        repo_root=tmp_path,
        remote_runner_factory=lambda: runner,
        remote_defaults=remote_defaults(tmp_path),
    )
    monkeypatch.setattr(window, "_confirm_remote_use", lambda _request: True)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    window.show()
    window.gpu_button.click()
    wait_until(app, runner.started.is_set)

    window.close()
    wait_until(app, lambda: runner.cancel_calls == 1)
    wait_until(app, lambda: "unknown" in window.phase_value.text().lower())

    assert window.isVisible()
    runner.release.set()
    wait_until(app, lambda: window.remote_controller.state.is_terminal)
    wait_until(app, lambda: not window.remote_controller.has_pending_workers)
    assert window.isVisible()
    assert "annulation locale, état pod inconnu" in window.result_banner.text().lower()
    window.close()


def test_close_after_unknown_cancel_accepts_authenticated_terminal_result(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = FakeRemoteRunner(
        cache_hit=False,
        result=RemoteTrialResult(
            state=RunState.CANCELLED,
            duration_seconds=0.2,
            cancel_state=CancelState.UNKNOWN,
            remote_terminal=True,
        ),
    )
    runner.block = True
    runner.cancel_state = CancelState.UNKNOWN
    window = MainWindow(
        commands=(local_command(tmp_path),),
        repo_root=tmp_path,
        remote_runner_factory=lambda: runner,
        remote_defaults=remote_defaults(tmp_path),
    )
    monkeypatch.setattr(window, "_confirm_remote_use", lambda _request: True)
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    window.show()
    window.gpu_button.click()
    wait_until(app, runner.started.is_set)

    window.close()
    wait_until(app, lambda: runner.cancel_calls == 1)
    wait_until(app, lambda: "unknown" in window.phase_value.text().lower())
    assert window.isVisible()

    runner.release.set()
    wait_until(app, lambda: not window.isVisible())


def test_environment_and_cli_defaults_merge_without_persisting_approval(
    tmp_path: Path,
) -> None:
    from pixaboost.gui.remote_trial import RemoteTrialDefaults

    defaults = RemoteTrialDefaults.from_environment(
        {
            "PIXABOOST_TRIAL_IMAGE": str(tmp_path / "env.png"),
            "PIXABOOST_SSH_HOST": "env-host",
            "PIXABOOST_SSH_USER": "env-user",
            "PIXABOOST_SSH_KEY_PATH": str(tmp_path / "env-key"),
            "PIXABOOST_SSH_KNOWN_HOSTS": str(tmp_path / "env-known"),
            "PIXABOOST_PIXAL3D_SHA": "a" * 40,
        }
    ).merged(host="cli-host", image_path=tmp_path / "cli.png")

    assert defaults.host == "cli-host"
    assert defaults.image_path == tmp_path / "cli.png"
    assert defaults.username == "env-user"
    assert defaults.expected_pixal3d_sha == "a" * 40
