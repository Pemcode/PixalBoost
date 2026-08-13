"""Offscreen PyQt6 integration tests with real child processes (F08)."""

from __future__ import annotations

import json
import os
import sys
import time
from collections.abc import Callable, Generator
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication

from pixaboost.gui.main_window import MainWindow
from pixaboost.gui.model import CommandSpec, RunResult, RunState
from pixaboost.gui.runner import CommandController


@pytest.fixture(scope="module")
def app() -> Generator[QApplication, None, None]:
    instance = QApplication.instance() or QApplication([])
    yield instance
    instance.processEvents()


def wait_until(
    app: QApplication, predicate: Callable[[], bool], *, timeout_seconds: float = 8.0
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return
        time.sleep(0.01)
    pytest.fail("Qt condition did not become true before the timeout")


def process_events_for(app: QApplication, duration_seconds: float) -> None:
    deadline = time.monotonic() + duration_seconds
    while time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)


def python_command(tmp_path: Path, code: str, *, artifact: bool = False) -> CommandSpec:
    required = (tmp_path / "result" / "manifest.json",) if artifact else ()
    return CommandSpec(
        key="fixture",
        label="Fixture",
        description="Deterministic child process used by the GUI integration tests.",
        program=sys.executable,
        arguments=("-u", "-c", code),
        working_directory=tmp_path,
        required_artifacts=required,
    )


def test_controller_streams_a_real_process_and_validates_its_artifact(
    app: QApplication, tmp_path: Path
) -> None:
    code = (
        "import pathlib; "
        "pathlib.Path('result').mkdir(); "
        "pathlib.Path('result/manifest.json').write_text("
        "'{\"schema_version\":1}', encoding='utf-8'); "
        'print(\'PIXABOOST_EVENT {"phase":"benchmark","stage":"rendu",'
        '"progress":0.5,"message":"half"}\', flush=True); '
        "print('complete', flush=True)"
    )
    controller = CommandController()
    logs: list[tuple[str, str]] = []
    events: list[object] = []
    results: list[RunResult] = []
    controller.log_line.connect(lambda stream, line: logs.append((stream, line)))
    controller.telemetry.connect(events.append)
    controller.completed.connect(results.append)

    assert controller.start(python_command(tmp_path, code, artifact=True)) is True
    wait_until(app, lambda: bool(results))

    result = results[0]
    assert result.state is RunState.SUCCEEDED
    assert result.exit_code == 0
    assert result.artifacts == (tmp_path / "result" / "manifest.json",)
    assert any(stream == "stdout" and line == "complete" for stream, line in logs)
    assert any(getattr(event, "progress", None) == 0.5 for event in events)
    assert all("PIXABOOST_EVENT" not in line for _stream, line in logs)
    assert ("stdout", "half") in logs


def test_controller_refuses_a_second_concurrent_command(app: QApplication, tmp_path: Path) -> None:
    controller = CommandController(cancel_grace_milliseconds=100)
    long_command = python_command(
        tmp_path, "import time; print('ready', flush=True); time.sleep(30)"
    )
    other_command = python_command(tmp_path, "print('must not run')")
    results: list[RunResult] = []
    logs: list[str] = []
    controller.completed.connect(results.append)
    controller.log_line.connect(lambda _stream, line: logs.append(line))

    assert controller.start(long_command) is True
    wait_until(app, lambda: "ready" in logs)
    assert controller.start(other_command) is False
    assert controller.cancel() is True
    wait_until(app, lambda: bool(results))
    assert results[0].state is RunState.CANCELLED
    assert "must not run" not in logs


def test_an_old_cancel_deadline_cannot_kill_the_next_command(
    app: QApplication, tmp_path: Path
) -> None:
    controller = CommandController(cancel_grace_milliseconds=250)
    first_results: list[RunResult] = []
    controller.completed.connect(first_results.append)
    first = python_command(tmp_path, "import time; print('first', flush=True); time.sleep(30)")
    assert controller.start(first)
    wait_until(app, lambda: controller.state is RunState.RUNNING)
    assert controller.cancel()
    wait_until(app, lambda: len(first_results) == 1)

    second = python_command(tmp_path, "import time; print('second', flush=True); time.sleep(0.5)")
    assert controller.start(second)
    wait_until(app, lambda: len(first_results) == 2)
    assert first_results[1].state is RunState.SUCCEEDED


def test_cancel_during_starting_never_transitions_back_to_running(
    app: QApplication, tmp_path: Path
) -> None:
    controller = CommandController(cancel_grace_milliseconds=50)
    states: list[RunState] = []
    results: list[RunResult] = []
    cancel_results: list[bool] = []

    def capture_state(state: RunState) -> None:
        states.append(state)
        if state is RunState.STARTING:
            cancel_results.append(controller.cancel())

    controller.state_changed.connect(capture_state)
    controller.completed.connect(results.append)
    assert controller.start(python_command(tmp_path, "import time; time.sleep(30)"))
    wait_until(app, lambda: bool(results))
    assert cancel_results == [True]
    cancelling_index = states.index(RunState.CANCELLING)
    assert RunState.RUNNING not in states[cancelling_index + 1 :]
    assert results[0].state is RunState.CANCELLED


def test_cancel_stops_descendants_of_the_local_process(app: QApplication, tmp_path: Path) -> None:
    marker = tmp_path / "orphan.txt"
    child_code = (
        "import signal,time,pathlib; "
        "signal.signal(signal.SIGTERM, signal.SIG_IGN); time.sleep(0.8); "
        f"pathlib.Path({str(marker)!r}).write_text('orphan')"
    )
    parent_code = (
        "import subprocess,sys,time; "
        f"subprocess.Popen([sys.executable, '-c', {child_code!r}]); "
        "print('child spawned', flush=True); time.sleep(30)"
    )
    controller = CommandController(cancel_grace_milliseconds=100)
    results: list[RunResult] = []
    logs: list[str] = []
    controller.completed.connect(results.append)
    controller.log_line.connect(lambda _stream, line: logs.append(line))
    assert controller.start(python_command(tmp_path, parent_code))
    wait_until(app, lambda: "child spawned" in logs)
    assert controller.cancel()
    wait_until(app, lambda: bool(results))
    process_events_for(app, 1.1)
    assert results[0].state is RunState.CANCELLED
    assert not marker.exists()


def test_a_zero_exit_without_its_required_manifest_is_a_failure(
    app: QApplication, tmp_path: Path
) -> None:
    controller = CommandController()
    results: list[RunResult] = []
    controller.completed.connect(results.append)
    assert controller.start(python_command(tmp_path, "print('no artifact')", artifact=True))
    wait_until(app, lambda: bool(results))
    assert results[0].state is RunState.FAILED
    assert "manifest.json" in results[0].error


def test_a_stale_required_manifest_cannot_validate_a_noop(
    app: QApplication, tmp_path: Path
) -> None:
    artifact = tmp_path / "result" / "manifest.json"
    artifact.parent.mkdir()
    artifact.write_text('{"old": true}', encoding="utf-8")
    controller = CommandController()
    results: list[RunResult] = []
    controller.completed.connect(results.append)
    assert controller.start(python_command(tmp_path, "print('noop')", artifact=True))
    wait_until(app, lambda: bool(results))
    assert results[0].state is RunState.FAILED
    assert "preexistant" in results[0].error


def test_rewriting_identical_manifest_content_is_still_stale(
    app: QApplication, tmp_path: Path
) -> None:
    artifact = tmp_path / "result" / "manifest.json"
    artifact.parent.mkdir()
    artifact.write_text('{"schema_version": 1}', encoding="utf-8")
    code = (
        "import pathlib,time; time.sleep(0.02); "
        "pathlib.Path('result/manifest.json').write_text("
        "'{\"schema_version\": 1}', encoding='utf-8')"
    )
    controller = CommandController()
    results: list[RunResult] = []
    controller.completed.connect(results.append)
    assert controller.start(python_command(tmp_path, code, artifact=True))
    wait_until(app, lambda: bool(results))
    assert results[0].state is RunState.FAILED
    assert "preexistant" in results[0].error


def test_a_required_manifest_must_be_valid_json(app: QApplication, tmp_path: Path) -> None:
    code = (
        "import pathlib; pathlib.Path('result').mkdir(); "
        "pathlib.Path('result/manifest.json').write_text('not json', encoding='utf-8')"
    )
    controller = CommandController()
    results: list[RunResult] = []
    controller.completed.connect(results.append)
    assert controller.start(python_command(tmp_path, code, artifact=True))
    wait_until(app, lambda: bool(results))
    assert results[0].state is RunState.FAILED
    assert "JSON" in results[0].error


def test_an_empty_manifest_object_is_not_valid_evidence(app: QApplication, tmp_path: Path) -> None:
    code = (
        "import pathlib; pathlib.Path('result').mkdir(); "
        "pathlib.Path('result/manifest.json').write_text('{}', encoding='utf-8')"
    )
    controller = CommandController()
    results: list[RunResult] = []
    controller.completed.connect(results.append)
    assert controller.start(python_command(tmp_path, code, artifact=True))
    wait_until(app, lambda: bool(results))
    assert results[0].state is RunState.FAILED
    assert "vide" in results[0].error


def test_stderr_and_nonzero_exit_are_reported(app: QApplication, tmp_path: Path) -> None:
    code = (
        "import sys; print('actionable failure', file=sys.stderr, flush=True); raise SystemExit(7)"
    )
    controller = CommandController()
    results: list[RunResult] = []
    logs: list[tuple[str, str]] = []
    controller.completed.connect(results.append)
    controller.log_line.connect(lambda stream, line: logs.append((stream, line)))
    assert controller.start(python_command(tmp_path, code))
    wait_until(app, lambda: bool(results))
    assert results[0].state is RunState.FAILED
    assert results[0].exit_code == 7
    assert ("stderr", "actionable failure") in logs


def test_malformed_structured_telemetry_invalidates_an_otherwise_green_process(
    app: QApplication, tmp_path: Path
) -> None:
    controller = CommandController()
    results: list[RunResult] = []
    controller.completed.connect(results.append)
    code = "print('PIXABOOST_EVENT {not-json}', flush=True)"
    assert controller.start(python_command(tmp_path, code))
    wait_until(app, lambda: bool(results))
    assert results[0].state is RunState.FAILED
    assert "telemetry" in results[0].error.lower()


def test_controller_runs_the_real_observed_pixaboost_benchmark(
    app: QApplication, tmp_path: Path
) -> None:
    output = tmp_path / "bench"
    command = CommandSpec(
        key="real-benchmark",
        label="Real benchmark",
        description="One real part at low resolution.",
        program=sys.executable,
        arguments=(
            "-u",
            "-m",
            "pixaboost.bench.build",
            "--output",
            str(output),
            "--resolution",
            "16",
            "--part",
            "l_bracket",
            "--events-jsonl",
        ),
        working_directory=tmp_path,
        required_artifacts=(output / "manifest.json",),
    )
    controller = CommandController()
    events: list[object] = []
    results: list[RunResult] = []
    controller.telemetry.connect(events.append)
    controller.completed.connect(results.append)

    assert controller.start(command)
    wait_until(app, lambda: bool(results))

    assert results[0].state is RunState.SUCCEEDED
    assert any(getattr(event, "stage", None) == "rendu" for event in events)
    assert any(getattr(event, "artifact", None) == output / "manifest.json" for event in events)
    manifest = json.loads((output / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parts"] == ["l_bracket"]
    assert manifest["config"]["resolution"] == 16


def test_a_missing_executable_is_a_failed_result_not_a_stuck_starting_state(
    app: QApplication, tmp_path: Path
) -> None:
    command = CommandSpec(
        key="missing",
        label="Missing",
        description="Cannot start.",
        program="pixaboost-command-that-does-not-exist-7f93",
        working_directory=tmp_path,
    )
    controller = CommandController()
    results: list[RunResult] = []
    controller.completed.connect(results.append)
    assert controller.start(command)
    wait_until(app, lambda: bool(results))
    assert results[0].state is RunState.FAILED
    assert results[0].exit_code is None
    assert "demarrer" in results[0].error


def test_the_real_window_runs_the_same_controller_path_offscreen(
    app: QApplication, tmp_path: Path
) -> None:
    code = (
        'print(\'PIXABOOST_EVENT {"phase":"essai","progress":1.0,"message":"done"}\', flush=True)'
    )
    command = python_command(tmp_path, code)
    window = MainWindow(commands=(command,), repo_root=tmp_path)
    window.show()

    assert not window.gpu_button.isEnabled()
    assert "F07" in window.gpu_reason.text()
    window.start_button.click()
    wait_until(app, lambda: window.controller.state.is_terminal)

    assert window.controller.state is RunState.SUCCEEDED
    assert window.state_value.text() == "Réussi"
    assert window.phase_value.text() == "essai"
    assert window.progress_bar.value() == 1000
    assert "PIXABOOST_EVENT" not in window.log_view.toPlainText()
    assert "done" in window.log_view.toPlainText()
    assert command.display_command in window.command_value.text()
    window.close()


def test_real_entrypoint_wires_the_existing_pod_runner_offscreen(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from pixaboost.gui import __main__ as gui_main

    image = tmp_path / "input.png"
    key = tmp_path / "key"
    known = tmp_path / "known_hosts"
    image.write_bytes(b"png")
    key.write_text("key", encoding="utf-8")
    known.write_text("host key", encoding="utf-8")
    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    exit_code = gui_main.main(
        [
            "--smoke-test",
            "--trial-image",
            str(image),
            "--ssh-host",
            "pod.example.test",
            "--ssh-user",
            "researcher",
            "--ssh-key",
            str(key),
            "--ssh-known-hosts",
            str(known),
            "--pixal3d-sha",
            "a" * 40,
        ]
    )
    assert exit_code == 0


def test_a_rejected_second_window_start_preserves_the_active_run_display(
    app: QApplication, tmp_path: Path
) -> None:
    command = python_command(
        tmp_path,
        "import time; print('active-log', flush=True); time.sleep(30)",
    )
    window = MainWindow(commands=(command,), repo_root=tmp_path)
    window.show()
    results: list[RunResult] = []
    window.controller.completed.connect(results.append)

    window.start_selected()
    wait_until(app, lambda: "active-log" in window.log_view.toPlainText())
    started_at = window._started_at
    window.start_selected()

    assert "active-log" in window.log_view.toPlainText()
    assert window._started_at == started_at
    assert window.controller.state is RunState.RUNNING
    assert window.controller.cancel()
    wait_until(app, lambda: bool(results))
    window.close()


@pytest.mark.parametrize(
    ("code", "cancel"),
    [
        ("raise SystemExit(9)", False),
        ("import time; time.sleep(30)", True),
    ],
)
def test_terminal_failure_and_cancellation_stop_the_progress_animation(
    app: QApplication,
    tmp_path: Path,
    code: str,
    cancel: bool,
) -> None:
    window = MainWindow(commands=(python_command(tmp_path, code),), repo_root=tmp_path)
    window.show()
    results: list[RunResult] = []
    window.controller.completed.connect(results.append)
    window.start_selected()
    if cancel:
        wait_until(app, lambda: window.controller.state is RunState.RUNNING)
        assert window.controller.cancel()
    wait_until(app, lambda: bool(results))

    assert results[0].state in {RunState.FAILED, RunState.CANCELLED}
    assert window.progress_bar.minimum() == 0
    assert window.progress_bar.maximum() == 1000
    window.close()


def test_a_window_failed_to_start_result_is_not_overwritten_after_start_returns(
    app: QApplication, tmp_path: Path
) -> None:
    command = CommandSpec(
        key="missing-window",
        label="Missing window executable",
        description="Exercise synchronous QProcess failure delivery.",
        program="pixaboost-window-command-that-does-not-exist-0d71",
        working_directory=tmp_path,
    )
    window = MainWindow(commands=(command,), repo_root=tmp_path)
    window.show()

    window.start_selected()
    wait_until(app, lambda: window.controller.state is RunState.FAILED)

    assert not window._elapsed_timer.isActive()
    assert "Impossible de demarrer" in window.log_view.toPlainText()
    assert window.exit_value.text() == "—"
    assert window.progress_bar.minimum() == 0
    assert window.progress_bar.maximum() == 1000
    window.close()


def test_a_fast_process_cannot_have_its_first_signals_erased_by_ui_reset(
    app: QApplication, tmp_path: Path
) -> None:
    command = python_command(tmp_path, "print('first-and-only-line', flush=True)")
    window = MainWindow(commands=(command,), repo_root=tmp_path)
    window.show()

    window.start_selected()
    wait_until(app, lambda: window.controller.state.is_terminal)

    assert window.controller.state is RunState.SUCCEEDED
    assert "first-and-only-line" in window.log_view.toPlainText()
    assert window.phase_value.text() == "Processus terminé"
    assert not window._elapsed_timer.isActive()
    window.close()
