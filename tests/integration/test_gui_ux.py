"""Offscreen usability and accessibility checks for the experiment window."""

from __future__ import annotations

import os
import sys
import time
from collections.abc import Generator
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QApplication

from pixaboost.gui.main_window import MainWindow
from pixaboost.gui.model import CommandSpec, RunResult, RunState
from pixaboost.gui.theme import ACCENT, STYLESHEET


@pytest.fixture(scope="module")
def app() -> Generator[QApplication, None, None]:
    instance = QApplication.instance() or QApplication([])
    yield instance
    instance.processEvents()


@pytest.fixture
def window(app: QApplication, tmp_path: Path) -> Generator[MainWindow, None, None]:
    command = CommandSpec(
        key="fixture",
        label="Fixture",
        description="Essai local déterministe.",
        program=sys.executable,
        arguments=("-c", "pass"),
        working_directory=tmp_path,
    )
    instance = MainWindow(commands=(command,), repo_root=tmp_path)
    instance.show()
    app.processEvents()
    yield instance
    instance.close()


def test_window_remains_usable_at_800_by_500(app: QApplication, window: MainWindow) -> None:
    window.resize(800, 500)
    app.processEvents()
    assert window.minimumWidth() <= 800
    assert window.minimumHeight() <= 500
    assert window.controls_scroll.viewport().height() > 0
    assert window.start_button.isVisible()
    assert window.tabs.isVisible()
    window.controls_scroll.ensureWidgetVisible(window.gpu_button)
    app.processEvents()
    assert window.gpu_button.isVisible()


def test_interactive_controls_expose_accessible_help(window: MainWindow) -> None:
    controls = (
        window.command_combo,
        window.start_button,
        window.cancel_button,
        window.gpu_button,
        window.command_value,
        window.progress_bar,
        window.log_view,
        window.artifact_list,
        window.refresh_button,
    )
    assert all(control.accessibleName().strip() for control in controls)
    assert all(control.accessibleDescription().strip() for control in controls)


def test_theme_has_contrasted_accent_disabled_primary_and_focus() -> None:
    assert ACCENT == "#3b73d1"
    assert "QPushButton#primary:disabled" in STYLESHEET
    assert "QPushButton:focus" in STYLESHEET
    assert "QTreeWidget:focus" in STYLESHEET


def test_artifact_inventory_has_an_explicit_empty_state(window: MainWindow) -> None:
    assert window.artifact_list.topLevelItemCount() == 1
    item = window.artifact_list.topLevelItem(0)
    assert item is not None
    assert "Aucun artefact" in item.text(1)
    assert not bool(item.flags() & Qt.ItemFlag.ItemIsSelectable)


def test_completion_persists_result_and_selects_its_artifact(
    app: QApplication, window: MainWindow, tmp_path: Path
) -> None:
    artifact = tmp_path / "artifacts" / "run-1" / "mesh.glb"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"glTF-result")
    manifest = artifact.parent / "manifest.json"
    manifest.write_text('{"schema_version": 1}', encoding="utf-8")
    result = RunResult(
        state=RunState.SUCCEEDED,
        exit_code=0,
        duration_seconds=1.25,
        artifacts=(artifact, manifest),
    )
    window._on_completed(result)
    app.processEvents()
    assert "réussi" in window.result_banner.text().lower()
    assert "2 artefacts" in window.result_banner.text()
    current = window.artifact_list.currentItem()
    assert current is not None
    assert current.data(1, Qt.ItemDataRole.UserRole) == str(artifact.resolve())
    selected_paths = {
        item.data(1, Qt.ItemDataRole.UserRole) for item in window.artifact_list.selectedItems()
    }
    assert selected_paths == {str(artifact.resolve()), str(manifest.resolve())}


def test_structured_telemetry_is_not_dumped_as_raw_json(
    app: QApplication, window: MainWindow
) -> None:
    window._on_log_line("stdout", 'PIXABOOST_EVENT {"phase":"essai","progress":1.0}')
    app.processEvents()
    assert "PIXABOOST_EVENT" not in window.log_view.toPlainText()


def test_log_updates_are_batched_and_keep_the_block_limit(
    app: QApplication, window: MainWindow
) -> None:
    started_at = time.monotonic()
    for index in range(5_000):
        window._on_log_line("stdout", f"line {index}")
    enqueue_duration = time.monotonic() - started_at
    assert window.log_view.document().blockCount() < 100
    deadline = time.monotonic() + 2.0
    while "line 4999" not in window.log_view.toPlainText() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.01)
    assert enqueue_duration < 1.0
    assert "line 4999" in window.log_view.toPlainText()
    assert window.log_view.document().blockCount() <= 2_500
