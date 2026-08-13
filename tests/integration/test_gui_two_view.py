"""Offscreen behaviour of the two-view panel (F15).

The trial is substituted: what is under test is the panel -- that the button
exists, that it refuses incomplete or nonsensical input before spending
anything, that the run happens off the GUI thread, and that a weak result is
reported as weak rather than as a GLB.
"""

from __future__ import annotations

import os
import time
from collections.abc import Generator
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PyQt6.QtWidgets import QApplication, QFileDialog

from pixaboost.gui.two_view_view import TwoViewPanel
from pixaboost.trials.two_view import TwoViewConfig, TwoViewResult


@pytest.fixture(scope="module")
def app() -> Generator[QApplication, None, None]:
    instance = QApplication.instance() or QApplication([])
    yield instance
    instance.processEvents()


def photos(tmp_path: Path) -> tuple[Path, Path]:
    front, back = tmp_path / "front.jpg", tmp_path / "back.jpg"
    front.write_bytes(b"not really a jpeg")
    back.write_bytes(b"nor is this")
    return front, back


def result_for(config: TwoViewConfig, *, pose: float, agreement: float, ambiguous: bool = False):
    run_dir = Path(config.runs_root) / "fixture"
    run_dir.mkdir(parents=True, exist_ok=True)
    glb = run_dir / "aligned.glb"
    glb.write_bytes(b"glTF fake")
    manifest = run_dir / "manifest.json"
    manifest.write_text("{}", encoding="utf-8")
    return TwoViewResult(
        glb_path=glb,
        run_dir=run_dir,
        manifest_path=manifest,
        pose_iou=pose,
        agreement_iou=agreement,
        is_ambiguous=ambiguous,
    )


def settle(app: QApplication, panel: TwoViewPanel, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while panel.is_running and time.monotonic() < deadline:
        app.processEvents()
    app.processEvents()


# --------------------------------------------------------------------------
# refusals, before anything is spent
# --------------------------------------------------------------------------


def test_the_button_is_disabled_until_both_photos_are_chosen(
    app: QApplication, tmp_path: Path
) -> None:
    panel = TwoViewPanel(
        runs_root=tmp_path / "runs", runner=lambda c: result_for(c, pose=1, agreement=1)
    )
    assert not panel.run_button.isEnabled()

    front, back = photos(tmp_path)
    panel.front_edit.setText(str(front))
    panel._refresh()
    assert not panel.run_button.isEnabled(), "one photo is not two views"

    panel.back_edit.setText(str(back))
    panel._refresh()
    assert panel.run_button.isEnabled()


def test_the_same_photo_twice_is_refused_without_starting_anything(
    app: QApplication, tmp_path: Path
) -> None:
    """There is no pose to derive between an image and itself."""
    started: list[TwoViewConfig] = []
    front, _ = photos(tmp_path)
    panel = TwoViewPanel(
        runs_root=tmp_path / "runs",
        runner=lambda c: (started.append(c), result_for(c, pose=1, agreement=1))[1],
    )
    panel.front_edit.setText(str(front))
    panel.back_edit.setText(str(front))
    panel._refresh()

    panel.run_button.click()
    settle(app, panel)

    assert started == []
    assert "identiques" in panel.status.text()


def test_a_missing_file_is_refused(app: QApplication, tmp_path: Path) -> None:
    started: list[TwoViewConfig] = []
    front, _ = photos(tmp_path)
    panel = TwoViewPanel(
        runs_root=tmp_path / "runs",
        runner=lambda c: (started.append(c), result_for(c, pose=1, agreement=1))[1],
    )
    panel.front_edit.setText(str(front))
    panel.back_edit.setText(str(tmp_path / "absent.jpg"))
    panel._refresh()

    panel.run_button.click()
    settle(app, panel)

    assert started == []
    assert "Introuvable" in panel.status.text()


def test_without_a_runner_the_panel_says_a_pod_is_required(
    app: QApplication, tmp_path: Path
) -> None:
    """Honest about F07 rather than failing obscurely at click time."""
    front, back = photos(tmp_path)
    panel = TwoViewPanel(runs_root=tmp_path / "runs")
    panel.front_edit.setText(str(front))
    panel.back_edit.setText(str(back))
    panel._refresh()

    panel.run_button.click()
    settle(app, panel)

    assert "Pod" in panel.status.text()


# --------------------------------------------------------------------------
# a real run
# --------------------------------------------------------------------------


def test_a_good_run_reports_both_numbers_and_the_file(
    app: QApplication, tmp_path: Path
) -> None:
    front, back = photos(tmp_path)
    panel = TwoViewPanel(
        runs_root=tmp_path / "runs", runner=lambda c: result_for(c, pose=0.93, agreement=0.87)
    )
    panel.front_edit.setText(str(front))
    panel.back_edit.setText(str(back))
    panel._refresh()

    panel.run_button.click()
    settle(app, panel)

    assert panel.result is not None and panel.result.glb_path.is_file()
    text = panel.status.text()
    assert "0.93" in text and "0.87" in text
    assert "aligned.glb" in text


def test_a_weak_alignment_is_reported_as_unusable_not_as_a_glb(
    app: QApplication, tmp_path: Path
) -> None:
    """The failure mode that matters: a file exists but means nothing."""
    front, back = photos(tmp_path)
    panel = TwoViewPanel(
        runs_root=tmp_path / "runs", runner=lambda c: result_for(c, pose=0.31, agreement=0.12)
    )
    panel.front_edit.setText(str(front))
    panel.back_edit.setText(str(back))
    panel._refresh()

    panel.run_button.click()
    settle(app, panel)

    assert "Ne pas exploiter" in panel.status.text()


def test_an_ambiguous_pose_is_explained_rather_than_alarming(
    app: QApplication, tmp_path: Path
) -> None:
    front, back = photos(tmp_path)
    panel = TwoViewPanel(
        runs_root=tmp_path / "runs",
        runner=lambda c: result_for(c, pose=0.95, agreement=0.9, ambiguous=True),
    )
    panel.front_edit.setText(str(front))
    panel.back_edit.setText(str(back))
    panel._refresh()

    panel.run_button.click()
    settle(app, panel)

    assert "axisymétrique" in panel.status.text()
    assert "Ne pas exploiter" not in panel.status.text()


def test_a_crash_in_the_trial_does_not_take_the_window_down(
    app: QApplication, tmp_path: Path
) -> None:
    front, back = photos(tmp_path)

    def exploding(config: TwoViewConfig) -> TwoViewResult:
        raise RuntimeError("the pod went away")

    panel = TwoViewPanel(runs_root=tmp_path / "runs", runner=exploding)
    panel.front_edit.setText(str(front))
    panel.back_edit.setText(str(back))
    panel._refresh()

    panel.run_button.click()
    settle(app, panel)

    assert panel.result is None
    assert "went away" in panel.status.text()
    assert not panel.is_running


def test_the_browse_button_fills_the_field(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    front, _ = photos(tmp_path)
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(front), ""))
    )
    panel = TwoViewPanel(runs_root=tmp_path / "runs")

    panel.findChildren(type(panel.run_button))[0].click()  # first "Parcourir…"

    assert panel.front_edit.text() == str(front)


def test_the_window_exposes_the_tab(app: QApplication, tmp_path: Path) -> None:
    import sys

    from pixaboost.gui.main_window import MainWindow
    from pixaboost.gui.model import CommandSpec

    command = CommandSpec(
        key="fixture",
        label="Fixture",
        description="Essai local déterministe.",
        program=sys.executable,
        arguments=("-c", "pass"),
        working_directory=tmp_path,
    )
    window = MainWindow(commands=(command,), repo_root=tmp_path)
    try:
        titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert any("2 vues" in title for title in titles), titles
        assert window.two_view_panel.result is None
    finally:
        window.close()
