"""Offscreen behaviour of the two-view panel (F15).

The engine is substituted: what is under test is the panel -- that it refuses
nonsensical input before spending anything, that a cache miss is confirmed by
name before any connection, that the run happens off the GUI thread, that it
can be cancelled, and that a weak result is reported as weak rather than as a
GLB.

The assertions that matter are the ones counting `engine.runs`. Each one is a
statement about money.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Callable, Generator
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest
from PIL import Image
from PyQt6.QtWidgets import QApplication, QFileDialog, QMessageBox

from pixaboost.backends.ssh_pod import CancelState
from pixaboost.core.segmentation import compose_rgba
from pixaboost.gui.two_view_adapter import TwoViewPreflight
from pixaboost.gui.two_view_view import TwoViewPanel
from pixaboost.trials.two_view import TwoViewConfig, TwoViewResult


@pytest.fixture(scope="module")
def app() -> Generator[QApplication, None, None]:
    instance = QApplication.instance() or QApplication([])
    yield instance
    instance.processEvents()


def cutouts(tmp_path: Path) -> tuple[Path, Path]:
    paths = []
    for name, shift in (("front_cutout.png", 0), ("back_cutout.png", 6)):
        mask = np.zeros((32, 32), dtype=bool)
        mask[8 : 24 + shift, 10:22] = True
        rgb = np.full((32, 32, 3), 160, dtype=np.uint8)
        path = tmp_path / name
        Image.fromarray(compose_rgba(rgb, mask), mode="RGBA").save(path, format="PNG")
        paths.append(path)
    return paths[0], paths[1]


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


class FakeEngine:
    """Records every decision the panel makes on its behalf."""

    def __init__(
        self,
        *,
        pose: float = 0.93,
        agreement: float = 0.87,
        ambiguous: bool = False,
        cache_hit: bool = False,
        preflight_error: Exception | None = None,
        run_error: Exception | None = None,
        block: threading.Event | None = None,
    ) -> None:
        self._pose, self._agreement, self._ambiguous = pose, agreement, ambiguous
        self._cache_hit = cache_hit
        self._preflight_error, self._run_error = preflight_error, run_error
        self._block = block
        self.entered = threading.Event()
        self.preflights: list[TwoViewConfig] = []
        self.runs: list[tuple[TwoViewConfig, bool]] = []
        self.cancels = 0
        self.cancel_state = CancelState.ACKNOWLEDGED

    def preflight(self, config: TwoViewConfig) -> TwoViewPreflight:
        self.preflights.append(config)
        if self._preflight_error is not None:
            raise self._preflight_error
        return TwoViewPreflight(
            front_image=Path(config.front_image),
            back_image=Path(config.back_image),
            front_cache_hit=self._cache_hit,
            back_cache_hit=self._cache_hit,
        )

    def run(
        self,
        config: TwoViewConfig,
        *,
        approve_existing_pod: bool,
        event_sink: Callable[[object], None] | None = None,
    ) -> TwoViewResult:
        self.runs.append((config, approve_existing_pod))
        self.entered.set()
        if self._block is not None:
            assert self._block.wait(10.0), "the test never released the run"
        if self._run_error is not None:
            raise self._run_error
        return result_for(
            config, pose=self._pose, agreement=self._agreement, ambiguous=self._ambiguous
        )

    def cancel(self) -> CancelState:
        self.cancels += 1
        return self.cancel_state


def panel_for(
    tmp_path: Path, engine: FakeEngine | None, *, confirm: bool = True
) -> TwoViewPanel:
    return TwoViewPanel(
        runs_root=tmp_path / "runs",
        engine_factory=(lambda: engine) if engine is not None else None,
        confirm=lambda _preflight: confirm,
    )


def drive(app: QApplication, panel: TwoViewPanel, front: Path, back: Path) -> None:
    panel.front_edit.setText(str(front))
    panel.back_edit.setText(str(back))
    panel._refresh()
    panel.run_button.click()
    settle(app, panel)


def launch(
    app: QApplication, panel: TwoViewPanel, front: Path, back: Path, engine: FakeEngine
) -> None:
    """Click Run and return once the engine is genuinely inside `run`."""
    panel.front_edit.setText(str(front))
    panel.back_edit.setText(str(back))
    panel._refresh()
    panel.run_button.click()
    wait_until(app, engine.entered.is_set)


def wait_until(app: QApplication, predicate, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while not predicate() and time.monotonic() < deadline:
        app.processEvents()
    app.processEvents()
    assert predicate(), "the expected state never arrived"


def settle(app: QApplication, panel: TwoViewPanel, timeout: float = 5.0) -> None:
    """Wait for every thread parented to the panel to be torn down.

    Not for a counter: the worker sets those from inside the thread, before its
    `finished` signal is delivered. Returning early leaves a live QThread
    parented to a widget about to be garbage collected, which aborts the whole
    process -- intermittently, and in whichever test happens to run next.
    """
    deadline = time.monotonic() + timeout
    while panel.is_busy and time.monotonic() < deadline:
        app.processEvents()
    app.processEvents()
    assert not panel.is_busy, "a worker thread outlived the test"


# --------------------------------------------------------------------------
# refusals, before anything is spent
# --------------------------------------------------------------------------


def test_the_button_is_disabled_until_both_cutouts_are_chosen(
    app: QApplication, tmp_path: Path
) -> None:
    panel = panel_for(tmp_path, FakeEngine())
    assert not panel.run_button.isEnabled()

    front, back = cutouts(tmp_path)
    panel.front_edit.setText(str(front))
    panel._refresh()
    assert not panel.run_button.isEnabled(), "one cutout is not two views"

    panel.back_edit.setText(str(back))
    panel._refresh()
    assert panel.run_button.isEnabled()


def test_the_same_cutout_twice_is_refused_without_starting_anything(
    app: QApplication, tmp_path: Path
) -> None:
    """There is no pose to derive between an image and itself."""
    engine = FakeEngine()
    front, _ = cutouts(tmp_path)
    panel = panel_for(tmp_path, engine)

    drive(app, panel, front, front)

    assert engine.preflights == [] and engine.runs == []
    assert "identiques" in panel.status.text()


def test_a_missing_file_is_refused(app: QApplication, tmp_path: Path) -> None:
    engine = FakeEngine()
    front, _ = cutouts(tmp_path)
    panel = panel_for(tmp_path, engine)

    drive(app, panel, front, tmp_path / "absent.png")

    assert engine.preflights == []
    assert "Introuvable" in panel.status.text()


def test_without_an_engine_the_panel_says_a_pod_is_required(
    app: QApplication, tmp_path: Path
) -> None:
    """Honest about F07 rather than failing obscurely at click time."""
    front, back = cutouts(tmp_path)
    panel = panel_for(tmp_path, None)

    drive(app, panel, front, back)

    assert "Pod" in panel.status.text()


def test_a_raw_photograph_is_named_and_the_fix_is_spelled_out(
    app: QApplication, tmp_path: Path
) -> None:
    """The mistake a user will make, and the sentence that unblocks them."""
    engine = FakeEngine(
        preflight_error=ValueError("view07.jpg: this image has no alpha channel")
    )
    front, back = cutouts(tmp_path)
    panel = panel_for(tmp_path, engine)

    drive(app, panel, front, back)

    text = panel.status.text()
    assert "view07.jpg" in text
    assert "Découpe (SAM 3)" in text
    assert engine.runs == [], "a bad input must not reach the Pod"


def test_refusing_the_confirmation_stops_before_any_connection(
    app: QApplication, tmp_path: Path
) -> None:
    """The cache miss dialog is a real gate, not a notification."""
    engine = FakeEngine(cache_hit=False)
    front, back = cutouts(tmp_path)
    panel = panel_for(tmp_path, engine, confirm=False)

    drive(app, panel, front, back)

    assert engine.preflights and engine.runs == []
    assert "annulé" in panel.status.text()


def test_a_full_cache_hit_is_never_confirmed(app: QApplication, tmp_path: Path) -> None:
    """Reusing a cached artefact costs nothing, so it must not ask."""
    engine = FakeEngine(cache_hit=True)
    front, back = cutouts(tmp_path)
    asked: list[TwoViewPreflight] = []
    panel = TwoViewPanel(
        runs_root=tmp_path / "runs",
        engine_factory=lambda: engine,
        confirm=lambda preflight: (asked.append(preflight), False)[1],
    )

    drive(app, panel, front, back)

    assert asked == [], "a cache hit asked for permission to spend nothing"
    assert engine.runs and engine.runs[0][1] is False, "approval granted for a cache hit"
    assert panel.result is not None


# --------------------------------------------------------------------------
# a real run
# --------------------------------------------------------------------------


def test_a_good_run_reports_both_numbers_and_the_file(
    app: QApplication, tmp_path: Path
) -> None:
    engine = FakeEngine(pose=0.93, agreement=0.87)
    front, back = cutouts(tmp_path)
    panel = panel_for(tmp_path, engine)

    drive(app, panel, front, back)

    assert panel.result is not None and panel.result.glb_path.is_file()
    text = panel.status.text()
    assert "0.93" in text and "0.87" in text
    assert "aligned.glb" in text
    assert engine.runs[0][1] is True, "a cache miss must run with the approval"


def test_a_weak_alignment_is_reported_as_unusable_not_as_a_glb(
    app: QApplication, tmp_path: Path
) -> None:
    """The failure mode that matters: a file exists but means nothing."""
    front, back = cutouts(tmp_path)
    panel = panel_for(tmp_path, FakeEngine(pose=0.31, agreement=0.12))

    drive(app, panel, front, back)

    assert "Ne pas exploiter" in panel.status.text()


def test_an_ambiguous_pose_is_explained_rather_than_alarming(
    app: QApplication, tmp_path: Path
) -> None:
    front, back = cutouts(tmp_path)
    panel = panel_for(tmp_path, FakeEngine(pose=0.95, agreement=0.9, ambiguous=True))

    drive(app, panel, front, back)

    assert "axisymétrique" in panel.status.text()
    assert "Ne pas exploiter" not in panel.status.text()


def test_a_crash_in_the_trial_does_not_take_the_window_down(
    app: QApplication, tmp_path: Path
) -> None:
    front, back = cutouts(tmp_path)
    panel = panel_for(tmp_path, FakeEngine(run_error=RuntimeError("the pod went away")))

    drive(app, panel, front, back)

    assert panel.result is None
    assert "went away" in panel.status.text()
    assert not panel.is_running


def test_the_opposite_faces_prior_is_on_by_default_and_reaches_the_trial(
    app: QApplication, tmp_path: Path
) -> None:
    """Front and back of a revolved part have the same outline.

    Without the prior the search returns the identity and the two halves land
    on top of each other instead of completing one another.
    """
    engine = FakeEngine()
    front, back = cutouts(tmp_path)
    panel = panel_for(tmp_path, engine)
    assert panel.opposite_check.isChecked(), "the labelled case is the default"

    drive(app, panel, front, back)
    assert engine.runs[-1][0].opposite_faces is True

    panel.opposite_check.setChecked(False)
    panel.run_button.click()
    settle(app, panel)
    assert engine.runs[-1][0].opposite_faces is False, "unchecking must reach the trial"


# --------------------------------------------------------------------------
# cancellation
# --------------------------------------------------------------------------


def test_cancel_is_offered_exactly_while_the_trial_runs(
    app: QApplication, tmp_path: Path
) -> None:
    release = threading.Event()
    engine = FakeEngine(block=release)
    panel = panel_for(tmp_path, engine)
    assert not panel.cancel_button.isEnabled()

    front, back = cutouts(tmp_path)
    launch(app, panel, front, back, engine)
    assert panel.cancel_button.isEnabled()
    assert not panel.run_button.isEnabled(), "a second run would bill the Pod twice"

    release.set()
    settle(app, panel)
    assert not panel.cancel_button.isEnabled(), "the run is over"


def test_cancelling_a_running_trial_reaches_the_engine(
    app: QApplication, tmp_path: Path
) -> None:
    release = threading.Event()
    engine = FakeEngine(block=release)
    panel = panel_for(tmp_path, engine)
    front, back = cutouts(tmp_path)
    launch(app, panel, front, back, engine)

    panel.cancel_button.click()
    wait_until(app, lambda: engine.cancels == 1)

    release.set()
    settle(app, panel)
    assert engine.cancels == 1


def test_an_unknown_cancel_state_is_reported_as_unknown(
    app: QApplication, tmp_path: Path
) -> None:
    """Silence about a Pod that may still be burning GPU time is the bug.

    The acknowledgement and the run's own outcome land on the GUI thread in
    either order, and whichever arrives second used to erase the first.
    """
    release = threading.Event()
    engine = FakeEngine(block=release, run_error=RuntimeError("connexion perdue"))
    engine.cancel_state = CancelState.UNKNOWN
    panel = panel_for(tmp_path, engine)
    front, back = cutouts(tmp_path)
    launch(app, panel, front, back, engine)

    panel.cancel_button.click()
    release.set()
    settle(app, panel)

    text = panel.status.text()
    assert "inconnu" in text, text
    assert "connexion perdue" in text, "the failure itself was swallowed"


def test_a_completed_run_is_not_muddied_by_an_unknown_cancel(
    app: QApplication, tmp_path: Path
) -> None:
    """Both reconstructions finished, so the Pod is not still working on them."""
    release = threading.Event()
    engine = FakeEngine(block=release)
    engine.cancel_state = CancelState.UNKNOWN
    panel = panel_for(tmp_path, engine)
    front, back = cutouts(tmp_path)
    launch(app, panel, front, back, engine)

    panel.cancel_button.click()
    release.set()
    settle(app, panel)

    assert "inconnu" not in panel.status.text()
    assert panel.result is not None


def test_closing_the_window_is_refused_over_a_live_worker_thread(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A QThread parented to a widget aborts the process if it outlives it.

    And a two-view run is billed GPU time, so it must not be detached in
    silence either.
    """
    monkeypatch.setattr(
        QMessageBox,
        "question",
        staticmethod(lambda *args, **kwargs: QMessageBox.StandardButton.No),
    )
    release = threading.Event()
    # Cached, so the run starts without the panel's own confirmation dialog:
    # the only modal left is the one `closeEvent` raises, which is the subject.
    engine = FakeEngine(block=release, cache_hit=True)
    window = window_for(tmp_path)
    window.two_view_panel._engine_factory = lambda: engine
    try:
        front, back = cutouts(tmp_path)
        launch(app, window.two_view_panel, front, back, engine)

        assert window.close() is False, "the window closed over a live worker thread"
    finally:
        release.set()
        settle(app, window.two_view_panel)
        window.close()


# --------------------------------------------------------------------------
# the window
# --------------------------------------------------------------------------


def test_the_browse_button_fills_the_field(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    front, _ = cutouts(tmp_path)
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(front), ""))
    )
    panel = panel_for(tmp_path, FakeEngine())

    panel.findChildren(type(panel.run_button))[0].click()  # first "Parcourir…"

    assert panel.front_edit.text() == str(front)


def window_for(tmp_path: Path):
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
    return MainWindow(commands=(command,), repo_root=tmp_path)


def test_the_window_exposes_the_tab(app: QApplication, tmp_path: Path) -> None:
    window = window_for(tmp_path)
    try:
        titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert any("2 vues" in title for title in titles), titles
        assert window.two_view_panel.result is None
    finally:
        window.close()


def test_an_unconfigured_pod_is_reported_rather_than_dialled(
    app: QApplication, tmp_path: Path
) -> None:
    """Empty fields must fail on the GUI thread, not inside a worker."""
    window = window_for(tmp_path)
    try:
        front, back = cutouts(tmp_path)
        drive(app, window.two_view_panel, front, back)
        assert "Pod" in window.two_view_panel.status.text()
    finally:
        window.close()
