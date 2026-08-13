"""Offscreen behaviour of the click-to-segment panel (F14).

The model is substituted throughout: `poe check` must not download 3.4 GB of
gated weights, and none of the behaviour under test is Meta's.

What is under test is the part that quietly ruins a run -- the mapping from a
mouse position to an image pixel -- plus the contract that makes Pixal3D honour
the result instead of re-running its own background removal.
"""

from __future__ import annotations

import os
from collections.abc import Generator
from pathlib import Path

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest
from PIL import Image
from PyQt6.QtCore import QPoint, QPointF, Qt
from PyQt6.QtGui import QMouseEvent
from PyQt6.QtWidgets import QApplication

from pixaboost.backends.sam3 import PointPrompt, Sam3Error, SegmentationResult
from pixaboost.gui.segmentation_view import (
    ImageCanvas,
    SegmentationPanel,
    auto_prompt_from_coarse_mask,
)

IMAGE_H, IMAGE_W = 120, 200


@pytest.fixture(scope="module")
def app() -> Generator[QApplication, None, None]:
    instance = QApplication.instance() or QApplication([])
    yield instance
    instance.processEvents()


def photo() -> np.ndarray:
    rng = np.random.default_rng(7)
    return rng.integers(0, 256, size=(IMAGE_H, IMAGE_W, 3), dtype=np.uint8)


class FakeRunner:
    """Records prompts and returns a disc mask centred on the first prompt."""

    def __init__(self, radius: int = 20, score: float = 0.93) -> None:
        self.calls: list[tuple[PointPrompt, ...]] = []
        self.radius, self.score = radius, score

    def segment(
        self, image: np.ndarray, prompts: tuple[PointPrompt, ...]
    ) -> SegmentationResult:
        self.calls.append(prompts)
        rows, cols = np.ogrid[: image.shape[0], : image.shape[1]]
        first = prompts[0]
        mask = ((rows - first.y) ** 2 + (cols - first.x) ** 2) <= self.radius**2
        return SegmentationResult(mask=mask, iou_score=self.score, candidate_scores=(self.score,))


class FailingRunner:
    def segment(self, image: np.ndarray, prompts: tuple[PointPrompt, ...]) -> SegmentationResult:
        raise Sam3Error("the pod is on fire")


def settle(app: QApplication, panel: SegmentationPanel, timeout: float = 5.0) -> None:
    """Pump the event loop until the worker thread has finished."""
    import time

    deadline = time.monotonic() + timeout
    while panel.is_running and time.monotonic() < deadline:
        app.processEvents()
    app.processEvents()


def click(canvas: ImageCanvas, x: int, y: int, button: Qt.MouseButton) -> None:
    event = QMouseEvent(
        QMouseEvent.Type.MouseButtonPress,
        QPointF(x, y),
        button,
        button,
        Qt.KeyboardModifier.NoModifier,
    )
    canvas.mousePressEvent(event)


# --------------------------------------------------------------------------
# coordinate mapping -- the silent killer
# --------------------------------------------------------------------------


def test_a_click_maps_to_image_pixels_not_widget_pixels(app: QApplication) -> None:
    canvas = ImageCanvas()
    canvas.resize(800, 400)  # far larger than the 200x120 image
    canvas.set_image(photo())

    left, top, drawn_w, drawn_h = canvas.displayed_rect()
    centre = canvas.widget_to_image(QPoint(left + drawn_w // 2, top + drawn_h // 2))

    assert centre is not None
    x, y = centre
    assert abs(x - IMAGE_W // 2) <= 1
    assert abs(y - IMAGE_H // 2) <= 1


def test_the_corners_of_the_drawn_image_map_to_the_corners_of_the_photo(
    app: QApplication,
) -> None:
    canvas = ImageCanvas()
    canvas.resize(640, 480)
    canvas.set_image(photo())
    left, top, drawn_w, drawn_h = canvas.displayed_rect()

    assert canvas.widget_to_image(QPoint(left, top)) == (0, 0)
    assert canvas.widget_to_image(QPoint(left + drawn_w - 1, top + drawn_h - 1)) == (
        IMAGE_W - 1,
        IMAGE_H - 1,
    )


def test_a_click_in_the_letterbox_margin_is_dropped_not_clamped(app: QApplication) -> None:
    """A clamped prompt is a wrong prompt that looks deliberate."""
    canvas = ImageCanvas()
    canvas.resize(800, 400)
    canvas.set_image(photo())
    left, top, _, _ = canvas.displayed_rect()
    assert top > 0 or left > 0, "the fixture must actually letterbox"

    if left > 0:
        assert canvas.widget_to_image(QPoint(left - 1, top + 5)) is None
    if top > 0:
        assert canvas.widget_to_image(QPoint(left + 5, top - 1)) is None


def test_clicking_before_an_image_is_loaded_does_nothing(app: QApplication) -> None:
    canvas = ImageCanvas()
    canvas.resize(400, 300)
    received: list[tuple[int, int, bool]] = []
    canvas.clicked.connect(lambda x, y, p: received.append((x, y, p)))
    click(canvas, 100, 100, Qt.MouseButton.LeftButton)
    assert received == []


# --------------------------------------------------------------------------
# prompts
# --------------------------------------------------------------------------


def test_left_click_is_a_positive_prompt_and_right_click_a_negative_one(
    app: QApplication,
) -> None:
    """Excluding the lifting clamp is a right click; that is why SAM is here."""
    runner = FakeRunner()
    panel = SegmentationPanel(runner=runner)
    panel.resize(400, 300)
    panel.set_image(photo())
    left, top, drawn_w, drawn_h = panel.canvas.displayed_rect()

    click(panel.canvas, left + drawn_w // 2, top + drawn_h // 2, Qt.MouseButton.LeftButton)
    settle(app, panel)
    click(panel.canvas, left + 5, top + 5, Qt.MouseButton.RightButton)
    settle(app, panel)

    assert [p.positive for p in panel.prompts] == [True, False]
    assert runner.calls[-1] == panel.prompts


def test_undo_removes_the_last_point_and_re_runs(app: QApplication) -> None:
    runner = FakeRunner()
    panel = SegmentationPanel(runner=runner)
    panel.resize(400, 300)
    panel.set_image(photo())
    left, top, drawn_w, drawn_h = panel.canvas.displayed_rect()

    click(panel.canvas, left + drawn_w // 2, top + drawn_h // 2, Qt.MouseButton.LeftButton)
    settle(app, panel)
    click(panel.canvas, left + drawn_w // 2 + 4, top + drawn_h // 2, Qt.MouseButton.LeftButton)
    settle(app, panel)
    assert len(panel.prompts) == 2

    panel.undo_last_prompt()
    settle(app, panel)

    assert len(panel.prompts) == 1
    assert len(runner.calls[-1]) == 1


def test_resetting_clears_the_mask_and_disables_saving(app: QApplication) -> None:
    panel = SegmentationPanel(runner=FakeRunner())
    panel.resize(400, 300)
    panel.set_image(photo())
    left, top, drawn_w, drawn_h = panel.canvas.displayed_rect()
    click(panel.canvas, left + drawn_w // 2, top + drawn_h // 2, Qt.MouseButton.LeftButton)
    settle(app, panel)
    assert panel.save_button.isEnabled()

    panel.reset_prompts()

    assert panel.prompts == ()
    assert panel.result is None
    assert not panel.save_button.isEnabled()


# --------------------------------------------------------------------------
# BiRefNet as a prompt generator, not as a mask
# --------------------------------------------------------------------------


def test_the_automatic_prompt_is_the_deepest_interior_point_not_the_centroid() -> None:
    """The test piece is a wheel: its centroid is the bore, which is background."""
    size = 201
    rows, cols = np.ogrid[:size, :size]
    radius_sq = (rows - 100) ** 2 + (cols - 100) ** 2
    coarse = (radius_sq <= 90**2) & (radius_sq > 30**2)

    prompt = auto_prompt_from_coarse_mask(coarse)

    assert coarse[prompt.y, prompt.x], "the auto prompt must land on the part"
    assert (prompt.y, prompt.x) != (100, 100), "that is the hole"
    assert prompt.positive


def test_a_coarse_mask_speck_does_not_capture_the_automatic_prompt() -> None:
    rows, cols = np.ogrid[:200, :200]
    coarse = ((rows - 120) ** 2 + (cols - 120) ** 2) <= 40**2
    coarse[2:6, 2:6] = True  # a bit of workbench

    prompt = auto_prompt_from_coarse_mask(coarse)

    assert np.hypot(prompt.y - 120, prompt.x - 120) < 40


def test_sams_mask_wins_over_the_coarse_mask(app: QApplication) -> None:
    """BiRefNet only picks the click. What gets saved is SAM's answer."""
    runner = FakeRunner(radius=10)
    panel = SegmentationPanel(runner=runner)
    panel.resize(400, 300)
    panel.set_image(photo())

    coarse = np.zeros((IMAGE_H, IMAGE_W), dtype=bool)
    coarse[10:110, 20:180] = True  # a huge, wrong, BiRefNet-style blob
    panel.prompt_automatically(coarse)
    settle(app, panel)

    assert panel.result is not None
    assert panel.result.mask.sum() < coarse.sum() / 4
    assert not np.array_equal(panel.result.mask, coarse)


# --------------------------------------------------------------------------
# the RGBA contract with Pixal3D
# --------------------------------------------------------------------------


def test_the_saved_png_has_non_uniform_alpha_so_pixal3d_honours_it(
    app: QApplication, tmp_path: Path
) -> None:
    """preprocess_image only skips rembg when `not np.all(alpha == 255)`."""
    panel = SegmentationPanel(runner=FakeRunner())
    panel.resize(400, 300)
    panel.set_image(photo())
    left, top, drawn_w, drawn_h = panel.canvas.displayed_rect()
    click(panel.canvas, left + drawn_w // 2, top + drawn_h // 2, Qt.MouseButton.LeftButton)
    settle(app, panel)

    written = panel.save_rgba(tmp_path / "cutout.png")

    with Image.open(written) as reopened:
        assert reopened.mode == "RGBA"
        alpha = np.array(reopened)[:, :, 3]
    assert not np.all(alpha == 255), "a fully opaque alpha is ignored by Pixal3D"
    assert alpha.max() == 255 and alpha.min() == 0


def test_saving_a_jpeg_is_refused_because_it_cannot_carry_alpha(
    app: QApplication, tmp_path: Path
) -> None:
    panel = SegmentationPanel(runner=FakeRunner())
    panel.resize(400, 300)
    panel.set_image(photo())
    left, top, drawn_w, drawn_h = panel.canvas.displayed_rect()
    click(panel.canvas, left + drawn_w // 2, top + drawn_h // 2, Qt.MouseButton.LeftButton)
    settle(app, panel)

    with pytest.raises(Sam3Error, match="PNG"):
        panel.save_rgba(tmp_path / "cutout.jpg")


def test_saving_without_a_mask_is_refused(app: QApplication, tmp_path: Path) -> None:
    panel = SegmentationPanel(runner=FakeRunner())
    panel.set_image(photo())
    with pytest.raises(Sam3Error, match="click the part first"):
        panel.save_rgba(tmp_path / "cutout.png")


def test_a_backend_failure_is_shown_and_leaves_no_stale_mask(app: QApplication) -> None:
    panel = SegmentationPanel(runner=FailingRunner())
    panel.resize(400, 300)
    panel.set_image(photo())
    left, top, drawn_w, drawn_h = panel.canvas.displayed_rect()

    click(panel.canvas, left + drawn_w // 2, top + drawn_h // 2, Qt.MouseButton.LeftButton)
    settle(app, panel)

    assert panel.result is None
    assert not panel.save_button.isEnabled()
    assert "on fire" in panel.status.text()


def test_the_runner_is_built_lazily_on_the_first_click_not_on_construction(
    app: QApplication,
) -> None:
    """Constructing the runner downloads 3.4 GB of gated weights.

    Opening a tab must never do that, so the factory is not called until a
    prompt actually exists.
    """
    built: list[FakeRunner] = []

    def factory() -> FakeRunner:
        built.append(FakeRunner())
        return built[-1]

    panel = SegmentationPanel(runner_factory=factory)
    panel.resize(400, 300)
    panel.set_image(photo())
    assert built == [], "loading an image must not build the model"
    assert not panel.has_engine

    left, top, drawn_w, drawn_h = panel.canvas.displayed_rect()
    click(panel.canvas, left + drawn_w // 2, top + drawn_h // 2, Qt.MouseButton.LeftButton)
    settle(app, panel)

    assert len(built) == 1, "the first click builds the runner exactly once"


def test_opening_the_window_does_not_touch_the_model(app: QApplication, tmp_path: Path) -> None:
    """End-to-end guard: `pixaboost-gui` must start with no network and no GPU."""
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
        window.show()
        app.processEvents()
        titles = [window.tabs.tabText(i) for i in range(window.tabs.count())]
        assert any("SAM" in title for title in titles), f"segmentation tab missing: {titles}"
        assert window.segmentation_panel.result is None
        assert window.segmentation_panel.prompts == ()
        assert not window.segmentation_panel.has_engine, (
            "opening the window must not construct the model"
        )
    finally:
        window.close()


def test_the_worker_leaves_no_thread_running(app: QApplication) -> None:
    panel = SegmentationPanel(runner=FakeRunner())
    panel.resize(400, 300)
    panel.set_image(photo())
    left, top, drawn_w, drawn_h = panel.canvas.displayed_rect()
    click(panel.canvas, left + drawn_w // 2, top + drawn_h // 2, Qt.MouseButton.LeftButton)
    settle(app, panel)
    assert not panel.is_running


# --------------------------------------------------------------------------
# the actual user path: can a human reach any of this from the window?
# --------------------------------------------------------------------------


def test_the_save_button_actually_writes_a_file(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Calling save_rgba() from a test proves nothing about the button.

    The first version of this panel wired undo and reset but not save, and the
    whole suite stayed green because every test called the method directly.
    """
    from PyQt6.QtWidgets import QFileDialog

    destination = tmp_path / "cutout.png"
    monkeypatch.setattr(
        QFileDialog, "getSaveFileName", staticmethod(lambda *a, **k: (str(destination), ""))
    )

    panel = SegmentationPanel(runner=FakeRunner())
    panel.resize(400, 300)
    panel.set_image(photo())
    left, top, drawn_w, drawn_h = panel.canvas.displayed_rect()
    click(panel.canvas, left + drawn_w // 2, top + drawn_h // 2, Qt.MouseButton.LeftButton)
    settle(app, panel)

    panel.save_button.click()
    app.processEvents()

    assert destination.is_file(), "the save button is not connected to anything"


def test_an_image_can_be_loaded_from_the_panel_itself(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without this the tab opens on 'no image' with no way out."""
    from PyQt6.QtWidgets import QFileDialog

    source = tmp_path / "view01.png"
    Image.fromarray(photo(), mode="RGB").save(source)
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(source), ""))
    )

    panel = SegmentationPanel(runner=FakeRunner())
    panel.resize(400, 300)
    panel.load_button.click()
    app.processEvents()

    assert panel.canvas.image is not None
    assert panel.canvas.image.shape == (IMAGE_H, IMAGE_W, 3)


def test_an_exif_rotated_photo_is_uprighted_on_load(
    app: QApplication, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """All 18 real photos carry EXIF Orientation=6; PIL does not apply it.

    Loading them raw would show the part lying on its side, so every click
    would be recorded against a differently-oriented image than the one
    Pixal3D eventually sees.
    """
    from PyQt6.QtWidgets import QFileDialog

    source = tmp_path / "rotated.jpg"
    portrait = np.zeros((40, 90, 3), dtype=np.uint8)  # wide
    exif = Image.Exif()
    exif[274] = 6  # rotate 90 degrees clockwise on display
    Image.fromarray(portrait, mode="RGB").save(source, exif=exif)

    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", staticmethod(lambda *a, **k: (str(source), ""))
    )
    panel = SegmentationPanel(runner=FakeRunner())
    panel.load_button.click()
    app.processEvents()

    assert panel.canvas.image is not None
    assert panel.canvas.image.shape[:2] == (90, 40), "EXIF orientation was not applied"


# --------------------------------------------------------------------------
# the full chain: BiRefNet excludes the scene, SAM segments the part
# --------------------------------------------------------------------------


class FakeSaliency:
    """Returns a deliberately over-inclusive mask, like BiRefNet on these photos.

    The blob spans the part *and* the clamp above it, because salient object
    detection has no reason to separate them.
    """

    def __init__(self) -> None:
        self.calls = 0

    def coarse_mask(self, image: np.ndarray) -> np.ndarray:
        self.calls += 1
        mask = np.zeros(image.shape[:2], dtype=bool)
        mask[30:110, 60:150] = True  # the part
        mask[0:30, 95:115] = True  # the clamp, still attached
        return mask


def test_the_auto_button_runs_birefnet_then_prompts_then_keeps_sams_mask(
    app: QApplication,
) -> None:
    """The whole point of the chain: BiRefNet's blob must not be the output."""
    saliency, runner = FakeSaliency(), FakeRunner(radius=12)
    panel = SegmentationPanel(runner=runner, saliency=saliency)
    panel.resize(400, 300)
    panel.set_image(photo())

    panel.auto_button.click()
    settle(app, panel)

    assert saliency.calls == 1, "BiRefNet must actually be consulted"
    assert len(runner.calls) == 1, "its mask must become exactly one prompt"
    assert panel.result is not None
    coarse = saliency.coarse_mask(photo())
    assert not np.array_equal(panel.result.mask, coarse), "SAM's mask must win"
    assert panel.result.mask.sum() < coarse.sum()


def test_the_automatic_prompt_lands_inside_the_coarse_mask(app: QApplication) -> None:
    saliency, runner = FakeSaliency(), FakeRunner()
    panel = SegmentationPanel(runner=runner, saliency=saliency)
    panel.resize(400, 300)
    panel.set_image(photo())

    panel.auto_button.click()
    settle(app, panel)

    prompt = panel.prompts[0]
    assert saliency.coarse_mask(photo())[prompt.y, prompt.x]
    assert prompt.positive


def test_the_auto_button_is_disabled_until_a_photo_is_loaded(app: QApplication) -> None:
    panel = SegmentationPanel(runner=FakeRunner(), saliency=FakeSaliency())
    assert not panel.auto_button.isEnabled()
    panel.set_image(photo())
    assert panel.auto_button.isEnabled()


def test_birefnet_finding_nothing_is_reported_and_manual_clicking_still_works(
    app: QApplication,
) -> None:
    """A dark or low-contrast shot must not dead-end the user."""
    from pixaboost.backends.birefnet import SaliencyError

    class EmptySaliency:
        def coarse_mask(self, image: np.ndarray) -> np.ndarray:
            raise SaliencyError("BiRefNet found no foreground above 0.50")

    runner = FakeRunner()
    panel = SegmentationPanel(runner=runner, saliency=EmptySaliency())
    panel.resize(400, 300)
    panel.set_image(photo())

    panel.auto_button.click()
    settle(app, panel)
    assert panel.result is None
    assert "no foreground" in panel.status.text()

    left, top, drawn_w, drawn_h = panel.canvas.displayed_rect()
    click(panel.canvas, left + drawn_w // 2, top + drawn_h // 2, Qt.MouseButton.LeftButton)
    settle(app, panel)
    assert panel.result is not None, "manual clicking must survive a saliency failure"


def test_neither_model_is_built_before_it_is_needed(app: QApplication) -> None:
    """Two lazy engines, two separate downloads, neither on tab open."""
    sam_built, saliency_built = [], []
    panel = SegmentationPanel(
        runner_factory=lambda: (sam_built.append(1), FakeRunner())[1],
        saliency_factory=lambda: (saliency_built.append(1), FakeSaliency())[1],
    )
    panel.resize(400, 300)
    panel.set_image(photo())
    assert not panel.has_engine and not panel.has_saliency_engine

    panel.auto_button.click()
    settle(app, panel)

    assert saliency_built == [1] and sam_built == [1]


def test_the_window_wires_both_engines_lazily(app: QApplication, tmp_path: Path) -> None:
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
        panel = window.segmentation_panel
        # Wired, but nothing downloaded yet -- both must hold at once.
        assert panel.can_auto_prompt, "the window did not wire BiRefNet up at all"
        assert not panel.has_engine
        assert not panel.has_saliency_engine
        assert panel.load_button.isEnabled(), "the user must be able to open a photo"
        assert not panel.auto_button.isEnabled(), "no photo yet"
    finally:
        window.close()


def test_the_overlay_dims_the_background_and_leaves_the_selection_alone(
    app: QApplication,
) -> None:
    """Tinting the selection was unreadable on a blue part with a blue accent.

    Dimming the background instead previews the real output: what goes dark is
    what becomes transparent.
    """
    canvas = ImageCanvas()
    canvas.resize(400, 300)
    flat = np.full((60, 80, 3), 200, dtype=np.uint8)
    canvas.set_image(flat)

    mask = np.zeros((60, 80), dtype=bool)
    mask[20:40, 30:50] = True
    canvas.set_mask(mask)

    rendered = canvas._render_composite()
    assert (rendered[mask] == 200).all(), "the selection must keep its true colours"
    assert (rendered[~mask] < 200).all(), "the background must be visibly dimmed"


def test_clearing_the_mask_restores_the_untouched_photo(app: QApplication) -> None:
    canvas = ImageCanvas()
    canvas.resize(400, 300)
    flat = np.full((20, 20, 3), 180, dtype=np.uint8)
    canvas.set_image(flat)
    mask = np.zeros((20, 20), dtype=bool)
    mask[5:10, 5:10] = True
    canvas.set_mask(mask)
    canvas.set_mask(None)

    assert (canvas._render_composite() == 180).all()


def test_each_click_leaves_a_visible_marker_coloured_by_its_polarity(
    app: QApplication,
) -> None:
    """Without markers the user cannot tell an include from an exclude."""
    panel = SegmentationPanel(runner=FakeRunner())
    panel.resize(400, 300)
    panel.set_image(np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8))
    left, top, drawn_w, drawn_h = panel.canvas.displayed_rect()

    click(panel.canvas, left + drawn_w // 2, top + drawn_h // 2, Qt.MouseButton.LeftButton)
    settle(app, panel)
    positive = panel.prompts[0]
    rendered = panel.canvas._render_composite()
    assert tuple(rendered[positive.y, positive.x]) == (57, 200, 137), "positive marker is green"

    click(panel.canvas, left + 10, top + 10, Qt.MouseButton.RightButton)
    settle(app, panel)
    negative = panel.prompts[1]
    rendered = panel.canvas._render_composite()
    assert tuple(rendered[negative.y, negative.x]) == (255, 111, 112), "negative marker is red"


def test_markers_follow_undo_and_reset(app: QApplication) -> None:
    panel = SegmentationPanel(runner=FakeRunner())
    panel.resize(400, 300)
    panel.set_image(np.zeros((IMAGE_H, IMAGE_W, 3), dtype=np.uint8))
    left, top, drawn_w, drawn_h = panel.canvas.displayed_rect()
    click(panel.canvas, left + drawn_w // 2, top + drawn_h // 2, Qt.MouseButton.LeftButton)
    settle(app, panel)
    marked = panel.prompts[0]

    panel.reset_prompts()
    rendered = panel.canvas._render_composite()

    assert tuple(rendered[marked.y, marked.x]) == (0, 0, 0), "a cleared prompt must leave no dot"


def test_a_marker_outside_the_image_is_ignored_rather_than_wrapping(
    app: QApplication,
) -> None:
    """Negative indices would wrap around and stamp a dot on the far edge."""
    from pixaboost.gui.segmentation_view import _stamp_marker

    canvas = np.zeros((20, 20, 3), dtype=np.uint8)
    _stamp_marker(canvas, PointPrompt(x=-5, y=10))
    _stamp_marker(canvas, PointPrompt(x=10, y=99))
    assert not canvas.any()


# --------------------------------------------------------------------------
# what the status line is allowed to claim
# --------------------------------------------------------------------------


def test_a_tiny_mask_is_flagged_even_though_sam_scores_it_highly(
    app: QApplication,
) -> None:
    """Measured on view01: bore click = 0.95 for 1.4 %, strap click = 0.97 for 1.8 %.

    The score does not separate "the part" from "a bit of the rigging". Showing
    it alone reads as reassurance for a mask that is plainly wrong.
    """

    class TinyButConfident:
        def segment(
            self, image: np.ndarray, prompts: tuple[PointPrompt, ...]
        ) -> SegmentationResult:
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask[0:8, 0:8] = True  # ~0.3 % of a 120x200 image
            return SegmentationResult(mask=mask, iou_score=0.97, candidate_scores=(0.97, 0.10))

    panel = SegmentationPanel(runner=TinyButConfident())
    panel.resize(400, 300)
    panel.set_image(photo())
    left, top, drawn_w, drawn_h = panel.canvas.displayed_rect()
    click(panel.canvas, left + drawn_w // 2, top + drawn_h // 2, Qt.MouseButton.LeftButton)
    settle(app, panel)

    text = panel.status.text()
    assert "%" in text, "coverage must be shown, not only the score"
    assert "alésage" in text or "élingue" in text, "the likely cause must be named"


def test_a_full_sized_mask_is_reported_without_a_warning(app: QApplication) -> None:
    panel = SegmentationPanel(runner=FakeRunner(radius=40))
    panel.resize(400, 300)
    panel.set_image(photo())
    left, top, drawn_w, drawn_h = panel.canvas.displayed_rect()
    click(panel.canvas, left + drawn_w // 2, top + drawn_h // 2, Qt.MouseButton.LeftButton)
    settle(app, panel)

    text = panel.status.text()
    assert "%" in text
    assert "alésage" not in text and "élingue" not in text
