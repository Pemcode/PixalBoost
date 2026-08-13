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
