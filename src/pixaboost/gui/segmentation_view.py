"""Click-to-segment panel: image in, RGBA cutout out (F14).

The panel owns no policy. It displays a photograph, turns a mouse press into a
`PointPrompt` in *image* pixel coordinates, hands the prompts to an injected
`Sam3Runner`, and writes the RGBA that Pixal3D will accept in place of its own
background removal. Every geometric decision -- which point to auto-prompt,
how to build the alpha -- comes from `core/segmentation.py`.

The coordinate mapping is the part that silently ruins everything: the view
letterboxes the image to fit, so a click at widget (0, 0) is almost never image
(0, 0). A prompt off by a scale factor lands on the workbench and SAM
cheerfully segments the workbench.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PyQt6.QtCore import QObject, QPoint, Qt, QThread, pyqtSignal
from PyQt6.QtGui import QImage, QMouseEvent, QPixmap
from PyQt6.QtWidgets import (
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from pixaboost.backends.birefnet import SaliencyRunner
from pixaboost.backends.sam3 import PointPrompt, Sam3Error, Sam3Runner, SegmentationResult
from pixaboost.core.segmentation import (
    compose_rgba,
    deepest_interior_point,
    largest_connected_component,
)
from pixaboost.gui.theme import ACCENT, DANGER, SUCCESS, TEXT_DIM

#: How much of the original brightness survives outside the mask.
#:
#: The overlay dims the *background* rather than tinting the selection. Tinting
#: was tried first and is unreadable: the theme accent is blue and the test
#: piece is painted blue, so the mask vanished into the part. Dimming also
#: previews the actual output -- what goes dark is what becomes transparent.
_BACKGROUND_DIM = 0.30

#: Below this share of the image, a mask is almost certainly a sub-part rather
#: than the piece. Measured on piece_test/upright/view01.jpg: the wheel comes
#: back at ~19 %, while a click in the bore gives 1.4 % and a click on the
#: lifting strap 1.8 %. Advisory only -- a genuinely small part in a wide shot
#: is legitimate, so nothing is refused.
_SUSPICIOUSLY_SMALL_PERCENT = 4.0


@dataclass(frozen=True)
class CutoutRequest:
    """Everything needed to reproduce one cutout."""

    image_path: Path
    prompts: tuple[PointPrompt, ...]


#: Marker colours. Green includes, red excludes -- the same convention as SAM's
#: own label values (1 and 0).
_POSITIVE_RGB = (57, 200, 137)
_NEGATIVE_RGB = (255, 111, 112)


def _stamp_marker(canvas: np.ndarray, prompt: PointPrompt, radius: int | None = None) -> None:
    """Draw a filled dot with a dark rim at one prompt, in place.

    Sized as a fraction of the image so a marker stays visible on a 4000 px
    photograph scaled down to fit a panel.
    """
    height, width = canvas.shape[:2]
    if not (0 <= prompt.y < height and 0 <= prompt.x < width):
        return
    # ~2 % of the short side. At 90 the dot was barely two pixels once a 4000 px
    # photograph had been scaled into the panel.
    span = radius if radius is not None else max(4, min(height, width) // 50)
    rows, cols = np.ogrid[:height, :width]
    distance = (rows - prompt.y) ** 2 + (cols - prompt.x) ** 2
    colour = _POSITIVE_RGB if prompt.positive else _NEGATIVE_RGB
    canvas[distance <= (span + max(2, span // 2)) ** 2] = (20, 20, 24)  # rim, for contrast
    canvas[distance <= span**2] = colour


def auto_prompt_from_coarse_mask(coarse: np.ndarray) -> PointPrompt:
    """Derive a click from a saliency mask without asking the user.

    BiRefNet is demoted to a prompt generator here: its mask says roughly where
    the foreground is, and the deepest interior point of its largest blob is a
    point that is certainly *on* the part. The centroid is not -- the test
    piece is a wheel and its centroid is the bore. See `core/segmentation.py`.
    """
    row, col = deepest_interior_point(largest_connected_component(coarse))
    return PointPrompt(x=int(col), y=int(row), positive=True)


class ImageCanvas(QLabel):
    """Shows an image scaled to fit and reports clicks in image pixels.

    Left button adds a positive prompt ("this is the part"), right button a
    negative one ("this is the clamp"). Clicks in the letterbox margins are
    dropped rather than clamped: a clamped prompt is a wrong prompt that looks
    deliberate.
    """

    clicked = pyqtSignal(int, int, bool)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._image: np.ndarray | None = None
        self._overlay: np.ndarray | None = None
        self._markers: tuple[PointPrompt, ...] = ()
        self.setMinimumSize(320, 240)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(f"border: 1px solid {TEXT_DIM};")
        self.setText("Aucune image chargée")
        self.setAccessibleName("Zone de segmentation")
        self.setAccessibleDescription(
            "Clic gauche : point sur la pièce. Clic droit : point à exclure."
        )

    @property
    def image(self) -> np.ndarray | None:
        return self._image

    def set_image(self, image: np.ndarray) -> None:
        self._image = np.ascontiguousarray(image, dtype=np.uint8)
        self._overlay = None
        self._repaint()

    def set_mask(self, mask: np.ndarray | None) -> None:
        self._overlay = None if mask is None else np.asarray(mask, dtype=bool)
        self._repaint()

    def set_markers(self, prompts: tuple[PointPrompt, ...]) -> None:
        """Show where each click landed, and whether it included or excluded.

        Without this the user has no way to tell a positive click from a
        negative one after the fact, which makes correcting a bad mask guesswork.
        """
        self._markers = prompts
        self._repaint()

    def displayed_rect(self) -> tuple[int, int, int, int]:
        """Return `(left, top, width, height)` of the image inside the widget.

        Exposed because it is the single source of the scale factor, and the
        tests assert the round trip against it rather than re-deriving it.
        """
        if self._image is None:
            return (0, 0, 0, 0)
        height, width = self._image.shape[:2]
        available_w, available_h = max(1, self.width()), max(1, self.height())
        scale = min(available_w / width, available_h / height)
        drawn_w, drawn_h = max(1, round(width * scale)), max(1, round(height * scale))
        return ((available_w - drawn_w) // 2, (available_h - drawn_h) // 2, drawn_w, drawn_h)

    def widget_to_image(self, point: QPoint) -> tuple[int, int] | None:
        """Map a widget position to image pixels, or None if outside the image."""
        if self._image is None:
            return None
        left, top, drawn_w, drawn_h = self.displayed_rect()
        dx, dy = point.x() - left, point.y() - top
        if not (0 <= dx < drawn_w and 0 <= dy < drawn_h):
            return None
        height, width = self._image.shape[:2]
        x = min(width - 1, int(dx * width / drawn_w))
        y = min(height - 1, int(dy * height / drawn_h))
        return x, y

    def mousePressEvent(self, ev: QMouseEvent | None) -> None:
        if ev is None or self._image is None:
            return
        mapped = self.widget_to_image(ev.position().toPoint())
        if mapped is None:
            return
        if ev.button() not in (Qt.MouseButton.LeftButton, Qt.MouseButton.RightButton):
            return
        x, y = mapped
        self.clicked.emit(x, y, ev.button() == Qt.MouseButton.LeftButton)

    def resizeEvent(self, a0: object) -> None:
        super().resizeEvent(a0)  # type: ignore[arg-type]
        self._repaint()

    def _render_composite(self) -> np.ndarray:
        """Photo with the background dimmed wherever a mask is set.

        Split out from `_repaint` so the overlay can be asserted on pixels
        rather than on "a QPixmap was produced".
        """
        if self._image is None:
            raise ValueError("no image loaded")
        composite = self._image.copy()
        if self._overlay is not None and self._overlay.shape == composite.shape[:2]:
            background = ~self._overlay
            dimmed = composite[background].astype(np.float32) * _BACKGROUND_DIM
            composite[background] = dimmed.astype(np.uint8)
        for prompt in self._markers:
            _stamp_marker(composite, prompt)
        return composite

    def _repaint(self) -> None:
        if self._image is None:
            return
        composite = self._render_composite()
        height, width = composite.shape[:2]
        # `tobytes()` rather than the buffer: QImage does not own a numpy view,
        # and `composite` is a local that dies at the end of this method.
        image = QImage(
            composite.tobytes(), width, height, 3 * width, QImage.Format.Format_RGB888
        )
        pixmap = QPixmap.fromImage(image)
        self.setPixmap(
            pixmap.scaled(
                self.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )


class _SegmentationWorker(QObject):
    """Runs one inference off the GUI thread."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(
        self, runner: Sam3Runner, image: np.ndarray, prompts: tuple[PointPrompt, ...]
    ) -> None:
        super().__init__()
        self._runner, self._image, self._prompts = runner, image, prompts

    def run(self) -> None:
        try:
            self.finished.emit(self._runner.segment(self._image, self._prompts))
        except Sam3Error as error:
            self.failed.emit(str(error))
        except Exception as error:  # keep the window alive on any backend fault
            self.failed.emit(f"segmentation failed: {type(error).__name__}: {error}")


class SegmentationPanel(QWidget):
    """Load a photo, click the part, save the RGBA cutout.

    `runner` is injected so the gate never downloads 3.4 GB of gated weights;
    the real application passes a `Sam3TrackerRunner`.
    """

    mask_changed = pyqtSignal(object)
    status_changed = pyqtSignal(str)

    def __init__(
        self,
        runner: Sam3Runner | None = None,
        parent: QWidget | None = None,
        runner_factory: Callable[[], Sam3Runner] | None = None,
        saliency: SaliencyRunner | None = None,
        saliency_factory: Callable[[], SaliencyRunner] | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._runner_factory = runner_factory
        self._saliency = saliency
        self._saliency_factory = saliency_factory
        self._prompts: list[PointPrompt] = []
        self._result: SegmentationResult | None = None
        self._thread: QThread | None = None
        self._worker: _SegmentationWorker | None = None
        self._image_path: Path | None = None
        self._build_ui()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        self.canvas = ImageCanvas(self)
        self.canvas.clicked.connect(self._on_canvas_clicked)
        layout.addWidget(self.canvas, stretch=1)

        self.status = QLabel("Chargez une image, puis cliquez sur la pièce.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {TEXT_DIM};")
        self.status.setAccessibleName("État de la segmentation")
        layout.addWidget(self.status)

        buttons = QHBoxLayout()
        self.load_button = QPushButton("Ouvrir une photo…")
        self.load_button.setToolTip(
            "Charge une photo et applique son orientation EXIF. "
            "Les 18 photos reelles portent Orientation=6."
        )
        self.load_button.clicked.connect(self.browse_for_image)

        self.auto_button = QPushButton("Amorcer (BiRefNet)")
        self.auto_button.setToolTip(
            "Detoure grossierement la scene, puis place automatiquement le point "
            "au plus profond du masque. Le masque BiRefNet ne sort pas d'ici : "
            "seul celui de SAM est conserve."
        )
        self.auto_button.clicked.connect(self.prompt_from_saliency)

        self.undo_button = QPushButton("Annuler le dernier point")
        self.undo_button.setToolTip("Retire le dernier clic et relance la segmentation.")
        self.undo_button.clicked.connect(self.undo_last_prompt)

        self.reset_button = QPushButton("Effacer les points")
        self.reset_button.setToolTip("Repart d'une image sans aucun point.")
        self.reset_button.clicked.connect(self.reset_prompts)

        self.save_button = QPushButton("Enregistrer le PNG RGBA")
        self.save_button.setToolTip(
            "Écrit un PNG dont l'alpha est le masque SAM. Pixal3D l'utilise tel quel "
            "et n'exécute pas son propre détourage."
        )
        self.save_button.clicked.connect(self.browse_and_save)

        for button in (
            self.load_button,
            self.auto_button,
            self.undo_button,
            self.reset_button,
            self.save_button,
        ):
            buttons.addWidget(button)
        layout.addLayout(buttons)
        self._refresh_buttons()

    # -- file dialogs ------------------------------------------------------

    def browse_for_image(self) -> Path | None:
        """Pick a photo, upright it, and load it."""
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Choisir une photo",
            "",
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;Tous les fichiers (*)",
        )
        if not selected:
            return None
        path = Path(selected)
        try:
            self.load_image(path)
        except Exception as error:
            self._announce(f"Image illisible : {type(error).__name__}", status="error")
            return None
        return path

    def load_image(self, path: Path) -> None:
        """Read a photo, applying its EXIF orientation.

        All 18 real photos carry `Orientation=6` and PIL does not apply it on
        its own, so a raw read shows the part on its side. Every click would
        then be recorded against a different orientation than the one Pixal3D
        finally sees.
        """
        from PIL import Image, ImageOps

        with Image.open(path) as opened:
            upright = ImageOps.exif_transpose(opened)
            rgb = (upright or opened).convert("RGB")
            self.set_image(np.asarray(rgb, dtype=np.uint8))
        self._image_path = Path(path)
        self._announce(f"{Path(path).name} chargée. Cliquez sur la pièce.")

    def browse_and_save(self) -> Path | None:
        """Ask where to write the cutout, then write it."""
        if self._result is None:
            self._announce("Aucun masque à enregistrer.", status="error")
            return None
        suggested = "cutout.png"
        if self._image_path is not None:
            suggested = f"{self._image_path.stem}_cutout.png"
        selected, _filter = QFileDialog.getSaveFileName(
            self, "Enregistrer la découpe", suggested, "PNG (*.png)"
        )
        if not selected:
            return None
        target = Path(selected).with_suffix(".png")
        try:
            return self.save_rgba(target)
        except Exception as error:
            self._announce(f"Écriture impossible : {error}", status="error")
            return None

    # -- state -------------------------------------------------------------

    @property
    def prompts(self) -> tuple[PointPrompt, ...]:
        return tuple(self._prompts)

    @property
    def result(self) -> SegmentationResult | None:
        return self._result

    @property
    def is_running(self) -> bool:
        return self._thread is not None

    @property
    def has_engine(self) -> bool:
        """True once a segmentation runner has actually been constructed.

        Public because "opening this tab did not download 3.4 GB of gated
        weights" is a property worth asserting from outside.
        """
        return self._runner is not None

    def set_image(self, image: np.ndarray) -> None:
        self.canvas.set_image(image)
        self.reset_prompts()
        self._refresh_buttons()

    def reset_prompts(self) -> None:
        self._prompts.clear()
        self._sync_markers()
        self._set_result(None)
        self._announce("Cliquez sur la pièce. Clic droit pour exclure ce qui la touche.")

    def undo_last_prompt(self) -> None:
        if not self._prompts:
            return
        self._prompts.pop()
        self._sync_markers()
        if self._prompts:
            self._request_segmentation()
        else:
            self._set_result(None)
            self._announce("Plus aucun point. Cliquez sur la pièce.")

    def prompt_from_saliency(self) -> None:
        """Run BiRefNet, then place the prompt at the deepest point of its mask.

        The coarse mask is consumed here and discarded. What the panel keeps,
        displays and saves is SAM's answer -- otherwise two masks would be in
        play with no rule saying which one wins.
        """
        image = self.canvas.image
        if image is None:
            self._announce("Chargez d'abord une photo.", status="error")
            return
        saliency = self._resolve_saliency()
        if saliency is None:
            self._announce(
                "Aucun modèle de détourage configuré : cliquez directement sur la pièce.",
                status="error",
            )
            return
        self._announce("Détourage grossier (BiRefNet) en cours…")
        try:
            coarse = saliency.coarse_mask(image)
        except Exception as error:
            self._announce(f"Détourage impossible : {error}", status="error")
            return
        try:
            self.prompt_automatically(coarse)
        except ValueError as error:
            self._announce(f"Masque grossier inexploitable : {error}", status="error")

    def _resolve_saliency(self) -> SaliencyRunner | None:
        if self._saliency is None and self._saliency_factory is not None:
            self._saliency = self._saliency_factory()
        return self._saliency

    @property
    def has_saliency_engine(self) -> bool:
        """True once BiRefNet has actually been constructed."""
        return self._saliency is not None

    @property
    def can_auto_prompt(self) -> bool:
        """True when a saliency engine exists or can be built on demand.

        Distinct from `has_saliency_engine`: this one is true before anything
        is downloaded, and is what proves the window actually wired BiRefNet up
        rather than leaving the button to fail at click time.
        """
        return self._saliency is not None or self._saliency_factory is not None

    def prompt_automatically(self, coarse_mask: np.ndarray) -> None:
        """Seed the click from a coarse saliency mask instead of a user gesture.

        This is the BiRefNet path. The coarse mask is *only* a prompt source:
        it never reaches the output, and SAM's mask is what gets saved.
        """
        prompt = auto_prompt_from_coarse_mask(coarse_mask)
        self._prompts = [prompt]
        self._sync_markers()
        self._request_segmentation()

    # -- interaction -------------------------------------------------------

    def _on_canvas_clicked(self, x: int, y: int, positive: bool) -> None:
        self._prompts.append(PointPrompt(x=x, y=y, positive=positive))
        self._sync_markers()
        self._request_segmentation()

    def _request_segmentation(self) -> None:
        image = self.canvas.image
        if image is None or not self._prompts or self._thread is not None:
            return
        runner = self._resolve_runner()
        if runner is None:
            self._announce("Aucun moteur de segmentation configuré.", status="error")
            return

        self._announce(f"Segmentation en cours ({len(self._prompts)} point(s))…")
        self._refresh_buttons()
        thread = QThread(self)
        worker = _SegmentationWorker(runner, image, self.prompts)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        self._thread, self._worker = thread, worker
        thread.start()

    def _sync_markers(self) -> None:
        """Mirror the prompt list onto the canvas.

        Called from every mutation rather than from the click handler, so
        undo, reset and the automatic prompt cannot drift out of sync.
        """
        self.canvas.set_markers(self.prompts)

    def _resolve_runner(self) -> Sam3Runner | None:
        if self._runner is None and self._runner_factory is not None:
            self._runner = self._runner_factory()
        return self._runner

    def _on_finished(self, result: object) -> None:
        self._teardown_thread()
        if not isinstance(result, SegmentationResult):
            self._announce("Résultat de segmentation inattendu.", status="error")
            return
        self._set_result(result)
        coverage = float(result.mask.mean()) * 100.0
        message = f"Masque obtenu — {coverage:.1f} % de l'image (score SAM {result.iou_score:.2f})."
        status = "ok"
        if coverage < _SUSPICIOUSLY_SMALL_PERCENT:
            # Measured on view01: a click in the bore scored 0.95 for 1.4 % of the
            # image, and a click on the lifting strap scored 0.97 for 1.8 %. The
            # score does not separate "the part" from "a bit of the rigging"; the
            # area does.
            message += (
                " Très petit : le clic est probablement tombé dans l'alésage ou sur"
                " l'élingue. Cliquez sur la matière pleine de la pièce."
            )
            status = "warn"
        elif result.is_ambiguous:
            message += " Candidats très proches : ajoutez un point pour lever le doute."
            status = "warn"
        self._announce(message, status=status)

    def _on_failed(self, message: str) -> None:
        self._teardown_thread()
        self._set_result(None)
        self._announce(message, status="error")

    def _teardown_thread(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
        self._thread, self._worker = None, None
        self._refresh_buttons()

    def _set_result(self, result: SegmentationResult | None) -> None:
        self._result = result
        self.canvas.set_mask(None if result is None else result.mask)
        self._refresh_buttons()
        self.mask_changed.emit(result)

    def _announce(self, message: str, *, status: str = "info") -> None:
        colour = {"ok": SUCCESS, "warn": ACCENT, "error": DANGER}.get(status, TEXT_DIM)
        self.status.setStyleSheet(f"color: {colour};")
        self.status.setText(message)
        self.status_changed.emit(message)

    def _refresh_buttons(self) -> None:
        busy = self._thread is not None
        has_image = self.canvas.image is not None
        self.load_button.setEnabled(not busy)
        self.auto_button.setEnabled(has_image and not busy)
        self.undo_button.setEnabled(bool(self._prompts) and not busy)
        self.reset_button.setEnabled(bool(self._prompts) and not busy)
        self.save_button.setEnabled(self._result is not None and not busy)

    # -- output ------------------------------------------------------------

    def rgba(self) -> np.ndarray:
        """Compose the RGBA cutout from SAM's mask.

        Raises rather than returning a placeholder: an accidental fully opaque
        or fully empty image would be silently re-processed by Pixal3D's own
        background removal, undoing the whole point of this panel.
        """
        image, result = self.canvas.image, self._result
        if image is None or result is None:
            raise Sam3Error("no mask to export: click the part first")
        return compose_rgba(image, result.mask)

    def save_rgba(self, destination: Path) -> Path:
        """Write the cutout as a PNG. PNG only -- JPEG has no alpha channel."""
        from PIL import Image

        target = Path(destination)
        if target.suffix.lower() != ".png":
            raise Sam3Error(f"the cutout must be a PNG to carry alpha, got {target.suffix or 'no'}")
        rgba = self.rgba()
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(rgba, mode="RGBA").save(target, format="PNG")
        self._announce(f"Découpe enregistrée : {target.name}", status="ok")
        return target
