"""Pick two photographs, get one aligned GLB (F15).

Face avant and face arriere in, `runs/<id>/aligned.glb` out. The panel owns no
geometry: it collects two paths, runs `trials.two_view` off the GUI thread, and
reports the two numbers that say whether to believe the result.

It does **not** fuse anything. The GLB holds both reconstructions in one frame,
unmerged, so that opening it answers the only question that matters at this
stage: do the two halves line up?
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QFileDialog,
    QGridLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pixaboost.gui.theme import DANGER, SUCCESS, TEXT_DIM, WARNING
from pixaboost.trials.two_view import TwoViewConfig, TwoViewResult

#: Injected so the panel never decides to spend GPU money, and so the tests
#: can substitute the whole chain. Signature: (config) -> TwoViewResult.
TwoViewRunner = Callable[[TwoViewConfig], TwoViewResult]


class _TwoViewWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)

    def __init__(self, runner: TwoViewRunner, config: TwoViewConfig) -> None:
        super().__init__()
        self._runner, self._config = runner, config

    def run(self) -> None:
        try:
            self.finished.emit(self._runner(self._config))
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")


class TwoViewPanel(QWidget):
    """Two photographs, no calibration, one GLB."""

    completed = pyqtSignal(object)

    def __init__(
        self,
        runs_root: Path,
        runner: TwoViewRunner | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._runs_root = Path(runs_root)
        self._runner = runner
        self._thread: QThread | None = None
        self._worker: _TwoViewWorker | None = None
        self._result: TwoViewResult | None = None
        self._build_ui()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Deux photos de la même pièce, sans calibration. La pose relative est "
            "déduite en cherchant la rotation dont la silhouette rendue colle au "
            "masque de la seconde photo.\n"
            "Le GLB contient les DEUX reconstructions dans un repère commun. "
            "Rien n'est fusionné : ouvre-le et regarde si les moitiés se superposent."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(intro)

        grid = QGridLayout()
        self.front_edit = QLineEdit()
        self.front_edit.setPlaceholderText("photo face avant")
        self.front_edit.setAccessibleName("Photo face avant")
        self.back_edit = QLineEdit()
        self.back_edit.setPlaceholderText("photo face arrière")
        self.back_edit.setAccessibleName("Photo face arrière")
        front_button = QPushButton("Parcourir…")
        front_button.clicked.connect(lambda: self._browse(self.front_edit, "face avant"))
        back_button = QPushButton("Parcourir…")
        back_button.clicked.connect(lambda: self._browse(self.back_edit, "face arrière"))
        for row, (label, edit, button) in enumerate(
            (
                ("Face avant", self.front_edit, front_button),
                ("Face arrière", self.back_edit, back_button),
            )
        ):
            grid.addWidget(QLabel(label), row, 0)
            grid.addWidget(edit, row, 1)
            grid.addWidget(button, row, 2)
        layout.addLayout(grid)

        self.run_button = QPushButton("Reconstruire depuis 2 vues")
        self.run_button.setToolTip(
            "Reconstruit chaque photo (cache d'abord), déduit la pose relative "
            "par rendu-comparaison, et écrit runs/<id>/aligned.glb."
        )
        self.run_button.clicked.connect(self.start)
        layout.addWidget(self.run_button)

        self.status = QLabel("Choisissez deux photos.")
        self.status.setWordWrap(True)
        self.status.setStyleSheet(f"color: {TEXT_DIM};")
        self.status.setAccessibleName("État de la reconstruction deux vues")
        layout.addWidget(self.status)
        layout.addStretch(1)
        self._refresh()

    # -- state -------------------------------------------------------------

    @property
    def is_running(self) -> bool:
        return self._thread is not None

    @property
    def result(self) -> TwoViewResult | None:
        return self._result

    def request(self) -> TwoViewConfig | None:
        """The config the two fields describe, or None if either is missing."""
        front, back = self.front_edit.text().strip(), self.back_edit.text().strip()
        if not front or not back:
            return None
        return TwoViewConfig(
            front_image=Path(front), back_image=Path(back), runs_root=self._runs_root
        )

    # -- actions -----------------------------------------------------------

    def _browse(self, edit: QLineEdit, what: str) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            f"Choisir la photo {what}",
            edit.text(),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;Tous les fichiers (*)",
        )
        if selected:
            edit.setText(selected)
            self._refresh()

    def start(self) -> None:
        if self._thread is not None:
            return
        config = self.request()
        if config is None:
            self._announce("Il faut les deux photos.", status="error")
            return
        for path in (config.front_image, config.back_image):
            if not path.is_file():
                self._announce(f"Introuvable : {path}", status="error")
                return
        if config.front_image == config.back_image:
            self._announce(
                "Les deux photos sont identiques : il n'y a aucune pose à déduire.",
                status="error",
            )
            return
        if self._runner is None:
            self._announce(
                "Aucun moteur configuré : la reconstruction mono-vue exige un Pod actif (F07).",
                status="error",
            )
            return

        self._announce("Reconstruction des deux vues, puis recherche de pose…")
        thread = QThread(self)
        worker = _TwoViewWorker(self._runner, config)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        self._thread, self._worker = thread, worker
        self._refresh()
        thread.start()

    def _on_finished(self, result: object) -> None:
        self._teardown()
        if not isinstance(result, TwoViewResult):
            self._announce("Résultat inattendu.", status="error")
            return
        self._result = result
        message = (
            f"GLB écrit : {result.glb_path.name} — "
            f"pose IoU {result.pose_iou:.2f}, accord {result.agreement_iou:.2f}."
        )
        status = "ok"
        if not result.is_trustworthy:
            message += (
                " Scores faibles : la seconde vue n'a probablement pas été localisée. "
                "Ne pas exploiter ce GLB."
            )
            status = "error"
        elif result.is_ambiguous:
            message += (
                " Pose ambiguë — normal sur une pièce axisymétrique : plusieurs azimuts "
                "sont équivalents et l'un vaut l'autre."
            )
            status = "warn"
        self._announce(message, status=status)
        self.completed.emit(result)

    def _on_failed(self, message: str) -> None:
        self._teardown()
        self._result = None
        self._announce(f"Échec : {message}", status="error")

    def _teardown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
        self._thread, self._worker = None, None
        self._refresh()

    def _announce(self, message: str, *, status: str = "info") -> None:
        colour = {"ok": SUCCESS, "warn": WARNING, "error": DANGER}.get(status, TEXT_DIM)
        self.status.setStyleSheet(f"color: {colour};")
        self.status.setText(message)

    def _refresh(self) -> None:
        self.run_button.setEnabled(self.request() is not None and self._thread is None)
