"""Pick two cutouts, get one aligned GLB (F15).

Face avant and face arriere in, `runs/<id>/aligned.glb` out. The panel owns no
geometry and no transport: it collects two paths, runs a free local preflight,
names what a cache miss will actually buy, and only then starts the billed work
off the GUI thread.

Inputs are the **RGBA cutouts** from the "Decoupe (SAM 3)" tab, not raw
photographs. One file then carries both the image Pixal3D reconstructs and the
silhouette the pose search matches against -- see ADR-0016.

It does **not** fuse anything. The GLB holds both reconstructions in one frame,
unmerged, so that opening it answers the only question that matters at this
stage: do the two halves line up?
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from PyQt6.QtCore import QObject, QThread, pyqtSignal
from PyQt6.QtWidgets import (
    QCheckBox,
    QFileDialog,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from pixaboost.backends.ssh_pod import CancelState
from pixaboost.gui.theme import DANGER, SUCCESS, TEXT_DIM, WARNING
from pixaboost.gui.two_view_adapter import TwoViewPreflight
from pixaboost.observability import TelemetryEvent
from pixaboost.trials.two_view import TwoViewConfig, TwoViewResult


class TwoViewEngine(Protocol):
    """The billed surface, injected so the panel never decides to spend money."""

    def preflight(self, config: TwoViewConfig) -> TwoViewPreflight:
        """Validate the inputs and resolve the cache, with no connection."""
        ...

    def run(
        self,
        config: TwoViewConfig,
        *,
        approve_existing_pod: bool,
        event_sink: Callable[[TelemetryEvent], None] | None = ...,
    ) -> TwoViewResult: ...

    def cancel(self) -> CancelState: ...


#: Built on the GUI thread at click time, so it can read the Pod fields once
#: and hand a worker a frozen copy.
TwoViewEngineFactory = Callable[[], TwoViewEngine]
#: Asks the user to approve a cache miss. Injected so the confirmation is
#: testable without driving a modal dialog.
TwoViewConfirm = Callable[[TwoViewPreflight], bool]

_UNKNOWN_CANCEL = "Annulation locale, état du Pod inconnu : vérifie-le avant de relancer."


class _PreflightWorker(QObject):
    """Local, free, and the only step allowed to run before confirmation."""

    finished = pyqtSignal(object)
    failed = pyqtSignal(str, bool)

    def __init__(self, engine: TwoViewEngine, config: TwoViewConfig) -> None:
        super().__init__()
        self._engine, self._config = engine, config

    def run(self) -> None:
        try:
            self.finished.emit(self._engine.preflight(self._config))
        except ValueError as error:
            self.failed.emit(str(error), True)
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}", False)


class _RunWorker(QObject):
    finished = pyqtSignal(object)
    failed = pyqtSignal(str)
    progress = pyqtSignal(object)

    def __init__(
        self, engine: TwoViewEngine, config: TwoViewConfig, *, approve_existing_pod: bool
    ) -> None:
        super().__init__()
        self._engine, self._config = engine, config
        self._approve = approve_existing_pod

    def run(self) -> None:
        try:
            self.finished.emit(
                self._engine.run(
                    self._config,
                    approve_existing_pod=self._approve,
                    event_sink=self.progress.emit,
                )
            )
        except Exception as error:
            self.failed.emit(f"{type(error).__name__}: {error}")


class _CancelWorker(QObject):
    finished = pyqtSignal(object)

    def __init__(self, engine: TwoViewEngine) -> None:
        super().__init__()
        self._engine = engine

    def run(self) -> None:
        try:
            self.finished.emit(self._engine.cancel())
        except Exception:
            self.finished.emit(CancelState.UNKNOWN)


class TwoViewPanel(QWidget):
    """Two cutouts, no calibration, one GLB."""

    completed = pyqtSignal(object)
    busy_changed = pyqtSignal(bool)

    def __init__(
        self,
        runs_root: Path,
        engine_factory: TwoViewEngineFactory | None = None,
        confirm: TwoViewConfirm | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._runs_root = Path(runs_root)
        self._engine_factory = engine_factory
        self._confirm = confirm or self._confirm_with_dialog
        self._engine: TwoViewEngine | None = None
        self._thread: QThread | None = None
        self._worker: QObject | None = None
        self._cancel_thread: QThread | None = None
        self._cancel_worker: _CancelWorker | None = None
        self._pending: TwoViewConfig | None = None
        self._result: TwoViewResult | None = None
        self._cancel_state: CancelState | None = None
        self._last_terminal: tuple[str, str] | None = None
        self._blocked = ""
        self._build_ui()

    # -- construction ------------------------------------------------------

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Deux découpes RGBA de la même pièce (onglet « Découpe (SAM 3) »), sans "
            "calibration. La pose relative est déduite en cherchant la rotation dont la "
            "silhouette rendue colle au masque de la seconde découpe.\n"
            "Le GLB contient les DEUX reconstructions dans un repère commun. "
            "Rien n'est fusionné : ouvre-le et regarde si les moitiés se superposent."
        )
        intro.setWordWrap(True)
        intro.setStyleSheet(f"color: {TEXT_DIM};")
        layout.addWidget(intro)

        grid = QGridLayout()
        self.front_edit = QLineEdit()
        self.front_edit.setPlaceholderText("découpe face avant (PNG RGBA)")
        self.front_edit.setAccessibleName("Découpe face avant")
        self.back_edit = QLineEdit()
        self.back_edit.setPlaceholderText("découpe face arrière (PNG RGBA)")
        self.back_edit.setAccessibleName("Découpe face arrière")
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

        self.opposite_check = QCheckBox("Les deux vues sont les faces OPPOSÉES de la pièce")
        self.opposite_check.setChecked(True)
        self.opposite_check.setToolTip(
            "Sur une pièce de révolution, l'avant et l'arrière ont la MÊME silhouette : "
            "aucun contour ne peut les distinguer et la recherche renverrait l'identité. "
            "Cochée, cette case affirme le demi-tour au lieu de le deviner. "
            "Décoche-la si la seconde vue est un autre angle, pas un retournement."
        )
        layout.addWidget(self.opposite_check)

        actions = QHBoxLayout()
        self.run_button = QPushButton("Reconstruire depuis 2 vues")
        self.run_button.setObjectName("primary")
        self.run_button.setToolTip(
            "Vérifie d'abord le cache local, sans rien dépenser. Un cache miss "
            "demande une confirmation nommant les vues concernées avant toute "
            "connexion au Pod."
        )
        self.run_button.clicked.connect(self.start)
        self.cancel_button = QPushButton("Annuler")
        self.cancel_button.setObjectName("danger")
        self.cancel_button.setEnabled(False)
        self.cancel_button.setToolTip(
            "Arrête l'essai. Une reconstruction pas encore commencée n'est jamais achetée."
        )
        self.cancel_button.clicked.connect(self.cancel)
        actions.addWidget(self.run_button, 2)
        actions.addWidget(self.cancel_button, 1)
        layout.addLayout(actions)

        self.status = QLabel("Choisissez deux découpes RGBA.")
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
    def is_cancelling(self) -> bool:
        return self._cancel_thread is not None

    @property
    def is_busy(self) -> bool:
        """Whether a thread parented to this widget is still alive.

        Destroying the panel while one runs aborts the process, so the window
        consults this before closing -- and so must any test that clicks
        `cancel`.
        """
        return self.is_running or self.is_cancelling

    @property
    def result(self) -> TwoViewResult | None:
        return self._result

    def request(self) -> TwoViewConfig | None:
        """The config the two fields describe, or None if either is missing."""
        front, back = self.front_edit.text().strip(), self.back_edit.text().strip()
        if not front or not back:
            return None
        return TwoViewConfig(
            front_image=Path(front),
            back_image=Path(back),
            runs_root=self._runs_root,
            opposite_faces=self.opposite_check.isChecked(),
        )

    def set_blocked(self, reason: str) -> None:
        """Refuse to start while another trial owns the Pod.

        Idempotent: `_refresh` re-emits `busy_changed`, whose listener calls
        back into here, so an unchanged reason must not restart that cycle.
        """
        if reason == self._blocked:
            return
        self._blocked = reason
        self._refresh()

    # -- actions -----------------------------------------------------------

    def _browse(self, edit: QLineEdit, what: str) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            f"Choisir la découpe {what}",
            edit.text(),
            "Découpes RGBA (*.png);;Tous les fichiers (*)",
        )
        if selected:
            edit.setText(selected)
            self._refresh()

    def start(self) -> None:
        if self._thread is not None:
            return
        if self._blocked:
            self._announce(self._blocked, status="error")
            return
        config = self.request()
        if config is None:
            self._announce("Il faut les deux découpes.", status="error")
            return
        for path in (config.front_image, config.back_image):
            if not path.is_file():
                self._announce(f"Introuvable : {path}", status="error")
                return
        if config.front_image == config.back_image:
            self._announce(
                "Les deux découpes sont identiques : il n'y a aucune pose à déduire.",
                status="error",
            )
            return
        if self._engine_factory is None:
            self._announce(
                "Aucun moteur configuré : la reconstruction mono-vue exige un Pod actif (F07).",
                status="error",
            )
            return
        try:
            self._engine = self._engine_factory()
        except Exception as error:
            self._announce(f"Configuration du Pod invalide : {error}", status="error")
            return

        self._pending = config
        self._cancel_state = None
        self._last_terminal = None
        self._announce("Vérification locale du cache — rien n'est dépensé…")
        self._start_worker(_PreflightWorker(self._engine, config), self._connect_preflight)

    def cancel(self) -> None:
        if self._engine is None or self._cancel_thread is not None:
            return
        self._announce("Annulation demandée…", status="warn")
        worker = _CancelWorker(self._engine)
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        worker.finished.connect(self._on_cancelled)
        self._cancel_thread, self._cancel_worker = thread, worker
        self._refresh()
        thread.start()

    # -- plumbing ----------------------------------------------------------

    def _start_worker(self, worker: QObject, connect: Callable[[QObject], None]) -> None:
        thread = QThread(self)
        worker.moveToThread(thread)
        thread.started.connect(worker.run)  # type: ignore[attr-defined]
        connect(worker)
        self._thread, self._worker = thread, worker
        self._refresh()
        thread.start()

    def _connect_preflight(self, worker: QObject) -> None:
        assert isinstance(worker, _PreflightWorker)
        worker.finished.connect(self._on_prepared)
        worker.failed.connect(self._on_preflight_failed)

    def _connect_run(self, worker: QObject) -> None:
        assert isinstance(worker, _RunWorker)
        worker.finished.connect(self._on_finished)
        worker.failed.connect(self._on_failed)
        worker.progress.connect(self._on_progress)

    def _on_prepared(self, preflight: object) -> None:
        self._teardown()
        if not isinstance(preflight, TwoViewPreflight) or self._pending is None:
            self._announce("Préflight inattendu.", status="error")
            return
        config, self._pending = self._pending, None
        if preflight.approval_required and not self._confirm(preflight):
            self._announce(
                "Essai annulé avant toute connexion au Pod.", status="warn"
            )
            return
        self._announce(
            "Lecture du cache local — aucun usage GPU…"
            if not preflight.approval_required
            else "Reconstruction sur le Pod, puis recherche de pose…"
        )
        assert self._engine is not None
        self._start_worker(
            _RunWorker(
                self._engine, config, approve_existing_pod=preflight.approval_required
            ),
            self._connect_run,
        )

    def _on_preflight_failed(self, message: str, is_input_error: bool) -> None:
        self._teardown()
        self._pending = None
        if is_input_error:
            self._announce(
                f"{message}\nCe panneau attend les découpes RGBA produites par "
                "l'onglet « Découpe (SAM 3) » : c'est le masque que tu as validé qui "
                "détermine la pose.",
                status="error",
            )
            return
        self._announce(f"Préflight impossible : {message}", status="error")

    def _on_progress(self, event: object) -> None:
        message = str(getattr(event, "message", "")).strip()
        if message:
            self._announce(message)

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
        self._announce_terminal(message, status)
        self.completed.emit(result)

    def _on_failed(self, message: str) -> None:
        self._teardown()
        self._result = None
        self._announce_terminal(f"Échec : {message}", "error")

    def _on_cancelled(self, state: object) -> None:
        if self._cancel_thread is not None:
            self._cancel_thread.quit()
            self._cancel_thread.wait(5000)
        self._cancel_thread, self._cancel_worker = None, None
        self._cancel_state = state if isinstance(state, CancelState) else CancelState.UNKNOWN
        if self._last_terminal is not None:
            self._render_terminal()
        elif self._cancel_state is CancelState.UNKNOWN:
            self._announce(_UNKNOWN_CANCEL, status="error")
        self._refresh()

    def _announce_terminal(self, message: str, status: str) -> None:
        self._last_terminal = (message, status)
        self._render_terminal()

    def _render_terminal(self) -> None:
        """Show the outcome, never losing an unresolved cancellation.

        The cancel acknowledgement and the run's own result arrive on the GUI
        thread in either order, so whichever lands second must not erase the
        other. A *successful* run is the one case that needs no warning: both
        reconstructions finished, so the Pod is not still working on them.
        """
        if self._last_terminal is None:
            return
        message, status = self._last_terminal
        if self._cancel_state is CancelState.UNKNOWN and status != "ok":
            message, status = f"{message} {_UNKNOWN_CANCEL}", "error"
        self._announce(message, status=status)

    def _teardown(self) -> None:
        if self._thread is not None:
            self._thread.quit()
            self._thread.wait(5000)
        self._thread, self._worker = None, None
        self._refresh()

    def _confirm_with_dialog(self, preflight: TwoViewPreflight) -> bool:
        names = ", ".join(path.name for path in preflight.missing)
        count = len(preflight.missing)
        answer = QMessageBox.question(
            self,
            "Confirmer l'utilisation du Pod existant",
            f"{count} vue(s) absente(s) du cache local : {names}.\n\n"
            f"Continuer lancera {count} reconstruction(s) sur le Pod déjà actif et son "
            "temps GPU peut être facturé. PixaBoost n'achètera, ne rechargera, ne "
            "provisionnera, ne démarrera et n'activera aucun crédit ou Pod. Cette "
            "autorisation est éphémère et valable pour cet essai seulement.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer is QMessageBox.StandardButton.Yes

    def _announce(self, message: str, *, status: str = "info") -> None:
        colour = {"ok": SUCCESS, "warn": WARNING, "error": DANGER}.get(status, TEXT_DIM)
        self.status.setStyleSheet(f"color: {colour};")
        self.status.setText(message)

    def _refresh(self) -> None:
        busy = self._thread is not None
        self.run_button.setEnabled(
            self.request() is not None and not busy and not self._blocked
        )
        self.cancel_button.setEnabled(busy and self._cancel_thread is None)
        for widget in (self.front_edit, self.back_edit, self.opposite_check):
            widget.setEnabled(not busy)
        self.busy_changed.emit(busy)
