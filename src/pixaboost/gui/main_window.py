"""Main PyQt6 window for observable, local PixaBoost experiments."""

from __future__ import annotations

import time
from collections import deque
from pathlib import Path

from PyQt6.QtCore import Qt, QTimer, pyqtSlot
from PyQt6.QtGui import QCloseEvent, QFont
from PyQt6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPlainTextEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSplitter,
    QStatusBar,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from pixaboost.backends.ssh_pod import CancelState
from pixaboost.gui.model import (
    ArtifactEntry,
    CommandSpec,
    RunResult,
    RunState,
    default_commands,
    discover_artifacts,
)
from pixaboost.gui.remote_trial import (
    RemoteRunnerFactory,
    RemoteTrialController,
    RemoteTrialDefaults,
    RemoteTrialRequest,
    RemoteTrialResult,
)
from pixaboost.gui.runner import CommandController
from pixaboost.observability import EVENT_PREFIX, TelemetryEvent

_UI_STATE_LABELS = {
    RunState.IDLE: "Prêt",
    RunState.STARTING: "Démarrage",
    RunState.RUNNING: "En cours",
    RunState.CANCELLING: "Arrêt demandé",
    RunState.SUCCEEDED: "Réussi",
    RunState.FAILED: "Échoué",
    RunState.CANCELLED: "Arrêté",
}


class MainWindow(QMainWindow):
    """Small experiment console for local checks and approved existing-Pod trials."""

    def __init__(
        self,
        *,
        commands: tuple[CommandSpec, ...] | None = None,
        repo_root: Path | None = None,
        remote_runner_factory: RemoteRunnerFactory | None = None,
        remote_defaults: RemoteTrialDefaults | None = None,
    ) -> None:
        super().__init__()
        self.repo_root = (repo_root or Path(__file__).resolve().parents[3]).resolve()
        self.commands = commands or default_commands(self.repo_root)
        if not self.commands:
            raise ValueError("the GUI needs at least one command")

        self.controller = CommandController(self)
        self.remote_controller = RemoteTrialController(remote_runner_factory, self)
        self.remote_defaults = remote_defaults or RemoteTrialDefaults()
        self._started_at = 0.0
        self._close_requested = False
        self._remote_cancel_acknowledged = False
        self._remote_cancel_state: CancelState | None = None
        self._remote_close_after_cancel = False
        self._pending_remote_request: RemoteTrialRequest | None = None
        self._active_kind: str | None = None
        self._has_detailed_telemetry = False
        self._pending_log_lines: deque[str] = deque(maxlen=5_000)

        self._elapsed_timer = QTimer(self)
        self._elapsed_timer.setInterval(250)
        self._elapsed_timer.timeout.connect(self._update_elapsed)
        self._log_flush_timer = QTimer(self)
        self._log_flush_timer.setInterval(40)
        self._log_flush_timer.setSingleShot(True)
        self._log_flush_timer.timeout.connect(self._flush_logs)

        self.setWindowTitle("PixaBoost — Laboratoire local")
        self._status_bar = QStatusBar(self)
        self.setStatusBar(self._status_bar)
        self.setMinimumSize(760, 480)
        self.resize(1120, 720)
        self._build_ui()
        self._connect_signals()
        self._update_selection(0)
        self._on_state_changed(RunState.IDLE)
        self.refresh_artifacts()

    def _build_ui(self) -> None:
        central = QWidget()
        root = QVBoxLayout(central)
        root.setContentsMargins(20, 16, 20, 16)
        root.setSpacing(14)

        title = QLabel("PixaBoost")
        title.setObjectName("title")
        subtitle = QLabel(
            "Laboratoire d'essais — commandes, progression, télémétrie et artefacts"
        )
        subtitle.setObjectName("subtitle")
        subtitle.setWordWrap(True)
        root.addWidget(title)
        root.addWidget(subtitle)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.controls_scroll = QScrollArea()
        self.controls_scroll.setWidgetResizable(True)
        self.controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.controls_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.controls_scroll.setAccessibleName("Panneau des essais")
        self.controls_scroll.setAccessibleDescription(
            "Commandes locales et état de disponibilité de la reconstruction GPU."
        )
        self.controls_scroll.setWidget(self._build_controls())
        self.tabs = self._build_tabs()
        splitter.addWidget(self.controls_scroll)
        splitter.addWidget(self.tabs)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([340, 740])
        root.addWidget(splitter, 1)
        self.setCentralWidget(central)
        self._status_bar.showMessage("Prêt — aucun essai en cours")

    def _build_controls(self) -> QWidget:
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 10, 0)
        layout.setSpacing(12)

        experiment = QGroupBox("Essai local")
        experiment_layout = QVBoxLayout(experiment)
        self.command_combo = QComboBox()
        self.command_combo.setAccessibleName("Commande d'essai")
        self.command_combo.setAccessibleDescription(
            "Choisit la vérification locale et gratuite à exécuter."
        )
        for command in self.commands:
            self.command_combo.addItem(command.label, command.key)
        self.description_label = QLabel()
        self.description_label.setWordWrap(True)
        self.description_label.setObjectName("dim")
        self.cost_label = QLabel()
        self.cost_label.setObjectName("dim")
        experiment_layout.addWidget(self.command_combo)
        experiment_layout.addWidget(self.description_label)
        experiment_layout.addWidget(self.cost_label)

        actions = QHBoxLayout()
        self.start_button = QPushButton("Lancer")
        self.start_button.setObjectName("primary")
        self.start_button.setDefault(True)
        self.start_button.setAccessibleName("Lancer l'essai")
        self.start_button.setAccessibleDescription(
            "Démarre la commande locale sélectionnée sans bloquer l'interface."
        )
        self.cancel_button = QPushButton("Arreter")
        self.cancel_button.setObjectName("danger")
        self.cancel_button.setAccessibleName("Arrêter l'essai")
        self.cancel_button.setAccessibleDescription(
            "Demande l'arrêt du processus local actuellement suivi."
        )
        actions.addWidget(self.start_button, 2)
        actions.addWidget(self.cancel_button, 1)
        experiment_layout.addLayout(actions)
        layout.addWidget(experiment)

        gpu = QGroupBox("Essai mono-vue — Pod existant")
        gpu_layout = QVBoxLayout(gpu)
        gpu_form = QFormLayout()
        image_row = QWidget()
        image_layout = QHBoxLayout(image_row)
        image_layout.setContentsMargins(0, 0, 0, 0)
        self.gpu_image_edit = QLineEdit(str(self.remote_defaults.image_path or ""))
        self.gpu_image_edit.setAccessibleName("Image de reconstruction")
        self.gpu_image_edit.setAccessibleDescription(
            "Image locale utilisée pour la reconstruction mono-vue."
        )
        self.gpu_browse_button = QPushButton("Parcourir")
        self.gpu_browse_button.setAccessibleName("Choisir une image")
        self.gpu_browse_button.setAccessibleDescription("Ouvre le sélecteur de fichier image.")
        image_layout.addWidget(self.gpu_image_edit, 1)
        image_layout.addWidget(self.gpu_browse_button)
        gpu_form.addRow("Image", image_row)

        self.gpu_host_edit = QLineEdit(self.remote_defaults.host)
        self.gpu_user_edit = QLineEdit(self.remote_defaults.username)
        self.gpu_key_edit = QLineEdit(str(self.remote_defaults.private_key_path or ""))
        self.gpu_known_hosts_edit = QLineEdit(str(self.remote_defaults.known_hosts_path or ""))
        self.gpu_revision_edit = QLineEdit(self.remote_defaults.expected_pixal3d_sha)
        remote_fields = (
            (
                self.gpu_host_edit,
                "Hôte SSH",
                "Hôte du Pod déjà actif; aucune ressource n'est provisionnée.",
            ),
            (
                self.gpu_user_edit,
                "Utilisateur SSH",
                "Compte SSH fourni pour le Pod existant.",
            ),
            (
                self.gpu_key_edit,
                "Chemin de clé SSH",
                "Chemin local de la clé privée; son contenu n'est jamais affiché.",
            ),
            (
                self.gpu_known_hosts_edit,
                "Fichier known_hosts",
                "Clés hôte de confiance; les hôtes inconnus sont refusés.",
            ),
            (
                self.gpu_revision_edit,
                "Révision Pixal3D",
                "SHA Git complet attendu et vérifié avant toute inférence.",
            ),
        )
        for field, name, description in remote_fields:
            field.setAccessibleName(name)
            field.setAccessibleDescription(description)
        gpu_form.addRow("Hôte", self.gpu_host_edit)
        gpu_form.addRow("Utilisateur", self.gpu_user_edit)
        gpu_form.addRow("Clé privée", self.gpu_key_edit)
        gpu_form.addRow("known_hosts", self.gpu_known_hosts_edit)
        gpu_form.addRow("Révision", self.gpu_revision_edit)
        gpu_layout.addLayout(gpu_form)

        gpu_actions = QHBoxLayout()
        self.gpu_button = QPushButton("Lancer une reconstruction")
        self.gpu_button.setObjectName("primary")
        self.gpu_button.setEnabled(False)
        self.gpu_button.setAccessibleName("Lancer une reconstruction GPU")
        self.gpu_button.setAccessibleDescription(
            "Utilise uniquement le Pod existant après confirmation explicite sur un cache miss."
        )
        self.gpu_cancel_button = QPushButton("Annuler")
        self.gpu_cancel_button.setObjectName("danger")
        self.gpu_cancel_button.setAccessibleName("Annuler la reconstruction distante")
        self.gpu_cancel_button.setAccessibleDescription(
            "Demande l'annulation et indique explicitement si l'état distant reste inconnu."
        )
        gpu_actions.addWidget(self.gpu_button, 2)
        gpu_actions.addWidget(self.gpu_cancel_button, 1)
        gpu_layout.addLayout(gpu_actions)
        self.gpu_reason = QLabel()
        self.gpu_reason.setWordWrap(True)
        self.gpu_reason.setObjectName("warning")
        gpu_layout.addWidget(self.gpu_reason)
        layout.addWidget(gpu)
        layout.addStretch(1)
        return panel

    def _build_tabs(self) -> QTabWidget:
        tabs = QTabWidget()
        tabs.setAccessibleName("Résultats de l'essai")
        tabs.setAccessibleDescription("Suivi en direct et inventaire des artefacts locaux.")
        tabs.addTab(self._build_live_tab(), "Suivi")
        tabs.addTab(self._build_artifacts_tab(), "Artefacts locaux")
        tabs.addTab(self._build_segmentation_tab(), "Découpe (SAM 3)")
        return tabs

    def _build_segmentation_tab(self) -> QWidget:
        """Click-to-segment panel (F14).

        The runner is built lazily, on the first click: constructing it downloads
        3.4 GB of gated weights, which must never happen because someone opened
        a tab. Until then the panel simply has no engine, and says so.
        """
        from pixaboost.gui.segmentation_view import SegmentationPanel

        def build_runner() -> object:
            from pixaboost.backends.sam3 import Sam3TrackerRunner

            return Sam3TrackerRunner()

        def build_saliency() -> object:
            from pixaboost.backends.birefnet import BiRefNetRunner

            return BiRefNetRunner()

        self.segmentation_panel = SegmentationPanel(
            runner_factory=build_runner,  # type: ignore[arg-type]
            saliency_factory=build_saliency,  # type: ignore[arg-type]
        )
        return self.segmentation_panel

    def _build_live_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        self.result_banner = QLabel("Résultat — aucun essai exécuté dans cette session.")
        self.result_banner.setObjectName("result")
        self.result_banner.setProperty("status", "neutral")
        self.result_banner.setTextFormat(Qt.TextFormat.PlainText)
        self.result_banner.setWordWrap(True)
        self.result_banner.setAccessibleName("Résultat du dernier essai")
        self.result_banner.setAccessibleDescription(self.result_banner.text())
        layout.addWidget(self.result_banner)

        metrics = QGroupBox("Telemetrie")
        form = QFormLayout(metrics)
        self.state_value = QLabel("—")
        self.phase_value = QLabel("—")
        self.elapsed_value = QLabel("0.0 s")
        self.exit_value = QLabel("—")
        for label in (self.state_value, self.phase_value, self.elapsed_value, self.exit_value):
            font = QFont(label.font())
            font.setBold(True)
            label.setFont(font)
        self.state_value.setAccessibleName("État de l'essai")
        self.phase_value.setAccessibleName("Phase de l'essai")
        self.elapsed_value.setAccessibleName("Temps écoulé")
        self.exit_value.setAccessibleName("Code retour")
        form.addRow("Etat", self.state_value)
        form.addRow("Phase", self.phase_value)
        form.addRow("Temps ecoule", self.elapsed_value)
        form.addRow("Code retour", self.exit_value)
        layout.addWidget(metrics)

        command_label = QLabel("Commande courante (sanitisee)")
        command_label.setObjectName("dim")
        self.command_value = QLineEdit()
        self.command_value.setReadOnly(True)
        self.command_value.setAccessibleName("Commande courante sanitisée")
        self.command_value.setAccessibleDescription(
            "Commande exécutée, avec les informations sensibles masquées."
        )
        layout.addWidget(command_label)
        layout.addWidget(self.command_value)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setValue(0)
        self.progress_bar.setFormat("%p%")
        self.progress_bar.setAccessibleName("Progression de l'essai")
        self.progress_bar.setAccessibleDescription(
            "Progression déterminée quand la commande publie de la télémétrie."
        )
        layout.addWidget(self.progress_bar)

        self.log_view = QPlainTextEdit()
        self.log_view.setObjectName("log")
        self.log_view.setReadOnly(True)
        self.log_view.setAccessibleName("Journal de l'essai")
        self.log_view.setAccessibleDescription(
            "Sorties textuelles sanitisées du processus, limitées aux 2500 dernières lignes."
        )
        document = self.log_view.document()
        assert document is not None
        document.setMaximumBlockCount(2500)
        layout.addWidget(self.log_view, 1)
        return tab

    def _build_artifacts_tab(self) -> QWidget:
        tab = QWidget()
        self.artifacts_tab = tab
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(14, 14, 14, 14)
        explanation = QLabel(
            "Inventaire de navigation uniquement. Un fichier local n'est une preuve que si son "
            "manifeste et sa verification officielle sont valides."
        )
        explanation.setWordWrap(True)
        explanation.setObjectName("dim")
        layout.addWidget(explanation)

        self.artifact_list = QTreeWidget()
        self.artifact_list.setHeaderLabels(["Type", "Chemin", "Taille"])
        self.artifact_list.setAlternatingRowColors(True)
        self.artifact_list.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.artifact_list.setAccessibleName("Artefacts locaux")
        self.artifact_list.setAccessibleDescription(
            "Fichiers produits localement; l'artefact du dernier essai est sélectionné."
        )
        header = self.artifact_list.header()
        assert header is not None
        header.setStretchLastSection(False)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        layout.addWidget(self.artifact_list, 1)
        self.refresh_button = QPushButton("Actualiser")
        self.refresh_button.setAccessibleName("Actualiser les artefacts")
        self.refresh_button.setAccessibleDescription(
            "Relit l'inventaire léger des artefacts locaux."
        )
        self.refresh_button.clicked.connect(self.refresh_artifacts)
        layout.addWidget(self.refresh_button, 0, Qt.AlignmentFlag.AlignRight)
        return tab

    def _connect_signals(self) -> None:
        self.command_combo.currentIndexChanged.connect(self._update_selection)
        self.start_button.clicked.connect(self.start_selected)
        self.cancel_button.clicked.connect(self.cancel_current)
        self.controller.state_changed.connect(self._on_local_state_changed)
        self.controller.command_changed.connect(self.command_value.setText)
        self.controller.telemetry.connect(self._on_telemetry)
        self.controller.log_line.connect(self._on_log_line)
        self.controller.completed.connect(self._on_local_completed)

        self.gpu_browse_button.clicked.connect(self._browse_remote_image)
        self.gpu_button.clicked.connect(self.start_remote_trial)
        self.gpu_cancel_button.clicked.connect(self.cancel_remote_trial)
        for field in (
            self.gpu_image_edit,
            self.gpu_host_edit,
            self.gpu_user_edit,
            self.gpu_key_edit,
            self.gpu_known_hosts_edit,
            self.gpu_revision_edit,
        ):
            field.textChanged.connect(self._update_action_availability)
        self.remote_controller.state_changed.connect(self._on_remote_state_changed)
        self.remote_controller.command_changed.connect(self.command_value.setText)
        self.remote_controller.telemetry.connect(self._on_telemetry)
        self.remote_controller.log_line.connect(self._on_log_line)
        self.remote_controller.completed.connect(self._on_remote_completed)
        self.remote_controller.cancellation_changed.connect(self._on_remote_cancellation)
        self.remote_controller.prepared.connect(self._on_remote_prepared)
        self.remote_controller.preparation_failed.connect(
            self._on_remote_preparation_failed
        )
        self.remote_controller.availability_changed.connect(
            self._on_remote_availability_changed
        )
        self._update_action_availability()

    @property
    def selected_command(self) -> CommandSpec:
        return self.commands[self.command_combo.currentIndex()]

    @pyqtSlot(int)
    def _update_selection(self, _index: int) -> None:
        command = self.selected_command
        self.description_label.setText(command.description)
        self.cost_label.setText(command.cost)
        if self.controller.state is RunState.IDLE or self.controller.state.is_terminal:
            self.command_value.setText(command.display_command)

    @pyqtSlot()
    def start_selected(self) -> None:
        if not self.controller.can_start or self._remote_is_active():
            self._status_bar.showMessage("Une commande est deja en cours", 5000)
            return
        self._active_kind = "local"
        self._prepare_run_display("Préparation du processus")
        if not self.controller.start(self.selected_command):
            self._active_kind = None
            self._elapsed_timer.stop()
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(0)
            self._set_result_banner("Résultat — lancement refusé.", "failure")
            self._status_bar.showMessage("La commande n'a pas pu etre reservee", 5000)
        self._update_action_availability()

    def _prepare_run_display(self, phase: str) -> None:
        self._log_flush_timer.stop()
        self._pending_log_lines.clear()
        self.log_view.clear()
        self._has_detailed_telemetry = False
        self._set_result_banner("Résultat — essai en cours…", "running")
        self.exit_value.setText("—")
        self.phase_value.setText(phase)
        self.progress_bar.setRange(0, 0)
        self._started_at = time.monotonic()
        self._elapsed_timer.start()

    @pyqtSlot()
    def cancel_current(self) -> None:
        if not self.controller.cancel():
            self._status_bar.showMessage("Aucun processus local a arreter", 5000)

    @pyqtSlot()
    def _browse_remote_image(self) -> None:
        selected, _filter = QFileDialog.getOpenFileName(
            self,
            "Choisir une image",
            self.gpu_image_edit.text(),
            "Images (*.png *.jpg *.jpeg *.webp *.bmp);;Tous les fichiers (*)",
        )
        if selected:
            self.gpu_image_edit.setText(selected)

    def _remote_request(self) -> RemoteTrialRequest:
        return RemoteTrialRequest(
            image_path=Path(self.gpu_image_edit.text()).expanduser().resolve(),
            host=self.gpu_host_edit.text().strip(),
            username=self.gpu_user_edit.text().strip(),
            private_key_path=Path(self.gpu_key_edit.text()).expanduser().resolve(),
            known_hosts_path=Path(self.gpu_known_hosts_edit.text()).expanduser().resolve(),
            expected_pixal3d_sha=self.gpu_revision_edit.text().strip(),
            project_git_sha=self.remote_defaults.project_git_sha,
        )

    def _remote_validation_error(self) -> str:
        if not self.remote_controller.is_configured:
            return (
                "F07 non configurée : aucun runner SSH n'a été fourni. "
                "Aucun achat, recharge ou provisionnement n'est possible."
            )
        try:
            self._remote_request()
        except ValueError as error:
            return str(error)
        return ""

    @pyqtSlot()
    def start_remote_trial(self) -> None:
        if self._local_is_active() or not self.remote_controller.can_start:
            self._status_bar.showMessage("Un autre essai est déjà en cours", 5000)
            return
        try:
            request = self._remote_request()
            self._active_kind = "remote"
            self._pending_remote_request = request
            self._remote_cancel_acknowledged = False
            self._remote_cancel_state = None
            self._remote_close_after_cancel = False
            self._close_requested = False
            self._prepare_run_display("Préflight du cache local")
            if not self.remote_controller.prepare(request):
                raise RuntimeError("le préflight local n'a pas pu être réservé")
        except Exception as error:
            message = f"Préflight local impossible : {error}"
            self._active_kind = None
            self._pending_remote_request = None
            self._elapsed_timer.stop()
            self._set_result_banner(f"Résultat — {message}", "failure")
            self._status_bar.showMessage(message, 10000)
            self._update_action_availability()
            return

        self._update_action_availability()

    @pyqtSlot(bool)
    def _on_remote_prepared(self, cache_hit: bool) -> None:
        request = self._pending_remote_request
        if request is None:
            self.remote_controller.discard_prepared()
            return
        if self._close_requested:
            self.remote_controller.discard_prepared()
            self._active_kind = None
            self._pending_remote_request = None
            QTimer.singleShot(0, self.close)
            return
        approved = cache_hit or self._confirm_remote_use(request)
        if not approved:
            self.remote_controller.discard_prepared()
            self._active_kind = None
            self._pending_remote_request = None
            self._elapsed_timer.stop()
            self._set_result_banner(
                "Résultat — essai annulé avant toute connexion au Pod.",
                "cancelled",
            )
            self._status_bar.showMessage("Essai annulé avant toute connexion au Pod", 7000)
            self._update_action_availability()
            return

        self.phase_value.setText(
            "Lecture du cache local" if cache_hit else "Connexion au Pod existant"
        )
        if not self.remote_controller.start_prepared(approve_existing_pod=not cache_hit):
            self._active_kind = None
            self._pending_remote_request = None
            self._elapsed_timer.stop()
            self._set_result_banner("Résultat — lancement distant refusé.", "failure")
        self._update_action_availability()

    @pyqtSlot(str)
    def _on_remote_preparation_failed(self, message: str) -> None:
        self._active_kind = None
        self._pending_remote_request = None
        self._elapsed_timer.stop()
        self._set_result_banner(f"Résultat — préflight local échoué : {message}", "failure")
        self._status_bar.showMessage(message, 10000)
        self._update_action_availability()

    def _confirm_remote_use(self, request: RemoteTrialRequest) -> bool:
        answer = QMessageBox.question(
            self,
            "Confirmer l'utilisation du Pod existant",
            f"Image : {request.image_path}\n"
            f"Cible : {request.username}@{request.host}\n"
            f"Révision Pixal3D : {request.expected_pixal3d_sha}\n\n"
            "Le cache local ne contient pas cet artefact. Continuer utilisera le Pod "
            "déjà actif et son temps GPU peut être facturé. PixaBoost n'achètera, ne "
            "rechargera, ne provisionnera, ne démarrera et n'activera aucun crédit ou "
            "Pod. Cette autorisation est éphémère et valable pour cet essai seulement.",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        return answer is QMessageBox.StandardButton.Yes

    @pyqtSlot()
    def cancel_remote_trial(self) -> None:
        self._remote_close_after_cancel = False
        self._close_requested = False
        if not self.remote_controller.cancel():
            self._status_bar.showMessage("Aucun processus distant actif à annuler", 5000)

    @pyqtSlot(object)
    def _on_local_state_changed(self, state: RunState) -> None:
        if self._active_kind in {None, "local"}:
            self._on_state_changed(state)

    @pyqtSlot(object)
    def _on_remote_state_changed(self, state: RunState) -> None:
        if self._active_kind in {None, "remote"}:
            self._on_state_changed(state)

    @pyqtSlot(object)
    def _on_state_changed(self, state: RunState) -> None:
        self.state_value.setText(_UI_STATE_LABELS[state])
        lifecycle_phase = {
            RunState.IDLE: "En attente",
            RunState.STARTING: "Démarrage du processus",
            RunState.RUNNING: "Processus en cours",
            RunState.CANCELLING: "Arrêt du processus en cours",
            RunState.SUCCEEDED: "Processus terminé",
            RunState.FAILED: "Processus échoué",
            RunState.CANCELLED: "Processus arrêté",
        }
        if state is RunState.IDLE or not self._has_detailed_telemetry:
            self.phase_value.setText(lifecycle_phase[state])
        if state.is_terminal:
            was_indeterminate = self.progress_bar.minimum() == self.progress_bar.maximum() == 0
            self.progress_bar.setRange(0, 1000)
            if state is RunState.SUCCEEDED:
                self.progress_bar.setValue(1000)
            elif was_indeterminate:
                self.progress_bar.setValue(0)
            self._elapsed_timer.stop()
        self._update_action_availability()

    def _local_is_active(self) -> bool:
        return self.controller.state in {
            RunState.STARTING,
            RunState.RUNNING,
            RunState.CANCELLING,
        }

    def _remote_is_active(self) -> bool:
        return self.remote_controller.owns_single_flight

    @pyqtSlot()
    def _on_remote_availability_changed(self) -> None:
        self._update_action_availability()
        if self._close_requested and not self.remote_controller.has_pending_workers:
            QTimer.singleShot(0, self.close)

    @pyqtSlot()
    def _update_action_availability(self) -> None:
        local_active = self._local_is_active()
        remote_active = self._remote_is_active()
        active = local_active or remote_active
        self.command_combo.setEnabled(not active)
        self.start_button.setEnabled(not active and self.controller.can_start)
        self.cancel_button.setEnabled(
            local_active and self.controller.state in {RunState.STARTING, RunState.RUNNING}
        )

        validation_error = self._remote_validation_error()
        configuration_enabled = not active
        for field in (
            self.gpu_image_edit,
            self.gpu_host_edit,
            self.gpu_user_edit,
            self.gpu_key_edit,
            self.gpu_known_hosts_edit,
            self.gpu_revision_edit,
        ):
            field.setEnabled(configuration_enabled)
        self.gpu_browse_button.setEnabled(configuration_enabled)
        self.gpu_reason.setText(
            validation_error
            or (
                "Prêt. Le préflight vérifie d'abord le cache local. Un cache miss "
                "demandera une confirmation éphémère avant toute connexion; aucun achat, "
                "recharge ou provisionnement n'est possible."
            )
        )
        self.gpu_button.setEnabled(
            not active and self.remote_controller.can_start and not validation_error
        )
        self.gpu_cancel_button.setEnabled(
            self.remote_controller.has_worker
            and self.remote_controller.state in {RunState.STARTING, RunState.RUNNING}
        )

    @pyqtSlot(object)
    def _on_remote_cancellation(self, state: CancelState) -> None:
        if state is CancelState.ACKNOWLEDGED:
            self._remote_cancel_acknowledged = True
            self._remote_cancel_state = state
            self._close_requested = self._remote_close_after_cancel
        elif state is CancelState.UNKNOWN:
            self._remote_cancel_acknowledged = False
            self._remote_cancel_state = state
            self._close_requested = False
        self._update_action_availability()

    @pyqtSlot(object)
    def _on_telemetry(self, event: TelemetryEvent) -> None:
        is_process_lifecycle = (
            event.phase == "processus"
            and not event.stage
            and event.progress is None
            and event.artifact is None
        )
        self._has_detailed_telemetry = self._has_detailed_telemetry or not is_process_lifecycle
        phase = event.phase if not event.stage else f"{event.phase} · {event.stage}"
        if is_process_lifecycle:
            phase = "Processus en cours"
        elif event.phase == "annulation distante" and event.message:
            phase = f"{phase} — {event.message}"
        self.phase_value.setText(phase)
        if event.progress is not None:
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(round(event.progress * 1000))
        if event.message:
            self._status_bar.showMessage(event.message, 5000)

    @pyqtSlot(str, str)
    def _on_log_line(self, stream: str, line: str) -> None:
        if line.lstrip().startswith(EVENT_PREFIX):
            return
        prefix = "" if stream == "stdout" else f"[{stream}] "
        self._pending_log_lines.append(prefix + line)
        if not self._log_flush_timer.isActive():
            self._log_flush_timer.start()

    @pyqtSlot()
    def _flush_logs(self) -> None:
        if not self._pending_log_lines:
            return
        lines = "\n".join(self._pending_log_lines)
        self._pending_log_lines.clear()
        self.log_view.appendPlainText(lines)

    @pyqtSlot(object)
    def _on_local_completed(self, result: RunResult) -> None:
        self._on_completed(result)
        self._active_kind = None
        self._update_action_availability()

    @pyqtSlot(object)
    def _on_remote_completed(self, result: RemoteTrialResult) -> None:
        effective_cancel_state = result.cancel_state or self._remote_cancel_state
        uncertain_cancel = (
            result.state is RunState.CANCELLED
            and effective_cancel_state is CancelState.UNKNOWN
            and not result.remote_terminal
        )
        if uncertain_cancel:
            self._close_requested = False
            self._remote_close_after_cancel = False
        elif (
            self._remote_close_after_cancel
            and result.state is RunState.CANCELLED
            and (
                effective_cancel_state is CancelState.ACKNOWLEDGED
                or result.remote_terminal
            )
        ):
            self._close_requested = True
        local_result = RunResult(
            state=result.state,
            exit_code=result.exit_code,
            duration_seconds=result.duration_seconds,
            error=result.error,
            artifacts=result.artifacts,
        )
        self._on_completed(local_result)
        if result.cache_hit and result.state is RunState.SUCCEEDED:
            self._set_result_banner(
                f"{self.result_banner.text()} · Cache local : aucun usage GPU.",
                "success",
            )
        self._active_kind = None
        self._pending_remote_request = None
        self._update_action_availability()

    @pyqtSlot(object)
    def _on_completed(self, result: RunResult) -> None:
        self.elapsed_value.setText(f"{result.duration_seconds:.1f} s")
        self.exit_value.setText("—" if result.exit_code is None else str(result.exit_code))
        if result.error:
            self._on_log_line("résultat", result.error)
        self._flush_logs()
        self._populate_artifacts(result.artifacts)
        artifact_note = (
            "aucun artefact déclaré"
            if not result.artifacts
            else "1 artefact produit"
            if len(result.artifacts) == 1
            else f"{len(result.artifacts)} artefacts produits"
        )
        outcome = {
            RunState.SUCCEEDED: ("essai réussi", "success"),
            RunState.FAILED: ("essai échoué", "failure"),
            RunState.CANCELLED: ("essai arrêté", "cancelled"),
        }[result.state]
        detail = f" · {result.error}" if result.error else ""
        self._set_result_banner(
            f"Résultat — {outcome[0]} en {result.duration_seconds:.1f} s · "
            f"{artifact_note}.{detail}",
            outcome[1],
        )
        status_message = {
            RunState.SUCCEEDED: "Essai réussi",
            RunState.FAILED: result.error or "Essai échoué",
            RunState.CANCELLED: "Essai arrêté",
        }[result.state]
        self._status_bar.showMessage(status_message, 10000)
        if self._close_requested and not self.remote_controller.has_pending_workers:
            QTimer.singleShot(0, self.close)

    def _update_elapsed(self) -> None:
        if self._started_at:
            self.elapsed_value.setText(f"{time.monotonic() - self._started_at:.1f} s")

    @pyqtSlot()
    def refresh_artifacts(self) -> None:
        self._populate_artifacts(())

    def _populate_artifacts(self, preferred_paths: tuple[Path, ...]) -> None:
        self.artifact_list.clear()
        entries_by_path = {
            str(entry.path.resolve()): entry for entry in discover_artifacts(self.repo_root)
        }
        for path in preferred_paths:
            resolved = path.resolve()
            key = str(resolved)
            if key in entries_by_path or not resolved.is_file():
                continue
            try:
                size = resolved.stat().st_size
            except OSError:
                continue
            entries_by_path[key] = ArtifactEntry(
                path=resolved,
                kind=_artifact_kind_for_display(resolved),
                size_bytes=size,
            )
        if not entries_by_path:
            item = QTreeWidgetItem(["", "Aucun artefact local détecté", ""])
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable & ~Qt.ItemFlag.ItemIsEnabled)
            self.artifact_list.addTopLevelItem(item)
            return

        preferred = {str(path.resolve()) for path in preferred_paths}
        selected: QTreeWidgetItem | None = None
        selected_items: list[QTreeWidgetItem] = []
        for entry in sorted(entries_by_path.values(), key=lambda value: str(value.path).lower()):
            relative = _relative_or_absolute(entry.path, self.repo_root)
            item = QTreeWidgetItem([entry.kind, str(relative), _format_bytes(entry.size_bytes)])
            resolved_text = str(entry.path.resolve())
            item.setData(1, Qt.ItemDataRole.UserRole, resolved_text)
            item.setToolTip(1, str(entry.path))
            self.artifact_list.addTopLevelItem(item)
            if resolved_text in preferred:
                item.setSelected(True)
                selected_items.append(item)
                if selected is None and resolved_text == str(preferred_paths[0].resolve()):
                    selected = item
        if selected is not None:
            self.artifact_list.setCurrentItem(selected)
            for item in selected_items:
                item.setSelected(True)
            self.artifact_list.scrollToItem(selected)

    def _set_result_banner(self, text: str, status: str) -> None:
        self.result_banner.setText(text)
        self.result_banner.setAccessibleDescription(text)
        self.result_banner.setProperty("status", status)
        style = self.result_banner.style()
        assert style is not None
        style.unpolish(self.result_banner)
        style.polish(self.result_banner)

    def closeEvent(self, event: QCloseEvent | None) -> None:
        if event is None:
            super().closeEvent(event)
            return
        if self.controller.state in {
            RunState.STARTING,
            RunState.RUNNING,
            RunState.CANCELLING,
        }:
            answer = QMessageBox.question(
                self,
                "PixaBoost",
                "Un processus CPU local est en cours. L'arreter et fermer ?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer is QMessageBox.StandardButton.Yes:
                self._close_requested = True
                self.controller.cancel()
            event.ignore()
            return
        if self.remote_controller.has_pending_workers:
            if self.remote_controller.state.is_terminal:
                self._close_requested = True
                self._status_bar.showMessage(
                    "Fermeture différée : finalisation du worker distant.",
                    3000,
                )
                event.ignore()
                return
            if not self.remote_controller.has_worker:
                self._close_requested = True
                self._status_bar.showMessage(
                    "Fermeture différée : finalisation du préflight local.",
                    3000,
                )
                event.ignore()
                return
            if self.remote_controller.state is RunState.CANCELLING:
                self._status_bar.showMessage(
                    "Fermeture impossible : attente de l'état d'annulation distante.",
                    7000,
                )
                event.ignore()
                return
            answer = QMessageBox.question(
                self,
                "PixaBoost",
                "Une reconstruction distante est en cours. Demander son arrêt ? "
                "La fenêtre restera ouverte tant que le Pod n'aura pas confirmé; "
                "aucun détachement silencieux n'est autorisé.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer is QMessageBox.StandardButton.Yes:
                self._close_requested = False
                self._remote_close_after_cancel = True
                if not self.remote_controller.cancel():
                    self._remote_close_after_cancel = False
            event.ignore()
            return
        super().closeEvent(event)


def _relative_or_absolute(path: Path, root: Path) -> Path:
    try:
        return path.relative_to(root)
    except ValueError:
        return path


def _format_bytes(value: int) -> str:
    size = float(value)
    for unit in ("o", "Kio", "Mio", "Gio"):
        if size < 1024.0 or unit == "Gio":
            return f"{size:.0f} {unit}" if unit == "o" else f"{size:.1f} {unit}"
        size /= 1024.0
    return f"{value} o"


def _artifact_kind_for_display(path: Path) -> str:
    if path.suffix.lower() == ".glb":
        return "GLB"
    if path.name == "manifest.json":
        return "manifeste"
    if path.name == "metrics.json":
        return "métriques"
    if path.name == "logs.jsonl":
        return "journal"
    if path.suffix.lower() == ".png":
        return "aperçu"
    return "rapport"
