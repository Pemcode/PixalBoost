"""Non-blocking Qt boundary for one existing-Pod single-view trial.

The GUI deliberately depends on a small injectable protocol instead of on the
SSH implementation.  ``preflight`` is required to be local and side-effect
free: it may inspect the artifact cache, but it must not create a client or
open a connection.  Consequently a cache miss can be presented to the user
before any worker thread or remote action exists.
"""

from __future__ import annotations

import os
import re
import subprocess
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Protocol

from PyQt6.QtCore import QObject, QThread, pyqtSignal, pyqtSlot

from pixaboost.backends.ssh_pod import CancelState, SshPodError
from pixaboost.gui.model import RunState, sanitise_command, sanitise_log_line
from pixaboost.observability import TelemetryEvent

_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
RemoteEventSink = Callable[[object], None]


@dataclass(frozen=True)
class RemoteTrialDefaults:
    """Non-secret values used to prefill the existing-Pod form."""

    image_path: Path | None = None
    host: str = ""
    username: str = ""
    private_key_path: Path | None = None
    known_hosts_path: Path | None = None
    expected_pixal3d_sha: str = ""
    project_git_sha: str = ""

    def merged(
        self,
        *,
        image_path: Path | None = None,
        host: str | None = None,
        username: str | None = None,
        private_key_path: Path | None = None,
        known_hosts_path: Path | None = None,
        expected_pixal3d_sha: str | None = None,
        project_git_sha: str | None = None,
    ) -> RemoteTrialDefaults:
        """Return CLI values overlaid on environment defaults."""
        return RemoteTrialDefaults(
            image_path=image_path or self.image_path,
            host=host if host is not None else self.host,
            username=username if username is not None else self.username,
            private_key_path=private_key_path or self.private_key_path,
            known_hosts_path=known_hosts_path or self.known_hosts_path,
            expected_pixal3d_sha=(
                expected_pixal3d_sha
                if expected_pixal3d_sha is not None
                else self.expected_pixal3d_sha
            ),
            project_git_sha=(
                project_git_sha if project_git_sha is not None else self.project_git_sha
            ),
        )

    @classmethod
    def from_environment(
        cls,
        environment: Mapping[str, str] | None = None,
    ) -> RemoteTrialDefaults:
        """Load optional form defaults; none of these values is a credential."""
        env = os.environ if environment is None else environment

        def optional_path(name: str) -> Path | None:
            value = env.get(name, "").strip()
            return Path(value).expanduser() if value else None

        return cls(
            image_path=optional_path("PIXABOOST_TRIAL_IMAGE"),
            host=env.get("PIXABOOST_SSH_HOST", "").strip(),
            username=env.get("PIXABOOST_SSH_USER", "").strip(),
            private_key_path=optional_path("PIXABOOST_SSH_KEY_PATH"),
            known_hosts_path=optional_path("PIXABOOST_SSH_KNOWN_HOSTS"),
            expected_pixal3d_sha=env.get("PIXABOOST_PIXAL3D_SHA", "").strip(),
            project_git_sha=env.get("PIXABOOST_PROJECT_SHA", "").strip(),
        )


@dataclass(frozen=True)
class RemoteTrialRequest:
    """Validated, non-secret description of a single-view Pod trial."""

    image_path: Path
    host: str
    username: str
    private_key_path: Path
    known_hosts_path: Path
    expected_pixal3d_sha: str
    project_git_sha: str

    def __post_init__(self) -> None:
        error = self.validation_error()
        if error:
            raise ValueError(error)

    def validation_error(self) -> str:
        if not self.image_path.is_file():
            return "L'image d'entrée n'existe pas."
        if not self.host.strip():
            return "L'hôte SSH est requis."
        if not self.username.strip():
            return "L'utilisateur SSH est requis."
        if not self.private_key_path.is_file():
            return "La clé privée SSH n'existe pas."
        if not self.known_hosts_path.is_file():
            return "Le fichier known_hosts n'existe pas."
        if not _FULL_GIT_SHA.fullmatch(self.expected_pixal3d_sha):
            return "La révision Pixal3D doit être un SHA Git complet en minuscules."
        if not _FULL_GIT_SHA.fullmatch(self.project_git_sha):
            return "La révision PixaBoost doit être un SHA Git complet en minuscules."
        return ""

    @property
    def display_command(self) -> str:
        """Render the cache-first equivalent without a Pod-use approval flag."""
        return self.display_command_for(approve_existing_pod=False)

    def display_command_for(self, *, approve_existing_pod: bool) -> str:
        """Render the real public CLI, with its one-shot approval made explicit."""
        arguments = [
            "reconstruct",
            "single-view",
            str(self.image_path),
            "--backend",
            "ssh-pod",
            "--host",
            self.host,
            "--user",
            self.username,
            "--key",
            str(self.private_key_path),
            "--known-hosts",
            str(self.known_hosts_path),
            "--revision",
            self.expected_pixal3d_sha,
            "--project-sha",
            self.project_git_sha,
            "--resolution",
            "1024",
            "--low-vram",
        ]
        if approve_existing_pod:
            arguments.append("--confirm-existing-pod-use")
        return sanitise_command(
            "pixaboost",
            tuple(arguments),
        )


@dataclass(frozen=True)
class RemoteTrialResult:
    """Terminal result shown by the same widgets as a local command."""

    state: RunState
    duration_seconds: float
    exit_code: int | None = None
    error: str = ""
    artifacts: tuple[Path, ...] = ()
    cache_hit: bool = False
    cancel_state: CancelState | None = None
    remote_terminal: bool = False

    def __post_init__(self) -> None:
        if not self.state.is_terminal:
            raise ValueError("remote trial result state must be terminal")
        if self.duration_seconds < 0:
            raise ValueError("remote trial duration cannot be negative")


class RemoteTrialRunner(Protocol):
    """Injectable service surface consumed by :class:`RemoteTrialController`."""

    def preflight(self, request: RemoteTrialRequest) -> bool:
        """Return cache-hit status without creating a client or a connection."""
        ...

    def run(
        self,
        request: RemoteTrialRequest,
        *,
        approve_existing_pod: bool,
        event_sink: RemoteEventSink,
    ) -> RemoteTrialResult: ...

    def cancel(self) -> CancelState: ...


RemoteRunnerFactory = Callable[[], RemoteTrialRunner]


class _PreflightWorker(QThread):
    """Run the bounded local cache lookup without blocking Qt's event loop."""

    def __init__(
        self,
        runner: RemoteTrialRunner,
        request: RemoteTrialRequest,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._request = request
        self.cache_hit: bool | None = None
        self.error = ""

    def run(self) -> None:
        try:
            self.cache_hit = bool(self._runner.preflight(self._request))
        except Exception as error:
            self.error = sanitise_log_line(f"{type(error).__name__}: {error}")


class _RemoteWorker(QThread):
    telemetry_event = pyqtSignal(object)
    log_line = pyqtSignal(str, str)
    completed = pyqtSignal(object)

    def __init__(
        self,
        runner: RemoteTrialRunner,
        request: RemoteTrialRequest,
        *,
        approve_existing_pod: bool,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner = runner
        self._request = request
        self._approve_existing_pod = approve_existing_pod

    def run(self) -> None:
        started_at = time.monotonic()
        try:
            result = self._runner.run(
                self._request,
                approve_existing_pod=self._approve_existing_pod,
                event_sink=self.telemetry_event.emit,
            )
        except SshPodError as error:
            message = sanitise_log_line(f"{type(error).__name__}: {error}")
            self.log_line.emit("remote", message)
            result = RemoteTrialResult(
                state=(
                    RunState.CANCELLED
                    if error.code == "remote_cancelled"
                    else RunState.FAILED
                ),
                duration_seconds=time.monotonic() - started_at,
                error=message,
                cancel_state=error.cancel_state,
                remote_terminal=error.remote_terminal,
            )
        except Exception as error:
            message = sanitise_log_line(f"{type(error).__name__}: {error}")
            self.log_line.emit("remote", message)
            result = RemoteTrialResult(
                state=RunState.FAILED,
                duration_seconds=time.monotonic() - started_at,
                error=message,
            )
        self.completed.emit(result)


class _CancelWorker(QThread):
    completed = pyqtSignal(object)

    def __init__(self, runner: RemoteTrialRunner, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._runner = runner

    def run(self) -> None:
        try:
            state = self._runner.cancel()
        except Exception:
            state = CancelState.UNKNOWN
        self.completed.emit(state)


class RemoteTrialController(QObject):
    """Own remote workers and expose the local controller's signal surface."""

    state_changed = pyqtSignal(object)
    command_changed = pyqtSignal(str)
    telemetry = pyqtSignal(object)
    log_line = pyqtSignal(str, str)
    completed = pyqtSignal(object)
    cancellation_changed = pyqtSignal(object)
    prepared = pyqtSignal(bool)
    preparation_failed = pyqtSignal(str)
    availability_changed = pyqtSignal()

    def __init__(
        self,
        runner_factory: RemoteRunnerFactory | None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._runner_factory = runner_factory
        self._runner: RemoteTrialRunner | None = None
        self._request: RemoteTrialRequest | None = None
        self._cache_hit = False
        self._preflight_worker: _PreflightWorker | None = None
        self._worker: _RemoteWorker | None = None
        self._cancel_worker: _CancelWorker | None = None
        self._last_cancel_state: CancelState | None = None
        self.state = RunState.IDLE

    @property
    def is_configured(self) -> bool:
        return self._runner_factory is not None

    @property
    def has_worker(self) -> bool:
        return self._worker is not None

    @property
    def has_pending_workers(self) -> bool:
        """Whether a run or cancellation thread still owns GUI resources."""
        return (
            self._preflight_worker is not None
            or self._worker is not None
            or self._cancel_worker is not None
        )

    @property
    def owns_single_flight(self) -> bool:
        """Whether a preflight, prepared request, run or cancellation owns the slot."""
        return self._runner is not None or self.has_pending_workers

    @property
    def can_start(self) -> bool:
        return (
            self.is_configured
            and not self.owns_single_flight
            and self.state not in {RunState.STARTING, RunState.RUNNING, RunState.CANCELLING}
        )

    def prepare(self, request: RemoteTrialRequest) -> bool:
        """Start the local-only preflight on a worker and reserve single-flight."""
        if not self.can_start:
            raise RuntimeError("another remote trial already owns the single-flight slot")
        assert self._runner_factory is not None
        runner = self._runner_factory()
        self._runner = runner
        self._request = request
        self._cache_hit = False
        self._last_cancel_state = None
        self._set_state(RunState.STARTING)
        worker = _PreflightWorker(runner, request, self)
        self._preflight_worker = worker
        worker.finished.connect(self._on_preflight_finished)
        worker.start()
        self.availability_changed.emit()
        return True

    def discard_prepared(self) -> None:
        """Discard a refused cache miss; no worker or connection was created."""
        if self.has_pending_workers:
            raise RuntimeError("cannot discard an active remote trial")
        self._runner = None
        self._request = None
        self._cache_hit = False
        self._set_state(RunState.IDLE)
        self.availability_changed.emit()

    def start_prepared(self, *, approve_existing_pod: bool) -> bool:
        if self._runner is None or self._request is None or self.has_pending_workers:
            return False
        if not self._cache_hit and not approve_existing_pod:
            return False
        self.command_changed.emit(
            self._request.display_command_for(
                approve_existing_pod=not self._cache_hit and approve_existing_pod
            )
        )
        self._set_state(RunState.STARTING)
        worker = _RemoteWorker(
            self._runner,
            self._request,
            approve_existing_pod=approve_existing_pod,
            parent=self,
        )
        self._worker = worker
        worker.started.connect(self._on_started)
        worker.telemetry_event.connect(self._on_event)
        worker.log_line.connect(self.log_line)
        worker.completed.connect(self._on_completed)
        worker.finished.connect(self._on_worker_finished)
        worker.start()
        self.availability_changed.emit()
        return True

    def cancel(self) -> bool:
        """Request remote cancellation on another thread and wait for a real ACK."""
        if self._runner is None or not self.has_worker or self._cancel_worker is not None:
            return False
        self._set_state(RunState.CANCELLING)
        self.log_line.emit("system", "Annulation distante demandée")
        cancel_worker = _CancelWorker(self._runner, self)
        self._cancel_worker = cancel_worker
        cancel_worker.completed.connect(self._on_cancel_completed)
        cancel_worker.finished.connect(self._on_cancel_worker_finished)
        cancel_worker.start()
        self.availability_changed.emit()
        return True

    def _set_state(self, state: RunState) -> None:
        self.state = state
        self.state_changed.emit(state)

    @pyqtSlot()
    def _on_started(self) -> None:
        if self.state is RunState.STARTING:
            self._set_state(RunState.RUNNING)

    @pyqtSlot(object)
    def _on_event(self, event: object) -> None:
        raw_phase = sanitise_log_line(str(getattr(event, "phase", "remote")))
        phase = {
            "connecting": "connexion",
            "preflight": "préflight",
            "transfer": "transfert",
            "inference": "inférence",
            "completed": "terminé",
        }.get(raw_phase, raw_phase)
        stage = sanitise_log_line(str(getattr(event, "stage", "")))
        progress_value = getattr(event, "progress", None)
        progress = float(progress_value) if progress_value is not None else None
        message = sanitise_log_line(str(getattr(event, "message", "")))
        artifact_value = getattr(event, "artifact", None)
        artifact = Path(artifact_value) if artifact_value is not None else None
        telemetry = TelemetryEvent(
            phase=phase,
            stage=stage,
            progress=progress,
            message=message,
            artifact=artifact,
        )
        self.telemetry.emit(telemetry)
        if message:
            self.log_line.emit("remote", message)

    @pyqtSlot(object)
    def _on_cancel_completed(self, state: CancelState) -> None:
        self._last_cancel_state = state
        self.cancellation_changed.emit(state)
        if state is CancelState.ACKNOWLEDGED:
            stage = "acknowledged"
            message = (
                "Annulation prise en compte — aucune nouvelle étape distante "
                "ne sera lancée."
            )
        elif state is CancelState.UNKNOWN:
            stage = "unknown"
            message = "Annulation locale, état Pod inconnu."
        else:
            stage = "not_running"
            message = "Aucun processus distant actif au moment de l'annulation."
        self.telemetry.emit(
            TelemetryEvent(phase="annulation distante", stage=stage, message=message)
        )
        self.log_line.emit("system", message)

    @pyqtSlot(object)
    def _on_completed(self, result: RemoteTrialResult) -> None:
        cancel_state = result.cancel_state or self._last_cancel_state
        error = sanitise_log_line(result.error)
        if (
            result.state is RunState.CANCELLED
            and cancel_state is CancelState.UNKNOWN
            and not result.remote_terminal
        ):
            error = "Annulation locale, état Pod inconnu."
        safe_result = replace(result, error=error, cancel_state=cancel_state)
        self._set_state(safe_result.state)
        self.completed.emit(safe_result)

    @pyqtSlot()
    def _on_preflight_finished(self) -> None:
        worker = self._preflight_worker
        self._preflight_worker = None
        if worker is None:
            return
        cache_hit = worker.cache_hit
        error = worker.error
        worker.deleteLater()
        if error or cache_hit is None:
            message = error or "Préflight local interrompu sans résultat."
            self._runner = None
            self._request = None
            self._cache_hit = False
            self.log_line.emit("remote", message)
            self._set_state(RunState.FAILED)
            self.preparation_failed.emit(message)
        else:
            self._cache_hit = cache_hit
            self.prepared.emit(cache_hit)
        self.availability_changed.emit()

    @pyqtSlot()
    def _on_worker_finished(self) -> None:
        worker = self._worker
        self._worker = None
        if worker is not None:
            worker.deleteLater()
        self._runner = None
        self._request = None
        self._cache_hit = False
        self.availability_changed.emit()

    @pyqtSlot()
    def _on_cancel_worker_finished(self) -> None:
        worker = self._cancel_worker
        self._cancel_worker = None
        if worker is not None:
            worker.deleteLater()
        self.availability_changed.emit()


def project_git_sha(repo_root: Path) -> str:
    """Read the local project revision without invoking a shell."""
    completed = subprocess.run(
        ("git", "rev-parse", "HEAD"),
        cwd=repo_root,
        check=True,
        capture_output=True,
        text=True,
        timeout=10,
    )
    revision = completed.stdout.strip()
    if not _FULL_GIT_SHA.fullmatch(revision):
        raise RuntimeError("git rev-parse did not return a full lowercase SHA")
    return revision
