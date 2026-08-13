"""Non-blocking QProcess controller for the experiment GUI."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import signal
import stat
import subprocess
import sys
import time
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

from PyQt6.QtCore import QObject, QProcess, QTimer, pyqtSignal

from pixaboost.gui.model import (
    CommandSpec,
    RunResult,
    RunState,
    interpret_output_line,
    sanitise_log_line,
)
from pixaboost.observability import EVENT_PREFIX, EventProtocolError, TelemetryEvent

_POSIX_SESSION_LAUNCHER = "import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])"
_MAX_MANIFEST_BYTES = 8 * 1024 * 1024


@dataclass(frozen=True)
class _ArtifactSnapshot:
    is_file: bool
    sha256: str | None
    manifest_bytes: bytes | None
    manifest_too_large: bool = False


class CommandController(QObject):
    """Run at most one local command and translate its lifecycle into signals."""

    state_changed = pyqtSignal(object)
    command_changed = pyqtSignal(str)
    telemetry = pyqtSignal(object)
    log_line = pyqtSignal(str, str)
    completed = pyqtSignal(object)

    def __init__(
        self,
        parent: QObject | None = None,
        *,
        cancel_grace_milliseconds: int = 3000,
    ) -> None:
        super().__init__(parent)
        if cancel_grace_milliseconds < 0:
            raise ValueError("cancel grace period must not be negative")
        self.cancel_grace_milliseconds = cancel_grace_milliseconds
        self.state = RunState.IDLE
        self.current_command = ""
        self._spec: CommandSpec | None = None
        self._started_at = 0.0
        self._cancel_requested = False
        self._protocol_error = ""
        self._finalised = True
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._artifact_snapshots: dict[Path, _ArtifactSnapshot] = {}
        self._cancel_pid = 0
        self._uses_process_group = False

        self._process = QProcess(self)
        self._process.setProcessChannelMode(QProcess.ProcessChannelMode.SeparateChannels)
        self._process.started.connect(self._on_started)
        self._process.readyReadStandardOutput.connect(self._read_stdout)
        self._process.readyReadStandardError.connect(self._read_stderr)
        self._process.errorOccurred.connect(self._on_process_error)
        self._process.finished.connect(self._on_finished)
        self._kill_timer = QTimer(self)
        self._kill_timer.setSingleShot(True)
        self._kill_timer.timeout.connect(self._kill_if_running)
        self._direct_kill_timer = QTimer(self)
        self._direct_kill_timer.setSingleShot(True)
        self._direct_kill_timer.timeout.connect(self._kill_wrapper_if_same_process)

    @property
    def can_start(self) -> bool:
        """Whether the controller's single-flight slot is currently available."""
        return self._process.state() is QProcess.ProcessState.NotRunning and self.state not in {
            RunState.STARTING,
            RunState.RUNNING,
            RunState.CANCELLING,
        }

    def start(self, spec: CommandSpec) -> bool:
        """Start ``spec`` unless another process still owns the single-flight slot."""
        if not self.can_start:
            return False

        self._spec = spec
        self._started_at = time.monotonic()
        self._cancel_requested = False
        self._protocol_error = ""
        self._finalised = False
        self._stdout_buffer = ""
        self._stderr_buffer = ""
        self._cancel_pid = 0
        self._artifact_snapshots = {
            path: _snapshot_artifact(path) for path in spec.required_artifacts
        }
        self.current_command = spec.display_command
        self.command_changed.emit(self.current_command)
        self._set_state(RunState.STARTING)

        program, arguments, self._uses_process_group = _process_invocation(spec)
        self._process.setWorkingDirectory(str(spec.working_directory))
        self._process.setProgram(program)
        self._process.setArguments(list(arguments))
        self._process.start()
        return True

    def cancel(self) -> bool:
        """Request termination of the local process; no remote semantics are implied."""
        if self.state not in {
            RunState.STARTING,
            RunState.RUNNING,
        }:
            return False
        self._cancel_requested = True
        self._set_state(RunState.CANCELLING)
        self._emit_log("system", "arret local demande")
        if self._process.state() is not QProcess.ProcessState.NotRunning:
            self._terminate_process_tree(force=False)
            self._kill_timer.start(self.cancel_grace_milliseconds)
        return True

    def _set_state(self, state: RunState) -> None:
        self.state = state
        self.state_changed.emit(state)

    def _on_started(self) -> None:
        if self._cancel_requested:
            self._set_state(RunState.CANCELLING)
            self._terminate_process_tree(force=False)
            self._kill_timer.start(self.cancel_grace_milliseconds)
            return
        self._set_state(RunState.RUNNING)
        self.telemetry.emit(TelemetryEvent(phase="processus", message="processus enfant demarre"))

    def _read_stdout(self) -> None:
        chunk = self._process.readAllStandardOutput().data().decode("utf-8", errors="replace")
        self._stdout_buffer = self._consume("stdout", self._stdout_buffer, chunk)

    def _read_stderr(self) -> None:
        chunk = self._process.readAllStandardError().data().decode("utf-8", errors="replace")
        self._stderr_buffer = self._consume("stderr", self._stderr_buffer, chunk)

    def _consume(self, stream: str, buffered: str, chunk: str) -> str:
        combined = (buffered + chunk).replace("\r\n", "\n").replace("\r", "\n")
        lines = combined.split("\n")
        for line in lines[:-1]:
            self._handle_line(stream, line)
        return lines[-1]

    def _handle_line(self, stream: str, line: str) -> None:
        if not line:
            return
        is_structured = line.strip().startswith(EVENT_PREFIX)
        candidate = line if is_structured else sanitise_log_line(line)
        try:
            event = interpret_output_line(candidate)
        except EventProtocolError as error:
            if not self._protocol_error:
                self._protocol_error = f"Telemetrie invalide : {error}"
                self._emit_log("system", self._protocol_error)
            return
        if event is None:
            self._emit_log(stream, candidate)
            return
        safe_event = _sanitise_telemetry_event(event)
        if is_structured:
            if safe_event.message:
                self._emit_log(stream, safe_event.message)
        else:
            self._emit_log(stream, candidate)
        self.telemetry.emit(safe_event)

    def _emit_log(self, stream: str, line: str) -> None:
        self.log_line.emit(stream, sanitise_log_line(line))

    def _flush_buffers(self) -> None:
        self._read_stdout()
        self._read_stderr()
        if self._stdout_buffer:
            self._handle_line("stdout", self._stdout_buffer)
        if self._stderr_buffer:
            self._handle_line("stderr", self._stderr_buffer)
        self._stdout_buffer = ""
        self._stderr_buffer = ""

    def _kill_if_running(self) -> None:
        current_pid = int(self._process.processId())
        if (
            self.state is RunState.CANCELLING
            and self._process.state() is not QProcess.ProcessState.NotRunning
            and (self._cancel_pid == 0 or current_pid == self._cancel_pid)
        ):
            self._emit_log("system", "le processus ne repond pas ; arret force")
            self._terminate_process_tree(force=True)

    def _kill_wrapper_if_same_process(self) -> None:
        current_pid = int(self._process.processId())
        if (
            self.state is RunState.CANCELLING
            and self._process.state() is not QProcess.ProcessState.NotRunning
            and current_pid == self._cancel_pid
        ):
            self._process.kill()

    def _terminate_process_tree(self, *, force: bool) -> None:
        """Stop the local command tree, including ``uv``/``poe`` descendants on Windows."""
        pid = int(self._process.processId())
        if pid <= 0:
            return
        self._cancel_pid = pid
        if os.name == "nt":
            # Console children do not have a reliable graceful-close primitive on Windows.
            # Force the complete tree before the wrapper can exit and orphan descendants.
            arguments = ["taskkill", "/PID", str(pid), "/T", "/F"]
            creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
            try:
                subprocess.Popen(
                    arguments,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                    creationflags=creation_flags,
                )
            except OSError:
                (self._process.kill if force else self._process.terminate)()
            else:
                if force:
                    self._direct_kill_timer.start(250)
            return
        try:
            kill_signal = getattr(signal, "SIGKILL", signal.SIGTERM) if force else signal.SIGTERM
            if self._uses_process_group:
                kill_process_group = cast(Callable[[int, int], None], vars(os)["killpg"])
                kill_process_group(pid, kill_signal)
            else:
                os.kill(pid, kill_signal)
        except OSError:
            (self._process.kill if force else self._process.terminate)()

    def _force_remaining_process_group(self) -> None:
        if os.name == "nt" or not self._uses_process_group or self._cancel_pid <= 0:
            return
        kill_process_group = cast(Callable[[int, int], None], vars(os)["killpg"])
        with suppress(OSError):
            kill_process_group(
                self._cancel_pid,
                getattr(signal, "SIGKILL", signal.SIGTERM),
            )

    def _on_process_error(self, error: QProcess.ProcessError) -> None:
        if error is QProcess.ProcessError.FailedToStart:
            self._finalise(
                RunState.FAILED,
                exit_code=None,
                error=f"Impossible de demarrer la commande : {self._process.errorString()}",
            )

    def _on_finished(self, exit_code: int, exit_status: QProcess.ExitStatus) -> None:
        if self._finalised:
            return
        self._flush_buffers()
        self._kill_timer.stop()

        if self._cancel_requested:
            self._force_remaining_process_group()
            self._finalise(RunState.CANCELLED, exit_code=exit_code, error="arret local demande")
            return
        if self._protocol_error:
            self._finalise(RunState.FAILED, exit_code=exit_code, error=self._protocol_error)
            return
        if exit_status is QProcess.ExitStatus.CrashExit:
            self._finalise(
                RunState.FAILED,
                exit_code=exit_code,
                error="Le processus enfant s'est termine brutalement.",
            )
            return
        if exit_code != 0:
            self._finalise(
                RunState.FAILED,
                exit_code=exit_code,
                error=f"La commande s'est terminee avec le code {exit_code}.",
            )
            return

        spec = self._spec
        assert spec is not None
        current_snapshots = {path: _snapshot_artifact(path) for path in spec.required_artifacts}
        missing = tuple(
            path for path, snapshot in current_snapshots.items() if not snapshot.is_file
        )
        if missing:
            paths = ", ".join(str(path) for path in missing)
            self._finalise(
                RunState.FAILED,
                exit_code=exit_code,
                error=f"Commande verte mais artefact requis absent : {paths}",
            )
            return
        stale = tuple(
            path
            for path in spec.required_artifacts
            if self._artifact_snapshots[path].is_file
            and current_snapshots[path].is_file
            and current_snapshots[path].sha256 == self._artifact_snapshots[path].sha256
        )
        if stale:
            paths = ", ".join(str(path) for path in stale)
            self._finalise(
                RunState.FAILED,
                exit_code=exit_code,
                error=f"Artefact requis preexistant et non modifie par la commande : {paths}",
            )
            return
        invalid_manifest = _first_invalid_manifest(spec.required_artifacts, current_snapshots)
        if invalid_manifest is not None:
            self._finalise(
                RunState.FAILED,
                exit_code=exit_code,
                error=invalid_manifest,
            )
            return
        self._finalise(
            RunState.SUCCEEDED,
            exit_code=exit_code,
            artifacts=spec.required_artifacts,
        )

    def _finalise(
        self,
        state: RunState,
        *,
        exit_code: int | None,
        error: str = "",
        artifacts: tuple[Path, ...] = (),
    ) -> None:
        if self._finalised:
            return
        self._finalised = True
        self._kill_timer.stop()
        self._direct_kill_timer.stop()
        self._cancel_pid = 0
        self._uses_process_group = False
        self._set_state(state)
        result = RunResult(
            state=state,
            exit_code=exit_code,
            duration_seconds=max(0.0, time.monotonic() - self._started_at),
            error=error,
            artifacts=artifacts,
        )
        self.completed.emit(result)


def _snapshot_artifact(path: Path) -> _ArtifactSnapshot:
    hasher = hashlib.sha256()
    manifest = bytearray() if path.name == "manifest.json" else None
    manifest_too_large = False
    try:
        with path.open("rb") as handle:
            if not stat.S_ISREG(os.fstat(handle.fileno()).st_mode):
                return _ArtifactSnapshot(False, None, None)
            while chunk := handle.read(1024 * 1024):
                hasher.update(chunk)
                if manifest is not None and not manifest_too_large:
                    if len(manifest) + len(chunk) > _MAX_MANIFEST_BYTES:
                        manifest.clear()
                        manifest_too_large = True
                    else:
                        manifest.extend(chunk)
    except OSError:
        return _ArtifactSnapshot(False, None, None)
    return _ArtifactSnapshot(
        True,
        hasher.hexdigest(),
        bytes(manifest) if manifest is not None and not manifest_too_large else None,
        manifest_too_large,
    )


def _first_invalid_manifest(
    paths: tuple[Path, ...], snapshots: dict[Path, _ArtifactSnapshot]
) -> str | None:
    for path in paths:
        if path.name != "manifest.json":
            continue
        snapshot = snapshots[path]
        if snapshot.manifest_too_large:
            return f"Manifest JSON invalide ({path}) : taille superieure a 8 Mio"
        if snapshot.manifest_bytes is None:
            return f"Manifest JSON invalide ({path}) : contenu illisible"
        try:
            payload = json.loads(snapshot.manifest_bytes)
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            return f"Manifest JSON invalide ({path}) : {error}"
        if not isinstance(payload, dict):
            return f"Manifest JSON invalide ({path}) : la racine doit etre un objet"
        if not payload:
            return f"Manifest JSON invalide ({path}) : l'objet racine est vide"
    return None


def _process_invocation(spec: CommandSpec) -> tuple[str, tuple[str, ...], bool]:
    if os.name == "nt" or shutil.which(spec.program) is None:
        return spec.program, spec.arguments, False
    return (
        sys.executable,
        ("-c", _POSIX_SESSION_LAUNCHER, spec.program, *spec.arguments),
        True,
    )


def _sanitise_telemetry_event(event: TelemetryEvent) -> TelemetryEvent:
    artifact = Path(sanitise_log_line(str(event.artifact))) if event.artifact is not None else None
    return TelemetryEvent(
        phase=sanitise_log_line(event.phase),
        stage=sanitise_log_line(event.stage),
        progress=event.progress,
        message=sanitise_log_line(event.message),
        artifact=artifact,
    )
