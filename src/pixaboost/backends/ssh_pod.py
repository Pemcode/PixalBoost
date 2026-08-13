"""SSH transport for an already-running research Pod.

This module deliberately has no RunPod lifecycle or billing API.  It can only
connect to a host that already exists, and it refuses to do even that until a
short-lived, one-shot :class:`ExistingPodUseApproval` is supplied by the
caller.  The class implements the ``JobRunner.run(payload)`` slice consumed by
``generate_single_view``.

Paramiko is imported lazily.  Unit tests inject the higher-level ``PodSession``
boundary, so the CPU gate never installs Paramiko and never opens a socket.
"""

from __future__ import annotations

import base64
import binascii
import hashlib
import importlib
import json
import os
import re
import shlex
import struct
import tempfile
import threading
import time
import uuid
from collections.abc import Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from pixaboost.backends.runpod_client import RunPodError
from pixaboost.backends.ssh_worker_source import REMOTE_WORKER_SOURCE

PROTOCOL_VERSION = "pixaboost.ssh.v1"
FRAME_PREFIX = "PIXABOOST_SSH_FRAME "
DEFAULT_MAX_INPUT_BYTES = 32 * 1024 * 1024
DEFAULT_MAX_OUTPUT_BYTES = 256 * 1024 * 1024
_MAX_WORKER_BYTES = 1024 * 1024
_MAX_WORKER_LOG_BYTES = 16 * 1024 * 1024
_UPLOAD_CHUNK_CHARS = 48 * 1024
_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


class SshPodError(RunPodError):
    """A safe, classified failure in the existing-Pod SSH transport."""

    def __init__(
        self,
        message: str,
        *,
        code: str = "ssh_pod_error",
        cancel_state: CancelState | None = None,
        remote_terminal: bool = False,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.cancel_state = cancel_state
        self.remote_terminal = remote_terminal


class CancelState(StrEnum):
    """What is actually known about an attempted remote cancellation."""

    NOT_RUNNING = "not_running"
    ACKNOWLEDGED = "acknowledged"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class SshPodConfig:
    """Connection and integrity settings for one already-running Pod."""

    host: str
    username: str
    private_key_path: Path
    known_hosts_path: Path
    expected_pixal3d_sha: str
    project_git_sha: str
    local_runs_root: Path = Path("runs")
    remote_root: str = "/workspace/pixaboost-jobs"
    port: int = 22
    connect_timeout_seconds: float = 30.0
    command_timeout_seconds: float = 120.0
    inference_timeout_seconds: float = 3600.0
    max_input_bytes: int = DEFAULT_MAX_INPUT_BYTES
    max_output_bytes: int = DEFAULT_MAX_OUTPUT_BYTES

    def __post_init__(self) -> None:
        if not self.host.strip():
            raise ValueError("SSH host must not be blank")
        if not self.username.strip():
            raise ValueError("SSH username must not be blank")
        if not _GIT_SHA.fullmatch(self.expected_pixal3d_sha):
            raise ValueError("expected Pixal3D revision must be a full lowercase Git SHA")
        if not _GIT_SHA.fullmatch(self.project_git_sha):
            raise ValueError("project Git revision must be a full lowercase Git SHA")
        remote_root = PurePosixPath(self.remote_root)
        if not remote_root.is_absolute() or remote_root == PurePosixPath("/"):
            raise ValueError("remote_root must be a scoped absolute POSIX path")
        if ".." in remote_root.parts:
            raise ValueError("remote_root must not contain parent traversal")
        if self.port < 1 or self.port > 65535:
            raise ValueError("SSH port must be between 1 and 65535")
        if self.max_input_bytes < 1 or self.max_output_bytes < 12:
            raise ValueError("transfer limits must be positive")

    def approval_fingerprint(self) -> str:
        """Bind an approval to this host, account, port and model revision."""
        material = (f"{self.username}@{self.host}:{self.port}|{self.expected_pixal3d_sha}").encode()
        return hashlib.sha256(material).hexdigest()


@dataclass
class ExistingPodUseApproval:
    """In-memory, short-lived and one-shot confirmation to use an existing Pod.

    Construct it only after the user has explicitly confirmed the action.  It
    is never persisted and cannot be reused for another SSH connection.
    """

    _config_fingerprint: str
    _expires_at: float
    _used: bool = field(default=False, init=False, repr=False)

    @classmethod
    def grant(
        cls,
        config: SshPodConfig,
        *,
        valid_for_seconds: float = 120.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> ExistingPodUseApproval:
        if valid_for_seconds <= 0 or valid_for_seconds > 600:
            raise ValueError("approval lifetime must be in (0, 600] seconds")
        return cls(config.approval_fingerprint(), clock() + valid_for_seconds)

    def consume(self, config: SshPodConfig, *, now: float) -> None:
        if self._used:
            raise SshPodError(
                "existing-Pod approval has already been used; confirm the action again",
                code="approval_required",
            )
        if now > self._expires_at:
            raise SshPodError(
                "existing-Pod approval has expired; confirm the action again",
                code="approval_required",
            )
        if self._config_fingerprint != config.approval_fingerprint():
            raise SshPodError(
                "existing-Pod approval does not match this host or model revision",
                code="approval_required",
            )
        self._used = True


@dataclass(frozen=True)
class SshPodEvent:
    """Typed progress emitted by the local harness and remote worker."""

    run_id: str
    sequence: int
    phase: str
    progress: float | None
    message: str


class PodSession(Protocol):
    """Injectable boundary around one strict-host-key PTY session."""

    def connect(self, config: SshPodConfig) -> None: ...

    def remote_revision(self) -> str: ...

    def upload_bytes(
        self,
        remote_path: str,
        payload: bytes,
        *,
        expected_sha256: str,
        max_bytes: int,
    ) -> None: ...

    def run_worker(
        self,
        worker_path: str,
        request: dict[str, Any],
        frame_sink: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]: ...

    def download_bytes(
        self,
        remote_path: str,
        *,
        expected_sha256: str,
        expected_size: int,
        max_bytes: int,
    ) -> bytes: ...

    def cancel(self, run_id: str) -> bool: ...

    def close(self) -> None: ...


SessionFactory = Callable[[], PodSession]
EventSink = Callable[[SshPodEvent], None]


class SshPodClient:
    """A synchronous, manifest-writing ``JobRunner`` for an existing Pod."""

    def __init__(
        self,
        config: SshPodConfig,
        *,
        approval: ExistingPodUseApproval | None,
        session_factory: SessionFactory | None = None,
        event_sink: EventSink | None = None,
        run_id_factory: Callable[[], str] | None = None,
        clock: Callable[[], float] = time.monotonic,
        wall_clock: Callable[[], datetime] | None = None,
    ) -> None:
        self.config = config
        self._approval = approval
        self._session_factory = session_factory or ParamikoPtySession
        self._event_sink = event_sink or (lambda _event: None)
        self._run_id_factory = run_id_factory or self._new_run_id
        self._clock = clock
        self._wall_clock = wall_clock or (lambda: datetime.now(UTC))
        self._lock = threading.Lock()
        self._session: PodSession | None = None
        self._active_run_id: str | None = None
        self._cancel_requested = False
        self._cancel_state: CancelState | None = None
        self._remote_terminal_observed = False
        self._sequence = 0
        self._last_manifest_path: Path | None = None

    @property
    def last_manifest_path(self) -> Path:
        if self._last_manifest_path is None:
            raise RuntimeError("no SSH Pod run has been attempted")
        return self._last_manifest_path

    @staticmethod
    def _new_run_id() -> str:
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"ssh-{timestamp}-{uuid.uuid4().hex[:12]}"

    def run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Run one remote inference and return the serverless-compatible payload."""
        run_id = self._run_id_factory()
        if not _SAFE_RUN_ID.fullmatch(run_id):
            raise SshPodError("run id is not filesystem-safe", code="invalid_request")

        run_dir = self.config.local_runs_root / run_id
        self._last_manifest_path = run_dir / "manifest.json"
        manifest: dict[str, Any] = {
            "schema_version": 1,
            "protocol": PROTOCOL_VERSION,
            "run_id": run_id,
            "transport": "existing-pod-ssh",
            "git_sha": self.config.project_git_sha,
            "created_at": self._wall_clock().isoformat(),
            "status": "starting",
            "remote": {
                "host": self.config.host,
                "username": self.config.username,
                "expected_pixal3d_sha": self.config.expected_pixal3d_sha,
            },
            "cancel_state": None,
            "remote_terminal_observed": False,
            "poses": [],
        }
        session: PodSession | None = None

        try:
            image, params = self._validate_payload(payload)
            input_sha = hashlib.sha256(image).hexdigest()
            manifest["input"] = {"bytes": len(image), "sha256": input_sha}
            manifest["params"] = params
            self._write_manifest(manifest)

            if self._approval is None:
                raise SshPodError(
                    "explicit approval is required before using an existing Pod",
                    code="approval_required",
                )
            self._raise_if_cancelled()
            self._approval.consume(self.config, now=self._clock())

            self._emit(run_id, "connecting", 0.0, "Opening strict-host-key SSH session")
            self._raise_if_cancelled()
            session = self._session_factory()
            self._raise_if_cancelled()
            session.connect(self.config)
            with self._lock:
                self._raise_if_cancelled_locked()
                self._session = session
                self._active_run_id = run_id
            self._raise_if_cancelled()

            remote_revision = session.remote_revision().strip()
            self._raise_if_cancelled()
            manifest["remote"]["pixal3d_sha"] = remote_revision
            if remote_revision != self.config.expected_pixal3d_sha:
                raise SshPodError(
                    "remote Pixal3D revision mismatch: inference was not started",
                    code="revision_mismatch",
                )
            self._emit(run_id, "preflight", 0.05, "Pinned Pixal3D revision verified")

            remote_dir = str(PurePosixPath(self.config.remote_root) / run_id)
            input_path = str(PurePosixPath(remote_dir) / "input.png")
            output_path = str(PurePosixPath(remote_dir) / "output.glb")
            worker_path = str(PurePosixPath(remote_dir) / "ssh_worker.py")
            remote_manifest_path = str(PurePosixPath(remote_dir) / "manifest.json")

            worker = REMOTE_WORKER_SOURCE.encode("utf-8")
            if len(worker) > _MAX_WORKER_BYTES:
                raise SshPodError("versioned worker exceeds its transfer limit", code="internal")
            worker_sha = hashlib.sha256(worker).hexdigest()
            manifest["worker"] = {"bytes": len(worker), "sha256": worker_sha}
            self._write_manifest(manifest)
            self._emit(run_id, "transfer", 0.1, "Transferring bounded, hashed inputs")
            self._raise_if_cancelled()
            session.upload_bytes(
                worker_path,
                worker,
                expected_sha256=worker_sha,
                max_bytes=_MAX_WORKER_BYTES,
            )
            self._raise_if_cancelled()
            session.upload_bytes(
                input_path,
                image,
                expected_sha256=input_sha,
                max_bytes=self.config.max_input_bytes,
            )
            self._raise_if_cancelled()

            request = {
                "protocol": PROTOCOL_VERSION,
                "run_id": run_id,
                "input_path": input_path,
                "input_sha256": input_sha,
                "output_path": output_path,
                "manifest_path": remote_manifest_path,
                "expected_pixal3d_sha": self.config.expected_pixal3d_sha,
                "project_git_sha": self.config.project_git_sha,
                "worker_sha256": worker_sha,
                "params": params,
                "poses": [],
            }
            result_frame = session.run_worker(
                worker_path,
                request,
                lambda frame: self._accept_event_frame(run_id, frame),
            )
            artifact = self._validate_result_frame(run_id, output_path, result_frame)
            self._raise_if_cancelled()
            glb_bytes = session.download_bytes(
                artifact["path"],
                expected_sha256=artifact["sha256"],
                expected_size=artifact["bytes"],
                max_bytes=self.config.max_output_bytes,
            )
            self._raise_if_cancelled()
            self._validate_glb(glb_bytes, artifact)
            self._raise_if_cancelled()

            encoded = base64.b64encode(glb_bytes).decode("ascii")
            self._raise_if_cancelled()
            manifest["status"] = "completed"
            manifest["completed_at"] = self._wall_clock().isoformat()
            manifest["artifact"] = {
                "bytes": len(glb_bytes),
                "sha256": hashlib.sha256(glb_bytes).hexdigest(),
                "magic": "glTF",
                "version": 2,
                "remote_path": artifact["path"],
            }
            manifest["cancel_state"] = self._cancel_value()
            manifest["remote_terminal_observed"] = self._remote_terminal_observed
            self._write_manifest(manifest)
            self._emit(run_id, "completed", 1.0, "GLB downloaded and verified")
            return {
                "glb_base64": encoded,
                "glb_bytes": len(glb_bytes),
                "seed": int(params["seed"]),
                "pixal3d_sha": self.config.expected_pixal3d_sha,
                "run_id": run_id,
                "manifest_path": str(self.last_manifest_path),
            }
        except SshPodError as caught:
            error = self._cancel_error(caught)
            manifest["status"] = "cancelled" if error.code == "remote_cancelled" else "failed"
            manifest["completed_at"] = self._wall_clock().isoformat()
            manifest["cancel_state"] = self._cancel_value()
            manifest["remote_terminal_observed"] = self._remote_terminal_observed
            manifest["error"] = {"code": error.code, "message": str(error)}
            self._write_manifest(manifest)
            if error is caught:
                raise
            raise error from caught
        except Exception as error:
            safe = self._cancel_error(
                SshPodError(
                    f"SSH Pod transport failed: {type(error).__name__}: {error}",
                    code="transport_error",
                )
            )
            manifest["status"] = (
                "cancelled" if safe.code == "remote_cancelled" else "failed"
            )
            manifest["completed_at"] = self._wall_clock().isoformat()
            manifest["cancel_state"] = self._cancel_value()
            manifest["remote_terminal_observed"] = self._remote_terminal_observed
            manifest["error"] = {"code": safe.code, "message": str(safe)}
            self._write_manifest(manifest)
            raise safe from None
        finally:
            if session is not None:
                session.close()
            with self._lock:
                self._session = None
                self._active_run_id = None

    def cancel(self) -> CancelState:
        """Request cancellation once; never claim success without an ACK."""
        with self._lock:
            self._cancel_requested = True
            if self._cancel_state is not None:
                return self._cancel_state
            if self._session is None or self._active_run_id is None:
                # A client represents one approved run.  Cancellation before
                # its session is published is therefore a definitive local
                # stop request: latch it so run() cannot race on to SSH.
                self._cancel_state = CancelState.ACKNOWLEDGED
                return self._cancel_state
            try:
                acknowledged = self._session.cancel(self._active_run_id)
            except Exception:
                acknowledged = False
            self._cancel_state = CancelState.ACKNOWLEDGED if acknowledged else CancelState.UNKNOWN
            return self._cancel_state

    def _raise_if_cancelled(self) -> None:
        with self._lock:
            self._raise_if_cancelled_locked()

    def _raise_if_cancelled_locked(self) -> None:
        if self._cancel_requested:
            raise SshPodError(
                "local cancellation prevents further SSH work",
                code="remote_cancelled",
            )

    def _cancel_error(self, error: SshPodError) -> SshPodError:
        with self._lock:
            requested = self._cancel_requested
            cancel_state = self._cancel_state
            remote_terminal = self._remote_terminal_observed
        if not requested:
            return error
        if error.code == "remote_cancelled":
            if error.cancel_state is cancel_state and error.remote_terminal == remote_terminal:
                return error
            return SshPodError(
                str(error),
                code=error.code,
                cancel_state=cancel_state,
                remote_terminal=remote_terminal,
            )
        return SshPodError(
            f"SSH operation stopped after local cancellation request: {error}",
            code="remote_cancelled",
            cancel_state=cancel_state,
            remote_terminal=remote_terminal,
        )

    def _validate_payload(self, payload: Mapping[str, Any]) -> tuple[bytes, dict[str, Any]]:
        encoded = payload.get("image")
        if not isinstance(encoded, str) or not encoded:
            raise SshPodError(
                "input.image is required and must be base64 text", code="invalid_request"
            )
        if encoded.startswith("data:"):
            _, separator, encoded = encoded.partition(",")
            if not separator:
                raise SshPodError("input image data URI is malformed", code="invalid_request")
        max_encoded = 4 * ((self.config.max_input_bytes + 2) // 3)
        if len(encoded) > max_encoded + 4:
            raise SshPodError(
                f"input image exceeds the {self.config.max_input_bytes}-byte limit",
                code="invalid_request",
            )
        try:
            image = base64.b64decode(encoded, validate=True)
        except (binascii.Error, ValueError):
            raise SshPodError("input image is not valid base64", code="invalid_request") from None
        if not image:
            raise SshPodError("input image decoded to zero bytes", code="invalid_request")
        if len(image) > self.config.max_input_bytes:
            raise SshPodError(
                f"input image exceeds the {self.config.max_input_bytes}-byte limit",
                code="invalid_request",
            )
        low_vram = payload.get("low_vram", True)
        if not isinstance(low_vram, bool):
            raise SshPodError("low_vram must be a boolean", code="invalid_request")
        try:
            params = {
                "seed": int(payload.get("seed", 42)),
                "resolution": int(payload.get("resolution", -1)),
                "low_vram": low_vram,
                "fov": float(payload.get("fov", -1.0)),
            }
        except (TypeError, ValueError):
            raise SshPodError(
                "generation parameters have invalid types", code="invalid_request"
            ) from None
        return image, params

    def _accept_event_frame(self, run_id: str, frame: dict[str, Any]) -> None:
        if frame.get("protocol") != PROTOCOL_VERSION:
            raise SshPodError("remote event uses an unsupported protocol", code="protocol_error")
        if frame.get("kind") != "event" or frame.get("run_id") != run_id:
            raise SshPodError("remote event frame is malformed", code="protocol_error")
        phase = frame.get("phase")
        message = frame.get("message")
        progress = frame.get("progress")
        if not isinstance(phase, str) or not phase or not isinstance(message, str):
            raise SshPodError("remote event fields are malformed", code="protocol_error")
        if progress is not None:
            if not isinstance(progress, (int, float)) or isinstance(progress, bool):
                raise SshPodError("remote progress is not numeric", code="protocol_error")
            progress = float(progress)
            if progress < 0.0 or progress > 1.0:
                raise SshPodError("remote progress is outside [0, 1]", code="protocol_error")
        self._emit(run_id, phase, progress, message)

    def _validate_result_frame(
        self, run_id: str, output_path: str, frame: Mapping[str, Any]
    ) -> dict[str, Any]:
        if frame.get("protocol") != PROTOCOL_VERSION or frame.get("kind") != "result":
            raise SshPodError("remote result uses an unsupported protocol", code="protocol_error")
        if frame.get("run_id") != run_id:
            raise SshPodError("remote result belongs to another run", code="protocol_error")
        status = frame.get("status")
        if status not in {"completed", "cancelled", "failed"}:
            raise SshPodError("remote result status is invalid", code="protocol_error")
        with self._lock:
            self._remote_terminal_observed = True
        if status != "completed":
            if status == "cancelled":
                with self._lock:
                    self._cancel_state = CancelState.ACKNOWLEDGED
            detail = frame.get("error")
            message = str(detail) if detail else str(status or "unknown")
            code = "remote_cancelled" if status == "cancelled" else "remote_failed"
            raise SshPodError(
                f"remote inference failed: {message}",
                code=code,
                cancel_state=self._cancel_state,
                remote_terminal=True,
            )
        if frame.get("pixal3d_sha") != self.config.expected_pixal3d_sha:
            raise SshPodError("remote result revision mismatch", code="revision_mismatch")
        raw = frame.get("artifact")
        if not isinstance(raw, Mapping):
            raise SshPodError("remote result has no artifact frame", code="protocol_error")
        path = raw.get("path")
        size = raw.get("bytes")
        digest = raw.get("sha256")
        if path != output_path:
            raise SshPodError("remote result artifact path is unexpected", code="protocol_error")
        if not isinstance(size, int) or isinstance(size, bool) or size < 12:
            raise SshPodError("remote result artifact size is invalid", code="protocol_error")
        if size > self.config.max_output_bytes:
            raise SshPodError("remote result exceeds the output limit", code="protocol_error")
        if not isinstance(digest, str) or not _SHA256.fullmatch(digest):
            raise SshPodError("remote result artifact SHA-256 is invalid", code="protocol_error")
        if raw.get("magic") != "glTF" or raw.get("version") != 2:
            raise SshPodError("remote result does not describe a GLB v2", code="protocol_error")
        return {"path": path, "bytes": size, "sha256": digest}

    @staticmethod
    def _validate_glb(glb: bytes, artifact: Mapping[str, Any]) -> None:
        if len(glb) < 12 or glb[:4] != b"glTF":
            raise SshPodError("downloaded artifact has invalid GLB magic", code="invalid_artifact")
        version, declared_length = struct.unpack("<II", glb[4:12])
        if version != 2:
            raise SshPodError(
                f"downloaded artifact has unsupported GLB version {version}",
                code="invalid_artifact",
            )
        if declared_length != len(glb):
            raise SshPodError(
                "downloaded artifact GLB length does not match its header",
                code="invalid_artifact",
            )
        if len(glb) != artifact["bytes"]:
            raise SshPodError(
                "downloaded artifact size does not match frame", code="invalid_artifact"
            )
        digest = hashlib.sha256(glb).hexdigest()
        if digest != artifact["sha256"]:
            raise SshPodError("downloaded artifact SHA-256 mismatch", code="invalid_artifact")

    def _emit(self, run_id: str, phase: str, progress: float | None, message: str) -> None:
        event = SshPodEvent(run_id, self._sequence, phase, progress, message)
        self._sequence += 1
        self._event_sink(event)

    def _cancel_value(self) -> str | None:
        with self._lock:
            return self._cancel_state.value if self._cancel_state is not None else None

    def _write_manifest(self, manifest: Mapping[str, Any]) -> None:
        destination = self.last_manifest_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        data = json.dumps(dict(manifest), indent=2, sort_keys=True).encode("utf-8")
        handle, temporary = tempfile.mkstemp(
            dir=destination.parent, prefix="manifest.", suffix=".partial"
        )
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


@dataclass(frozen=True)
class _ShellResult:
    output: bytes
    exit_code: int


class ParamikoPtySession:
    """Strict-host-key Paramiko implementation using ``invoke_shell`` only."""

    def __init__(self, *, paramiko_loader: Callable[[], Any] | None = None) -> None:
        self._paramiko_loader = paramiko_loader or (lambda: importlib.import_module("paramiko"))
        self._client: Any = None
        self._channel: Any = None
        self._config: SshPodConfig | None = None
        self._send_lock = threading.Lock()
        self._cancel_requested = False

    def connect(self, config: SshPodConfig) -> None:
        if not config.private_key_path.is_file():
            raise SshPodError(f"SSH private key not found: {config.private_key_path}")
        if not config.known_hosts_path.is_file():
            raise SshPodError(
                f"known_hosts not found: {config.known_hosts_path}; refusing trust-on-first-use"
            )
        try:
            paramiko = self._paramiko_loader()
        except ImportError:
            raise SshPodError(
                "Paramiko is required for SSH Pod execution; install the SSH/GUI extra"
            ) from None

        client = paramiko.SSHClient()
        channel: Any = None
        try:
            client.load_host_keys(str(config.known_hosts_path))
            client.set_missing_host_key_policy(paramiko.RejectPolicy())
            client.connect(
                hostname=config.host,
                port=config.port,
                username=config.username,
                key_filename=str(config.private_key_path),
                look_for_keys=False,
                allow_agent=False,
                timeout=config.connect_timeout_seconds,
                banner_timeout=config.connect_timeout_seconds,
                auth_timeout=config.connect_timeout_seconds,
            )
            channel = client.invoke_shell(term="dumb", width=200, height=1000)
            channel.settimeout(config.connect_timeout_seconds)
            self._client = client
            self._channel = channel
            self._config = config
            configured = self._execute(
                "stty -echo -icanon min 1 time 0",
                timeout_seconds=config.command_timeout_seconds,
                max_output_bytes=4096,
            )
            self._require_success(configured, "configure PTY")
        except BaseException:
            if channel is not None:
                with suppress(Exception):
                    channel.close()
            with suppress(Exception):
                client.close()
            self._client = None
            self._channel = None
            self._config = None
            raise

    def remote_revision(self) -> str:
        config = self._require_config()
        result = self._execute(
            "git -C /opt/pixal3d rev-parse HEAD && "
            "test -z \"$(git -C /opt/pixal3d status --porcelain --untracked-files=all)\"",
            timeout_seconds=config.command_timeout_seconds,
            max_output_bytes=4096,
        )
        self._require_success(result, "attest clean remote Pixal3D checkout")
        return result.output.decode("ascii", errors="strict").strip().splitlines()[-1]

    def upload_bytes(
        self,
        remote_path: str,
        payload: bytes,
        *,
        expected_sha256: str,
        max_bytes: int,
    ) -> None:
        config = self._require_config()
        if len(payload) > max_bytes:
            raise SshPodError(f"upload exceeds its {max_bytes}-byte limit")
        if hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise SshPodError("local upload SHA-256 mismatch")

        destination = PurePosixPath(remote_path)
        if not destination.is_absolute():
            raise SshPodError("remote upload path must be absolute")
        parent = shlex.quote(str(destination.parent))
        target = shlex.quote(str(destination))
        encoded_path = shlex.quote(f"{destination}.base64.partial")
        binary_path = shlex.quote(f"{destination}.binary.partial")
        initialise = self._execute(
            f"mkdir -p {parent} && test ! -e {target} && : > {encoded_path}",
            timeout_seconds=config.command_timeout_seconds,
            max_output_bytes=4096,
        )
        self._require_success(initialise, "prepare bounded upload")

        encoded = base64.b64encode(payload).decode("ascii")
        try:
            for offset in range(0, len(encoded), _UPLOAD_CHUNK_CHARS):
                chunk = shlex.quote(encoded[offset : offset + _UPLOAD_CHUNK_CHARS])
                appended = self._execute(
                    f"printf '%s' {chunk} >> {encoded_path}",
                    timeout_seconds=config.command_timeout_seconds,
                    max_output_bytes=4096,
                )
                self._require_success(appended, "transfer upload chunk")
            verify = self._execute(
                f"base64 -d {encoded_path} > {binary_path} && "
                f"printf '%s %s\\n' \"$(sha256sum {binary_path} | cut -d' ' -f1)\" "
                f'"$(stat -c %s {binary_path})"',
                timeout_seconds=config.command_timeout_seconds,
                max_output_bytes=4096,
            )
            self._require_success(verify, "decode uploaded bytes")
            fields = verify.output.decode("ascii", errors="strict").strip().split()
            if fields != [expected_sha256, str(len(payload))]:
                raise SshPodError("remote upload SHA-256 or size mismatch")
            moved = self._execute(
                f"mv {binary_path} {target} && rm -f {encoded_path}",
                timeout_seconds=config.command_timeout_seconds,
                max_output_bytes=4096,
            )
            self._require_success(moved, "finalise uploaded bytes")
        except BaseException:
            with suppress(SshPodError):
                self._execute(
                    f"rm -f {encoded_path} {binary_path}",
                    timeout_seconds=config.command_timeout_seconds,
                    max_output_bytes=4096,
                )
            raise

    def run_worker(
        self,
        worker_path: str,
        request: dict[str, Any],
        frame_sink: Callable[[dict[str, Any]], None],
    ) -> dict[str, Any]:
        config = self._require_config()
        request_bytes = json.dumps(request, separators=(",", ":"), sort_keys=True).encode("utf-8")
        if len(request_bytes) > 128 * 1024:
            raise SshPodError("remote worker request is unexpectedly large")
        encoded = base64.b64encode(request_bytes).decode("ascii")
        frames: list[dict[str, Any]] = []
        pending = bytearray()

        def consume(chunk: bytes) -> None:
            pending.extend(chunk)
            while b"\n" in pending:
                raw, _, remainder = pending.partition(b"\n")
                pending[:] = remainder
                line = raw.strip().decode("utf-8", errors="replace")
                if not line.startswith(FRAME_PREFIX):
                    continue
                try:
                    frame = json.loads(line[len(FRAME_PREFIX) :])
                except json.JSONDecodeError as error:
                    raise SshPodError(f"malformed remote protocol frame: {error}") from None
                if not isinstance(frame, dict):
                    raise SshPodError("remote protocol frame is not an object")
                if frame.get("kind") == "event":
                    frame_sink(frame)
                elif frame.get("kind") == "result":
                    frames.append(frame)
                else:
                    raise SshPodError("remote protocol frame has an unknown kind")

        command = f"python3 -u {shlex.quote(worker_path)} --request-b64 {shlex.quote(encoded)}"
        result = self._execute(
            command,
            timeout_seconds=config.inference_timeout_seconds,
            max_output_bytes=_MAX_WORKER_LOG_BYTES,
            output_sink=consume,
        )
        if not frames:
            self._require_success(result, "run remote inference worker")
            raise SshPodError("remote worker returned no final result frame")
        if len(frames) != 1:
            raise SshPodError("remote worker returned multiple final result frames")
        return frames[0]

    def download_bytes(
        self,
        remote_path: str,
        *,
        expected_sha256: str,
        expected_size: int,
        max_bytes: int,
    ) -> bytes:
        config = self._require_config()
        if expected_size > max_bytes:
            raise SshPodError("download exceeds its configured byte limit", code="invalid_artifact")
        quoted = shlex.quote(remote_path)
        metadata = self._execute(
            f"printf '%s %s\\n' \"$(sha256sum {quoted} | cut -d' ' -f1)\" "
            f'"$(stat -c %s {quoted})"',
            timeout_seconds=config.command_timeout_seconds,
            max_output_bytes=4096,
        )
        self._require_success(metadata, "read remote artifact metadata")
        if metadata.output.decode("ascii", errors="strict").strip().split() != [
            expected_sha256,
            str(expected_size),
        ]:
            raise SshPodError("remote artifact changed before download", code="invalid_artifact")

        encoded_limit = 4 * ((max_bytes + 2) // 3) + 16
        downloaded = self._execute(
            f"base64 -w 0 {quoted}",
            timeout_seconds=config.command_timeout_seconds,
            max_output_bytes=encoded_limit,
        )
        self._require_success(downloaded, "download remote artifact")
        try:
            payload = base64.b64decode(downloaded.output.strip(), validate=True)
        except (binascii.Error, ValueError):
            raise SshPodError(
                "remote artifact transfer is not valid base64", code="invalid_artifact"
            ) from None
        if len(payload) != expected_size or hashlib.sha256(payload).hexdigest() != expected_sha256:
            raise SshPodError(
                "downloaded artifact SHA-256 or size mismatch", code="invalid_artifact"
            )
        return payload

    def cancel(self, run_id: str) -> bool:
        """Interrupt the PTY, but report unknown until a worker ACK is observed."""
        del run_id
        channel = self._require_channel()
        with self._send_lock:
            self._cancel_requested = True
            self._send_channel(channel, "\x03", action="cancellation")
        return False

    def close(self) -> None:
        if self._channel is not None:
            self._channel.close()
            self._channel = None
        if self._client is not None:
            self._client.close()
            self._client = None
        self._config = None

    def _execute(
        self,
        command: str,
        *,
        timeout_seconds: float,
        max_output_bytes: int,
        output_sink: Callable[[bytes], None] | None = None,
    ) -> _ShellResult:
        channel = self._require_channel()
        token = uuid.uuid4().hex
        begin = f"__PIXABOOST_BEGIN_{token}__".encode("ascii")
        end = f"__PIXABOOST_END_{token}__:".encode("ascii")
        wrapped = (
            f"printf '\\n{begin.decode()}\\n'; {{ {command}; }}; "
            f"pixaboost_rc=$?; printf '\\n{end.decode()}%s\\n' \"$pixaboost_rc\"\n"
        )
        with self._send_lock:
            if self._cancel_requested:
                raise SshPodError(
                    "PTY command refused after cancellation request",
                    code="remote_cancelled",
                )
            self._send_channel(channel, wrapped, action="command write")

        deadline = time.monotonic() + timeout_seconds
        buffer = bytearray()
        started = False
        streamed = 0
        while time.monotonic() <= deadline:
            if not channel.recv_ready():
                if channel.closed:
                    raise SshPodError("SSH PTY closed before command completion")
                time.sleep(0.01)
                continue
            chunk = channel.recv(65536)
            if not chunk:
                raise SshPodError("SSH PTY returned EOF before command completion")
            buffer.extend(chunk)
            if not started:
                marker = buffer.find(begin + b"\r\n")
                delimiter_size = len(begin) + 2
                if marker < 0:
                    marker = buffer.find(begin + b"\n")
                    delimiter_size = len(begin) + 1
                if marker < 0:
                    if len(buffer) > 1024 * 1024:
                        raise SshPodError("SSH PTY never emitted the command start frame")
                    continue
                del buffer[: marker + delimiter_size]
                started = True
            marker = buffer.find(end)
            end_match = None
            if marker >= 0:
                trailer = bytes(buffer[marker + len(end) : marker + len(end) + 32])
                end_match = re.match(rb"(-?\d+)\r?\n", trailer)
            visible_end = marker if marker >= 0 else len(buffer)
            if output_sink is not None and visible_end > streamed:
                output_sink(bytes(buffer[streamed:visible_end]))
                streamed = visible_end
            if visible_end > max_output_bytes:
                raise SshPodError(f"remote command output exceeded {max_output_bytes} bytes")
            if end_match is None:
                continue
            output = bytes(buffer[:marker]).rstrip(b"\r\n")
            return _ShellResult(output=output, exit_code=int(end_match.group(1)))
        raise SshPodError(f"remote command timed out after {timeout_seconds:.0f}s")

    def _require_channel(self) -> Any:
        if self._channel is None:
            raise SshPodError("SSH PTY is not connected")
        return self._channel

    def _require_config(self) -> SshPodConfig:
        if self._config is None:
            raise SshPodError("SSH PTY is not connected")
        return self._config

    @staticmethod
    def _send_channel(channel: Any, payload: str, *, action: str) -> None:
        try:
            channel.sendall(payload)
        except TimeoutError:
            raise SshPodError(
                f"SSH PTY {action} timed out",
                code="transport_timeout",
            ) from None
        except OSError as error:
            raise SshPodError(
                f"SSH PTY {action} failed: {error}",
                code="transport_error",
            ) from None

    @staticmethod
    def _require_success(result: _ShellResult, action: str) -> None:
        if result.exit_code != 0:
            raise SshPodError(f"could not {action} (remote exit {result.exit_code})")
