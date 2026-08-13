"""Observable, cache-first single-view reconstruction trial.

This public CPU harness proves the cache decision locally, requires a one-shot
approval before constructing a remote client, delegates generation to the
existing adapter, and records reproducible evidence for every attempt.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import tempfile
import time
import uuid
import warnings
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from io import BytesIO
from pathlib import Path
from typing import Any, Protocol

from PIL import Image

from pixaboost.backends.cache import (
    ArtifactCache,
    CacheCorruptionError,
    CachedArtifact,
    cache_key,
)
from pixaboost.backends.pixal3d import GenerationParams, JobRunner, generate_single_view
from pixaboost.backends.ssh_pod import (
    ExistingPodUseApproval,
    SshPodClient,
    SshPodConfig,
    SshPodError,
    SshPodEvent,
)
from pixaboost.observability import TelemetryEvent

_SAFE_RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
_ALLOWED_IMAGE_FORMATS = frozenset({"BMP", "JPEG", "PNG", "WEBP"})

TrialEventSink = Callable[[TelemetryEvent], None]
TransportEventSink = Callable[[SshPodEvent], None]


class SingleViewClientFactory(Protocol):
    """Construct the remote boundary only after approval has been checked."""

    def __call__(
        self,
        config: SshPodConfig,
        approval: ExistingPodUseApproval | None,
        event_sink: TransportEventSink,
    ) -> JobRunner: ...


@dataclass(frozen=True)
class SingleViewTrialConfig:
    """All inputs needed to reproduce one mono-view trial."""

    image_path: Path
    params: GenerationParams
    cache: ArtifactCache
    ssh: SshPodConfig
    poses: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class SingleViewPreflight:
    """Purely local cache decision, safe to call before user confirmation."""

    cache_key: str
    cache_hit: bool
    artifact: CachedArtifact | None
    approval_required: bool


@dataclass(frozen=True)
class SingleViewTrialResult:
    """Completed artefact and its local evidence bundle."""

    artifact: CachedArtifact
    cache_hit: bool
    run_dir: Path
    manifest_path: Path
    metrics_path: Path
    logs_path: Path


@dataclass
class _Evidence:
    run_dir: Path
    manifest: dict[str, Any]
    records: list[dict[str, Any]]
    started: float
    clock: Callable[[], float]
    event_sink: TrialEventSink

    @property
    def manifest_path(self) -> Path:
        return self.run_dir / "manifest.json"

    @property
    def metrics_path(self) -> Path:
        return self.run_dir / "metrics.json"

    @property
    def logs_path(self) -> Path:
        return self.run_dir / "logs.jsonl"

    def emit(self, event: TelemetryEvent) -> None:
        self.records.append(
            {
                "elapsed_seconds": max(0.0, self.clock() - self.started),
                "phase": event.phase,
                "stage": event.stage,
                "progress": event.progress,
                "message": event.message,
                "artifact": str(event.artifact) if event.artifact is not None else None,
            }
        )
        _atomic_write_jsonl(self.logs_path, self.records)
        self.event_sink(event)

    def finish(self, *, status: str, cache_hit: bool, error: SshPodError | None = None) -> None:
        duration = max(0.0, self.clock() - self.started)
        self.manifest["status"] = status
        self.manifest["completed_at"] = datetime.now(UTC).isoformat()
        if error is not None:
            self.manifest["error"] = {"code": error.code, "message": str(error)}
        _atomic_write_json(self.manifest_path, self.manifest)
        metrics: dict[str, Any] = {
            "schema_version": 1,
            "status": status,
            "cache_hit": cache_hit,
            "duration_seconds": duration,
        }
        if error is not None:
            metrics["error_code"] = error.code
        _atomic_write_json(self.metrics_path, metrics)
        if not self.logs_path.exists():
            _atomic_write_jsonl(self.logs_path, self.records)


def preflight_single_view(config: SingleViewTrialConfig) -> SingleViewPreflight:
    """Resolve the cache locally without approval, SSH, GPU or network access."""
    image = _read_input_image(config)
    key = cache_key(
        image=image,
        params=asdict(config.params),
        model_revision=config.ssh.expected_pixal3d_sha,
    )
    artifact = config.cache.load(key)
    return SingleViewPreflight(
        cache_key=key,
        cache_hit=artifact is not None,
        artifact=artifact,
        approval_required=artifact is None,
    )


def run_single_view_trial(
    config: SingleViewTrialConfig,
    *,
    approval: ExistingPodUseApproval | None = None,
    event_sink: TrialEventSink | None = None,
    client_factory: SingleViewClientFactory | None = None,
    run_id_factory: Callable[[], str] | None = None,
) -> SingleViewTrialResult:
    """Run or reuse one reconstruction and atomically write its evidence."""
    run_id = (run_id_factory or _new_run_id)()
    if not _SAFE_RUN_ID.fullmatch(run_id):
        raise ValueError("trial run id is not filesystem-safe")
    run_dir = config.ssh.local_runs_root / run_id
    image = _read_input_image(config)
    params = asdict(config.params)
    key = cache_key(
        image=image,
        params=params,
        model_revision=config.ssh.expected_pixal3d_sha,
    )
    started = time.monotonic()
    evidence = _Evidence(
        run_dir=run_dir,
        manifest={
            "schema_version": 1,
            "run_id": run_id,
            "kind": "single-view-reconstruction",
            "backend": "ssh-pod",
            "git_sha": config.ssh.project_git_sha,
            "model_revision": config.ssh.expected_pixal3d_sha,
            "created_at": datetime.now(UTC).isoformat(),
            "status": "starting",
            "input": {
                "path": str(config.image_path),
                "bytes": len(image),
                "sha256": hashlib.sha256(image).hexdigest(),
            },
            "seeds": [config.params.seed],
            "params": params,
            "poses": list(config.poses),
            "cache": {"key": key, "hit": False},
        },
        records=[],
        started=started,
        clock=time.monotonic,
        event_sink=event_sink or (lambda _event: None),
    )

    try:
        artifact = config.cache.load(key)
    except CacheCorruptionError as error:
        safe = SshPodError(str(error), code="cache_corruption")
        evidence.emit(TelemetryEvent(phase="failed", stage="cache", message=str(safe)))
        evidence.finish(status="failed", cache_hit=False, error=safe)
        raise safe from None
    cache_hit = artifact is not None
    evidence.manifest["cache"]["hit"] = cache_hit
    evidence.emit(
        TelemetryEvent(
            phase="preflight",
            stage="cache",
            progress=0.0,
            message="Artefact cache hit" if cache_hit else "Artefact cache miss",
        )
    )
    if artifact is not None:
        evidence.manifest["artifact"] = _artifact_record(artifact)
        evidence.emit(
            TelemetryEvent(
                phase="completed",
                stage="cache",
                progress=1.0,
                message="Reconstruction loaded from the local cache",
                artifact=artifact.glb_path,
            )
        )
        evidence.finish(status="completed", cache_hit=True)
        return _result(artifact, True, evidence)

    if approval is None:
        approval_error = SshPodError(
            "explicit approval is required before using an existing Pod",
            code="approval_required",
        )
        evidence.emit(TelemetryEvent(phase="failed", stage="approval", message=str(approval_error)))
        evidence.finish(status="failed", cache_hit=False, error=approval_error)
        raise approval_error

    factory = client_factory or _default_client_factory

    def receive_transport_event(remote: SshPodEvent) -> None:
        evidence.emit(
            TelemetryEvent(
                phase=remote.phase,
                stage="ssh-pod",
                progress=remote.progress,
                message=remote.message,
            )
        )

    try:
        client = factory(config.ssh, approval, receive_transport_event)
        artifact = generate_single_view(
            image=image,
            params=config.params,
            client=client,
            cache=config.cache,
            model_revision=config.ssh.expected_pixal3d_sha,
        )
        transport_manifest = getattr(client, "last_manifest_path", None)
        if isinstance(transport_manifest, Path):
            evidence.manifest["transport_manifest"] = str(transport_manifest)
        evidence.manifest["artifact"] = _artifact_record(artifact)
        evidence.emit(
            TelemetryEvent(
                phase="completed",
                stage="cache",
                progress=1.0,
                message="Verified GLB stored in the local artefact cache",
                artifact=artifact.glb_path,
            )
        )
        evidence.finish(status="completed", cache_hit=False)
        return _result(artifact, False, evidence)
    except SshPodError as error:
        status = "cancelled" if error.code == "remote_cancelled" else "failed"
        evidence.emit(TelemetryEvent(phase=status, stage="ssh-pod", message=str(error)))
        evidence.finish(status=status, cache_hit=False, error=error)
        raise
    except Exception as error:
        safe = SshPodError(
            f"single-view trial failed: {type(error).__name__}: {error}",
            code="trial_error",
        )
        evidence.emit(TelemetryEvent(phase="failed", stage="trial", message=str(safe)))
        evidence.finish(status="failed", cache_hit=False, error=safe)
        raise safe from None


def _default_client_factory(
    config: SshPodConfig,
    approval: ExistingPodUseApproval | None,
    event_sink: TransportEventSink,
) -> JobRunner:
    return SshPodClient(config, approval=approval, event_sink=event_sink)


def _read_input_image(config: SingleViewTrialConfig) -> bytes:
    """Read at most the transport limit, including under a concurrent file change."""
    path = config.image_path
    limit = config.ssh.max_input_bytes
    try:
        announced_size = path.stat().st_size
    except OSError as error:
        raise SshPodError(
            f"input image cannot be inspected: {error}",
            code="invalid_request",
        ) from None
    if announced_size > limit:
        raise SshPodError(
            f"input image exceeds the {limit}-byte limit",
            code="invalid_request",
        )
    try:
        with path.open("rb") as stream:
            image = stream.read(limit + 1)
    except OSError as error:
        raise SshPodError(
            f"input image cannot be read: {error}",
            code="invalid_request",
        ) from None
    if len(image) > limit:
        raise SshPodError(
            f"input image exceeds the {limit}-byte limit",
            code="invalid_request",
        )
    if not image:
        raise SshPodError("input image is empty", code="invalid_request")
    _validate_input_image(image)
    return image


def _validate_input_image(image: bytes) -> None:
    """Reject unsupported, malformed, or decompression-bomb inputs locally."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(BytesIO(image)) as opened:
                image_format = opened.format
                if image_format not in _ALLOWED_IMAGE_FORMATS:
                    supported = ", ".join(sorted(_ALLOWED_IMAGE_FORMATS))
                    raise SshPodError(
                        f"unsupported input image format: {image_format or 'unknown'}; "
                        f"expected one of {supported}",
                        code="invalid_request",
                    )
                width, height = opened.size
                if width < 1 or height < 1:
                    raise SshPodError(
                        "input image dimensions must be positive",
                        code="invalid_request",
                    )
                opened.verify()
    except SshPodError:
        raise
    except Exception as error:
        raise SshPodError(
            f"input image is invalid or unsafe ({type(error).__name__})",
            code="invalid_request",
        ) from None


def _result(
    artifact: CachedArtifact, cache_hit: bool, evidence: _Evidence
) -> SingleViewTrialResult:
    return SingleViewTrialResult(
        artifact=artifact,
        cache_hit=cache_hit,
        run_dir=evidence.run_dir,
        manifest_path=evidence.manifest_path,
        metrics_path=evidence.metrics_path,
        logs_path=evidence.logs_path,
    )


def _artifact_record(artifact: CachedArtifact) -> dict[str, Any]:
    payload = artifact.glb_path.read_bytes()
    return {
        "cache_key": artifact.key,
        "path": str(artifact.glb_path),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _new_run_id() -> str:
    timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    return f"trial-{timestamp}-{uuid.uuid4().hex[:12]}"


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    _atomic_write(path, json.dumps(dict(payload), indent=2, sort_keys=True).encode("utf-8"))


def _atomic_write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    data = b"".join(
        json.dumps(record, ensure_ascii=False, sort_keys=True).encode("utf-8") + b"\n"
        for record in records
    )
    _atomic_write(path, data)


def _atomic_write(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, prefix=f"{path.name}.", suffix=".partial")
    try:
        with os.fdopen(handle, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise
