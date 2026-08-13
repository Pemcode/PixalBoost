"""Small JSONL telemetry contract for observable PixaBoost commands."""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path

EVENT_PREFIX = "PIXABOOST_EVENT "


class EventProtocolError(ValueError):
    """A child process emitted a line that violates the telemetry contract."""


@dataclass(frozen=True)
class TelemetryEvent:
    """One observable checkpoint emitted by a PixaBoost command."""

    phase: str
    stage: str = ""
    progress: float | None = None
    message: str = ""
    artifact: Path | None = None

    def __post_init__(self) -> None:
        if not self.phase.strip() or "\n" in self.phase or "\r" in self.phase:
            raise EventProtocolError("phase must be a non-empty single line")
        if self.progress is not None and (
            isinstance(self.progress, bool)
            or not math.isfinite(self.progress)
            or not 0.0 <= self.progress <= 1.0
        ):
            raise EventProtocolError("progress must be a finite number between 0 and 1")
        for name, value in (("stage", self.stage), ("message", self.message)):
            if "\n" in value or "\r" in value:
                raise EventProtocolError(f"{name} must be a single line")


def encode_event(event: TelemetryEvent) -> str:
    """Encode an event as one prefixed JSONL record for stdout."""
    payload = asdict(event)
    if event.artifact is not None:
        payload["artifact"] = str(event.artifact)
    return EVENT_PREFIX + json.dumps(payload, ensure_ascii=False, sort_keys=True)


def parse_event_line(line: str) -> TelemetryEvent:
    """Parse one prefixed telemetry record, rejecting malformed evidence."""
    stripped = line.strip()
    if not stripped.startswith(EVENT_PREFIX):
        raise EventProtocolError(f"telemetry line must start with {EVENT_PREFIX!r}")
    raw = stripped[len(EVENT_PREFIX) :]
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as error:
        raise EventProtocolError(f"telemetry payload is not valid JSON: {error.msg}") from None
    if not isinstance(payload, dict):
        raise EventProtocolError("telemetry payload must be a JSON object")

    phase = payload.get("phase")
    if not isinstance(phase, str):
        raise EventProtocolError("telemetry phase must be a string")
    stage = payload.get("stage", "")
    message = payload.get("message", "")
    progress = payload.get("progress")
    artifact = payload.get("artifact")
    if not isinstance(stage, str) or not isinstance(message, str):
        raise EventProtocolError("telemetry stage and message must be strings")
    if progress is not None and (
        isinstance(progress, bool) or not isinstance(progress, (int, float))
    ):
        raise EventProtocolError("telemetry progress must be numeric or null")
    if artifact is not None and not isinstance(artifact, str):
        raise EventProtocolError("telemetry artifact must be a path string or null")

    return TelemetryEvent(
        phase=phase,
        stage=stage,
        progress=float(progress) if progress is not None else None,
        message=message,
        artifact=Path(artifact) if artifact else None,
    )
