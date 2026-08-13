"""Qt-free command, state and artifact models for the experiment GUI."""

from __future__ import annotations

import re
import stat
import subprocess
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

from pixaboost.observability import EVENT_PREFIX, TelemetryEvent, parse_event_line


class RunState(StrEnum):
    IDLE = "idle"
    STARTING = "starting"
    RUNNING = "running"
    CANCELLING = "cancelling"
    SUCCEEDED = "succeeded"
    FAILED = "failed"
    CANCELLED = "cancelled"

    @property
    def is_terminal(self) -> bool:
        return self in {self.SUCCEEDED, self.FAILED, self.CANCELLED}


STATE_LABELS = {
    RunState.IDLE: "pret",
    RunState.STARTING: "demarrage",
    RunState.RUNNING: "en cours",
    RunState.CANCELLING: "arret demande",
    RunState.SUCCEEDED: "termine",
    RunState.FAILED: "echec",
    RunState.CANCELLED: "arrete",
}


_SECRET_NAME = re.compile(
    r"(?i)(?:access[_-]?token|api[_-]?key|apikey|authorization|password|secret|token)"
)
_LIKELY_TOKEN = re.compile(r"(?i)^(?:rpa_|ghp_|hf_|sk-)[A-Za-z0-9_.-]+$")
_AUTHORIZATION_HEADER = re.compile(
    r"(?i)(\b(?:proxy-)?authorization\s*:\s*)"
    r"(?:(?:bearer|basic)\s+)?[^\s,;\"']+"
)
_SECRET_FLAG_VALUE = re.compile(
    r"(?i)(?<!\S)(--(?:access[_-]?token|api[_-]?key|apikey|authorization|password|secret|token))"
    r"(=|\s+)(?:(?:bearer|basic)\s+)?[^\s,;\"']+"
)
_SECRET_ASSIGNMENT = re.compile(
    r"(?i)(?<![\w-])"
    r"((?:[A-Za-z0-9]+[_-])*"
    r"(?:access[_-]?token|api[_-]?key|apikey|authorization|password|secret|token)"
    r"(?:[_-][A-Za-z0-9]+)*\s*[:=]\s*)"
    r"(?:\"[^\"]*\"|'[^']*'|[^\s,;]+)"
)
_SHORT_SECRET_VALUE = re.compile(r"(?i)(?<!\S)(-p)(?:=|\s+)?([^\s,;\"']+)")
_BEARER_TOKEN = re.compile(r"(?i)(\b(?:bearer|basic)\s+)[^\s,;\"']+")
_URL = re.compile(r"[A-Za-z][A-Za-z0-9+.-]*://[^\s\"']+")
_SHORT_SECRET_FLAGS = {"-p"}
_LIKELY_TOKEN_INLINE = re.compile(r"(?i)(?<![A-Za-z0-9_.-])(?:rpa_|ghp_|hf_|sk-)[A-Za-z0-9_.-]+")
_PERCENT = re.compile(r"\[\s*(\d{1,3})%\]")


def sanitise_command(program: str, arguments: tuple[str, ...]) -> str:
    """Render argv for display without leaking common credentials."""
    safe: list[str] = [sanitise_log_line(program)]
    redact_next = False
    authorization_scheme_allowed = False
    for argument in arguments:
        lowered = argument.lower()
        if redact_next:
            safe.append("[REDACTED]")
            if authorization_scheme_allowed and lowered in {"bearer", "basic"}:
                authorization_scheme_allowed = False
            else:
                redact_next = False
                authorization_scheme_allowed = False
            continue

        sanitised_url = _sanitise_url(argument)
        if sanitised_url != argument:
            safe.append(sanitised_url)
            continue

        name, separator, value = argument.partition("=")
        if separator and (_SECRET_NAME.search(name) or name.lower() in _SHORT_SECRET_FLAGS):
            safe.append(f"{name}=[REDACTED]")
        elif separator:
            safe.append(f"{name}={sanitise_log_line(value)}")
        elif lowered.startswith("-p") and len(argument) > 2:
            safe.append("-p[REDACTED]")
        elif lowered in _SHORT_SECRET_FLAGS or (
            lowered.startswith("-") and _SECRET_NAME.search(lowered.lstrip("-"))
        ):
            safe.append(argument)
            redact_next = True
        elif lowered == "bearer":
            safe.append("[REDACTED]")
            redact_next = True
        elif _SECRET_NAME.fullmatch(argument.rstrip(":")):
            safe.append(argument)
            redact_next = True
            authorization_scheme_allowed = argument.rstrip(":").lower() == "authorization"
        elif _LIKELY_TOKEN.fullmatch(argument):
            safe.append("[REDACTED]")
        else:
            safe.append(sanitise_log_line(argument))
    return subprocess.list2cmdline(safe)


def sanitise_log_line(line: str) -> str:
    """Redact common credentials before process output reaches any Qt signal."""
    sanitised = _URL.sub(lambda match: _sanitise_url(match.group(0)), line)
    sanitised = _AUTHORIZATION_HEADER.sub(r"\1[REDACTED]", sanitised)
    sanitised = _SECRET_FLAG_VALUE.sub(r"\1\2[REDACTED]", sanitised)
    sanitised = _SHORT_SECRET_VALUE.sub(_redact_short_secret, sanitised)
    sanitised = _SECRET_ASSIGNMENT.sub(r"\1[REDACTED]", sanitised)
    sanitised = _BEARER_TOKEN.sub(r"\1[REDACTED]", sanitised)
    return _LIKELY_TOKEN_INLINE.sub("[REDACTED]", sanitised)


def _redact_short_secret(match: re.Match[str]) -> str:
    return f"{match.group(1)}[REDACTED]"


def _sanitise_url(argument: str) -> str:
    """Redact URL userinfo and credential-shaped query parameters."""
    try:
        parsed = urlsplit(argument)
    except ValueError:
        return argument
    if not parsed.scheme or not parsed.netloc:
        return argument
    netloc = parsed.netloc
    changed = False
    if parsed.username is not None or parsed.password is not None:
        netloc = f"[REDACTED]@{parsed.netloc.rsplit('@', maxsplit=1)[-1]}"
        changed = True
    pairs = [
        (key, "[REDACTED]" if _SECRET_NAME.search(key) else value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
    ]
    changed = changed or any(_SECRET_NAME.search(key) for key, _value in pairs)
    if not changed:
        return argument
    return urlunsplit(
        (parsed.scheme, netloc, parsed.path, urlencode(pairs, safe="[]"), parsed.fragment)
    )


@dataclass(frozen=True)
class CommandSpec:
    key: str
    label: str
    description: str
    program: str
    working_directory: Path
    arguments: tuple[str, ...] = ()
    required_artifacts: tuple[Path, ...] = ()
    cost: str = "gratuit — CPU local"

    def __post_init__(self) -> None:
        if not self.key.strip():
            raise ValueError("command key must not be blank")
        if not self.program.strip():
            raise ValueError("command program must not be blank")

    @property
    def display_command(self) -> str:
        return sanitise_command(self.program, self.arguments)


@dataclass(frozen=True)
class RunResult:
    state: RunState
    exit_code: int | None
    duration_seconds: float
    error: str = ""
    artifacts: tuple[Path, ...] = ()

    def __post_init__(self) -> None:
        if not self.state.is_terminal:
            raise ValueError("run result state must be terminal")


@dataclass(frozen=True)
class ArtifactEntry:
    path: Path
    kind: str
    size_bytes: int


def default_commands(repo_root: Path) -> tuple[CommandSpec, ...]:
    root = Path(repo_root).resolve()
    return (
        CommandSpec(
            key="check",
            label="Gate CPU complet",
            description="Lint, types et tests CPU. Verification canonique du depot.",
            program="uv",
            arguments=("run", "poe", "check"),
            working_directory=root,
        ),
        CommandSpec(
            key="test",
            label="Tests CPU",
            description="Tests unitaires et integration, sans GPU ni reseau.",
            program="uv",
            arguments=("run", "poe", "test"),
            working_directory=root,
        ),
        CommandSpec(
            key="bench-build",
            label="Benchmark synthetique",
            description="Rend 3 pieces sous 18 vues et ecrit un manifeste reproductible.",
            program="uv",
            arguments=(
                "run",
                "python",
                "-m",
                "pixaboost.bench.build",
                "--events-jsonl",
            ),
            working_directory=root,
            required_artifacts=(root / "data" / "bench" / "manifest.json",),
        ),
    )


def interpret_output_line(line: str) -> TelemetryEvent | None:
    """Turn structured or familiar tool output into best-effort telemetry."""
    stripped = line.strip()
    if not stripped:
        return None
    if stripped.startswith(EVENT_PREFIX):
        return parse_event_line(stripped)

    phase_markers = {
        "Poe => ruff": "lint",
        "Poe => mypy": "verification des types",
        "Poe => pytest": "tests",
    }
    for marker, phase in phase_markers.items():
        if marker in stripped:
            return TelemetryEvent(phase=phase, message=stripped)

    match = _PERCENT.search(stripped)
    if match:
        percentage = min(100, int(match.group(1)))
        return TelemetryEvent(phase="tests", progress=percentage / 100.0, message=stripped)

    if stripped.startswith("benchmark written to "):
        artifact_root = Path(stripped.removeprefix("benchmark written to ").strip())
        return TelemetryEvent(
            phase="finalisation",
            progress=1.0,
            message=stripped,
            artifact=artifact_root / "manifest.json",
        )
    return None


def discover_artifacts(repo_root: Path) -> tuple[ArtifactEntry, ...]:
    """Inventory lightweight paths without parsing or loading large artifacts."""
    root = Path(repo_root)
    candidates: set[Path] = set()

    artifact_root = root / "artifacts"
    if artifact_root.is_dir():
        for pattern in ("**/*.glb", "**/manifest.json", "**/batch_report.json", "**/*_sheet.png"):
            candidates.update(artifact_root.glob(pattern))

    bench_manifest = root / "data" / "bench" / "manifest.json"
    candidates.add(bench_manifest)

    runs_root = root / "runs"
    if runs_root.is_dir():
        for name in ("manifest.json", "metrics.json", "logs.jsonl"):
            candidates.update(runs_root.glob(f"*/{name}"))

    entries = [entry for path in candidates if (entry := _artifact_entry(path)) is not None]
    return tuple(sorted(entries, key=lambda entry: str(entry.path).lower()))


def _artifact_entry(path: Path) -> ArtifactEntry | None:
    try:
        path_stat = path.stat()
    except OSError:
        return None
    if not stat.S_ISREG(path_stat.st_mode):
        return None
    return ArtifactEntry(path=path, kind=_artifact_kind(path), size_bytes=path_stat.st_size)


def _artifact_kind(path: Path) -> str:
    if path.suffix.lower() == ".glb":
        return "GLB"
    if path.name == "manifest.json":
        return "manifeste"
    if path.name == "metrics.json":
        return "metriques"
    if path.name == "logs.jsonl":
        return "journal"
    if path.suffix.lower() == ".png":
        return "apercu"
    return "rapport"
