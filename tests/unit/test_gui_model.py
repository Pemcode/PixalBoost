"""Pure contracts shared by the observed commands and the PyQt6 GUI (F08)."""

from __future__ import annotations

from pathlib import Path

import pytest

from pixaboost.gui.model import (
    CommandSpec,
    RunState,
    default_commands,
    discover_artifacts,
    interpret_output_line,
    sanitise_command,
    sanitise_log_line,
)
from pixaboost.observability import (
    EventProtocolError,
    TelemetryEvent,
    encode_event,
    parse_event_line,
)

SECRET = "rpa_TEST_SECRET_0123456789"


def test_a_structured_telemetry_event_round_trips() -> None:
    event = TelemetryEvent(
        phase="benchmark",
        stage="rendu",
        progress=0.5,
        message="l_bracket — az000_el+00",
        artifact=Path("data/bench/manifest.json"),
    )
    assert parse_event_line(encode_event(event)) == event


@pytest.mark.parametrize("progress", [-0.01, 1.01, float("nan")])
def test_invalid_progress_is_rejected_instead_of_being_silently_clamped(progress: float) -> None:
    with pytest.raises(EventProtocolError, match="progress"):
        TelemetryEvent(phase="test", progress=progress)


def test_malformed_structured_output_is_an_actionable_protocol_error() -> None:
    with pytest.raises(EventProtocolError, match="valid JSON"):
        parse_event_line('PIXABOOST_EVENT {"phase":')


def test_boolean_progress_is_rejected_even_though_bool_is_an_int_subclass() -> None:
    with pytest.raises(EventProtocolError, match="progress"):
        parse_event_line('PIXABOOST_EVENT {"phase":"test","progress":true}')


def test_command_display_redacts_secret_flags_assignments_and_bearer_tokens() -> None:
    opaque_api_key = "opaque-api-key-value-4f82"
    opaque_authorization = "opaque-authorization-value-9b31"
    opaque_password = "opaque-password-value-2d17"
    opaque_query_token = "opaque-query-token-value-7e63"
    display = sanitise_command(
        "runner",
        (
            "--api-key",
            opaque_api_key,
            f"--token={SECRET}",
            f"RUNPOD_API_KEY={SECRET}",
            "Authorization:",
            "Bearer",
            opaque_authorization,
            "--api_key",
            SECRET,
            "--access-token",
            SECRET,
            "-p",
            opaque_password,
            f"https://user:password@example.test/run?ordinary=yes&token={opaque_query_token}",
            "-H",
            "Authorization: Bearer opaque-header-value-3c71",
            "--ordinary",
            "visible",
        ),
    )
    for secret in (
        SECRET,
        opaque_api_key,
        opaque_authorization,
        opaque_password,
        opaque_query_token,
        "opaque-header-value-3c71",
        "user:password",
    ):
        assert secret not in display
    assert display.count("[REDACTED]") >= 7
    assert "visible" in display


@pytest.mark.parametrize(
    "argument",
    [
        "--header=Authorization: Bearer TOPSECRET",
        "-pTOPSECRET",
        "RUN_URL=https://example.test/run?ordinary=yes&token=TOPSECRET",
        "--password=TOPSECRET",
        "https://user:TOPSECRET@example.test/run",
    ],
)
def test_command_display_redacts_attached_and_nested_credentials(argument: str) -> None:
    display = sanitise_command("runner", (argument, "visible"))
    assert "TOPSECRET" not in display
    assert "visible" in display


@pytest.mark.parametrize(
    "line",
    [
        "request --header=Authorization: Bearer TOPSECRET failed",
        "retrying -pTOPSECRET now",
        "RUN_URL=https://example.test/run?ordinary=yes&token=TOPSECRET",
        "Authorization: Basic TOPSECRET",
        "connected to https://user:TOPSECRET@example.test/run",
        "token=TOPSECRET",
        "RUNPOD_API_KEY=TOPSECRET",
        "X-Api-Key: TOPSECRET",
        "password=-p123456",
    ],
)
def test_log_lines_redact_common_credentials_without_hiding_context(line: str) -> None:
    sanitised = sanitise_log_line(line)
    assert "TOPSECRET" not in sanitised
    assert "[REDACTED]" in sanitised


def test_default_gui_commands_are_cpu_only_and_benchmark_requires_a_manifest(
    tmp_path: Path,
) -> None:
    commands = default_commands(tmp_path)
    assert {command.key for command in commands} == {"check", "test", "bench-build"}
    assert all(command.cost == "gratuit — CPU local" for command in commands)
    assert all("runpod" not in command.display_command.lower() for command in commands)
    benchmark = next(command for command in commands if command.key == "bench-build")
    assert benchmark.required_artifacts == (tmp_path / "data" / "bench" / "manifest.json",)
    assert "--events-jsonl" in benchmark.arguments


def test_command_spec_refuses_an_empty_program(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="program"):
        CommandSpec(
            key="broken",
            label="Broken",
            description="Cannot run",
            program=" ",
            working_directory=tmp_path,
        )


def test_plain_process_output_yields_best_effort_phases_and_progress() -> None:
    typecheck = interpret_output_line("Poe => mypy src")
    pytest_progress = interpret_output_line("........ [ 38%]")
    assert typecheck is not None and typecheck.phase == "verification des types"
    assert pytest_progress is not None and pytest_progress.progress == pytest.approx(0.38)


def test_artifact_discovery_is_bounded_to_the_three_repository_roots(tmp_path: Path) -> None:
    glb = tmp_path / "artifacts" / "sample" / "mesh.glb"
    run_manifest = tmp_path / "runs" / "20260813-120000" / "manifest.json"
    bench_manifest = tmp_path / "data" / "bench" / "manifest.json"
    ignored = tmp_path / "somewhere-else" / "not-an-artifact.txt"
    for path in (glb, run_manifest, bench_manifest, ignored):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"glTF" if path.suffix == ".glb" else b"{}")

    entries = discover_artifacts(tmp_path)

    assert {entry.path for entry in entries} == {glb, run_manifest, bench_manifest}
    assert next(entry for entry in entries if entry.path == glb).kind == "GLB"
    assert all(entry.size_bytes >= 2 for entry in entries)


def test_terminal_states_are_explicit() -> None:
    assert RunState.SUCCEEDED.is_terminal
    assert RunState.FAILED.is_terminal
    assert RunState.CANCELLED.is_terminal
    assert not RunState.RUNNING.is_terminal
