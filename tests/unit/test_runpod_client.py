"""TDD for backends.runpod_client (F07).

No network here. The HTTP transport is injected, so the polling state machine,
the URL construction and the credential loading are all exercised as pure logic.
The real endpoint is only touched by tests/e2e.

One requirement is a security requirement rather than a functional one: the API
key must never reach a log line or an exception message.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from pixaboost.backends.runpod_client import (
    RunPodClient,
    RunPodEndpoint,
    RunPodError,
    load_api_key,
)

SECRET = "rpa_TESTKEYVALUE0123456789"


# ---------------------------------------------------------------------------
# Credentials
# ---------------------------------------------------------------------------


def test_a_bare_token_on_one_line_is_accepted(tmp_path: Path) -> None:
    """This is the shape of the real runpod.env: a naked token, no key=value."""
    source = tmp_path / "runpod.env"
    source.write_text(f"{SECRET}\n", encoding="utf-8")
    assert load_api_key(source) == SECRET


def test_a_key_equals_value_line_is_accepted(tmp_path: Path) -> None:
    source = tmp_path / "runpod.env"
    source.write_text(f"# comment\n\nRUNPOD_API_KEY={SECRET}\n", encoding="utf-8")
    assert load_api_key(source) == SECRET


def test_quotes_and_whitespace_are_stripped(tmp_path: Path) -> None:
    source = tmp_path / "runpod.env"
    source.write_text(f'  RUNPOD_API_KEY = "{SECRET}"  \n', encoding="utf-8")
    assert load_api_key(source) == SECRET


def test_the_environment_variable_wins_over_the_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "runpod.env"
    source.write_text("from_file\n", encoding="utf-8")
    monkeypatch.setenv("RUNPOD_API_KEY", SECRET)
    assert load_api_key(source) == SECRET


def test_a_missing_key_fails_with_actionable_guidance(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    with pytest.raises(RunPodError, match="RUNPOD_API_KEY"):
        load_api_key(tmp_path / "absent.env")


def test_an_empty_file_is_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    source = tmp_path / "runpod.env"
    source.write_text("\n# nothing but a comment\n", encoding="utf-8")
    with pytest.raises(RunPodError, match="no API key"):
        load_api_key(source)


def test_a_malformed_file_never_leaks_the_secret_into_the_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("RUNPOD_API_KEY", raising=False)
    source = tmp_path / "runpod.env"
    source.write_text(f"KEY_ONE={SECRET}\nKEY_TWO={SECRET}\n", encoding="utf-8")
    try:
        load_api_key(source)
    except RunPodError as error:
        assert SECRET not in str(error)


def test_the_endpoint_never_reprs_its_key() -> None:
    endpoint = RunPodEndpoint(endpoint_id="abc", api_key=SECRET)
    assert SECRET not in repr(endpoint)
    assert SECRET not in str(endpoint)
    assert endpoint.headers()["Authorization"] == f"Bearer {SECRET}"


# ---------------------------------------------------------------------------
# URLs
# ---------------------------------------------------------------------------


def test_urls_are_built_from_the_endpoint_id() -> None:
    endpoint = RunPodEndpoint(endpoint_id="xyz789", api_key=SECRET)
    assert endpoint.run_url() == "https://api.runpod.ai/v2/xyz789/run"
    assert endpoint.status_url("job-1") == "https://api.runpod.ai/v2/xyz789/status/job-1"
    assert endpoint.health_url() == "https://api.runpod.ai/v2/xyz789/health"


def test_an_empty_endpoint_id_is_rejected() -> None:
    with pytest.raises(RunPodError, match="endpoint id"):
        RunPodEndpoint(endpoint_id="  ", api_key=SECRET)


# ---------------------------------------------------------------------------
# The polling state machine
# ---------------------------------------------------------------------------


class FakeTransport:
    """Replays a scripted sequence of responses and records the calls made."""

    def __init__(self, responses: list[dict[str, Any]]) -> None:
        self.responses = list(responses)
        self.calls: list[tuple[str, dict[str, Any] | None]] = []

    def __call__(self, url: str, body: dict[str, Any] | None, headers: dict[str, str]) -> Any:
        self.calls.append((url, body))
        return self.responses.pop(0)


def client_with(responses: list[dict[str, Any]]) -> tuple[RunPodClient, FakeTransport]:
    transport = FakeTransport(responses)
    client = RunPodClient(
        RunPodEndpoint(endpoint_id="xyz789", api_key=SECRET),
        transport=transport,
        sleep=lambda _seconds: None,
    )
    return client, transport


def test_a_job_that_completes_immediately_returns_its_output() -> None:
    client, transport = client_with(
        [
            {"id": "job-1", "status": "IN_QUEUE"},
            {"status": "COMPLETED", "output": {"glb_bytes": 12}},
        ]
    )
    assert client.run({"ping": True}) == {"glb_bytes": 12}
    assert transport.calls[0][0].endswith("/run")
    assert transport.calls[0][1] == {"input": {"ping": True}}


def test_the_client_waits_through_queue_and_progress_states() -> None:
    client, transport = client_with(
        [
            {"id": "job-1", "status": "IN_QUEUE"},
            {"status": "IN_QUEUE"},
            {"status": "IN_PROGRESS"},
            {"status": "COMPLETED", "output": {"ok": True}},
        ]
    )
    assert client.run({}) == {"ok": True}
    assert len(transport.calls) == 4


def test_a_failed_job_raises_with_the_worker_error() -> None:
    client, _ = client_with(
        [
            {"id": "job-1", "status": "IN_QUEUE"},
            {"status": "FAILED", "error": "CUDA out of memory"},
        ]
    )
    with pytest.raises(RunPodError, match="CUDA out of memory"):
        client.run({})


def test_a_handler_level_error_is_surfaced_even_when_the_job_completes() -> None:
    """The worker returns HTTP 200 with an `error` key when the handler itself
    fails. Treating that as success would cache a broken artefact."""
    client, _ = client_with(
        [
            {"id": "job-1", "status": "IN_QUEUE"},
            {"status": "COMPLETED", "output": {"error": "inference wrote no GLB"}},
        ]
    )
    with pytest.raises(RunPodError, match="wrote no GLB"):
        client.run({})


@pytest.mark.parametrize("status", ["CANCELLED", "TIMED_OUT"])
def test_terminal_non_success_states_raise(status: str) -> None:
    client, _ = client_with([{"id": "job-1", "status": "IN_QUEUE"}, {"status": status}])
    with pytest.raises(RunPodError, match=status):
        client.run({})


def test_a_submission_without_a_job_id_raises() -> None:
    client, _ = client_with([{"error": "endpoint not found"}])
    with pytest.raises(RunPodError, match="no job id"):
        client.run({})


def test_polling_gives_up_after_the_deadline() -> None:
    ticks = iter([0.0, 10.0, 4000.0])
    transport = FakeTransport(
        [{"id": "job-1", "status": "IN_QUEUE"}, {"status": "IN_QUEUE"}, {"status": "IN_QUEUE"}]
    )
    client = RunPodClient(
        RunPodEndpoint(endpoint_id="xyz789", api_key=SECRET),
        transport=transport,
        sleep=lambda _seconds: None,
        clock=lambda: next(ticks),
        timeout_seconds=1800.0,
    )
    with pytest.raises(RunPodError, match="did not finish within"):
        client.run({})
