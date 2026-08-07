"""Minimal RunPod serverless client.

Thin by design: submit a job, poll it, hand back the worker's output. No
business logic, no retry heuristics dressed up as intelligence -- see
backends/CONSTRAINTS.md.

The HTTP transport is injected rather than imported at the call site, so the
polling state machine is exercised as pure logic in unit tests and the network
only appears in tests/e2e.

The API key is treated as a secret throughout: it is never placed in a `repr`,
a log line, or an exception message.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

API_ROOT = "https://api.runpod.ai/v2"
DEFAULT_KEY_FILE = Path("runpod.env")

#: Statuses that mean "keep waiting". Anything else is terminal.
PENDING_STATUSES = frozenset({"IN_QUEUE", "IN_PROGRESS"})

Transport = Callable[[str, dict[str, Any] | None, dict[str, str]], Any]


class RunPodError(RuntimeError):
    """Anything that went wrong talking to RunPod, or that RunPod reported back."""


def load_api_key(source: Path | str | None = None) -> str:
    """Read the RunPod API key from ``RUNPOD_API_KEY`` or from a file.

    The file may hold a bare token on its own line -- which is how the real
    `runpod.env` is written -- or a ``NAME=value`` pair. Comments and blank
    lines are ignored. No error raised here ever contains the key itself.
    """
    from_env = os.environ.get("RUNPOD_API_KEY", "").strip()
    if from_env:
        return from_env

    path = Path(source) if source is not None else DEFAULT_KEY_FILE
    if not path.is_file():
        raise RunPodError(
            f"no RunPod credentials: set RUNPOD_API_KEY, or create {path}. "
            f"The file may contain the bare token on one line."
        )

    candidates: list[str] = []
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        _, separator, value = line.partition("=")
        candidates.append((value if separator else line).strip().strip("'\""))

    if not candidates:
        raise RunPodError(f"{path} contains no API key (only blanks or comments)")
    if len(candidates) > 1:
        raise RunPodError(
            f"{path} holds {len(candidates)} candidate values; it must contain exactly one key"
        )
    return candidates[0]


@dataclass(frozen=True)
class RunPodEndpoint:
    """Where to send jobs. Compares and prints without exposing the key."""

    endpoint_id: str
    api_key: str = field(repr=False)
    api_root: str = API_ROOT

    def __post_init__(self) -> None:
        if not self.endpoint_id.strip():
            raise RunPodError("endpoint id must not be blank")
        if not self.api_key.strip():
            raise RunPodError("api key must not be blank")

    def headers(self) -> dict[str, str]:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

    def run_url(self) -> str:
        return f"{self.api_root}/{self.endpoint_id.strip()}/run"

    def status_url(self, job_id: str) -> str:
        return f"{self.api_root}/{self.endpoint_id.strip()}/status/{job_id}"

    def health_url(self) -> str:
        return f"{self.api_root}/{self.endpoint_id.strip()}/health"


def _urllib_transport(url: str, body: dict[str, Any] | None, headers: dict[str, str]) -> Any:
    data = json.dumps(body).encode("utf-8") if body is not None else None
    request = urllib.request.Request(
        url, data=data, headers=headers, method="POST" if data else "GET"
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as error:
        # Deliberately does not echo headers: they carry the bearer token.
        raise RunPodError(f"RunPod returned HTTP {error.code} for {url}") from None
    except urllib.error.URLError as error:
        raise RunPodError(f"could not reach RunPod at {url}: {error.reason}") from None


class RunPodClient:
    """Submit one job and wait for it."""

    def __init__(
        self,
        endpoint: RunPodEndpoint,
        *,
        transport: Transport | None = None,
        sleep: Callable[[float], None] = time.sleep,
        clock: Callable[[], float] = time.monotonic,
        poll_interval_seconds: float = 5.0,
        timeout_seconds: float = 1800.0,
    ) -> None:
        self.endpoint = endpoint
        self._transport = transport or _urllib_transport
        self._sleep = sleep
        self._clock = clock
        self.poll_interval_seconds = poll_interval_seconds
        self.timeout_seconds = timeout_seconds

    def health(self) -> dict[str, Any]:
        result = self._transport(self.endpoint.health_url(), None, self.endpoint.headers())
        return dict(result)

    def run(self, payload: Mapping[str, Any]) -> dict[str, Any]:
        """Submit ``payload`` as the job input and block until it resolves."""
        submitted = self._transport(
            self.endpoint.run_url(), {"input": dict(payload)}, self.endpoint.headers()
        )
        job_id = submitted.get("id") if isinstance(submitted, Mapping) else None
        if not job_id:
            raise RunPodError(f"submission returned no job id: {submitted}")

        started = self._clock()
        while True:
            state = self._transport(
                self.endpoint.status_url(str(job_id)), None, self.endpoint.headers()
            )
            status = str(state.get("status", "UNKNOWN"))

            if status == "COMPLETED":
                output = state.get("output") or {}
                # The worker answers HTTP 200 with an `error` key when the
                # handler itself failed; treating that as success would cache a
                # broken artefact.
                if isinstance(output, Mapping) and output.get("error"):
                    raise RunPodError(f"worker reported: {output['error']}")
                return dict(output)

            if status not in PENDING_STATUSES:
                detail = state.get("error") or status
                raise RunPodError(f"job {job_id} ended as {status}: {detail}")

            if self._clock() - started > self.timeout_seconds:
                raise RunPodError(
                    f"job {job_id} did not finish within {self.timeout_seconds:.0f}s "
                    f"(last status {status})"
                )
            self._sleep(self.poll_interval_seconds)
