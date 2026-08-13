"""Production adapter between the Qt trial protocol and the public service."""

from __future__ import annotations

import threading
import time
from contextlib import suppress
from pathlib import Path
from typing import cast

from pixaboost.backends.cache import ArtifactCache
from pixaboost.backends.pixal3d import GenerationParams, JobRunner
from pixaboost.backends.ssh_pod import (
    CancelState,
    ExistingPodUseApproval,
    SshPodClient,
    SshPodConfig,
    SshPodError,
)
from pixaboost.gui.model import RunState
from pixaboost.gui.remote_trial import (
    RemoteEventSink,
    RemoteTrialRequest,
    RemoteTrialResult,
    project_git_sha,
)
from pixaboost.observability import TelemetryEvent
from pixaboost.trials.single_view import (
    SingleViewClientFactory,
    SingleViewTrialConfig,
    TransportEventSink,
    preflight_single_view,
    run_single_view_trial,
)


class ExistingPodSingleViewRunner:
    """Cache-first service adapter for one already-running SSH Pod.

    Construction and ``preflight`` are local-only.  The remote client is built
    inside ``run_single_view_trial`` after the one-shot approval is present.
    """

    def __init__(
        self,
        repo_root: Path,
        *,
        params: GenerationParams | None = None,
        project_git_revision: str | None = None,
        client_factory: SingleViewClientFactory | None = None,
    ) -> None:
        self._repo_root = Path(repo_root).resolve()
        self._params = params or GenerationParams(resolution=1024, low_vram=True)
        self._project_git_revision = project_git_revision or project_git_sha(self._repo_root)
        self._client_factory = client_factory
        self._prepared_request: RemoteTrialRequest | None = None
        self._prepared_config: SingleViewTrialConfig | None = None
        self._active_client: object | None = None
        self._cancel_requested = False
        self._client_lock = threading.Lock()

    def preflight(self, request: RemoteTrialRequest) -> bool:
        config = self._config_for(request)
        preflight = preflight_single_view(config)
        with self._client_lock:
            self._cancel_requested = False
        self._prepared_request = request
        self._prepared_config = config
        return preflight.cache_hit

    def run(
        self,
        request: RemoteTrialRequest,
        *,
        approve_existing_pod: bool,
        event_sink: RemoteEventSink,
    ) -> RemoteTrialResult:
        config = (
            self._prepared_config
            if request == self._prepared_request and self._prepared_config is not None
            else self._config_for(request)
        )
        approval = ExistingPodUseApproval.grant(config.ssh) if approve_existing_pod else None
        started_at = time.monotonic()

        def relay(event: TelemetryEvent) -> None:
            event_sink(event)

        def capture_client(
            ssh_config: SshPodConfig,
            one_shot_approval: ExistingPodUseApproval | None,
            transport_sink: TransportEventSink,
        ) -> JobRunner:
            with self._client_lock:
                if self._cancel_requested:
                    raise SshPodError(
                        "trial cancelled before SSH client construction",
                        code="remote_cancelled",
                        cancel_state=CancelState.ACKNOWLEDGED,
                    )
            factory = self._client_factory or _default_client_factory
            client = factory(ssh_config, one_shot_approval, transport_sink)
            if not self._publish_client(client):
                cancel = getattr(client, "cancel", None)
                if callable(cancel):
                    with suppress(Exception):
                        cancel()
                raise SshPodError(
                    "trial cancelled before SSH client publication",
                    code="remote_cancelled",
                    cancel_state=CancelState.ACKNOWLEDGED,
                )
            return client

        try:
            result = run_single_view_trial(
                config,
                approval=approval,
                event_sink=relay,
                client_factory=cast(SingleViewClientFactory, capture_client),
            )
        except SshPodError as error:
            if error.code != "remote_cancelled":
                raise
            return RemoteTrialResult(
                state=RunState.CANCELLED,
                duration_seconds=time.monotonic() - started_at,
                error=(
                    "Annulation locale, état Pod inconnu."
                    if error.cancel_state is CancelState.UNKNOWN
                    and not error.remote_terminal
                    else str(error)
                ),
                cancel_state=error.cancel_state,
                remote_terminal=error.remote_terminal,
            )
        finally:
            with self._client_lock:
                self._active_client = None

        return RemoteTrialResult(
            state=RunState.SUCCEEDED,
            duration_seconds=time.monotonic() - started_at,
            exit_code=0,
            artifacts=(
                result.artifact.glb_path,
                result.manifest_path,
                result.metrics_path,
                result.logs_path,
            ),
            cache_hit=result.cache_hit,
        )

    def cancel(self) -> CancelState:
        with self._client_lock:
            client = self._active_client
            if client is None:
                self._cancel_requested = True
                return CancelState.ACKNOWLEDGED
        cancel = getattr(client, "cancel", None)
        if not callable(cancel):
            return CancelState.UNKNOWN
        try:
            state = cancel()
        except Exception:
            return CancelState.UNKNOWN
        return state if isinstance(state, CancelState) else CancelState.UNKNOWN

    def _publish_client(self, client: object) -> bool:
        """Atomically publish the client unless cancellation won the race."""
        with self._client_lock:
            if self._cancel_requested:
                return False
            self._active_client = client
            return True

    def _config_for(self, request: RemoteTrialRequest) -> SingleViewTrialConfig:
        if request.project_git_sha != self._project_git_revision:
            raise ValueError("PixaBoost revision does not match the active GUI checkout")
        return SingleViewTrialConfig(
            image_path=request.image_path,
            params=self._params,
            cache=ArtifactCache(self._repo_root / "artifacts"),
            ssh=SshPodConfig(
                host=request.host,
                username=request.username,
                private_key_path=request.private_key_path,
                known_hosts_path=request.known_hosts_path,
                expected_pixal3d_sha=request.expected_pixal3d_sha,
                project_git_sha=self._project_git_revision,
                local_runs_root=self._repo_root / "runs",
            ),
        )


def _default_client_factory(
    config: SshPodConfig,
    approval: ExistingPodUseApproval | None,
    event_sink: TransportEventSink,
) -> JobRunner:
    return SshPodClient(config, approval=approval, event_sink=event_sink)
