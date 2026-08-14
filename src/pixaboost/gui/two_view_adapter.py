"""Run the two-view trial against one already-running SSH Pod (F15).

The sibling of `single_view_adapter.py`, and it holds the two wires the trial
deliberately does not own: where a reconstruction comes from, and where a mask
comes from.

**Reconstruction** is `run_single_view_trial`, once per photograph, cache-first.
Nothing here provisions, buys or starts anything; a cache miss needs the same
explicit one-shot approval as a mono-view trial, and a *fresh* approval is
granted per photograph because a grant is one-shot and expires in two minutes
while a reconstruction takes tens of them.

**Masks** are the alpha channel of the input cutouts. That is the whole reason
this engine demands RGBA files rather than photographs: the mask that decides
the derived pose is then, byte for byte, the mask the user validated in the
segmentation tab and the mask Pixal3D reconstructed from. See ADR-0016.
"""

from __future__ import annotations

import re
import threading
from collections.abc import Callable
from contextlib import suppress
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np

from pixaboost.backends.cache import ArtifactCache
from pixaboost.backends.images import read_object_mask
from pixaboost.backends.pixal3d import GenerationParams, JobRunner
from pixaboost.backends.ssh_pod import (
    CancelState,
    ExistingPodUseApproval,
    SshPodClient,
    SshPodConfig,
    SshPodError,
)
from pixaboost.observability import TelemetryEvent
from pixaboost.trials.single_view import (
    SingleViewClientFactory,
    SingleViewTrialConfig,
    TransportEventSink,
    preflight_single_view,
    run_single_view_trial,
)
from pixaboost.trials.two_view import TwoViewConfig, TwoViewResult, run_two_view_trial

_FULL_GIT_SHA = re.compile(r"^[0-9a-f]{40}$")

BoolArray = np.ndarray
#: Progress relay into the GUI log. Both reconstructions and the transport
#: report through it, so a 50-minute run is never a frozen window.
TelemetrySink = Callable[[TelemetryEvent], None]


@dataclass(frozen=True)
class PodSettings:
    """Non-secret description of the Pod, shared by both reconstructions.

    Frozen on purpose: it is read once on the GUI thread and handed to a
    worker, so no field can change under a run that is already paying for it.
    """

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
        """Same rules as a mono-view request, minus the image it has no say in."""
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


@dataclass(frozen=True)
class TwoViewPreflight:
    """Purely local cache decision for both photographs.

    Reports the two views separately because the confirmation dialog has to
    name what will actually be bought: one reconstruction or two.
    """

    front_image: Path
    back_image: Path
    front_cache_hit: bool
    back_cache_hit: bool

    @property
    def missing(self) -> tuple[Path, ...]:
        """The photographs that would require the Pod, in run order."""
        return tuple(
            image
            for image, hit in (
                (self.front_image, self.front_cache_hit),
                (self.back_image, self.back_cache_hit),
            )
            if not hit
        )

    @property
    def approval_required(self) -> bool:
        return bool(self.missing)


class ExistingPodTwoViewEngine:
    """Cache-first two-view engine bound to one already-running Pod."""

    def __init__(
        self,
        repo_root: Path,
        settings: PodSettings,
        *,
        params: GenerationParams | None = None,
        client_factory: SingleViewClientFactory | None = None,
    ) -> None:
        self._repo_root = Path(repo_root).resolve()
        self._settings = settings
        self._params = params or GenerationParams(resolution=1024, low_vram=True)
        self._client_factory = client_factory
        self._cache = ArtifactCache(self._repo_root / "artifacts")
        self._lock = threading.Lock()
        self._active_client: object | None = None
        self._cancel_requested = False

    # -- local, free -------------------------------------------------------

    def preflight(self, config: TwoViewConfig) -> TwoViewPreflight:
        """Validate the inputs and resolve the cache, with no connection.

        Reading both masks here is what makes a wrong file a free mistake: the
        user is told at click time, before the confirmation dialog, instead of
        an hour into a billed run.
        """
        front, back = Path(config.front_image), Path(config.back_image)
        read_object_mask(front)
        read_object_mask(back)
        with self._lock:
            self._cancel_requested = False
        return TwoViewPreflight(
            front_image=front,
            back_image=back,
            front_cache_hit=preflight_single_view(self._single_config(front)).cache_hit,
            back_cache_hit=preflight_single_view(self._single_config(back)).cache_hit,
        )

    # -- the run -----------------------------------------------------------

    def run(
        self,
        config: TwoViewConfig,
        *,
        approve_existing_pod: bool,
        event_sink: TelemetrySink | None = None,
    ) -> TwoViewResult:
        """Reconstruct both cutouts, derive the pose, write the aligned GLB."""
        relay = event_sink or (lambda _event: None)

        def reconstruct(photo: Path) -> Path:
            self._raise_if_cancelled()
            single = self._single_config(Path(photo))
            approval = (
                ExistingPodUseApproval.grant(single.ssh) if approve_existing_pod else None
            )
            relay(
                TelemetryEvent(
                    phase="reconstruction",
                    stage=Path(photo).name,
                    message=f"Reconstruction de {Path(photo).name}",
                )
            )
            result = run_single_view_trial(
                single,
                approval=approval,
                event_sink=relay,
                client_factory=cast(SingleViewClientFactory, self._capture_client),
            )
            return result.artifact.glb_path

        try:
            return run_two_view_trial(config, reconstruct=reconstruct, mask_of=self._mask_of)
        finally:
            with self._lock:
                self._active_client = None

    def cancel(self) -> CancelState:
        """Stop the run; a reconstruction not yet started is never bought."""
        with self._lock:
            self._cancel_requested = True
            client = self._active_client
        if client is None:
            return CancelState.ACKNOWLEDGED
        cancel = getattr(client, "cancel", None)
        if not callable(cancel):
            return CancelState.UNKNOWN
        try:
            state = cancel()
        except Exception:
            return CancelState.UNKNOWN
        return state if isinstance(state, CancelState) else CancelState.UNKNOWN

    # -- internals ---------------------------------------------------------

    def _mask_of(self, photo: Path) -> BoolArray:
        return read_object_mask(Path(photo))

    def _raise_if_cancelled(self) -> None:
        with self._lock:
            cancelled = self._cancel_requested
        if cancelled:
            raise SshPodError(
                "two-view trial cancelled before this reconstruction started",
                code="remote_cancelled",
                cancel_state=CancelState.ACKNOWLEDGED,
            )

    def _capture_client(
        self,
        ssh_config: SshPodConfig,
        approval: ExistingPodUseApproval | None,
        transport_sink: TransportEventSink,
    ) -> JobRunner:
        self._raise_if_cancelled()
        factory = self._client_factory or _default_client_factory
        client = factory(ssh_config, approval, transport_sink)
        with self._lock:
            if self._cancel_requested:
                published = False
            else:
                self._active_client = client
                published = True
        if not published:
            cancel = getattr(client, "cancel", None)
            if callable(cancel):
                with suppress(Exception):
                    cancel()
            raise SshPodError(
                "two-view trial cancelled before SSH client publication",
                code="remote_cancelled",
                cancel_state=CancelState.ACKNOWLEDGED,
            )
        return client

    def _single_config(self, image: Path) -> SingleViewTrialConfig:
        return SingleViewTrialConfig(
            image_path=image,
            params=self._params,
            cache=self._cache,
            ssh=SshPodConfig(
                host=self._settings.host,
                username=self._settings.username,
                private_key_path=self._settings.private_key_path,
                known_hosts_path=self._settings.known_hosts_path,
                expected_pixal3d_sha=self._settings.expected_pixal3d_sha,
                project_git_sha=self._settings.project_git_sha,
                local_runs_root=self._repo_root / "runs",
            ),
        )


def _default_client_factory(
    config: SshPodConfig,
    approval: ExistingPodUseApproval | None,
    event_sink: TransportEventSink,
) -> JobRunner:
    return SshPodClient(config, approval=approval, event_sink=event_sink)
