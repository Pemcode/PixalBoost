"""Wiring the two-view trial onto a real existing Pod (F15).

No SSH and no GPU: the transport is a fake `JobRunner` that returns a genuine
GLB, so what is under test is everything *around* the connection -- the cache
decision, the approval, where the mask comes from, and what happens when the
inputs are not cutouts.

The budget-critical assertions are the negative ones. Every test that says "no
client was constructed" is a test that a mistake costs nothing.
"""

from __future__ import annotations

import base64
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from pixaboost.backends.glb import write_placed_meshes
from pixaboost.backends.ssh_pod import CancelState, ExistingPodUseApproval, SshPodError
from pixaboost.bench.shapes import l_bracket
from pixaboost.core.segmentation import compose_rgba
from pixaboost.gui.two_view_adapter import ExistingPodTwoViewEngine, PodSettings
from pixaboost.trials.two_view import TwoViewConfig

SEARCH = {"search_resolution": 64, "azimuth_steps": 8, "elevation_steps": 1, "refine_rounds": 1}


def settings(tmp_path: Path) -> PodSettings:
    key, known_hosts = tmp_path / "id_ed25519", tmp_path / "known_hosts"
    key.write_text("key", encoding="utf-8")
    known_hosts.write_text("host key", encoding="utf-8")
    return PodSettings(
        host="pod.example.test",
        username="researcher",
        private_key_path=key,
        known_hosts_path=known_hosts,
        expected_pixal3d_sha="a" * 40,
        project_git_sha="b" * 40,
    )


def cutout(path: Path, *, shift: int = 0) -> Path:
    """An RGBA cutout, as the segmentation panel writes them."""
    mask = np.zeros((64, 64), dtype=bool)
    mask[16 : 48 + shift, 20:44] = True
    rgb = np.full((64, 64, 3), 170, dtype=np.uint8)
    Image.fromarray(compose_rgba(rgb, mask), mode="RGBA").save(path, format="PNG")
    return path


def photograph(path: Path) -> Path:
    """A raw JPEG: no alpha, therefore no mask."""
    Image.fromarray(np.full((64, 64, 3), 120, dtype=np.uint8), mode="RGB").save(path)
    return path


def real_glb_bytes(tmp_path: Path) -> bytes:
    vertices, faces = l_bracket()
    path = write_placed_meshes(tmp_path / "_fixture.glb", {"part": (vertices, faces, np.eye(3))})
    return path.read_bytes()


class FakeClient:
    """Stands in for `SshPodClient`; every construction is a billed connection."""

    def __init__(self, revision: str, payload: bytes) -> None:
        self.revision, self._payload = revision, payload
        self.cancelled = 0

    def run(self, _payload: dict[str, Any]) -> dict[str, Any]:
        return {
            "glb_base64": base64.b64encode(self._payload).decode("ascii"),
            "glb_bytes": len(self._payload),
            "pixal3d_sha": self.revision,
        }

    def cancel(self) -> CancelState:
        self.cancelled += 1
        return CancelState.ACKNOWLEDGED


def engine_for(
    tmp_path: Path,
    *,
    on_connect: Callable[[ExistingPodUseApproval | None], None] | None = None,
) -> tuple[ExistingPodTwoViewEngine, list[ExistingPodUseApproval | None]]:
    """An engine whose only remote surface is a fake, plus the approvals it saw."""
    approvals: list[ExistingPodUseApproval | None] = []
    glb = real_glb_bytes(tmp_path)

    def factory(ssh_config: Any, approval: Any, _sink: Any) -> FakeClient:
        approvals.append(approval)
        if on_connect is not None:
            on_connect(approval)
        return FakeClient(ssh_config.expected_pixal3d_sha, glb)

    engine = ExistingPodTwoViewEngine(
        tmp_path / "repo",
        settings(tmp_path),
        client_factory=factory,
    )
    return engine, approvals


def config_for(tmp_path: Path, front: Path, back: Path) -> TwoViewConfig:
    return TwoViewConfig(
        front_image=front, back_image=back, runs_root=tmp_path / "runs", **SEARCH
    )


# --------------------------------------------------------------------------
# refusals that cost nothing
# --------------------------------------------------------------------------


def test_a_raw_photograph_is_refused_before_any_connection(tmp_path: Path) -> None:
    """The mistake a user will make: picking the JPEG instead of the cutout."""
    engine, approvals = engine_for(tmp_path)
    config = config_for(
        tmp_path, cutout(tmp_path / "front.png"), photograph(tmp_path / "back.jpg")
    )

    with pytest.raises(ValueError, match=r"back\.jpg"):
        engine.preflight(config)
    assert approvals == [], "a bad input must not reach the Pod"


def test_the_front_photograph_is_checked_too(tmp_path: Path) -> None:
    engine, _ = engine_for(tmp_path)
    config = config_for(
        tmp_path, photograph(tmp_path / "front.jpg"), cutout(tmp_path / "back.png")
    )

    with pytest.raises(ValueError, match=r"front\.jpg"):
        engine.preflight(config)


def test_running_a_cache_miss_without_approval_never_connects(tmp_path: Path) -> None:
    """Constraint: no existing-Pod use without an explicit, one-shot approval."""
    engine, approvals = engine_for(tmp_path)
    config = config_for(
        tmp_path, cutout(tmp_path / "front.png"), cutout(tmp_path / "back.png", shift=8)
    )
    assert engine.preflight(config).approval_required

    with pytest.raises(SshPodError, match="approval"):
        engine.run(config, approve_existing_pod=False)
    assert approvals == []


# --------------------------------------------------------------------------
# an approved run
# --------------------------------------------------------------------------


def test_an_approved_run_produces_an_aligned_glb_and_a_manifest(tmp_path: Path) -> None:
    engine, approvals = engine_for(tmp_path)
    config = config_for(
        tmp_path, cutout(tmp_path / "front.png"), cutout(tmp_path / "back.png", shift=8)
    )

    result = engine.run(config, approve_existing_pod=True)

    assert result.glb_path.is_file()
    assert result.manifest_path.is_file()
    assert len(approvals) == 2, "two photographs is two reconstructions"


def test_each_reconstruction_gets_its_own_fresh_approval(tmp_path: Path) -> None:
    """`ExistingPodUseApproval` is one-shot and expires in two minutes.

    A single grant reused for the second photograph would be refused an hour
    into the run, after the first reconstruction had already been paid for.
    """
    engine, approvals = engine_for(tmp_path)
    config = config_for(
        tmp_path, cutout(tmp_path / "front.png"), cutout(tmp_path / "back.png", shift=8)
    )

    engine.run(config, approve_existing_pod=True)

    assert len(approvals) == 2
    assert approvals[0] is not approvals[1], "the same one-shot approval was reused"


def test_a_second_identical_run_costs_nothing(tmp_path: Path) -> None:
    """Constraint 9: an artefact already in the cache is never regenerated."""
    engine, approvals = engine_for(tmp_path)
    config = config_for(
        tmp_path, cutout(tmp_path / "front.png"), cutout(tmp_path / "back.png", shift=8)
    )
    engine.run(config, approve_existing_pod=True)
    assert len(approvals) == 2

    assert not engine.preflight(config).approval_required
    engine.run(config, approve_existing_pod=False)

    assert len(approvals) == 2, "the cached artefacts were regenerated"


def test_the_preflight_reports_each_photograph_separately(tmp_path: Path) -> None:
    """One cached view and one missing still needs the Pod, and says so."""
    engine, _ = engine_for(tmp_path)
    front, back = cutout(tmp_path / "front.png"), cutout(tmp_path / "back.png", shift=8)
    engine.run(config_for(tmp_path, front, back), approve_existing_pod=True)

    third = cutout(tmp_path / "third.png", shift=16)
    preflight = engine.preflight(config_for(tmp_path, front, third))

    assert preflight.front_cache_hit and not preflight.back_cache_hit
    assert preflight.approval_required
    assert preflight.missing == (third,)


def test_a_known_pose_is_recovered_from_the_cutouts_alpha_alone(tmp_path: Path) -> None:
    """End to end, with the answer known in advance.

    The fake transport always returns the same L-bracket, so the back cutout's
    alpha is set to a render of that bracket at a rotation the engine is never
    told. Recovering it proves the pose comes from the cutout's own alpha and
    from nothing else -- which is what makes the run reproducible from the two
    files alone.
    """
    from pixaboost.core.geometry import BlenderCamera, front_view_camera
    from pixaboost.core.pose_search import object_rotation, rotation_angle_between
    from pixaboost.core.render import rasterise_silhouette

    vertices, faces = l_bracket()
    truth = object_rotation(np.pi, 0.0, 0.0)
    silhouette = rasterise_silhouette(
        vertices @ truth.T,
        faces,
        front_view_camera(2.0),
        BlenderCamera(camera_angle_x=0.857556, resolution=96),
    )
    back = tmp_path / "back.png"
    rgb = np.full((*silhouette.shape, 3), 170, dtype=np.uint8)
    Image.fromarray(compose_rgba(rgb, silhouette), mode="RGBA").save(back, format="PNG")

    engine, _ = engine_for(tmp_path)
    config = TwoViewConfig(
        front_image=cutout(tmp_path / "front.png"),
        back_image=back,
        runs_root=tmp_path / "runs",
        search_resolution=96,
        azimuth_steps=12,
        elevation_steps=3,
        refine_rounds=2,
    )
    result = engine.run(config, approve_existing_pod=True)

    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    recovered = np.asarray(manifest["search"]["front_to_back_rotation"])
    assert rotation_angle_between(recovered, truth) < np.deg2rad(10.0), (
        f"recovered a pose {np.rad2deg(rotation_angle_between(recovered, truth)):.1f} deg away"
    )
    assert result.pose_iou > 0.90, f"the silhouettes do not agree: {result.pose_iou:.3f}"


def test_a_cancel_before_any_client_exists_is_acknowledged(tmp_path: Path) -> None:
    engine, _ = engine_for(tmp_path)

    assert engine.cancel() is CancelState.ACKNOWLEDGED


def test_a_cancel_requested_first_stops_the_run_from_connecting(tmp_path: Path) -> None:
    """Cancelling a 50-minute run must not wait for it to start."""
    engine, approvals = engine_for(tmp_path)
    engine.cancel()
    config = config_for(
        tmp_path, cutout(tmp_path / "front.png"), cutout(tmp_path / "back.png", shift=8)
    )

    with pytest.raises(SshPodError, match="cancel"):
        engine.run(config, approve_existing_pod=True)
    assert approvals == []


def test_the_second_reconstruction_is_skipped_once_cancelled(tmp_path: Path) -> None:
    """Cancelling during the first photograph must not buy the second."""
    engine, approvals = engine_for(tmp_path, on_connect=lambda _a: engine.cancel())
    config = config_for(
        tmp_path, cutout(tmp_path / "front.png"), cutout(tmp_path / "back.png", shift=8)
    )

    with pytest.raises(SshPodError):
        engine.run(config, approve_existing_pod=True)
    assert len(approvals) == 1, "the second photograph was reconstructed after a cancel"
