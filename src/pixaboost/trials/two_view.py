"""Two uncalibrated photographs into one aligned GLB (F15).

The capture records no poses, so the relation between the two reconstructions
has to be *derived*. It is, by rendering the first reconstruction and searching
for the orientation whose silhouette matches the second photograph's mask --
`core/pose_search.py`. The object supplies the frame the photographer did not.

What comes out is **two meshes in one coordinate system**, not one merged mesh.
Nothing is averaged, weighted, carved or scored per voxel: hard constraint 11
holds until F13 rules. The value of the artefact is that a human opens it and
sees immediately whether the two halves agree.

Orchestration only, per ADR-0012: the geometry lives in `core/`, the file I/O
in `backends/`, and the reconstruction itself is injected so this module never
decides to spend GPU money.
"""

from __future__ import annotations

import hashlib
import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from pixaboost.backends.glb import load_mesh, write_placed_meshes
from pixaboost.core.geometry import BlenderCamera, front_view_camera
from pixaboost.core.metrics import silhouette_iou
from pixaboost.core.pose_search import (
    crop_to_canonical_framing,
    search_object_pose,
)
from pixaboost.core.render import rasterise_silhouette

BoolArray = np.ndarray

#: Pixal3D's own conditioning FOV (`pixal3d_image_to_3d.py:196`).
DEFAULT_CAMERA_ANGLE_X = 0.857556
DEFAULT_DISTANCE = 2.0
#: Search resolution. Silhouette IoU is stable well below the export
#: resolution, and the search cost is quadratic in it.
DEFAULT_SEARCH_RESOLUTION = 128

#: Photo -> the GLB reconstructed from it. Injected: cache-first in the GUI,
#: a fake in the tests. This module never opens an SSH connection.
Reconstructor = Callable[[Path], Path]
#: Photo -> its boolean object mask, at the photo's own resolution.
MaskProvider = Callable[[Path], BoolArray]


@dataclass(frozen=True)
class TwoViewConfig:
    """Everything needed to reproduce one two-view alignment."""

    front_image: Path
    back_image: Path
    runs_root: Path
    camera_angle_x: float = DEFAULT_CAMERA_ANGLE_X
    distance: float = DEFAULT_DISTANCE
    search_resolution: int = DEFAULT_SEARCH_RESOLUTION
    azimuth_steps: int = 24
    elevation_steps: int = 7
    refine_rounds: int = 3


@dataclass(frozen=True)
class TwoViewResult:
    """The aligned artefact and the numbers that say whether to trust it."""

    glb_path: Path
    run_dir: Path
    manifest_path: Path
    #: Silhouette IoU between the rotated front reconstruction and the back mask.
    pose_iou: float
    #: Silhouette IoU between the two reconstructions once aligned, seen from
    #: the canonical camera. This is the "do the halves line up" number.
    agreement_iou: float
    is_ambiguous: bool

    @property
    def is_trustworthy(self) -> bool:
        """Advisory. A low pose IoU means the search never found the object."""
        return self.pose_iou >= 0.75 and self.agreement_iou >= 0.5


def run_two_view_trial(
    config: TwoViewConfig,
    *,
    reconstruct: Reconstructor,
    mask_of: MaskProvider,
    run_id: str | None = None,
) -> TwoViewResult:
    """Reconstruct both photographs, derive their relative pose, write the GLB."""
    started = time.monotonic()
    camera = BlenderCamera(
        camera_angle_x=config.camera_angle_x, resolution=config.search_resolution
    )

    front_glb = Path(reconstruct(Path(config.front_image)))
    back_glb = Path(reconstruct(Path(config.back_image)))
    front_vertices, front_faces = load_mesh(front_glb)
    back_vertices, back_faces = load_mesh(back_glb)

    target = crop_to_canonical_framing(mask_of(Path(config.back_image)), camera.resolution)
    search = search_object_pose(
        front_vertices,
        front_faces,
        target,
        camera=camera,
        distance=config.distance,
        azimuth_steps=config.azimuth_steps,
        elevation_steps=config.elevation_steps,
        refine_rounds=config.refine_rounds,
    )

    # `search.rotation` carries the front frame *into* the back frame, so the
    # back reconstruction comes back the other way to join the front.
    back_to_front = search.rotation.T
    agreement = _agreement(
        front_vertices,
        front_faces,
        back_vertices @ back_to_front.T,
        back_faces,
        camera,
        config.distance,
    )

    run_dir = Path(config.runs_root) / (run_id or _new_run_id())
    glb_path = write_placed_meshes(
        run_dir / "aligned.glb",
        {
            "front": (front_vertices, front_faces, np.eye(3)),
            "back": (back_vertices, back_faces, back_to_front),
        },
    )
    manifest_path = _write_manifest(
        run_dir,
        config=config,
        front_glb=front_glb,
        back_glb=back_glb,
        rotation=search.rotation,
        pose_iou=search.iou,
        agreement_iou=agreement,
        ambiguous=search.is_ambiguous,
        duration=time.monotonic() - started,
    )
    return TwoViewResult(
        glb_path=glb_path,
        run_dir=run_dir,
        manifest_path=manifest_path,
        pose_iou=search.iou,
        agreement_iou=agreement,
        is_ambiguous=search.is_ambiguous,
    )


def _agreement(
    front_vertices: np.ndarray,
    front_faces: np.ndarray,
    aligned_back_vertices: np.ndarray,
    back_faces: np.ndarray,
    camera: BlenderCamera,
    distance: float,
) -> float:
    """How much the two reconstructions cover each other once aligned.

    Deliberately a silhouette comparison rather than a Chamfer: it needs no
    surface sampling, it is the same metric the search optimises, and a human
    can reproduce it by looking at the GLB from the front.
    """
    view = front_view_camera(distance)
    return silhouette_iou(
        rasterise_silhouette(aligned_back_vertices, back_faces, view, camera),
        rasterise_silhouette(front_vertices, front_faces, view, camera),
    )


def _new_run_id() -> str:
    return f"twoview-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"


def _write_manifest(run_dir: Path, **fields: Any) -> Path:
    config: TwoViewConfig = fields["config"]
    payload = {
        "schema_version": 1,
        "kind": "two-view-alignment",
        "created_at": datetime.now(UTC).isoformat(),
        "duration_seconds": fields["duration"],
        "inputs": {
            "front_image": str(config.front_image),
            "back_image": str(config.back_image),
            "front_glb": str(fields["front_glb"]),
            "back_glb": str(fields["back_glb"]),
            "front_glb_sha256": _sha256(fields["front_glb"]),
            "back_glb_sha256": _sha256(fields["back_glb"]),
        },
        "camera": {
            "camera_angle_x": config.camera_angle_x,
            "distance": config.distance,
            "search_resolution": config.search_resolution,
        },
        "search": {
            "azimuth_steps": config.azimuth_steps,
            "elevation_steps": config.elevation_steps,
            "refine_rounds": config.refine_rounds,
            "front_to_back_rotation": np.asarray(fields["rotation"]).tolist(),
        },
        "metrics": {
            "pose_iou": fields["pose_iou"],
            "agreement_iou": fields["agreement_iou"],
            "is_ambiguous": fields["ambiguous"],
        },
        "fusion": "none -- meshes are juxtaposed in a common frame, constraint 11",
    }
    run_dir.mkdir(parents=True, exist_ok=True)
    path = run_dir / "manifest.json"
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
