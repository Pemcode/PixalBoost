"""Build the synthetic benchmark: reference meshes, ground-truth poses, images, masks.

Everything a run needs to be reproducible is written next to the data. Per the
hard constraint in CLAUDE.md, a result without a manifest is not evidence: the
git sha, the config and the part list all get recorded here.

Shading is deliberately plain -- a matte grey headlight over a black background.
That is both what Pixal3D preprocesses real photographs into (black background,
inference.py `preprocess_image`) and a fair stand-in for the matte metal parts
the real capture set contains.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image

from pixaboost import __version__
from pixaboost.bench.rig import capture_rig
from pixaboost.bench.shapes import CATALOGUE, FloatArray, IntArray
from pixaboost.core.geometry import BlenderCamera, Sim3
from pixaboost.core.render import rasterise_face_index

#: Pixal3D's own default conditioning FOV (pipelines/pixal3d_image_to_3d.py:196).
DEFAULT_CAMERA_ANGLE_X = 0.857556
DEFAULT_DISTANCE = 2.0

_AMBIENT = 0.18
_ALBEDO = np.array([0.72, 0.73, 0.75])  # neutral grey, faintly cool: matte metal


@dataclass(frozen=True)
class BuildConfig:
    resolution: int = 512
    camera_angle_x: float = DEFAULT_CAMERA_ANGLE_X
    distance: float = DEFAULT_DISTANCE
    parts: tuple[str, ...] | None = None

    def selected_parts(self) -> list[str]:
        if self.parts is None:
            return sorted(CATALOGUE)
        unknown = sorted(set(self.parts) - set(CATALOGUE))
        if unknown:
            raise ValueError(f"unknown part(s) {unknown}; known parts are {sorted(CATALOGUE)}")
        return sorted(self.parts)


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=Path(__file__).resolve().parents[3],
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return result.stdout.strip()


def _face_normals(vertices: FloatArray, faces: IntArray) -> FloatArray:
    corner = vertices[faces[:, 0]]
    normals = np.cross(vertices[faces[:, 1]] - corner, vertices[faces[:, 2]] - corner)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    return normals / np.where(lengths > 0.0, lengths, 1.0)


def shade(
    vertices: FloatArray,
    faces: IntArray,
    camera_to_world: Sim3,
    camera: BlenderCamera,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one view, returning ``(rgb_uint8, mask_bool)``.

    Lambertian with the light on the camera axis. Face normals are used without
    smoothing, on purpose: these are machined parts, and shading their facets
    flat keeps the sharp edges that carry the shape readable.
    """
    face_index = rasterise_face_index(vertices, faces, camera_to_world, camera)
    mask = face_index >= 0

    view_direction = camera_to_world.rotation @ np.array([0.0, 0.0, -1.0])
    normals = _face_normals(vertices, faces)
    # Two-sided: an inward-facing normal is a winding artefact, not a dark facet.
    lambert = np.abs(normals @ -view_direction)

    intensity = np.zeros(face_index.shape, dtype=np.float64)
    intensity[mask] = _AMBIENT + (1.0 - _AMBIENT) * lambert[face_index[mask]]
    rgb = np.clip(intensity[..., None] * _ALBEDO, 0.0, 1.0) * mask[..., None]
    return (rgb * 255.0 + 0.5).astype(np.uint8), mask


def build_dataset(output_root: Path, config: BuildConfig | None = None) -> Path:
    """Write the whole benchmark under `output_root` and return it."""
    config = config or BuildConfig()
    parts = config.selected_parts()  # validated before anything is written
    camera = BlenderCamera(camera_angle_x=config.camera_angle_x, resolution=config.resolution)
    rig = capture_rig(config.distance)

    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    for name in parts:
        vertices, faces = CATALOGUE[name]()
        part_dir = output_root / name
        (part_dir / "images").mkdir(parents=True, exist_ok=True)
        (part_dir / "masks").mkdir(parents=True, exist_ok=True)

        np.savez_compressed(part_dir / "mesh.npz", vertices=vertices, faces=faces)

        views = []
        for view in rig:
            rgb, mask = shade(vertices, faces, view.camera_to_world, camera)
            Image.fromarray(rgb, mode="RGB").save(part_dir / "images" / f"{view.name}.png")
            Image.fromarray((mask * 255).astype(np.uint8), mode="L").save(
                part_dir / "masks" / f"{view.name}.png"
            )
            views.append(
                {
                    "name": view.name,
                    "azimuth_deg": view.azimuth_deg,
                    "elevation_deg": view.elevation_deg,
                    "camera_to_world": view.camera_to_world.as_matrix().tolist(),
                }
            )

        (part_dir / "cameras.json").write_text(
            json.dumps(
                {
                    "intrinsics": {
                        "camera_angle_x": config.camera_angle_x,
                        "resolution": config.resolution,
                        "distance": config.distance,
                        "focal_px": camera.focal_px,
                        "convention": "blender: camera looks down -Z, 32 mm sensor",
                    },
                    "views": views,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "git_sha": _git_sha(),
                "pixaboost_version": __version__,
                "created_utc": datetime.now(UTC).isoformat(),
                "config": asdict(config),
                "parts": parts,
                "note": "Rig mirrors the real capture protocol: 6 azimuths x 3 elevations.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return output_root


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data/bench"))
    parser.add_argument("--resolution", type=int, default=BuildConfig.resolution)
    parser.add_argument("--part", action="append", dest="parts", choices=sorted(CATALOGUE))
    args = parser.parse_args(argv)

    root = build_dataset(
        args.output,
        BuildConfig(
            resolution=args.resolution,
            parts=tuple(args.parts) if args.parts else None,
        ),
    )
    print(f"benchmark written to {root.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
