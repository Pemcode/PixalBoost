"""Translation between GLB files and `core/` mesh arrays (F15).

No business logic: read a GLB into `(vertices, faces)`, write meshes back out
with a placement transform each. Whether two meshes *should* be placed
together, and where, is decided in `trials/`.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import trimesh

FloatArray = np.ndarray
IntArray = np.ndarray


class GlbError(RuntimeError):
    """A GLB could not be read, or held nothing usable."""


def load_mesh(path: Path | str) -> tuple[FloatArray, IntArray]:
    """Read a GLB and return its concatenated triangle soup.

    A Pixal3D export is a single mesh, but a scene with several nodes is
    flattened rather than refused: the pose search only needs a silhouette, and
    a silhouette does not care how the geometry was grouped.
    """
    source = Path(path)
    try:
        loaded = trimesh.load(source, force="mesh", process=False)
    except Exception as error:
        raise GlbError(f"{source} could not be read as a GLB: {type(error).__name__}") from None
    if not isinstance(loaded, trimesh.Trimesh) or loaded.faces.size == 0:
        raise GlbError(f"{source} holds no triangles")
    return (
        np.asarray(loaded.vertices, dtype=np.float64),
        np.asarray(loaded.faces, dtype=np.int64),
    )


def write_placed_meshes(
    destination: Path | str,
    parts: dict[str, tuple[FloatArray, IntArray, FloatArray]],
) -> Path:
    """Write several meshes into one GLB, each under its own rotation.

    `parts` maps a node name to `(vertices, faces, rotation)`. The rotation is
    applied to the vertices rather than stored as a node transform, so the file
    reads identically in any viewer that ignores the scene graph.

    This is **juxtaposition, not fusion**: no vertex is merged, averaged,
    weighted or carved. Hard constraint 11 of CLAUDE.md stands until the F13
    gate rules.
    """
    target = Path(destination)
    if not parts:
        raise GlbError("refusing to write a GLB with no meshes")

    scene = trimesh.Scene()
    for name, (vertices, faces, rotation) in parts.items():
        points = np.asarray(vertices, dtype=np.float64)
        matrix = np.asarray(rotation, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise GlbError(f"{name}: rotation must be 3x3, got {matrix.shape}")
        scene.add_geometry(
            trimesh.Trimesh(vertices=points @ matrix.T, faces=np.asarray(faces), process=False),
            node_name=name,
            geom_name=name,
        )

    target.parent.mkdir(parents=True, exist_ok=True)
    exported: bytes = scene.export(file_type="glb")  # type: ignore[no-untyped-call]
    target.write_bytes(exported)
    return target
