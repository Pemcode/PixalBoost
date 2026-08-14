"""Two uncalibrated photographs into one aligned GLB (F15).

Synthetic throughout: the "reconstructions" are benchmark meshes written to
real GLB files, and the "photograph mask" is a render of the same mesh at a
rotation the trial is not told about. No GPU, no network, no Pixal3D.

That substitution is honest for what is under test -- the alignment, the
artefact and the manifest. It says nothing about Pixal3D's own output quality.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from pixaboost.backends.glb import GlbError, load_mesh, write_placed_meshes
from pixaboost.bench.shapes import flange_ring, l_bracket
from pixaboost.core.geometry import BlenderCamera, front_view_camera
from pixaboost.core.pose_search import crop_to_canonical_framing, object_rotation
from pixaboost.core.render import rasterise_silhouette
from pixaboost.trials.two_view import TwoViewConfig, run_two_view_trial

CAMERA = BlenderCamera(camera_angle_x=0.857556, resolution=96)


def write_glb(path: Path, mesh: tuple[np.ndarray, np.ndarray]) -> Path:
    vertices, faces = mesh
    return write_placed_meshes(path, {"part": (vertices, faces, np.eye(3))})


def mask_at(mesh: tuple[np.ndarray, np.ndarray], rotation: np.ndarray) -> np.ndarray:
    vertices, faces = mesh
    return rasterise_silhouette(vertices @ rotation.T, faces, front_view_camera(2.0), CAMERA)


def build(tmp_path: Path, mesh: tuple[np.ndarray, np.ndarray], back_rotation: np.ndarray):
    """Front and back GLBs plus the mask the trial must align against."""
    front = write_glb(tmp_path / "front.glb", mesh)
    vertices, faces = mesh
    back = write_glb(tmp_path / "back.glb", (vertices @ back_rotation.T, faces))
    target = mask_at(mesh, back_rotation)

    def reconstruct(photo: Path) -> Path:
        return front if photo.name == "front.jpg" else back

    def mask_of(photo: Path) -> np.ndarray:
        return target

    config = TwoViewConfig(
        front_image=tmp_path / "front.jpg",
        back_image=tmp_path / "back.jpg",
        runs_root=tmp_path / "runs",
        search_resolution=CAMERA.resolution,
        azimuth_steps=18,
        elevation_steps=3,
        refine_rounds=2,
    )
    return config, reconstruct, mask_of


# --------------------------------------------------------------------------
# the GLB adapter
# --------------------------------------------------------------------------


def test_a_mesh_survives_a_round_trip_through_a_glb(tmp_path: Path) -> None:
    vertices, faces = l_bracket()
    reloaded_vertices, reloaded_faces = load_mesh(write_glb(tmp_path / "a.glb", (vertices, faces)))
    assert reloaded_faces.shape == faces.shape
    assert np.allclose(np.sort(reloaded_vertices, axis=0), np.sort(vertices, axis=0), atol=1e-5)


def test_the_placement_rotation_is_baked_into_the_vertices(tmp_path: Path) -> None:
    """A viewer that ignores the scene graph must still see the alignment."""
    vertices, faces = l_bracket()
    rotation = object_rotation(np.deg2rad(90.0), 0.0, 0.0)
    path = write_placed_meshes(tmp_path / "r.glb", {"part": (vertices, faces, rotation)})

    reloaded, _ = load_mesh(path)

    assert np.allclose(
        np.sort(reloaded, axis=0), np.sort(vertices @ rotation.T, axis=0), atol=1e-5
    )


def test_an_unreadable_file_is_refused(tmp_path: Path) -> None:
    broken = tmp_path / "broken.glb"
    broken.write_bytes(b"not a glb at all")
    with pytest.raises(GlbError):
        load_mesh(broken)


def test_a_non_rotation_placement_is_refused(tmp_path: Path) -> None:
    vertices, faces = l_bracket()
    with pytest.raises(GlbError, match="3x3"):
        write_placed_meshes(tmp_path / "x.glb", {"part": (vertices, faces, np.eye(4))})


def test_an_empty_scene_is_refused(tmp_path: Path) -> None:
    with pytest.raises(GlbError, match="no meshes"):
        write_placed_meshes(tmp_path / "x.glb", {})


# --------------------------------------------------------------------------
# the alignment itself
# --------------------------------------------------------------------------


def test_two_views_of_an_asymmetric_part_are_brought_into_one_frame(tmp_path: Path) -> None:
    """The whole claim of F15, with no calibration anywhere in the inputs."""
    mesh = l_bracket()
    config, reconstruct, mask_of = build(tmp_path, mesh, object_rotation(np.pi, 0.0, 0.0))

    result = run_two_view_trial(config, reconstruct=reconstruct, mask_of=mask_of)

    assert result.pose_iou > 0.90, f"the back view was never located: {result.pose_iou:.3f}"
    assert result.agreement_iou > 0.80, "the two halves do not line up once aligned"
    assert result.is_trustworthy


def test_the_written_glb_contains_both_reconstructions(tmp_path: Path) -> None:
    mesh = l_bracket()
    config, reconstruct, mask_of = build(tmp_path, mesh, object_rotation(np.pi, 0.0, 0.0))

    result = run_two_view_trial(config, reconstruct=reconstruct, mask_of=mask_of)

    assert result.glb_path.is_file()
    vertices, faces = load_mesh(result.glb_path)
    front_vertices, front_faces = mesh
    assert len(faces) == 2 * len(front_faces), "both meshes must be present, unmerged"
    assert len(vertices) == 2 * len(front_vertices)


def test_nothing_is_fused_the_two_meshes_stay_separable(tmp_path: Path) -> None:
    """Constraint 11: juxtaposition only. Face count must be exactly additive."""
    mesh = flange_ring()
    config, reconstruct, mask_of = build(tmp_path, mesh, object_rotation(np.deg2rad(120.0), 0, 0))

    result = run_two_view_trial(config, reconstruct=reconstruct, mask_of=mask_of)

    _, faces = load_mesh(result.glb_path)
    assert len(faces) == 2 * len(mesh[1])
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert "none" in manifest["fusion"]


def test_an_axisymmetric_part_is_flagged_ambiguous_but_still_aligns(tmp_path: Path) -> None:
    mesh = flange_ring()
    config, reconstruct, mask_of = build(tmp_path, mesh, object_rotation(np.deg2rad(150.0), 0, 0))

    result = run_two_view_trial(config, reconstruct=reconstruct, mask_of=mask_of)

    assert result.is_ambiguous
    assert result.agreement_iou > 0.80, "a spin about the axis must not hurt the alignment"


def test_the_manifest_records_everything_needed_to_reproduce(tmp_path: Path) -> None:
    """Constraint 7: without a manifest a result does not exist."""
    mesh = l_bracket()
    config, reconstruct, mask_of = build(tmp_path, mesh, object_rotation(np.pi, 0.0, 0.0))

    result = run_two_view_trial(config, reconstruct=reconstruct, mask_of=mask_of)
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))

    assert manifest["kind"] == "two-view-alignment"
    assert len(manifest["search"]["front_to_back_rotation"]) == 3
    assert manifest["metrics"]["pose_iou"] == pytest.approx(result.pose_iou)
    assert manifest["metrics"]["agreement_iou"] == pytest.approx(result.agreement_iou)
    assert manifest["inputs"]["front_glb_sha256"] != manifest["inputs"]["back_glb_sha256"]


def test_a_topologically_impossible_mask_collapses_the_score(tmp_path: Path) -> None:
    """The number must collapse when the search has nothing to find.

    The mask is an annulus. A solid extruded L has no viewpoint whose
    silhouette contains a hole, so no rotation can score well.
    """
    mesh = l_bracket()
    config, reconstruct, _ = build(tmp_path, mesh, object_rotation(np.pi, 0.0, 0.0))

    def impossible_mask(photo: Path) -> np.ndarray:
        rows, cols = np.ogrid[:200, :200]
        radius = (rows - 100) ** 2 + (cols - 100) ** 2
        return (radius <= 80**2) & (radius > 45**2)

    result = run_two_view_trial(config, reconstruct=reconstruct, mask_of=impossible_mask)

    assert result.pose_iou < 0.75, f"scored {result.pose_iou:.3f} on an impossible target"
    assert not result.is_trustworthy


def test_a_bar_shaped_mask_does_NOT_collapse_the_score_and_that_is_a_real_limit(
    tmp_path: Path,
) -> None:
    """Measured limitation, pinned so nobody rediscovers it on real photographs.

    An L-bracket seen edge-on *is* a bar, so a bar-shaped mask still scores
    high. Silhouette matching cannot distinguish a genuine match from a
    degenerate view that happens to agree.

    The `opposite_faces` prior narrowed this -- 0.94 without it, 0.83 with --
    by putting most of the coincidental orientations out of reach. It did not
    remove it: `is_trustworthy` is still True here, on a mask that has nothing
    to do with the part. `pose_iou` must therefore never be read as proof.
    That is what `agreement_iou` and looking at the GLB are for.
    """
    mesh = l_bracket()
    config, reconstruct, _ = build(tmp_path, mesh, object_rotation(np.pi, 0.0, 0.0))

    def bar_mask(photo: Path) -> np.ndarray:
        mask = np.zeros((200, 200), dtype=bool)
        mask[20:40, 150:190] = True
        return mask

    result = run_two_view_trial(config, reconstruct=reconstruct, mask_of=bar_mask)

    assert result.pose_iou > 0.75, "if this ever drops, the limitation shrank -- update the doc"
    assert result.is_trustworthy, (
        "the honest, uncomfortable part: the guard does NOT catch this one"
    )


def test_the_photo_mask_goes_through_the_canonical_framing(tmp_path: Path) -> None:
    """A raw mask and a canonically framed one must not be treated alike."""
    mesh = l_bracket()
    raw = mask_at(mesh, np.eye(3))
    framed = crop_to_canonical_framing(raw, CAMERA.resolution)
    assert framed.shape == raw.shape
    assert not np.array_equal(framed, raw), "the framing step must actually change the mask"
