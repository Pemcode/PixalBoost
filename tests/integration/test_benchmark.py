"""Integration test for the synthetic benchmark builder (F05).

Integration, not unit: it touches the filesystem and writes a real dataset. Still
CPU-only, offline, and under a second -- it builds into `tmp_path` at a small
resolution rather than reading whatever `poe bench-build` last produced, so it
cannot pass because of a stale artefact.

The assertion that matters is the round trip: re-rendering the stored mesh with
the stored pose and intrinsics must reproduce the stored mask exactly. Poses and
masks that disagree would silently poison every metric in F12.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from pixaboost.bench.build import BuildConfig, build_dataset
from pixaboost.bench.rig import AZIMUTH_COUNT, ELEVATIONS_DEG
from pixaboost.core.geometry import BlenderCamera, Sim3
from pixaboost.core.render import rasterise_silhouette
from pixaboost.observability import TelemetryEvent

VIEW_COUNT = AZIMUTH_COUNT * len(ELEVATIONS_DEG)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("bench")
    return build_dataset(root, BuildConfig(resolution=64, parts=("flange_ring",)))


def test_the_manifest_records_what_would_be_needed_to_rebuild(dataset: Path) -> None:
    manifest = json.loads((dataset / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest) >= {"git_sha", "pixaboost_version", "config", "parts", "created_utc"}
    assert manifest["config"]["resolution"] == 64
    assert manifest["parts"] == ["flange_ring"]


def test_every_part_has_a_mesh_cameras_images_and_masks(dataset: Path) -> None:
    part = dataset / "flange_ring"
    assert (part / "mesh.npz").is_file()
    assert (part / "cameras.json").is_file()
    assert len(sorted((part / "images").glob("*.png"))) == VIEW_COUNT
    assert len(sorted((part / "masks").glob("*.png"))) == VIEW_COUNT


def test_the_stored_mesh_is_the_watertight_reference(dataset: Path) -> None:
    mesh = np.load(dataset / "flange_ring" / "mesh.npz")
    vertices, faces = mesh["vertices"], mesh["faces"]
    assert vertices.shape[1] == 3
    assert faces.shape[1] == 3
    counts: dict[tuple[int, int], int] = {}
    for face in faces:
        for i in range(3):
            key = (min(face[i], face[(i + 1) % 3]), max(face[i], face[(i + 1) % 3]))
            counts[key] = counts.get(key, 0) + 1
    assert set(counts.values()) == {2}


def test_cameras_are_rigid_and_evenly_distributed(dataset: Path) -> None:
    cameras = json.loads((dataset / "flange_ring" / "cameras.json").read_text(encoding="utf-8"))
    assert len(cameras["views"]) == VIEW_COUNT
    assert cameras["intrinsics"]["resolution"] == 64
    for view in cameras["views"]:
        pose = Sim3.from_matrix(np.array(view["camera_to_world"], dtype=np.float64))
        assert pose.scale == pytest.approx(1.0)
        assert np.linalg.norm(pose.translation) == pytest.approx(cameras["intrinsics"]["distance"])


def test_masks_show_the_part_without_filling_the_frame(dataset: Path) -> None:
    for path in sorted((dataset / "flange_ring" / "masks").glob("*.png")):
        mask = np.array(Image.open(path))
        coverage = float((mask > 127).mean())
        assert 0.01 < coverage < 0.9, f"{path.name} covers {coverage:.2%} of the frame"


def test_rerendering_the_mesh_with_the_stored_pose_reproduces_the_stored_mask(
    dataset: Path,
) -> None:
    part = dataset / "flange_ring"
    mesh = np.load(part / "mesh.npz")
    cameras = json.loads((part / "cameras.json").read_text(encoding="utf-8"))
    camera = BlenderCamera(
        camera_angle_x=cameras["intrinsics"]["camera_angle_x"],
        resolution=cameras["intrinsics"]["resolution"],
    )
    for view in cameras["views"]:
        pose = Sim3.from_matrix(np.array(view["camera_to_world"], dtype=np.float64))
        expected = rasterise_silhouette(mesh["vertices"], mesh["faces"], pose, camera)
        stored = np.array(Image.open(part / "masks" / f"{view['name']}.png")) > 127
        np.testing.assert_array_equal(stored, expected, err_msg=f"mask drift on {view['name']}")


def test_images_are_lit_inside_the_mask_and_black_outside(dataset: Path) -> None:
    part = dataset / "flange_ring"
    for mask_path in sorted((part / "masks").glob("*.png")):
        image = np.array(Image.open(part / "images" / mask_path.name).convert("RGB"))
        mask = np.array(Image.open(mask_path)) > 127
        assert image.shape == (64, 64, 3)
        assert image[~mask].max() == 0, "background must be black, as Pixal3D preprocesses to"
        assert image[mask].max() > 0, "the part must actually be lit"


def test_building_the_whole_catalogue_yields_three_distinct_parts(tmp_path: Path) -> None:
    root = build_dataset(tmp_path, BuildConfig(resolution=32))
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["parts"] == ["flange_ring", "l_bracket", "stepped_shaft"]
    silhouettes = {
        name: np.array(Image.open(root / name / "masks" / "az000_el+00.png")) > 127
        for name in manifest["parts"]
    }
    for name, other in [("flange_ring", "l_bracket"), ("l_bracket", "stepped_shaft")]:
        assert not np.array_equal(silhouettes[name], silhouettes[other])


def test_an_unknown_part_is_rejected_before_anything_is_written(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown part"):
        build_dataset(tmp_path, BuildConfig(parts=("not_a_part",)))
    assert not list(tmp_path.iterdir())


def test_the_builder_emits_countable_progress_and_the_manifest_artifact(tmp_path: Path) -> None:
    events: list[TelemetryEvent] = []

    build_dataset(
        tmp_path,
        BuildConfig(resolution=16, parts=("l_bracket",)),
        on_event=events.append,
    )

    render_events = [event for event in events if event.stage == "rendu"]
    assert len(render_events) == VIEW_COUNT
    assert events[0].progress == 0.0
    assert [event.progress for event in render_events] == sorted(
        event.progress for event in render_events if event.progress is not None
    )
    assert events[-1].progress == 1.0
    assert events[-1].artifact == tmp_path / "manifest.json"
