"""Characterisation tests pinning the Pixal3D facts that F10 depends on.

These assert the *upstream source*, parsed with `ast` -- no torch, no GPU, no
model weights. Two jobs:

1. They are the verification command for F01: they encode, as executable
   assertions, the claims made in docs/pixal3d-internals.md.
2. They are a drift detector. Bumping the `vendor/pixal3d` submodule SHA will
   fail these the moment upstream changes any fact F10 is built on -- which
   forces a re-read instead of a silent behaviour change.

Per docs/testing.md these belong to the *contract test* regime, not to strict
TDD: they describe code we do not own.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
VENDOR = REPO_ROOT / "vendor" / "pixal3d"

PROJ_MIXIN = (
    VENDOR / "pixal3d" / "trainers" / "flow_matching" / "mixins" / "image_conditioned_proj.py"
)
PIPELINE = VENDOR / "pixal3d" / "pipelines" / "pixal3d_image_to_3d.py"
INFERENCE = VENDOR / "inference.py"


def _load(path: Path) -> ast.Module:
    if not path.is_file():
        pytest.skip(
            f"Vendored Pixal3D source missing: {path.relative_to(REPO_ROOT)}. "
            f"The submodule is not checked out. "
            f"Fix: `git submodule update --init --recursive`."
        )
    return ast.parse(path.read_text(encoding="utf-8"))


def _find_function(tree: ast.Module, qualname: str) -> ast.FunctionDef:
    """Find `func` or `Class.method` in a module AST."""
    parts = qualname.split(".")
    nodes: list[ast.stmt] = list(tree.body)
    for depth, part in enumerate(parts):
        want_func = depth == len(parts) - 1
        for node in nodes:
            if want_func and isinstance(node, ast.FunctionDef) and node.name == part:
                return node
            if not want_func and isinstance(node, ast.ClassDef) and node.name == part:
                nodes = list(node.body)
                break
        else:
            raise AssertionError(f"{qualname!r}: could not resolve {part!r}")
    raise AssertionError(f"{qualname!r} did not resolve to a function")


def _arg_names(func: ast.FunctionDef) -> list[str]:
    return [a.arg for a in func.args.posonlyargs + func.args.args + func.args.kwonlyargs]


# ---------------------------------------------------------------------------
# The blocker: back-projection is hard-locked to a single front-facing camera.
# ---------------------------------------------------------------------------


def test_projgrid_forward_still_accepts_a_transform_matrix_argument() -> None:
    """The per-view camera parameter exists in the signature -- F10 needs it."""
    forward = _find_function(_load(PROJ_MIXIN), "ProjGrid.forward")
    assert "transform_matrix" in _arg_names(forward)


def test_projgrid_forward_still_asserts_transform_matrix_is_none() -> None:
    """The parameter is plumbed but disabled by a hard assert.

    This single statement is why multi-view inference is unavailable upstream:
    every back-projection is forced onto the canonical front view. F10 must
    subclass ProjGrid to lift it. If this test fails, upstream may have enabled
    arbitrary cameras -- re-read the module before touching backends/pixal3d.py.
    """
    forward = _find_function(_load(PROJ_MIXIN), "ProjGrid.forward")
    asserts = [n for n in ast.walk(forward) if isinstance(n, ast.Assert)]
    blocking = [
        n
        for n in asserts
        if isinstance(n.test, ast.Compare)
        and isinstance(n.test.left, ast.Name)
        and n.test.left.id == "transform_matrix"
        and any(isinstance(op, ast.Is) for op in n.test.ops)
        and any(isinstance(c, ast.Constant) and c.value is None for c in n.test.comparators)
    ]
    assert blocking, "expected `assert transform_matrix is None` in ProjGrid.forward"


def test_projgrid_forward_discards_the_visibility_mask() -> None:
    """`valid_mask` is computed then never used.

    It is the free per-voxel visibility signal a masked multi-view average
    needs, and `sample_features` uses padding_mode='border', so voxels that
    fall outside the image silently receive border features instead of being
    excluded. F10 must recover this mask rather than recompute it.
    """
    forward = _find_function(_load(PROJ_MIXIN), "ProjGrid.forward")
    bound = sum(
        1
        for n in ast.walk(forward)
        if isinstance(n, ast.Name) and n.id == "valid_mask" and isinstance(n.ctx, ast.Store)
    )
    used = sum(
        1
        for n in ast.walk(forward)
        if isinstance(n, ast.Name) and n.id == "valid_mask" and isinstance(n.ctx, ast.Load)
    )
    assert bound >= 1, "expected valid_mask to be bound in ProjGrid.forward"
    assert used == 0, "valid_mask is now consumed upstream -- re-read before reimplementing it"


# ---------------------------------------------------------------------------
# The conditioning path threads the camera end to end, so F10 is surgical.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("module", "qualname"),
    [
        ("PROJ_MIXIN", "DinoV3ProjFeatureExtractor.forward"),
        ("PROJ_MIXIN", "ImageConditionedProjMixin.encode_image_proj"),
    ],
)
def test_conditioning_path_threads_transform_matrix(module: str, qualname: str) -> None:
    tree = _load({"PROJ_MIXIN": PROJ_MIXIN}[module])
    assert "transform_matrix" in _arg_names(_find_function(tree, qualname))


# ---------------------------------------------------------------------------
# The inference-side entry points are single-view by construction.
# ---------------------------------------------------------------------------


def test_pipeline_shape_conditioning_hardcodes_batch_size_one() -> None:
    """`B = 1` means a second image would break the grid reshape downstream."""
    func = _find_function(_load(PIPELINE), "Pixal3DImageTo3DPipeline.get_proj_cond_shape")
    literals = [
        n.value.value
        for n in ast.walk(func)
        if isinstance(n, ast.Assign)
        and isinstance(n.value, ast.Constant)
        and any(isinstance(t, ast.Name) and t.id == "B" for t in n.targets)
    ]
    assert literals == [1], f"expected a single `B = 1` assignment, found {literals}"


def test_inference_cli_takes_exactly_one_image() -> None:
    """No `--images`, no `--views`: the released CLI is single-view."""
    source = INFERENCE.read_text(encoding="utf-8") if INFERENCE.is_file() else _load(INFERENCE)
    assert isinstance(source, str)
    flags = {
        node.args[0].value
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr == "add_argument"
        and node.args
        and isinstance(node.args[0], ast.Constant)
        and isinstance(node.args[0].value, str)
    }
    assert "--image" in flags
    assert not {"--images", "--views", "--image_dir", "--multiview"} & flags
