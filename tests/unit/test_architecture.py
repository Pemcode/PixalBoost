"""Enforces the `core/` boundary (F04).

`core/` must stay pure CPU: no torch, no network, no GPU, no implicit global
randomness, and no dependency on `backends/`. That boundary is not aesthetic --
it is what keeps `poe check` runnable offline, on free CI, in under 60 s, and
what keeps a metric reproducible bit for bit. See core/ARCHITECTURE.md.

Detection is static (`ast`), not runtime: a lazy `import torch` inside a
function body would slip past an import-and-inspect check, but not past this.

The detector is itself tested against a known-bad module, because an
architecture test that passes vacuously is worse than no test at all.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
CORE = REPO_ROOT / "src" / "pixaboost" / "core"

#: Top-level packages `core/` may never reach for, and why.
FORBIDDEN_IMPORTS = {
    "torch": "GPU tensor runtime",
    "torchvision": "GPU tensor runtime",
    "jax": "accelerator runtime",
    "tensorflow": "accelerator runtime",
    "cupy": "CUDA runtime",
    "pycuda": "CUDA runtime",
    "transformers": "pulls torch and downloads weights",
    "huggingface_hub": "network",
    "requests": "network",
    "httpx": "network",
    "aiohttp": "network",
    "urllib": "network",
    "urllib3": "network",
    "http": "network",
    "socket": "network",
    "ftplib": "network",
    "smtplib": "network",
    "boto3": "network",
}

FORBIDDEN_INTERNAL = ("pixaboost.backends", "pixaboost.bench")

#: `np.random.default_rng(seed)` is fine; the legacy global-state helpers are not.
FORBIDDEN_RANDOM_ATTRS = {"seed", "rand", "randn", "random", "randint", "choice", "shuffle"}


def core_modules() -> list[Path]:
    return sorted(CORE.rglob("*.py"))


def imported_roots(tree: ast.AST) -> set[str]:
    roots: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            roots.update(alias.name.split(".")[0] for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            roots.add(node.module.split(".")[0])
    return roots


def imported_modules(tree: ast.AST) -> set[str]:
    modules: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.add(node.module)
    return modules


def global_random_calls(tree: ast.AST) -> set[str]:
    """Find `np.random.<attr>` / `numpy.random.<attr>` uses of the legacy global RNG."""
    found: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute) or node.attr not in FORBIDDEN_RANDOM_ATTRS:
            continue
        parent = node.value
        if (
            isinstance(parent, ast.Attribute)
            and parent.attr == "random"
            and isinstance(parent.value, ast.Name)
            and parent.value.id in {"np", "numpy"}
        ):
            found.add(f"{parent.value.id}.random.{node.attr}")
    return found


def test_the_core_package_is_actually_being_scanned() -> None:
    """Guards against a vacuous pass if the layout moves and the glob finds nothing."""
    modules = core_modules()
    assert modules, f"no modules found under {CORE}; this test would pass vacuously"
    assert {path.name for path in modules} >= {"geometry.py", "metrics.py", "render.py"}


@pytest.mark.parametrize("path", core_modules(), ids=lambda p: p.name)
def test_core_module_avoids_torch_gpu_and_network(path: Path) -> None:
    offenders = imported_roots(ast.parse(path.read_text(encoding="utf-8"))) & set(
        FORBIDDEN_IMPORTS
    )
    assert not offenders, (
        f"WHAT: {path.relative_to(REPO_ROOT)} imports {sorted(offenders)}.\n"
        f"WHY: core/ must stay pure CPU and offline -- "
        f"{', '.join(FORBIDDEN_IMPORTS[name] for name in sorted(offenders))}. "
        f"Allowing it breaks the sub-60s offline gate (core/ARCHITECTURE.md).\n"
        f"FIX: move the code needing this into src/pixaboost/backends/ and have it "
        f"return a core datatype, or drop the dependency."
    )


@pytest.mark.parametrize("path", core_modules(), ids=lambda p: p.name)
def test_core_module_does_not_depend_on_outer_layers(path: Path) -> None:
    modules = imported_modules(ast.parse(path.read_text(encoding="utf-8")))
    offenders = {m for m in modules if m.startswith(FORBIDDEN_INTERNAL)}
    assert not offenders, (
        f"WHAT: {path.relative_to(REPO_ROOT)} imports {sorted(offenders)}.\n"
        f"WHY: dependencies point inwards. core/ decides; backends/ and bench/ feed it. "
        f"An edge the other way makes core/ untestable without a GPU.\n"
        f"FIX: invert it -- pass the data in as an argument instead of importing the caller."
    )


@pytest.mark.parametrize("path", core_modules(), ids=lambda p: p.name)
def test_core_module_uses_no_implicit_global_randomness(path: Path) -> None:
    offenders = global_random_calls(ast.parse(path.read_text(encoding="utf-8")))
    assert not offenders, (
        f"WHAT: {path.relative_to(REPO_ROOT)} calls {sorted(offenders)}.\n"
        f"WHY: the legacy numpy RNG carries hidden global state, so a metric stops being "
        f"reproducible bit for bit and a benchmark result stops being evidence.\n"
        f"FIX: take an explicit `seed` argument and use np.random.default_rng(seed)."
    )


# ---------------------------------------------------------------------------
# The detectors must actually detect. Without this, the checks above could pass
# for the wrong reason and silently stop protecting the boundary.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "source",
    [
        "import torch",
        "import torch.nn as nn",
        "from torch import Tensor",
        "import requests",
        "def f():\n    import torch\n    return torch",  # lazy import inside a function
        "class C:\n    def m(self):\n        from huggingface_hub import hf_hub_download",
    ],
)
def test_forbidden_import_detector_catches_violations(source: str) -> None:
    assert imported_roots(ast.parse(source)) & set(FORBIDDEN_IMPORTS)


def test_layering_detector_catches_a_backend_import() -> None:
    tree = ast.parse("from pixaboost.backends.pixal3d import run")
    assert {m for m in imported_modules(tree) if m.startswith(FORBIDDEN_INTERNAL)}


@pytest.mark.parametrize(
    "source",
    ["np.random.seed(0)", "x = np.random.rand(3)", "numpy.random.choice(a)"],
)
def test_global_randomness_detector_catches_violations(source: str) -> None:
    assert global_random_calls(ast.parse(source))


def test_global_randomness_detector_allows_a_seeded_generator() -> None:
    assert not global_random_calls(ast.parse("rng = np.random.default_rng(seed)"))
