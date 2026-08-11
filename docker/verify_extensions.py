"""Fail the image build, not a billed GPU-hour, when a dependency is missing.

Two classes of check, because the build runs on a machine with no NVIDIA driver:

- **Importable**: pure-Python packages. Actually imported, since a package can
  be installed yet broken.
- **Present**: compiled CUDA extensions. Only their module spec is looked up.
  Importing them here would dlopen `libcuda.so.1`, which ships with the driver
  and not the toolkit, so a real import would fail on a CPU runner even when
  the image is perfectly good.

`natten` is in the second list even though nothing in the vendored source
imports it: it is pulled in at runtime by the NAF upsampler that
`image_conditioned_proj.py:415` fetches through `torch.hub`. A missing natten
therefore surfaces mid-inference, on paid hardware, which is exactly what this
script exists to prevent.
"""

from __future__ import annotations

import importlib
import importlib.util  # `import importlib` alone does NOT expose `.util`
import os
import shutil
import sys

IMPORTABLE = ("torch", "trimesh", "transformers", "runpod", "utils3d", "moge")
PRESENT_ONLY = ("natten", "o_voxel", "flex_gemm", "cumesh", "nvdiffrast", "flash_attn_3")

#: What each attention backend actually imports, read off
#: pixal3d/modules/attention/full_attn.py:97-144. None means "always available".
#:
#: This mapping exists because the names differ from the wheel names in a way
#: that already cost one broken image: the flash_attn_3 wheel ships
#: `flash_attn_interface`, not `flash_attn`, so upstream's default backend
#: raises ModuleNotFoundError -- and it does so at inference time, after the
#: 26 GB weight download rather than at build time.
BACKEND_MODULE = {
    "xformers": "xformers.ops",
    "flash_attn": "flash_attn",
    "flash_attn_3": "flash_attn_interface",
    "flash_attn_4": "flash_attn.cute",
    "sdpa": None,
    "naive": None,
}


def main() -> int:
    failures: list[str] = []

    for name in IMPORTABLE:
        try:
            importlib.import_module(name)
        except Exception as error:  # noqa: BLE001 - we want the reason, whatever it is
            failures.append(f"{name}: import failed: {type(error).__name__}: {error}")
            print(f"FAIL  {name:<16} {type(error).__name__}: {error}", file=sys.stderr)
        else:
            print(f"ok    {name:<16} imported")

    for name in PRESENT_ONLY:
        try:
            found = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError) as error:
            found = False
            print(f"FAIL  {name:<16} spec lookup raised {error}", file=sys.stderr)
        if not found:
            failures.append(f"{name}: not installed")
            print(f"FAIL  {name:<16} not installed", file=sys.stderr)
        else:
            print(f"ok    {name:<16} present")

    # flex_gemm imports Triton, which JIT compiles a C helper at *run* time to
    # reach the CUDA driver. No compiler means "Failed to find C compiler", and
    # it only surfaces on the first real import -- i.e. on paid hardware, after
    # the weight download. Checking for the binary catches it during the build.
    compiler = os.environ.get("CC") or shutil.which("cc") or shutil.which("gcc")
    if compiler is None:
        failures.append("no C compiler: Triton cannot build its CUDA driver module")
        print(f"FAIL  {'C compiler':<16} not found (needed by Triton at runtime)", file=sys.stderr)
    else:
        print(f"ok    {'C compiler':<16} {compiler}")

    backend = os.environ.get("ATTN_BACKEND", "flash_attn")
    if backend not in BACKEND_MODULE:
        failures.append(f"ATTN_BACKEND={backend!r} is not one of {sorted(BACKEND_MODULE)}")
        print(f"FAIL  {'ATTN_BACKEND':<16} unknown value {backend!r}", file=sys.stderr)
    else:
        required = BACKEND_MODULE[backend]
        if required is None:
            print(f"ok    {'ATTN_BACKEND':<16} {backend} needs no extra module")
        elif importlib.util.find_spec(required.split(".")[0]) is None:
            failures.append(f"ATTN_BACKEND={backend} needs {required}, which is not installed")
            print(f"FAIL  {'ATTN_BACKEND':<16} {backend} needs {required}", file=sys.stderr)
        else:
            print(f"ok    {'ATTN_BACKEND':<16} {backend} -> {required}")

    if failures:
        print(f"\nMISSING MODULES: {failures}", file=sys.stderr)
        return 1

    print("\nall extensions present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
