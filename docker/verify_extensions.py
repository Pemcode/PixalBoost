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
import sys

IMPORTABLE = ("torch", "trimesh", "transformers", "runpod", "utils3d", "moge")
PRESENT_ONLY = ("natten", "o_voxel", "flex_gemm", "cumesh", "nvdiffrast", "flash_attn_3")


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

    if failures:
        print(f"\nMISSING MODULES: {failures}", file=sys.stderr)
        return 1

    print("\nall extensions present")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
