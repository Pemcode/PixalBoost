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
import struct
import sys
from pathlib import Path

IMPORTABLE = ("torch", "trimesh", "transformers", "runpod", "utils3d", "moge")
PRESENT_ONLY = ("natten", "o_voxel", "flex_gemm", "cumesh", "nvdiffrast", "flash_attn_3")

#: Compute capabilities the image must carry precompiled kernels for.
#: 8.9 = Ada (RTX 4090/L40S, what the batch runs use), 9.0 = Hopper (H100).
#:
#: This exists because a wheel can install cleanly, import cleanly, and still
#: die at the first kernel launch with "no kernel image is available for
#: execution on the device" -- which happens mid-inference, on paid hardware.
#: The upstream-pinned natten wheel carried sm_90 only and did exactly that.
REQUIRED_CUDA_ARCHS = {89, 90}

#: flex_gemm ships sm_80 cubins only but compiles its hot kernels through
#: Triton at runtime, so it is exempt from the architecture check.
ARCH_CHECKED = ("natten",)

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


def cuda_archs(library: Path) -> set[int]:
    """Compute capabilities of the CUDA cubins embedded in a shared object.

    CUDA cubins are ELF images (e_machine 190) stitched into the library's
    fatbinary. The architecture sits in the low byte of e_flags, so it can be
    read without nvcc or a GPU present.
    """
    blob = library.read_bytes()
    archs: set[int] = set()
    position = 0
    while True:
        position = blob.find(b"\x7fELF", position)
        if position == -1 or position + 0x34 > len(blob):
            return archs
        try:
            machine = struct.unpack_from("<H", blob, position + 0x12)[0]
            if machine == 190 and blob[position + 4] == 2:  # EM_CUDA, ELF64
                archs.add(struct.unpack_from("<I", blob, position + 0x30)[0] & 0xFF)
        except struct.error:
            pass
        position += 4


def check_architectures(module: str) -> str | None:
    """Return a failure message when `module` lacks kernels for a target GPU."""
    spec = importlib.util.find_spec(module)
    if spec is None or not spec.origin:
        return f"{module}: not installed, cannot check architectures"

    libraries = sorted(Path(spec.origin).parent.glob("*.so"))
    if not libraries:
        print(f"ok    {module:<16} pure python, no kernels to check")
        return None

    present: set[int] = set()
    for library in libraries:
        present |= cuda_archs(library)

    missing = REQUIRED_CUDA_ARCHS - present
    listed = ", ".join(f"sm_{a}" for a in sorted(present)) or "none"
    if missing:
        print(f"FAIL  {module:<16} has {listed}", file=sys.stderr)
        return (
            f"{module}: missing kernels for "
            f"{', '.join(f'sm_{a}' for a in sorted(missing))} (has {listed})"
        )
    print(f"ok    {module:<16} kernels for {listed}")
    return None


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

    for name in ARCH_CHECKED:
        problem = check_architectures(name)
        if problem:
            failures.append(problem)

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
