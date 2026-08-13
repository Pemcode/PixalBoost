"""Versioned source transferred to an existing Pod by :mod:`ssh_pod`.

The worker stays a standalone Python 3.10 script: it is copied to a unique
remote run directory and never mutates the pinned Pixal3D checkout.
"""

REMOTE_WORKER_SOURCE = r"""#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import signal
import struct
import subprocess
import sys
import tempfile
import traceback
from pathlib import Path
from typing import Any, Dict

PROTOCOL = "pixaboost.ssh.v1"
FRAME_PREFIX = "PIXABOOST_SSH_FRAME "
MIT_REMBG_MODEL = "ZhengPeng7/BiRefNet"
sequence = 0


class Cancelled(RuntimeError):
    pass


def emit(kind: str, run_id: str, **fields: Any) -> None:
    global sequence
    frame = {
        "protocol": PROTOCOL,
        "kind": kind,
        "run_id": run_id,
        "sequence": sequence,
        **fields,
    }
    sequence += 1
    print(FRAME_PREFIX + json.dumps(frame, separators=(",", ":"), sort_keys=True), flush=True)


def atomic_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=str(path.parent), suffix=".partial")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def decode_request(value: str) -> Dict[str, Any]:
    raw = base64.b64decode(value, validate=True)
    decoded = json.loads(raw.decode("utf-8"))
    if not isinstance(decoded, dict):
        raise ValueError("request must be a JSON object")
    if decoded.get("protocol") != PROTOCOL:
        raise ValueError("unsupported request protocol")
    return decoded


def validate_glb(path: Path) -> Dict[str, Any]:
    data = path.read_bytes()
    if len(data) < 12 or data[:4] != b"glTF":
        raise ValueError("inference output has invalid GLB magic")
    version, declared_length = struct.unpack("<II", data[4:12])
    if version != 2:
        raise ValueError(f"inference output uses GLB version {version}, expected 2")
    if declared_length != len(data):
        raise ValueError("inference output GLB length does not match its header")
    return {
        "path": str(path),
        "bytes": len(data),
        "sha256": hashlib.sha256(data).hexdigest(),
        "magic": "glTF",
        "version": 2,
    }


def install_cancel_handlers() -> None:
    def cancel(_signum: int, _frame: Any) -> None:
        raise Cancelled("remote inference interrupted")

    signal.signal(signal.SIGINT, cancel)
    signal.signal(signal.SIGTERM, cancel)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--request-b64", required=True)
    args = parser.parse_args()
    request = decode_request(args.request_b64)
    run_id = str(request["run_id"])
    manifest_path = Path(str(request["manifest_path"]))
    manifest: Dict[str, Any] = {
        "schema_version": 1,
        "protocol": PROTOCOL,
        "run_id": run_id,
        "status": "running",
        "expected_pixal3d_sha": request["expected_pixal3d_sha"],
        "git_sha": request["project_git_sha"],
        "worker_sha256": request["worker_sha256"],
        "input_sha256": request["input_sha256"],
        "params": request["params"],
        "poses": request["poses"],
    }
    atomic_json(manifest_path, manifest)
    install_cancel_handlers()
    revision = "unknown"

    try:
        actual_worker_sha = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
        if actual_worker_sha != request["worker_sha256"]:
            raise RuntimeError("remote worker SHA-256 mismatch")
        revision = subprocess.run(
            ["git", "-C", "/opt/pixal3d", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != request["expected_pixal3d_sha"]:
            raise RuntimeError("Pixal3D revision mismatch inside remote worker")
        checkout_status = subprocess.run(
            ["git", "-C", "/opt/pixal3d", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if checkout_status:
            raise RuntimeError("Pixal3D checkout is not clean inside remote worker")
        manifest["pixal3d_checkout_clean"] = True
        atomic_json(manifest_path, manifest)

        input_path = Path(str(request["input_path"]))
        if not input_path.is_file():
            raise RuntimeError("transferred input image is missing")
        actual_input_sha = hashlib.sha256(input_path.read_bytes()).hexdigest()
        if actual_input_sha != request["input_sha256"]:
            raise RuntimeError("transferred input image SHA-256 mismatch")

        output_path = Path(str(request["output_path"]))
        if output_path.exists():
            raise RuntimeError("refusing to overwrite an existing remote GLB")

        emit(
            "event",
            run_id,
            phase="inference",
            progress=0.2,
            message="Loading pinned Pixal3D pipeline",
        )
        os.environ.setdefault("ATTN_BACKEND", "sdpa")
        os.environ.setdefault("HF_HOME", "/workspace/huggingface")
        sys.path.insert(0, "/opt/pixal3d")

        from pixal3d.pipelines import rembg

        original_birefnet_init = rembg.BiRefNet.__init__

        def force_mit_birefnet(
            self: Any, model_name: str = MIT_REMBG_MODEL, **kwargs: Any
        ) -> None:
            del model_name, kwargs
            original_birefnet_init(self, MIT_REMBG_MODEL)

        rembg.BiRefNet.__init__ = force_mit_birefnet

        from inference import run_inference

        params = request["params"]
        emit(
            "event",
            run_id,
            phase="inference",
            progress=0.35,
            message="Running single-view reconstruction",
        )
        run_inference(
            image_path=str(input_path),
            output_path=str(output_path),
            seed=int(params["seed"]),
            manual_fov=float(params["fov"]),
            low_vram=bool(params["low_vram"]),
            resolution=int(params["resolution"]),
        )
        emit(
            "event",
            run_id,
            phase="finalisation",
            progress=0.9,
            message="Validating GLB integrity",
        )
        artifact = validate_glb(output_path)
        result = {
            "protocol": PROTOCOL,
            "kind": "result",
            "run_id": run_id,
            "status": "completed",
            "pixal3d_sha": revision,
            "seed": int(params["seed"]),
            "artifact": artifact,
        }
        manifest.update({"status": "completed", "artifact": artifact})
        atomic_json(manifest_path, manifest)
        public_result = {
            key: value
            for key, value in result.items()
            if key not in {"protocol", "kind", "run_id"}
        }
        emit("result", run_id, **public_result)
        return 0
    except Cancelled as error:
        result = {
            "status": "cancelled",
            "pixal3d_sha": revision,
            "error": str(error),
        }
        manifest.update(result)
        atomic_json(manifest_path, manifest)
        emit("result", run_id, **result)
        return 130
    except Exception as error:
        result = {
            "status": "failed",
            "pixal3d_sha": revision,
            "error": f"{type(error).__name__}: {error}",
        }
        manifest.update(result)
        manifest["traceback"] = traceback.format_exc()
        atomic_json(manifest_path, manifest)
        emit("result", run_id, **result)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
"""
