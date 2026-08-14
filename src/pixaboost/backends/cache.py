"""Content-addressed cache for GPU artefacts.

This module is what makes a GPU-less development loop possible. Once a
generation exists on disk, every downstream experiment reads it instead of
paying for it again -- and with no local GPU, that is the difference between
iterating in seconds and iterating in billed minutes.

Two invariants carry the weight:

- **The key covers the model revision.** Bumping the `vendor/pixal3d` submodule
  must invalidate every artefact. Otherwise we would compare outputs from two
  different models and call the difference a result.
- **`meta.json` is written last and is the only completion marker.** A GLB on
  disk without it is a crashed transfer, not a cache hit. A truncated file
  reported as present would poison every metric downstream while looking fine.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import socket
import tempfile
import time
import uuid
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any

GLB_NAME = "output.glb"
META_NAME = "meta.json"
_KEY_LENGTH = 32
_SHA256 = re.compile(r"^[0-9a-f]{64}$")
_RESERVATION_OWNER_NAME = "owner.json"
_DEFAULT_RESERVATION_TIMEOUT_SECONDS = 2 * 60 * 60.0
_DEFAULT_RESERVATION_STALE_AFTER_SECONDS = 6 * 60 * 60.0
_DEFAULT_RESERVATION_POLL_INTERVAL_SECONDS = 0.1


class CacheCorruptionError(RuntimeError):
    """A completed cache entry exists but no longer matches its integrity metadata."""


class CacheReservationTimeoutError(TimeoutError):
    """Another process kept ownership of a cache key beyond the bounded wait."""


def cache_key(*, image: bytes, params: Mapping[str, Any], model_revision: str) -> str:
    """Derive a stable, filesystem-safe key for one generation request.

    Parameters are serialised with sorted keys so that argument order cannot
    produce two keys for the same request.
    """
    digest = hashlib.sha256()
    digest.update(model_revision.encode("utf-8"))
    digest.update(b"\x00")
    digest.update(json.dumps(dict(params), sort_keys=True, default=str).encode("utf-8"))
    digest.update(b"\x00")
    digest.update(image)
    return digest.hexdigest()[:_KEY_LENGTH]


@dataclass(frozen=True)
class CachedArtifact:
    """A completed cache entry."""

    key: str
    glb_path: Path
    metadata: dict[str, Any]


class ArtifactCache:
    """A directory of completed generations, one subdirectory per key."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)

    def directory_for(self, key: str) -> Path:
        return self.root / key

    def has(self, key: str) -> bool:
        return self.load(key) is not None

    def reserve(
        self,
        key: str,
        *,
        timeout_seconds: float = _DEFAULT_RESERVATION_TIMEOUT_SECONDS,
        stale_after_seconds: float = _DEFAULT_RESERVATION_STALE_AFTER_SECONDS,
        poll_interval_seconds: float = _DEFAULT_RESERVATION_POLL_INTERVAL_SECONDS,
    ) -> ArtifactReservation:
        """Reserve one content key across processes before starting billed work.

        Directory creation is the portable atomic primitive shared by Windows
        and Linux. The owner record lets a later process reclaim a reservation
        abandoned by a dead worker, while a live local owner is never stolen.
        """
        return ArtifactReservation(
            cache=self,
            key=key,
            timeout_seconds=timeout_seconds,
            stale_after_seconds=stale_after_seconds,
            poll_interval_seconds=poll_interval_seconds,
        )

    def load(self, key: str) -> CachedArtifact | None:
        entry = self.directory_for(key)
        meta_path = entry / META_NAME
        glb_path = entry / GLB_NAME
        if not meta_path.is_file():
            return None
        if not glb_path.is_file():
            raise CacheCorruptionError(
                f"cache entry {key} has a completion marker but its GLB is missing"
            )
        try:
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as error:
            raise CacheCorruptionError(
                f"cache entry {key} metadata is unreadable: {error}"
            ) from None
        if not isinstance(metadata, dict):
            raise CacheCorruptionError(f"cache entry {key} metadata must be a JSON object")
        expected_size = metadata.get("glb_bytes")
        expected_sha = metadata.get("glb_sha256")
        if (
            metadata.get("key") != key
            or not isinstance(expected_size, int)
            or isinstance(expected_size, bool)
            or expected_size < 1
            or not isinstance(expected_sha, str)
            or not _SHA256.fullmatch(expected_sha)
        ):
            raise CacheCorruptionError(f"cache entry {key} metadata is incomplete or inconsistent")
        actual_size, actual_sha = self._file_integrity(glb_path)
        if actual_size != expected_size:
            raise CacheCorruptionError(
                f"cache entry {key} size mismatch: expected {expected_size}, got {actual_size}"
            )
        if actual_sha != expected_sha:
            raise CacheCorruptionError(f"cache entry {key} SHA-256 mismatch")
        return CachedArtifact(
            key=key,
            glb_path=glb_path,
            metadata=metadata,
        )

    def store(
        self,
        key: str,
        *,
        glb: bytes,
        metadata: Mapping[str, Any],
        overwrite: bool = False,
    ) -> CachedArtifact:
        """Write an artefact, leaving an existing one untouched unless asked.

        Refusing to overwrite by default is the enforcement point for hard
        constraint 9: never regenerate an artefact that already exists.
        """
        if not glb:
            raise ValueError("refusing to cache an empty GLB payload")

        existing = self.load(key)
        if existing is not None and not overwrite:
            return existing

        entry = self.directory_for(key)
        entry.mkdir(parents=True, exist_ok=True)
        self._atomic_write(entry / GLB_NAME, glb)
        self._atomic_write(
            entry / META_NAME,
            json.dumps(
                {
                    **dict(metadata),
                    "key": key,
                    "glb_bytes": len(glb),
                    "glb_sha256": hashlib.sha256(glb).hexdigest(),
                },
                indent=2,
            ).encode("utf-8"),
        )
        loaded = self.load(key)
        assert loaded is not None  # just written
        return loaded

    @staticmethod
    def _file_integrity(path: Path) -> tuple[int, str]:
        digest = hashlib.sha256()
        size = 0
        try:
            with path.open("rb") as stream:
                while chunk := stream.read(1024 * 1024):
                    size += len(chunk)
                    digest.update(chunk)
        except OSError as error:
            raise CacheCorruptionError(f"cache artifact {path} is unreadable: {error}") from None
        return size, digest.hexdigest()

    @staticmethod
    def _atomic_write(destination: Path, payload: bytes) -> None:
        """Write via a sibling temp file and rename, so readers never see a partial file."""
        handle, temporary = tempfile.mkstemp(dir=destination.parent, suffix=".partial")
        try:
            with os.fdopen(handle, "wb") as stream:
                stream.write(payload)
                stream.flush()
                os.fsync(stream.fileno())
            os.replace(temporary, destination)
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise


@dataclass
class ArtifactReservation:
    """Exclusive, interprocess ownership of one cache key."""

    cache: ArtifactCache
    key: str
    timeout_seconds: float
    stale_after_seconds: float
    poll_interval_seconds: float
    _token: str = field(default_factory=lambda: uuid.uuid4().hex, init=False)
    _acquired: bool = field(default=False, init=False)

    def __post_init__(self) -> None:
        if self.timeout_seconds < 0.0:
            raise ValueError("reservation timeout must be non-negative")
        if self.stale_after_seconds < 0.0:
            raise ValueError("reservation stale age must be non-negative")
        if self.poll_interval_seconds <= 0.0:
            raise ValueError("reservation poll interval must be positive")

    @property
    def _path(self) -> Path:
        safe_key = hashlib.sha256(self.key.encode("utf-8")).hexdigest()
        return self.cache.root / f".{safe_key}.reservation"

    @property
    def _owner_path(self) -> Path:
        return self._path / _RESERVATION_OWNER_NAME

    def __enter__(self) -> ArtifactReservation:
        if self._acquired:
            raise RuntimeError("cache reservation is already acquired")
        self.cache.root.mkdir(parents=True, exist_ok=True)
        deadline = time.monotonic() + self.timeout_seconds
        while True:
            try:
                self._path.mkdir()
            except FileExistsError:
                self._reclaim_if_stale()
            else:
                try:
                    self._write_owner()
                except BaseException:
                    shutil.rmtree(self._path, ignore_errors=True)
                    raise
                self._acquired = True
                return self

            remaining = deadline - time.monotonic()
            if remaining <= 0.0:
                raise CacheReservationTimeoutError(
                    f"timed out waiting for cache reservation {self.key}"
                )
            time.sleep(min(self.poll_interval_seconds, remaining))

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        self.release()

    def release(self) -> None:
        """Release only the reservation carrying this instance's owner token."""
        if not self._acquired:
            return
        self._acquired = False
        owner = self._read_owner()
        if owner is None or owner.get("token") != self._token:
            return
        quarantine = self._path.with_name(f"{self._path.name}.release-{self._token}")
        try:
            self._path.rename(quarantine)
        except OSError:
            return
        shutil.rmtree(quarantine, ignore_errors=True)

    def _write_owner(self) -> None:
        payload = {
            "schema_version": 1,
            "key": self.key,
            "token": self._token,
            "pid": os.getpid(),
            "hostname": socket.gethostname(),
            "created_at_epoch": time.time(),
        }
        self.cache._atomic_write(
            self._owner_path,
            json.dumps(payload, sort_keys=True).encode("utf-8"),
        )

    def _read_owner(self) -> dict[str, Any] | None:
        try:
            payload = json.loads(self._owner_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError):
            return None
        return payload if isinstance(payload, dict) else None

    def _reclaim_if_stale(self) -> bool:
        owner = self._read_owner()
        if owner is not None and owner.get("hostname") == socket.gethostname():
            pid = owner.get("pid")
            if isinstance(pid, int) and not isinstance(pid, bool):
                if _pid_is_alive(pid):
                    return False
                return self._quarantine_stale_reservation()

        # Missing, malformed or foreign-host ownership is reclaimed only after
        # a conservative age. This avoids stealing a directory in the tiny
        # interval between its atomic creation and owner-record write.
        try:
            age = max(0.0, time.time() - self._path.stat().st_mtime)
        except FileNotFoundError:
            return True
        if age < self.stale_after_seconds:
            return False

        return self._quarantine_stale_reservation()

    def _quarantine_stale_reservation(self) -> bool:
        quarantine = self._path.with_name(
            f"{self._path.name}.stale-{os.getpid()}-{uuid.uuid4().hex}"
        )
        try:
            self._path.rename(quarantine)
        except (FileNotFoundError, FileExistsError, PermissionError):
            return False
        shutil.rmtree(quarantine, ignore_errors=True)
        return True


def _pid_is_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if pid == os.getpid():
        return True
    if os.name == "nt":
        return _windows_pid_is_alive(pid)
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def _windows_pid_is_alive(pid: int) -> bool:
    """Query a process handle without using ``os.kill(pid, 0)`` on Windows.

    `ctypes.WinDLL` and `ctypes.get_last_error` only exist on Windows, so they
    are reached through `getattr`: a direct attribute access type-checks here
    and fails on the Linux CI runner, which is precisely the kind of
    platform-dependent gate the project cannot afford. `poe typecheck` now runs
    both platforms for the same reason.
    """
    import ctypes
    from ctypes import wintypes

    process_query_limited_information = 0x1000
    still_active = 259
    load_library = getattr(ctypes, "WinDLL", None)
    if load_library is None:  # pragma: no cover - unreachable on Windows
        raise RuntimeError("_windows_pid_is_alive called off Windows")
    kernel32 = load_library("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [wintypes.DWORD, wintypes.BOOL, wintypes.DWORD]
    kernel32.OpenProcess.restype = wintypes.HANDLE
    kernel32.GetExitCodeProcess.argtypes = [wintypes.HANDLE, ctypes.POINTER(wintypes.DWORD)]
    kernel32.GetExitCodeProcess.restype = wintypes.BOOL
    kernel32.CloseHandle.argtypes = [wintypes.HANDLE]
    kernel32.CloseHandle.restype = wintypes.BOOL

    handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
    if not handle:
        # Access denied (5) still means the process exists. The indirection is
        # deliberate and ruff's B009 is wrong here: `ctypes.get_last_error`
        # exists only on Windows in typeshed, so a direct access type-checks on
        # this machine and fails the Linux CI run.
        last_error = getattr(ctypes, "get_last_error")  # noqa: B009
        return bool(last_error() == 5)
    try:
        exit_code = wintypes.DWORD()
        if not kernel32.GetExitCodeProcess(handle, ctypes.byref(exit_code)):
            return True
        return bool(exit_code.value == still_active)
    finally:
        kernel32.CloseHandle(handle)
