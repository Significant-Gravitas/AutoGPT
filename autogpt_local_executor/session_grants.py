"""Tamper-evident grants binding one chat session to one canonical root."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import hmac
import json
import os
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from .directory_browser import _is_reparse_point


class RootGrantError(Exception):
    pass


@dataclass(frozen=True)
class RootBinding:
    session_id: str
    allowed_root: Path
    fingerprint: str
    revision: int
    root_grant: str


class RootGrantSigner:
    def __init__(
        self,
        key: bytes,
        machine_id: str,
        *,
        now: Callable[[], float] = time.time,
    ) -> None:
        if len(key) < 16:
            raise ValueError("root grant key must be at least 16 bytes")
        self._key = hmac.new(key, b"autogpt-root-grant-v1", hashlib.sha256).digest()
        self.machine_id = machine_id
        self._now = now

    def issue(self, session_id: str, allowed_root: Path, revision: int) -> RootBinding:
        if revision < 1:
            raise ValueError("revision must be positive")
        root = canonical_directory(allowed_root)
        fingerprint = directory_fingerprint(root)
        claims = {
            "allowed_root": str(root),
            "birth": directory_birth(root),
            "fingerprint": fingerprint,
            "issued_at": self._now(),
            "machine_id": self.machine_id,
            "revision": revision,
            "session_id": session_id,
        }
        encoded = _b64encode(
            json.dumps(claims, sort_keys=True, separators=(",", ":")).encode("utf-8")
        )
        signature = _b64encode(
            hmac.new(self._key, encoded.encode("ascii"), hashlib.sha256).digest()
        )
        grant = f"v1.{encoded}.{signature}"
        return RootBinding(
            session_id=session_id,
            allowed_root=root,
            fingerprint=fingerprint,
            revision=revision,
            root_grant=grant,
        )

    def verify(self, root_grant: str, expected_session_id: str) -> RootBinding:
        try:
            version, encoded, signature = root_grant.split(".", 2)
            if version != "v1":
                raise RootGrantError("Unsupported root grant version")
            expected = _b64encode(
                hmac.new(self._key, encoded.encode("ascii"), hashlib.sha256).digest()
            )
            if not hmac.compare_digest(signature, expected):
                raise RootGrantError("Root grant signature is invalid")
            claims = json.loads(_b64decode(encoded))
            session_id = claims["session_id"]
            machine_id = claims["machine_id"]
            revision = claims["revision"]
            allowed_root = claims["allowed_root"]
            fingerprint = claims["fingerprint"]
            if not isinstance(session_id, str) or session_id != expected_session_id:
                raise RootGrantError("Root grant belongs to another session")
            if not isinstance(machine_id, str) or machine_id != self.machine_id:
                raise RootGrantError("Root grant belongs to another machine")
            if not isinstance(revision, int) or revision < 1:
                raise RootGrantError("Root grant revision is invalid")
            if not isinstance(allowed_root, str) or not isinstance(fingerprint, str):
                raise RootGrantError("Root grant claims are invalid")
            root = canonical_directory(Path(allowed_root))
            # A grant issued while every system's fingerprint held st_dev
            # restores while that number holds.
            if fingerprint not in (directory_fingerprint(root), _fingerprint(root, device=True)):
                raise RootGrantError("The selected directory changed after the grant was issued")
            # A filesystem that reuses a freed inode (ext4, tmpfs) gives a
            # recreated directory the old identity; its creation time still
            # differs. None where the platform keeps no creation time.
            birth = claims["birth"]
            if birth is not None and birth != directory_birth(root):
                raise RootGrantError("The selected directory changed after the grant was issued")
        except RootGrantError:
            raise
        except (KeyError, TypeError, ValueError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RootGrantError("Root grant is malformed") from exc
        return RootBinding(
            session_id=session_id,
            allowed_root=root,
            fingerprint=fingerprint,
            revision=revision,
            root_grant=root_grant,
        )


def canonical_directory(path: Path) -> Path:
    try:
        if path.is_symlink() or _is_reparse_point(path):
            raise RootGrantError("Selected root cannot be a symbolic link")
        resolved = Path(os.path.realpath(path, strict=True))
        if not resolved.is_dir() or not os.access(resolved, os.R_OK | os.W_OK | os.X_OK):
            raise RootGrantError("Selected root is unavailable")
        return resolved
    except RootGrantError:
        raise
    except (OSError, RuntimeError) as exc:
        raise RootGrantError("Selected root is unavailable") from exc


def directory_fingerprint(path: Path) -> str:
    """What tells the selected folder from one put in its place: its inode
    number and canonical path, and on Windows its volume's serial number,
    which is st_dev there and kept from when the volume was formatted.

    A POSIX st_dev is the number the system gives a device when it is
    attached or mounted: a reboot can number a btrfs subvolume or an LVM
    volume afresh, a share or a FUSE folder has a new one each time it is
    mounted, and a removable disk's depends on what else is plugged in. A
    grant that held it stopped restoring once it changed. A folder put in
    place of the selected one has another inode number or, on a filesystem
    that reuses a freed one, another creation time (directory_birth, which
    the grant checks too).
    """
    return _fingerprint(canonical_directory(path), device=sys.platform == "win32")


def _fingerprint(root: Path, *, device: bool) -> str:
    try:
        stat = root.stat(follow_symlinks=False)
    except OSError as exc:
        raise RootGrantError("Selected root is unavailable") from exc
    identity: dict[str, int | str] = {
        "inode": stat.st_ino,
        "path": os.path.normcase(str(root)),
    }
    if device:
        identity["device"] = stat.st_dev
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def directory_birth(path: Path) -> str | None:
    """The directory's creation time, or None where the platform keeps none.

    macOS, BSD and Windows (Python 3.12+) report it through ``stat``; Linux
    only through ``statx(STATX_BTIME)`` (glibc 2.28+), which ext4, btrfs and
    xfs answer.
    """
    try:
        stat = path.stat(follow_symlinks=False)
    except OSError as exc:
        raise RootGrantError("Selected root is unavailable") from exc
    birth_ns = getattr(stat, "st_birthtime_ns", None)
    if isinstance(birth_ns, int):
        return f"ns:{birth_ns}"
    birth = getattr(stat, "st_birthtime", None)
    if isinstance(birth, float):
        return f"s:{birth!r}"
    if sys.platform.startswith("linux"):
        return _linux_birth(path)
    return None


class _StatxTimestamp(ctypes.Structure):
    _fields_ = [
        ("tv_sec", ctypes.c_int64),
        ("tv_nsec", ctypes.c_uint32),
        ("_reserved", ctypes.c_int32),
    ]


class _Statx(ctypes.Structure):
    # struct statx from <linux/stat.h>: 256 bytes, the tail reserved.
    _fields_ = [
        ("stx_mask", ctypes.c_uint32),
        ("stx_blksize", ctypes.c_uint32),
        ("stx_attributes", ctypes.c_uint64),
        ("stx_nlink", ctypes.c_uint32),
        ("stx_uid", ctypes.c_uint32),
        ("stx_gid", ctypes.c_uint32),
        ("stx_mode", ctypes.c_uint16),
        ("_spare0", ctypes.c_uint16),
        ("stx_ino", ctypes.c_uint64),
        ("stx_size", ctypes.c_uint64),
        ("stx_blocks", ctypes.c_uint64),
        ("stx_attributes_mask", ctypes.c_uint64),
        ("stx_atime", _StatxTimestamp),
        ("stx_btime", _StatxTimestamp),
        ("stx_ctime", _StatxTimestamp),
        ("stx_mtime", _StatxTimestamp),
        ("stx_rdev_major", ctypes.c_uint32),
        ("stx_rdev_minor", ctypes.c_uint32),
        ("stx_dev_major", ctypes.c_uint32),
        ("stx_dev_minor", ctypes.c_uint32),
        ("_spare2", ctypes.c_uint64 * 14),
    ]


_AT_FDCWD = -100
_AT_SYMLINK_NOFOLLOW = 0x100
_STATX_BTIME = 0x800


def _linux_birth(path: Path) -> str | None:
    try:
        statx = ctypes.CDLL(None, use_errno=True).statx
    except (AttributeError, OSError):
        return None
    statx.argtypes = [
        ctypes.c_int,
        ctypes.c_char_p,
        ctypes.c_int,
        ctypes.c_uint,
        ctypes.POINTER(_Statx),
    ]
    statx.restype = ctypes.c_int
    result = _Statx()
    if statx(
        _AT_FDCWD,
        os.fsencode(path),
        _AT_SYMLINK_NOFOLLOW,
        _STATX_BTIME,
        ctypes.byref(result),
    ):
        return None
    if not result.stx_mask & _STATX_BTIME:
        return None
    return f"ns:{result.stx_btime.tv_sec * 1_000_000_000 + result.stx_btime.tv_nsec}"


def _b64encode(value: bytes) -> str:
    return base64.urlsafe_b64encode(value).rstrip(b"=").decode("ascii")


def _b64decode(value: str) -> str:
    padding = "=" * (-len(value) % 4)
    return base64.urlsafe_b64decode(value + padding).decode("utf-8")


__all__ = [
    "RootBinding",
    "RootGrantError",
    "RootGrantSigner",
    "canonical_directory",
    "directory_birth",
    "directory_fingerprint",
]
