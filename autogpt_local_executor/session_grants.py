"""Tamper-evident grants binding one chat session to one canonical root."""

from __future__ import annotations

import base64
import ctypes
import hashlib
import hmac
import json
import os
import platform
import re
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
        birth = directory_birth(root)
        fingerprint = _issued_fingerprint(root, birth)
        claims = {
            "allowed_root": str(root),
            "birth": birth,
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
            birth = claims["birth"]
            # The fingerprint a grant issued now would hold, or the one every
            # system's grant held before most of them left out st_dev, which
            # restores while that number holds. So on Btrfs a grant issued
            # with neither the number nor the folder's subvolume is refused:
            # a snapshot put in the folder's place matched it.
            accepted = {_issued_fingerprint(root, birth), _fingerprint(root, device=True)}
            if fingerprint not in accepted:
                raise RootGrantError("The selected directory changed after the grant was issued")
            # A filesystem that reuses a freed inode (ext4, tmpfs) gives a
            # recreated directory the old identity; its creation time still
            # differs. None where the platform keeps no creation time.
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
    number and canonical path, and its device number on Windows (the
    volume's serial number, kept from when the volume was formatted) and
    wherever the folder has no creation time.

    A POSIX st_dev is the number the system gives a device when it is
    attached or mounted: a reboot can number a btrfs subvolume or an LVM
    volume afresh, a share or a FUSE folder has a new one each time it is
    mounted, and a removable disk's depends on what else is plugged in. A
    grant that held it stopped restoring once it changed. A folder put in
    place of the selected one has another inode number or, on a filesystem
    that reuses a freed one, another creation time (directory_birth, which
    the grant checks too).

    Where the filesystem reports no creation time (an NFS share on Linux
    6.12 or earlier, say), the inode number can't tell the folder from
    another filesystem mounted at its path: the root folders of two
    filesystems of one type usually share an inode number (2 on ext4).
    There the device number counts, and a renumbered device asks for the
    folder again.

    On Linux the number alone doesn't tell them apart either: a share or a
    FUSE folder is numbered from a pool, its number freed when it is
    unmounted, and the next one mounted can be given it. So there the
    fingerprint also names the mount that holds the folder, by its source
    (server:/export, /dev/sdb1) and filesystem type, which the same
    filesystem mounted again keeps. Where the mount can't be found, the
    device number counts alone.

    On Btrfs a snapshot of the folder's subvolume keeps the folder's inode
    number and creation time, and the subvolume's owner can make one and
    put it in the folder's place without root. So there the fingerprint
    holds the subvolume's UUID, which the snapshot doesn't share and a
    renumbered device keeps. Where it can't be read (Linux before 4.18), the
    device number counts, as for a folder with no creation time.
    """
    root = canonical_directory(path)
    return _issued_fingerprint(root, directory_birth(root))


def _needs_device(birth: str | None) -> bool:
    """Whether the fingerprint of every folder with creation time `birth`
    holds its device number (directory_fingerprint says why). One on Btrfs
    may hold it too."""
    return sys.platform == "win32" or birth is None


def _issued_fingerprint(root: Path, birth: str | None) -> str:
    """The fingerprint a grant issued now holds for `root`, whose creation
    time is `birth`."""
    if birth is None:
        mounted = _mount_fingerprint(root)
        if mounted is not None:
            return mounted
    elif not _needs_device(birth) and _on_btrfs(root):
        subvolume = _btrfs_subvolume(root)
        if subvolume is None:
            return _fingerprint(root, device=True)
        return _digest(root, _root_stat(root), device=False, subvolume=subvolume)
    return _fingerprint(root, device=_needs_device(birth))


def _fingerprint(root: Path, *, device: bool) -> str:
    return _digest(root, _root_stat(root), device=device)


def _mount_fingerprint(root: Path) -> str | None:
    """The fingerprint of a Linux folder with no creation time: its device
    number, inode number and path, and the mount that holds it. None
    elsewhere, and where the mount can't be found."""
    if not sys.platform.startswith("linux"):
        return None
    stat = _root_stat(root)
    mount = _linux_mount(root, stat.st_dev)
    if mount is None:
        return None
    return _digest(root, stat, device=True, mount=mount)


def _root_stat(root: Path) -> os.stat_result:
    try:
        return root.stat(follow_symlinks=False)
    except OSError as exc:
        raise RootGrantError("Selected root is unavailable") from exc


def _digest(
    root: Path,
    stat: os.stat_result,
    *,
    device: bool,
    mount: dict[str, str] | None = None,
    subvolume: str | None = None,
) -> str:
    identity: dict[str, object] = {
        "inode": stat.st_ino,
        "path": os.path.normcase(str(root)),
    }
    if device:
        identity["device"] = stat.st_dev
    if mount is not None:
        identity["mount"] = mount
    if subvolume is not None:
        identity["subvolume"] = subvolume
    raw = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


# statfs(2)'s f_type for Btrfs (BTRFS_SUPER_MAGIC, <linux/magic.h>).
_BTRFS_SUPER_MAGIC = 0x9123683E
# BTRFS_IOC_GET_SUBVOL_INFO (<linux/btrfs.h>, Linux 4.18+), which any user
# who can open a folder may ask: _IOR(0x94, 60, struct
# btrfs_ioctl_get_subvol_info_args), a 504-byte struct holding the
# subvolume's UUID at bytes 296 to 311.
_BTRFS_IOC_GET_SUBVOL_INFO = 0x81F8943C
_SUBVOL_INFO_SIZE = 504
_SUBVOL_UUID = slice(296, 312)


class _Statfs(ctypes.Structure):
    # struct statfs begins with f_type: a long, except on s390, where it is
    # an unsigned int. The rest is room enough for the struct anywhere.
    _fields_ = [
        ("f_type", ctypes.c_uint if platform.machine().startswith("s390") else ctypes.c_long),
        ("_rest", ctypes.c_byte * 256),
    ]


def _on_btrfs(root: Path) -> bool:
    """Whether `root` is on Btrfs, by statfs(2). False off Linux, and where
    statfs can't be called."""
    if not sys.platform.startswith("linux"):
        return False
    try:
        statfs = ctypes.CDLL(None, use_errno=True).statfs
    except (AttributeError, OSError):
        return False
    statfs.argtypes = [ctypes.c_char_p, ctypes.POINTER(_Statfs)]
    statfs.restype = ctypes.c_int
    result = _Statfs()
    if statfs(os.fsencode(root), ctypes.byref(result)):
        return False
    return result.f_type & 0xFFFFFFFF == _BTRFS_SUPER_MAGIC


def _btrfs_subvolume(root: Path) -> str | None:
    """The UUID of the Btrfs subvolume that holds `root`, as hex, or None
    if it can't be read."""
    if not sys.platform.startswith("linux"):
        return None
    import fcntl

    try:
        fd = os.open(root, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC)
    except OSError:
        return None
    try:
        info = bytearray(_SUBVOL_INFO_SIZE)
        fcntl.ioctl(fd, _BTRFS_IOC_GET_SUBVOL_INFO, info, True)
    except OSError:
        return None
    finally:
        os.close(fd)
    return bytes(info[_SUBVOL_UUID]).hex()


# The mounts the shim can see, one a line (proc(5)): mount ID, parent ID,
# major:minor, the mount's root, its mount point, options, optional fields,
# "-", filesystem type, source, superblock options. A space, tab, newline or
# backslash in a field is written as an octal escape (\040 for a space).
_MOUNTINFO = "/proc/self/mountinfo"
_MOUNTINFO_ESCAPE = re.compile(rb"\\([0-7]{3})")


def _linux_mount(root: Path, device: int) -> dict[str, str] | None:
    """The source and filesystem type of the mount that holds `root`, a
    canonical path on device `device`: of the mounts of that device, the one
    whose mount point is the longest that holds the path. None if the mount
    table can't be read or no mount matches."""
    if not sys.platform.startswith("linux"):
        return None
    try:
        with open(_MOUNTINFO, "rb") as table:
            lines = table.read().splitlines()
    except OSError:
        return None
    path = os.fsencode(root)
    found: tuple[bytes, bytes, bytes] | None = None
    for line in lines:
        fields = line.split(b" ")
        try:
            separator = fields.index(b"-", 6)
            major, minor = fields[2].split(b":")
            number = os.makedev(int(major), int(minor))
            point = _unescape(fields[4])
            fs_type = _unescape(fields[separator + 1])
            source = _unescape(fields[separator + 2])
        except (IndexError, ValueError, OverflowError):
            continue
        holds = point == b"/" or path == point or path.startswith(point + b"/")
        if number != device or not holds:
            continue
        if found is None or len(point) >= len(found[0]):
            found = (point, fs_type, source)
    if found is None:
        return None
    return {"source": os.fsdecode(found[2]), "type": os.fsdecode(found[1])}


def _unescape(field: bytes) -> bytes:
    return _MOUNTINFO_ESCAPE.sub(lambda match: bytes([int(match.group(1), 8)]), field)


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
