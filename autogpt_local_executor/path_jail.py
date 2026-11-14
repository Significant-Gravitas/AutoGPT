"""
Path jail — the single security-critical primitive that every FILE_*
handler must call before touching the filesystem.

The full algorithm is in docs/CROSS_PLATFORM.md "Path Jail Strategy".
This module is a faithful implementation of it; if you change behavior
here, update the doc first.
"""

from __future__ import annotations

import contextlib
import errno
import os
import re
import stat
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path, PurePath
from typing import BinaryIO

from .fsops import file_acl, temporary_sibling
from .platform_info import detect_platform

# Windows reserved device names (case-insensitive, with or without extension).
# Per docs/CROSS_PLATFORM.md → "Reserved filenames" row.
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}

# Invalid Windows path chars (per the same table).
_WINDOWS_INVALID_CHARS = re.compile(r'[<>"|?*\x00-\x1f]')

# Per-OS PATH_MAX limits, per CROSS_PLATFORM.md.
_PATH_MAX = {
    "darwin": 1024,
    "linux": 4096,
    "wsl2": 4096,
    "windows": 32767,
}


class PathJailError(Exception):
    """Raised when a path violates the jail.

    `code` is one of the ErrorCode enum values for direct mapping to the
    wire ERROR.code field.
    """

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def _is_reserved_name(p: PurePath) -> bool:
    if sys.platform != "win32":
        # Only Windows treats these as reserved.
        return False
    # Check every part — reserved names are reserved at every depth.
    for part in p.parts:
        stem = part.split(".", 1)[0].lower() if "." in part else part.lower()
        if stem in _WINDOWS_RESERVED:
            return True
    return False


def _has_alternate_data_stream(raw: str) -> bool:
    """NTFS Alternate Data Streams use `:` after the drive letter.

    On Windows, `C:\\foo\\bar.txt:hidden` is an ADS reference. We reject
    any `:` outside the drive prefix (positions 0-1).
    """
    if sys.platform != "win32":
        return False
    # Strip drive prefix (e.g., "C:") so we don't false-positive on it.
    rest = raw[2:] if len(raw) >= 2 and raw[1] == ":" else raw
    # Also strip UNC `\\?\` prefix.
    if rest.startswith("\\\\?\\"):
        rest = rest[4:]
        # And the next drive letter colon.
        if len(rest) >= 2 and rest[1] == ":":
            rest = rest[2:]
    return ":" in rest


def _has_invalid_chars(raw: str) -> bool:
    if sys.platform != "win32":
        # POSIX: only \x00 is illegal in paths.
        return "\x00" in raw
    # Strip drive letter colon before checking.
    rest = raw[2:] if len(raw) >= 2 and raw[1] == ":" else raw
    return bool(_WINDOWS_INVALID_CHARS.search(rest))


def _exceeds_path_max(raw: str) -> bool:
    limit = _PATH_MAX.get(detect_platform())
    if limit is None:
        return False
    return len(raw) > limit


def is_inside_jail(
    requested_path: str | os.PathLike[str], allowed_root: str | os.PathLike[str]
) -> bool:
    """Return True iff `requested_path` resolves inside `allowed_root`.

    Faithful implementation of the algorithm in CROSS_PLATFORM.md "Path
    Jail Strategy". Returns False on any check failure (reserved name,
    ADS, invalid chars, path-max exceeded, realpath OSError, or outside
    the jail). For diagnosable errors, use `assert_inside_jail()`.
    """
    try:
        assert_inside_jail(requested_path, allowed_root)
        return True
    except PathJailError:
        return False


def assert_inside_jail(
    requested_path: str | os.PathLike[str],
    allowed_root: str | os.PathLike[str],
) -> Path:
    """Raise PathJailError if `requested_path` is outside `allowed_root`.

    Returns the resolved Path on success — callers should use this Path
    rather than the original, so they always operate on the canonicalized
    form.
    """
    raw = str(requested_path)

    # 1. Lexical normalize. expanduser handles `~`; we keep the result as a
    #    PurePath for the reserved-name / ADS / invalid-char checks.
    try:
        lex = Path(raw).expanduser()
    except (RuntimeError, OSError) as exc:
        raise PathJailError("PATH_INVALID_CHARS", f"Bad path: {exc}") from exc

    if _exceeds_path_max(str(lex)):
        raise PathJailError(
            "PATH_INVALID_CHARS",
            f"Path exceeds OS PATH_MAX: len={len(str(lex))}",
        )

    if _is_reserved_name(lex):
        raise PathJailError(
            "PATH_RESERVED_NAME",
            f"Path uses a Windows reserved device name: {raw!r}",
        )

    if _has_alternate_data_stream(str(lex)):
        raise PathJailError(
            "PATH_INVALID_CHARS",
            f"Path contains an NTFS alternate data stream: {raw!r}",
        )

    if _has_invalid_chars(str(lex)):
        raise PathJailError(
            "PATH_INVALID_CHARS",
            f"Path contains characters illegal on this OS: {raw!r}",
        )

    # 2. Resolve symlinks, junctions, firmlinks, reparse points.
    #    Use os.path.realpath (Python 3.8+ handles Windows junctions) —
    #    pathlib's resolve() can hang on broken symlinks on some FS.
    try:
        resolved = os.path.realpath(from_root(lex, allowed_root), strict=False)
    except (OSError, ValueError) as exc:
        raise PathJailError(
            "PATH_INVALID_CHARS",
            f"Cannot resolve path {raw!r}: {exc}",
        ) from exc

    try:
        root_resolved = os.path.realpath(str(allowed_root), strict=True)
    except (OSError, ValueError) as exc:
        # Treat a missing allowed_root as a config error, not a path error,
        # but still surface it via the jail interface so handlers don't crash.
        raise PathJailError(
            "PATH_OUTSIDE_ALLOWED_ROOT",
            f"allowed_root {str(allowed_root)!r} does not exist: {exc}",
        ) from exc

    # 3. Exact prefix with a path separator boundary, so /workspace doesn't
    #    match /workspaceother.
    if _has_prefix(resolved, root_resolved):
        return Path(resolved)

    # 4. A case-insensitive filesystem can spell the root differently (macOS
    #    realpath keeps the spelling it was given), so ask the filesystem
    #    instead of folding case: the path is inside only if the root is one
    #    of its existing ancestors. Folding strings would let a case-variant
    #    sibling of the root (`WS` beside `Ws`) in wherever a directory on the
    #    way is case-sensitive.
    if _root_is_an_ancestor(resolved, root_resolved):
        return Path(resolved)

    raise PathJailError(
        "PATH_OUTSIDE_ALLOWED_ROOT",
        f"Path {raw!r} (resolved to {resolved!r}) is outside allowed_root {root_resolved!r}",
    )


def from_root(requested_path: str | os.PathLike[str], allowed_root: str | os.PathLike[str]) -> str:
    """The path a request names, before symlinks are resolved: `~` expanded,
    and a relative path taken from the root, as the platform means one (it
    joins a relative path to allowed_root itself), not from whatever
    directory the shim was started in."""
    return os.path.join(os.fspath(allowed_root), Path(os.fspath(requested_path)).expanduser())


def _has_prefix(path: str, root: str) -> bool:
    root_prefix = root if root.endswith(os.sep) else root + os.sep
    return path == root or path.startswith(root_prefix)


def _root_is_an_ancestor(path: str, root: str) -> bool:
    """Whether `root` is `path` itself or one of its existing ancestors, by
    device and inode, whatever case either is spelled in."""
    try:
        root_info = os.stat(root)
    except (OSError, ValueError):
        return False
    if root_info.st_ino == 0:
        # A filesystem without inode numbers can't establish identity.
        return False
    candidate = Path(path)
    for ancestor in (candidate, *candidate.parents):
        try:
            info = os.stat(ancestor)
        except (OSError, ValueError):
            continue  # not created yet; keep walking up
        if (info.st_dev, info.st_ino) == (root_info.st_dev, root_info.st_ino):
            return True
    return False


def assert_entry_inside_jail(
    requested_path: str | os.PathLike[str],
    allowed_root: str | os.PathLike[str],
) -> Path:
    """Return the directory entry `requested_path` names, for operations on the
    entry itself (delete, move), which must not follow a final symlink to its
    target.

    The entry's directory is resolved and must be inside `allowed_root`; its
    name is kept as given. That rules out the root itself, whose entry lives in
    the directory above it. Callers check `assert_inside_jail` too, so a link
    whose target is outside the root stays refused.
    """
    raw = str(requested_path)
    lex = Path(raw).expanduser()
    if lex.name in ("", ".."):
        raise PathJailError("PATH_INVALID_CHARS", f"Path {raw!r} does not name an entry")
    try:
        directory = assert_inside_jail(lex.parent, allowed_root)
    except PathJailError as exc:
        raise PathJailError(
            exc.code,
            f"Path {raw!r} names an entry outside allowed_root {str(allowed_root)!r}",
        ) from exc
    return directory / lex.name


def _assert_safe_content_stat(info: os.stat_result) -> None:
    if not stat.S_ISREG(info.st_mode):
        raise PathJailError("CAPABILITY_NOT_GRANTED", "Content access requires a regular file")
    if info.st_nlink != 1:
        raise PathJailError(
            "PATH_OUTSIDE_ALLOWED_ROOT",
            "Content access requires exactly one file link; hard-link scope cannot be established",
        )


@contextmanager
def open_jail_file(
    requested_path: str | os.PathLike[str],
    allowed_root: str | os.PathLike[str],
    *,
    write: bool = False,
) -> Iterator[BinaryIO]:
    """Open a single-linked regular file, without truncating before validation.

    This checks the content descriptor but does not pin or traverse ancestor
    directory descriptors. Concurrent same-user filesystem changes remain out
    of scope for this path-based preview.
    """
    path = assert_inside_jail(requested_path, allowed_root)
    try:
        _assert_safe_content_stat(path.stat())
    except FileNotFoundError:
        if not write:
            raise

    flags = (os.O_WRONLY | os.O_CREAT) if write else os.O_RDONLY
    flags |= getattr(os, "O_BINARY", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags, 0o666)
    try:
        _assert_safe_content_stat(os.fstat(descriptor))
        with os.fdopen(descriptor, "wb" if write else "rb", closefd=False) as file:
            yield file
    finally:
        os.close(descriptor)


def write_jail_file(
    requested_path: str | os.PathLike[str],
    allowed_root: str | os.PathLike[str],
    data: bytes,
) -> None:
    """Write `data` to a single-linked regular file inside the jail, or create it.

    The data goes to a new file beside the target, which is then renamed over
    it, so a write that fails part way (a full disk, say) leaves the old
    contents whole. An existing file keeps its mode, less set-user-ID and
    set-group-ID, and its owner and group; a new one gets 0o666 less the
    umask. Where the directory takes no new entries, (on Windows) another
    program has the file open, the shim may not give the new file the old
    one's owner and group (a file another account owns, a group the shim
    isn't in), the old file's ACL isn't what the new one gets from the
    folder (on Linux and macOS), or (on Windows) its owner or DACL isn't,
    the file is overwritten in place.
    """
    path = assert_inside_jail(requested_path, allowed_root)
    owner: tuple[int, int] | None = None
    set_id = False
    try:
        existing = path.stat()
    except FileNotFoundError:
        mode = None
    else:
        owner = (existing.st_uid, existing.st_gid)
        set_id = bool(existing.st_mode & (stat.S_ISUID | stat.S_ISGID))
        _assert_safe_content_stat(existing)
        # A rename ignores the target's mode, so a file the person made
        # read-only would be replaced whole; refuse it, as writing it in
        # place would be refused.
        if not os.access(path, os.W_OK):
            raise PermissionError(errno.EACCES, os.strerror(errno.EACCES), str(path))
        # New contents must not run with the file owner's privileges. Linux
        # drops these bits on a write by an unprivileged process; the shim
        # drops them itself, so a privileged one (root, say) does too.
        mode = stat.S_IMODE(existing.st_mode) & ~(stat.S_ISUID | stat.S_ISGID)
    try:
        _replace_with(path, data, mode, owner)
    except PermissionError:
        with open_jail_file(path, allowed_root, write=True) as file:
            if mode is not None and hasattr(os, "fchmod"):
                # Before the contents change, so they never carry the bits,
                # even when the write fails part way. Best effort for the
                # rest of the mode (a file someone else owns), but a file
                # whose set-ID bits stay isn't written: the kernel doesn't
                # drop them for a shim with CAP_FSETID.
                try:
                    os.fchmod(file.fileno(), mode)
                except OSError:
                    if set_id:
                        raise
            file.truncate(0)
            file.write(data)


def _replace_with(
    path: Path, data: bytes, mode: int | None, owner: tuple[int, int] | None = None
) -> None:
    flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
    flags |= getattr(os, "O_BINARY", 0) | getattr(os, "O_NOFOLLOW", 0)
    # A new file is opened with 0o666 so the umask applies, as it does to a
    # plain create. One that replaces a file starts owner-only and gets the
    # old mode below: another account that opened it under the umask's mode
    # (watching the directory, say) could read the new contents through that
    # descriptor once they were written, however private the old file was.
    temp = temporary_sibling(path)
    descriptor = os.open(temp, flags, 0o666 if mode is None else 0o600)
    try:
        with os.fdopen(descriptor, "wb") as file:
            if owner is not None and hasattr(os, "fchown"):
                created = os.fstat(file.fileno())
                if (created.st_uid, created.st_gid) != owner:
                    # The new file is the shim's, in its group or the
                    # directory's, and the old mode would then apply to them.
                    # A PermissionError here (an owner or group the shim may
                    # not give) sends the write in place, which keeps both.
                    os.fchown(file.fileno(), *owner)
            if mode is not None and hasattr(os, "fchmod"):
                os.fchmod(file.fileno(), mode)
            if mode is not None and sys.platform == "win32":
                _refuse_other_windows_permissions(path, temp)
            elif mode is not None:
                _refuse_another_acl(path, file.fileno())
            file.write(data)
            file.flush()
            os.fsync(file.fileno())
        if mode is not None and not hasattr(os, "fchmod"):
            os.chmod(temp, mode)
        os.replace(temp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temp)
        raise


def _refuse_another_acl(path: Path, descriptor: int) -> None:
    """PermissionError, which writes the file in place, unless the new file
    beside `path`, open on `descriptor` and given the old mode, has the ACL
    `path` has (Linux, macOS). The rename keeps the new file's ACL, none or
    the folder's default entries, so a file with entries of its own would
    lose them, and the new one could get the folder's. On Linux the mode's
    group bits are an ACL's mask: a file that let another account in and
    kept its own group out would come back open to the group. Checked
    before any content is written; written in place, the file keeps its
    own ACL. One that can't be read is taken for one that differs."""
    try:
        same = file_acl(descriptor) == file_acl(path)
    except OSError as exc:
        raise PermissionError(
            errno.EACCES, f"The ACL of {path} couldn't be compared ({exc})"
        ) from exc
    if not same:
        raise PermissionError(errno.EACCES, f"{path} has an ACL a replacement wouldn't keep")


def _refuse_other_windows_permissions(path: Path, temp: Path) -> None:
    """PermissionError, which writes the file in place, unless the new file
    beside `path` has the owner and DACL `path` has (Windows). The new file
    takes the folder's inheritable entries and the rename keeps them, so a
    file with entries of its own (inheritance off, its owner alone, in a
    folder other accounts can read) would become readable to them. Checked
    before any content is written; written in place, the file keeps its
    own permissions."""
    try:
        same = _windows_owner_and_dacl(temp) == _windows_owner_and_dacl(path)
    except OSError as exc:
        raise PermissionError(
            errno.EACCES, f"The permissions of {path} couldn't be compared ({exc})"
        ) from exc
    if not same:
        raise PermissionError(errno.EACCES, f"{path} has permissions a replacement wouldn't keep")


def _windows_owner_and_dacl(path: Path) -> str:
    """`path`'s owner and DACL, as an SDDL string (Windows)."""
    if sys.platform != "win32":
        raise OSError(errno.ENOTSUP, "Windows only")
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    get_security = advapi32.GetNamedSecurityInfoW
    get_security.argtypes = [
        wintypes.LPCWSTR,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.c_void_p,
        ctypes.POINTER(ctypes.c_void_p),
    ]
    get_security.restype = wintypes.DWORD
    as_text = advapi32.ConvertSecurityDescriptorToStringSecurityDescriptorW
    as_text.argtypes = [
        ctypes.c_void_p,
        wintypes.DWORD,
        wintypes.DWORD,
        ctypes.POINTER(ctypes.c_void_p),
        ctypes.c_void_p,
    ]
    as_text.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = [ctypes.c_void_p]
    kernel32.LocalFree.restype = ctypes.c_void_p
    se_file_object, sddl_revision_1 = 1, 1
    owner_and_dacl = 0x1 | 0x4  # OWNER_ and DACL_SECURITY_INFORMATION

    descriptor = ctypes.c_void_p()
    status = get_security(
        str(path),
        se_file_object,
        owner_and_dacl,
        None,
        None,
        None,
        None,
        ctypes.byref(descriptor),
    )
    if status:
        raise ctypes.WinError(status)
    try:
        text = ctypes.c_void_p()
        if not as_text(descriptor, sddl_revision_1, owner_and_dacl, ctypes.byref(text), None):
            raise ctypes.WinError(ctypes.get_last_error())
        try:
            return ctypes.wstring_at(text.value) if text.value else ""
        finally:
            kernel32.LocalFree(text)
    finally:
        kernel32.LocalFree(descriptor)


__all__ = [
    "PathJailError",
    "assert_entry_inside_jail",
    "assert_inside_jail",
    "from_root",
    "is_inside_jail",
    "open_jail_file",
    "write_jail_file",
]
