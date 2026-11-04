"""
Filesystem operations the standard library has no safe form of.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import os
import sys

# The rename flags that refuse an existing destination.
_RENAME_NOREPLACE = 1  # Linux renameat2(2)
_RENAME_EXCL = 0x00000004  # macOS renamex_np(2)
_AT_FDCWD = -100
# A filesystem or kernel without the flag answers with one of these.
_UNSUPPORTED = {errno.EINVAL, errno.ENOSYS, getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)}


def rename_no_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    """Rename `src` to `dst`, raising FileExistsError if `dst` exists.

    On POSIX `os.rename` replaces an existing destination silently, so
    checking first leaves a gap in which another file can appear and be
    replaced. Linux and macOS refuse it atomically (`renameat2` with
    RENAME_NOREPLACE, `renamex_np` with RENAME_EXCL); Windows' rename never
    replaces. Where neither is available, the check and the rename are two
    steps.
    """
    if sys.platform != "win32" and _native_no_replace_rename(src, dst):
        return
    if sys.platform != "win32" and os.path.lexists(dst):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), os.fspath(dst))
    os.rename(src, dst)


def _native_no_replace_rename(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> bool:
    """Do the rename with the OS's no-replace flag. False when it has none."""
    try:
        # By name: macOS keeps renamex_np in libSystem, not the main image.
        libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    except OSError:
        return False
    old, new = os.fsencode(src), os.fsencode(dst)
    if sys.platform.startswith("linux"):
        call = getattr(libc, "renameat2", None)
        if call is None:
            return False
        call.argtypes = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
        result = call(_AT_FDCWD, old, _AT_FDCWD, new, _RENAME_NOREPLACE)
    elif sys.platform == "darwin":
        call = getattr(libc, "renamex_np", None)
        if call is None:
            return False
        call.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        result = call(old, new, _RENAME_EXCL)
    else:
        return False
    if result == 0:
        return True
    error = ctypes.get_errno()
    if error in _UNSUPPORTED:
        return False
    raise OSError(error, os.strerror(error), os.fspath(src), None, os.fspath(dst))


__all__ = ["rename_no_replace"]
