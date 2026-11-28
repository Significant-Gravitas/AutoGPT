"""Publish a complete sibling file atomically without replacing a destination."""

from __future__ import annotations

import ctypes
import errno
import os
import sys
from pathlib import Path


def publish_file(pending: Path, target: Path) -> None:
    """Publish closed, flushed bytes; the caller removes any remaining pending file."""
    if sys.platform == "win32":
        os.rename(pending, target)
        return
    try:
        os.link(pending, target)
    except OSError as exc:
        if exc.errno not in (
            errno.EPERM,
            errno.ENOSYS,
            errno.ENOTSUP,
            errno.EOPNOTSUPP,
        ):
            raise
        rename_noreplace(pending, target)


def rename_noreplace(pending: Path, target: Path) -> None:
    source, destination = os.fsencode(pending), os.fsencode(target)
    if b"\0" in source or b"\0" in destination:
        raise ValueError("embedded null byte")
    library = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "linux":
        name = "renameat2"
        arguments = (-100, source, -100, destination, 1)
        types = [
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_int,
            ctypes.c_char_p,
            ctypes.c_uint,
        ]
    elif sys.platform == "darwin":
        name = "renamex_np"
        arguments = (source, destination, 4)
        types = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
    else:
        raise OSError(errno.ENOTSUP, "Atomic backup publication is unavailable")
    try:
        rename = getattr(library, name)
    except AttributeError as exc:
        raise OSError(
            errno.ENOTSUP, "Atomic backup publication is unavailable"
        ) from exc
    rename.argtypes = types
    rename.restype = ctypes.c_int
    if rename(*arguments):
        code = ctypes.get_errno()
        if code in (errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP):
            raise OSError(
                code,
                "This filesystem cannot publish a backup atomically without overwriting; choose another destination",
                str(target),
            )
        raise OSError(code, os.strerror(code), str(pending), None, str(target))
