"""Local drive file operations with retained, non-reparse parent handles.

UNC, device/extended namespaces, ADS and ambiguous Win32 names fail closed.
Parents deny write/delete sharing while walking to block replacement. Relative
opens also use OBJ_DONT_REPARSE: sharing alone cannot block reparse conversion.
Leaves are inspected before truncation and never reopened by name.
"""

import ctypes as c
import errno
import msvcrt
import ntpath
import os
import re
import sys
from contextlib import ExitStack, contextmanager

from backend.copilot import windows_handles as wh

if sys.platform != "win32":
    raise ImportError("Native Windows filesystem operations require Windows")

_RESERVED = re.compile(
    r"^(CON|PRN|AUX|NUL|CONIN\$|CONOUT\$|COM[1-9¹²³]|LPT[1-9¹²³])(?:\.|$)",
    re.IGNORECASE,
)


def components(path: str) -> tuple[str, list[str]]:
    if not ntpath.isabs(path) or ntpath.normpath(path) != path:
        raise ValueError(f"needs an absolute, normalised path: {path!r}")
    drive, tail = ntpath.splitdrive(path)
    if not re.fullmatch("[a-zA-Z]:", drive) or not tail.startswith("\\"):
        raise OSError(errno.ENOTSUP, "Only local drive paths are supported", path)
    parts = tail[1:].split("\\")
    if parts == [""]:
        raise IsADirectoryError(errno.EISDIR, "Is a directory", path)
    for part in parts:
        if (
            not part
            or part in (".", "..")
            or part.endswith((".", " "))
            or _RESERVED.match(part)
            or any(ord(ch) < 32 or ch in '<>:"/|?*' for ch in part)
            or len(part.encode("utf-16-le")) > 510
        ):
            raise OSError(errno.EINVAL, "Unsupported Windows path component", path)
    return drive, parts


@contextmanager
def parents(path: str, *, make: bool = False, security: bool = False):
    drive, parts = components(path)
    with ExitStack() as stack:
        root = wh.open_root(drive, security=security)
        stack.callback(wh.close, root)
        handles = [root]
        for name in parts[:-1]:
            root = wh.relative(
                root,
                name,
                wh.DIRECTORY_ACCESS | (wh.READ_CONTROL if security else 0),
                wh.FILE_OPEN_IF if make else wh.FILE_OPEN,
                directory=True,
            )
            stack.callback(wh.close, root)
            wh.inspect(root, path, directory=True)
            handles.append(root)
        yield root, parts[-1], handles


def open_no_follow(path: str, flags: int, *, make_parents: bool = False) -> int:
    supported = (
        os.O_WRONLY
        | os.O_RDWR
        | os.O_APPEND
        | os.O_CREAT
        | os.O_EXCL
        | os.O_TRUNC
        | os.O_BINARY
        | os.O_TEXT
        | os.O_NOINHERIT
    )
    if flags & ~supported or flags & 3 == 3:
        raise OSError(errno.ENOTSUP, "Unsupported open flags", path)
    if flags & os.O_TRUNC and not flags & (os.O_WRONLY | os.O_RDWR):
        raise OSError(errno.EINVAL, "Truncation requires write access", path)
    access = wh.READ_ATTRIBUTES
    if flags & 3 != os.O_WRONLY:
        access |= 1
    if flags & (os.O_WRONLY | os.O_RDWR):
        access |= 2 | 4
    disposition = wh.FILE_OPEN
    if flags & os.O_CREAT:
        disposition = wh.FILE_CREATE if flags & os.O_EXCL else wh.FILE_OPEN_IF
    with parents(path, make=make_parents) as (parent, name, _):
        # FILE_CREATE must report collisions even when the existing entry is a link.
        handle = wh.relative(
            parent, name, access, disposition, allow_link=disposition == wh.FILE_CREATE
        )
        try:
            wh.inspect(handle, path, directory=False)
            if flags & os.O_TRUNC:
                end = c.c_int64(0)
                if not wh.set_info(handle, 6, c.byref(end), c.sizeof(end)):
                    raise wh.win_error(c.get_last_error(), path)
            fd = msvcrt.open_osfhandle(
                handle, (flags & (os.O_APPEND | os.O_TEXT)) | os.O_NOINHERIT
            )
        except BaseException:
            wh.close(handle)
            raise
    try:
        msvcrt.setmode(fd, os.O_TEXT if flags & os.O_TEXT else os.O_BINARY)
        os.set_inheritable(fd, False)
    except BaseException:
        os.close(fd)
        raise
    return fd


def unlink_no_follow(path: str) -> None:
    with parents(path) as (parent, name, _):
        handle = wh.relative(
            parent, name, wh.DELETE | wh.READ_ATTRIBUTES, wh.FILE_OPEN, allow_link=True
        )
        try:
            attrs = wh.attributes(handle)
            if attrs & wh.DIRECTORY and not attrs & wh.REPARSE:
                raise IsADirectoryError(errno.EISDIR, "Is a directory", path)
            delete = c.c_ubyte(1)
            if not wh.set_info(handle, 4, c.byref(delete), c.sizeof(delete)):
                raise wh.win_error(c.get_last_error(), path)
        finally:
            wh.close(handle)
