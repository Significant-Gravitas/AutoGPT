"""
Filesystem operations the standard library has no safe form of.
"""

from __future__ import annotations

import ctypes
import ctypes.util
import errno
import functools
import os
import secrets
import sys
from pathlib import Path

# The rename flags that refuse an existing destination.
_RENAME_NOREPLACE = 1  # Linux renameat2(2)
_RENAME_EXCL = 0x00000004  # macOS renamex_np(2)
_AT_FDCWD = -100
# A filesystem or kernel without the flag answers with one of these.
_UNSUPPORTED = {errno.EINVAL, errno.ENOSYS, getattr(errno, "ENOTSUP", errno.EOPNOTSUPP)}


# How much of a file's name a temporary name beside it keeps, in bytes, so
# the temporary name stays well under the 255 bytes filesystems allow.
_TEMPORARY_NAME_BYTES = 100


def temporary_sibling(path: Path) -> Path:
    """A fresh hidden name beside `path`, `.<name>.<random>.tmp`, for a file
    that is then renamed over it. The name is cut to 100 bytes on a
    character boundary: one of multi-byte characters, valid at 255 bytes,
    would otherwise make a temporary name too long to create."""
    name = path.name
    while len(os.fsencode(name)) > _TEMPORARY_NAME_BYTES:
        name = name[:-1]
    return path.with_name(f".{name}.{secrets.token_hex(6)}.tmp")


def rename_no_replace(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> None:
    """Rename `src` to `dst`, raising FileExistsError if `dst` exists.

    On POSIX `os.rename` replaces an existing destination silently, so
    checking first leaves a gap in which another file can appear and be
    replaced. Linux and macOS refuse it atomically (`renameat2` with
    RENAME_NOREPLACE, `renamex_np` with RENAME_EXCL); Windows' rename never
    replaces. Where neither is available, the check and the rename are two
    steps.
    """
    if rename_no_replace_in_one_step(src, dst):
        return
    if os.path.lexists(dst):
        raise FileExistsError(errno.EEXIST, os.strerror(errno.EEXIST), os.fspath(dst))
    os.rename(src, dst)


def rename_no_replace_in_one_step(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> bool:
    """`rename_no_replace` where the system does it in one step: Windows'
    rename, and Linux's and macOS's with their no-replace flag. False, with
    nothing done, where there is no such step (an old kernel, a filesystem
    that doesn't take the flag), for a caller that can't have the two-step
    way: another file could take the name between its look and its rename,
    and be replaced."""
    if sys.platform == "win32":
        os.rename(src, dst)
        return True
    return _native_no_replace_rename(src, dst)


@functools.cache
def _libc() -> ctypes.CDLL | None:
    """The C library, looked up once: on Linux, find_library runs
    `ldconfig -p` in a subprocess. None when it can't be loaded."""
    try:
        # By name: macOS keeps renamex_np in libSystem, not the main image.
        return ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)
    except OSError:
        return None


def _native_no_replace_rename(src: str | os.PathLike[str], dst: str | os.PathLike[str]) -> bool:
    """Do the rename with the OS's no-replace flag. False when it has none."""
    libc = _libc()
    if libc is None:
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


# The extended attributes Linux keeps a file's ACL in: a POSIX ACL, and the
# ACL an NFSv4 or SMB mount reads from its server.
_LINUX_ACL_ATTRIBUTES = ("system.posix_acl_access", "system.nfs4_acl", "system.cifs_acl")
# macOS's acl_type_t for its ACLs, the only kind it keeps, and the errors its
# acl_get_* functions give for a file without one.
_ACL_TYPE_EXTENDED = 0x00000100
_NO_MACOS_ACL = {
    errno.ENOENT,
    getattr(errno, "ENOATTR", errno.ENOENT),
    getattr(errno, "ENOTSUP", errno.EOPNOTSUPP),
    errno.EOPNOTSUPP,
}


def file_acl(target: int | str | os.PathLike[str]) -> object:
    """The ACL of `target`, a path (not followed if a link) or an open file:
    on Linux its ACL attributes, on macOS its extended ACL as text. None, or
    on Linux None for each attribute, where it has none or its filesystem
    keeps none, and None on other systems. Two files with equal values have
    the same ACL. OSError when it can't be read."""
    if sys.platform.startswith("linux"):
        return tuple(_linux_attribute(target, name) for name in _LINUX_ACL_ATTRIBUTES)
    if sys.platform == "darwin":
        return _macos_acl(target)
    return None


def _linux_attribute(target: int | str | os.PathLike[str], name: str) -> bytes | None:
    try:
        if isinstance(target, int):
            return os.getxattr(target, name)
        return os.getxattr(target, name, follow_symlinks=False)
    except OSError as exc:
        if exc.errno in (errno.ENODATA, errno.ENOTSUP, errno.EOPNOTSUPP):
            return None
        raise


def _macos_acl(target: int | str | os.PathLike[str]) -> bytes | None:
    libc = _libc()
    try:
        if libc is None:
            raise AttributeError("no C library")
        read = libc.acl_get_fd_np if isinstance(target, int) else libc.acl_get_link_np
        as_text, free = libc.acl_to_text, libc.acl_free
    except AttributeError:
        raise OSError(errno.ENOSYS, "The C library's ACL functions couldn't be found") from None
    subject: int | bytes
    if isinstance(target, int):
        read.argtypes = [ctypes.c_int, ctypes.c_int]
        subject = target
    else:
        read.argtypes = [ctypes.c_char_p, ctypes.c_int]
        subject = os.fsencode(target)
    read.restype = ctypes.c_void_p
    as_text.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    as_text.restype = ctypes.c_void_p
    free.argtypes = [ctypes.c_void_p]
    free.restype = ctypes.c_int
    acl = read(subject, _ACL_TYPE_EXTENDED)
    if not acl:
        error = ctypes.get_errno()
        # No ACL (ENOENT), or a filesystem that keeps none.
        if error in _NO_MACOS_ACL:
            return None
        raise OSError(error, os.strerror(error))
    try:
        text = as_text(acl, None)
        if not text:
            error = ctypes.get_errno()
            raise OSError(error, os.strerror(error))
        try:
            return ctypes.string_at(text)
        finally:
            free(text)
    finally:
        free(acl)


def can_open_folder(path: str | os.PathLike[str]) -> bool:
    """Whether this account can list the folder at `path` and reach what is
    in it.

    On POSIX, access(2) for read and search. It takes ACLs into account,
    and it asks without opening the folder: on macOS, opening one the
    privacy settings guard (Desktop, Documents, Downloads) puts a prompt on
    the person's screen, for a folder they only browsed past.

    On Windows os.access looks at the read-only attribute and no further,
    so it calls every folder usable, another account's profile among them.
    There the folder is opened for listing, which its access list allows or
    refuses.

    The answer is Windows' for this process. An administrator's process
    whose backup and restore privileges are switched on (one started from
    Git Bash, say) is let in whatever the access list says, here and in
    can_write_folder. Listing then works too; a file made the ordinary way
    doesn't use those privileges and can still be refused."""
    if sys.platform != "win32":
        return os.access(path, os.R_OK | os.X_OK)
    try:
        with os.scandir(path):
            return True
    except OSError:
        return False


def can_write_folder(path: str | os.PathLike[str]) -> bool:
    """Whether this account can make a file in the folder at `path`.

    On POSIX, access(2) for write and search. On Windows, where os.access
    calls every folder writable, the system is asked for a handle on the
    folder with the right to add a file to it (FILE_ADD_FILE): the folder's
    access list grants or refuses it, and a read-only volume refuses it.
    Nothing is written to find out: no file is made and removed, which a
    sync client or a file watcher would see."""
    if sys.platform != "win32":
        return os.access(path, os.W_OK | os.X_OK)
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    create_file = kernel32.CreateFileW
    create_file.argtypes = [
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    ]
    create_file.restype = wintypes.HANDLE
    close_handle = kernel32.CloseHandle
    close_handle.argtypes = [wintypes.HANDLE]
    file_add_file, open_existing = 0x0002, 3
    share_all = 0x1 | 0x2 | 0x4  # read, write and delete
    # What opening a folder, not a file, takes.
    backup_semantics = 0x02000000
    handle = create_file(
        os.fspath(path), file_add_file, share_all, None, open_existing, backup_semantics, None
    )
    if handle is None or handle == ctypes.c_void_p(-1).value:
        return False
    close_handle(handle)
    return True


__all__ = ["can_open_folder", "can_write_folder", "file_acl", "rename_no_replace"]
