"""Host-side file access to the chat's working directory.

Each helper walks a ``realpath``-checked path one component at a time without
following a link, so it reaches the checked location or fails.  The rule in
``backend/util/architecture_test.py`` keeps other by-name file calls out of the
modules that touch that directory.
"""

import errno
import os
import re
import stat
import sys

# O_PATH walks a directory with search permission only, as path lookup does.
_DIR_ACCESS = os.O_PATH if sys.platform == "linux" else os.O_RDONLY
_DIR_FLAGS = _DIR_ACCESS | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC
_UNSAFE_FILENAME = re.compile(r"[^\w.\-]")


def open_no_follow(path: str, flags: int, *, make_parents: bool = False) -> int:
    """``os.open`` for a ``realpath``-resolved *path*, refusing a link at any
    component and anything but a regular file.  Fits ``open(..., opener=...)``."""
    *parents, name = _components(path)
    dir_fd = _open_dir(parents, make_parents=make_parents, path=path)
    try:
        fd = os.open(name, flags | os.O_NOFOLLOW | os.O_NONBLOCK, 0o666, dir_fd=dir_fd)
    except OSError as exc:
        raise type(exc)(exc.errno, exc.strerror, path) from None
    finally:
        os.close(dir_fd)
    try:
        # O_NONBLOCK keeps a FIFO from hanging the open; it is refused here.
        if not stat.S_ISREG(os.fstat(fd).st_mode):
            raise OSError(errno.EINVAL, "Not a regular file", path)
        os.set_blocking(fd, True)
    except BaseException:
        os.close(fd)
        raise
    return fd


def unlink_no_follow(path: str) -> None:
    """Remove the entry *path* names, refusing a link among its parents.  A link
    as the last component is removed itself, as ``rm`` does."""
    *parents, name = _components(path)
    dir_fd = _open_dir(parents, make_parents=False, path=path)
    try:
        os.unlink(name, dir_fd=dir_fd)
    except OSError as exc:
        raise type(exc)(exc.errno, exc.strerror, path) from None
    finally:
        os.close(dir_fd)


def save_new_file(directory: str, filename: str, content: bytes) -> str:
    """Write *content* under a name in *directory* that nothing holds yet, adding
    a numeric suffix on a collision; returns the path.  An existing entry, a
    dangling link included, is a collision and is never written.  *directory* is
    used as given: pass the working directory's root or one this process made."""
    safe = _UNSAFE_FILENAME.sub("_", filename)
    if safe in ("", ".", ".."):
        safe = "file"
    stem, ext = os.path.splitext(safe)
    root = os.path.realpath(directory)
    for idx in range(1000):
        candidate = os.path.join(root, f"{stem}_{idx}{ext}" if idx else safe)
        try:
            fd = open_no_follow(candidate, os.O_WRONLY | os.O_CREAT | os.O_EXCL)
        except FileExistsError:
            continue
        with open(fd, "wb") as f:
            f.write(content)
        return candidate
    raise FileExistsError(errno.EEXIST, "No free name", os.path.join(root, safe))


def cli_tmpdir(sdk_cwd: str) -> str:
    """The CLI's temp directory, beside *sdk_cwd*; the CLI asks for a private
    (0700) one of its own."""
    return f"{sdk_cwd}.cli"


def _components(path: str) -> list[str]:
    if not os.path.isabs(path) or os.path.normpath(path) != path:
        raise ValueError(f"needs an absolute, normalised path: {path!r}")
    parts = path.split(os.sep)[1:]
    if not parts or not parts[-1]:
        raise IsADirectoryError(errno.EISDIR, "Is a directory", path)
    return parts


def _open_dir(parents: list[str], *, make_parents: bool, path: str) -> int:
    dir_fd = os.open(os.sep, _DIR_FLAGS)
    try:
        for parent in parents:
            try:
                child = os.open(parent, _DIR_FLAGS, dir_fd=dir_fd)
            except FileNotFoundError:
                if not make_parents:
                    raise
                try:
                    os.mkdir(parent, dir_fd=dir_fd)
                except FileExistsError:
                    pass
                child = os.open(parent, _DIR_FLAGS, dir_fd=dir_fd)
            os.close(dir_fd)
            dir_fd = child
    except OSError as exc:
        os.close(dir_fd)
        raise type(exc)(exc.errno, exc.strerror, path) from None
    return dir_fd
