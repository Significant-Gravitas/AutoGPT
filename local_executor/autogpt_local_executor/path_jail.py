"""
Path jail — the single security-critical primitive that every FILE_*
handler must call before touching the filesystem.

The full algorithm is in docs/CROSS_PLATFORM.md "Path Jail Strategy".
This module is a faithful implementation of it; if you change behavior
here, update the doc first.

A path is first placed by its name alone, and one that is plainly outside
the root is refused before anything looks at it (_plainly_outside): looking
at a path on another host contacts that host. A path whose name is inside is
then resolved, links included, and the resolved path decides.
"""

from __future__ import annotations

import contextlib
import errno
import ntpath
import os
import re
import stat
import sys
import unicodedata
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path, PurePath
from types import ModuleType
from typing import BinaryIO

from .fsops import file_acl, temporary_sibling
from .platform_info import detect_platform

# Windows reserved device names (case-insensitive, with or without extension).
# Per docs/CROSS_PLATFORM.md → "Reserved filenames" row. The digits include
# the superscript ones, which Windows takes for 1, 2 and 3.
_WINDOWS_RESERVED = {
    "con",
    "prn",
    "aux",
    "nul",
    "conin$",
    "conout$",
    *(f"com{digit}" for digit in "123456789\u00b9\u00b2\u00b3"),
    *(f"lpt{digit}" for digit in "123456789\u00b9\u00b2\u00b3"),
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
    wire ERROR.code field. `message` goes to the platform with it, and names
    only the path that was asked for and the root as the platform was told
    it. `detail` stays on this computer, for its log: where the path
    resolved to, the root's own real place, the system's error with its
    file name. Sent on, those told the platform where a link in the shared
    folder points outside it, and on Windows, by the letter case of what
    came back, whether a path exists.
    """

    def __init__(self, code: str, message: str, *, detail: str | None = None) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.detail = detail


def _outside(raw: str, allowed_root: str | os.PathLike[str], detail: str) -> PathJailError:
    """The one refusal of a path outside the root, whichever check found it:
    by its name alone (step 3) or once it was resolved (steps 5 and 6). A
    message of its own for each said which, and so whether a link is on the
    way."""
    return PathJailError(
        "PATH_OUTSIDE_ALLOWED_ROOT",
        f"Path {raw!r} is outside allowed_root {os.fspath(allowed_root)!r}",
        detail=detail,
    )


def _is_reserved_name(p: PurePath) -> bool:
    if sys.platform != "win32":
        # Only Windows treats these as reserved.
        return False
    # Check every part — reserved names are reserved at every depth.
    return any(_is_windows_device_name(part) for part in p.parts)


def _is_windows_device_name(name: str) -> bool:
    """Whether Windows opens a device for a file of this name, in whatever
    folder: `NUL`, and `nul.txt` and `NUL .txt` too, since it drops what
    follows the first dot and the spaces before it."""
    return name.partition(".")[0].rstrip(" ").lower() in _WINDOWS_RESERVED


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
    ADS, invalid chars, path-max exceeded, a name outside the root,
    realpath OSError, or outside the jail). For diagnosable errors, use
    `assert_inside_jail()`.
    """
    try:
        assert_inside_jail(requested_path, allowed_root)
        return True
    except PathJailError:
        return False


def assert_inside_jail(
    requested_path: str | os.PathLike[str],
    allowed_root: str | os.PathLike[str],
    *,
    asked: str | os.PathLike[str] | None = None,
) -> Path:
    """Raise PathJailError if `requested_path` is outside `allowed_root`.

    Returns the resolved Path on success — callers should use this Path
    rather than the original, so they always operate on the canonicalized
    form.

    Only the root is looked at before the path's name is known to be
    inside it (step 3): a path on another drive, share, host or device
    namespace, elsewhere on the disk, or one that climbs above the root
    through `..` at any point, is refused without a `realpath`, `stat` or
    `open` of it.

    `asked` is the path as the request named it, for a check of a path an
    earlier check resolved (open_jail_file): a refusal's message names
    that one, and the resolved path it was given stays in `detail`.
    """
    raw = str(requested_path)
    named = raw if asked is None else str(asked)

    # 1. Lexical normalize. expanduser handles `~`; we keep the result as a
    #    PurePath for the reserved-name / ADS / invalid-char checks. Nothing
    #    in this step or the next two looks at the requested path on disk.
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
            f"Path uses a Windows reserved device name: {named!r}",
        )

    if _has_alternate_data_stream(str(lex)):
        raise PathJailError(
            "PATH_INVALID_CHARS",
            f"Path contains an NTFS alternate data stream: {named!r}",
        )

    if _has_invalid_chars(str(lex)):
        raise PathJailError(
            "PATH_INVALID_CHARS",
            f"Path contains characters illegal on this OS: {named!r}",
        )

    # 2. Resolve the root: the folder the person chose, and the one place
    #    that is looked at before the requested path is known to be in it.
    try:
        root_resolved = os.path.realpath(str(allowed_root), strict=True)
    except (OSError, ValueError) as exc:
        # Treat a missing allowed_root as a config error, not a path error,
        # but still surface it via the jail interface so handlers don't crash.
        raise PathJailError(
            "PATH_OUTSIDE_ALLOWED_ROOT",
            f"allowed_root {str(allowed_root)!r} does not exist or can't be reached",
            # The system's error names the part it stopped at, with the
            # links before it followed.
            detail=f"allowed_root {str(allowed_root)!r} couldn't be resolved: {exc}",
        ) from exc

    # 3. Refuse by its name a path that is plainly outside the root, before
    #    anything looks at it. Resolving a path asks the system about each
    #    folder on the way: for `\\host\share\x` Windows connects to the
    #    host and signs in with the person's account, `\\.\pipe\x` opens
    #    a device, and on macOS `/net/<host>/x` has autofs contact the host,
    #    all before the path could be refused.
    if _plainly_outside(str(lex), _root_spellings(allowed_root, root_resolved)):
        raise _outside(
            named,
            allowed_root,
            # Worked out on the names alone, as the check was.
            f"{raw!r} comes to {os.path.normpath(from_root(lex, allowed_root))!r} by its "
            f"name, outside the root, which is {root_resolved!r}",
        )

    # 4. Resolve symlinks, junctions, firmlinks, reparse points. Its name
    #    is inside the root, so this reads the root and what is in it, and
    #    whatever a link in there points at: resolving is how the jail
    #    learns where one points, and the resolved path decides (5, 6).
    #    Use os.path.realpath (Python 3.8+ handles Windows junctions) —
    #    pathlib's resolve() can hang on broken symlinks on some FS.
    try:
        resolved = os.path.realpath(from_root(lex, allowed_root), strict=False)
    except (OSError, ValueError) as exc:
        raise PathJailError(
            "PATH_INVALID_CHARS",
            f"Cannot resolve path {named!r}: {_reason(exc)}",
            detail=f"{raw!r} couldn't be resolved: {exc}",
        ) from exc

    # 5. Exact prefix with a path separator boundary, so /workspace doesn't
    #    match /workspaceother.
    if _has_prefix(resolved, root_resolved):
        return Path(resolved)

    # 6. A case-insensitive filesystem can spell the root differently (macOS
    #    realpath keeps the spelling it was given), so ask the filesystem
    #    instead of folding case: the path is inside only if the root is one
    #    of its existing ancestors. Folding strings would let a case-variant
    #    sibling of the root (`WS` beside `Ws`) in wherever a directory on the
    #    way is case-sensitive.
    if _root_is_an_ancestor(resolved, root_resolved):
        return Path(resolved)

    raise _outside(
        named,
        allowed_root,
        f"{raw!r} resolves to {resolved!r}, outside the root, which is {root_resolved!r}",
    )


def _reason(exc: BaseException) -> str:
    """Why the system refused, in its own words, without the file name its
    error ends with: that is the path it was given, a resolved one."""
    if isinstance(exc, OSError):
        return exc.strerror or type(exc).__name__
    return str(exc)


def from_root(requested_path: str | os.PathLike[str], allowed_root: str | os.PathLike[str]) -> str:
    """The path a request names, before symlinks are resolved: `~` expanded,
    and a relative path taken from the root, as the platform means one (it
    joins a relative path to allowed_root itself), not from whatever
    directory the shim was started in."""
    return os.path.join(os.fspath(allowed_root), Path(os.fspath(requested_path)).expanduser())


def _root_spellings(allowed_root: str | os.PathLike[str], root_resolved: str) -> tuple[str, ...]:
    """The names a path inside the root can start with.

    The platform's own tools send a path under the root the shim reported
    to it, with `.` and `..` already worked out on the names, and refuse
    any other themselves: the root as configured in a legacy session, and
    the resolved one the folder browser gave otherwise. A path copied from
    a command's output is under the resolved root, where the command ran.
    So both are taken, and they differ wherever a link is on the way: `/tmp`
    on macOS, a home folder that is a link, a mapped drive (`Z:\\ws` and
    the share it stands for), a Windows name typed in its 8.3 form.

    The third is the root by the home folder's own name, when the home
    folder is a link and the root is in it: `~` expands to that name.

    A path that reaches the root through some other link outside it is not
    taken: it has to be written by one of these names. Nor is a path that
    writes a long-named root by 8.3 names: the 8.3 form counts where the
    root was configured in it.

    The configured root comes first: it is the one a request's path is
    joined to (from_root). A root configured with `.` or `..` in it
    (`/work/staging/../shared`) is kept as it was given and reported that
    way in HELLO, so the platform's paths begin with it as written: that
    spelling is one of the root's, ahead of the same root with its dots
    worked out.
    """
    configured = os.fspath(allowed_root)
    spellings = [os.path.abspath(configured), root_resolved]
    if os.path.isabs(configured) and configured != spellings[0]:
        spellings.insert(0, configured)
    home = os.path.expanduser("~")
    if os.path.isabs(home):
        home_resolved = _resolved_folder(home)
        if home_resolved != home and _has_prefix(root_resolved, home_resolved):
            spellings.append(home + root_resolved[len(home_resolved) :])
    return tuple(spellings)


@lru_cache(maxsize=8)
def _resolved_folder(folder: str) -> str:
    """`folder` with its links followed, or as it is when it can't be read.
    Kept: the home folder doesn't move while the shim runs."""
    try:
        return os.path.realpath(folder)
    except (OSError, ValueError):
        return folder


def _plainly_outside(
    requested: str, root_spellings: Iterable[str], *, pathmod: ModuleType = os.path
) -> bool:
    """Whether `requested`, a request's path with `~` expanded, is outside
    every spelling of the root by its name alone. Nothing is looked at.

    A relative path is taken from the first spelling, the configured root,
    as the path that goes on to be resolved is (from_root). The path has to
    begin with a spelling of the root as it is written, and stay under it
    name by name: a `..` is followed in its place, not worked out first.
    Resolving a path looks at each folder it passes through, so one that
    leaves the root and comes back (`/net/host/../../workspace/file`, for a
    root `/workspace`) would have the folder outside looked at on the way,
    though it ends inside. The platform works `.` and `..` out before it
    sends a path, so its own paths never do that.
    Names are compared without regard to case or to how an accented letter
    is written, since a filesystem may take either for the same folder (on
    Windows and macOS it does), and with either separator on Windows. That
    lets through a path the resolved check then refuses; it never lets one
    in. What is outside by its name:

    - a path that is not under the root: elsewhere on the disk, above the
      root through `..` at any point, even to come back into it, or a name
      that only starts like the root's;
    - on Windows, another drive, a path relative to another drive's folder
      (`D:x`), another share or host (`\\\\host\\share\\x`,
      `//host/share/x`), and the device namespaces (`\\\\.\\`,
      `\\\\?\\`), the root's own drive in them included.

    `pathmod` is os.path, or ntpath or posixpath to place another system's
    paths.
    """
    windows = pathmod is ntpath
    spellings = tuple(root_spellings)
    candidate = _named(pathmod.join(spellings[0], requested), windows, keep_dots=True)
    if candidate is None:
        return True
    for spelling in spellings:
        # As it is written too: a path under a root configured with `..`
        # in it begins with those same names.
        root = _named(spelling, windows, keep_dots=True)
        if root is not None and candidate[0] == root[0] and _stays_under(candidate[1], root[1]):
            return False
    return True


def _stays_under(names: tuple[str, ...], root: tuple[str, ...]) -> bool:
    """Whether a path's names, with each `..` still in its place, begin
    with the root's and never climb above it on the way. The top of a disk
    or a share has nothing above it: its own `..` is the top."""
    if names[: len(root)] != root:
        return False
    depth = 0
    for name in names[len(root) :]:
        if name != "..":
            depth += 1
        elif depth > 0:
            depth -= 1
        elif root:
            return False
    return True


def _named(
    path: str, windows: bool, *, keep_dots: bool = False
) -> tuple[str, tuple[str, ...]] | None:
    """An absolute path as its anchor (a drive or a share on Windows, and
    nothing elsewhere) and the names after it, each in the form names are
    compared in, with `.` dropped and `..` worked out on the names, or with
    `keep_dots` left in its place. None for what is no plain absolute path:
    on Windows a device path, a path relative to a drive's folder, and one
    with no drive."""
    if windows:
        path = path.replace("/", "\\")
        if path.startswith("\\\\"):
            host, _, rest = path[2:].partition("\\")
            share, _, rest = rest.partition("\\")
            # A host of `.` or `?` is a device namespace, and a third
            # separator leaves no host.
            if host in ("", ".", "?") or not share:
                return None
            anchor = f"\\\\{host}\\{share}"
        elif len(path) >= 3 and path[0].isalpha() and path[1:3] == ":\\":
            anchor, rest = path[:2], path[3:]
        else:
            return None
        separator = "\\"
    else:
        if not path.startswith("/"):
            return None
        anchor, rest, separator = "", path, "/"
    names: list[str] = []
    for name in rest.split(separator):
        if name in ("", "."):
            continue
        if name == ".." and not keep_dots:
            # The top's own `..` is the top.
            if names:
                names.pop()
            continue
        names.append(_compared(name))
    return _compared(anchor), tuple(names)


def _compared(name: str) -> str:
    return unicodedata.normalize("NFC", name).casefold()


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
            detail=exc.detail,
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
    asked: str | os.PathLike[str] | None = None,
) -> Iterator[BinaryIO]:
    """Open a single-linked regular file, without truncating before validation.

    This checks the content descriptor but does not pin or traverse ancestor
    directory descriptors. Concurrent same-user filesystem changes remain out
    of scope for this path-based preview.

    A handler gives the path its first check resolved, which is checked
    again here, and the path as the request named it as `asked`: a refusal
    at this second check names that one (assert_inside_jail).
    """
    path = assert_inside_jail(requested_path, allowed_root, asked=asked)
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
    *,
    asked: str | os.PathLike[str] | None = None,
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

    `asked` is the path as the request named it, as for open_jail_file.
    """
    path = assert_inside_jail(requested_path, allowed_root, asked=asked)
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
        with open_jail_file(path, allowed_root, write=True, asked=asked) as file:
            if set_id and mode is not None and hasattr(os, "fchmod"):
                # Before the contents change, so they never carry the bits,
                # even when the write fails part way. A file whose set-ID
                # bits stay isn't written: the kernel doesn't drop them for a
                # shim with CAP_FSETID. Otherwise the mode is left alone,
                # since it is the file's own already: on an NFS mount a
                # chmod reaches the server even when it changes nothing, and
                # a server that drops ACL entries on a chmod (ZFS's
                # aclmode=discard, say) would drop the ACL this write keeps.
                os.fchmod(file.fileno(), mode)
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
