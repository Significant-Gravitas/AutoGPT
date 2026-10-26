"""Windows only: a second way to a directory, whose path RabbitMQ's batch
scripts can take.

They cannot take a space, and the 8.3 short name that normally stands in for
a long one does not exist on a volume that has short names turned off: the
default for every drive but the system one, and a setting on that one too. A
user called "John Smith" on such a machine has a space in every path under
their profile. A directory junction does what the short name did, and making
one needs no privilege.

Where the junctions live decides who can point them somewhere else, and
RabbitMQ reads its configuration and its cookie through them:

* under %LOCALAPPDATA%, when that path (or its short name) is itself usable.
  It is the user's own profile, as private as the data directory.
* otherwise directly under %ProgramData%, in a folder named after the install
  and the user's SID. %ProgramData% has no user's name in it; only
  administrators can rename or delete what is in it, but any user can create
  a folder there. So the folder is created closed to everyone but its user,
  the system and administrators, in the one call that creates it, and one
  that is already there is used only when it is a real directory, owned by
  one of those three, that nobody else may change. All of that is read from
  one open handle to the folder, never from its path a second time: whoever
  made the folder could otherwise show a different object to each question.
  A folder that passes cannot be moved or replaced by anyone but those
  three, so the path stays what was checked. Another user of the machine
  can take the name first; that makes the app say it cannot run, and never
  lets them redirect it.

Each install of the app (install.py) has a folder of its own in either place.
"""

from __future__ import annotations

import contextlib
import hashlib
import logging
import os
import stat
import subprocess
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger("autogpt_desktop")

ADMINISTRATORS = "S-1-5-32-544"
SYSTEM = "S-1-5-18"
ERROR_ALREADY_EXISTS = 183
CREATE_NO_WINDOW = 0x08000000
ACCESS_ALLOWED, ACCESS_DENIED = 0, 1
# The rights that let their holder put something into a directory, take
# something out of it, or change who may: FILE_ADD_FILE, FILE_ADD_SUBDIRECTORY,
# FILE_WRITE_EA, FILE_DELETE_CHILD, FILE_WRITE_ATTRIBUTES, DELETE, WRITE_DAC,
# WRITE_OWNER, GENERIC_ALL, GENERIC_WRITE.
CHANGES = (
    0x2
    | 0x4
    | 0x10
    | 0x40
    | 0x100
    | 0x10000
    | 0x40000
    | 0x80000
    | 0x10000000
    | 0x40000000
)


@dataclass(frozen=True)
class Root:
    """A folder for junctions. `guarded` is the one outside the user's
    profile, which has to be created closed and checked before use."""

    path: Path
    guarded: bool


@dataclass(frozen=True)
class Ace:
    """One entry of an access list: allowed or denied, what, to whom."""

    kind: int
    mask: int
    sid: str


@dataclass(frozen=True)
class Seen:
    """What one open handle says the object behind it is. `access` is None
    for an object with no access list, which is open to everyone."""

    attributes: int
    owner: str
    access: tuple[Ace, ...] | None


def plain(text: str) -> bool:
    """Whether the batch scripts can take it: ASCII, and no spaces."""
    return text.isascii() and not any(character.isspace() for character in text)


def alias(path: Path, install_name: str) -> str | None:
    """A plain path to `path` through a junction, or None when there is
    nowhere to put one (the log says why)."""
    try:
        user = current_user_sid()
    except OSError as exc:
        logger.warning(f"could not read the user's SID ({exc}); no alias for {path}")
        return None
    for root in roots(install_name, os.environ, user, short_name):
        try:
            with held(root, user):
                return str(junction(root.path, path))
        except OSError as exc:
            logger.warning(f"cannot keep an alias for {path} in {root.path}: {exc}")
    return None


def roots(
    install_name: str, environ: Mapping[str, str], user_sid: str, shorten
) -> list[Root]:
    """Where the junctions may live, best first. `shorten` gives a path's
    8.3 spelling (or the path back when it has none)."""
    found = []
    local = environ.get("LOCALAPPDATA")
    if local:
        spelling = next((s for s in (local, shorten(local)) if plain(s)), None)
        if spelling:
            found.append(Root(Path(spelling) / install_name / "links", guarded=False))
    shared = environ.get("PROGRAMDATA") or environ.get("ALLUSERSPROFILE")
    if shared and plain(shared):
        found.append(Root(Path(shared) / f"{install_name}-{user_sid}", guarded=True))
    return found


@contextlib.contextmanager
def held(root: Root, user_sid: str):
    """The folder, there and fit to hold junctions, for as long as the
    caller is making one. A guarded folder is judged through a handle that
    stays open meanwhile, and that does not share the right to delete: the
    object that was judged cannot be renamed or removed until it closes."""
    if not root.guarded:
        root.path.mkdir(parents=True, exist_ok=True)
        yield
        return
    create_closed(root.path, user_sid)
    handle = open_as_it_is(root.path)
    try:
        problem = refusal(look(handle), user_sid)
        if problem:
            raise PermissionError(f"{root.path} {problem}")
        yield
    finally:
        _kernel32().CloseHandle(handle)


def refusal(seen: Seen, user_sid: str) -> str | None:
    """Why a folder outside the user's profile may not hold the junctions,
    or None when it may."""
    if seen.attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT:
        return "is a link, which someone else can repoint"
    if not seen.attributes & stat.FILE_ATTRIBUTE_DIRECTORY:
        return "is not a folder"
    if not trusted(seen.owner, user_sid):
        return f"belongs to another user ({seen.owner})"
    if seen.access is None:
        return "has no access list, so anyone can change what is in it"
    for ace in seen.access:
        if ace.kind == ACCESS_DENIED or trusted(ace.sid, user_sid):
            continue
        if ace.kind != ACCESS_ALLOWED:
            return f"has an access entry of a kind ({ace.kind}) that cannot be judged"
        if ace.mask & CHANGES:
            return f"can be changed by someone else ({ace.sid})"
    return None


def trusted(sid: str, user_sid: str) -> bool:
    """Administrators and the system own what an elevated run creates, and
    could redirect anything anyway. Nobody else can give a folder to them or
    to the user: taking ownership works only towards oneself."""
    return sid in (user_sid, ADMINISTRATORS, SYSTEM)


def junction(root: Path, target: Path) -> Path:
    """The junction for `target` under `root`, made, or made again when it
    points somewhere else (the data directory was moved; an old install)."""
    forget_dangling(root)
    name = hashlib.sha256(os.path.normcase(str(target)).encode()).hexdigest()[:16]
    link = root / name
    if os.path.lexists(link) and not points_at(link, target):
        os.rmdir(link)  # a junction is removed like an empty directory
    if not os.path.lexists(link):
        create_junction(target, link)
    if not points_at(link, target):
        with contextlib.suppress(OSError):
            if is_reparse_point(link):
                os.rmdir(link)
        raise OSError(f"{link} does not lead to {target}")
    return link


def forget_dangling(root: Path) -> None:
    """Junctions to folders that are gone: a data directory that was moved
    or deleted, an install that was removed. Nothing else takes them away,
    and each start is a chance to."""
    with contextlib.suppress(OSError):
        for entry in list(os.scandir(root)):
            link = Path(entry.path)
            if is_reparse_point(link) and not os.path.exists(link):
                with contextlib.suppress(OSError):
                    os.rmdir(link)


def points_at(link: Path, target: Path) -> bool:
    if not is_reparse_point(link):
        return False
    try:
        return os.path.samefile(link, target)
    except OSError:
        return False


def is_reparse_point(path: Path) -> bool:
    attributes = getattr(os.lstat(path), "st_file_attributes", 0)
    return bool(attributes & stat.FILE_ATTRIBUTE_REPARSE_POINT)


def create_junction(target: Path, link: Path) -> None:
    """Through CPython's own helper where it has one (private, and there
    since 3.5), otherwise the command prompt's `mklink /J`."""
    import _winapi

    make = getattr(_winapi, "CreateJunction", None)
    if make:
        make(str(target), str(link))
        return
    result = subprocess.run(
        ["cmd", "/c", "mklink", "/J", str(link), str(target)],
        stdin=subprocess.DEVNULL,
        capture_output=True,
        creationflags=CREATE_NO_WINDOW,
    )
    if result.returncode != 0:
        raise OSError(
            result.stderr.decode(errors="replace").strip() or "mklink /J failed"
        )


def short_name(path: str) -> str:
    """The 8.3 spelling of an existing path; the path itself when the volume
    keeps no short names."""
    import ctypes

    buffer = ctypes.create_unicode_buffer(32768)
    length = _kernel32().GetShortPathNameW(path, buffer, len(buffer))
    return buffer.value if 0 < length < len(buffer) else path


def create_closed(path: Path, user_sid: str) -> bool:
    """Create the directory with access for its user, the system and
    administrators only, not inherited from its parent. False when something
    by that name is already there."""
    import ctypes
    from ctypes import wintypes

    class SecurityAttributes(ctypes.Structure):
        _fields_ = [
            ("nLength", wintypes.DWORD),
            ("lpSecurityDescriptor", wintypes.LPVOID),
            ("bInheritHandle", wintypes.BOOL),
        ]

    full_access = "(A;OICI;FA;;;{})"
    sddl = "D:P" + "".join(full_access.format(sid) for sid in (user_sid, "SY", "BA"))
    descriptor = wintypes.LPVOID()
    sddl_revision_1 = 1
    if not _advapi32().ConvertStringSecurityDescriptorToSecurityDescriptorW(
        sddl, sddl_revision_1, ctypes.byref(descriptor), None
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        attributes = SecurityAttributes(
            ctypes.sizeof(SecurityAttributes), descriptor, False
        )
        if _kernel32().CreateDirectoryW(str(path), ctypes.byref(attributes)):
            return True
        error = ctypes.get_last_error()
    finally:
        _kernel32().LocalFree(descriptor)
    if error == ERROR_ALREADY_EXISTS:
        return False
    raise ctypes.WinError(error)


def open_as_it_is(path: Path) -> int:
    """A handle to whatever is at `path` itself: a link there is opened, not
    followed. Others may read and write while it is open, but not delete or
    rename."""
    import ctypes

    # Listing is asked for although nothing is listed: Windows only holds
    # others to a handle's sharing when the handle can read, write or delete.
    read_control, read_attributes, list_directory = 0x00020000, 0x0080, 0x0001
    share_read, share_write, open_existing = 0x1, 0x2, 3
    backup_semantics, open_reparse_point = (
        0x02000000,
        0x00200000,
    )  # a directory; no following
    handle = _kernel32().CreateFileW(
        str(path),
        read_control | read_attributes | list_directory,
        share_read | share_write,
        None,
        open_existing,
        backup_semantics | open_reparse_point,
        None,
    )
    if handle is None or handle == ctypes.c_void_p(-1).value:
        raise ctypes.WinError(ctypes.get_last_error())
    return handle


def look(handle: int) -> Seen:
    """Kind, owner and access list of the object a handle is open on."""
    import ctypes
    from ctypes import wintypes

    class FileInformation(ctypes.Structure):  # BY_HANDLE_FILE_INFORMATION
        _fields_ = [("dwFileAttributes", wintypes.DWORD), ("rest", wintypes.DWORD * 12)]

    information = FileInformation()
    if not _kernel32().GetFileInformationByHandle(handle, ctypes.byref(information)):
        raise ctypes.WinError(ctypes.get_last_error())
    owner, dacl, descriptor = wintypes.LPVOID(), wintypes.LPVOID(), wintypes.LPVOID()
    file_object, owner_and_dacl = 1, 0x1 | 0x4
    error = _advapi32().GetSecurityInfo(
        handle,
        file_object,
        owner_and_dacl,
        ctypes.byref(owner),
        None,
        ctypes.byref(dacl),
        None,
        ctypes.byref(descriptor),
    )
    if error:
        raise ctypes.WinError(error)
    try:
        access = _entries(dacl) if dacl else None
        return Seen(information.dwFileAttributes, _sid_text(owner), access)
    finally:
        _kernel32().LocalFree(descriptor)


def _entries(dacl) -> tuple[Ace, ...]:
    import ctypes
    from ctypes import wintypes

    class AclSize(ctypes.Structure):
        _fields_ = [
            ("count", wintypes.DWORD),
            ("used", wintypes.DWORD),
            ("free", wintypes.DWORD),
        ]

    class AceStart(ctypes.Structure):  # ACE_HEADER, then the mask; the SID follows
        _fields_ = [
            ("kind", ctypes.c_ubyte),
            ("flags", ctypes.c_ubyte),
            ("size", wintypes.WORD),
            ("mask", wintypes.DWORD),
        ]

    advapi32 = _advapi32()
    size = AclSize()
    acl_size_information = 2
    if not advapi32.GetAclInformation(
        dacl, ctypes.byref(size), ctypes.sizeof(size), acl_size_information
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    found = []
    for index in range(size.count):
        ace = wintypes.LPVOID()
        if not advapi32.GetAce(dacl, index, ctypes.byref(ace)):
            raise ctypes.WinError(ctypes.get_last_error())
        start = ctypes.cast(ace, ctypes.POINTER(AceStart)).contents
        if start.kind in (ACCESS_ALLOWED, ACCESS_DENIED):
            sid = _sid_text(wintypes.LPVOID((ace.value or 0) + ctypes.sizeof(AceStart)))
        else:
            sid = ""  # laid out differently; refusal() turns the kind away
        found.append(Ace(start.kind, start.mask, sid))
    return tuple(found)


def current_user_sid() -> str:
    import ctypes
    from ctypes import wintypes

    kernel32, advapi32 = _kernel32(), _advapi32()
    token = wintypes.HANDLE()
    token_query, token_user = 0x0008, 1
    if not advapi32.OpenProcessToken(
        kernel32.GetCurrentProcess(), token_query, ctypes.byref(token)
    ):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        needed = wintypes.DWORD()
        advapi32.GetTokenInformation(token, token_user, None, 0, ctypes.byref(needed))
        buffer = ctypes.create_string_buffer(needed.value)
        if not advapi32.GetTokenInformation(
            token, token_user, buffer, needed, ctypes.byref(needed)
        ):
            raise ctypes.WinError(ctypes.get_last_error())
        # TOKEN_USER begins with a pointer to the SID.
        return _sid_text(ctypes.cast(buffer, ctypes.POINTER(wintypes.LPVOID)).contents)
    finally:
        kernel32.CloseHandle(token)


def _sid_text(sid) -> str:
    import ctypes
    from ctypes import wintypes

    text = wintypes.LPWSTR()
    if not _advapi32().ConvertSidToStringSidW(sid, ctypes.byref(text)):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        return str(text.value)
    finally:
        with contextlib.suppress(OSError):
            _kernel32().LocalFree(ctypes.cast(text, wintypes.LPVOID))


def _kernel32():
    import ctypes
    from ctypes import wintypes

    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.GetShortPathNameW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.LPWSTR,
        wintypes.DWORD,
    )
    kernel32.GetShortPathNameW.restype = wintypes.DWORD
    kernel32.CreateDirectoryW.argtypes = (wintypes.LPCWSTR, wintypes.LPVOID)
    kernel32.CreateDirectoryW.restype = wintypes.BOOL
    kernel32.LocalFree.argtypes = (wintypes.LPVOID,)
    kernel32.LocalFree.restype = wintypes.LPVOID
    kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    kernel32.CloseHandle.argtypes = (wintypes.HANDLE,)
    kernel32.CreateFileW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.LPVOID,
        wintypes.DWORD,
        wintypes.DWORD,
        wintypes.HANDLE,
    )
    kernel32.CreateFileW.restype = wintypes.HANDLE
    kernel32.GetFileInformationByHandle.argtypes = (wintypes.HANDLE, wintypes.LPVOID)
    kernel32.GetFileInformationByHandle.restype = wintypes.BOOL
    return kernel32


def _advapi32():
    import ctypes
    from ctypes import wintypes

    advapi32 = ctypes.WinDLL("advapi32", use_last_error=True)
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.argtypes = (
        wintypes.LPCWSTR,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        wintypes.LPVOID,
    )
    advapi32.ConvertStringSecurityDescriptorToSecurityDescriptorW.restype = (
        wintypes.BOOL
    )
    advapi32.GetSecurityInfo.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.LPVOID),
    )
    advapi32.GetSecurityInfo.restype = wintypes.DWORD
    advapi32.GetAclInformation.argtypes = (
        wintypes.LPVOID,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.c_int,
    )
    advapi32.GetAclInformation.restype = wintypes.BOOL
    advapi32.GetAce.argtypes = (
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.LPVOID),
    )
    advapi32.GetAce.restype = wintypes.BOOL
    advapi32.OpenProcessToken.argtypes = (
        wintypes.HANDLE,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.HANDLE),
    )
    advapi32.OpenProcessToken.restype = wintypes.BOOL
    advapi32.GetTokenInformation.argtypes = (
        wintypes.HANDLE,
        ctypes.c_int,
        wintypes.LPVOID,
        wintypes.DWORD,
        ctypes.POINTER(wintypes.DWORD),
    )
    advapi32.GetTokenInformation.restype = wintypes.BOOL
    advapi32.ConvertSidToStringSidW.argtypes = (
        wintypes.LPVOID,
        ctypes.POINTER(wintypes.LPWSTR),
    )
    advapi32.ConvertSidToStringSidW.restype = wintypes.BOOL
    return advapi32
