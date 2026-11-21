"""Atomic private directory creation and handle-based ACL validation.

Trust TokenUser, SYSTEM and administrators, never arbitrary token groups.
Parents may be readable but cannot permit outsider replacement or reparse
mutation. The private leaf permits no outsider access, including inheritance.
Existing security descriptors are only read, never repaired.

Ancestor ACLs must exclude outsider mutation even while handles are pinned:
write-attributes access can change reparse data despite denying write sharing.
The leaf must have a protected DACL to prevent future inherited grants.
Lifetime handles permit child-file renames after the complete chain is checked.
They deny directory deletion and replace the original pins without an open gap.
"""

import ctypes as c
import errno
import sys
from contextlib import ExitStack, contextmanager
from ctypes import wintypes as w

from backend.copilot import windows_handles as wh

if sys.platform == "win32":
    from backend.copilot.host_fs_windows import parents
else:
    raise ImportError("Native Windows private directories require Windows")

advapi = c.WinDLL("advapi32", use_last_error=True)
open_token = wh.bind(
    advapi, "OpenProcessToken", w.BOOL, w.HANDLE, w.DWORD, c.POINTER(w.HANDLE)
)
get_token = wh.bind(
    advapi,
    "GetTokenInformation",
    w.BOOL,
    w.HANDLE,
    c.c_int,
    w.LPVOID,
    w.DWORD,
    c.POINTER(w.DWORD),
)
sid_string = wh.bind(
    advapi, "ConvertSidToStringSidW", w.BOOL, w.LPVOID, c.POINTER(w.LPWSTR)
)
convert_sd = wh.bind(
    advapi,
    "ConvertStringSecurityDescriptorToSecurityDescriptorW",
    w.BOOL,
    w.LPCWSTR,
    w.DWORD,
    c.POINTER(w.LPVOID),
    c.POINTER(w.ULONG),
)
get_security = wh.bind(
    advapi,
    "GetSecurityInfo",
    w.DWORD,
    w.HANDLE,
    c.c_int,
    w.DWORD,
    c.POINTER(w.LPVOID),
    c.POINTER(w.LPVOID),
    c.POINTER(w.LPVOID),
    c.POINTER(w.LPVOID),
    c.POINTER(w.LPVOID),
)
get_ace = wh.bind(advapi, "GetAce", w.BOOL, w.LPVOID, w.DWORD, c.POINTER(w.LPVOID))
get_control = wh.bind(
    advapi,
    "GetSecurityDescriptorControl",
    w.BOOL,
    w.LPVOID,
    c.POINTER(w.WORD),
    c.POINTER(w.DWORD),
)

SYSTEM = "S-1-5-18"
ADMINISTRATORS = "S-1-5-32-544"
# Windows servicing owns system roots on some installs. This service SID is
# trusted for ancestors only, never as the private directory's owner/reader.
TRUSTED_INSTALLER = "S-1-5-80-956008885-3418522649-1831038044-1853292631-2271478464"
PARENT_MUTATION = (
    0x2 | 0x10 | 0x40 | 0x100 | 0x10000 | 0x40000 | 0x80000 | 0x10000000 | 0x40000000
)


class Acl(c.Structure):
    _fields_ = [
        ("revision", w.BYTE),
        ("reserved", w.BYTE),
        ("size", w.WORD),
        ("count", w.WORD),
        ("reserved2", w.WORD),
    ]


class Ace(c.Structure):
    _fields_ = [
        ("kind", w.BYTE),
        ("flags", w.BYTE),
        ("size", w.WORD),
        ("mask", w.DWORD),
    ]


def ensure_private_directory(path: str, *, lifetime: ExitStack | None = None) -> str:
    user = current_user_sid()
    with ExitStack() as held:
        parent, name, ancestors = held.enter_context(parents(path, security=True))
        for handle in ancestors:
            verify_acl(handle, user, private=False)
        with security_descriptor(user) as descriptor:
            handle = wh.relative(
                parent,
                name,
                wh.DIRECTORY_ACCESS | wh.READ_CONTROL,
                wh.FILE_OPEN_IF,
                directory=True,
                descriptor=descriptor,
            )
        held.callback(wh.close, handle)
        wh.inspect(handle, path, directory=True)
        verify_acl(handle, user, private=True)
        if lifetime is not None:
            with ExitStack() as retained:
                for index, original in enumerate([*ancestors, handle]):
                    replacement = wh.retain_directory(original, path)
                    retained.callback(wh.close, replacement)
                    verify_acl(replacement, user, private=index == len(ancestors))
                lifetime.enter_context(retained.pop_all())
    return path


def current_user_sid() -> str:
    token = w.HANDLE()
    if not open_token(w.HANDLE(-1), 8, c.byref(token)):
        raise c.WinError(c.get_last_error())
    try:
        size = w.DWORD()
        get_token(token, 1, None, 0, c.byref(size))
        if c.get_last_error() != 122:
            raise c.WinError(c.get_last_error())
        buffer = c.create_string_buffer(size.value)
        if not get_token(token, 1, buffer, size, c.byref(size)):
            raise c.WinError(c.get_last_error())
        return sid_text(c.cast(buffer, c.POINTER(w.LPVOID))[0])
    finally:
        wh.close(token)


def sid_text(sid) -> str:
    result = w.LPWSTR()
    if not sid_string(sid, c.byref(result)):
        raise c.WinError(c.get_last_error())
    try:
        value = result.value
        if value is None:
            raise OSError(errno.EIO, "Native SID conversion returned no text")
        return value
    finally:
        wh.local_free(c.cast(result, w.LPVOID))


@contextmanager
def security_descriptor(user: str):
    descriptor = w.LPVOID()
    sddl = f"O:{user}D:P(A;OICI;FA;;;{user})(A;OICI;FA;;;SY)(A;OICI;FA;;;BA)"
    if not convert_sd(sddl, 1, c.byref(descriptor), None):
        raise c.WinError(c.get_last_error())
    try:
        yield descriptor
    finally:
        wh.local_free(descriptor)


def verify_acl(handle: int, user: str, *, private: bool) -> None:
    owner, dacl, descriptor = w.LPVOID(), w.LPVOID(), w.LPVOID()
    error = get_security(
        handle, 1, 5, c.byref(owner), None, c.byref(dacl), None, c.byref(descriptor)
    )
    if error:
        raise c.WinError(error)
    try:
        control, revision = w.WORD(), w.DWORD()
        if not get_control(descriptor, c.byref(control), c.byref(revision)):
            raise c.WinError(c.get_last_error())
        if private and not control.value & 0x1000:
            raise PermissionError(errno.EACCES, "Private DACL must reject inheritance")
        trusted = {user, SYSTEM, ADMINISTRATORS}
        if not private:
            trusted.add(TRUSTED_INSTALLER)
        if sid_text(owner) not in ({user} if private else trusted) or not dacl:
            raise PermissionError(errno.EACCES, "Untrusted owner or missing DACL")
        # Python 3.13's 0700 directories grant OWNER RIGHTS. This SID denotes
        # the current owner, whose identity was checked above, not a group.
        trusted.add("S-1-3-4")
        acl = c.cast(dacl, c.POINTER(Acl)).contents
        for index in range(acl.count):
            entry = w.LPVOID()
            if not get_ace(dacl, index, c.byref(entry)):
                raise c.WinError(c.get_last_error())
            address = entry.value
            if address is None:
                raise PermissionError(errno.EACCES, "Missing ACL entry")
            ace = c.cast(entry, c.POINTER(Ace)).contents
            if ace.kind not in (0, 1):
                raise PermissionError(errno.EACCES, "Unsupported ACE type")
            if ace.kind == 1 or (not private and ace.flags & 8):
                continue
            sid = sid_text(address + c.sizeof(Ace))
            if sid not in trusted and ace.mask & (
                0xFFFFFFFF if private else PARENT_MUTATION
            ):
                raise PermissionError(errno.EACCES, "Outsider access in DACL")
    finally:
        wh.local_free(descriptor)
