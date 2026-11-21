"""Typed Windows ABI boundary. Imported only on Windows.

NtCreateFile receives one component relative to an already opened directory;
retention uses an empty name to reopen that same directory object.
No OBJ_INHERIT, no reparsing, no destructive create disposition. Callers own
every returned handle until they explicitly transfer it to the CRT.
"""

import ctypes as c
import errno
import re
import sys
from ctypes import wintypes as w

if sys.platform != "win32":
    raise ImportError("Native Windows handles require Windows")

kernel = c.WinDLL("kernel32", use_last_error=True)
ntdll = c.WinDLL("ntdll")


class UnicodeString(c.Structure):
    _fields_ = [("Length", w.USHORT), ("MaximumLength", w.USHORT), ("Buffer", w.LPWSTR)]


class ObjectAttributes(c.Structure):
    _fields_ = [
        ("Length", w.ULONG),
        ("RootDirectory", w.HANDLE),
        ("ObjectName", c.POINTER(UnicodeString)),
        ("Attributes", w.ULONG),
        ("SecurityDescriptor", w.LPVOID),
        ("SecurityQualityOfService", w.LPVOID),
    ]


class IoStatus(c.Structure):
    _fields_ = [("Status", c.c_void_p), ("Information", c.c_size_t)]


class AttributeTag(c.Structure):
    _fields_ = [("attributes", w.DWORD), ("tag", w.DWORD)]


def bind(dll, name, result, *args):
    function = getattr(dll, name)
    function.restype = result
    function.argtypes = args
    return function


close = bind(kernel, "CloseHandle", w.BOOL, w.HANDLE)
local_free = bind(kernel, "LocalFree", w.HANDLE, w.HANDLE)
create_file = bind(
    kernel,
    "CreateFileW",
    w.HANDLE,
    w.LPCWSTR,
    w.DWORD,
    w.DWORD,
    w.LPVOID,
    w.DWORD,
    w.DWORD,
    w.HANDLE,
)
get_type = bind(kernel, "GetFileType", w.DWORD, w.HANDLE)
get_final_path = bind(
    kernel, "GetFinalPathNameByHandleW", w.DWORD, w.HANDLE, w.LPWSTR, w.DWORD, w.DWORD
)
get_info = bind(
    kernel, "GetFileInformationByHandleEx", w.BOOL, w.HANDLE, c.c_int, w.LPVOID, w.DWORD
)
set_info = bind(
    kernel, "SetFileInformationByHandle", w.BOOL, w.HANDLE, c.c_int, w.LPVOID, w.DWORD
)
nt_create = bind(
    ntdll,
    "NtCreateFile",
    w.LONG,
    c.POINTER(w.HANDLE),
    w.ULONG,
    c.POINTER(ObjectAttributes),
    c.POINTER(IoStatus),
    w.LPVOID,
    w.ULONG,
    w.ULONG,
    w.ULONG,
    w.ULONG,
    w.LPVOID,
    w.ULONG,
)
dos_error = bind(ntdll, "RtlNtStatusToDosError", w.ULONG, w.LONG)

DIRECTORY = 0x10
REPARSE = 0x400
READ_CONTROL = 0x20000
READ_ATTRIBUTES = 0x80
SYNCHRONIZE = 0x100000
DELETE = 0x10000
FILE_OPEN = 1
FILE_CREATE = 2
FILE_OPEN_IF = 3
DIRECTORY_ACCESS = READ_ATTRIBUTES | SYNCHRONIZE | 0x20


def win_error(code: int, path: str) -> OSError:
    return OSError(0, c.FormatError(code), path, code)


def attributes(handle: int) -> int:
    result = AttributeTag()
    if not get_info(handle, 9, c.byref(result), c.sizeof(result)):
        raise c.WinError(c.get_last_error())
    if get_type(handle) != 1:
        raise OSError(errno.EINVAL, "Not a disk file")
    return result.attributes


def inspect(handle: int, path: str, *, directory: bool) -> int:
    attrs = attributes(handle)
    if attrs & REPARSE:
        raise OSError(errno.ELOOP, "Reparse points are not allowed", path)
    if bool(attrs & DIRECTORY) != directory:
        if directory:
            raise NotADirectoryError(errno.ENOTDIR, "Not a directory", path)
        raise IsADirectoryError(errno.EISDIR, "Is a directory", path)
    return handle


def open_root(drive: str, *, security: bool = False) -> int:
    handle = create_file(
        drive + "\\",
        DIRECTORY_ACCESS | (READ_CONTROL if security else 0),
        1,
        None,
        3,
        0x02200000,
        None,
    )
    if handle == c.c_void_p(-1).value:
        raise win_error(c.get_last_error(), drive)
    try:
        final_path = c.create_unicode_buffer(64)
        length = get_final_path(handle, final_path, len(final_path), 1)
        if not length:
            raise win_error(c.get_last_error(), drive)
        if length >= len(final_path) or not re.fullmatch(
            r"\\\\\?\\Volume\{[0-9a-fA-F-]{36}\}\\", final_path.value
        ):
            raise OSError(errno.ENOTSUP, "Drive must name a local volume root", drive)
        return inspect(handle, drive, directory=True)
    except BaseException:
        close(handle)
        raise


def relative(
    parent: int,
    name: str,
    access: int,
    disposition: int,
    *,
    directory: bool = False,
    allow_link: bool = False,
    descriptor=None,
    share_write: bool = False,
) -> int:
    buffer = c.create_unicode_buffer(name)
    length = len(name.encode("utf-16-le"))
    string = UnicodeString(length, length + 2, c.cast(buffer, w.LPWSTR))
    # OBJ_DONT_REPARSE refuses traversal; omit only to open a final link itself.
    obj = ObjectAttributes(
        c.sizeof(ObjectAttributes),
        parent,
        c.pointer(string),
        0x40 | (0 if allow_link else 0x1000),
        descriptor,
        None,
    )
    handle = w.HANDLE()
    status = IoStatus()
    options = 0x00200000 | 0x20 | (1 if directory else 0)
    code = nt_create(
        c.byref(handle),
        access | SYNCHRONIZE,
        c.byref(obj),
        c.byref(status),
        None,
        0,
        (3 if share_write else 1) if directory else 7,
        disposition,
        options,
        None,
        0,
    )
    if code < 0:
        raise win_error(dos_error(code), name)
    value = handle.value
    if value is None:
        raise OSError(errno.EIO, "Native open returned a null handle")
    return value


def retain_directory(handle: int, path: str) -> int:
    """Reopen a validated directory object without releasing its identity pin."""
    identity = c.create_string_buffer(24)
    if not get_info(handle, 18, identity, len(identity)):
        raise win_error(c.get_last_error(), path)
    retained = relative(
        handle,
        "",
        DIRECTORY_ACCESS | READ_CONTROL,
        FILE_OPEN,
        directory=True,
        share_write=True,
    )
    try:
        inspect(retained, path, directory=True)
        current = c.create_string_buffer(24)
        if not get_info(retained, 18, current, len(current)):
            raise win_error(c.get_last_error(), path)
        if current.raw != identity.raw:
            raise PermissionError("Private directory identity changed")
        return retained
    except BaseException:
        close(retained)
        raise
