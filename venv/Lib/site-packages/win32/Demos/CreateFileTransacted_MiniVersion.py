"""
This demonstrates the creation of miniversions of a file during a transaction.
The FSCTL_TXFS_CREATE_MINIVERSION control code saves any changes to a new
miniversion (effectively a savepoint within a transaction).
"""

import os
import struct

import win32api
import win32con
import win32file
import win32transaction
import winerror
import winioctlcon
from pywin32_testutil import str2bytes  # py3k-friendly helper


def demo():
    """
    Definition of buffer used with FSCTL_TXFS_CREATE_MINIVERSION:
    typedef struct _TXFS_CREATE_MINIVERSION_INFO{
        USHORT StructureVersion;
        USHORT StructureLength;
        ULONG BaseVersion;
        USHORT MiniVersion;}
    """
    buf_fmt = "HHLH0L"  ## buffer size must include struct padding
    buf_size = struct.calcsize(buf_fmt)

    tempdir = win32api.GetTempPath()
    tempfile = win32api.GetTempFileName(tempdir, "cft")[0]
    print("Demonstrating transactions on tempfile", tempfile)
    f = open(tempfile, "w")
    f.write("This is original file.\n")
    f.close()

    trans = win32transaction.CreateTransaction(
        Description="Test creating miniversions of a file"
    )
    hfile = win32file.CreateFileW(
        tempfile,
        win32con.GENERIC_READ | win32con.GENERIC_WRITE,
        win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
        None,
        win32con.OPEN_EXISTING,
        0,
        None,
        Transaction=trans,
    )

    win32file.WriteFile(hfile, str2bytes("This is first miniversion.\n"))
    buf = win32file.DeviceIoControl(
        hfile, winioctlcon.FSCTL_TXFS_CREATE_MINIVERSION, None, buf_size, None
    )
    struct_ver, struct_len, base_ver, ver_1 = struct.unpack(buf_fmt, buf)

    win32file.SetFilePointer(hfile, 0, win32con.FILE_BEGIN)
    win32file.WriteFile(hfile, str2bytes("This is second miniversion!\n"))
    buf = win32file.DeviceIoControl(
        hfile, winioctlcon.FSCTL_TXFS_CREATE_MINIVERSION, None, buf_size, None
    )
    struct_ver, struct_len, base_ver, ver_2 = struct.unpack(buf_fmt, buf)
    hfile.Close()

    ## miniversions can't be opened with write access
    hfile_0 = win32file.CreateFileW(
        tempfile,
        win32con.GENERIC_READ,
        win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
        None,
        win32con.OPEN_EXISTING,
        0,
        None,
        Transaction=trans,
        MiniVersion=base_ver,
    )
    print("version:", base_ver, win32file.ReadFile(hfile_0, 100))
    hfile_0.Close()

    hfile_1 = win32file.CreateFileW(
        tempfile,
        win32con.GENERIC_READ,
        win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
        None,
        win32con.OPEN_EXISTING,
        0,
        None,
        Transaction=trans,
        MiniVersion=ver_1,
    )
    print("version:", ver_1, win32file.ReadFile(hfile_1, 100))
    hfile_1.Close()

    hfile_2 = win32file.CreateFileW(
        tempfile,
        win32con.GENERIC_READ,
        win32con.FILE_SHARE_READ | win32con.FILE_SHARE_WRITE,
        None,
        win32con.OPEN_EXISTING,
        0,
        None,
        Transaction=trans,
        MiniVersion=ver_2,
    )
    print("version:", ver_2, win32file.ReadFile(hfile_2, 100))
    hfile_2.Close()

    ## MiniVersions are destroyed when transaction is committed or rolled back
    win32transaction.CommitTransaction(trans)

    os.unlink(tempfile)


if __name__ == "__main__":
    # When run on CI, this fails with NOT_SUPPORTED, so don't have that cause "failure"
    try:
        demo()
    except win32file.error as e:
        if e.winerror == winerror.ERROR_NOT_SUPPORTED:
            print("These features are not supported by this filesystem.")
        else:
            raise
