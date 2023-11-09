import os

import ntsecuritycon
import win32api
import win32con
import win32security
import winnt

temp_dir = win32api.GetTempPath()
fname = win32api.GetTempFileName(temp_dir, "rsk")[0]
print(fname)
## file can't exist
os.remove(fname)

## enable backup and restore privs
required_privs = (
    (
        win32security.LookupPrivilegeValue("", ntsecuritycon.SE_BACKUP_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", ntsecuritycon.SE_RESTORE_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
)
ph = win32api.GetCurrentProcess()
th = win32security.OpenProcessToken(
    ph, win32con.TOKEN_READ | win32con.TOKEN_ADJUST_PRIVILEGES
)
adjusted_privs = win32security.AdjustTokenPrivileges(th, 0, required_privs)

try:
    sa = win32security.SECURITY_ATTRIBUTES()
    my_sid = win32security.GetTokenInformation(th, ntsecuritycon.TokenUser)[0]
    sa.SECURITY_DESCRIPTOR.SetSecurityDescriptorOwner(my_sid, 0)

    k, disp = win32api.RegCreateKeyEx(
        win32con.HKEY_CURRENT_USER,
        "Python test key",
        SecurityAttributes=sa,
        samDesired=win32con.KEY_ALL_ACCESS,
        Class="some class",
        Options=0,
    )
    win32api.RegSetValue(k, None, win32con.REG_SZ, "Default value for python test key")

    subk, disp = win32api.RegCreateKeyEx(
        k,
        "python test subkey",
        SecurityAttributes=sa,
        samDesired=win32con.KEY_ALL_ACCESS,
        Class="some other class",
        Options=0,
    )
    win32api.RegSetValue(subk, None, win32con.REG_SZ, "Default value for subkey")

    win32api.RegSaveKeyEx(
        k, fname, Flags=winnt.REG_STANDARD_FORMAT, SecurityAttributes=sa
    )

    restored_key, disp = win32api.RegCreateKeyEx(
        win32con.HKEY_CURRENT_USER,
        "Python test key(restored)",
        SecurityAttributes=sa,
        samDesired=win32con.KEY_ALL_ACCESS,
        Class="restored class",
        Options=0,
    )
    win32api.RegRestoreKey(restored_key, fname)
finally:
    win32security.AdjustTokenPrivileges(th, 0, adjusted_privs)
