fname = "h:\\tmp.reg"

import os

import ntsecuritycon
import pywintypes
import win32api
import win32con
import win32security

## regsave will not overwrite a file
if os.path.isfile(fname):
    os.remove(fname)

new_privs = (
    (
        win32security.LookupPrivilegeValue("", ntsecuritycon.SE_SECURITY_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", ntsecuritycon.SE_TCB_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
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
    ph, win32security.TOKEN_ALL_ACCESS | win32con.TOKEN_ADJUST_PRIVILEGES
)
win32security.AdjustTokenPrivileges(th, 0, new_privs)
my_sid = win32security.GetTokenInformation(th, ntsecuritycon.TokenUser)[0]

hklm = win32api.RegOpenKey(
    win32con.HKEY_LOCAL_MACHINE, None, 0, win32con.KEY_ALL_ACCESS
)
skey = win32api.RegOpenKey(hklm, "SYSTEM", 0, win32con.KEY_ALL_ACCESS)

sa = pywintypes.SECURITY_ATTRIBUTES()
sd = pywintypes.SECURITY_DESCRIPTOR()
sa.SECURITY_DESCRIPTOR = sd
acl = pywintypes.ACL()

pwr_sid = win32security.LookupAccountName("", "Power Users")[0]
acl.AddAccessAllowedAce(
    win32con.ACL_REVISION,
    win32con.GENERIC_READ | win32con.ACCESS_SYSTEM_SECURITY,
    my_sid,
)
sd.SetSecurityDescriptorDacl(1, acl, 0)
sd.SetSecurityDescriptorOwner(pwr_sid, 0)
sa.bInheritHandle = 1
assert sa.SECURITY_DESCRIPTOR is sd

win32api.RegSaveKey(skey, fname, sa)
