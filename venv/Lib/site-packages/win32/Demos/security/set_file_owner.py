fname = r"h:\tmp.txt"

import ntsecuritycon
import win32api
import win32con
import win32file
import win32security

new_privs = (
    (
        win32security.LookupPrivilegeValue("", ntsecuritycon.SE_SECURITY_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", ntsecuritycon.SE_SHUTDOWN_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", ntsecuritycon.SE_TCB_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", ntsecuritycon.SE_RESTORE_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", ntsecuritycon.SE_TAKE_OWNERSHIP_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", ntsecuritycon.SE_CREATE_PERMANENT_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", "SeEnableDelegationPrivilege"),
        win32con.SE_PRIVILEGE_ENABLED,
    ),  ##doesn't seem to be in ntsecuritycon.py ?
)

ph = win32api.GetCurrentProcess()
th = win32security.OpenProcessToken(
    ph, win32security.TOKEN_ALL_ACCESS | win32con.TOKEN_ADJUST_PRIVILEGES
)
win32security.AdjustTokenPrivileges(th, 0, new_privs)

all_security_info = (
    win32security.OWNER_SECURITY_INFORMATION
    | win32security.GROUP_SECURITY_INFORMATION
    | win32security.DACL_SECURITY_INFORMATION
    | win32security.SACL_SECURITY_INFORMATION
)

sd = win32security.GetFileSecurity(fname, all_security_info)
old_dacl = sd.GetSecurityDescriptorDacl()
old_sacl = sd.GetSecurityDescriptorSacl()
old_group = sd.GetSecurityDescriptorGroup()

new_sd = win32security.SECURITY_DESCRIPTOR()
print(
    "relative, valid, size: ",
    new_sd.IsSelfRelative(),
    new_sd.IsValid(),
    new_sd.GetLength(),
)

my_sid = win32security.GetTokenInformation(th, ntsecuritycon.TokenUser)[0]
tmp_sid = win32security.LookupAccountName("", "tmp")[0]

new_sd.SetSecurityDescriptorSacl(1, old_sacl, 1)
new_sd.SetSecurityDescriptorDacl(1, old_dacl, 1)
new_sd.SetSecurityDescriptorOwner(tmp_sid, 0)
new_sd.SetSecurityDescriptorGroup(old_group, 0)

win32security.SetFileSecurity(fname, all_security_info, new_sd)
