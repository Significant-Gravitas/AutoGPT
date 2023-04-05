import os

import ntsecuritycon
import win32api
import win32con
import win32file
import win32security
from win32security import (
    ACL_REVISION_DS,
    CONTAINER_INHERIT_ACE,
    DACL_SECURITY_INFORMATION,
    GROUP_SECURITY_INFORMATION,
    OBJECT_INHERIT_ACE,
    OWNER_SECURITY_INFORMATION,
    PROTECTED_DACL_SECURITY_INFORMATION,
    SACL_SECURITY_INFORMATION,
    SE_FILE_OBJECT,
)

## SE_SECURITY_NAME needed to access SACL, SE_RESTORE_NAME needed to change owner to someone other than yourself
new_privs = (
    (
        win32security.LookupPrivilegeValue("", ntsecuritycon.SE_SECURITY_NAME),
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
modified_privs = win32security.AdjustTokenPrivileges(th, 0, new_privs)

## look up a few sids that should be available on most systems
my_sid = win32security.GetTokenInformation(th, ntsecuritycon.TokenUser)[0]
pwr_sid = win32security.LookupAccountName("", "Power Users")[0]
admin_sid = win32security.LookupAccountName("", "Administrators")[0]
everyone_sid = win32security.LookupAccountName("", "EveryOne")[0]

## create a dir and set security so Everyone has read permissions, and all files and subdirs inherit its ACLs
temp_dir = win32api.GetTempPath()
dir_name = win32api.GetTempFileName(temp_dir, "sfa")[0]
os.remove(dir_name)
os.mkdir(dir_name)
dir_dacl = win32security.ACL()
dir_dacl.AddAccessAllowedAceEx(
    ACL_REVISION_DS,
    CONTAINER_INHERIT_ACE | OBJECT_INHERIT_ACE,
    win32con.GENERIC_READ,
    everyone_sid,
)
## make sure current user has permissions on dir
dir_dacl.AddAccessAllowedAceEx(
    ACL_REVISION_DS,
    CONTAINER_INHERIT_ACE | OBJECT_INHERIT_ACE,
    win32con.GENERIC_ALL,
    my_sid,
)
## keep dir from inheriting any permissions so it only has ACEs explicitely set here
win32security.SetNamedSecurityInfo(
    dir_name,
    SE_FILE_OBJECT,
    OWNER_SECURITY_INFORMATION
    | GROUP_SECURITY_INFORMATION
    | DACL_SECURITY_INFORMATION
    | PROTECTED_DACL_SECURITY_INFORMATION,
    pwr_sid,
    pwr_sid,
    dir_dacl,
    None,
)

## Create a file in the dir and add some specific permissions to it
fname = win32api.GetTempFileName(dir_name, "sfa")[0]
print(fname)
file_sd = win32security.GetNamedSecurityInfo(
    fname, SE_FILE_OBJECT, DACL_SECURITY_INFORMATION | SACL_SECURITY_INFORMATION
)
file_dacl = file_sd.GetSecurityDescriptorDacl()
file_sacl = file_sd.GetSecurityDescriptorSacl()

if file_dacl is None:
    file_dacl = win32security.ACL()
if file_sacl is None:
    file_sacl = win32security.ACL()

file_dacl.AddAccessDeniedAce(file_dacl.GetAclRevision(), win32con.DELETE, admin_sid)
file_dacl.AddAccessDeniedAce(file_dacl.GetAclRevision(), win32con.DELETE, my_sid)
file_dacl.AddAccessAllowedAce(file_dacl.GetAclRevision(), win32con.GENERIC_ALL, pwr_sid)
file_sacl.AddAuditAccessAce(
    file_dacl.GetAclRevision(), win32con.GENERIC_ALL, my_sid, True, True
)

win32security.SetNamedSecurityInfo(
    fname,
    SE_FILE_OBJECT,
    DACL_SECURITY_INFORMATION | SACL_SECURITY_INFORMATION,
    None,
    None,
    file_dacl,
    file_sacl,
)

win32security.AdjustTokenPrivileges(th, 0, modified_privs)
