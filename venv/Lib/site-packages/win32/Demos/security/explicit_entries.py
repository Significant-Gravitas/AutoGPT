import os

import ntsecuritycon
import win32api
import win32con
import win32file
import win32security
from security_enums import ACCESS_MODE, ACE_FLAGS, TRUSTEE_FORM, TRUSTEE_TYPE

fname = os.path.join(win32api.GetTempPath(), "win32security_test.txt")
f = open(fname, "w")
f.write("Hello from Python\n")
f.close()
print("Testing on file", fname)

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
    ph, win32security.TOKEN_ALL_ACCESS
)  ##win32con.TOKEN_ADJUST_PRIVILEGES)
win32security.AdjustTokenPrivileges(th, 0, new_privs)

all_security_info = (
    win32security.OWNER_SECURITY_INFORMATION
    | win32security.GROUP_SECURITY_INFORMATION
    | win32security.DACL_SECURITY_INFORMATION
    | win32security.SACL_SECURITY_INFORMATION
)

sd = win32security.GetFileSecurity(fname, all_security_info)

old_sacl = sd.GetSecurityDescriptorSacl()
if old_sacl == None:
    old_sacl = win32security.ACL()
old_dacl = sd.GetSecurityDescriptorDacl()
if old_dacl == None:
    old_dacl = win32security.ACL()

my_sid = win32security.GetTokenInformation(th, ntsecuritycon.TokenUser)[0]
tmp_sid = win32security.LookupAccountName("", "tmp")[0]
pwr_sid = win32security.LookupAccountName("", "Power Users")[0]


## MultipleTrustee,MultipleTrusteeOperation,TrusteeForm,TrusteeType,Identifier
## first two are ignored
my_trustee = {}
my_trustee["MultipleTrustee"] = None
my_trustee["MultipleTrusteeOperation"] = 0
my_trustee["TrusteeForm"] = TRUSTEE_FORM.TRUSTEE_IS_SID
my_trustee["TrusteeType"] = TRUSTEE_TYPE.TRUSTEE_IS_USER
my_trustee["Identifier"] = my_sid

tmp_trustee = {}
tmp_trustee["MultipleTrustee"] = None
tmp_trustee["MultipleTrusteeOperation"] = 0
tmp_trustee["TrusteeForm"] = TRUSTEE_FORM.TRUSTEE_IS_NAME
tmp_trustee["TrusteeType"] = TRUSTEE_TYPE.TRUSTEE_IS_USER
tmp_trustee["Identifier"] = "rupole\\tmp"

pwr_trustee = {}
pwr_trustee["MultipleTrustee"] = None
pwr_trustee["MultipleTrusteeOperation"] = 0
pwr_trustee["TrusteeForm"] = TRUSTEE_FORM.TRUSTEE_IS_SID
pwr_trustee["TrusteeType"] = TRUSTEE_TYPE.TRUSTEE_IS_USER
pwr_trustee["Identifier"] = pwr_sid

expl_list = []
expl_list.append(
    {
        "Trustee": my_trustee,
        "Inheritance": ACE_FLAGS.NO_INHERITANCE,
        "AccessMode": ACCESS_MODE.SET_AUDIT_SUCCESS,  ##|ACCESS_MODE.SET_AUDIT_FAILURE,
        "AccessPermissions": win32con.GENERIC_ALL,
    }
)

expl_list.append(
    {
        "Trustee": my_trustee,
        "Inheritance": ACE_FLAGS.NO_INHERITANCE,
        "AccessMode": ACCESS_MODE.SET_AUDIT_FAILURE,
        "AccessPermissions": win32con.GENERIC_ALL,
    }
)

expl_list.append(
    {
        "Trustee": tmp_trustee,
        "Inheritance": ACE_FLAGS.NO_INHERITANCE,
        "AccessMode": ACCESS_MODE.SET_AUDIT_SUCCESS,
        "AccessPermissions": win32con.GENERIC_ALL,
    }
)

expl_list.append(
    {
        "Trustee": tmp_trustee,
        "Inheritance": ACE_FLAGS.NO_INHERITANCE,
        "AccessMode": ACCESS_MODE.SET_AUDIT_FAILURE,
        "AccessPermissions": win32con.GENERIC_ALL,
    }
)
old_sacl.SetEntriesInAcl(expl_list)

expl_list = []
expl_list.append(
    {
        "Trustee": tmp_trustee,
        "Inheritance": ACE_FLAGS.NO_INHERITANCE,
        "AccessMode": ACCESS_MODE.DENY_ACCESS,
        "AccessPermissions": win32con.DELETE,
    }
)

expl_list.append(
    {
        "Trustee": tmp_trustee,
        "Inheritance": ACE_FLAGS.NO_INHERITANCE,
        "AccessMode": ACCESS_MODE.GRANT_ACCESS,
        "AccessPermissions": win32con.WRITE_OWNER,
    }
)
expl_list.append(
    {
        "Trustee": pwr_trustee,
        "Inheritance": ACE_FLAGS.NO_INHERITANCE,
        "AccessMode": ACCESS_MODE.GRANT_ACCESS,
        "AccessPermissions": win32con.GENERIC_READ,
    }
)
expl_list.append(
    {
        "Trustee": my_trustee,
        "Inheritance": ACE_FLAGS.NO_INHERITANCE,
        "AccessMode": ACCESS_MODE.GRANT_ACCESS,
        "AccessPermissions": win32con.GENERIC_ALL,
    }
)

old_dacl.SetEntriesInAcl(expl_list)
sd.SetSecurityDescriptorSacl(1, old_sacl, 1)
sd.SetSecurityDescriptorDacl(1, old_dacl, 1)
sd.SetSecurityDescriptorOwner(pwr_sid, 1)

win32security.SetFileSecurity(fname, all_security_info, sd)
