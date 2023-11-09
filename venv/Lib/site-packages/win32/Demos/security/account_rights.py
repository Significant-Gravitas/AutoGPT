import ntsecuritycon
import win32api
import win32con
import win32file
import win32security
from security_enums import ACCESS_MODE, ACE_FLAGS, TRUSTEE_FORM, TRUSTEE_TYPE

new_privs = (
    (
        win32security.LookupPrivilegeValue("", ntsecuritycon.SE_SECURITY_NAME),
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

policy_handle = win32security.GetPolicyHandle("", win32security.POLICY_ALL_ACCESS)
tmp_sid = win32security.LookupAccountName("", "tmp")[0]

privs = [
    ntsecuritycon.SE_DEBUG_NAME,
    ntsecuritycon.SE_TCB_NAME,
    ntsecuritycon.SE_RESTORE_NAME,
    ntsecuritycon.SE_REMOTE_SHUTDOWN_NAME,
]
win32security.LsaAddAccountRights(policy_handle, tmp_sid, privs)

privlist = win32security.LsaEnumerateAccountRights(policy_handle, tmp_sid)
for priv in privlist:
    print(priv)

privs = [ntsecuritycon.SE_DEBUG_NAME, ntsecuritycon.SE_TCB_NAME]
win32security.LsaRemoveAccountRights(policy_handle, tmp_sid, 0, privs)

privlist = win32security.LsaEnumerateAccountRights(policy_handle, tmp_sid)
for priv in privlist:
    print(priv)

win32security.LsaClose(policy_handle)
