import ntsecuritycon
import win32api
import win32con
import win32security

new_privs = (
    (
        win32security.LookupPrivilegeValue("", ntsecuritycon.SE_SECURITY_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", ntsecuritycon.SE_TCB_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
)
ph = win32api.GetCurrentProcess()
th = win32security.OpenProcessToken(
    ph, win32security.TOKEN_ALL_ACCESS | win32con.TOKEN_ADJUST_PRIVILEGES
)

win32security.AdjustTokenPrivileges(th, 0, new_privs)
hkey = win32api.RegOpenKey(
    win32con.HKEY_LOCAL_MACHINE, None, 0, win32con.KEY_ALL_ACCESS
)
win32api.RegCreateKey(hkey, "SYSTEM\\NOTMP")
notmpkey = win32api.RegOpenKey(
    hkey, "SYSTEM\\notmp", 0, win32con.ACCESS_SYSTEM_SECURITY
)

tmp_sid = win32security.LookupAccountName("", "tmp")[0]
sacl = win32security.ACL()
sacl.AddAuditAccessAce(win32security.ACL_REVISION, win32con.GENERIC_ALL, tmp_sid, 1, 1)

sd = win32security.SECURITY_DESCRIPTOR()
sd.SetSecurityDescriptorSacl(1, sacl, 1)
win32api.RegSetKeySecurity(notmpkey, win32con.SACL_SECURITY_INFORMATION, sd)
