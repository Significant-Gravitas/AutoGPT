import win32api
import win32con
import win32process
import win32security

new_privs = (
    (
        win32security.LookupPrivilegeValue("", win32security.SE_SECURITY_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", win32security.SE_TCB_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", win32security.SE_SHUTDOWN_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", win32security.SE_RESTORE_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", win32security.SE_TAKE_OWNERSHIP_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", win32security.SE_CREATE_PERMANENT_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", win32security.SE_ENABLE_DELEGATION_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", win32security.SE_CHANGE_NOTIFY_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", win32security.SE_DEBUG_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue(
            "", win32security.SE_PROF_SINGLE_PROCESS_NAME
        ),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", win32security.SE_SYSTEM_PROFILE_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
    (
        win32security.LookupPrivilegeValue("", win32security.SE_LOCK_MEMORY_NAME),
        win32con.SE_PRIVILEGE_ENABLED,
    ),
)

all_info = (
    win32security.OWNER_SECURITY_INFORMATION
    | win32security.GROUP_SECURITY_INFORMATION
    | win32security.DACL_SECURITY_INFORMATION
    | win32security.SACL_SECURITY_INFORMATION
)
info = (
    win32security.OWNER_SECURITY_INFORMATION
    | win32security.GROUP_SECURITY_INFORMATION
    | win32security.DACL_SECURITY_INFORMATION
)

ph = win32process.GetCurrentProcess()
th = win32security.OpenProcessToken(
    ph, win32security.TOKEN_ALL_ACCESS
)  ##win32con.TOKEN_ADJUST_PRIVILEGES)
win32security.AdjustTokenPrivileges(th, 0, new_privs)
my_sid = win32security.GetTokenInformation(th, win32security.TokenUser)[0]
pwr_sid = win32security.LookupAccountName("", "Power Users")[0]

h = win32process.GetProcessWindowStation()
sd = win32security.GetUserObjectSecurity(h, info)
dacl = sd.GetSecurityDescriptorDacl()
ace_cnt = dacl.GetAceCount()

dacl.AddAccessAllowedAce(
    dacl.GetAclRevision(), win32con.ACCESS_SYSTEM_SECURITY | win32con.WRITE_DAC, my_sid
)
sd.SetSecurityDescriptorDacl(1, dacl, 0)
sd.SetSecurityDescriptorGroup(pwr_sid, 0)
sd.SetSecurityDescriptorOwner(pwr_sid, 0)

win32security.SetUserObjectSecurity(h, info, sd)
new_sd = win32security.GetUserObjectSecurity(h, info)
assert (
    new_sd.GetSecurityDescriptorDacl().GetAceCount() == ace_cnt + 1
), "Did not add an ace to the Dacl !!!!!!"
assert (
    win32security.LookupAccountSid("", new_sd.GetSecurityDescriptorOwner())[0]
    == "Power Users"
), "Owner not successfully set to Power Users !!!!!"
assert (
    win32security.LookupAccountSid("", new_sd.GetSecurityDescriptorGroup())[0]
    == "Power Users"
), "Group not successfully set to Power Users !!!!!"
