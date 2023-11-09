import win32api
import win32con
import win32process
import win32security

## You need SE_RESTORE_NAME to be able to set the owner of a security descriptor to anybody
## other than yourself or your primary group.  Most admin logins don't have it by default, so
## enabling it may fail
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

pid = win32api.GetCurrentProcessId()
ph = win32api.OpenProcess(win32con.PROCESS_ALL_ACCESS, 0, pid)
## PROCESS_ALL_ACCESS does not contain ACCESS_SYSTEM_SECURITY (neccessy to do SACLs)
th = win32security.OpenProcessToken(
    ph, win32security.TOKEN_ALL_ACCESS
)  ##win32con.TOKEN_ADJUST_PRIVILEGES)
old_privs = win32security.AdjustTokenPrivileges(th, 0, new_privs)
my_sid = win32security.GetTokenInformation(th, win32security.TokenUser)[0]
pwr_sid = win32security.LookupAccountName("", "Power Users")[0]
## reopen process with ACCESS_SYSTEM_SECURITY now that sufficent privs are enabled
ph = win32api.OpenProcess(
    win32con.PROCESS_ALL_ACCESS | win32con.ACCESS_SYSTEM_SECURITY, 0, pid
)

sd = win32security.GetSecurityInfo(ph, win32security.SE_KERNEL_OBJECT, all_info)
dacl = sd.GetSecurityDescriptorDacl()
if dacl is None:
    dacl = win32security.ACL()
sacl = sd.GetSecurityDescriptorSacl()
if sacl is None:
    sacl = win32security.ACL()

dacl_ace_cnt = dacl.GetAceCount()
sacl_ace_cnt = sacl.GetAceCount()

dacl.AddAccessAllowedAce(
    dacl.GetAclRevision(), win32con.ACCESS_SYSTEM_SECURITY | win32con.WRITE_DAC, my_sid
)
sacl.AddAuditAccessAce(sacl.GetAclRevision(), win32con.GENERIC_ALL, my_sid, 1, 1)

win32security.SetSecurityInfo(
    ph, win32security.SE_KERNEL_OBJECT, all_info, pwr_sid, pwr_sid, dacl, sacl
)
new_sd = win32security.GetSecurityInfo(ph, win32security.SE_KERNEL_OBJECT, all_info)

if new_sd.GetSecurityDescriptorDacl().GetAceCount() != dacl_ace_cnt + 1:
    print("New dacl doesn" "t contain extra ace ????")
if new_sd.GetSecurityDescriptorSacl().GetAceCount() != sacl_ace_cnt + 1:
    print("New Sacl doesn" "t contain extra ace ????")
if (
    win32security.LookupAccountSid("", new_sd.GetSecurityDescriptorOwner())[0]
    != "Power Users"
):
    print("Owner not successfully set to Power Users !!!!!")
if (
    win32security.LookupAccountSid("", new_sd.GetSecurityDescriptorGroup())[0]
    != "Power Users"
):
    print("Group not successfully set to Power Users !!!!!")

win32security.SetSecurityInfo(
    ph,
    win32security.SE_KERNEL_OBJECT,
    win32security.SACL_SECURITY_INFORMATION,
    None,
    None,
    None,
    None,
)
new_sd_1 = win32security.GetSecurityInfo(
    ph, win32security.SE_KERNEL_OBJECT, win32security.SACL_SECURITY_INFORMATION
)
if new_sd_1.GetSecurityDescriptorSacl() is not None:
    print("Unable to set Sacl to NULL !!!!!!!!")
