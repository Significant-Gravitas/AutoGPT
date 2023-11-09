import ntsecuritycon
import win32api
import win32file
import win32security

policy_handle = win32security.GetPolicyHandle("rupole", win32security.POLICY_ALL_ACCESS)

event_audit_info = win32security.LsaQueryInformationPolicy(
    policy_handle, win32security.PolicyAuditEventsInformation
)
print(event_audit_info)

new_audit_info = list(event_audit_info[1])
new_audit_info[win32security.AuditCategoryPolicyChange] = (
    win32security.POLICY_AUDIT_EVENT_SUCCESS | win32security.POLICY_AUDIT_EVENT_FAILURE
)
new_audit_info[win32security.AuditCategoryAccountLogon] = (
    win32security.POLICY_AUDIT_EVENT_SUCCESS | win32security.POLICY_AUDIT_EVENT_FAILURE
)
new_audit_info[win32security.AuditCategoryLogon] = (
    win32security.POLICY_AUDIT_EVENT_SUCCESS | win32security.POLICY_AUDIT_EVENT_FAILURE
)

win32security.LsaSetInformationPolicy(
    policy_handle, win32security.PolicyAuditEventsInformation, (1, new_audit_info)
)

win32security.LsaClose(policy_handle)
