import ntsecuritycon
import win32api
import win32file
import win32security

policy_handle = win32security.GetPolicyHandle("rupole", win32security.POLICY_ALL_ACCESS)

## mod_nbr, mod_time = win32security.LsaQueryInformationPolicy(policy_handle,win32security.PolicyModificationInformation)
## print mod_nbr, mod_time

(
    domain_name,
    dns_domain_name,
    dns_forest_name,
    domain_guid,
    domain_sid,
) = win32security.LsaQueryInformationPolicy(
    policy_handle, win32security.PolicyDnsDomainInformation
)
print(domain_name, dns_domain_name, dns_forest_name, domain_guid, domain_sid)

event_audit_info = win32security.LsaQueryInformationPolicy(
    policy_handle, win32security.PolicyAuditEventsInformation
)
print(event_audit_info)

domain_name, sid = win32security.LsaQueryInformationPolicy(
    policy_handle, win32security.PolicyPrimaryDomainInformation
)
print(domain_name, sid)

domain_name, sid = win32security.LsaQueryInformationPolicy(
    policy_handle, win32security.PolicyAccountDomainInformation
)
print(domain_name, sid)

server_role = win32security.LsaQueryInformationPolicy(
    policy_handle, win32security.PolicyLsaServerRoleInformation
)
print("server role: ", server_role)

win32security.LsaClose(policy_handle)
