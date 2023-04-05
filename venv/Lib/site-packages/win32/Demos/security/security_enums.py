import ntsecuritycon
import win32security
import winnt


class Enum:
    def __init__(self, *const_names):
        """Accepts variable number of constant names that can be found in either
        win32security, ntsecuritycon, or winnt."""
        for const_name in const_names:
            try:
                const_val = getattr(win32security, const_name)
            except AttributeError:
                try:
                    const_val = getattr(ntsecuritycon, const_name)
                except AttributeError:
                    try:
                        const_val = getattr(winnt, const_name)
                    except AttributeError:
                        raise AttributeError(
                            'Constant "%s" not found in win32security, ntsecuritycon, or winnt.'
                            % const_name
                        )
            setattr(self, const_name, const_val)

    def lookup_name(self, const_val):
        """Looks up the name of a particular value."""
        for k, v in self.__dict__.items():
            if v == const_val:
                return k
        raise AttributeError("Value %s not found in enum" % const_val)

    def lookup_flags(self, flags):
        """Returns the names of all recognized flags in input, and any flags not found in the enum."""
        flag_names = []
        unknown_flags = flags
        for k, v in self.__dict__.items():
            if flags & v == v:
                flag_names.append(k)
                unknown_flags = unknown_flags & ~v
        return flag_names, unknown_flags


TOKEN_INFORMATION_CLASS = Enum(
    "TokenUser",
    "TokenGroups",
    "TokenPrivileges",
    "TokenOwner",
    "TokenPrimaryGroup",
    "TokenDefaultDacl",
    "TokenSource",
    "TokenType",
    "TokenImpersonationLevel",
    "TokenStatistics",
    "TokenRestrictedSids",
    "TokenSessionId",
    "TokenGroupsAndPrivileges",
    "TokenSessionReference",
    "TokenSandBoxInert",
    "TokenAuditPolicy",
    "TokenOrigin",
    "TokenElevationType",
    "TokenLinkedToken",
    "TokenElevation",
    "TokenHasRestrictions",
    "TokenAccessInformation",
    "TokenVirtualizationAllowed",
    "TokenVirtualizationEnabled",
    "TokenIntegrityLevel",
    "TokenUIAccess",
    "TokenMandatoryPolicy",
    "TokenLogonSid",
)

TOKEN_TYPE = Enum("TokenPrimary", "TokenImpersonation")

TOKEN_ELEVATION_TYPE = Enum(
    "TokenElevationTypeDefault", "TokenElevationTypeFull", "TokenElevationTypeLimited"
)

POLICY_AUDIT_EVENT_TYPE = Enum(
    "AuditCategorySystem",
    "AuditCategoryLogon",
    "AuditCategoryObjectAccess",
    "AuditCategoryPrivilegeUse",
    "AuditCategoryDetailedTracking",
    "AuditCategoryPolicyChange",
    "AuditCategoryAccountManagement",
    "AuditCategoryDirectoryServiceAccess",
    "AuditCategoryAccountLogon",
)

POLICY_INFORMATION_CLASS = Enum(
    "PolicyAuditLogInformation",
    "PolicyAuditEventsInformation",
    "PolicyPrimaryDomainInformation",
    "PolicyPdAccountInformation",
    "PolicyAccountDomainInformation",
    "PolicyLsaServerRoleInformation",
    "PolicyReplicaSourceInformation",
    "PolicyDefaultQuotaInformation",
    "PolicyModificationInformation",
    "PolicyAuditFullSetInformation",
    "PolicyAuditFullQueryInformation",
    "PolicyDnsDomainInformation",
)

POLICY_LSA_SERVER_ROLE = Enum("PolicyServerRoleBackup", "PolicyServerRolePrimary")

## access modes for opening a policy handle - this is not a real enum
POLICY_ACCESS_MODES = Enum(
    "POLICY_VIEW_LOCAL_INFORMATION",
    "POLICY_VIEW_AUDIT_INFORMATION",
    "POLICY_GET_PRIVATE_INFORMATION",
    "POLICY_TRUST_ADMIN",
    "POLICY_CREATE_ACCOUNT",
    "POLICY_CREATE_SECRET",
    "POLICY_CREATE_PRIVILEGE",
    "POLICY_SET_DEFAULT_QUOTA_LIMITS",
    "POLICY_SET_AUDIT_REQUIREMENTS",
    "POLICY_AUDIT_LOG_ADMIN",
    "POLICY_SERVER_ADMIN",
    "POLICY_LOOKUP_NAMES",
    "POLICY_NOTIFICATION",
    "POLICY_ALL_ACCESS",
    "POLICY_READ",
    "POLICY_WRITE",
    "POLICY_EXECUTE",
)

## EventAuditingOptions flags - not a real enum
POLICY_AUDIT_EVENT_OPTIONS_FLAGS = Enum(
    "POLICY_AUDIT_EVENT_UNCHANGED",
    "POLICY_AUDIT_EVENT_SUCCESS",
    "POLICY_AUDIT_EVENT_FAILURE",
    "POLICY_AUDIT_EVENT_NONE",
)

# AceType in ACE_HEADER - not a real enum
ACE_TYPE = Enum(
    "ACCESS_MIN_MS_ACE_TYPE",
    "ACCESS_ALLOWED_ACE_TYPE",
    "ACCESS_DENIED_ACE_TYPE",
    "SYSTEM_AUDIT_ACE_TYPE",
    "SYSTEM_ALARM_ACE_TYPE",
    "ACCESS_MAX_MS_V2_ACE_TYPE",
    "ACCESS_ALLOWED_COMPOUND_ACE_TYPE",
    "ACCESS_MAX_MS_V3_ACE_TYPE",
    "ACCESS_MIN_MS_OBJECT_ACE_TYPE",
    "ACCESS_ALLOWED_OBJECT_ACE_TYPE",
    "ACCESS_DENIED_OBJECT_ACE_TYPE",
    "SYSTEM_AUDIT_OBJECT_ACE_TYPE",
    "SYSTEM_ALARM_OBJECT_ACE_TYPE",
    "ACCESS_MAX_MS_OBJECT_ACE_TYPE",
    "ACCESS_MAX_MS_V4_ACE_TYPE",
    "ACCESS_MAX_MS_ACE_TYPE",
    "ACCESS_ALLOWED_CALLBACK_ACE_TYPE",
    "ACCESS_DENIED_CALLBACK_ACE_TYPE",
    "ACCESS_ALLOWED_CALLBACK_OBJECT_ACE_TYPE",
    "ACCESS_DENIED_CALLBACK_OBJECT_ACE_TYPE",
    "SYSTEM_AUDIT_CALLBACK_ACE_TYPE",
    "SYSTEM_ALARM_CALLBACK_ACE_TYPE",
    "SYSTEM_AUDIT_CALLBACK_OBJECT_ACE_TYPE",
    "SYSTEM_ALARM_CALLBACK_OBJECT_ACE_TYPE",
    "SYSTEM_MANDATORY_LABEL_ACE_TYPE",
    "ACCESS_MAX_MS_V5_ACE_TYPE",
)

# bit flags for AceFlags - not a real enum
ACE_FLAGS = Enum(
    "CONTAINER_INHERIT_ACE",
    "FAILED_ACCESS_ACE_FLAG",
    "INHERIT_ONLY_ACE",
    "INHERITED_ACE",
    "NO_PROPAGATE_INHERIT_ACE",
    "OBJECT_INHERIT_ACE",
    "SUCCESSFUL_ACCESS_ACE_FLAG",
    "NO_INHERITANCE",
    "SUB_CONTAINERS_AND_OBJECTS_INHERIT",
    "SUB_CONTAINERS_ONLY_INHERIT",
    "SUB_OBJECTS_ONLY_INHERIT",
)

# used in SetEntriesInAcl - very similar to ACE_TYPE
ACCESS_MODE = Enum(
    "NOT_USED_ACCESS",
    "GRANT_ACCESS",
    "SET_ACCESS",
    "DENY_ACCESS",
    "REVOKE_ACCESS",
    "SET_AUDIT_SUCCESS",
    "SET_AUDIT_FAILURE",
)

# Bit flags in PSECURITY_DESCRIPTOR->Control - not a real enum
SECURITY_DESCRIPTOR_CONTROL_FLAGS = Enum(
    "SE_DACL_AUTO_INHERITED",  ## win2k and up
    "SE_SACL_AUTO_INHERITED",  ## win2k and up
    "SE_DACL_PROTECTED",  ## win2k and up
    "SE_SACL_PROTECTED",  ## win2k and up
    "SE_DACL_DEFAULTED",
    "SE_DACL_PRESENT",
    "SE_GROUP_DEFAULTED",
    "SE_OWNER_DEFAULTED",
    "SE_SACL_PRESENT",
    "SE_SELF_RELATIVE",
    "SE_SACL_DEFAULTED",
)

# types of SID
SID_NAME_USE = Enum(
    "SidTypeUser",
    "SidTypeGroup",
    "SidTypeDomain",
    "SidTypeAlias",
    "SidTypeWellKnownGroup",
    "SidTypeDeletedAccount",
    "SidTypeInvalid",
    "SidTypeUnknown",
    "SidTypeComputer",
    "SidTypeLabel",
)

## bit flags, not a real enum
TOKEN_ACCESS_PRIVILEGES = Enum(
    "TOKEN_ADJUST_DEFAULT",
    "TOKEN_ADJUST_GROUPS",
    "TOKEN_ADJUST_PRIVILEGES",
    "TOKEN_ALL_ACCESS",
    "TOKEN_ASSIGN_PRIMARY",
    "TOKEN_DUPLICATE",
    "TOKEN_EXECUTE",
    "TOKEN_IMPERSONATE",
    "TOKEN_QUERY",
    "TOKEN_QUERY_SOURCE",
    "TOKEN_READ",
    "TOKEN_WRITE",
)

SECURITY_IMPERSONATION_LEVEL = Enum(
    "SecurityAnonymous",
    "SecurityIdentification",
    "SecurityImpersonation",
    "SecurityDelegation",
)

POLICY_SERVER_ENABLE_STATE = Enum("PolicyServerEnabled", "PolicyServerDisabled")

POLICY_NOTIFICATION_INFORMATION_CLASS = Enum(
    "PolicyNotifyAuditEventsInformation",
    "PolicyNotifyAccountDomainInformation",
    "PolicyNotifyServerRoleInformation",
    "PolicyNotifyDnsDomainInformation",
    "PolicyNotifyDomainEfsInformation",
    "PolicyNotifyDomainKerberosTicketInformation",
    "PolicyNotifyMachineAccountPasswordInformation",
)

TRUSTED_INFORMATION_CLASS = Enum(
    "TrustedDomainNameInformation",
    "TrustedControllersInformation",
    "TrustedPosixOffsetInformation",
    "TrustedPasswordInformation",
    "TrustedDomainInformationBasic",
    "TrustedDomainInformationEx",
    "TrustedDomainAuthInformation",
    "TrustedDomainFullInformation",
    "TrustedDomainAuthInformationInternal",
    "TrustedDomainFullInformationInternal",
    "TrustedDomainInformationEx2Internal",
    "TrustedDomainFullInformation2Internal",
)

TRUSTEE_FORM = Enum(
    "TRUSTEE_IS_SID",
    "TRUSTEE_IS_NAME",
    "TRUSTEE_BAD_FORM",
    "TRUSTEE_IS_OBJECTS_AND_SID",
    "TRUSTEE_IS_OBJECTS_AND_NAME",
)

TRUSTEE_TYPE = Enum(
    "TRUSTEE_IS_UNKNOWN",
    "TRUSTEE_IS_USER",
    "TRUSTEE_IS_GROUP",
    "TRUSTEE_IS_DOMAIN",
    "TRUSTEE_IS_ALIAS",
    "TRUSTEE_IS_WELL_KNOWN_GROUP",
    "TRUSTEE_IS_DELETED",
    "TRUSTEE_IS_INVALID",
    "TRUSTEE_IS_COMPUTER",
)

## SE_OBJECT_TYPE - securable objects
SE_OBJECT_TYPE = Enum(
    "SE_UNKNOWN_OBJECT_TYPE",
    "SE_FILE_OBJECT",
    "SE_SERVICE",
    "SE_PRINTER",
    "SE_REGISTRY_KEY",
    "SE_LMSHARE",
    "SE_KERNEL_OBJECT",
    "SE_WINDOW_OBJECT",
    "SE_DS_OBJECT",
    "SE_DS_OBJECT_ALL",
    "SE_PROVIDER_DEFINED_OBJECT",
    "SE_WMIGUID_OBJECT",
    "SE_REGISTRY_WOW64_32KEY",
)

PRIVILEGE_FLAGS = Enum(
    "SE_PRIVILEGE_ENABLED_BY_DEFAULT",
    "SE_PRIVILEGE_ENABLED",
    "SE_PRIVILEGE_USED_FOR_ACCESS",
)

# Group flags used with TokenGroups
TOKEN_GROUP_ATTRIBUTES = Enum(
    "SE_GROUP_MANDATORY",
    "SE_GROUP_ENABLED_BY_DEFAULT",
    "SE_GROUP_ENABLED",
    "SE_GROUP_OWNER",
    "SE_GROUP_USE_FOR_DENY_ONLY",
    "SE_GROUP_INTEGRITY",
    "SE_GROUP_INTEGRITY_ENABLED",
    "SE_GROUP_LOGON_ID",
    "SE_GROUP_RESOURCE",
)

# Privilege flags returned by TokenPrivileges
TOKEN_PRIVILEGE_ATTRIBUTES = Enum(
    "SE_PRIVILEGE_ENABLED_BY_DEFAULT",
    "SE_PRIVILEGE_ENABLED",
    "SE_PRIVILEGE_REMOVED",
    "SE_PRIVILEGE_USED_FOR_ACCESS",
)
