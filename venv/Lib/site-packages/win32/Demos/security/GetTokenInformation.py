""" Lists various types of information about current user's access token,
    including UAC status on Vista
"""

import pywintypes
import win32api
import win32con
import win32security
import winerror
from security_enums import (
    SECURITY_IMPERSONATION_LEVEL,
    TOKEN_ELEVATION_TYPE,
    TOKEN_GROUP_ATTRIBUTES,
    TOKEN_PRIVILEGE_ATTRIBUTES,
    TOKEN_TYPE,
)


def dump_token(th):
    token_type = win32security.GetTokenInformation(th, win32security.TokenType)
    print("TokenType:", token_type, TOKEN_TYPE.lookup_name(token_type))
    if token_type == win32security.TokenImpersonation:
        imp_lvl = win32security.GetTokenInformation(
            th, win32security.TokenImpersonationLevel
        )
        print(
            "TokenImpersonationLevel:",
            imp_lvl,
            SECURITY_IMPERSONATION_LEVEL.lookup_name(imp_lvl),
        )

    print(
        "TokenSessionId:",
        win32security.GetTokenInformation(th, win32security.TokenSessionId),
    )

    privs = win32security.GetTokenInformation(th, win32security.TokenPrivileges)
    print("TokenPrivileges:")
    for priv_luid, priv_flags in privs:
        flag_names, unk = TOKEN_PRIVILEGE_ATTRIBUTES.lookup_flags(priv_flags)
        flag_desc = " ".join(flag_names)
        if unk:
            flag_desc += "(" + str(unk) + ")"

        priv_name = win32security.LookupPrivilegeName("", priv_luid)
        priv_desc = win32security.LookupPrivilegeDisplayName("", priv_name)
        print("\t", priv_name, priv_desc, priv_flags, flag_desc)

    print("TokenGroups:")
    groups = win32security.GetTokenInformation(th, win32security.TokenGroups)
    for group_sid, group_attr in groups:
        flag_names, unk = TOKEN_GROUP_ATTRIBUTES.lookup_flags(group_attr)
        flag_desc = " ".join(flag_names)
        if unk:
            flag_desc += "(" + str(unk) + ")"
        if group_attr & TOKEN_GROUP_ATTRIBUTES.SE_GROUP_LOGON_ID:
            sid_desc = "Logon sid"
        else:
            sid_desc = win32security.LookupAccountSid("", group_sid)
        print("\t", group_sid, sid_desc, group_attr, flag_desc)

    ## Vista token information types, will throw (87, 'GetTokenInformation', 'The parameter is incorrect.') on earier OS
    try:
        is_elevated = win32security.GetTokenInformation(
            th, win32security.TokenElevation
        )
        print("TokenElevation:", is_elevated)
    except pywintypes.error as details:
        if details.winerror != winerror.ERROR_INVALID_PARAMETER:
            raise
        return None
    print(
        "TokenHasRestrictions:",
        win32security.GetTokenInformation(th, win32security.TokenHasRestrictions),
    )
    print(
        "TokenMandatoryPolicy",
        win32security.GetTokenInformation(th, win32security.TokenMandatoryPolicy),
    )
    print(
        "TokenVirtualizationAllowed:",
        win32security.GetTokenInformation(th, win32security.TokenVirtualizationAllowed),
    )
    print(
        "TokenVirtualizationEnabled:",
        win32security.GetTokenInformation(th, win32security.TokenVirtualizationEnabled),
    )

    elevation_type = win32security.GetTokenInformation(
        th, win32security.TokenElevationType
    )
    print(
        "TokenElevationType:",
        elevation_type,
        TOKEN_ELEVATION_TYPE.lookup_name(elevation_type),
    )
    if elevation_type != win32security.TokenElevationTypeDefault:
        lt = win32security.GetTokenInformation(th, win32security.TokenLinkedToken)
        print("TokenLinkedToken:", lt)
    else:
        lt = None
    return lt


ph = win32api.GetCurrentProcess()
th = win32security.OpenProcessToken(ph, win32con.MAXIMUM_ALLOWED)
lt = dump_token(th)
if lt:
    print("\n\nlinked token info:")
    dump_token(lt)
