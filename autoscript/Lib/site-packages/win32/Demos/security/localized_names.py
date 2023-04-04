# A Python port of the MS knowledge base article Q157234
# "How to deal with localized and renamed user and group names"
# http://support.microsoft.com/default.aspx?kbid=157234

import sys

import pywintypes
from ntsecuritycon import *
from win32net import NetUserModalsGet
from win32security import LookupAccountSid


def LookupAliasFromRid(TargetComputer, Rid):
    # Sid is the same regardless of machine, since the well-known
    # BUILTIN domain is referenced.
    sid = pywintypes.SID()
    sid.Initialize(SECURITY_NT_AUTHORITY, 2)

    for i, r in enumerate((SECURITY_BUILTIN_DOMAIN_RID, Rid)):
        sid.SetSubAuthority(i, r)

    name, domain, typ = LookupAccountSid(TargetComputer, sid)
    return name


def LookupUserGroupFromRid(TargetComputer, Rid):
    # get the account domain Sid on the target machine
    # note: if you were looking up multiple sids based on the same
    # account domain, only need to call this once.
    umi2 = NetUserModalsGet(TargetComputer, 2)
    domain_sid = umi2["domain_id"]

    SubAuthorityCount = domain_sid.GetSubAuthorityCount()

    # create and init new sid with acct domain Sid + acct Rid
    sid = pywintypes.SID()
    sid.Initialize(domain_sid.GetSidIdentifierAuthority(), SubAuthorityCount + 1)

    # copy existing subauthorities from account domain Sid into
    # new Sid
    for i in range(SubAuthorityCount):
        sid.SetSubAuthority(i, domain_sid.GetSubAuthority(i))

    # append Rid to new Sid
    sid.SetSubAuthority(SubAuthorityCount, Rid)

    name, domain, typ = LookupAccountSid(TargetComputer, sid)
    return name


def main():
    if len(sys.argv) == 2:
        targetComputer = sys.argv[1]
    else:
        targetComputer = None

    name = LookupUserGroupFromRid(targetComputer, DOMAIN_USER_RID_ADMIN)
    print("'Administrator' user name = %s" % (name,))

    name = LookupAliasFromRid(targetComputer, DOMAIN_ALIAS_RID_ADMINS)
    print("'Administrators' local group/alias name = %s" % (name,))


if __name__ == "__main__":
    main()
