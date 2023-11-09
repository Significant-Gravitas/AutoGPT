# Contributed by Kelly Kranabetter.
import os
import sys

import ntsecuritycon
import pywintypes
import win32security
import winerror

# get security information
# name=r"c:\autoexec.bat"
# name= r"g:\!workgrp\lim"
name = sys.argv[0]

if not os.path.exists(name):
    print(name, "does not exist!")
    sys.exit()

print("On file ", name, "\n")

# get owner SID
print("OWNER")
try:
    sd = win32security.GetFileSecurity(name, win32security.OWNER_SECURITY_INFORMATION)
    sid = sd.GetSecurityDescriptorOwner()
    print("  ", win32security.LookupAccountSid(None, sid))
except pywintypes.error as exc:
    # in automation and network shares we see:
    # pywintypes.error: (1332, 'LookupAccountName', 'No mapping between account names and security IDs was done.')
    if exc.winerror != winerror.ERROR_NONE_MAPPED:
        raise
    print("No owner information is available")

# get group SID
try:
    print("GROUP")
    sd = win32security.GetFileSecurity(name, win32security.GROUP_SECURITY_INFORMATION)
    sid = sd.GetSecurityDescriptorGroup()
    print("  ", win32security.LookupAccountSid(None, sid))
except pywintypes.error as exc:
    if exc.winerror != winerror.ERROR_NONE_MAPPED:
        raise
    print("No group information is available")

# get ACEs
sd = win32security.GetFileSecurity(name, win32security.DACL_SECURITY_INFORMATION)
dacl = sd.GetSecurityDescriptorDacl()
if dacl == None:
    print("No Discretionary ACL")
else:
    for ace_no in range(0, dacl.GetAceCount()):
        ace = dacl.GetAce(ace_no)
        print("ACE", ace_no)

        print("  -Type")
        for i in (
            "ACCESS_ALLOWED_ACE_TYPE",
            "ACCESS_DENIED_ACE_TYPE",
            "SYSTEM_AUDIT_ACE_TYPE",
            "SYSTEM_ALARM_ACE_TYPE",
        ):
            if getattr(ntsecuritycon, i) == ace[0][0]:
                print("    ", i)

        print("  -Flags", hex(ace[0][1]))
        for i in (
            "OBJECT_INHERIT_ACE",
            "CONTAINER_INHERIT_ACE",
            "NO_PROPAGATE_INHERIT_ACE",
            "INHERIT_ONLY_ACE",
            "SUCCESSFUL_ACCESS_ACE_FLAG",
            "FAILED_ACCESS_ACE_FLAG",
        ):
            if getattr(ntsecuritycon, i) & ace[0][1] == getattr(ntsecuritycon, i):
                print("    ", i)

        print("  -mask", hex(ace[1]))

        # files and directories do permissions differently
        permissions_file = (
            "DELETE",
            "READ_CONTROL",
            "WRITE_DAC",
            "WRITE_OWNER",
            "SYNCHRONIZE",
            "FILE_GENERIC_READ",
            "FILE_GENERIC_WRITE",
            "FILE_GENERIC_EXECUTE",
            "FILE_DELETE_CHILD",
        )
        permissions_dir = (
            "DELETE",
            "READ_CONTROL",
            "WRITE_DAC",
            "WRITE_OWNER",
            "SYNCHRONIZE",
            "FILE_ADD_SUBDIRECTORY",
            "FILE_ADD_FILE",
            "FILE_DELETE_CHILD",
            "FILE_LIST_DIRECTORY",
            "FILE_TRAVERSE",
            "FILE_READ_ATTRIBUTES",
            "FILE_WRITE_ATTRIBUTES",
            "FILE_READ_EA",
            "FILE_WRITE_EA",
        )
        permissions_dir_inherit = (
            "DELETE",
            "READ_CONTROL",
            "WRITE_DAC",
            "WRITE_OWNER",
            "SYNCHRONIZE",
            "GENERIC_READ",
            "GENERIC_WRITE",
            "GENERIC_EXECUTE",
            "GENERIC_ALL",
        )
        if os.path.isfile(name):
            permissions = permissions_file
        else:
            permissions = permissions_dir
            # directories also contain an ACE that is inherited by children (files) within them
            if (
                ace[0][1] & ntsecuritycon.OBJECT_INHERIT_ACE
                == ntsecuritycon.OBJECT_INHERIT_ACE
                and ace[0][1] & ntsecuritycon.INHERIT_ONLY_ACE
                == ntsecuritycon.INHERIT_ONLY_ACE
            ):
                permissions = permissions_dir_inherit

        calc_mask = 0  # calculate the mask so we can see if we are printing all of the permissions
        for i in permissions:
            if getattr(ntsecuritycon, i) & ace[1] == getattr(ntsecuritycon, i):
                calc_mask = calc_mask | getattr(ntsecuritycon, i)
                print("    ", i)
        print("  ", "Calculated Check Mask=", hex(calc_mask))
        print("  -SID\n    ", win32security.LookupAccountSid(None, ace[2]))
