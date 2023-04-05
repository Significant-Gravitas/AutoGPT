"""A demo of using win32net.NetValidatePasswordPolicy.

Example usage:

% NetValidatePasswordPolicy.py --password=foo change
which might return:

> Result of 'change' validation is 0: The operation completed successfully.

or depending on the policy:

> Result of 'change' validation is 2245: The password does not meet the
> password policy requirements. Check the minimum password length,
> password complexity and password history requirements.

Adding --user doesn't seem to change the output (even the PasswordLastSet seen
when '-f' is used doesn't depend on the username), but theoretically it will
also check the password history for the specified user.

% NetValidatePasswordPolicy.py auth

which always (with and without '-m') seems to return:

> Result of 'auth' validation is 2701: Password must change at next logon
"""

import optparse
import sys
from pprint import pprint

import win32api
import win32net
import win32netcon


def main():
    parser = optparse.OptionParser(
        "%prog [options] auth|change ...",
        description="A win32net.NetValidatePasswordPolicy demo.",
    )

    parser.add_option(
        "-u",
        "--username",
        action="store",
        help="The username to pass to the function (only for the " "change command",
    )

    parser.add_option(
        "-p",
        "--password",
        action="store",
        help="The clear-text password to pass to the function "
        "(only for the 'change' command)",
    )

    parser.add_option(
        "-m",
        "--password-matched",
        action="store_false",
        default=True,
        help="Used to specify the password does NOT match (ie, "
        "uses False for the PasswordMatch/PasswordMatched "
        "arg, both 'auth' and 'change' commands)",
    )

    parser.add_option(
        "-s",
        "--server",
        action="store",
        help="The name of the server to execute the command on",
    )

    parser.add_option(
        "-f",
        "--show_fields",
        action="store_true",
        default=False,
        help="Print the NET_VALIDATE_PERSISTED_FIELDS returned",
    )

    options, args = parser.parse_args()

    if not args:
        args = ["auth"]

    for arg in args:
        if arg == "auth":
            input = {
                "PasswordMatched": options.password_matched,
            }
            val_type = win32netcon.NetValidateAuthentication
        elif arg == "change":
            input = {
                "ClearPassword": options.password,
                "PasswordMatch": options.password_matched,
                "UserAccountName": options.username,
            }
            val_type = win32netcon.NetValidatePasswordChange
        else:
            parser.error("Invalid arg - must be 'auth' or 'change'")

        try:
            fields, status = win32net.NetValidatePasswordPolicy(
                options.server, None, val_type, input
            )
        except NotImplementedError:
            print("NetValidatePasswordPolicy not implemented on this platform.")
            return 1
        except win32net.error as exc:
            print("NetValidatePasswordPolicy failed: ", exc)
            return 1

        if options.show_fields:
            print("NET_VALIDATE_PERSISTED_FIELDS fields:")
            pprint(fields)

        print(
            "Result of %r validation is %d: %s"
            % (arg, status, win32api.FormatMessage(status).strip())
        )

    return 0


if __name__ == "__main__":
    sys.exit(main())
