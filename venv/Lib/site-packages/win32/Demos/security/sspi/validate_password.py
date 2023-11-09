# Demonstrates how to validate a password.
# See also MSKB article Q180548
#
# To use with Kerberos you need to jump through the 'targetspn' hoops.

import sys

import win32security
from sspi import ClientAuth, ServerAuth


def validate(username, password, domain=""):
    auth_info = username, domain, password
    ca = ClientAuth("NTLM", auth_info=auth_info)
    sa = ServerAuth("NTLM")

    data = err = None
    while err != 0:
        err, data = ca.authorize(data)
        err, data = sa.authorize(data)
    # If we get here without exception, we worked!


if __name__ == "__main__":
    if len(sys.argv) not in [2, 3, 4]:
        print("Usage: %s username [password [domain]]" % (__file__,))
        sys.exit(1)

    # password and domain are optional!
    password = None
    if len(sys.argv) >= 3:
        password = sys.argv[2]
    domain = ""
    if len(sys.argv) >= 4:
        domain = sys.argv[3]
    try:
        validate(sys.argv[1], password, domain)
        print("Validated OK")
    except win32security.error as details:
        hr, func, msg = details
        print("Validation failed: %s (%d)" % (msg, hr))
