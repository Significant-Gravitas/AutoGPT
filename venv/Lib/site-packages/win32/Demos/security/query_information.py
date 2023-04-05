import win32api
import win32security
import winerror
from ntsecuritycon import *


# This is a Python implementation of win32api.GetDomainName()
def GetDomainName():
    try:
        tok = win32security.OpenThreadToken(win32api.GetCurrentThread(), TOKEN_QUERY, 1)
    except win32api.error as details:
        if details[0] != winerror.ERROR_NO_TOKEN:
            raise
        # attempt to open the process token, since no thread token
        # exists
        tok = win32security.OpenProcessToken(win32api.GetCurrentProcess(), TOKEN_QUERY)
    sid, attr = win32security.GetTokenInformation(tok, TokenUser)
    win32api.CloseHandle(tok)

    name, dom, typ = win32security.LookupAccountSid(None, sid)
    return dom


if __name__ == "__main__":
    print("Domain name is", GetDomainName())
