# A demo of using the RAS API from Python
import sys

import win32ras


# The error raised if we can not
class ConnectionError(Exception):
    pass


def Connect(rasEntryName, numRetries=5):
    """Make a connection to the specified RAS entry.

    Returns a tuple of (bool, handle) on success.
    - bool is 1 if a new connection was established, or 0 is a connection already existed.
    - handle is a RAS HANDLE that can be passed to Disconnect() to end the connection.

    Raises a ConnectionError if the connection could not be established.
    """
    assert numRetries > 0
    for info in win32ras.EnumConnections():
        if info[1].lower() == rasEntryName.lower():
            print("Already connected to", rasEntryName)
            return 0, info[0]

    dial_params, have_pw = win32ras.GetEntryDialParams(None, rasEntryName)
    if not have_pw:
        print("Error: The password is not saved for this connection")
        print(
            "Please connect manually selecting the 'save password' option and try again"
        )
        sys.exit(1)

    print("Connecting to", rasEntryName, "...")
    retryCount = numRetries
    while retryCount > 0:
        rasHandle, errCode = win32ras.Dial(None, None, dial_params, None)
        if win32ras.IsHandleValid(rasHandle):
            bValid = 1
            break
        print("Retrying...")
        win32api.Sleep(5000)
        retryCount = retryCount - 1

    if errCode:
        raise ConnectionError(errCode, win32ras.GetErrorString(errCode))
    return 1, rasHandle


def Disconnect(handle):
    if type(handle) == type(""):  # have they passed a connection name?
        for info in win32ras.EnumConnections():
            if info[1].lower() == handle.lower():
                handle = info[0]
                break
        else:
            raise ConnectionError(0, "Not connected to entry '%s'" % handle)

    win32ras.HangUp(handle)


usage = """rasutil.py - Utilities for using RAS

Usage:
  rasutil [-r retryCount] [-c rasname] [-d rasname]
  
  -r retryCount - Number of times to retry the RAS connection
  -c rasname - Connect to the phonebook entry specified by rasname
  -d rasname - Disconnect from the phonebook entry specified by rasname
"""


def Usage(why):
    print(why)
    print(usage)
    sys.exit(1)


if __name__ == "__main__":
    import getopt

    try:
        opts, args = getopt.getopt(sys.argv[1:], "r:c:d:")
    except getopt.error as why:
        Usage(why)
    retries = 5
    if len(args) != 0:
        Usage("Invalid argument")

    for opt, val in opts:
        if opt == "-c":
            Connect(val, retries)
        if opt == "-d":
            Disconnect(val)
        if opt == "-r":
            retries = int(val)
