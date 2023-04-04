## AHH - I cant make this work!!!

# But this is the general idea.

import sys

import netscape

error = "Netscape Test Error"

if __name__ == "__main__":
    n = netscape.CNetworkCX()
    rc = n.Open("http://d|/temp/apyext.html", 0, None, 0, None)
    if not rc:
        raise error("Open method of Netscape failed")
    while 1:
        num, str = n.Read(None, 0)
        print("Got ", num, str)
        if num == 0:
            break  # used to be continue - no idea!!
        if num == -1:
            break
    #               sys.stdout.write(str)
    n.Close()
    print("Done!")
    del n
    sys.last_type = sys.last_value = sys.last_traceback = None
