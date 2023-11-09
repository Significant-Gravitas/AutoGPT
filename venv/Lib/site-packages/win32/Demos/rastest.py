# rastest.py - test/demonstrate the win32ras module.
# Much of the code here contributed by Jethro Wright.

import os
import sys

import win32ras

# Build a little dictionary of RAS states to decent strings.
# eg win32ras.RASCS_OpenPort -> "OpenPort"
stateMap = {}
for name, val in list(win32ras.__dict__.items()):
    if name[:6] == "RASCS_":
        stateMap[val] = name[6:]

# Use a lock so the callback can tell the main thread when it is finished.
import win32event

callbackEvent = win32event.CreateEvent(None, 0, 0, None)


def Callback(hras, msg, state, error, exterror):
    #       print "Callback called with ", hras, msg, state, error, exterror
    stateName = stateMap.get(state, "Unknown state?")
    print("Status is %s (%04lx), error code is %d" % (stateName, state, error))
    finished = state in [win32ras.RASCS_Connected]
    if finished:
        win32event.SetEvent(callbackEvent)
    if error != 0 or int(state) == win32ras.RASCS_Disconnected:
        #       we know for sure this is a good place to hangup....
        print("Detected call failure: %s" % win32ras.GetErrorString(error))
        HangUp(hras)
        win32event.SetEvent(callbackEvent)


def ShowConnections():
    print("All phone-book entries:")
    for (name,) in win32ras.EnumEntries():
        print(" ", name)
    print("Current Connections:")
    for con in win32ras.EnumConnections():
        print(" ", con)


def EditEntry(entryName):
    try:
        win32ras.EditPhonebookEntry(0, None, entryName)
    except win32ras.error as xxx_todo_changeme:
        (rc, function, msg) = xxx_todo_changeme.args
        print("Can not edit/find the RAS entry -", msg)


def HangUp(hras):
    #       trap potential, irrelevant errors from win32ras....
    try:
        win32ras.HangUp(hras)
    except:
        print("Tried to hang up gracefully on error, but didn't work....")
    return None


def Connect(entryName, bUseCallback):
    if bUseCallback:
        theCallback = Callback
        win32event.ResetEvent(callbackEvent)
    else:
        theCallback = None
    #       in order to *use* the username/password of a particular dun entry, one must
    #       explicitly get those params under win95....
    try:
        dp, b = win32ras.GetEntryDialParams(None, entryName)
    except:
        print("Couldn't find DUN entry: %s" % entryName)
    else:
        hras, rc = win32ras.Dial(
            None, None, (entryName, "", "", dp[3], dp[4], ""), theCallback
        )
        #       hras, rc = win32ras.Dial(None, None, (entryName, ),theCallback)
        #       print hras, rc
        if not bUseCallback and rc != 0:
            print("Could not dial the RAS connection:", win32ras.GetErrorString(rc))
            hras = HangUp(hras)
        #       don't wait here if there's no need to....
        elif (
            bUseCallback
            and win32event.WaitForSingleObject(callbackEvent, 60000)
            != win32event.WAIT_OBJECT_0
        ):
            print("Gave up waiting for the process to complete!")
            #       sdk docs state one must explcitly hangup, even if there's an error....
            try:
                cs = win32ras.GetConnectStatus(hras)
            except:
                #       on error, attempt a hang up anyway....
                hras = HangUp(hras)
            else:
                if int(cs[0]) == win32ras.RASCS_Disconnected:
                    hras = HangUp(hras)
    return hras, rc


def Disconnect(rasEntry):
    # Need to find the entry
    name = rasEntry.lower()
    for hcon, entryName, devName, devType in win32ras.EnumConnections():
        if entryName.lower() == name:
            win32ras.HangUp(hcon)
            print("Disconnected from", rasEntry)
            break
    else:
        print("Could not find an open connection to", entryName)


usage = """
Usage: %s [-s] [-l] [-c connection] [-d connection]
-l : List phone-book entries and current connections.
-s : Show status while connecting/disconnecting (uses callbacks)
-c : Connect to the specified phonebook name.
-d : Disconnect from the specified phonebook name.
-e : Edit the specified phonebook entry.
"""


def main():
    import getopt

    try:
        opts, args = getopt.getopt(sys.argv[1:], "slc:d:e:")
    except getopt.error as why:
        print(why)
        print(
            usage
            % (
                os.path.basename(
                    sys.argv[0],
                )
            )
        )
        return

    bCallback = 0
    if args or not opts:
        print(
            usage
            % (
                os.path.basename(
                    sys.argv[0],
                )
            )
        )
        return
    for opt, val in opts:
        if opt == "-s":
            bCallback = 1
        if opt == "-l":
            ShowConnections()
        if opt == "-c":
            hras, rc = Connect(val, bCallback)
            if hras != None:
                print("hras: 0x%8lx, rc: 0x%04x" % (hras, rc))
        if opt == "-d":
            Disconnect(val)
        if opt == "-e":
            EditEntry(val)


if __name__ == "__main__":
    main()
