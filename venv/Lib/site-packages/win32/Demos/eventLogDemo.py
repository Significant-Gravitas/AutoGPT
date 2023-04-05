import win32api  # To translate NT Sids to account names.
import win32con
import win32evtlog
import win32evtlogutil
import win32security


def ReadLog(computer, logType="Application", dumpEachRecord=0):
    # read the entire log back.
    h = win32evtlog.OpenEventLog(computer, logType)
    numRecords = win32evtlog.GetNumberOfEventLogRecords(h)
    #       print "There are %d records" % numRecords

    num = 0
    while 1:
        objects = win32evtlog.ReadEventLog(
            h,
            win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ,
            0,
        )
        if not objects:
            break
        for object in objects:
            # get it for testing purposes, but dont print it.
            msg = win32evtlogutil.SafeFormatMessage(object, logType)
            if object.Sid is not None:
                try:
                    domain, user, typ = win32security.LookupAccountSid(
                        computer, object.Sid
                    )
                    sidDesc = "%s/%s" % (domain, user)
                except win32security.error:
                    sidDesc = str(object.Sid)
                user_desc = "Event associated with user %s" % (sidDesc,)
            else:
                user_desc = None
            if dumpEachRecord:
                print(
                    "Event record from %r generated at %s"
                    % (object.SourceName, object.TimeGenerated.Format())
                )
                if user_desc:
                    print(user_desc)
                try:
                    print(msg)
                except UnicodeError:
                    print("(unicode error printing message: repr() follows...)")
                    print(repr(msg))

        num = num + len(objects)

    if numRecords == num:
        print("Successfully read all", numRecords, "records")
    else:
        print(
            "Couldn't get all records - reported %d, but found %d" % (numRecords, num)
        )
        print(
            "(Note that some other app may have written records while we were running!)"
        )
    win32evtlog.CloseEventLog(h)


def usage():
    print("Writes an event to the event log.")
    print("-w : Dont write any test records.")
    print("-r : Dont read the event log")
    print("-c : computerName : Process the log on the specified computer")
    print("-v : Verbose")
    print("-t : LogType - Use the specified log - default = 'Application'")


def test():
    # check if running on Windows NT, if not, display notice and terminate
    if win32api.GetVersion() & 0x80000000:
        print("This sample only runs on NT")
        return

    import getopt
    import sys

    opts, args = getopt.getopt(sys.argv[1:], "rwh?c:t:v")
    computer = None
    do_read = do_write = 1

    logType = "Application"
    verbose = 0

    if len(args) > 0:
        print("Invalid args")
        usage()
        return 1
    for opt, val in opts:
        if opt == "-t":
            logType = val
        if opt == "-c":
            computer = val
        if opt in ["-h", "-?"]:
            usage()
            return
        if opt == "-r":
            do_read = 0
        if opt == "-w":
            do_write = 0
        if opt == "-v":
            verbose = verbose + 1
    if do_write:
        ph = win32api.GetCurrentProcess()
        th = win32security.OpenProcessToken(ph, win32con.TOKEN_READ)
        my_sid = win32security.GetTokenInformation(th, win32security.TokenUser)[0]

        win32evtlogutil.ReportEvent(
            logType,
            2,
            strings=["The message text for event 2", "Another insert"],
            data="Raw\0Data".encode("ascii"),
            sid=my_sid,
        )
        win32evtlogutil.ReportEvent(
            logType,
            1,
            eventType=win32evtlog.EVENTLOG_WARNING_TYPE,
            strings=["A warning", "An even more dire warning"],
            data="Raw\0Data".encode("ascii"),
            sid=my_sid,
        )
        win32evtlogutil.ReportEvent(
            logType,
            1,
            eventType=win32evtlog.EVENTLOG_INFORMATION_TYPE,
            strings=["An info", "Too much info"],
            data="Raw\0Data".encode("ascii"),
            sid=my_sid,
        )
        print("Successfully wrote 3 records to the log")

    if do_read:
        ReadLog(computer, logType, verbose > 0)


if __name__ == "__main__":
    test()
