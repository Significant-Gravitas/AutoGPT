# Generate a base file name
import os
import time

import win32api
import win32evtlog


def BackupClearLog(logType):
    datePrefix = time.strftime("%Y%m%d", time.localtime(time.time()))
    fileExists = 1
    retry = 0
    while fileExists:
        if retry == 0:
            index = ""
        else:
            index = "-%d" % retry
        try:
            fname = os.path.join(
                win32api.GetTempPath(),
                "%s%s-%s" % (datePrefix, index, logType) + ".evt",
            )
            os.stat(fname)
        except os.error:
            fileExists = 0
        retry = retry + 1
    # OK - have unique file name.
    try:
        hlog = win32evtlog.OpenEventLog(None, logType)
    except win32evtlogutil.error as details:
        print("Could not open the event log", details)
        return
    try:
        if win32evtlog.GetNumberOfEventLogRecords(hlog) == 0:
            print("No records in event log %s - not backed up" % logType)
            return
        win32evtlog.ClearEventLog(hlog, fname)
        print("Backed up %s log to %s" % (logType, fname))
    finally:
        win32evtlog.CloseEventLog(hlog)


if __name__ == "__main__":
    BackupClearLog("Application")
    BackupClearLog("System")
    BackupClearLog("Security")
