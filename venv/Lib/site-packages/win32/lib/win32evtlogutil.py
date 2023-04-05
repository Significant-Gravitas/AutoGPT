"""Event Log Utilities - helper for win32evtlog.pyd
"""

import win32api
import win32con
import win32evtlog
import winerror

error = win32api.error  # The error the evtlog module raises.

langid = win32api.MAKELANGID(win32con.LANG_NEUTRAL, win32con.SUBLANG_NEUTRAL)


def AddSourceToRegistry(
    appName,
    msgDLL=None,
    eventLogType="Application",
    eventLogFlags=None,
    categoryDLL=None,
    categoryCount=0,
):
    """Add a source of messages to the event log.

    Allows Python program to register a custom source of messages in the
    registry.  You must also provide the DLL name that has the message table, so the
    full message text appears in the event log.

    Note that the win32evtlog.pyd file has a number of string entries with just "%1"
    built in, so many Python programs can simply use this DLL.  Disadvantages are that
    you do not get language translation, and the full text is stored in the event log,
    blowing the size of the log up.
    """

    # When an application uses the RegisterEventSource or OpenEventLog
    # function to get a handle of an event log, the event logging service
    # searches for the specified source name in the registry. You can add a
    # new source name to the registry by opening a new registry subkey
    # under the Application key and adding registry values to the new
    # subkey.

    if msgDLL is None:
        msgDLL = win32evtlog.__file__
    # Create a new key for our application
    hkey = win32api.RegCreateKey(
        win32con.HKEY_LOCAL_MACHINE,
        "SYSTEM\\CurrentControlSet\\Services\\EventLog\\%s\\%s"
        % (eventLogType, appName),
    )

    # Add the Event-ID message-file name to the subkey.
    win32api.RegSetValueEx(
        hkey,
        "EventMessageFile",  # value name \
        0,  # reserved \
        win32con.REG_EXPAND_SZ,  # value type \
        msgDLL,
    )

    # Set the supported types flags and add it to the subkey.
    if eventLogFlags is None:
        eventLogFlags = (
            win32evtlog.EVENTLOG_ERROR_TYPE
            | win32evtlog.EVENTLOG_WARNING_TYPE
            | win32evtlog.EVENTLOG_INFORMATION_TYPE
        )
    win32api.RegSetValueEx(
        hkey,  # subkey handle \
        "TypesSupported",  # value name \
        0,  # reserved \
        win32con.REG_DWORD,  # value type \
        eventLogFlags,
    )

    if categoryCount > 0:
        # Optionally, you can specify a message file that contains the categories
        if categoryDLL is None:
            categoryDLL = win32evtlog.__file__
        win32api.RegSetValueEx(
            hkey,  # subkey handle \
            "CategoryMessageFile",  # value name \
            0,  # reserved \
            win32con.REG_EXPAND_SZ,  # value type \
            categoryDLL,
        )

        win32api.RegSetValueEx(
            hkey,  # subkey handle \
            "CategoryCount",  # value name \
            0,  # reserved \
            win32con.REG_DWORD,  # value type \
            categoryCount,
        )
    win32api.RegCloseKey(hkey)


def RemoveSourceFromRegistry(appName, eventLogType="Application"):
    """Removes a source of messages from the event log."""

    # Delete our key
    try:
        win32api.RegDeleteKey(
            win32con.HKEY_LOCAL_MACHINE,
            "SYSTEM\\CurrentControlSet\\Services\\EventLog\\%s\\%s"
            % (eventLogType, appName),
        )
    except win32api.error as exc:
        if exc.winerror != winerror.ERROR_FILE_NOT_FOUND:
            raise


def ReportEvent(
    appName,
    eventID,
    eventCategory=0,
    eventType=win32evtlog.EVENTLOG_ERROR_TYPE,
    strings=None,
    data=None,
    sid=None,
):
    """Report an event for a previously added event source."""
    # Get a handle to the Application event log
    hAppLog = win32evtlog.RegisterEventSource(None, appName)

    # Now report the event, which will add this event to the event log */
    win32evtlog.ReportEvent(
        hAppLog,  # event-log handle \
        eventType,
        eventCategory,
        eventID,
        sid,
        strings,
        data,
    )

    win32evtlog.DeregisterEventSource(hAppLog)


def FormatMessage(eventLogRecord, logType="Application"):
    """Given a tuple from ReadEventLog, and optionally where the event
    record came from, load the message, and process message inserts.

    Note that this function may raise win32api.error.  See also the
    function SafeFormatMessage which will return None if the message can
    not be processed.
    """

    # From the event log source name, we know the name of the registry
    # key to look under for the name of the message DLL that contains
    # the messages we need to extract with FormatMessage. So first get
    # the event log source name...
    keyName = "SYSTEM\\CurrentControlSet\\Services\\EventLog\\%s\\%s" % (
        logType,
        eventLogRecord.SourceName,
    )

    # Now open this key and get the EventMessageFile value, which is
    # the name of the message DLL.
    handle = win32api.RegOpenKey(win32con.HKEY_LOCAL_MACHINE, keyName)
    try:
        dllNames = win32api.RegQueryValueEx(handle, "EventMessageFile")[0].split(";")
        # Win2k etc appear to allow multiple DLL names
        data = None
        for dllName in dllNames:
            try:
                # Expand environment variable strings in the message DLL path name,
                # in case any are there.
                dllName = win32api.ExpandEnvironmentStrings(dllName)

                dllHandle = win32api.LoadLibraryEx(
                    dllName, 0, win32con.LOAD_LIBRARY_AS_DATAFILE
                )
                try:
                    data = win32api.FormatMessageW(
                        win32con.FORMAT_MESSAGE_FROM_HMODULE,
                        dllHandle,
                        eventLogRecord.EventID,
                        langid,
                        eventLogRecord.StringInserts,
                    )
                finally:
                    win32api.FreeLibrary(dllHandle)
            except win32api.error:
                pass  # Not in this DLL - try the next
            if data is not None:
                break
    finally:
        win32api.RegCloseKey(handle)
    return data or ""  # Don't want "None" ever being returned.


def SafeFormatMessage(eventLogRecord, logType=None):
    """As for FormatMessage, except returns an error message if
    the message can not be processed.
    """
    if logType is None:
        logType = "Application"
    try:
        return FormatMessage(eventLogRecord, logType)
    except win32api.error:
        if eventLogRecord.StringInserts is None:
            desc = ""
        else:
            desc = ", ".join(eventLogRecord.StringInserts)
        return (
            "<The description for Event ID ( %d ) in Source ( %r ) could not be found. It contains the following insertion string(s):%r.>"
            % (
                winerror.HRESULT_CODE(eventLogRecord.EventID),
                eventLogRecord.SourceName,
                desc,
            )
        )


def FeedEventLogRecords(
    feeder, machineName=None, logName="Application", readFlags=None
):
    if readFlags is None:
        readFlags = (
            win32evtlog.EVENTLOG_BACKWARDS_READ | win32evtlog.EVENTLOG_SEQUENTIAL_READ
        )
    h = win32evtlog.OpenEventLog(machineName, logName)
    try:
        while 1:
            objects = win32evtlog.ReadEventLog(h, readFlags, 0)
            if not objects:
                break
            map(lambda item, feeder=feeder: feeder(*(item,)), objects)
    finally:
        win32evtlog.CloseEventLog(h)
