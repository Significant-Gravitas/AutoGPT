## Demonstrates how to create a "pull" subscription
import win32con
import win32event
import win32evtlog

query_text = '*[System[Provider[@Name="Microsoft-Windows-Winlogon"]]]'

h = win32event.CreateEvent(None, 0, 0, None)
s = win32evtlog.EvtSubscribe(
    "System",
    win32evtlog.EvtSubscribeStartAtOldestRecord,
    SignalEvent=h,
    Query=query_text,
)

while 1:
    while 1:
        events = win32evtlog.EvtNext(s, 10)
        if len(events) == 0:
            break
        ##for event in events:
        ##	print(win32evtlog.EvtRender(event, win32evtlog.EvtRenderEventXml))
        print("retrieved %s events" % len(events))
    while 1:
        print("waiting...")
        w = win32event.WaitForSingleObjectEx(h, 2000, True)
        if w == win32con.WAIT_OBJECT_0:
            break
