## Demonstrates a "push" subscription with a callback function
import win32evtlog

query_text = '*[System[Provider[@Name="Microsoft-Windows-Winlogon"]]]'


def c(reason, context, evt):
    if reason == win32evtlog.EvtSubscribeActionError:
        print("EvtSubscribeActionError")
    elif reason == win32evtlog.EvtSubscribeActionDeliver:
        print("EvtSubscribeActionDeliver")
    else:
        print("??? Unknown action ???", reason)
    context.append(win32evtlog.EvtRender(evt, win32evtlog.EvtRenderEventXml))
    return 0


evttext = []
s = win32evtlog.EvtSubscribe(
    "System",
    win32evtlog.EvtSubscribeStartAtOldestRecord,
    Query="*",
    Callback=c,
    Context=evttext,
)
