import win32event
import win32security

evt = win32event.CreateEvent(None, 0, 0, None)
win32security.LsaRegisterPolicyChangeNotification(
    win32security.PolicyNotifyAuditEventsInformation, evt
)
print("Waiting for you change Audit policy in Management console ...")
ret_code = win32event.WaitForSingleObject(evt, 1000000000)
## should come back when you change Audit policy in Management console ...
print(ret_code)
win32security.LsaUnregisterPolicyChangeNotification(
    win32security.PolicyNotifyAuditEventsInformation, evt
)
