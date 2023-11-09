import os
import tempfile

import pythoncom
import win32api
import win32event
from win32com.bits import bits
from win32com.server.util import wrap

TIMEOUT = 200  # ms
StopEvent = win32event.CreateEvent(None, 0, 0, None)

job_name = "bits-pywin32-test"
states = dict(
    [
        (val, (name[13:]))
        for name, val in vars(bits).items()
        if name.startswith("BG_JOB_STATE_")
    ]
)

bcm = pythoncom.CoCreateInstance(
    bits.CLSID_BackgroundCopyManager,
    None,
    pythoncom.CLSCTX_LOCAL_SERVER,
    bits.IID_IBackgroundCopyManager,
)


class BackgroundJobCallback:
    _com_interfaces_ = [bits.IID_IBackgroundCopyCallback]
    _public_methods_ = ["JobTransferred", "JobError", "JobModification"]

    def JobTransferred(self, job):
        print("Job Transferred", job)
        job.Complete()
        win32event.SetEvent(StopEvent)  # exit msg pump

    def JobError(self, job, error):
        print("Job Error", job, error)
        f = error.GetFile()
        print("While downloading", f.GetRemoteName())
        print("To", f.GetLocalName())
        print("The following error happened:")
        self._print_error(error)
        if f.GetRemoteName().endswith("missing-favicon.ico"):
            print("Changing to point to correct file")
            f2 = f.QueryInterface(bits.IID_IBackgroundCopyFile2)
            favicon = "http://www.python.org/favicon.ico"
            print("Changing RemoteName from", f2.GetRemoteName(), "to", favicon)
            f2.SetRemoteName(favicon)
            job.Resume()
        else:
            job.Cancel()

    def _print_error(self, err):
        ctx, hresult = err.GetError()
        try:
            hresult_msg = win32api.FormatMessage(hresult)
        except win32api.error:
            hresult_msg = ""
        print("Context=0x%x, hresult=0x%x (%s)" % (ctx, hresult, hresult_msg))
        print(err.GetErrorDescription())

    def JobModification(self, job, reserved):
        state = job.GetState()
        print("Job Modification", job.GetDisplayName(), states.get(state))
        # Need to catch TRANSIENT_ERROR here, as JobError doesn't get
        # called (apparently) when the error is transient.
        if state == bits.BG_JOB_STATE_TRANSIENT_ERROR:
            print("Error details:")
            err = job.GetError()
            self._print_error(err)


job = bcm.CreateJob(job_name, bits.BG_JOB_TYPE_DOWNLOAD)

job.SetNotifyInterface(wrap(BackgroundJobCallback()))
job.SetNotifyFlags(
    bits.BG_NOTIFY_JOB_TRANSFERRED
    | bits.BG_NOTIFY_JOB_ERROR
    | bits.BG_NOTIFY_JOB_MODIFICATION
)


# The idea here is to intentionally make one of the files fail to be
# downloaded. Then the JobError notification will be triggered, where
# we do fix the failing file by calling SetRemoteName to a valid URL
# and call Resume() on the job, making the job finish successfully.
#
# Note to self: A domain that cannot be resolved will cause
# TRANSIENT_ERROR instead of ERROR, and the JobError notification will
# not be triggered! This can bite you during testing depending on how
# your DNS is configured. For example, if you use OpenDNS.org's DNS
# servers, an invalid hostname will *always* be resolved (they
# redirect you to a search page), so be careful when testing.
job.AddFile(
    "http://www.python.org/favicon.ico",
    os.path.join(tempfile.gettempdir(), "bits-favicon.ico"),
)
job.AddFile(
    "http://www.python.org/missing-favicon.ico",
    os.path.join(tempfile.gettempdir(), "bits-missing-favicon.ico"),
)

for f in job.EnumFiles():
    print("Downloading", f.GetRemoteName())
    print("To", f.GetLocalName())

job.Resume()

while True:
    rc = win32event.MsgWaitForMultipleObjects(
        (StopEvent,), 0, TIMEOUT, win32event.QS_ALLEVENTS
    )

    if rc == win32event.WAIT_OBJECT_0:
        break
    elif rc == win32event.WAIT_OBJECT_0 + 1:
        if pythoncom.PumpWaitingMessages():
            break  # wm_quit
