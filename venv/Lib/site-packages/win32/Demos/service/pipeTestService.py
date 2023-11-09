# A Demo of services and named pipes.

# A multi-threaded service that simply echos back its input.

# * Install as a service using "pipeTestService.py install"
# * Use Control Panel to change the user name of the service
#   to a real user name (ie, NOT the SystemAccount)
# * Start the service.
# * Run the "pipeTestServiceClient.py" program as the client pipe side.

import _thread
import traceback

# Old versions of the service framework would not let you import this
# module at the top-level.  Now you can, and can check 'Debugging()' and
# 'RunningAsService()' to check your context.
import pywintypes
import servicemanager
import win32con
import win32service
import win32serviceutil
import winerror
from ntsecuritycon import *
from win32api import *

# Use "import *" to keep this looking as much as a "normal" service
# as possible.  Real code shouldn't do this.
from win32event import *
from win32file import *
from win32pipe import *


def ApplyIgnoreError(fn, args):
    try:
        return fn(*args)
    except error:  # Ignore win32api errors.
        return None


class TestPipeService(win32serviceutil.ServiceFramework):
    _svc_name_ = "PyPipeTestService"
    _svc_display_name_ = "Python Pipe Test Service"
    _svc_description_ = "Tests Python service framework by receiving and echoing messages over a named pipe"

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = CreateEvent(None, 0, 0, None)
        self.overlapped = pywintypes.OVERLAPPED()
        self.overlapped.hEvent = CreateEvent(None, 0, 0, None)
        self.thread_handles = []

    def CreatePipeSecurityObject(self):
        # Create a security object giving World read/write access,
        # but only "Owner" modify access.
        sa = pywintypes.SECURITY_ATTRIBUTES()
        sidEveryone = pywintypes.SID()
        sidEveryone.Initialize(SECURITY_WORLD_SID_AUTHORITY, 1)
        sidEveryone.SetSubAuthority(0, SECURITY_WORLD_RID)
        sidCreator = pywintypes.SID()
        sidCreator.Initialize(SECURITY_CREATOR_SID_AUTHORITY, 1)
        sidCreator.SetSubAuthority(0, SECURITY_CREATOR_OWNER_RID)

        acl = pywintypes.ACL()
        acl.AddAccessAllowedAce(FILE_GENERIC_READ | FILE_GENERIC_WRITE, sidEveryone)
        acl.AddAccessAllowedAce(FILE_ALL_ACCESS, sidCreator)

        sa.SetSecurityDescriptorDacl(1, acl, 0)
        return sa

    # The functions executed in their own thread to process a client request.
    def DoProcessClient(self, pipeHandle, tid):
        try:
            try:
                # Create a loop, reading large data.  If we knew the data stream was
                # was small, a simple ReadFile would do.
                d = "".encode("ascii")  # ensure bytes on py2k and py3k...
                hr = winerror.ERROR_MORE_DATA
                while hr == winerror.ERROR_MORE_DATA:
                    hr, thisd = ReadFile(pipeHandle, 256)
                    d = d + thisd
                print("Read", d)
                ok = 1
            except error:
                # Client disconnection - do nothing
                ok = 0

            # A secure service would handle (and ignore!) errors writing to the
            # pipe, but for the sake of this demo we dont (if only to see what errors
            # we can get when our clients break at strange times :-)
            if ok:
                msg = (
                    "%s (on thread %d) sent me %s"
                    % (GetNamedPipeHandleState(pipeHandle, False, True)[4], tid, d)
                ).encode("ascii")
                WriteFile(pipeHandle, msg)
        finally:
            ApplyIgnoreError(DisconnectNamedPipe, (pipeHandle,))
            ApplyIgnoreError(CloseHandle, (pipeHandle,))

    def ProcessClient(self, pipeHandle):
        try:
            procHandle = GetCurrentProcess()
            th = DuplicateHandle(
                procHandle,
                GetCurrentThread(),
                procHandle,
                0,
                0,
                win32con.DUPLICATE_SAME_ACCESS,
            )
            try:
                self.thread_handles.append(th)
                try:
                    return self.DoProcessClient(pipeHandle, th)
                except:
                    traceback.print_exc()
            finally:
                self.thread_handles.remove(th)
        except:
            traceback.print_exc()

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        # Write an event log record - in debug mode we will also
        # see this message printed.
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STARTED,
            (self._svc_name_, ""),
        )

        num_connections = 0
        while 1:
            pipeHandle = CreateNamedPipe(
                "\\\\.\\pipe\\PyPipeTest",
                PIPE_ACCESS_DUPLEX | FILE_FLAG_OVERLAPPED,
                PIPE_TYPE_MESSAGE | PIPE_READMODE_BYTE,
                PIPE_UNLIMITED_INSTANCES,  # max instances
                0,
                0,
                6000,
                self.CreatePipeSecurityObject(),
            )
            try:
                hr = ConnectNamedPipe(pipeHandle, self.overlapped)
            except error as details:
                print("Error connecting pipe!", details)
                CloseHandle(pipeHandle)
                break
            if hr == winerror.ERROR_PIPE_CONNECTED:
                # Client is already connected - signal event
                SetEvent(self.overlapped.hEvent)
            rc = WaitForMultipleObjects(
                (self.hWaitStop, self.overlapped.hEvent), 0, INFINITE
            )
            if rc == WAIT_OBJECT_0:
                # Stop event
                break
            else:
                # Pipe event - spawn thread to deal with it.
                _thread.start_new_thread(self.ProcessClient, (pipeHandle,))
                num_connections = num_connections + 1

        # Sleep to ensure that any new threads are in the list, and then
        # wait for all current threads to finish.
        # What is a better way?
        Sleep(500)
        while self.thread_handles:
            self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING, 5000)
            print("Waiting for %d threads to finish..." % (len(self.thread_handles)))
            WaitForMultipleObjects(self.thread_handles, 1, 3000)
        # Write another event log record.
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STOPPED,
            (self._svc_name_, " after processing %d connections" % (num_connections,)),
        )


if __name__ == "__main__":
    win32serviceutil.HandleCommandLine(TestPipeService)
