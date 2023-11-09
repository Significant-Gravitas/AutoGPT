# A Demo of a service that takes advantage of the additional notifications
# available in later Windows versions.

# Note that all output is written as event log entries - so you must install
# and start the service, then look at the event log for messages as events
# are generated.

# Events are generated for USB device insertion and removal, power state
# changes and hardware profile events - so try putting your computer to
# sleep and waking it, inserting a memory stick, etc then check the event log

# Most event notification support lives around win32gui
import servicemanager
import win32con
import win32event
import win32gui
import win32gui_struct
import win32service
import win32serviceutil

GUID_DEVINTERFACE_USB_DEVICE = "{A5DCBF10-6530-11D2-901F-00C04FB951ED}"


class EventDemoService(win32serviceutil.ServiceFramework):
    _svc_name_ = "PyServiceEventDemo"
    _svc_display_name_ = "Python Service Event Demo"
    _svc_description_ = (
        "Demonstrates a Python service which takes advantage of the extra notifications"
    )

    def __init__(self, args):
        win32serviceutil.ServiceFramework.__init__(self, args)
        self.hWaitStop = win32event.CreateEvent(None, 0, 0, None)
        # register for a device notification - we pass our service handle
        # instead of a window handle.
        filter = win32gui_struct.PackDEV_BROADCAST_DEVICEINTERFACE(
            GUID_DEVINTERFACE_USB_DEVICE
        )
        self.hdn = win32gui.RegisterDeviceNotification(
            self.ssh, filter, win32con.DEVICE_NOTIFY_SERVICE_HANDLE
        )

    # Override the base class so we can accept additional events.
    def GetAcceptedControls(self):
        # say we accept them all.
        rc = win32serviceutil.ServiceFramework.GetAcceptedControls(self)
        rc |= (
            win32service.SERVICE_ACCEPT_PARAMCHANGE
            | win32service.SERVICE_ACCEPT_NETBINDCHANGE
            | win32service.SERVICE_CONTROL_DEVICEEVENT
            | win32service.SERVICE_ACCEPT_HARDWAREPROFILECHANGE
            | win32service.SERVICE_ACCEPT_POWEREVENT
            | win32service.SERVICE_ACCEPT_SESSIONCHANGE
        )
        return rc

    # All extra events are sent via SvcOtherEx (SvcOther remains as a
    # function taking only the first args for backwards compat)
    def SvcOtherEx(self, control, event_type, data):
        # This is only showing a few of the extra events - see the MSDN
        # docs for "HandlerEx callback" for more info.
        if control == win32service.SERVICE_CONTROL_DEVICEEVENT:
            info = win32gui_struct.UnpackDEV_BROADCAST(data)
            msg = "A device event occurred: %x - %s" % (event_type, info)
        elif control == win32service.SERVICE_CONTROL_HARDWAREPROFILECHANGE:
            msg = "A hardware profile changed: type=%s, data=%s" % (event_type, data)
        elif control == win32service.SERVICE_CONTROL_POWEREVENT:
            msg = "A power event: setting %s" % data
        elif control == win32service.SERVICE_CONTROL_SESSIONCHANGE:
            # data is a single elt tuple, but this could potentially grow
            # in the future if the win32 struct does
            msg = "Session event: type=%s, data=%s" % (event_type, data)
        else:
            msg = "Other event: code=%d, type=%s, data=%s" % (control, event_type, data)

        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            0xF000,  #  generic message
            (msg, ""),
        )

    def SvcStop(self):
        self.ReportServiceStatus(win32service.SERVICE_STOP_PENDING)
        win32event.SetEvent(self.hWaitStop)

    def SvcDoRun(self):
        # do nothing at all - just wait to be stopped
        win32event.WaitForSingleObject(self.hWaitStop, win32event.INFINITE)
        # Write a stop message.
        servicemanager.LogMsg(
            servicemanager.EVENTLOG_INFORMATION_TYPE,
            servicemanager.PYS_SERVICE_STOPPED,
            (self._svc_name_, ""),
        )


if __name__ == "__main__":
    win32serviceutil.HandleCommandLine(EventDemoService)
