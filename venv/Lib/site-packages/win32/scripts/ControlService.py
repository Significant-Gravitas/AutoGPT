# ControlService.py
#
# A simple app which duplicates some of the functionality in the
# Services applet of the control panel.
#
# Suggested enhancements (in no particular order):
#
# 1. When changing the service status, continue to query the status
# of the service until the status change is complete.  Use this
# information to put up some kind of a progress dialog like the CP
# applet does.  Unlike the CP, allow canceling out in the event that
# the status change hangs.
# 2. When starting or stopping a service with dependencies, alert
# the user about the dependent services, then start (or stop) all
# dependent services as appropriate.
# 3. Allow toggling between service view and device view
# 4. Allow configuration of other service parameters such as startup
# name and password.
# 5. Allow connection to remote SCMs.  This is just a matter of
# reconnecting to the SCM on the remote machine; the rest of the
# code should still work the same.
# 6. Either implement the startup parameters or get rid of the editbox.
# 7. Either implement or get rid of "H/W Profiles".
# 8. Either implement or get rid of "Help".
# 9. Improve error handling.  Ideally, this would also include falling
# back to lower levels of functionality for users with less rights.
# Right now, we always try to get all the rights and fail when we can't


import win32con
import win32service
import win32ui
from pywin.mfc import dialog


class StartupDlg(dialog.Dialog):
    IDC_LABEL = 127
    IDC_DEVICE = 128
    IDC_BOOT = 129
    IDC_SYSTEM = 130
    IDC_AUTOMATIC = 131
    IDC_MANUAL = 132
    IDC_DISABLED = 133

    def __init__(self, displayname, service):
        dialog.Dialog.__init__(self, self.GetResource())
        self.name = displayname
        self.service = service

    def __del__(self):
        win32service.CloseServiceHandle(self.service)

    def OnInitDialog(self):
        cfg = win32service.QueryServiceConfig(self.service)
        self.GetDlgItem(self.IDC_BOOT + cfg[1]).SetCheck(1)

        status = win32service.QueryServiceStatus(self.service)
        if (status[0] & win32service.SERVICE_KERNEL_DRIVER) or (
            status[0] & win32service.SERVICE_FILE_SYSTEM_DRIVER
        ):
            # driver
            self.GetDlgItem(self.IDC_LABEL).SetWindowText("Device:")
        else:
            # service
            self.GetDlgItem(self.IDC_LABEL).SetWindowText("Service:")
            self.GetDlgItem(self.IDC_BOOT).EnableWindow(0)
            self.GetDlgItem(self.IDC_SYSTEM).EnableWindow(0)
        self.GetDlgItem(self.IDC_DEVICE).SetWindowText(str(self.name))

        return dialog.Dialog.OnInitDialog(self)

    def OnOK(self):
        self.BeginWaitCursor()
        starttype = (
            self.GetCheckedRadioButton(self.IDC_BOOT, self.IDC_DISABLED) - self.IDC_BOOT
        )
        try:
            win32service.ChangeServiceConfig(
                self.service,
                win32service.SERVICE_NO_CHANGE,
                starttype,
                win32service.SERVICE_NO_CHANGE,
                None,
                None,
                0,
                None,
                None,
                None,
                None,
            )
        except:
            self.MessageBox(
                "Unable to change startup configuration",
                None,
                win32con.MB_ICONEXCLAMATION,
            )
        self.EndWaitCursor()
        return dialog.Dialog.OnOK(self)

    def GetResource(self):
        style = (
            win32con.WS_POPUP
            | win32con.DS_SETFONT
            | win32con.WS_SYSMENU
            | win32con.WS_CAPTION
            | win32con.WS_VISIBLE
            | win32con.DS_MODALFRAME
        )
        exstyle = None
        t = [
            ["Service Startup", (6, 18, 188, 107), style, exstyle, (8, "MS Shell Dlg")],
        ]
        t.append(
            [
                130,
                "Device:",
                self.IDC_LABEL,
                (6, 7, 40, 8),
                win32con.WS_VISIBLE | win32con.WS_CHILD | win32con.SS_LEFT,
            ]
        )
        t.append(
            [
                130,
                "",
                self.IDC_DEVICE,
                (48, 7, 134, 8),
                win32con.WS_VISIBLE | win32con.WS_CHILD | win32con.SS_LEFT,
            ]
        )
        t.append(
            [
                128,
                "Startup Type",
                -1,
                (6, 21, 130, 80),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_GROUP
                | win32con.BS_GROUPBOX,
            ]
        )
        t.append(
            [
                128,
                "&Boot",
                self.IDC_BOOT,
                (12, 33, 39, 10),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_TABSTOP
                | win32con.BS_AUTORADIOBUTTON,
            ]
        )
        t.append(
            [
                128,
                "&System",
                self.IDC_SYSTEM,
                (12, 46, 39, 10),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_TABSTOP
                | win32con.BS_AUTORADIOBUTTON,
            ]
        )
        t.append(
            [
                128,
                "&Automatic",
                self.IDC_AUTOMATIC,
                (12, 59, 118, 10),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_TABSTOP
                | win32con.BS_AUTORADIOBUTTON,
            ]
        )
        t.append(
            [
                128,
                "&Manual",
                self.IDC_MANUAL,
                (12, 72, 118, 10),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_TABSTOP
                | win32con.BS_AUTORADIOBUTTON,
            ]
        )
        t.append(
            [
                128,
                "&Disabled",
                self.IDC_DISABLED,
                (12, 85, 118, 10),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_TABSTOP
                | win32con.BS_AUTORADIOBUTTON,
            ]
        )
        t.append(
            [
                128,
                "OK",
                win32con.IDOK,
                (142, 25, 40, 14),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_TABSTOP
                | win32con.WS_GROUP
                | win32con.BS_DEFPUSHBUTTON,
            ]
        )
        t.append(
            [
                128,
                "Cancel",
                win32con.IDCANCEL,
                (142, 43, 40, 14),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_TABSTOP
                | win32con.BS_PUSHBUTTON,
            ]
        )
        t.append(
            [
                128,
                "&Help",
                win32con.IDHELP,
                (142, 61, 40, 14),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_TABSTOP
                | win32con.BS_PUSHBUTTON,
            ]
        )
        return t


class ServiceDlg(dialog.Dialog):
    IDC_LIST = 128
    IDC_START = 129
    IDC_STOP = 130
    IDC_PAUSE = 131
    IDC_CONTINUE = 132
    IDC_STARTUP = 133
    IDC_PROFILES = 134
    IDC_PARAMS = 135

    def __init__(self, machineName=""):
        dialog.Dialog.__init__(self, self.GetResource())
        self.HookCommand(self.OnListEvent, self.IDC_LIST)
        self.HookCommand(self.OnStartCmd, self.IDC_START)
        self.HookCommand(self.OnStopCmd, self.IDC_STOP)
        self.HookCommand(self.OnPauseCmd, self.IDC_PAUSE)
        self.HookCommand(self.OnContinueCmd, self.IDC_CONTINUE)
        self.HookCommand(self.OnStartupCmd, self.IDC_STARTUP)
        self.machineName = machineName
        self.scm = win32service.OpenSCManager(
            self.machineName, None, win32service.SC_MANAGER_ALL_ACCESS
        )

    def __del__(self):
        win32service.CloseServiceHandle(self.scm)

    def OnInitDialog(self):
        self.listCtrl = self.GetDlgItem(self.IDC_LIST)
        self.listCtrl.SetTabStops([158, 200])
        if self.machineName:
            self.SetWindowText("Services on %s" % self.machineName)
        self.ReloadData()
        return dialog.Dialog.OnInitDialog(self)

    def ReloadData(self):
        service = self.GetSelService()
        self.listCtrl.SetRedraw(0)
        self.listCtrl.ResetContent()
        svcs = win32service.EnumServicesStatus(self.scm)
        i = 0
        self.data = []
        for svc in svcs:
            try:
                status = (
                    "Unknown",
                    "Stopped",
                    "Starting",
                    "Stopping",
                    "Running",
                    "Continuing",
                    "Pausing",
                    "Paused",
                )[svc[2][1]]
            except:
                status = "Unknown"
            s = win32service.OpenService(
                self.scm, svc[0], win32service.SERVICE_ALL_ACCESS
            )
            cfg = win32service.QueryServiceConfig(s)
            try:
                startup = ("Boot", "System", "Automatic", "Manual", "Disabled")[cfg[1]]
            except:
                startup = "Unknown"
            win32service.CloseServiceHandle(s)

            # svc[2][2] control buttons
            pos = self.listCtrl.AddString(str(svc[1]) + "\t" + status + "\t" + startup)
            self.listCtrl.SetItemData(pos, i)
            self.data.append(
                tuple(svc[2])
                + (
                    svc[1],
                    svc[0],
                )
            )
            i = i + 1

            if service and service[1] == svc[0]:
                self.listCtrl.SetCurSel(pos)
        self.OnListEvent(self.IDC_LIST, win32con.LBN_SELCHANGE)
        self.listCtrl.SetRedraw(1)

    def OnListEvent(self, id, code):
        if code == win32con.LBN_SELCHANGE or code == win32con.LBN_SELCANCEL:
            pos = self.listCtrl.GetCurSel()
            if pos >= 0:
                data = self.data[self.listCtrl.GetItemData(pos)][2]
                canstart = (
                    self.data[self.listCtrl.GetItemData(pos)][1]
                    == win32service.SERVICE_STOPPED
                )
            else:
                data = 0
                canstart = 0
            self.GetDlgItem(self.IDC_START).EnableWindow(canstart)
            self.GetDlgItem(self.IDC_STOP).EnableWindow(
                data & win32service.SERVICE_ACCEPT_STOP
            )
            self.GetDlgItem(self.IDC_PAUSE).EnableWindow(
                data & win32service.SERVICE_ACCEPT_PAUSE_CONTINUE
            )
            self.GetDlgItem(self.IDC_CONTINUE).EnableWindow(
                data & win32service.SERVICE_ACCEPT_PAUSE_CONTINUE
            )

    def GetSelService(self):
        pos = self.listCtrl.GetCurSel()
        if pos < 0:
            return None
        pos = self.listCtrl.GetItemData(pos)
        return self.data[pos][-2:]

    def OnStartCmd(self, id, code):
        service = self.GetSelService()
        if not service:
            return
        s = win32service.OpenService(
            self.scm, service[1], win32service.SERVICE_ALL_ACCESS
        )
        win32service.StartService(s, None)
        win32service.CloseServiceHandle(s)
        self.ReloadData()

    def OnStopCmd(self, id, code):
        service = self.GetSelService()
        if not service:
            return
        s = win32service.OpenService(
            self.scm, service[1], win32service.SERVICE_ALL_ACCESS
        )
        win32service.ControlService(s, win32service.SERVICE_CONTROL_STOP)
        win32service.CloseServiceHandle(s)
        self.ReloadData()

    def OnPauseCmd(self, id, code):
        service = self.GetSelService()
        if not service:
            return
        s = win32service.OpenService(
            self.scm, service[1], win32service.SERVICE_ALL_ACCESS
        )
        win32service.ControlService(s, win32service.SERVICE_CONTROL_PAUSE)
        win32service.CloseServiceHandle(s)
        self.ReloadData()

    def OnContinueCmd(self, id, code):
        service = self.GetSelService()
        if not service:
            return
        s = win32service.OpenService(
            self.scm, service[1], win32service.SERVICE_ALL_ACCESS
        )
        win32service.ControlService(s, win32service.SERVICE_CONTROL_CONTINUE)
        win32service.CloseServiceHandle(s)
        self.ReloadData()

    def OnStartupCmd(self, id, code):
        service = self.GetSelService()
        if not service:
            return
        s = win32service.OpenService(
            self.scm, service[1], win32service.SERVICE_ALL_ACCESS
        )
        if StartupDlg(service[0], s).DoModal() == win32con.IDOK:
            self.ReloadData()

    def GetResource(self):
        style = (
            win32con.WS_POPUP
            | win32con.DS_SETFONT
            | win32con.WS_SYSMENU
            | win32con.WS_CAPTION
            | win32con.WS_VISIBLE
            | win32con.DS_MODALFRAME
        )
        exstyle = None
        t = [
            ["Services", (16, 16, 333, 157), style, exstyle, (8, "MS Shell Dlg")],
        ]
        t.append(
            [
                130,
                "Ser&vice",
                -1,
                (6, 6, 70, 8),
                win32con.WS_VISIBLE | win32con.WS_CHILD | win32con.SS_LEFT,
            ]
        )
        t.append(
            [
                130,
                "Status",
                -1,
                (164, 6, 42, 8),
                win32con.WS_VISIBLE | win32con.WS_CHILD | win32con.SS_LEFT,
            ]
        )
        t.append(
            [
                130,
                "Startup",
                -1,
                (206, 6, 50, 8),
                win32con.WS_VISIBLE | win32con.WS_CHILD | win32con.SS_LEFT,
            ]
        )
        t.append(
            [
                131,
                "",
                self.IDC_LIST,
                (6, 16, 255, 106),
                win32con.LBS_USETABSTOPS
                | win32con.LBS_SORT
                | win32con.LBS_NOINTEGRALHEIGHT
                | win32con.WS_BORDER
                | win32con.WS_CHILD
                | win32con.WS_VISIBLE
                | win32con.WS_TABSTOP
                | win32con.LBS_NOTIFY
                | win32con.WS_VSCROLL,
            ]
        )
        t.append(
            [
                128,
                "Close",
                win32con.IDOK,
                (267, 6, 60, 14),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_GROUP
                | win32con.WS_TABSTOP
                | win32con.BS_DEFPUSHBUTTON,
            ]
        )
        t.append(
            [
                128,
                "&Start",
                self.IDC_START,
                (267, 27, 60, 14),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_TABSTOP
                | win32con.BS_PUSHBUTTON,
            ]
        )
        t.append(
            [
                128,
                "S&top",
                self.IDC_STOP,
                (267, 44, 60, 14),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_TABSTOP
                | win32con.BS_PUSHBUTTON,
            ]
        )
        t.append(
            [
                128,
                "&Pause",
                self.IDC_PAUSE,
                (267, 61, 60, 14),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_TABSTOP
                | win32con.BS_PUSHBUTTON,
            ]
        )
        t.append(
            [
                128,
                "&Continue",
                self.IDC_CONTINUE,
                (267, 78, 60, 14),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_TABSTOP
                | win32con.BS_PUSHBUTTON,
            ]
        )
        t.append(
            [
                128,
                "Sta&rtup...",
                self.IDC_STARTUP,
                (267, 99, 60, 14),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_TABSTOP
                | win32con.BS_PUSHBUTTON,
            ]
        )
        t.append(
            [
                128,
                "H&W Profiles...",
                self.IDC_PROFILES,
                (267, 116, 60, 14),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_TABSTOP
                | win32con.BS_PUSHBUTTON,
            ]
        )
        t.append(
            [
                128,
                "&Help",
                win32con.IDHELP,
                (267, 137, 60, 14),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_TABSTOP
                | win32con.BS_PUSHBUTTON,
            ]
        )
        t.append(
            [
                130,
                "St&artup Parameters:",
                -1,
                (6, 128, 70, 8),
                win32con.WS_VISIBLE | win32con.WS_CHILD | win32con.SS_LEFT,
            ]
        )
        t.append(
            [
                129,
                "",
                self.IDC_PARAMS,
                (6, 139, 247, 12),
                win32con.WS_VISIBLE
                | win32con.WS_CHILD
                | win32con.WS_GROUP
                | win32con.WS_BORDER
                | win32con.ES_AUTOHSCROLL,
            ]
        )
        return t


if __name__ == "__main__":
    import sys

    machine = ""
    if len(sys.argv) > 1:
        machine = sys.argv[1]
    ServiceDlg(machine).DoModal()
