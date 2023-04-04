# ocxserialtest.py
#
# Sample that uses the mscomm OCX to talk to a serial
# device.

# Very simple -  queries a modem for ATI responses

import pythoncom
import win32con
import win32ui
import win32uiole
from pywin.mfc import activex, dialog
from win32com.client import gencache

SERIAL_SETTINGS = "19200,n,8,1"
SERIAL_PORT = 2

win32ui.DoWaitCursor(1)
serialModule = gencache.EnsureModule("{648A5603-2C6E-101B-82B6-000000000014}", 0, 1, 1)
win32ui.DoWaitCursor(0)
if serialModule is None:
    raise ImportError("MS COMM Control does not appear to be installed on the PC")


def MakeDlgTemplate():
    style = (
        win32con.DS_MODALFRAME
        | win32con.WS_POPUP
        | win32con.WS_VISIBLE
        | win32con.WS_CAPTION
        | win32con.WS_SYSMENU
        | win32con.DS_SETFONT
    )
    cs = win32con.WS_CHILD | win32con.WS_VISIBLE
    dlg = [
        ["Very Basic Terminal", (0, 0, 350, 180), style, None, (8, "MS Sans Serif")],
    ]
    s = win32con.WS_TABSTOP | cs
    dlg.append(
        [
            "RICHEDIT",
            None,
            132,
            (5, 5, 340, 170),
            s
            | win32con.ES_WANTRETURN
            | win32con.ES_MULTILINE
            | win32con.ES_AUTOVSCROLL
            | win32con.WS_VSCROLL,
        ]
    )
    return dlg


####################################
#
# Serial Control
#
class MySerialControl(activex.Control, serialModule.MSComm):
    def __init__(self, parent):
        activex.Control.__init__(self)
        serialModule.MSComm.__init__(self)
        self.parent = parent

    def OnComm(self):
        self.parent.OnComm()


class TestSerDialog(dialog.Dialog):
    def __init__(self, *args):
        dialog.Dialog.__init__(*(self,) + args)
        self.olectl = None

    def OnComm(self):
        event = self.olectl.CommEvent
        if event == serialModule.OnCommConstants.comEvReceive:
            self.editwindow.ReplaceSel(self.olectl.Input)

    def OnKey(self, key):
        if self.olectl:
            self.olectl.Output = chr(key)

    def OnInitDialog(self):
        rc = dialog.Dialog.OnInitDialog(self)
        self.editwindow = self.GetDlgItem(132)
        self.editwindow.HookAllKeyStrokes(self.OnKey)

        self.olectl = MySerialControl(self)
        try:
            self.olectl.CreateControl(
                "OCX",
                win32con.WS_TABSTOP | win32con.WS_VISIBLE,
                (7, 43, 500, 300),
                self._obj_,
                131,
            )
        except win32ui.error:
            self.MessageBox("The Serial Control could not be created")
            self.olectl = None
            self.EndDialog(win32con.IDCANCEL)
        if self.olectl:
            self.olectl.Settings = SERIAL_SETTINGS
            self.olectl.CommPort = SERIAL_PORT
            self.olectl.RThreshold = 1
            try:
                self.olectl.PortOpen = 1
            except pythoncom.com_error as details:
                print(
                    "Could not open the specified serial port - %s"
                    % (details.excepinfo[2])
                )
                self.EndDialog(win32con.IDCANCEL)
        return rc

    def OnDestroy(self, msg):
        if self.olectl:
            try:
                self.olectl.PortOpen = 0
            except pythoncom.com_error as details:
                print("Error closing port - %s" % (details.excepinfo[2]))
        return dialog.Dialog.OnDestroy(self, msg)


def test():
    d = TestSerDialog(MakeDlgTemplate())
    d.DoModal()


if __name__ == "__main__":
    from . import demoutils

    if demoutils.NeedGoodGUI():
        test()
