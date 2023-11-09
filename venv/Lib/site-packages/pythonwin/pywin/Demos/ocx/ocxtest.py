# OCX Tester for Pythonwin
#
# This file _is_ ready to run.  All that is required is that the OCXs being tested
# are installed on your machine.
#
# The .py files behind the OCXs will be automatically generated and imported.

import glob
import os

import win32api
import win32con
import win32ui
import win32uiole
from pywin.mfc import activex, dialog, window
from win32com.client import gencache


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
        ["OCX Demos", (0, 0, 350, 350), style, None, (8, "MS Sans Serif")],
    ]
    s = win32con.WS_TABSTOP | cs
    # 	dlg.append([131, None, 130, (5, 40, 110, 48),
    # 		s | win32con.LBS_NOTIFY | win32con.LBS_SORT | win32con.LBS_NOINTEGRALHEIGHT | win32con.WS_VSCROLL | win32con.WS_BORDER])
    # 	dlg.append(["{8E27C92B-1264-101C-8A2F-040224009C02}", None, 131, (5, 40, 110, 48),win32con.WS_TABSTOP])

    dlg.append(
        [128, "About", win32con.IDOK, (124, 5, 50, 14), s | win32con.BS_DEFPUSHBUTTON]
    )
    s = win32con.BS_PUSHBUTTON | s
    dlg.append([128, "Close", win32con.IDCANCEL, (124, 22, 50, 14), s])

    return dlg


####################################
#
# Calendar test code
#


def GetTestCalendarClass():
    global calendarParentModule
    win32ui.DoWaitCursor(1)
    calendarParentModule = gencache.EnsureModule(
        "{8E27C92E-1264-101C-8A2F-040224009C02}", 0, 7, 0
    )
    win32ui.DoWaitCursor(0)
    if calendarParentModule is None:
        return None

    class TestCalDialog(dialog.Dialog):
        def OnInitDialog(self):
            class MyCal(activex.Control, calendarParentModule.Calendar):
                def OnAfterUpdate(self):
                    print("OnAfterUpdate")

                def OnClick(self):
                    print("OnClick")

                def OnDblClick(self):
                    print("OnDblClick")

                def OnKeyDown(self, KeyCode, Shift):
                    print("OnKeyDown", KeyCode, Shift)

                def OnKeyPress(self, KeyAscii):
                    print("OnKeyPress", KeyAscii)

                def OnKeyUp(self, KeyCode, Shift):
                    print("OnKeyUp", KeyCode, Shift)

                def OnBeforeUpdate(self, Cancel):
                    print("OnBeforeUpdate", Cancel)

                def OnNewMonth(self):
                    print("OnNewMonth")

                def OnNewYear(self):
                    print("OnNewYear")

            rc = dialog.Dialog.OnInitDialog(self)
            self.olectl = MyCal()
            try:
                self.olectl.CreateControl(
                    "OCX",
                    win32con.WS_TABSTOP | win32con.WS_VISIBLE,
                    (7, 43, 500, 300),
                    self._obj_,
                    131,
                )
            except win32ui.error:
                self.MessageBox("The Calendar Control could not be created")
                self.olectl = None
                self.EndDialog(win32con.IDCANCEL)

            return rc

        def OnOK(self):
            self.olectl.AboutBox()

    return TestCalDialog


####################################
#
# Video Control
#
def GetTestVideoModule():
    global videoControlModule, videoControlFileName
    win32ui.DoWaitCursor(1)
    videoControlModule = gencache.EnsureModule(
        "{05589FA0-C356-11CE-BF01-00AA0055595A}", 0, 2, 0
    )
    win32ui.DoWaitCursor(0)
    if videoControlModule is None:
        return None
    fnames = glob.glob(os.path.join(win32api.GetWindowsDirectory(), "*.avi"))
    if not fnames:
        print("No AVI files available in system directory")
        return None
    videoControlFileName = fnames[0]
    return videoControlModule


def GetTestVideoDialogClass():
    if GetTestVideoModule() is None:
        return None

    class TestVideoDialog(dialog.Dialog):
        def OnInitDialog(self):
            rc = dialog.Dialog.OnInitDialog(self)
            try:
                self.olectl = activex.MakeControlInstance(
                    videoControlModule.ActiveMovie
                )
                self.olectl.CreateControl(
                    "",
                    win32con.WS_TABSTOP | win32con.WS_VISIBLE,
                    (7, 43, 500, 300),
                    self._obj_,
                    131,
                )
            except win32ui.error:
                self.MessageBox("The Video Control could not be created")
                self.olectl = None
                self.EndDialog(win32con.IDCANCEL)
                return

            self.olectl.FileName = videoControlFileName
            # 			self.olectl.Run()
            return rc

        def OnOK(self):
            self.olectl.AboutBox()

    return TestVideoDialog


###############
#
# An OCX in an MDI Frame
#
class OCXFrame(window.MDIChildWnd):
    def __init__(self):
        pass  # Dont call base class doc/view version...

    def Create(self, controlClass, title, rect=None, parent=None):
        style = win32con.WS_CHILD | win32con.WS_VISIBLE | win32con.WS_OVERLAPPEDWINDOW
        self._obj_ = win32ui.CreateMDIChild()
        self._obj_.AttachObject(self)
        self._obj_.CreateWindow(None, title, style, rect, parent)

        rect = self.GetClientRect()
        rect = (0, 0, rect[2] - rect[0], rect[3] - rect[1])
        self.ocx = controlClass()
        self.ocx.CreateControl(
            "", win32con.WS_VISIBLE | win32con.WS_CHILD, rect, self, 1000
        )


def MDITest():
    calendarParentModule = gencache.EnsureModule(
        "{8E27C92E-1264-101C-8A2F-040224009C02}", 0, 7, 0
    )

    class MyCal(activex.Control, calendarParentModule.Calendar):
        def OnAfterUpdate(self):
            print("OnAfterUpdate")

        def OnClick(self):
            print("OnClick")

    f = OCXFrame()
    f.Create(MyCal, "Calendar Test")


def test1():
    klass = GetTestCalendarClass()
    if klass is None:
        print(
            "Can not test the MSAccess Calendar control - it does not appear to be installed"
        )
        return

    d = klass(MakeDlgTemplate())
    d.DoModal()


def test2():
    klass = GetTestVideoDialogClass()
    if klass is None:
        print("Can not test the Video OCX - it does not appear to be installed,")
        print("or no AVI files can be found.")
        return
    d = klass(MakeDlgTemplate())
    d.DoModal()
    d = None


def test3():
    d = TestCOMMDialog(MakeDlgTemplate())
    d.DoModal()
    d = None


def testall():
    test1()
    test2()


def demo():
    testall()


if __name__ == "__main__":
    from . import demoutils

    if demoutils.NeedGoodGUI():
        testall()
