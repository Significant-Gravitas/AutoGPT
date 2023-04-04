#
# Window creation example
#
# 	This example creates a minimal "control" that just fills in its
# 	window with red.  To make your own control, subclass Control and
# 	write your own OnPaint() method.  See PyCWnd.HookMessage for what
# 	the parameters to OnPaint are.
#

import win32api
import win32con
import win32ui
from pywin.mfc import dialog, window


class Control(window.Wnd):
    """Generic control class"""

    def __init__(self):
        window.Wnd.__init__(self, win32ui.CreateWnd())

    def OnPaint(self):
        dc, paintStruct = self.BeginPaint()
        self.DoPaint(dc)
        self.EndPaint(paintStruct)

    def DoPaint(self, dc):  # Override this!
        pass


class RedBox(Control):
    def DoPaint(self, dc):
        dc.FillSolidRect(self.GetClientRect(), win32api.RGB(255, 0, 0))


class RedBoxWithPie(RedBox):
    def DoPaint(self, dc):
        RedBox.DoPaint(self, dc)
        r = self.GetClientRect()
        dc.Pie(r[0], r[1], r[2], r[3], 0, 0, r[2], r[3] // 2)


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

    w = 64
    h = 64

    dlg = [
        ["Red box", (0, 0, w, h), style, None, (8, "MS Sans Serif")],
    ]

    s = win32con.WS_TABSTOP | cs

    dlg.append(
        [
            128,
            "Cancel",
            win32con.IDCANCEL,
            (7, h - 18, 50, 14),
            s | win32con.BS_PUSHBUTTON,
        ]
    )

    return dlg


class TestDialog(dialog.Dialog):
    def OnInitDialog(self):
        rc = dialog.Dialog.OnInitDialog(self)
        self.redbox = RedBox()
        self.redbox.CreateWindow(
            None,
            "RedBox",
            win32con.WS_CHILD | win32con.WS_VISIBLE,
            (5, 5, 90, 68),
            self,
            1003,
        )
        return rc


class TestPieDialog(dialog.Dialog):
    def OnInitDialog(self):
        rc = dialog.Dialog.OnInitDialog(self)
        self.control = RedBoxWithPie()
        self.control.CreateWindow(
            None,
            "RedBox with Pie",
            win32con.WS_CHILD | win32con.WS_VISIBLE,
            (5, 5, 90, 68),
            self,
            1003,
        )


def demo(modal=0):
    d = TestPieDialog(MakeDlgTemplate())
    if modal:
        d.DoModal()
    else:
        d.CreateWindow()


if __name__ == "__main__":
    demo(1)
