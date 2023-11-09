#
# Progress bar control example
#
# 	PyCProgressCtrl encapsulates the MFC CProgressCtrl class.  To use it,
# 	you:
#
# 	- Create the control with win32ui.CreateProgressCtrl()
# 	- Create the control window with PyCProgressCtrl.CreateWindow()
# 	- Initialize the range if you want it to be other than (0, 100) using
# 	  PyCProgressCtrl.SetRange()
# 	- Either:
# 	  - Set the step size with PyCProgressCtrl.SetStep(), and
# 	  - Increment using PyCProgressCtrl.StepIt()
# 	  or:
# 	  - Set the amount completed using PyCProgressCtrl.SetPos()
#
# Example and progress bar code courtesy of KDL Technologies, Ltd., Hong Kong SAR, China.
#

import win32con
import win32ui
from pywin.mfc import dialog


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

    w = 215
    h = 36

    dlg = [
        [
            "Progress bar control example",
            (0, 0, w, h),
            style,
            None,
            (8, "MS Sans Serif"),
        ],
    ]

    s = win32con.WS_TABSTOP | cs

    dlg.append(
        [
            128,
            "Tick",
            win32con.IDOK,
            (10, h - 18, 50, 14),
            s | win32con.BS_DEFPUSHBUTTON,
        ]
    )

    dlg.append(
        [
            128,
            "Cancel",
            win32con.IDCANCEL,
            (w - 60, h - 18, 50, 14),
            s | win32con.BS_PUSHBUTTON,
        ]
    )

    return dlg


class TestDialog(dialog.Dialog):
    def OnInitDialog(self):
        rc = dialog.Dialog.OnInitDialog(self)
        self.pbar = win32ui.CreateProgressCtrl()
        self.pbar.CreateWindow(
            win32con.WS_CHILD | win32con.WS_VISIBLE, (10, 10, 310, 24), self, 1001
        )
        # self.pbar.SetStep (5)
        self.progress = 0
        self.pincr = 5
        return rc

    def OnOK(self):
        # NB: StepIt wraps at the end if you increment past the upper limit!
        # self.pbar.StepIt()
        self.progress = self.progress + self.pincr
        if self.progress > 100:
            self.progress = 100
        if self.progress <= 100:
            self.pbar.SetPos(self.progress)


def demo(modal=0):
    d = TestDialog(MakeDlgTemplate())
    if modal:
        d.DoModal()
    else:
        d.CreateWindow()


if __name__ == "__main__":
    demo(1)
