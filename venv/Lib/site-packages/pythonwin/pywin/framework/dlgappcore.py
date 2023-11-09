# dlgappcore.
#
# base classes for dialog based apps.


import win32api
import win32con
import win32ui
from pywin.mfc import dialog

from . import app

error = "Dialog Application Error"


class AppDialog(dialog.Dialog):
    "The dialog box for the application"

    def __init__(self, id, dll=None):
        self.iconId = win32ui.IDR_MAINFRAME
        dialog.Dialog.__init__(self, id, dll)

    def OnInitDialog(self):
        return dialog.Dialog.OnInitDialog(self)

    # Provide support for a dlg app using an icon
    def OnPaint(self):
        if not self.IsIconic():
            return self._obj_.OnPaint()
        self.DefWindowProc(win32con.WM_ICONERASEBKGND, dc.GetHandleOutput(), 0)
        left, top, right, bottom = self.GetClientRect()
        left = (right - win32api.GetSystemMetrics(win32con.SM_CXICON)) >> 1
        top = (bottom - win32api.GetSystemMetrics(win32con.SM_CYICON)) >> 1
        hIcon = win32ui.GetApp().LoadIcon(self.iconId)
        self.GetDC().DrawIcon((left, top), hIcon)

    # Only needed to provide a minimized icon (and this seems
    # less important under win95/NT4
    def OnEraseBkgnd(self, dc):
        if self.IsIconic():
            return 1
        else:
            return self._obj_.OnEraseBkgnd(dc)

    def OnQueryDragIcon(self):
        return win32ui.GetApp().LoadIcon(self.iconId)

    def PreDoModal(self):
        pass


class DialogApp(app.CApp):
    "An application class, for an app with main dialog box"

    def InitInstance(self):
        # 		win32ui.SetProfileFileName('dlgapp.ini')
        win32ui.LoadStdProfileSettings()
        win32ui.EnableControlContainer()
        win32ui.Enable3dControls()
        self.dlg = self.frame = self.CreateDialog()

        if self.frame is None:
            raise error("No dialog was created by CreateDialog()")
            return

        self._obj_.InitDlgInstance(self.dlg)
        self.PreDoModal()
        self.dlg.PreDoModal()
        self.dlg.DoModal()

    def CreateDialog(self):
        pass

    def PreDoModal(self):
        pass
