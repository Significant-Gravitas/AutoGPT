# dojobapp - do a job, show the result in a dialog, and exit.
#
# Very simple - faily minimal dialog based app.
#
# This should be run using the command line:
# pythonwin /app demos\dojobapp.py


import win32api
import win32con
import win32ui
from pywin.framework import app, dlgappcore


class DoJobAppDialog(dlgappcore.AppDialog):
    softspace = 1

    def __init__(self, appName=""):
        self.appName = appName
        dlgappcore.AppDialog.__init__(self, win32ui.IDD_GENERAL_STATUS)

    def PreDoModal(self):
        pass

    def ProcessArgs(self, args):
        pass

    def OnInitDialog(self):
        self.SetWindowText(self.appName)
        butCancel = self.GetDlgItem(win32con.IDCANCEL)
        butCancel.ShowWindow(win32con.SW_HIDE)
        p1 = self.GetDlgItem(win32ui.IDC_PROMPT1)
        p2 = self.GetDlgItem(win32ui.IDC_PROMPT2)

        # Do something here!

        p1.SetWindowText("Hello there")
        p2.SetWindowText("from the demo")

    def OnDestroy(self, msg):
        pass


# 	def OnOK(self):
# 		pass
# 	def OnCancel(self): default behaviour - cancel == close.
# 		return


class DoJobDialogApp(dlgappcore.DialogApp):
    def CreateDialog(self):
        return DoJobAppDialog("Do Something")


class CopyToDialogApp(DoJobDialogApp):
    def __init__(self):
        DoJobDialogApp.__init__(self)


app.AppBuilder = DoJobDialogApp


def t():
    t = DoJobAppDialog("Copy To")
    t.DoModal()
    return t


if __name__ == "__main__":
    import demoutils

    demoutils.NeedApp()
