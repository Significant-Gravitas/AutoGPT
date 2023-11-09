# sliderdemo.py
# Demo of the slider control courtesy of Mike Fletcher.

import win32con
import win32ui
from pywin.mfc import dialog


class MyDialog(dialog.Dialog):
    """
    Example using simple controls
    """

    _dialogstyle = (
        win32con.WS_MINIMIZEBOX
        | win32con.WS_DLGFRAME
        | win32con.DS_MODALFRAME
        | win32con.WS_POPUP
        | win32con.WS_VISIBLE
        | win32con.WS_CAPTION
        | win32con.WS_SYSMENU
        | win32con.DS_SETFONT
    )
    _buttonstyle = (
        win32con.BS_PUSHBUTTON
        | win32con.WS_TABSTOP
        | win32con.WS_CHILD
        | win32con.WS_VISIBLE
    )
    ### The static template, contains all "normal" dialog items
    DIALOGTEMPLATE = [
        # the dialog itself is the first element in the template
        ["Example slider", (0, 0, 50, 43), _dialogstyle, None, (8, "MS SansSerif")],
        # rest of elements are the controls within the dialog
        # standard "Close" button
        [128, "Close", win32con.IDCANCEL, (0, 30, 50, 13), _buttonstyle],
    ]
    ### ID of the control to be created during dialog initialisation
    IDC_SLIDER = 9500

    def __init__(self):
        dialog.Dialog.__init__(self, self.DIALOGTEMPLATE)

    def OnInitDialog(self):
        rc = dialog.Dialog.OnInitDialog(self)
        # now initialise your controls that you want to create
        # programmatically, including those which are OLE controls
        # those created directly by win32ui.Create*
        # and your "custom controls" which are subclasses/whatever
        win32ui.EnableControlContainer()
        self.slider = win32ui.CreateSliderCtrl()
        self.slider.CreateWindow(
            win32con.WS_TABSTOP | win32con.WS_VISIBLE,
            (0, 0, 100, 30),
            self._obj_,
            self.IDC_SLIDER,
        )
        self.HookMessage(self.OnSliderMove, win32con.WM_HSCROLL)
        return rc

    def OnSliderMove(self, params):
        print("Slider moved")

    def OnCancel(self):
        print("The slider control is at position", self.slider.GetPos())
        self._obj_.OnCancel()


###
def demo():
    dia = MyDialog()
    dia.DoModal()


if __name__ == "__main__":
    demo()
