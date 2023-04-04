# Demo of ToolBars

# Shows the toolbar control.
# Demos how to make custom tooltips, etc.

import commctrl
import win32api
import win32con
import win32ui
from pywin.mfc import afxres, docview, window


class GenericFrame(window.MDIChildWnd):
    def OnCreateClient(self, cp, context):
        # handlers for toolbar buttons
        self.HookCommand(self.OnPrevious, 401)
        self.HookCommand(self.OnNext, 402)
        # Its not necessary for us to hook both of these - the
        # common controls should fall-back all by themselves.
        # Indeed, given we hook TTN_NEEDTEXTW, commctrl.TTN_NEEDTEXTA
        # will not be called.
        self.HookNotify(self.GetTTText, commctrl.TTN_NEEDTEXT)
        self.HookNotify(self.GetTTText, commctrl.TTN_NEEDTEXTW)

        # 		parent = win32ui.GetMainFrame()
        parent = self
        style = (
            win32con.WS_CHILD
            | win32con.WS_VISIBLE
            | afxres.CBRS_SIZE_DYNAMIC
            | afxres.CBRS_TOP
            | afxres.CBRS_TOOLTIPS
            | afxres.CBRS_FLYBY
        )

        buttons = (win32ui.ID_APP_ABOUT, win32ui.ID_VIEW_INTERACTIVE)
        bitmap = win32ui.IDB_BROWSER_HIER
        tbid = 0xE840
        self.toolbar = tb = win32ui.CreateToolBar(parent, style, tbid)
        tb.LoadBitmap(bitmap)
        tb.SetButtons(buttons)

        tb.EnableDocking(afxres.CBRS_ALIGN_ANY)
        tb.SetWindowText("Test")
        parent.EnableDocking(afxres.CBRS_ALIGN_ANY)
        parent.DockControlBar(tb)
        parent.LoadBarState("ToolbarTest")
        window.MDIChildWnd.OnCreateClient(self, cp, context)
        return 1

    def OnDestroy(self, msg):
        self.SaveBarState("ToolbarTest")

    def GetTTText(self, std, extra):
        (hwndFrom, idFrom, code) = std
        text, hinst, flags = extra
        if flags & commctrl.TTF_IDISHWND:
            return  # Not handled
        if idFrom == win32ui.ID_APP_ABOUT:
            # our 'extra' return value needs to be the following
            # entries from a NMTTDISPINFO[W] struct:
            # (szText, hinst, uFlags).  None means 'don't change
            # the value'
            return 0, ("It works!", None, None)
        return None  # not handled.

    def GetMessageString(self, id):
        if id == win32ui.ID_APP_ABOUT:
            return "Dialog Test\nTest"
        else:
            return self._obj_.GetMessageString(id)

    def OnSize(self, params):
        print("OnSize called with ", params)

    def OnNext(self, id, cmd):
        print("OnNext called")

    def OnPrevious(self, id, cmd):
        print("OnPrevious called")


msg = """\
This toolbar was dynamically created.\r
\r
The first item's tooltips is provided by Python code.\r
\r
(Dont close the window with the toolbar in a floating state - it may not re-appear!)\r
"""


def test():
    template = docview.DocTemplate(
        win32ui.IDR_PYTHONTYPE, None, GenericFrame, docview.EditView
    )
    doc = template.OpenDocumentFile(None)
    doc.SetTitle("Toolbar Test")
    view = doc.GetFirstView()
    view.SetWindowText(msg)


if __name__ == "__main__":
    import demoutils

    if demoutils.NeedGoodGUI():
        test()
