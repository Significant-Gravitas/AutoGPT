# A demo of the win32rcparser module and using win32gui

import os

import commctrl
import win32api
import win32con
import win32gui
import win32rcparser

this_dir = os.path.abspath(os.path.dirname(__file__))
g_rcname = os.path.abspath(
    os.path.join(this_dir, "..", "test", "win32rcparser", "test.rc")
)

if not os.path.isfile(g_rcname):
    raise RuntimeError("Can't locate test.rc (should be at '%s')" % (g_rcname,))


class DemoWindow:
    def __init__(self, dlg_template):
        self.dlg_template = dlg_template

    def CreateWindow(self):
        self._DoCreate(win32gui.CreateDialogIndirect)

    def DoModal(self):
        return self._DoCreate(win32gui.DialogBoxIndirect)

    def _DoCreate(self, fn):
        message_map = {
            win32con.WM_INITDIALOG: self.OnInitDialog,
            win32con.WM_CLOSE: self.OnClose,
            win32con.WM_DESTROY: self.OnDestroy,
            win32con.WM_COMMAND: self.OnCommand,
        }
        return fn(0, self.dlg_template, 0, message_map)

    def OnInitDialog(self, hwnd, msg, wparam, lparam):
        self.hwnd = hwnd
        # centre the dialog
        desktop = win32gui.GetDesktopWindow()
        l, t, r, b = win32gui.GetWindowRect(self.hwnd)
        dt_l, dt_t, dt_r, dt_b = win32gui.GetWindowRect(desktop)
        centre_x, centre_y = win32gui.ClientToScreen(
            desktop, ((dt_r - dt_l) // 2, (dt_b - dt_t) // 2)
        )
        win32gui.MoveWindow(
            hwnd, centre_x - (r // 2), centre_y - (b // 2), r - l, b - t, 0
        )

    def OnCommand(self, hwnd, msg, wparam, lparam):
        # Needed to make OK/Cancel work - no other controls are handled.
        id = win32api.LOWORD(wparam)
        if id in [win32con.IDOK, win32con.IDCANCEL]:
            win32gui.EndDialog(hwnd, id)

    def OnClose(self, hwnd, msg, wparam, lparam):
        win32gui.EndDialog(hwnd, 0)

    def OnDestroy(self, hwnd, msg, wparam, lparam):
        pass


def DemoModal():
    # Load the .rc file.
    resources = win32rcparser.Parse(g_rcname)
    for id, ddef in resources.dialogs.items():
        print("Displaying dialog", id)
        w = DemoWindow(ddef)
        w.DoModal()


if __name__ == "__main__":
    flags = 0
    for flag in """ICC_DATE_CLASSES ICC_ANIMATE_CLASS ICC_ANIMATE_CLASS 
                   ICC_BAR_CLASSES ICC_COOL_CLASSES ICC_DATE_CLASSES
                   ICC_HOTKEY_CLASS ICC_INTERNET_CLASSES ICC_LISTVIEW_CLASSES
                   ICC_PAGESCROLLER_CLASS ICC_PROGRESS_CLASS ICC_TAB_CLASSES
                   ICC_TREEVIEW_CLASSES ICC_UPDOWN_CLASS ICC_USEREX_CLASSES
                   ICC_WIN95_CLASSES  """.split():
        flags |= getattr(commctrl, flag)
    win32gui.InitCommonControlsEx(flags)
    # Need to do this go get rich-edit working.
    win32api.LoadLibrary("riched20.dll")
    DemoModal()
