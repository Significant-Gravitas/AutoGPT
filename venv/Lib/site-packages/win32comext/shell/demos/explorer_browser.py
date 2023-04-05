# A sample of using Vista's IExplorerBrowser interfaces...
# Currently doesn't quite work:
# * CPU sits at 100% while running.

import sys

import pythoncom
import win32api
import win32con
import win32gui
from win32com.server.util import unwrap, wrap
from win32com.shell import shell, shellcon

# event handler for the browser.
IExplorerBrowserEvents_Methods = """OnNavigationComplete OnNavigationFailed 
                                    OnNavigationPending OnViewCreated""".split()


class EventHandler:
    _com_interfaces_ = [shell.IID_IExplorerBrowserEvents]
    _public_methods_ = IExplorerBrowserEvents_Methods

    def OnNavigationComplete(self, pidl):
        print("OnNavComplete", pidl)

    def OnNavigationFailed(self, pidl):
        print("OnNavigationFailed", pidl)

    def OnNavigationPending(self, pidl):
        print("OnNavigationPending", pidl)

    def OnViewCreated(self, view):
        print("OnViewCreated", view)
        # And if our demo view has been registered, it may well
        # be that view!
        try:
            pyview = unwrap(view)
            print("and look - its a Python implemented view!", pyview)
        except ValueError:
            pass


class MainWindow:
    def __init__(self):
        message_map = {
            win32con.WM_DESTROY: self.OnDestroy,
            win32con.WM_COMMAND: self.OnCommand,
            win32con.WM_SIZE: self.OnSize,
        }
        # Register the Window class.
        wc = win32gui.WNDCLASS()
        hinst = wc.hInstance = win32api.GetModuleHandle(None)
        wc.lpszClassName = "test_explorer_browser"
        wc.lpfnWndProc = message_map  # could also specify a wndproc.
        classAtom = win32gui.RegisterClass(wc)
        # Create the Window.
        style = win32con.WS_OVERLAPPEDWINDOW | win32con.WS_VISIBLE
        self.hwnd = win32gui.CreateWindow(
            classAtom,
            "Python IExplorerBrowser demo",
            style,
            0,
            0,
            win32con.CW_USEDEFAULT,
            win32con.CW_USEDEFAULT,
            0,
            0,
            hinst,
            None,
        )
        eb = pythoncom.CoCreateInstance(
            shellcon.CLSID_ExplorerBrowser,
            None,
            pythoncom.CLSCTX_ALL,
            shell.IID_IExplorerBrowser,
        )
        # as per MSDN docs, hook up events early
        self.event_cookie = eb.Advise(wrap(EventHandler()))

        eb.SetOptions(shellcon.EBO_SHOWFRAMES)
        rect = win32gui.GetClientRect(self.hwnd)
        # Set the flags such that the folders autoarrange and non web view is presented
        flags = (shellcon.FVM_LIST, shellcon.FWF_AUTOARRANGE | shellcon.FWF_NOWEBVIEW)
        eb.Initialize(self.hwnd, rect, (0, shellcon.FVM_DETAILS))
        if len(sys.argv) == 2:
            # If an arg was specified, ask the desktop parse it.
            # You can pass anything explorer accepts as its '/e' argument -
            # eg, "::{guid}\::{guid}" etc.
            # "::{20D04FE0-3AEA-1069-A2D8-08002B30309D}" is "My Computer"
            pidl = shell.SHGetDesktopFolder().ParseDisplayName(0, None, sys.argv[1])[1]
        else:
            # And start browsing at the root of the namespace.
            pidl = []
        eb.BrowseToIDList(pidl, shellcon.SBSP_ABSOLUTE)
        # and for some reason the "Folder" view in the navigator pane doesn't
        # magically synchronize itself - so let's do that ourself.
        # Get the tree control.
        sp = eb.QueryInterface(pythoncom.IID_IServiceProvider)
        try:
            tree = sp.QueryService(
                shell.IID_INameSpaceTreeControl, shell.IID_INameSpaceTreeControl
            )
        except pythoncom.com_error as exc:
            # this should really only fail if no "nav" frame exists...
            print(
                "Strange - failed to get the tree control even though "
                "we asked for a EBO_SHOWFRAMES"
            )
            print(exc)
        else:
            # get the IShellItem for the selection.
            si = shell.SHCreateItemFromIDList(pidl, shell.IID_IShellItem)
            # set it to selected.
            tree.SetItemState(si, shellcon.NSTCIS_SELECTED, shellcon.NSTCIS_SELECTED)

        # eb.FillFromObject(None, shellcon.EBF_NODROPTARGET);
        # eb.SetEmptyText("No known folders yet...");
        self.eb = eb

    def OnCommand(self, hwnd, msg, wparam, lparam):
        pass

    def OnDestroy(self, hwnd, msg, wparam, lparam):
        print("tearing down ExplorerBrowser...")
        self.eb.Unadvise(self.event_cookie)
        self.eb.Destroy()
        self.eb = None
        print("shutting down app...")
        win32gui.PostQuitMessage(0)

    def OnSize(self, hwnd, msg, wparam, lparam):
        x = win32api.LOWORD(lparam)
        y = win32api.HIWORD(lparam)
        self.eb.SetRect(None, (0, 0, x, y))


def main():
    w = MainWindow()
    win32gui.PumpMessages()


if __name__ == "__main__":
    main()
