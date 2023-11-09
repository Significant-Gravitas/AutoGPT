##
## helloapp.py
##
##
## A nice, small 'hello world' Pythonwin application.
## NOT an MDI application - just a single, normal, top-level window.
##
## MUST be run with the command line "pythonwin.exe /app helloapp.py"
## (or if you are really keen, rename "pythonwin.exe" to something else, then
## using MSVC or similar, edit the string section in the .EXE to name this file)
##
## Originally by Willy Heineman <wheineman@uconect.net>


import win32con
import win32ui
from pywin.mfc import afxres, dialog, window
from pywin.mfc.thread import WinApp


# The main frame.
# Does almost nothing at all - doesnt even create a child window!
class HelloWindow(window.Wnd):
    def __init__(self):
        # The window.Wnd ctor creates a Window object, and places it in
        # self._obj_.  Note the window object exists, but the window itself
        # does not!
        window.Wnd.__init__(self, win32ui.CreateWnd())

        # Now we ask the window object to create the window itself.
        self._obj_.CreateWindowEx(
            win32con.WS_EX_CLIENTEDGE,
            win32ui.RegisterWndClass(0, 0, win32con.COLOR_WINDOW + 1),
            "Hello World!",
            win32con.WS_OVERLAPPEDWINDOW,
            (100, 100, 400, 300),
            None,
            0,
            None,
        )


# The application object itself.
class HelloApp(WinApp):
    def InitInstance(self):
        self.frame = HelloWindow()
        self.frame.ShowWindow(win32con.SW_SHOWNORMAL)
        # We need to tell MFC what our main frame is.
        self.SetMainFrame(self.frame)


# Now create the application object itself!
app = HelloApp()
