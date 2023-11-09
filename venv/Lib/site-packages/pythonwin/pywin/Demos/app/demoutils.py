# Utilities for the demos

import sys

import win32api
import win32con
import win32ui

NotScriptMsg = """\
This demo program is not designed to be run as a Script, but is
probably used by some other test program.  Please try another demo.
"""

NeedGUIMsg = """\
This demo program can only be run from inside of Pythonwin

You must start Pythonwin, and select 'Run' from the toolbar or File menu
"""


NeedAppMsg = """\
This demo program is a 'Pythonwin Application'.

It is more demo code than an example of Pythonwin's capabilities.

To run it, you must execute the command:
pythonwin.exe /app "%s"

Would you like to execute it now?
"""


def NotAScript():
    import win32ui

    win32ui.MessageBox(NotScriptMsg, "Demos")


def NeedGoodGUI():
    from pywin.framework.app import HaveGoodGUI

    rc = HaveGoodGUI()
    if not rc:
        win32ui.MessageBox(NeedGUIMsg, "Demos")
    return rc


def NeedApp():
    import win32ui

    rc = win32ui.MessageBox(NeedAppMsg % sys.argv[0], "Demos", win32con.MB_YESNO)
    if rc == win32con.IDYES:
        try:
            parent = win32ui.GetMainFrame().GetSafeHwnd()
            win32api.ShellExecute(
                parent, None, "pythonwin.exe", '/app "%s"' % sys.argv[0], None, 1
            )
        except win32api.error as details:
            win32ui.MessageBox("Error executing command - %s" % (details), "Demos")


if __name__ == "__main__":
    import demoutils

    demoutils.NotAScript()
