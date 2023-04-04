# A Python file that can be used to start Pythonwin, instead of using
# pythonwin.exe
import os
import sys

import win32ui

import pywin.framework.intpyapp  # InteractivePythonApp()

assert pywin.framework.intpyapp  # not unused
# Pretend this script doesn't exist, or pythonwin tries to edit it
sys.argv[:] = sys.argv[1:] or [""]  # like PySys_SetArgv(Ex)
if sys.path[0] not in ("", ".", os.getcwd()):
    sys.path.insert(0, os.getcwd())
# And bootstrap the app.
app = win32ui.GetApp()
if not app.InitInstance():
    # Run when not already handled by DDE
    app.Run()
