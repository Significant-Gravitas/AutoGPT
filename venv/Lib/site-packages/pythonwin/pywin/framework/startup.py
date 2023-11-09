# startup.py
#
"The main application startup code for PythonWin."

#
# This does the basic command line handling.

# Keep this as short as possible, cos error output is only redirected if
# this runs OK.  Errors in imported modules are much better - the messages go somewhere (not any more :-)

import os
import sys

import win32api
import win32ui

if not sys.argv:
    # Initialize sys.argv from commandline. When sys.argv is empty list (
    # different from [''] meaning "no cmd line arguments" ), then C
    # bootstrapping or another method of invocation failed to initialize
    # sys.argv and it will be done here. ( This was a workaround for a bug in
    # win32ui but is retained for other situations. )
    argv = win32api.CommandLineToArgv(win32api.GetCommandLine())
    sys.argv = argv[1:]
    if os.getcwd() not in sys.path and "." not in sys.path:
        sys.path.insert(0, os.getcwd())

# You may wish to redirect error output somewhere useful if you have startup errors.
# eg, 'import win32traceutil' will do this for you.
# import win32traceutil # Just uncomment this line to see error output!

# An old class I used to use - generally only useful if Pythonwin is running under MSVC
# class DebugOutput:
# 	softspace=1
# 	def write(self,message):
# 		win32ui.OutputDebug(message)
# sys.stderr=sys.stdout=DebugOutput()

# To fix a problem with Pythonwin when started from the Pythonwin directory,
# we update the pywin path to ensure it is absolute.
# If it is indeed relative, it will be relative to our current directory.
# If its already absolute, then this will have no affect.
import pywin
import pywin.framework

pywin.__path__[0] = win32ui.FullPath(pywin.__path__[0])
pywin.framework.__path__[0] = win32ui.FullPath(pywin.framework.__path__[0])

# make a few wierd sys values.  This is so later we can clobber sys.argv to trick
# scripts when running under a GUI environment.

moduleName = "pywin.framework.intpyapp"
sys.appargvoffset = 0
sys.appargv = sys.argv[:]
# Must check for /app param here.
if len(sys.argv) >= 2 and sys.argv[0].lower() in ("/app", "-app"):
    from . import cmdline

    moduleName = cmdline.FixArgFileName(sys.argv[1])
    sys.appargvoffset = 2
    newargv = sys.argv[sys.appargvoffset :]
    # 	newargv.insert(0, sys.argv[0])
    sys.argv = newargv

# Import the application module.
__import__(moduleName)

try:
    win32ui.GetApp()._obj_
    # This worked - an app already exists - do nothing more
except (AttributeError, win32ui.error):
    # This means either no app object exists at all, or the one
    # that does exist does not have a Python class (ie, was created
    # by the host .EXE).  In this case, we do the "old style" init...
    from . import app

    if app.AppBuilder is None:
        raise TypeError("No application object has been registered")

    app.App = app.AppBuilder()
