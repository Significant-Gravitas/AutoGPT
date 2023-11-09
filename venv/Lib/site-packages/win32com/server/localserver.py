# LocalServer .EXE support for Python.
#
# This is designed to be used as a _script_ file by pythonw.exe
#
# In some cases, you could also use Python.exe, which will create
# a console window useful for debugging.
#
# NOTE: When NOT running in any sort of debugging mode,
# 'print' statements may fail, as sys.stdout is not valid!!!

#
# Usage:
#  wpython.exe LocalServer.py clsid [, clsid]
import sys

sys.coinit_flags = 2
import pythoncom
import win32api
from win32com.server import factory

usage = """\
Invalid command line arguments

This program provides LocalServer COM support
for Python COM objects.

It is typically run automatically by COM, passing as arguments
The ProgID or CLSID of the Python Server(s) to be hosted
"""


def serve(clsids):
    infos = factory.RegisterClassFactories(clsids)

    pythoncom.EnableQuitMessage(win32api.GetCurrentThreadId())
    pythoncom.CoResumeClassObjects()

    pythoncom.PumpMessages()

    factory.RevokeClassFactories(infos)

    pythoncom.CoUninitialize()


def main():
    if len(sys.argv) == 1:
        win32api.MessageBox(0, usage, "Python COM Server")
        sys.exit(1)
    serve(sys.argv[1:])


if __name__ == "__main__":
    main()
