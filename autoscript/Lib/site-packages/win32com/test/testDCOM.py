# testDCOM
usage = """\
testDCOM.py - Simple DCOM test
Usage: testDCOM.py serverName

Attempts to start the Python.Interpreter object on the named machine,
and checks that the object is indeed running remotely.

Requires the named server be configured to run DCOM (using dcomcnfg.exe),
and the Python.Interpreter object installed and registered on that machine.

The Python.Interpreter object must be installed on the local machine,
but no special DCOM configuration should be necessary.
"""
import string
import sys

# NOTE: If you configured the object locally using dcomcnfg, you could
# simple use Dispatch rather than DispatchEx.
import pythoncom
import win32api
import win32com.client


def test(serverName):
    if string.lower(serverName) == string.lower(win32api.GetComputerName()):
        print("You must specify a remote server name, not the local machine!")
        return

    # Hack to overcome a DCOM limitation.  As the Python.Interpreter object
    # is probably installed locally as an InProc object, DCOM seems to ignore
    # all settings, and use the local object.
    clsctx = pythoncom.CLSCTX_SERVER & ~pythoncom.CLSCTX_INPROC_SERVER
    ob = win32com.client.DispatchEx("Python.Interpreter", serverName, clsctx=clsctx)
    ob.Exec("import win32api")
    actualName = ob.Eval("win32api.GetComputerName()")
    if string.lower(serverName) != string.lower(actualName):
        print(
            "Error: The object created on server '%s' reported its name as '%s'"
            % (serverName, actualName)
        )
    else:
        print("Object created and tested OK on server '%s'" % serverName)


if __name__ == "__main__":
    if len(sys.argv) == 2:
        test(sys.argv[1])
    else:
        print(usage)
