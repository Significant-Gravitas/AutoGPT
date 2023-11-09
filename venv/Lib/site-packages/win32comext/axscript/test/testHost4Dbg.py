import os
import sys
import traceback

import pythoncom
import win32ui
from win32com.axscript import axscript
from win32com.axscript.server import axsite
from win32com.axscript.server.error import Exception
from win32com.server import util

version = "0.0.1"


class MySite(axsite.AXSite):
    def OnScriptError(self, error):
        print("An error occurred in the Script Code")
        exc = error.GetExceptionInfo()
        try:
            text = error.GetSourceLineText()
        except:
            text = "<unknown>"
        context, line, char = error.GetSourcePosition()
        print(
            "Exception: %s (line %d)\n%s\n%s^\n%s"
            % (exc[1], line, text, " " * (char - 1), exc[2])
        )


class ObjectModel:
    _public_methods_ = ["echo", "msgbox"]

    def echo(self, *args):
        print("".join(map(str, args)))

    def msgbox(self, *args):
        msg = "".join(map(str, args))
        win32ui.MessageBox(msg)


def TestEngine():
    model = {"Test": util.wrap(ObjectModel())}
    scriptDir = "."
    site = MySite(model)
    pyEngine = site._AddEngine("Python")
    #  pyEngine2 = site._AddEngine("Python")
    vbEngine = site._AddEngine("VBScript")
    #  forthEngine = site._AddEngine("ForthScript")
    try:
        #    code = open(os.path.join(scriptDir, "debugTest.4ths"),"rb").read()
        #    forthEngine.AddCode(code)
        code = open(os.path.join(scriptDir, "debugTest.pys"), "rb").read()
        pyEngine.AddCode(code)
        code = open(os.path.join(scriptDir, "debugTest.vbs"), "rb").read()
        vbEngine.AddCode(code)
        #    code = open(os.path.join(scriptDir, "debugTestFail.pys"),"rb").read()
        #    pyEngine2.AddCode(code)

        #    from win32com.axdebug import axdebug
        #    sessionProvider=pythoncom.CoCreateInstance(axdebug.CLSID_DefaultDebugSessionProvider,None,pythoncom.CLSCTX_ALL, axdebug.IID_IDebugSessionProvider)
        #    sessionProvider.StartDebugSession(None)

        input("Press enter to continue")
        #   forthEngine.Start()
        pyEngine.Start()  # Actually run the Python code
        vbEngine.Start()  # Actually run the VB code
    except pythoncom.com_error as details:
        print("Script failed: %s (0x%x)" % (details[1], details[0]))
    # Now run the code expected to fail!
    #  try:
    #    pyEngine2.Start() # Actually run the Python code that fails!
    #    print "Script code worked when it should have failed."
    #  except pythoncom.com_error:
    #    pass

    site._Close()


if __name__ == "__main__":
    import win32com.axdebug.util

    try:
        TestEngine()
    except:
        traceback.print_exc()
    win32com.axdebug.util._dump_wrapped()
    sys.exc_type = sys.exc_value = sys.exc_traceback = None
    print(pythoncom._GetInterfaceCount(), "com objects still alive")
