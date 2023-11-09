import sys

import pythoncom
import win32com.server.policy
from win32com.axscript import axscript
from win32com.axscript.server import axsite
from win32com.axscript.server.error import Exception
from win32com.server import connect, util


class MySite(axsite.AXSite):
    def OnScriptError(self, error):
        exc = error.GetExceptionInfo()
        context, line, char = error.GetSourcePosition()
        print(" >Exception:", exc[1])
        try:
            st = error.GetSourceLineText()
        except pythoncom.com_error:
            st = None
        if st is None:
            st = ""
        text = st + "\n" + (" " * (char - 1)) + "^" + "\n" + exc[2]
        for line in text.splitlines():
            print("  >" + line)


class MyCollection(util.Collection):
    def _NewEnum(self):
        print("Making new Enumerator")
        return util.Collection._NewEnum(self)


class Test:
    _public_methods_ = ["echo"]
    _public_attrs_ = ["collection", "verbose"]

    def __init__(self):
        self.verbose = 0
        self.collection = util.wrap(MyCollection([1, "Two", 3]))
        self.last = ""

    #    self._connect_server_ = TestConnectServer(self)

    def echo(self, *args):
        self.last = "".join(map(str, args))
        if self.verbose:
            for arg in args:
                print(arg, end=" ")
            print()


#    self._connect_server_.Broadcast(last)


#### Connections currently wont work, as there is no way for the engine to
#### know what events we support.  We need typeinfo support.

IID_ITestEvents = pythoncom.MakeIID("{8EB72F90-0D44-11d1-9C4B-00AA00125A98}")


class TestConnectServer(connect.ConnectableServer):
    _connect_interfaces_ = [IID_ITestEvents]

    # The single public method that the client can call on us
    # (ie, as a normal COM server, this exposes just this single method.
    def __init__(self, object):
        self.object = object

    def Broadcast(self, arg):
        # Simply broadcast a notification.
        self._BroadcastNotify(self.NotifyDoneIt, (arg,))

    def NotifyDoneIt(self, interface, arg):
        interface.Invoke(1000, 0, pythoncom.DISPATCH_METHOD, 1, arg)


VBScript = """\
prop = "Property Value"

sub hello(arg1)
   test.echo arg1
end sub
  
sub testcollection
   test.verbose = 1
   for each item in test.collection
     test.echo "Collection item is", item
   next
end sub
"""
if sys.version_info < (3,):
    PyScript = """print "PyScript is being parsed..."\n"""
else:
    PyScript = """print("PyScript is being parsed...")\n"""
PyScript += """\
prop = "Property Value"
def hello(arg1):
   test.echo(arg1)
   pass
   
def testcollection():
   test.verbose = 1
#   test.collection[1] = "New one"
   for item in test.collection:
     test.echo("Collection item is", item)
   pass
"""

ErrScript = """\
bad code for everyone!
"""


def TestEngine(engineName, code, bShouldWork=1):
    echoer = Test()
    model = {
        "test": util.wrap(echoer),
    }

    site = MySite(model)
    engine = site._AddEngine(engineName)
    engine.AddCode(code, axscript.SCRIPTTEXT_ISPERSISTENT)
    try:
        engine.Start()
    finally:
        if not bShouldWork:
            engine.Close()
            return
    doTestEngine(engine, echoer)
    # re-transition the engine back to the UNINITIALIZED state, a-la ASP.
    engine.eScript.SetScriptState(axscript.SCRIPTSTATE_UNINITIALIZED)
    engine.eScript.SetScriptSite(util.wrap(site))
    print("restarting")
    engine.Start()
    # all done!
    engine.Close()


def doTestEngine(engine, echoer):
    # Now call into the scripts IDispatch
    from win32com.client.dynamic import Dispatch

    ob = Dispatch(engine.GetScriptDispatch())
    try:
        ob.hello("Goober")
    except pythoncom.com_error as exc:
        print("***** Calling 'hello' failed", exc)
        return
    if echoer.last != "Goober":
        print("***** Function call didnt set value correctly", repr(echoer.last))

    if str(ob.prop) != "Property Value":
        print("***** Property Value not correct - ", repr(ob.prop))

    ob.testcollection()

    # Now make sure my engines can evaluate stuff.
    result = engine.eParse.ParseScriptText(
        "1+1", None, None, None, 0, 0, axscript.SCRIPTTEXT_ISEXPRESSION
    )
    if result != 2:
        print("Engine could not evaluate '1+1' - said the result was", result)


def dotestall():
    for i in range(10):
        TestEngine("Python", PyScript)
        print(sys.gettotalrefcount())


##  print "Testing Exceptions"
##  try:
##    TestEngine("Python", ErrScript, 0)
##  except pythoncom.com_error:
##    pass


def testall():
    dotestall()
    pythoncom.CoUninitialize()
    print(
        "AXScript Host worked correctly - %d/%d COM objects left alive."
        % (pythoncom._GetInterfaceCount(), pythoncom._GetGatewayCount())
    )


if __name__ == "__main__":
    testall()
