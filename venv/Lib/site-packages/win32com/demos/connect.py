# Implements _both_ a connectable client, and a connectable server.
#
# Note that we cheat just a little - the Server in this demo is not created
# via Normal COM - this means we can avoid registering the server.
# However, the server _is_ accessed as a COM object - just the creation
# is cheated on - so this is still working as a fully-fledged server.

import pythoncom
import win32com.server.connect
import win32com.server.util
from pywin32_testutil import str2bytes
from win32com.server.exception import Exception

# This is the IID of the Events interface both Client and Server support.
IID_IConnectDemoEvents = pythoncom.MakeIID("{A4988850-49C3-11d0-AE5D-52342E000000}")

# The server which implements
# Create a connectable class, that has a single public method
# 'DoIt', which echos to a single sink 'DoneIt'


class ConnectableServer(win32com.server.connect.ConnectableServer):
    _public_methods_ = [
        "DoIt"
    ] + win32com.server.connect.ConnectableServer._public_methods_
    _connect_interfaces_ = [IID_IConnectDemoEvents]

    # The single public method that the client can call on us
    # (ie, as a normal COM server, this exposes just this single method.
    def DoIt(self, arg):
        # Simply broadcast a notification.
        self._BroadcastNotify(self.NotifyDoneIt, (arg,))

    def NotifyDoneIt(self, interface, arg):
        interface.Invoke(1000, 0, pythoncom.DISPATCH_METHOD, 1, arg)


# Here is the client side of the connection world.
# Define a COM object which implements the methods defined by the
# IConnectDemoEvents interface.
class ConnectableClient:
    # This is another cheat - I _know_ the server defines the "DoneIt" event
    # as DISPID==1000 - I also know from the implementation details of COM
    # that the first method in _public_methods_ gets 1000.
    # Normally some explicit DISPID->Method mapping is required.
    _public_methods_ = ["OnDoneIt"]

    def __init__(self):
        self.last_event_arg = None

    # A client must implement QI, and respond to a query for the Event interface.
    # In addition, it must provide a COM object (which server.util.wrap) does.
    def _query_interface_(self, iid):
        import win32com.server.util

        # Note that this seems like a necessary hack.  I am responding to IID_IConnectDemoEvents
        # but only creating an IDispatch gateway object.
        if iid == IID_IConnectDemoEvents:
            return win32com.server.util.wrap(self)

    # And here is our event method which gets called.
    def OnDoneIt(self, arg):
        self.last_event_arg = arg


def CheckEvent(server, client, val, verbose):
    client.last_event_arg = None
    server.DoIt(val)
    if client.last_event_arg != val:
        raise RuntimeError("Sent %r, but got back %r" % (val, client.last_event_arg))
    if verbose:
        print("Sent and received %r" % val)


# A simple test script for all this.
# In the real world, it is likely that the code controlling the server
# will be in the same class as that getting the notifications.
def test(verbose=0):
    import win32com.client.connect
    import win32com.client.dynamic
    import win32com.server.policy

    server = win32com.client.dynamic.Dispatch(
        win32com.server.util.wrap(ConnectableServer())
    )
    connection = win32com.client.connect.SimpleConnection()
    client = ConnectableClient()
    connection.Connect(server, client, IID_IConnectDemoEvents)
    CheckEvent(server, client, "Hello", verbose)
    CheckEvent(server, client, str2bytes("Here is a null>\x00<"), verbose)
    CheckEvent(server, client, "Here is a null>\x00<", verbose)
    val = "test-\xe0\xf2"  # 2 extended characters.
    CheckEvent(server, client, val, verbose)
    if verbose:
        print("Everything seemed to work!")
    # Aggressive memory leak checking (ie, do nothing!) :-)  All should cleanup OK???


if __name__ == "__main__":
    test(1)
