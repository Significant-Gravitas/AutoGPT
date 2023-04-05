"""Utilities for Server Side connections.

  A collection of helpers for server side connection points.
"""
import pythoncom
import win32com.server.util
import winerror
from win32com import olectl

from .exception import Exception

# Methods implemented by the interfaces.
IConnectionPointContainer_methods = ["EnumConnectionPoints", "FindConnectionPoint"]
IConnectionPoint_methods = [
    "EnumConnections",
    "Unadvise",
    "Advise",
    "GetConnectionPointContainer",
    "GetConnectionInterface",
]


class ConnectableServer:
    _public_methods_ = IConnectionPointContainer_methods + IConnectionPoint_methods
    _com_interfaces_ = [
        pythoncom.IID_IConnectionPoint,
        pythoncom.IID_IConnectionPointContainer,
    ]

    # Clients must set _connect_interfaces_ = [...]
    def __init__(self):
        self.cookieNo = 0
        self.connections = {}

    # IConnectionPoint interfaces
    def EnumConnections(self):
        raise Exception(winerror.E_NOTIMPL)

    def GetConnectionInterface(self):
        raise Exception(winerror.E_NOTIMPL)

    def GetConnectionPointContainer(self):
        return win32com.server.util.wrap(self)

    def Advise(self, pUnk):
        # Creates a connection to the client.  Simply allocate a new cookie,
        # find the clients interface, and store it in a dictionary.
        try:
            interface = pUnk.QueryInterface(
                self._connect_interfaces_[0], pythoncom.IID_IDispatch
            )
        except pythoncom.com_error:
            raise Exception(scode=olectl.CONNECT_E_NOCONNECTION)
        self.cookieNo = self.cookieNo + 1
        self.connections[self.cookieNo] = interface
        return self.cookieNo

    def Unadvise(self, cookie):
        # Destroy a connection - simply delete interface from the map.
        try:
            del self.connections[cookie]
        except KeyError:
            raise Exception(scode=winerror.E_UNEXPECTED)

    # IConnectionPointContainer interfaces
    def EnumConnectionPoints(self):
        raise Exception(winerror.E_NOTIMPL)

    def FindConnectionPoint(self, iid):
        # Find a connection we support.  Only support the single event interface.
        if iid in self._connect_interfaces_:
            return win32com.server.util.wrap(self)

    def _BroadcastNotify(self, broadcaster, extraArgs):
        # Broadcasts a notification to all connections.
        # Ignores clients that fail.
        for interface in self.connections.values():
            try:
                broadcaster(*(interface,) + extraArgs)
            except pythoncom.com_error as details:
                self._OnNotifyFail(interface, details)

    def _OnNotifyFail(self, interface, details):
        print("Ignoring COM error to connection - %s" % (repr(details)))
