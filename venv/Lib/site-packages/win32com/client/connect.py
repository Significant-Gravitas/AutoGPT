"""Utilities for working with Connections"""
import pythoncom
import win32com.server.util


class SimpleConnection:
    "A simple, single connection object"

    def __init__(self, coInstance=None, eventInstance=None, eventCLSID=None, debug=0):
        self.cp = None
        self.cookie = None
        self.debug = debug
        if not coInstance is None:
            self.Connect(coInstance, eventInstance, eventCLSID)

    def __del__(self):
        try:
            self.Disconnect()
        except pythoncom.error:
            # Ignore disconnection as we are torn down.
            pass

    def _wrap(self, obj):
        useDispatcher = None
        if self.debug:
            from win32com.server import dispatcher

            useDispatcher = dispatcher.DefaultDebugDispatcher
        return win32com.server.util.wrap(obj, useDispatcher=useDispatcher)

    def Connect(self, coInstance, eventInstance, eventCLSID=None):
        try:
            oleobj = coInstance._oleobj_
        except AttributeError:
            oleobj = coInstance
        cpc = oleobj.QueryInterface(pythoncom.IID_IConnectionPointContainer)
        if eventCLSID is None:
            eventCLSID = eventInstance.CLSID
        comEventInstance = self._wrap(eventInstance)
        self.cp = cpc.FindConnectionPoint(eventCLSID)
        self.cookie = self.cp.Advise(comEventInstance)

    def Disconnect(self):
        if not self.cp is None:
            if self.cookie:
                self.cp.Unadvise(self.cookie)
                self.cookie = None
            self.cp = None
