""" A module for managing the AXDebug I*Contexts

"""
import pythoncom
import win32com.server.util

from . import adb, axdebug, gateways

# Utility function for wrapping object created by this module.
from .util import _wrap, _wrap_remove, trace


class DebugCodeContext(gateways.DebugCodeContext, gateways.DebugDocumentContext):
    # NOTE: We also implement the IDebugDocumentContext interface for Simple Hosts.
    # Thus, debugDocument may be NULL when we have smart hosts - but in that case, we
    # wont be called upon to provide it.
    _public_methods_ = (
        gateways.DebugCodeContext._public_methods_
        + gateways.DebugDocumentContext._public_methods_
    )
    _com_interfaces_ = (
        gateways.DebugCodeContext._com_interfaces_
        + gateways.DebugDocumentContext._com_interfaces_
    )

    def __init__(self, lineNo, charPos, len, codeContainer, debugSite):
        self.debugSite = debugSite
        self.offset = charPos
        self.length = len
        self.breakPointState = 0
        self.lineno = lineNo
        gateways.DebugCodeContext.__init__(self)
        self.codeContainer = codeContainer

    def _Close(self):
        self.debugSite = None

    def GetDocumentContext(self):
        if self.debugSite is not None:
            # We have a smart host - let him give it to us.
            return self.debugSite.GetDocumentContextFromPosition(
                self.codeContainer.sourceContext, self.offset, self.length
            )
        else:
            # Simple host - Fine - Ill do it myself!
            return _wrap(self, axdebug.IID_IDebugDocumentContext)

    def SetBreakPoint(self, bps):
        self.breakPointState = bps
        adb.OnSetBreakPoint(self, bps, self.lineno)

    # The DebugDocumentContext methods for simple hosts.
    def GetDocument(self):
        return self.codeContainer.debugDocument

    def EnumCodeContexts(self):
        return _wrap(EnumDebugCodeContexts([self]), axdebug.IID_IEnumDebugCodeContexts)


class EnumDebugCodeContexts(gateways.EnumDebugCodeContexts):
    def _wrap(self, obj):
        return _wrap(obj, axdebug.IID_IDebugCodeContext)
