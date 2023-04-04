"""Support for stack-frames.

Provides Implements a nearly complete wrapper for a stack frame.
"""

import pythoncom
from win32com.server.exception import COMException

from . import axdebug, expressions, gateways
from .util import RaiseNotImpl, _wrap, trace

# def trace(*args):
#       pass


class EnumDebugStackFrames(gateways.EnumDebugStackFrames):
    """A class that given a debugger object, can return an enumerator
    of DebugStackFrame objects.
    """

    def __init__(self, debugger):
        infos = []
        frame = debugger.currentframe
        #               print "Stack check"
        while frame:
            #                       print " Checking frame", frame.f_code.co_filename, frame.f_lineno-1, frame.f_trace,
            # Get a DebugCodeContext for the stack frame.  If we fail, then it
            # is not debuggable, and therefore not worth displaying.
            cc = debugger.codeContainerProvider.FromFileName(frame.f_code.co_filename)
            if cc is not None:
                try:
                    address = frame.f_locals["__axstack_address__"]
                except KeyError:
                    #                                       print "Couldnt find stack address for",frame.f_code.co_filename, frame.f_lineno-1
                    # Use this one, even tho it is wrong :-(
                    address = axdebug.GetStackAddress()
                frameInfo = (
                    DebugStackFrame(frame, frame.f_lineno - 1, cc),
                    address,
                    address + 1,
                    0,
                    None,
                )
                infos.append(frameInfo)
            #                               print "- Kept!"
            #                       else:
            #                               print "- rejected"
            frame = frame.f_back

        gateways.EnumDebugStackFrames.__init__(self, infos, 0)

    #       def __del__(self):
    #               print "EnumDebugStackFrames dieing"

    def Next(self, count):
        return gateways.EnumDebugStackFrames.Next(self, count)

    #       def _query_interface_(self, iid):
    #               from win32com.util import IIDToInterfaceName
    #               print "EnumDebugStackFrames QI with %s (%s)" % (IIDToInterfaceName(iid), str(iid))
    #               return 0
    def _wrap(self, obj):
        # This enum returns a tuple, with 2 com objects in it.
        obFrame, min, lim, fFinal, obFinal = obj
        obFrame = _wrap(obFrame, axdebug.IID_IDebugStackFrame)
        if obFinal:
            obFinal = _wrap(obFinal, pythoncom.IID_IUnknown)
        return obFrame, min, lim, fFinal, obFinal


class DebugStackFrame(gateways.DebugStackFrame):
    def __init__(self, frame, lineno, codeContainer):
        self.frame = frame
        self.lineno = lineno
        self.codeContainer = codeContainer
        self.expressionContext = None

    #       def __del__(self):
    #               print "DSF dieing"
    def _query_interface_(self, iid):
        if iid == axdebug.IID_IDebugExpressionContext:
            if self.expressionContext is None:
                self.expressionContext = _wrap(
                    expressions.ExpressionContext(self.frame),
                    axdebug.IID_IDebugExpressionContext,
                )
            return self.expressionContext
        #               from win32com.util import IIDToInterfaceName
        #               print "DebugStackFrame QI with %s (%s)" % (IIDToInterfaceName(iid), str(iid))
        return 0

    #
    # The following need implementation
    def GetThread(self):
        """Returns the thread associated with this stack frame.

        Result must be a IDebugApplicationThread
        """
        RaiseNotImpl("GetThread")

    def GetCodeContext(self):
        offset = self.codeContainer.GetPositionOfLine(self.lineno)
        return self.codeContainer.GetCodeContextAtPosition(offset)

    #
    # The following are usefully implemented
    def GetDescriptionString(self, fLong):
        filename = self.frame.f_code.co_filename
        s = ""
        if 0:  # fLong:
            s = s + filename
        if self.frame.f_code.co_name:
            s = s + self.frame.f_code.co_name
        else:
            s = s + "<lambda>"
        return s

    def GetLanguageString(self, fLong):
        if fLong:
            return "Python ActiveX Scripting Engine"
        else:
            return "Python"

    def GetDebugProperty(self):
        return _wrap(StackFrameDebugProperty(self.frame), axdebug.IID_IDebugProperty)


class DebugStackFrameSniffer:
    _public_methods_ = ["EnumStackFrames"]
    _com_interfaces_ = [axdebug.IID_IDebugStackFrameSniffer]

    def __init__(self, debugger):
        self.debugger = debugger
        trace("DebugStackFrameSniffer instantiated")

    #       def __del__(self):
    #               print "DSFS dieing"
    def EnumStackFrames(self):
        trace("DebugStackFrameSniffer.EnumStackFrames called")
        return _wrap(
            EnumDebugStackFrames(self.debugger), axdebug.IID_IEnumDebugStackFrames
        )


# A DebugProperty for a stack frame.
class StackFrameDebugProperty:
    _com_interfaces_ = [axdebug.IID_IDebugProperty]
    _public_methods_ = [
        "GetPropertyInfo",
        "GetExtendedInfo",
        "SetValueAsString",
        "EnumMembers",
        "GetParent",
    ]

    def __init__(self, frame):
        self.frame = frame

    def GetPropertyInfo(self, dwFieldSpec, nRadix):
        RaiseNotImpl("StackFrameDebugProperty::GetPropertyInfo")

    def GetExtendedInfo(self):  ### Note - not in the framework.
        RaiseNotImpl("StackFrameDebugProperty::GetExtendedInfo")

    def SetValueAsString(self, value, radix):
        #
        RaiseNotImpl("DebugProperty::SetValueAsString")

    def EnumMembers(self, dwFieldSpec, nRadix, iid):
        print("EnumMembers", dwFieldSpec, nRadix, iid)
        from . import expressions

        return expressions.MakeEnumDebugProperty(
            self.frame.f_locals, dwFieldSpec, nRadix, iid, self.frame
        )

    def GetParent(self):
        # return IDebugProperty
        RaiseNotImpl("DebugProperty::GetParent")
