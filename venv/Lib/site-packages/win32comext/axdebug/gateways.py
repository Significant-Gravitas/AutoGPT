# Classes which describe interfaces.

import pythoncom
import win32com.server.connect
import winerror
from win32com.axdebug import axdebug
from win32com.axdebug.util import RaiseNotImpl, _wrap
from win32com.server.exception import Exception
from win32com.server.util import ListEnumeratorGateway


class EnumDebugCodeContexts(ListEnumeratorGateway):
    """A class to expose a Python sequence as an EnumDebugCodeContexts

    Create an instance of this class passing a sequence (list, tuple, or
    any sequence protocol supporting object) and it will automatically
    support the EnumDebugCodeContexts interface for the object.

    """

    _com_interfaces_ = [axdebug.IID_IEnumDebugCodeContexts]


class EnumDebugStackFrames(ListEnumeratorGateway):
    """A class to expose a Python sequence as an EnumDebugStackFrames

    Create an instance of this class passing a sequence (list, tuple, or
    any sequence protocol supporting object) and it will automatically
    support the EnumDebugStackFrames interface for the object.

    """

    _com_interfaces_ = [axdebug.IID_IEnumDebugStackFrames]


class EnumDebugApplicationNodes(ListEnumeratorGateway):
    """A class to expose a Python sequence as an EnumDebugStackFrames

    Create an instance of this class passing a sequence (list, tuple, or
    any sequence protocol supporting object) and it will automatically
    support the EnumDebugApplicationNodes interface for the object.

    """

    _com_interfaces_ = [axdebug.IID_IEnumDebugApplicationNodes]


class EnumRemoteDebugApplications(ListEnumeratorGateway):
    _com_interfaces_ = [axdebug.IID_IEnumRemoteDebugApplications]


class EnumRemoteDebugApplicationThreads(ListEnumeratorGateway):
    _com_interfaces_ = [axdebug.IID_IEnumRemoteDebugApplicationThreads]


class DebugDocumentInfo:
    _public_methods_ = ["GetName", "GetDocumentClassId"]
    _com_interfaces_ = [axdebug.IID_IDebugDocumentInfo]

    def __init__(self):
        pass

    def GetName(self, dnt):
        """Get the one of the name of the document
        dnt -- int DOCUMENTNAMETYPE
        """
        RaiseNotImpl("GetName")

    def GetDocumentClassId(self):
        """
        Result must be an IID object (or string representing one).
        """
        RaiseNotImpl("GetDocumentClassId")


class DebugDocumentProvider(DebugDocumentInfo):
    _public_methods_ = DebugDocumentInfo._public_methods_ + ["GetDocument"]
    _com_interfaces_ = DebugDocumentInfo._com_interfaces_ + [
        axdebug.IID_IDebugDocumentProvider
    ]

    def GetDocument(self):
        RaiseNotImpl("GetDocument")


class DebugApplicationNode(DebugDocumentProvider):
    """Provides the functionality of IDebugDocumentProvider, plus a context within a project tree."""

    _public_methods_ = (
        """EnumChildren GetParent SetDocumentProvider
                    Close Attach Detach""".split()
        + DebugDocumentProvider._public_methods_
    )
    _com_interfaces_ = [
        axdebug.IID_IDebugDocumentProvider
    ] + DebugDocumentProvider._com_interfaces_

    def __init__(self):
        DebugDocumentProvider.__init__(self)

    def EnumChildren(self):
        # Result is type PyIEnumDebugApplicationNodes
        RaiseNotImpl("EnumChildren")

    def GetParent(self):
        # result is type PyIDebugApplicationNode
        RaiseNotImpl("GetParent")

    def SetDocumentProvider(self, pddp):  # PyIDebugDocumentProvider pddp
        # void result.
        RaiseNotImpl("SetDocumentProvider")

    def Close(self):
        # void result.
        RaiseNotImpl("Close")

    def Attach(self, parent):  # PyIDebugApplicationNode
        # void result.
        RaiseNotImpl("Attach")

    def Detach(self):
        # void result.
        RaiseNotImpl("Detach")


class DebugApplicationNodeEvents:
    """Event interface for DebugApplicationNode object."""

    _public_methods_ = "onAddChild onRemoveChild onDetach".split()
    _com_interfaces_ = [axdebug.IID_IDebugApplicationNodeEvents]

    def __init__(self):
        pass

    def onAddChild(self, child):  # PyIDebugApplicationNode
        # void result.
        RaiseNotImpl("onAddChild")

    def onRemoveChild(self, child):  # PyIDebugApplicationNode
        # void result.
        RaiseNotImpl("onRemoveChild")

    def onDetach(self):
        # void result.
        RaiseNotImpl("onDetach")

    def onAttach(self, parent):  # PyIDebugApplicationNode
        # void result.
        RaiseNotImpl("onAttach")


class DebugDocument(DebugDocumentInfo):
    """The base interface to all debug documents."""

    _public_methods_ = DebugDocumentInfo._public_methods_
    _com_interfaces_ = [axdebug.IID_IDebugDocument] + DebugDocumentInfo._com_interfaces_


class DebugDocumentText(DebugDocument):
    """The interface to a text only debug document."""

    _com_interfaces_ = [axdebug.IID_IDebugDocumentText] + DebugDocument._com_interfaces_
    _public_methods_ = [
        "GetDocumentAttributes",
        "GetSize",
        "GetPositionOfLine",
        "GetLineOfPosition",
        "GetText",
        "GetPositionOfContext",
        "GetContextOfPosition",
    ] + DebugDocument._public_methods_

    def __init__(self):
        pass

    # IDebugDocumentText
    def GetDocumentAttributes(self):
        # Result is int (TEXT_DOC_ATTR)
        RaiseNotImpl("GetDocumentAttributes")

    def GetSize(self):
        # Result is (numLines, numChars)
        RaiseNotImpl("GetSize")

    def GetPositionOfLine(self, cLineNumber):
        # Result is int char position
        RaiseNotImpl("GetPositionOfLine")

    def GetLineOfPosition(self, charPos):
        # Result is int, int (lineNo, offset)
        RaiseNotImpl("GetLineOfPosition")

    def GetText(self, charPos, maxChars, wantAttr):
        """Params
        charPos -- integer
        maxChars -- integer
        wantAttr -- Should the function compute attributes.

        Return value must be (string, attribtues).  attributes may be
        None if(not wantAttr)
        """
        RaiseNotImpl("GetText")

    def GetPositionOfContext(self, debugDocumentContext):
        """Params
        debugDocumentContext -- a PyIDebugDocumentContext object.

        Return value must be (charPos, numChars)
        """
        RaiseNotImpl("GetPositionOfContext")

    def GetContextOfPosition(self, charPos, maxChars):
        """Params are integers.
        Return value must be PyIDebugDocumentContext object
        """
        print(self)
        RaiseNotImpl("GetContextOfPosition")


class DebugDocumentTextExternalAuthor:
    """Allow external editors to edit file-based debugger documents, and to notify the document when the source file has been changed."""

    _public_methods_ = ["GetPathName", "GetFileName", "NotifyChanged"]
    _com_interfaces_ = [axdebug.IID_IDebugDocumentTextExternalAuthor]

    def __init__(self):
        pass

    def GetPathName(self):
        """Return the full path (including file name) to the document's source file.

        Result must be (filename, fIsOriginal), where
        - if fIsOriginalPath is TRUE if the path refers to the original file for the document.
        - if fIsOriginalPath is FALSE if the path refers to a newly created temporary file.

        raise Exception(winerror.E_FAIL) if no source file can be created/determined.
        """
        RaiseNotImpl("GetPathName")

    def GetFileName(self):
        """Return just the name of the document, with no path information.  (Used for "Save As...")

        Result is a string
        """
        RaiseNotImpl("GetFileName")

    def NotifyChanged(self):
        """Notify the host that the document's source file has been saved and
        that its contents should be refreshed.
        """
        RaiseNotImpl("NotifyChanged")


class DebugDocumentTextEvents:
    _public_methods_ = """onDestroy onInsertText onRemoveText
              onReplaceText onUpdateTextAttributes
              onUpdateDocumentAttributes""".split()
    _com_interfaces_ = [axdebug.IID_IDebugDocumentTextEvents]

    def __init__(self):
        pass

    def onDestroy(self):
        # Result is void.
        RaiseNotImpl("onDestroy")

    def onInsertText(self, cCharacterPosition, cNumToInsert):
        # Result is void.
        RaiseNotImpl("onInsertText")

    def onRemoveText(self, cCharacterPosition, cNumToRemove):
        # Result is void.
        RaiseNotImpl("onRemoveText")

    def onReplaceText(self, cCharacterPosition, cNumToReplace):
        # Result is void.
        RaiseNotImpl("onReplaceText")

    def onUpdateTextAttributes(self, cCharacterPosition, cNumToUpdate):
        # Result is void.
        RaiseNotImpl("onUpdateTextAttributes")

    def onUpdateDocumentAttributes(self, textdocattr):  # TEXT_DOC_ATTR
        # Result is void.
        RaiseNotImpl("onUpdateDocumentAttributes")


class DebugDocumentContext:
    _public_methods_ = ["GetDocument", "EnumCodeContexts"]
    _com_interfaces_ = [axdebug.IID_IDebugDocumentContext]

    def __init__(self):
        pass

    def GetDocument(self):
        """Return value must be a PyIDebugDocument object"""
        RaiseNotImpl("GetDocument")

    def EnumCodeContexts(self):
        """Return value must be a PyIEnumDebugCodeContexts object"""
        RaiseNotImpl("EnumCodeContexts")


class DebugCodeContext:
    _public_methods_ = ["GetDocumentContext", "SetBreakPoint"]
    _com_interfaces_ = [axdebug.IID_IDebugCodeContext]

    def __init__(self):
        pass

    def GetDocumentContext(self):
        """Return value must be a PyIDebugDocumentContext object"""
        RaiseNotImpl("GetDocumentContext")

    def SetBreakPoint(self, bps):
        """bps -- an integer with flags."""
        RaiseNotImpl("SetBreakPoint")


class DebugStackFrame:
    """Abstraction representing a logical stack frame on the stack of a thread."""

    _public_methods_ = [
        "GetCodeContext",
        "GetDescriptionString",
        "GetLanguageString",
        "GetThread",
        "GetDebugProperty",
    ]
    _com_interfaces_ = [axdebug.IID_IDebugStackFrame]

    def __init__(self):
        pass

    def GetCodeContext(self):
        """Returns the current code context associated with the stack frame.

        Return value must be a IDebugCodeContext object
        """
        RaiseNotImpl("GetCodeContext")

    def GetDescriptionString(self, fLong):
        """Returns a textual description of the stack frame.

        fLong -- A flag indicating if the long name is requested.
        """
        RaiseNotImpl("GetDescriptionString")

    def GetLanguageString(self):
        """Returns a short or long textual description of the language.

        fLong -- A flag indicating if the long name is requested.
        """
        RaiseNotImpl("GetLanguageString")

    def GetThread(self):
        """Returns the thread associated with this stack frame.

        Result must be a IDebugApplicationThread
        """
        RaiseNotImpl("GetThread")

    def GetDebugProperty(self):
        RaiseNotImpl("GetDebugProperty")


class DebugDocumentHost:
    """The interface from the IDebugDocumentHelper back to
    the smart host or language engine.  This interface
    exposes host specific functionality such as syntax coloring.
    """

    _public_methods_ = [
        "GetDeferredText",
        "GetScriptTextAttributes",
        "OnCreateDocumentContext",
        "GetPathName",
        "GetFileName",
        "NotifyChanged",
    ]
    _com_interfaces_ = [axdebug.IID_IDebugDocumentHost]

    def __init__(self):
        pass

    def GetDeferredText(self, dwTextStartCookie, maxChars, bWantAttr):
        RaiseNotImpl("GetDeferredText")

    def GetScriptTextAttributes(self, codeText, delimterText, flags):
        # Result must be an attribute sequence of same "length" as the code.
        RaiseNotImpl("GetScriptTextAttributes")

    def OnCreateDocumentContext(self):
        # Result must be a PyIUnknown
        RaiseNotImpl("OnCreateDocumentContext")

    def GetPathName(self):
        # Result must be (string, int) where the int is a BOOL
        # - TRUE if the path refers to the original file for the document.
        # - FALSE if the path refers to a newly created temporary file.
        # - raise Exception(scode=E_FAIL) if no source file can be created/determined.
        RaiseNotImpl("GetPathName")

    def GetFileName(self):
        # Result is a string with just the name of the document, no path information.
        RaiseNotImpl("GetFileName")

    def NotifyChanged(self):
        RaiseNotImpl("NotifyChanged")


# Additional gateway related functions.


class DebugDocumentTextConnectServer:
    _public_methods_ = (
        win32com.server.connect.IConnectionPointContainer_methods
        + win32com.server.connect.IConnectionPoint_methods
    )
    _com_interfaces_ = [
        pythoncom.IID_IConnectionPoint,
        pythoncom.IID_IConnectionPointContainer,
    ]

    # IConnectionPoint interfaces
    def __init__(self):
        self.cookieNo = -1
        self.connections = {}

    def EnumConnections(self):
        RaiseNotImpl("EnumConnections")

    def GetConnectionInterface(self):
        RaiseNotImpl("GetConnectionInterface")

    def GetConnectionPointContainer(self):
        return _wrap(self)

    def Advise(self, pUnk):
        # Creates a connection to the client.  Simply allocate a new cookie,
        # find the clients interface, and store it in a dictionary.
        interface = pUnk.QueryInterface(axdebug.IID_IDebugDocumentTextEvents, 1)
        self.cookieNo = self.cookieNo + 1
        self.connections[self.cookieNo] = interface
        return self.cookieNo

    def Unadvise(self, cookie):
        # Destroy a connection - simply delete interface from the map.
        try:
            del self.connections[cookie]
        except KeyError:
            return Exception(scode=winerror.E_UNEXPECTED)

    # IConnectionPointContainer interfaces
    def EnumConnectionPoints(self):
        RaiseNotImpl("EnumConnectionPoints")

    def FindConnectionPoint(self, iid):
        # Find a connection we support.  Only support the single event interface.
        if iid == axdebug.IID_IDebugDocumentTextEvents:
            return _wrap(self)
        raise Exception(scode=winerror.E_NOINTERFACE)  # ??


class RemoteDebugApplicationEvents:
    _public_methods_ = [
        "OnConnectDebugger",
        "OnDisconnectDebugger",
        "OnSetName",
        "OnDebugOutput",
        "OnClose",
        "OnEnterBreakPoint",
        "OnLeaveBreakPoint",
        "OnCreateThread",
        "OnDestroyThread",
        "OnBreakFlagChange",
    ]
    _com_interfaces_ = [axdebug.IID_IRemoteDebugApplicationEvents]

    def OnConnectDebugger(self, appDebugger):
        """appDebugger -- a PyIApplicationDebugger"""
        RaiseNotImpl("OnConnectDebugger")

    def OnDisconnectDebugger(self):
        RaiseNotImpl("OnDisconnectDebugger")

    def OnSetName(self, name):
        RaiseNotImpl("OnSetName")

    def OnDebugOutput(self, string):
        RaiseNotImpl("OnDebugOutput")

    def OnClose(self):
        RaiseNotImpl("OnClose")

    def OnEnterBreakPoint(self, rdat):
        """rdat -- PyIRemoteDebugApplicationThread"""
        RaiseNotImpl("OnEnterBreakPoint")

    def OnLeaveBreakPoint(self, rdat):
        """rdat -- PyIRemoteDebugApplicationThread"""
        RaiseNotImpl("OnLeaveBreakPoint")

    def OnCreateThread(self, rdat):
        """rdat -- PyIRemoteDebugApplicationThread"""
        RaiseNotImpl("OnCreateThread")

    def OnDestroyThread(self, rdat):
        """rdat -- PyIRemoteDebugApplicationThread"""
        RaiseNotImpl("OnDestroyThread")

    def OnBreakFlagChange(self, abf, rdat):
        """abf -- int - one of the axdebug.APPBREAKFLAGS constants
        rdat -- PyIRemoteDebugApplicationThread
        RaiseNotImpl("OnBreakFlagChange")
        """


class DebugExpressionContext:
    _public_methods_ = ["ParseLanguageText", "GetLanguageInfo"]
    _com_interfaces_ = [axdebug.IID_IDebugExpressionContext]

    def __init__(self):
        pass

    def ParseLanguageText(self, code, radix, delim, flags):
        """
        result is IDebugExpression
        """
        RaiseNotImpl("ParseLanguageText")

    def GetLanguageInfo(self):
        """
        result is (string langName, iid langId)
        """
        RaiseNotImpl("GetLanguageInfo")


class DebugExpression:
    _public_methods_ = [
        "Start",
        "Abort",
        "QueryIsComplete",
        "GetResultAsString",
        "GetResultAsDebugProperty",
    ]
    _com_interfaces_ = [axdebug.IID_IDebugExpression]

    def Start(self, callback):
        """
        callback -- an IDebugExpressionCallback

        result - void
        """
        RaiseNotImpl("Start")

    def Abort(self):
        """
        no params
        result -- void
        """
        RaiseNotImpl("Abort")

    def QueryIsComplete(self):
        """
        no params
        result -- void
        """
        RaiseNotImpl("QueryIsComplete")

    def GetResultAsString(self):
        RaiseNotImpl("GetResultAsString")

    def GetResultAsDebugProperty(self):
        RaiseNotImpl("GetResultAsDebugProperty")


class ProvideExpressionContexts:
    _public_methods_ = ["EnumExpressionContexts"]
    _com_interfaces_ = [axdebug.IID_IProvideExpressionContexts]

    def EnumExpressionContexts(self):
        RaiseNotImpl("EnumExpressionContexts")
