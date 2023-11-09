"""AXScript Client Framework

  This module provides a core framework for an ActiveX Scripting client.
  Derived classes actually implement the AX Client itself, including the
  scoping rules, etc.

  There are classes defined for the engine itself, and for ScriptItems
"""
import re
import sys

import pythoncom  # Need simple connection point support
import win32api
import win32com.client.connect
import win32com.server.util
import winerror
from win32com.axscript import axscript


def RemoveCR(text):
    # No longer just "RemoveCR" - should be renamed to
    # FixNewlines, or something.  Idea is to fix arbitary newlines into
    # something Python can compile...
    return re.sub("(\r\n)|\r|(\n\r)", "\n", text)


SCRIPTTEXT_FORCEEXECUTION = -2147483648  # 0x80000000
SCRIPTTEXT_ISEXPRESSION = 0x00000020
SCRIPTTEXT_ISPERSISTENT = 0x00000040

from win32com.server.exception import Exception, IsCOMServerException

from . import error  # ax.client.error

state_map = {
    axscript.SCRIPTSTATE_UNINITIALIZED: "SCRIPTSTATE_UNINITIALIZED",
    axscript.SCRIPTSTATE_INITIALIZED: "SCRIPTSTATE_INITIALIZED",
    axscript.SCRIPTSTATE_STARTED: "SCRIPTSTATE_STARTED",
    axscript.SCRIPTSTATE_CONNECTED: "SCRIPTSTATE_CONNECTED",
    axscript.SCRIPTSTATE_DISCONNECTED: "SCRIPTSTATE_DISCONNECTED",
    axscript.SCRIPTSTATE_CLOSED: "SCRIPTSTATE_CLOSED",
}


def profile(fn, *args):
    import profile

    prof = profile.Profile()
    try:
        # roll on 1.6 :-)
        # 		return prof.runcall(fn, *args)
        return prof.runcall(*(fn,) + args)
    finally:
        import pstats

        # Damn - really want to send this to Excel!
        #      width, list = pstats.Stats(prof).strip_dirs().get_print_list([])
        pstats.Stats(prof).strip_dirs().sort_stats("time").print_stats()


class SafeOutput:
    softspace = 1

    def __init__(self, redir=None):
        if redir is None:
            redir = sys.stdout
        self.redir = redir

    def write(self, message):
        try:
            self.redir.write(message)
        except:
            win32api.OutputDebugString(message)

    def flush(self):
        pass

    def close(self):
        pass


# Make sure we have a valid sys.stdout/stderr, otherwise out
# print and trace statements may raise an exception
def MakeValidSysOuts():
    if not isinstance(sys.stdout, SafeOutput):
        sys.stdout = sys.stderr = SafeOutput()
        # and for the sake of working around something I can't understand...
        # prevent keyboard interrupts from killing IIS
        import signal

        def noOp(a, b):
            # it would be nice to get to the bottom of this, so a warning to
            # the debug console can't hurt.
            print("WARNING: Ignoring keyboard interrupt from ActiveScripting engine")

        # If someone else has already redirected, then assume they know what they are doing!
        if signal.getsignal(signal.SIGINT) == signal.default_int_handler:
            try:
                signal.signal(signal.SIGINT, noOp)
            except ValueError:
                # Not the main thread - can't do much.
                pass


def trace(*args):
    """A function used instead of "print" for debugging output."""
    for arg in args:
        print(arg, end=" ")
    print()


def RaiseAssert(scode, desc):
    """A debugging function that raises an exception considered an "Assertion"."""
    print("**************** ASSERTION FAILED *******************")
    print(desc)
    raise Exception(desc, scode)


class AXScriptCodeBlock:
    """An object which represents a chunk of code in an AX Script"""

    def __init__(self, name, codeText, sourceContextCookie, startLineNumber, flags):
        self.name = name
        self.codeText = codeText
        self.codeObject = None
        self.sourceContextCookie = sourceContextCookie
        self.startLineNumber = startLineNumber
        self.flags = flags
        self.beenExecuted = 0

    def GetFileName(self):
        # Gets the "file name" for Python - uses <...> so Python doesnt think
        # it is a real file.
        return "<%s>" % self.name

    def GetDisplayName(self):
        return self.name

    def GetLineNo(self, no):
        pos = -1
        for i in range(no - 1):
            pos = self.codeText.find("\n", pos + 1)
            if pos == -1:
                pos = len(self.codeText)
        epos = self.codeText.find("\n", pos + 1)
        if epos == -1:
            epos = len(self.codeText)
        return self.codeText[pos + 1 : epos].strip()


class Event:
    """A single event for a ActiveX named object."""

    def __init__(self):
        self.name = "<None>"

    def __repr__(self):
        return "<%s at %d: %s>" % (self.__class__.__name__, id(self), self.name)

    def Reset(self):
        pass

    def Close(self):
        pass

    def Build(self, typeinfo, funcdesc):
        self.dispid = funcdesc[0]
        self.name = typeinfo.GetNames(self.dispid)[0]


# 		print "Event.Build() - Event Name is ", self.name


class EventSink:
    """A set of events against an item.  Note this is a COM client for connection points."""

    _public_methods_ = []

    def __init__(self, myItem, coDispatch):
        self.events = {}
        self.connection = None
        self.coDispatch = coDispatch
        self.myScriptItem = myItem
        self.myInvokeMethod = myItem.GetEngine().ProcessScriptItemEvent
        self.iid = None

    def Reset(self):
        self.Disconnect()

    def Close(self):
        self.iid = None
        self.myScriptItem = None
        self.myInvokeMethod = None
        self.coDispatch = None
        for event in self.events.values():
            event.Reset()
        self.events = {}
        self.Disconnect()

    # COM Connection point methods.
    def _query_interface_(self, iid):
        if iid == self.iid:
            return win32com.server.util.wrap(self)

    def _invoke_(self, dispid, lcid, wFlags, args):
        try:
            event = self.events[dispid]
        except:
            raise Exception(scode=winerror.DISP_E_MEMBERNOTFOUND)
        # print "Invoke for ", event, "on", self.myScriptItem, " - calling",  self.myInvokeMethod
        return self.myInvokeMethod(self.myScriptItem, event, lcid, wFlags, args)

    def GetSourceTypeInfo(self, typeinfo):
        """Gets the typeinfo for the Source Events for the passed typeinfo"""
        attr = typeinfo.GetTypeAttr()
        cFuncs = attr[6]
        typeKind = attr[5]
        if typeKind not in [pythoncom.TKIND_COCLASS, pythoncom.TKIND_INTERFACE]:
            RaiseAssert(
                winerror.E_UNEXPECTED, "The typeKind of the object is unexpected"
            )
        cImplType = attr[8]
        for i in range(cImplType):
            # Look for the [source, default] interface on the coclass
            # that isn't marked as restricted.
            flags = typeinfo.GetImplTypeFlags(i)
            flagsNeeded = (
                pythoncom.IMPLTYPEFLAG_FDEFAULT | pythoncom.IMPLTYPEFLAG_FSOURCE
            )
            if (flags & (flagsNeeded | pythoncom.IMPLTYPEFLAG_FRESTRICTED)) == (
                flagsNeeded
            ):
                # Get the handle to the implemented interface.
                href = typeinfo.GetRefTypeOfImplType(i)
                return typeinfo.GetRefTypeInfo(href)

    def BuildEvents(self):
        # See if it is an extender object.
        try:
            mainTypeInfo = self.coDispatch.QueryInterface(
                axscript.IID_IProvideMultipleClassInfo
            )
            isMulti = 1
            numTypeInfos = mainTypeInfo.GetMultiTypeInfoCount()
        except pythoncom.com_error:
            isMulti = 0
            numTypeInfos = 1
            try:
                mainTypeInfo = self.coDispatch.QueryInterface(
                    pythoncom.IID_IProvideClassInfo
                )
            except pythoncom.com_error:
                numTypeInfos = 0
        # Create an event handler for the item.
        for item in range(numTypeInfos):
            if isMulti:
                typeinfo, flags = mainTypeInfo.GetInfoOfIndex(
                    item, axscript.MULTICLASSINFO_GETTYPEINFO
                )
            else:
                typeinfo = mainTypeInfo.GetClassInfo()
            sourceType = self.GetSourceTypeInfo(typeinfo)
            cFuncs = 0
            if sourceType:
                attr = sourceType.GetTypeAttr()
                self.iid = attr[0]
                cFuncs = attr[6]
                for i in range(cFuncs):
                    funcdesc = sourceType.GetFuncDesc(i)
                    event = Event()
                    event.Build(sourceType, funcdesc)
                    self.events[event.dispid] = event

    def Connect(self):
        if self.connection is not None or self.iid is None:
            return
        # 		trace("Connect for sink item", self.myScriptItem.name, "with IID",str(self.iid))
        self.connection = win32com.client.connect.SimpleConnection(
            self.coDispatch, self, self.iid
        )

    def Disconnect(self):
        if self.connection:
            try:
                self.connection.Disconnect()
            except pythoncom.com_error:
                pass  # Ignore disconnection errors.
            self.connection = None


class ScriptItem:
    """An item (or subitem) that is exposed to the ActiveX script"""

    def __init__(self, parentItem, name, dispatch, flags):
        self.parentItem = parentItem
        self.dispatch = dispatch
        self.name = name
        self.flags = flags
        self.eventSink = None
        self.subItems = {}
        self.createdConnections = 0
        self.isRegistered = 0

    # 		trace("Creating ScriptItem", name, "of parent", parentItem,"with dispatch", dispatch)

    def __repr__(self):
        flagsDesc = ""
        if self.flags is not None and self.flags & axscript.SCRIPTITEM_GLOBALMEMBERS:
            flagsDesc = "/Global"
        return "<%s at %d: %s%s>" % (
            self.__class__.__name__,
            id(self),
            self.name,
            flagsDesc,
        )

    def _dump_(self, level):
        flagDescs = []
        if self.flags is not None and self.flags & axscript.SCRIPTITEM_GLOBALMEMBERS:
            flagDescs.append("GLOBAL!")
        if self.flags is None or self.flags & axscript.SCRIPTITEM_ISVISIBLE == 0:
            flagDescs.append("NOT VISIBLE")
        if self.flags is not None and self.flags & axscript.SCRIPTITEM_ISSOURCE:
            flagDescs.append("EVENT SINK")
        if self.flags is not None and self.flags & axscript.SCRIPTITEM_CODEONLY:
            flagDescs.append("CODE ONLY")
        print(" " * level, "Name=", self.name, ", flags=", "/".join(flagDescs), self)
        for subItem in self.subItems.values():
            subItem._dump_(level + 1)

    def Reset(self):
        self.Disconnect()
        if self.eventSink:
            self.eventSink.Reset()
        self.isRegistered = 0
        for subItem in self.subItems.values():
            subItem.Reset()

    def Close(self):
        self.Reset()
        self.dispatch = None
        self.parentItem = None
        if self.eventSink:
            self.eventSink.Close()
            self.eventSink = None
        for subItem in self.subItems.values():
            subItem.Close()
        self.subItems = []
        self.createdConnections = 0

    def Register(self):
        if self.isRegistered:
            return
        # Get the type info to use to build this item.
        # 		if not self.dispatch:
        # 			id = self.parentItem.dispatch.GetIDsOfNames(self.name)
        # 			print "DispID of me is", id
        # 			result = self.parentItem.dispatch.Invoke(id, 0, pythoncom.DISPATCH_PROPERTYGET,1)
        # 			if type(result)==pythoncom.TypeIIDs[pythoncom.IID_IDispatch]:
        # 				self.dispatch = result
        # 			else:
        # 				print "*** No dispatch"
        # 				return
        # 			print "**** Made dispatch"
        self.isRegistered = 1
        # Register the sub-items.
        for item in self.subItems.values():
            if not item.isRegistered:
                item.Register()

    def IsGlobal(self):
        return self.flags & axscript.SCRIPTITEM_GLOBALMEMBERS

    def IsVisible(self):
        return (
            self.flags & (axscript.SCRIPTITEM_ISVISIBLE | axscript.SCRIPTITEM_ISSOURCE)
        ) != 0

    def GetEngine(self):
        item = self
        while item.parentItem.__class__ == self.__class__:
            item = item.parentItem
        return item.parentItem

    def _GetFullItemName(self):
        ret = self.name
        if self.parentItem:
            try:
                ret = self.parentItem._GetFullItemName() + "." + ret
            except AttributeError:
                pass
        return ret

    def GetSubItemClass(self):
        return self.__class__

    def GetSubItem(self, name):
        return self.subItems[name.lower()]

    def GetCreateSubItem(self, parentItem, name, dispatch, flags):
        keyName = name.lower()
        try:
            rc = self.subItems[keyName]
            # No changes allowed to existing flags.
            if not rc.flags is None and not flags is None and rc.flags != flags:
                raise Exception(scode=winerror.E_INVALIDARG)
            # Existing item must not have a dispatch.
            if not rc.dispatch is None and not dispatch is None:
                raise Exception(scode=winerror.E_INVALIDARG)
            rc.flags = flags  # Setup the real flags.
            rc.dispatch = dispatch
        except KeyError:
            rc = self.subItems[keyName] = self.GetSubItemClass()(
                parentItem, name, dispatch, flags
            )
        return rc

    # 		if self.dispatch is None:
    # 			RaiseAssert(winerror.E_UNEXPECTED, "??")

    def CreateConnections(self):
        # Create (but do not connect to) the connection points.
        if self.createdConnections:
            return
        self.createdConnections = 1
        # Nothing to do unless this is an event source
        # This flags means self, _and_ children, are connectable.
        if self.flags & axscript.SCRIPTITEM_ISSOURCE:
            self.BuildEvents()
            self.FindBuildSubItemEvents()

    def Connect(self):
        # Connect to the already created connection points.
        if self.eventSink:
            self.eventSink.Connect()
        for subItem in self.subItems.values():
            subItem.Connect()

    def Disconnect(self):
        # Disconnect from the connection points.
        if self.eventSink:
            self.eventSink.Disconnect()
        for subItem in self.subItems.values():
            subItem.Disconnect()

    def BuildEvents(self):
        if self.eventSink is not None or self.dispatch is None:
            RaiseAssert(
                winerror.E_UNEXPECTED,
                "Item already has built events, or no dispatch available?",
            )

        # 		trace("BuildEvents for named item", self._GetFullItemName())
        self.eventSink = EventSink(self, self.dispatch)
        self.eventSink.BuildEvents()

    def FindBuildSubItemEvents(self):
        # Called during connection to event source.  Seeks out and connects to
        # all children.  As per the AX spec, this is not recursive
        # (ie, children sub-items are not seeked)
        try:
            multiTypeInfo = self.dispatch.QueryInterface(
                axscript.IID_IProvideMultipleClassInfo
            )
            numTypeInfos = multiTypeInfo.GetMultiTypeInfoCount()
        except pythoncom.com_error:
            return
        for item in range(numTypeInfos):
            typeinfo, flags = multiTypeInfo.GetInfoOfIndex(
                item, axscript.MULTICLASSINFO_GETTYPEINFO
            )
            defaultType = self.GetDefaultSourceTypeInfo(typeinfo)
            index = 0
            while 1:
                try:
                    fdesc = defaultType.GetFuncDesc(index)
                except pythoncom.com_error:
                    break  # No more funcs
                index = index + 1
                dispid = fdesc[0]
                funckind = fdesc[3]
                invkind = fdesc[4]
                elemdesc = fdesc[8]
                funcflags = fdesc[9]
                try:
                    isSubObject = (
                        not (funcflags & pythoncom.FUNCFLAG_FRESTRICTED)
                        and funckind == pythoncom.FUNC_DISPATCH
                        and invkind == pythoncom.INVOKE_PROPERTYGET
                        and elemdesc[0][0] == pythoncom.VT_PTR
                        and elemdesc[0][1][0] == pythoncom.VT_USERDEFINED
                    )
                except:
                    isSubObject = 0
                if isSubObject:
                    try:
                        # We found a sub-object.
                        names = typeinfo.GetNames(dispid)
                        result = self.dispatch.Invoke(
                            dispid, 0x0, pythoncom.DISPATCH_PROPERTYGET, 1
                        )
                        # IE has an interesting problem - there are lots of synonyms for the same object.  Eg
                        # in a simple form, "window.top", "window.window", "window.parent", "window.self"
                        # all refer to the same object.  Our event implementation code does not differentiate
                        # eg, "window_onload" will fire for *all* objects named "window".  Thus,
                        # "window" and "window.window" will fire the same event handler :(
                        # One option would be to check if the sub-object is indeed the
                        # parent object - however, this would stop "top_onload" from firing,
                        # as no event handler for "top" would work.
                        # I think we simply need to connect to a *single* event handler.
                        # As use in IE is deprecated, I am not solving this now.
                        if type(result) == pythoncom.TypeIIDs[pythoncom.IID_IDispatch]:
                            name = names[0]
                            subObj = self.GetCreateSubItem(
                                self, name, result, axscript.SCRIPTITEM_ISVISIBLE
                            )
                            # print "subobj", name, "flags are", subObj.flags, "mydisp=", self.dispatch, "result disp=", result, "compare=", self.dispatch==result
                            subObj.BuildEvents()
                            subObj.Register()
                    except pythoncom.com_error:
                        pass

    def GetDefaultSourceTypeInfo(self, typeinfo):
        """Gets the typeinfo for the Default Dispatch for the passed typeinfo"""
        attr = typeinfo.GetTypeAttr()
        cFuncs = attr[6]
        typeKind = attr[5]
        if typeKind not in [pythoncom.TKIND_COCLASS, pythoncom.TKIND_INTERFACE]:
            RaiseAssert(
                winerror.E_UNEXPECTED, "The typeKind of the object is unexpected"
            )
        cImplType = attr[8]
        for i in range(cImplType):
            # Look for the [source, default] interface on the coclass
            # that isn't marked as restricted.
            flags = typeinfo.GetImplTypeFlags(i)
            if (
                flags
                & (
                    pythoncom.IMPLTYPEFLAG_FDEFAULT
                    | pythoncom.IMPLTYPEFLAG_FSOURCE
                    | pythoncom.IMPLTYPEFLAG_FRESTRICTED
                )
            ) == pythoncom.IMPLTYPEFLAG_FDEFAULT:
                # Get the handle to the implemented interface.
                href = typeinfo.GetRefTypeOfImplType(i)
                defTypeInfo = typeinfo.GetRefTypeInfo(href)
                attr = defTypeInfo.GetTypeAttr()
                typeKind = attr[5]
                typeFlags = attr[11]
                if (
                    typeKind == pythoncom.TKIND_INTERFACE
                    and typeFlags & pythoncom.TYPEFLAG_FDUAL
                ):
                    # Get corresponding Disp interface
                    # -1 is a special value which does this for us.
                    href = typeinfo.GetRefTypeOfImplType(-1)
                    return defTypeInfo.GetRefTypeInfo(href)
                else:
                    return defTypeInfo


IActiveScriptMethods = [
    "SetScriptSite",
    "GetScriptSite",
    "SetScriptState",
    "GetScriptState",
    "Close",
    "AddNamedItem",
    "AddTypeLib",
    "GetScriptDispatch",
    "GetCurrentScriptThreadID",
    "GetScriptThreadID",
    "GetScriptThreadState",
    "InterruptScriptThread",
    "Clone",
]
IActiveScriptParseMethods = ["InitNew", "AddScriptlet", "ParseScriptText"]
IObjectSafetyMethods = ["GetInterfaceSafetyOptions", "SetInterfaceSafetyOptions"]

# ActiveScriptParseProcedure is a new interface with IIS4/IE4.
IActiveScriptParseProcedureMethods = ["ParseProcedureText"]


class COMScript:
    """An ActiveX Scripting engine base class.

    This class implements the required COM interfaces for ActiveX scripting.
    """

    _public_methods_ = (
        IActiveScriptMethods
        + IActiveScriptParseMethods
        + IObjectSafetyMethods
        + IActiveScriptParseProcedureMethods
    )
    _com_interfaces_ = [
        axscript.IID_IActiveScript,
        axscript.IID_IActiveScriptParse,
        axscript.IID_IObjectSafety,
    ]  # , axscript.IID_IActiveScriptParseProcedure]

    def __init__(self):
        # Make sure we can print/trace wihout an exception!
        MakeValidSysOuts()
        # 		trace("AXScriptEngine object created", self)
        self.baseThreadId = -1
        self.debugManager = None
        self.threadState = axscript.SCRIPTTHREADSTATE_NOTINSCRIPT
        self.scriptState = axscript.SCRIPTSTATE_UNINITIALIZED
        self.scriptSite = None
        self.safetyOptions = 0
        self.lcid = 0
        self.subItems = {}
        self.scriptCodeBlocks = {}

    def _query_interface_(self, iid):
        if self.debugManager:
            return self.debugManager._query_interface_for_debugger_(iid)
        # 		trace("ScriptEngine QI - unknown IID", iid)
        return 0

    # IActiveScriptParse
    def InitNew(self):
        if self.scriptSite is not None:
            self.SetScriptState(axscript.SCRIPTSTATE_INITIALIZED)

    def AddScriptlet(
        self,
        defaultName,
        code,
        itemName,
        subItemName,
        eventName,
        delimiter,
        sourceContextCookie,
        startLineNumber,
    ):
        # 		trace ("AddScriptlet", defaultName, code, itemName, subItemName, eventName, delimiter, sourceContextCookie, startLineNumber)
        self.DoAddScriptlet(
            defaultName,
            code,
            itemName,
            subItemName,
            eventName,
            delimiter,
            sourceContextCookie,
            startLineNumber,
        )

    def ParseScriptText(
        self,
        code,
        itemName,
        context,
        delimiter,
        sourceContextCookie,
        startLineNumber,
        flags,
        bWantResult,
    ):
        # 		trace ("ParseScriptText", code[:20],"...", itemName, context, delimiter, sourceContextCookie, startLineNumber, flags, bWantResult)
        if (
            bWantResult
            or self.scriptState == axscript.SCRIPTSTATE_STARTED
            or self.scriptState == axscript.SCRIPTSTATE_CONNECTED
            or self.scriptState == axscript.SCRIPTSTATE_DISCONNECTED
        ):
            flags = flags | SCRIPTTEXT_FORCEEXECUTION
        else:
            flags = flags & (~SCRIPTTEXT_FORCEEXECUTION)

        if flags & SCRIPTTEXT_FORCEEXECUTION:
            # About to execute the code.
            self.RegisterNewNamedItems()
        return self.DoParseScriptText(
            code, sourceContextCookie, startLineNumber, bWantResult, flags
        )

    #
    # IActiveScriptParseProcedure
    def ParseProcedureText(
        self,
        code,
        formalParams,
        procName,
        itemName,
        unkContext,
        delimiter,
        contextCookie,
        startingLineNumber,
        flags,
    ):
        trace(
            "ParseProcedureText",
            code,
            formalParams,
            procName,
            itemName,
            unkContext,
            delimiter,
            contextCookie,
            startingLineNumber,
            flags,
        )
        # NOTE - this is never called, as we have disabled this interface.
        # Problem is, once enabled all even code comes via here, rather than AddScriptlet.
        # However, the "procName" is always an empty string - ie, itemName is the object whose event we are handling,
        # but no idea what the specific event is!?
        # Problem is disabling this block is that AddScriptlet is _not_ passed
        # <SCRIPT for="whatever" event="onClick" language="Python">
        # (but even for those blocks, the "onClick" information is still missing!?!?!?)

        # 		self.DoAddScriptlet(None, code, itemName, subItemName, eventName, delimiter,sourceContextCookie, startLineNumber)
        return None

    #
    # IActiveScript
    def SetScriptSite(self, site):
        # We should still work with an existing site (or so MSXML believes :)
        self.scriptSite = site
        if self.debugManager is not None:
            self.debugManager.Close()
        import traceback

        try:
            import win32com.axdebug.axdebug  # see if the core exists.

            from . import debug

            self.debugManager = debug.DebugManager(self)
        except pythoncom.com_error:
            # COM errors will occur if the debugger interface has never been
            # seen on the target system
            trace("Debugging interfaces not available - debugging is disabled..")
            self.debugManager = None
        except ImportError:
            trace(
                "Debugging extensions (axdebug) module does not exist - debugging is disabled.."
            )
            self.debugManager = None
        except:
            traceback.print_exc()
            trace(
                "*** Debugger Manager could not initialize - %s: %s"
                % (sys.exc_info()[0], sys.exc_info()[1])
            )
            self.debugManager = None

        try:
            self.lcid = site.GetLCID()
        except pythoncom.com_error:
            self.lcid = win32api.GetUserDefaultLCID()
        self.Reset()

    def GetScriptSite(self, iid):
        if self.scriptSite is None:
            raise Exception(scode=winerror.S_FALSE)
        return self.scriptSite.QueryInterface(iid)

    def SetScriptState(self, state):
        # print "SetScriptState with %s - currentstate = %s" % (state_map.get(state),state_map.get(self.scriptState))
        if state == self.scriptState:
            return
        # If closed, allow no other state transitions
        if self.scriptState == axscript.SCRIPTSTATE_CLOSED:
            raise Exception(scode=winerror.E_INVALIDARG)

        if state == axscript.SCRIPTSTATE_INITIALIZED:
            # Re-initialize - shutdown then reset.
            if self.scriptState in [
                axscript.SCRIPTSTATE_CONNECTED,
                axscript.SCRIPTSTATE_STARTED,
            ]:
                self.Stop()
        elif state == axscript.SCRIPTSTATE_STARTED:
            if self.scriptState == axscript.SCRIPTSTATE_CONNECTED:
                self.Disconnect()
            if self.scriptState == axscript.SCRIPTSTATE_DISCONNECTED:
                self.Reset()
            self.Run()
            self.ChangeScriptState(axscript.SCRIPTSTATE_STARTED)
        elif state == axscript.SCRIPTSTATE_CONNECTED:
            if self.scriptState in [
                axscript.SCRIPTSTATE_UNINITIALIZED,
                axscript.SCRIPTSTATE_INITIALIZED,
            ]:
                self.ChangeScriptState(
                    axscript.SCRIPTSTATE_STARTED
                )  # report transition through started
                self.Run()
            if self.scriptState == axscript.SCRIPTSTATE_STARTED:
                self.Connect()
                self.ChangeScriptState(state)
        elif state == axscript.SCRIPTSTATE_DISCONNECTED:
            if self.scriptState == axscript.SCRIPTSTATE_CONNECTED:
                self.Disconnect()
        elif state == axscript.SCRIPTSTATE_CLOSED:
            self.Close()
        elif state == axscript.SCRIPTSTATE_UNINITIALIZED:
            if self.scriptState == axscript.SCRIPTSTATE_STARTED:
                self.Stop()
            if self.scriptState == axscript.SCRIPTSTATE_CONNECTED:
                self.Disconnect()
            if self.scriptState == axscript.SCRIPTSTATE_DISCONNECTED:
                self.Reset()
            self.ChangeScriptState(state)
        else:
            raise Exception(scode=winerror.E_INVALIDARG)

    def GetScriptState(self):
        return self.scriptState

    def Close(self):
        # 		trace("Close")
        if self.scriptState in [
            axscript.SCRIPTSTATE_CONNECTED,
            axscript.SCRIPTSTATE_DISCONNECTED,
        ]:
            self.Stop()
        if self.scriptState in [
            axscript.SCRIPTSTATE_CONNECTED,
            axscript.SCRIPTSTATE_DISCONNECTED,
            axscript.SCRIPTSTATE_INITIALIZED,
            axscript.SCRIPTSTATE_STARTED,
        ]:
            pass  # engine.close??
        if self.scriptState in [
            axscript.SCRIPTSTATE_UNINITIALIZED,
            axscript.SCRIPTSTATE_CONNECTED,
            axscript.SCRIPTSTATE_DISCONNECTED,
            axscript.SCRIPTSTATE_INITIALIZED,
            axscript.SCRIPTSTATE_STARTED,
        ]:
            self.ChangeScriptState(axscript.SCRIPTSTATE_CLOSED)
            # Completely reset all named items (including persistent)
            for item in self.subItems.values():
                item.Close()
            self.subItems = {}
            self.baseThreadId = -1
        if self.debugManager:
            self.debugManager.Close()
            self.debugManager = None
        self.scriptSite = None
        self.scriptCodeBlocks = {}
        self.persistLoaded = 0

    def AddNamedItem(self, name, flags):
        if self.scriptSite is None:
            raise Exception(scode=winerror.E_INVALIDARG)
        try:
            unknown = self.scriptSite.GetItemInfo(name, axscript.SCRIPTINFO_IUNKNOWN)[0]
            dispatch = unknown.QueryInterface(pythoncom.IID_IDispatch)
        except pythoncom.com_error:
            raise Exception(
                scode=winerror.E_NOINTERFACE,
                desc="Object has no dispatch interface available.",
            )
        newItem = self.subItems[name] = self.GetNamedItemClass()(
            self, name, dispatch, flags
        )
        if newItem.IsGlobal():
            newItem.CreateConnections()

    def GetScriptDispatch(self, name):
        # Base classes should override.
        raise Exception(scode=winerror.E_NOTIMPL)

    def GetCurrentScriptThreadID(self):
        return self.baseThreadId

    def GetScriptThreadID(self, win32ThreadId):
        if self.baseThreadId == -1:
            raise Exception(scode=winerror.E_UNEXPECTED)
        if self.baseThreadId != win32ThreadId:
            raise Exception(scode=winerror.E_INVALIDARG)
        return self.baseThreadId

    def GetScriptThreadState(self, scriptThreadId):
        if self.baseThreadId == -1:
            raise Exception(scode=winerror.E_UNEXPECTED)
        if scriptThreadId != self.baseThreadId:
            raise Exception(scode=winerror.E_INVALIDARG)
        return self.threadState

    def AddTypeLib(self, uuid, major, minor, flags):
        # Get the win32com gencache to register this library.
        from win32com.client import gencache

        gencache.EnsureModule(uuid, self.lcid, major, minor, bForDemand=1)

    # This is never called by the C++ framework - it does magic.
    # See PyGActiveScript.cpp
    # def InterruptScriptThread(self, stidThread, exc_info, flags):
    # 	raise Exception("Not Implemented", scode=winerror.E_NOTIMPL)

    def Clone(self):
        raise Exception("Not Implemented", scode=winerror.E_NOTIMPL)

    #
    # IObjectSafety

    # Note that IE seems to insist we say we support all the flags, even tho
    # we dont accept them all.  If unknown flags come in, they are ignored, and never
    # reflected in GetInterfaceSafetyOptions and the QIs obviously fail, but still IE
    # allows our engine to initialize.
    def SetInterfaceSafetyOptions(self, iid, optionsMask, enabledOptions):
        # 		trace ("SetInterfaceSafetyOptions", iid, optionsMask, enabledOptions)
        if optionsMask & enabledOptions == 0:
            return

        # See comments above.
        # 		if (optionsMask & enabledOptions & \
        # 			~(axscript.INTERFACESAFE_FOR_UNTRUSTED_DATA | axscript.INTERFACESAFE_FOR_UNTRUSTED_CALLER)):
        # 			# request for options we don't understand
        # 			RaiseAssert(scode=winerror.E_FAIL, desc="Unknown safety options")

        if iid in [
            pythoncom.IID_IPersist,
            pythoncom.IID_IPersistStream,
            pythoncom.IID_IPersistStreamInit,
            axscript.IID_IActiveScript,
            axscript.IID_IActiveScriptParse,
        ]:
            supported = self._GetSupportedInterfaceSafetyOptions()
            self.safetyOptions = supported & optionsMask & enabledOptions
        else:
            raise Exception(scode=winerror.E_NOINTERFACE)

    def _GetSupportedInterfaceSafetyOptions(self):
        return 0

    def GetInterfaceSafetyOptions(self, iid):
        if iid in [
            pythoncom.IID_IPersist,
            pythoncom.IID_IPersistStream,
            pythoncom.IID_IPersistStreamInit,
            axscript.IID_IActiveScript,
            axscript.IID_IActiveScriptParse,
        ]:
            supported = self._GetSupportedInterfaceSafetyOptions()
            return supported, self.safetyOptions
        else:
            raise Exception(scode=winerror.E_NOINTERFACE)

    #
    # Other helpers.
    def ExecutePendingScripts(self):
        self.RegisterNewNamedItems()
        self.DoExecutePendingScripts()

    def ProcessScriptItemEvent(self, item, event, lcid, wFlags, args):
        # 		trace("ProcessScriptItemEvent", item, event, lcid, wFlags, args)
        self.RegisterNewNamedItems()
        return self.DoProcessScriptItemEvent(item, event, lcid, wFlags, args)

    def _DumpNamedItems_(self):
        for item in self.subItems.values():
            item._dump_(0)

    def ResetNamedItems(self):
        # Due to the way we work, we re-create persistent ones.
        existing = self.subItems
        self.subItems = {}
        for name, item in existing.items():
            item.Close()
            if item.flags & axscript.SCRIPTITEM_ISPERSISTENT:
                self.AddNamedItem(item.name, item.flags)

    def GetCurrentSafetyOptions(self):
        return self.safetyOptions

    def ProcessNewNamedItemsConnections(self):
        # Process all sub-items.
        for item in self.subItems.values():
            if not item.createdConnections:  # Fast-track!
                item.CreateConnections()

    def RegisterNewNamedItems(self):
        # Register all sub-items.
        for item in self.subItems.values():
            if not item.isRegistered:  # Fast-track!
                self.RegisterNamedItem(item)

    def RegisterNamedItem(self, item):
        item.Register()

    def CheckConnectedOrDisconnected(self):
        if self.scriptState in [
            axscript.SCRIPTSTATE_CONNECTED,
            axscript.SCRIPTSTATE_DISCONNECTED,
        ]:
            return
        RaiseAssert(
            winerror.E_UNEXPECTED,
            "Not connected or disconnected - %d" % self.scriptState,
        )

    def Connect(self):
        self.ProcessNewNamedItemsConnections()
        self.RegisterNewNamedItems()
        self.ConnectEventHandlers()

    def Run(self):
        # 		trace("AXScript running...")
        if (
            self.scriptState != axscript.SCRIPTSTATE_INITIALIZED
            and self.scriptState != axscript.SCRIPTSTATE_STARTED
        ):
            raise Exception(scode=winerror.E_UNEXPECTED)
        # 		self._DumpNamedItems_()
        self.ExecutePendingScripts()
        self.DoRun()

    def Stop(self):
        # Stop all executing scripts, and disconnect.
        if self.scriptState == axscript.SCRIPTSTATE_CONNECTED:
            self.Disconnect()
        # Reset back to initialized.
        self.Reset()

    def Disconnect(self):
        self.CheckConnectedOrDisconnected()
        try:
            self.DisconnectEventHandlers()
        except pythoncom.com_error:
            # Ignore errors when disconnecting.
            pass

        self.ChangeScriptState(axscript.SCRIPTSTATE_DISCONNECTED)

    def ConnectEventHandlers(self):
        # 		trace ("Connecting to event handlers")
        for item in self.subItems.values():
            item.Connect()
        self.ChangeScriptState(axscript.SCRIPTSTATE_CONNECTED)

    def DisconnectEventHandlers(self):
        # 		trace ("Disconnecting from event handlers")
        for item in self.subItems.values():
            item.Disconnect()

    def Reset(self):
        # Keeping persistent engine state, reset back an initialized state
        self.ResetNamedItems()
        self.ChangeScriptState(axscript.SCRIPTSTATE_INITIALIZED)

    def ChangeScriptState(self, state):
        # print "  ChangeScriptState with %s - currentstate = %s" % (state_map.get(state),state_map.get(self.scriptState))
        self.DisableInterrupts()
        try:
            self.scriptState = state
            try:
                if self.scriptSite:
                    self.scriptSite.OnStateChange(state)
            except pythoncom.com_error as xxx_todo_changeme:
                (hr, desc, exc, arg) = xxx_todo_changeme.args
                # Ignore all errors here - E_NOTIMPL likely from scriptlets.
        finally:
            self.EnableInterrupts()

    # This stack frame is debugged - therefore we do as little as possible in it.
    def _ApplyInScriptedSection(self, fn, args):
        if self.debugManager:
            self.debugManager.OnEnterScript()
            if self.debugManager.adb.appDebugger:
                return self.debugManager.adb.runcall(fn, *args)
            else:
                return fn(*args)
        else:
            return fn(*args)

    def ApplyInScriptedSection(self, codeBlock, fn, args):
        self.BeginScriptedSection()
        try:
            try:
                # 				print "ApplyInSS", codeBlock, fn, args
                return self._ApplyInScriptedSection(fn, args)
            finally:
                if self.debugManager:
                    self.debugManager.OnLeaveScript()
                self.EndScriptedSection()
        except:
            self.HandleException(codeBlock)

    # This stack frame is debugged - therefore we do as little as possible in it.
    def _CompileInScriptedSection(self, code, name, type):
        if self.debugManager:
            self.debugManager.OnEnterScript()
        return compile(code, name, type)

    def CompileInScriptedSection(self, codeBlock, type, realCode=None):
        if codeBlock.codeObject is not None:  # already compiled
            return 1
        if realCode is None:
            code = codeBlock.codeText
        else:
            code = realCode
        name = codeBlock.GetFileName()
        self.BeginScriptedSection()
        try:
            try:
                codeObject = self._CompileInScriptedSection(RemoveCR(code), name, type)
                codeBlock.codeObject = codeObject
                return 1
            finally:
                if self.debugManager:
                    self.debugManager.OnLeaveScript()
                self.EndScriptedSection()
        except:
            self.HandleException(codeBlock)

    # This stack frame is debugged - therefore we do as little as possible in it.
    def _ExecInScriptedSection(self, codeObject, globals, locals=None):
        if self.debugManager:
            self.debugManager.OnEnterScript()
            if self.debugManager.adb.appDebugger:
                return self.debugManager.adb.run(codeObject, globals, locals)
            else:
                exec(codeObject, globals, locals)
        else:
            exec(codeObject, globals, locals)

    def ExecInScriptedSection(self, codeBlock, globals, locals=None):
        if locals is None:
            locals = globals
        assert (
            not codeBlock.beenExecuted
        ), "This code block should not have been executed"
        codeBlock.beenExecuted = 1
        self.BeginScriptedSection()
        try:
            try:
                self._ExecInScriptedSection(codeBlock.codeObject, globals, locals)
            finally:
                if self.debugManager:
                    self.debugManager.OnLeaveScript()
                self.EndScriptedSection()
        except:
            self.HandleException(codeBlock)

    def _EvalInScriptedSection(self, codeBlock, globals, locals=None):
        if self.debugManager:
            self.debugManager.OnEnterScript()
            if self.debugManager.adb.appDebugger:
                return self.debugManager.adb.runeval(codeBlock, globals, locals)
            else:
                return eval(codeBlock, globals, locals)
        else:
            return eval(codeBlock, globals, locals)

    def EvalInScriptedSection(self, codeBlock, globals, locals=None):
        if locals is None:
            locals = globals
        assert (
            not codeBlock.beenExecuted
        ), "This code block should not have been executed"
        codeBlock.beenExecuted = 1
        self.BeginScriptedSection()
        try:
            try:
                return self._EvalInScriptedSection(
                    codeBlock.codeObject, globals, locals
                )
            finally:
                if self.debugManager:
                    self.debugManager.OnLeaveScript()
                self.EndScriptedSection()
        except:
            self.HandleException(codeBlock)

    def HandleException(self, codeBlock):
        # NOTE - Never returns - raises a ComException
        exc_type, exc_value, exc_traceback = sys.exc_info()
        # If a SERVER exception, re-raise it.  If a client side COM error, it is
        # likely to have originated from the script code itself, and therefore
        # needs to be reported like any other exception.
        if IsCOMServerException(exc_type):
            # Ensure the traceback doesnt cause a cycle.
            exc_traceback = None
            raise
        # It could be an error by another script.
        if (
            issubclass(pythoncom.com_error, exc_type)
            and exc_value.hresult == axscript.SCRIPT_E_REPORTED
        ):
            # Ensure the traceback doesnt cause a cycle.
            exc_traceback = None
            raise Exception(scode=exc_value.hresult)

        exception = error.AXScriptException(
            self, codeBlock, exc_type, exc_value, exc_traceback
        )

        # Ensure the traceback doesnt cause a cycle.
        exc_traceback = None
        result_exception = error.ProcessAXScriptException(
            self.scriptSite, self.debugManager, exception
        )
        if result_exception is not None:
            try:
                self.scriptSite.OnScriptTerminate(None, result_exception)
            except pythoncom.com_error:
                pass  # Ignore errors telling engine we stopped.
            # reset ourselves to 'connected' so further events continue to fire.
            self.SetScriptState(axscript.SCRIPTSTATE_CONNECTED)
            raise result_exception
        # I think that in some cases this should just return - but the code
        # that could return None above is disabled, so it never happens.
        RaiseAssert(
            winerror.E_UNEXPECTED, "Don't have an exception to raise to the caller!"
        )

    def BeginScriptedSection(self):
        if self.scriptSite is None:
            raise Exception(scode=winerror.E_UNEXPECTED)
        self.scriptSite.OnEnterScript()

    def EndScriptedSection(self):
        if self.scriptSite is None:
            raise Exception(scode=winerror.E_UNEXPECTED)
        self.scriptSite.OnLeaveScript()

    def DisableInterrupts(self):
        pass

    def EnableInterrupts(self):
        pass

    def GetNamedItem(self, name):
        try:
            return self.subItems[name]
        except KeyError:
            raise Exception(scode=winerror.E_INVALIDARG)

    def GetNamedItemClass(self):
        return ScriptItem

    def _AddScriptCodeBlock(self, codeBlock):
        self.scriptCodeBlocks[codeBlock.GetFileName()] = codeBlock
        if self.debugManager:
            self.debugManager.AddScriptBlock(codeBlock)


if __name__ == "__main__":
    print("This is a framework class - please use pyscript.py etc")


def dumptypeinfo(typeinfo):
    return
    attr = typeinfo.GetTypeAttr()
    # Loop over all methods
    print("Methods")
    for j in range(attr[6]):
        fdesc = list(typeinfo.GetFuncDesc(j))
        id = fdesc[0]
        try:
            names = typeinfo.GetNames(id)
        except pythoncom.ole_error:
            names = None
        doc = typeinfo.GetDocumentation(id)

        print(" ", names, "has attr", fdesc)

    # Loop over all variables (ie, properties)
    print("Variables")
    for j in range(attr[7]):
        fdesc = list(typeinfo.GetVarDesc(j))
        names = typeinfo.GetNames(id)
        print(" ", names, "has attr", fdesc)
