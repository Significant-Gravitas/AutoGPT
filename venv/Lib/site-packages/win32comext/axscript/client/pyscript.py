"""Python ActiveX Scripting Implementation

This module implements the Python ActiveX Scripting client.

To register the implementation, simply "run" this Python program - ie
either double-click on it, or run "python.exe pyscript.py" from the
command line.
"""

import re

import pythoncom
import win32api
import win32com
import win32com.client.dynamic
import win32com.server.register
import winerror
from win32com.axscript import axscript
from win32com.axscript.client import framework, scriptdispatch
from win32com.axscript.client.framework import (
    SCRIPTTEXT_FORCEEXECUTION,
    SCRIPTTEXT_ISEXPRESSION,
    SCRIPTTEXT_ISPERSISTENT,
    Exception,
    RaiseAssert,
    trace,
)

PyScript_CLSID = "{DF630910-1C1D-11d0-AE36-8C0F5E000000}"

debugging_attr = 0


def debug_attr_print(*args):
    if debugging_attr:
        trace(*args)


def ExpandTabs(text):
    return re.sub("\t", "    ", text)


def AddCR(text):
    return re.sub("\n", "\r\n", text)


class AXScriptCodeBlock(framework.AXScriptCodeBlock):
    def GetDisplayName(self):
        return "PyScript - " + framework.AXScriptCodeBlock.GetDisplayName(self)


# There is only ever _one_ ax object - it exists in the global namespace
# for all script items.
# It performs a search from all global/visible objects
# down.
# This means that if 2 sub-objects of the same name are used
# then only one is ever reachable using the ax shortcut.
class AXScriptAttribute:
    "An attribute in a scripts namespace."

    def __init__(self, engine):
        self.__dict__["_scriptEngine_"] = engine

    def __getattr__(self, attr):
        if attr[1] == "_" and attr[:-1] == "_":
            raise AttributeError(attr)
        rc = self._FindAttribute_(attr)
        if rc is None:
            raise AttributeError(attr)
        return rc

    def _Close_(self):
        self.__dict__["_scriptEngine_"] = None

    def _DoFindAttribute_(self, obj, attr):
        try:
            return obj.subItems[attr.lower()].attributeObject
        except KeyError:
            pass
        # Check out the sub-items
        for item in obj.subItems.values():
            try:
                return self._DoFindAttribute_(item, attr)
            except AttributeError:
                pass
        raise AttributeError(attr)

    def _FindAttribute_(self, attr):
        for item in self._scriptEngine_.subItems.values():
            try:
                return self._DoFindAttribute_(item, attr)
            except AttributeError:
                pass
        # All else fails, see if it is a global
        # (mainly b/w compat)
        return getattr(self._scriptEngine_.globalNameSpaceModule, attr)


# 		raise AttributeError(attr)


class NamedScriptAttribute:
    "An explicitely named object in an objects namespace"

    # Each named object holds a reference to one of these.
    # Whenever a sub-item appears in a namespace, it is really one of these
    # objects.  Has a circular reference back to the item itself, which is
    # closed via _Close_()
    def __init__(self, scriptItem):
        self.__dict__["_scriptItem_"] = scriptItem

    def __repr__(self):
        return "<NamedItemAttribute" + repr(self._scriptItem_) + ">"

    def __getattr__(self, attr):
        # If a known subitem, return it.
        try:
            return self._scriptItem_.subItems[attr.lower()].attributeObject
        except KeyError:
            # Otherwise see if the dispatch can give it to us
            if self._scriptItem_.dispatchContainer:
                return getattr(self._scriptItem_.dispatchContainer, attr)
        raise AttributeError(attr)

    def __setattr__(self, attr, value):
        # XXX - todo - if a known item, then should call its default
        # dispatch method.
        attr = attr.lower()
        if self._scriptItem_.dispatchContainer:
            try:
                return setattr(self._scriptItem_.dispatchContainer, attr, value)
            except AttributeError:
                pass
        raise AttributeError(attr)

    def _Close_(self):
        self.__dict__["_scriptItem_"] = None


class ScriptItem(framework.ScriptItem):
    def __init__(self, parentItem, name, dispatch, flags):
        framework.ScriptItem.__init__(self, parentItem, name, dispatch, flags)
        self.scriptlets = {}
        self.attributeObject = None

    def Reset(self):
        framework.ScriptItem.Reset(self)
        if self.attributeObject:
            self.attributeObject._Close_()
        self.attributeObject = None

    def Close(self):
        framework.ScriptItem.Close(self)  # calls reset.
        self.dispatchContainer = None
        self.scriptlets = {}

    def Register(self):
        framework.ScriptItem.Register(self)
        self.attributeObject = NamedScriptAttribute(self)
        if self.dispatch:
            # Need to avoid the new Python "lazy" dispatch behaviour.
            try:
                engine = self.GetEngine()
                olerepr = clsid = None
                typeinfo = self.dispatch.GetTypeInfo()
                clsid = typeinfo.GetTypeAttr()[0]
                try:
                    olerepr = engine.mapKnownCOMTypes[clsid]
                except KeyError:
                    pass
            except pythoncom.com_error:
                typeinfo = None
            if olerepr is None:
                olerepr = win32com.client.dynamic.MakeOleRepr(
                    self.dispatch, typeinfo, None
                )
                if clsid is not None:
                    engine.mapKnownCOMTypes[clsid] = olerepr
            self.dispatchContainer = win32com.client.dynamic.CDispatch(
                self.dispatch, olerepr, self.name
            )


# 			self.dispatchContainer = win32com.client.dynamic.Dispatch(self.dispatch, userName = self.name)
# 			self.dispatchContainer = win32com.client.dynamic.DumbDispatch(self.dispatch, userName = self.name)

# 	def Connect(self):
# 		framework.ScriptItem.Connect(self)
# 	def Disconnect(self):
# 		framework.ScriptItem.Disconnect(self)


class PyScript(framework.COMScript):
    # Setup the auto-registration stuff...
    _reg_verprogid_ = "Python.AXScript.2"
    _reg_progid_ = "Python"
    # 	_reg_policy_spec_ = default
    _reg_catids_ = [axscript.CATID_ActiveScript, axscript.CATID_ActiveScriptParse]
    _reg_desc_ = "Python ActiveX Scripting Engine"
    _reg_clsid_ = PyScript_CLSID
    _reg_class_spec_ = "win32com.axscript.client.pyscript.PyScript"
    _reg_remove_keys_ = [(".pys",), ("pysFile",)]
    _reg_threading_ = "both"

    def __init__(self):
        framework.COMScript.__init__(self)
        self.globalNameSpaceModule = None
        self.codeBlocks = []
        self.scriptDispatch = None

    def InitNew(self):
        framework.COMScript.InitNew(self)
        import imp

        self.scriptDispatch = None
        self.globalNameSpaceModule = imp.new_module("__ax_main__")
        self.globalNameSpaceModule.__dict__["ax"] = AXScriptAttribute(self)

        self.codeBlocks = []
        self.persistedCodeBlocks = []
        self.mapKnownCOMTypes = {}  # Map of known CLSID to typereprs
        self.codeBlockCounter = 0

    def Stop(self):
        # Flag every pending script as already done
        for b in self.codeBlocks:
            b.beenExecuted = 1
        return framework.COMScript.Stop(self)

    def Reset(self):
        # Reset all code-blocks that are persistent, and discard the rest
        oldCodeBlocks = self.codeBlocks[:]
        self.codeBlocks = []
        for b in oldCodeBlocks:
            if b.flags & SCRIPTTEXT_ISPERSISTENT:
                b.beenExecuted = 0
                self.codeBlocks.append(b)
        return framework.COMScript.Reset(self)

    def _GetNextCodeBlockNumber(self):
        self.codeBlockCounter = self.codeBlockCounter + 1
        return self.codeBlockCounter

    def RegisterNamedItem(self, item):
        wasReg = item.isRegistered
        framework.COMScript.RegisterNamedItem(self, item)
        if not wasReg:
            # Insert into our namespace.
            # Add every item by name
            if item.IsVisible():
                self.globalNameSpaceModule.__dict__[item.name] = item.attributeObject
            if item.IsGlobal():
                # Global items means sub-items are also added...
                for subitem in item.subItems.values():
                    self.globalNameSpaceModule.__dict__[
                        subitem.name
                    ] = subitem.attributeObject
                # Also add all methods
                for name, entry in item.dispatchContainer._olerepr_.mapFuncs.items():
                    if not entry.hidden:
                        self.globalNameSpaceModule.__dict__[name] = getattr(
                            item.dispatchContainer, name
                        )

    def DoExecutePendingScripts(self):
        try:
            globs = self.globalNameSpaceModule.__dict__
            for codeBlock in self.codeBlocks:
                if not codeBlock.beenExecuted:
                    if self.CompileInScriptedSection(codeBlock, "exec"):
                        self.ExecInScriptedSection(codeBlock, globs)
        finally:
            pass

    def DoRun(self):
        pass

    def Close(self):
        self.ResetNamespace()
        self.globalNameSpaceModule = None
        self.codeBlocks = []
        self.scriptDispatch = None
        framework.COMScript.Close(self)

    def GetScriptDispatch(self, name):
        # 		trace("GetScriptDispatch with", name)
        # 		if name is not None: return None
        if self.scriptDispatch is None:
            self.scriptDispatch = scriptdispatch.MakeScriptDispatch(
                self, self.globalNameSpaceModule
            )
        return self.scriptDispatch

    def MakeEventMethodName(self, subItemName, eventName):
        return (
            subItemName[0].upper()
            + subItemName[1:]
            + "_"
            + eventName[0].upper()
            + eventName[1:]
        )

    def DoAddScriptlet(
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
        # Just store the code away - compile when called.  (JIT :-)
        item = self.GetNamedItem(itemName)
        if (
            itemName == subItemName
        ):  # Explicit handlers - eg <SCRIPT LANGUAGE="Python" for="TestForm" Event="onSubmit">
            subItem = item
        else:
            subItem = item.GetCreateSubItem(item, subItemName, None, None)
        funcName = self.MakeEventMethodName(subItemName, eventName)

        codeBlock = AXScriptCodeBlock(
            "Script Event %s" % funcName, code, sourceContextCookie, startLineNumber, 0
        )
        self._AddScriptCodeBlock(codeBlock)
        subItem.scriptlets[funcName] = codeBlock

    def DoProcessScriptItemEvent(self, item, event, lcid, wFlags, args):
        # 		trace("ScriptItemEvent", self, item, event, event.name, lcid, wFlags, args)
        funcName = self.MakeEventMethodName(item.name, event.name)
        codeBlock = function = None
        try:
            function = item.scriptlets[funcName]
            if type(function) == type(self):  # ie, is a CodeBlock instance
                codeBlock = function
                function = None
        except KeyError:
            pass
        if codeBlock is not None:
            realCode = "def %s():\n" % funcName
            for line in framework.RemoveCR(codeBlock.codeText).split("\n"):
                realCode = realCode + "\t" + line + "\n"
            realCode = realCode + "\n"
            if not self.CompileInScriptedSection(codeBlock, "exec", realCode):
                return
            dict = {}
            self.ExecInScriptedSection(
                codeBlock, self.globalNameSpaceModule.__dict__, dict
            )
            function = dict[funcName]
            # cache back in scriptlets as a function.
            item.scriptlets[funcName] = function
        if function is None:
            # still no function - see if in the global namespace.
            try:
                function = self.globalNameSpaceModule.__dict__[funcName]
            except KeyError:
                # Not there _exactly_ - do case ins search.
                funcNameLook = funcName.lower()
                for attr in self.globalNameSpaceModule.__dict__.keys():
                    if funcNameLook == attr.lower():
                        function = self.globalNameSpaceModule.__dict__[attr]
                        # cache back in scriptlets, to avoid this overhead next time
                        item.scriptlets[funcName] = function

        if function is None:
            raise Exception(scode=winerror.DISP_E_MEMBERNOTFOUND)
        return self.ApplyInScriptedSection(codeBlock, function, args)

    def DoParseScriptText(
        self, code, sourceContextCookie, startLineNumber, bWantResult, flags
    ):
        code = framework.RemoveCR(code) + "\n"
        if flags & SCRIPTTEXT_ISEXPRESSION:
            name = "Script Expression"
            exec_type = "eval"
        else:
            name = "Script Block"
            exec_type = "exec"
        num = self._GetNextCodeBlockNumber()
        if num == 1:
            num = ""
        name = "%s %s" % (name, num)
        codeBlock = AXScriptCodeBlock(
            name, code, sourceContextCookie, startLineNumber, flags
        )
        self._AddScriptCodeBlock(codeBlock)
        globs = self.globalNameSpaceModule.__dict__
        if bWantResult:  # always immediate.
            if self.CompileInScriptedSection(codeBlock, exec_type):
                if flags & SCRIPTTEXT_ISEXPRESSION:
                    return self.EvalInScriptedSection(codeBlock, globs)
                else:
                    return self.ExecInScriptedSection(codeBlock, globs)

            # else compile failed, but user chose to keep running...
        else:
            if flags & SCRIPTTEXT_FORCEEXECUTION:
                if self.CompileInScriptedSection(codeBlock, exec_type):
                    self.ExecInScriptedSection(codeBlock, globs)
            else:
                self.codeBlocks.append(codeBlock)

    def GetNamedItemClass(self):
        return ScriptItem

    def ResetNamespace(self):
        if self.globalNameSpaceModule is not None:
            try:
                self.globalNameSpaceModule.ax._Reset_()
            except AttributeError:
                pass  # ???
            globalNameSpaceModule = None


def DllRegisterServer():
    klass = PyScript
    win32com.server.register._set_subkeys(
        klass._reg_progid_ + "\\OLEScript", {}
    )  # Just a CreateKey
    # Basic Registration for wsh.
    win32com.server.register._set_string(".pys", "pysFile")
    win32com.server.register._set_string("pysFile\\ScriptEngine", klass._reg_progid_)
    guid_wsh_shellex = "{60254CA5-953B-11CF-8C96-00AA00B8708C}"
    win32com.server.register._set_string(
        "pysFile\\ShellEx\\DropHandler", guid_wsh_shellex
    )
    win32com.server.register._set_string(
        "pysFile\\ShellEx\\PropertySheetHandlers\\WSHProps", guid_wsh_shellex
    )


def Register(klass=PyScript):
    ret = win32com.server.register.UseCommandLine(
        klass, finalize_register=DllRegisterServer
    )
    return ret


if __name__ == "__main__":
    Register()
