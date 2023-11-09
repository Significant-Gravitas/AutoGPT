"""dynamic dispatch objects for AX Script.

 This is an IDispatch object that a scripting host may use to
 query and invoke methods on the main script.  Not may hosts use
 this yet, so it is not well tested!
"""

import types

import pythoncom
import win32com.server.policy
import win32com.server.util
import winerror
from win32com.axscript import axscript
from win32com.client import Dispatch
from win32com.server.exception import COMException

debugging = 0

PyIDispatchType = pythoncom.TypeIIDs[pythoncom.IID_IDispatch]


def _is_callable(obj):
    return type(obj) in [types.FunctionType, types.MethodType]
    # ignore hasattr(obj, "__call__") as this means all COM objects!


class ScriptDispatch:
    _public_methods_ = []

    def __init__(self, engine, scriptNamespace):
        self.engine = engine
        self.scriptNamespace = scriptNamespace

    def _dynamic_(self, name, lcid, wFlags, args):
        # Ensure any newly added items are available.
        self.engine.RegisterNewNamedItems()
        self.engine.ProcessNewNamedItemsConnections()
        if wFlags & pythoncom.INVOKE_FUNC:
            # attempt to call a function
            try:
                func = getattr(self.scriptNamespace, name)
                if not _is_callable(func):
                    raise AttributeError(name)  # Not a function.
                realArgs = []
                for arg in args:
                    if type(arg) == PyIDispatchType:
                        realArgs.append(Dispatch(arg))
                    else:
                        realArgs.append(arg)
                # xxx - todo - work out what code block to pass???
                return self.engine.ApplyInScriptedSection(None, func, tuple(realArgs))

            except AttributeError:
                if not wFlags & pythoncom.DISPATCH_PROPERTYGET:
                    raise COMException(scode=winerror.DISP_E_MEMBERNOTFOUND)
        if wFlags & pythoncom.DISPATCH_PROPERTYGET:
            # attempt to get a property
            try:
                ret = getattr(self.scriptNamespace, name)
                if _is_callable(ret):
                    raise AttributeError(name)  # Not a property.
            except AttributeError:
                raise COMException(scode=winerror.DISP_E_MEMBERNOTFOUND)
            except COMException as instance:
                raise
            except:
                ret = self.engine.HandleException()
            return ret

        raise COMException(scode=winerror.DISP_E_MEMBERNOTFOUND)


class StrictDynamicPolicy(win32com.server.policy.DynamicPolicy):
    def _wrap_(self, object):
        win32com.server.policy.DynamicPolicy._wrap_(self, object)
        if hasattr(self._obj_, "scriptNamespace"):
            for name in dir(self._obj_.scriptNamespace):
                self._dyn_dispid_to_name_[self._getdispid_(name, 0)] = name

    def _getmembername_(self, dispid):
        try:
            return str(self._dyn_dispid_to_name_[dispid])
        except KeyError:
            raise COMException(scode=winerror.DISP_E_UNKNOWNNAME, desc="Name not found")

    def _getdispid_(self, name, fdex):
        try:
            func = getattr(self._obj_.scriptNamespace, str(name))
        except AttributeError:
            raise COMException(scode=winerror.DISP_E_MEMBERNOTFOUND)
        # 		if not _is_callable(func):
        return win32com.server.policy.DynamicPolicy._getdispid_(self, name, fdex)


def _wrap_debug(obj):
    return win32com.server.util.wrap(
        obj,
        usePolicy=StrictDynamicPolicy,
        useDispatcher=win32com.server.policy.DispatcherWin32trace,
    )


def _wrap_nodebug(obj):
    return win32com.server.util.wrap(obj, usePolicy=StrictDynamicPolicy)


if debugging:
    _wrap = _wrap_debug
else:
    _wrap = _wrap_nodebug


def MakeScriptDispatch(engine, namespace):
    return _wrap(ScriptDispatch(engine, namespace))
