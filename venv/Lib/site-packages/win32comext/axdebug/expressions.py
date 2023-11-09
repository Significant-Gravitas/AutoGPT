import io
import string
import sys
import traceback
from pprint import pprint

import winerror
from win32com.server.exception import COMException

from . import axdebug, gateways
from .util import RaiseNotImpl, _wrap, _wrap_remove


# Given an object, return a nice string
def MakeNiceString(ob):
    stream = io.StringIO()
    pprint(ob, stream)
    return string.strip(stream.getvalue())


class ProvideExpressionContexts(gateways.ProvideExpressionContexts):
    pass


class ExpressionContext(gateways.DebugExpressionContext):
    def __init__(self, frame):
        self.frame = frame

    def ParseLanguageText(self, code, radix, delim, flags):
        return _wrap(
            Expression(self.frame, code, radix, delim, flags),
            axdebug.IID_IDebugExpression,
        )

    def GetLanguageInfo(self):
        #               print "GetLanguageInfo"
        return "Python", "{DF630910-1C1D-11d0-AE36-8C0F5E000000}"


class Expression(gateways.DebugExpression):
    def __init__(self, frame, code, radix, delim, flags):
        self.callback = None
        self.frame = frame
        self.code = code
        self.radix = radix
        self.delim = delim
        self.flags = flags
        self.isComplete = 0
        self.result = None
        self.hresult = winerror.E_UNEXPECTED

    def Start(self, callback):
        try:
            try:
                try:
                    self.result = eval(
                        self.code, self.frame.f_globals, self.frame.f_locals
                    )
                except SyntaxError:
                    exec(self.code, self.frame.f_globals, self.frame.f_locals)
                    self.result = ""
                self.hresult = 0
            except:
                l = traceback.format_exception_only(
                    sys.exc_info()[0], sys.exc_info()[1]
                )
                # l is a list of strings with trailing "\n"
                self.result = string.join(map(lambda s: s[:-1], l), "\n")
                self.hresult = winerror.E_FAIL
        finally:
            self.isComplete = 1
            callback.onComplete()

    def Abort(self):
        print("** ABORT **")

    def QueryIsComplete(self):
        return self.isComplete

    def GetResultAsString(self):
        #               print "GetStrAsResult returning", self.result
        return self.hresult, MakeNiceString(self.result)

    def GetResultAsDebugProperty(self):
        result = _wrap(
            DebugProperty(self.code, self.result, None, self.hresult),
            axdebug.IID_IDebugProperty,
        )
        return self.hresult, result


def MakeEnumDebugProperty(object, dwFieldSpec, nRadix, iid, stackFrame=None):
    name_vals = []
    if hasattr(object, "items") and hasattr(object, "keys"):  # If it is a dict.
        name_vals = iter(object.items())
        dictionary = object
    elif hasattr(object, "__dict__"):  # object with dictionary, module
        name_vals = iter(object.__dict__.items())
        dictionary = object.__dict__
    infos = []
    for name, val in name_vals:
        infos.append(
            GetPropertyInfo(name, val, dwFieldSpec, nRadix, 0, dictionary, stackFrame)
        )
    return _wrap(EnumDebugPropertyInfo(infos), axdebug.IID_IEnumDebugPropertyInfo)


def GetPropertyInfo(
    obname, obvalue, dwFieldSpec, nRadix, hresult=0, dictionary=None, stackFrame=None
):
    # returns a tuple
    name = typ = value = fullname = attrib = dbgprop = None
    if dwFieldSpec & axdebug.DBGPROP_INFO_VALUE:
        value = MakeNiceString(obvalue)
    if dwFieldSpec & axdebug.DBGPROP_INFO_NAME:
        name = obname
    if dwFieldSpec & axdebug.DBGPROP_INFO_TYPE:
        if hresult:
            typ = "Error"
        else:
            try:
                typ = type(obvalue).__name__
            except AttributeError:
                typ = str(type(obvalue))
    if dwFieldSpec & axdebug.DBGPROP_INFO_FULLNAME:
        fullname = obname
    if dwFieldSpec & axdebug.DBGPROP_INFO_ATTRIBUTES:
        if hasattr(obvalue, "has_key") or hasattr(
            obvalue, "__dict__"
        ):  # If it is a dict or object
            attrib = axdebug.DBGPROP_ATTRIB_VALUE_IS_EXPANDABLE
        else:
            attrib = 0
    if dwFieldSpec & axdebug.DBGPROP_INFO_DEBUGPROP:
        dbgprop = _wrap(
            DebugProperty(name, obvalue, None, hresult, dictionary, stackFrame),
            axdebug.IID_IDebugProperty,
        )
    return name, typ, value, fullname, attrib, dbgprop


from win32com.server.util import ListEnumeratorGateway


class EnumDebugPropertyInfo(ListEnumeratorGateway):
    """A class to expose a Python sequence as an EnumDebugCodeContexts

    Create an instance of this class passing a sequence (list, tuple, or
    any sequence protocol supporting object) and it will automatically
    support the EnumDebugCodeContexts interface for the object.

    """

    _public_methods_ = ListEnumeratorGateway._public_methods_ + ["GetCount"]
    _com_interfaces_ = [axdebug.IID_IEnumDebugPropertyInfo]

    def GetCount(self):
        return len(self._list_)

    def _wrap(self, ob):
        return ob


class DebugProperty:
    _com_interfaces_ = [axdebug.IID_IDebugProperty]
    _public_methods_ = [
        "GetPropertyInfo",
        "GetExtendedInfo",
        "SetValueAsString",
        "EnumMembers",
        "GetParent",
    ]

    def __init__(
        self, name, value, parent=None, hresult=0, dictionary=None, stackFrame=None
    ):
        self.name = name
        self.value = value
        self.parent = parent
        self.hresult = hresult
        self.dictionary = dictionary
        self.stackFrame = stackFrame

    def GetPropertyInfo(self, dwFieldSpec, nRadix):
        return GetPropertyInfo(
            self.name,
            self.value,
            dwFieldSpec,
            nRadix,
            self.hresult,
            dictionary,
            stackFrame,
        )

    def GetExtendedInfo(self):  ### Note - not in the framework.
        RaiseNotImpl("DebugProperty::GetExtendedInfo")

    def SetValueAsString(self, value, radix):
        if self.stackFrame and self.dictionary:
            self.dictionary[self.name] = eval(
                value, self.stackFrame.f_globals, self.stackFrame.f_locals
            )
        else:
            RaiseNotImpl("DebugProperty::SetValueAsString")

    def EnumMembers(self, dwFieldSpec, nRadix, iid):
        # Returns IEnumDebugPropertyInfo
        return MakeEnumDebugProperty(
            self.value, dwFieldSpec, nRadix, iid, self.stackFrame
        )

    def GetParent(self):
        # return IDebugProperty
        RaiseNotImpl("DebugProperty::GetParent")
