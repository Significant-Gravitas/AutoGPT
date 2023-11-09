"""Python.Dictionary COM Server.

This module implements a simple COM server that acts much like a Python
dictionary or as a standard string-keyed VB Collection.  The keys of
the dictionary are strings and are case-insensitive.

It uses a highly customized policy to fine-tune the behavior exposed to
the COM client.

The object exposes the following properties:

    int Count                       (readonly)
    VARIANT Item(BSTR key)          (propget for Item)
    Item(BSTR key, VARIANT value)   (propput for Item)

    Note that 'Item' is the default property, so the following forms of
    VB code are acceptable:

        set ob = CreateObject("Python.Dictionary")
        ob("hello") = "there"
        ob.Item("hi") = ob("HELLO")

All keys are defined, returning VT_NULL (None) if a value has not been
stored.  To delete a key, simply assign VT_NULL to the key.

The object responds to the _NewEnum method by returning an enumerator over
the dictionary's keys. This allows for the following type of VB code:

    for each name in ob
        debug.print name, ob(name)
    next
"""


import pythoncom
import pywintypes
import winerror
from pythoncom import DISPATCH_METHOD, DISPATCH_PROPERTYGET
from win32com.server import policy, util
from win32com.server.exception import COMException
from winerror import S_OK


class DictionaryPolicy(policy.BasicWrapPolicy):
    ### BasicWrapPolicy looks for this
    _com_interfaces_ = []

    ### BasicWrapPolicy looks for this
    _name_to_dispid_ = {
        "item": pythoncom.DISPID_VALUE,
        "_newenum": pythoncom.DISPID_NEWENUM,
        "count": 1,
    }

    ### Auto-Registration process looks for these...
    _reg_desc_ = "Python Dictionary"
    _reg_clsid_ = "{39b61048-c755-11d0-86fa-00c04fc2e03e}"
    _reg_progid_ = "Python.Dictionary"
    _reg_verprogid_ = "Python.Dictionary.1"
    _reg_policy_spec_ = "win32com.servers.dictionary.DictionaryPolicy"

    def _CreateInstance_(self, clsid, reqIID):
        self._wrap_({})
        return pythoncom.WrapObject(self, reqIID)

    def _wrap_(self, ob):
        self._obj_ = ob  # ob should be a dictionary

    def _invokeex_(self, dispid, lcid, wFlags, args, kwargs, serviceProvider):
        if dispid == 0:  # item
            l = len(args)
            if l < 1:
                raise COMException(
                    desc="not enough parameters", scode=winerror.DISP_E_BADPARAMCOUNT
                )

            key = args[0]
            if type(key) not in [str, str]:
                ### the nArgErr thing should be 0-based, not reversed... sigh
                raise COMException(
                    desc="Key must be a string", scode=winerror.DISP_E_TYPEMISMATCH
                )

            key = key.lower()

            if wFlags & (DISPATCH_METHOD | DISPATCH_PROPERTYGET):
                if l > 1:
                    raise COMException(scode=winerror.DISP_E_BADPARAMCOUNT)
                try:
                    return self._obj_[key]
                except KeyError:
                    return None  # unknown keys return None (VT_NULL)

            if l != 2:
                raise COMException(scode=winerror.DISP_E_BADPARAMCOUNT)
            if args[1] is None:
                # delete a key when None is assigned to it
                try:
                    del self._obj_[key]
                except KeyError:
                    pass
            else:
                self._obj_[key] = args[1]
            return S_OK

        if dispid == 1:  # count
            if not wFlags & DISPATCH_PROPERTYGET:
                raise COMException(scode=winerror.DISP_E_MEMBERNOTFOUND)  # not found
            if len(args) != 0:
                raise COMException(scode=winerror.DISP_E_BADPARAMCOUNT)
            return len(self._obj_)

        if dispid == pythoncom.DISPID_NEWENUM:
            return util.NewEnum(list(self._obj_.keys()))

        raise COMException(scode=winerror.DISP_E_MEMBERNOTFOUND)

    def _getidsofnames_(self, names, lcid):
        ### this is a copy of MappedWrapPolicy._getidsofnames_ ...

        name = names[0].lower()
        try:
            return (self._name_to_dispid_[name],)
        except KeyError:
            raise COMException(
                scode=winerror.DISP_E_MEMBERNOTFOUND, desc="Member not found"
            )


def Register():
    from win32com.server.register import UseCommandLine

    return UseCommandLine(DictionaryPolicy)


if __name__ == "__main__":
    Register()
