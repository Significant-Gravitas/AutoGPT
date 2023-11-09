import win32com
import win32com.client

if type(__path__) == type(""):
    # For freeze to work!
    import sys

    try:
        import adsi

        sys.modules["win32com.adsi.adsi"] = adsi
    except ImportError:
        pass
else:
    # See if we have a special directory for the binaries (for developers)
    win32com.__PackageSupportBuildPath__(__path__)


# Some helpers
# We want to _look_ like the ADSI module, but provide some additional
# helpers.

# Of specific note - most of the interfaces supported by ADSI
# derive from IDispatch - thus, you get the custome methods from the
# interface, as well as via IDispatch.
import pythoncom

from .adsi import *

LCID = 0

IDispatchType = pythoncom.TypeIIDs[pythoncom.IID_IDispatch]
IADsContainerType = pythoncom.TypeIIDs[adsi.IID_IADsContainer]


def _get_good_ret(
    ob,
    # Named arguments used internally
    resultCLSID=None,
):
    assert resultCLSID is None, "Now have type info for ADSI objects - fix me!"
    # See if the object supports IDispatch
    if hasattr(ob, "Invoke"):
        import win32com.client.dynamic

        name = "Dispatch wrapper around %r" % ob
        return win32com.client.dynamic.Dispatch(ob, name, ADSIDispatch)
    return ob


class ADSIEnumerator:
    def __init__(self, ob):
        # Query the object for the container interface.
        self._cont_ = ob.QueryInterface(IID_IADsContainer)
        self._oleobj_ = ADsBuildEnumerator(self._cont_)  # a PyIADsEnumVARIANT
        self.index = -1

    def __getitem__(self, index):
        return self.__GetIndex(index)

    def __call__(self, index):
        return self.__GetIndex(index)

    def __GetIndex(self, index):
        if type(index) != type(0):
            raise TypeError("Only integer indexes are supported for enumerators")
        if index != self.index + 1:
            # Index requested out of sequence.
            raise ValueError("You must index this object sequentially")
        self.index = index
        result = ADsEnumerateNext(self._oleobj_, 1)
        if len(result):
            return _get_good_ret(result[0])
        # Failed - reset for next time around.
        self.index = -1
        self._oleobj_ = ADsBuildEnumerator(self._cont_)  # a PyIADsEnumVARIANT
        raise IndexError("list index out of range")


class ADSIDispatch(win32com.client.CDispatch):
    def _wrap_dispatch_(
        self, ob, userName=None, returnCLSID=None, UnicodeToString=None
    ):
        assert UnicodeToString is None, "this is deprectated and will be removed"
        if not userName:
            userName = "ADSI-object"
        olerepr = win32com.client.dynamic.MakeOleRepr(ob, None, None)
        return ADSIDispatch(ob, olerepr, userName)

    def _NewEnum(self):
        try:
            return ADSIEnumerator(self)
        except pythoncom.com_error:
            # doesnt support it - let our base try!
            return win32com.client.CDispatch._NewEnum(self)

    def __getattr__(self, attr):
        try:
            return getattr(self._oleobj_, attr)
        except AttributeError:
            return win32com.client.CDispatch.__getattr__(self, attr)

    def QueryInterface(self, iid):
        ret = self._oleobj_.QueryInterface(iid)
        return _get_good_ret(ret)


# We override the global methods to do the right thing.
_ADsGetObject = ADsGetObject  # The one in the .pyd


def ADsGetObject(path, iid=pythoncom.IID_IDispatch):
    ret = _ADsGetObject(path, iid)
    return _get_good_ret(ret)


_ADsOpenObject = ADsOpenObject


def ADsOpenObject(path, username, password, reserved=0, iid=pythoncom.IID_IDispatch):
    ret = _ADsOpenObject(path, username, password, reserved, iid)
    return _get_good_ret(ret)
