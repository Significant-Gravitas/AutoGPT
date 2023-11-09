""" General Server side utilities 
"""
import pythoncom
import winerror

from . import policy
from .exception import COMException


def wrap(ob, iid=None, usePolicy=None, useDispatcher=None):
    """Wraps an object in a PyGDispatch gateway.

    Returns a client side PyI{iid} interface.

    Interface and gateway support must exist for the specified IID, as
    the QueryInterface() method is used.

    """
    if usePolicy is None:
        usePolicy = policy.DefaultPolicy
    if useDispatcher == 1:  # True will also work here.
        import win32com.server.dispatcher

        useDispatcher = win32com.server.dispatcher.DefaultDebugDispatcher
    if useDispatcher is None or useDispatcher == 0:
        ob = usePolicy(ob)
    else:
        ob = useDispatcher(usePolicy, ob)

    # get a PyIDispatch, which interfaces to PyGDispatch
    ob = pythoncom.WrapObject(ob)
    if iid is not None:
        ob = ob.QueryInterface(iid)  # Ask the PyIDispatch if it supports it?
    return ob


def unwrap(ob):
    """Unwraps an interface.

    Given an interface which wraps up a Gateway, return the object behind
    the gateway.
    """
    ob = pythoncom.UnwrapObject(ob)
    # see if the object is a dispatcher
    if hasattr(ob, "policy"):
        ob = ob.policy
    return ob._obj_


class ListEnumerator:
    """A class to expose a Python sequence as an EnumVARIANT.

    Create an instance of this class passing a sequence (list, tuple, or
    any sequence protocol supporting object) and it will automatically
    support the EnumVARIANT interface for the object.

    See also the @NewEnum@ function, which can be used to turn the
    instance into an actual COM server.
    """

    _public_methods_ = ["Next", "Skip", "Reset", "Clone"]

    def __init__(self, data, index=0, iid=pythoncom.IID_IEnumVARIANT):
        self._list_ = data
        self.index = index
        self._iid_ = iid

    def _query_interface_(self, iid):
        if iid == self._iid_:
            return 1

    def Next(self, count):
        result = self._list_[self.index : self.index + count]
        self.Skip(count)
        return result

    def Skip(self, count):
        end = self.index + count
        if end > len(self._list_):
            end = len(self._list_)
        self.index = end

    def Reset(self):
        self.index = 0

    def Clone(self):
        return self._wrap(self.__class__(self._list_, self.index))

    def _wrap(self, ob):
        return wrap(ob)


class ListEnumeratorGateway(ListEnumerator):
    """A List Enumerator which wraps a sequence's items in gateways.

    If a sequence contains items (objects) that have not been wrapped for
    return through the COM layers, then a ListEnumeratorGateway can be
    used to wrap those items before returning them (from the Next() method).

    See also the @ListEnumerator@ class and the @NewEnum@ function.
    """

    def Next(self, count):
        result = self._list_[self.index : self.index + count]
        self.Skip(count)
        return map(self._wrap, result)


def NewEnum(
    seq,
    cls=ListEnumerator,
    iid=pythoncom.IID_IEnumVARIANT,
    usePolicy=None,
    useDispatcher=None,
):
    """Creates a new enumerator COM server.

    This function creates a new COM Server that implements the
    IID_IEnumVARIANT interface.

    A COM server that can enumerate the passed in sequence will be
    created, then wrapped up for return through the COM framework.
    Optionally, a custom COM server for enumeration can be passed
    (the default is @ListEnumerator@), and the specific IEnum
    interface can be specified.
    """
    ob = cls(seq, iid=iid)
    return wrap(ob, iid, usePolicy=usePolicy, useDispatcher=useDispatcher)


class Collection:
    "A collection of VARIANT values."

    _public_methods_ = ["Item", "Count", "Add", "Remove", "Insert"]

    def __init__(self, data=None, readOnly=0):
        if data is None:
            data = []
        self.data = data

        # disable Add/Remove if read-only. note that we adjust _public_methods_
        # on this instance only.
        if readOnly:
            self._public_methods_ = ["Item", "Count"]

    # This method is also used as the "default" method.
    # Thus "print ob" will cause this to be called with zero
    # params.  Handle this slightly more elegantly here.
    # Ideally the  policy should handle this.
    def Item(self, *args):
        if len(args) != 1:
            raise COMException(scode=winerror.DISP_E_BADPARAMCOUNT)

        try:
            return self.data[args[0]]
        except IndexError as desc:
            raise COMException(scode=winerror.DISP_E_BADINDEX, desc=str(desc))

    _value_ = Item

    def Count(self):
        return len(self.data)

    def Add(self, value):
        self.data.append(value)

    def Remove(self, index):
        try:
            del self.data[index]
        except IndexError as desc:
            raise COMException(scode=winerror.DISP_E_BADINDEX, desc=str(desc))

    def Insert(self, index, value):
        try:
            index = int(index)
        except (ValueError, TypeError):
            raise COMException(scode=winerror.DISP_E_TYPEMISMATCH)
        self.data.insert(index, value)

    def _NewEnum(self):
        return NewEnum(self.data)


def NewCollection(seq, cls=Collection):
    """Creates a new COM collection object

    This function creates a new COM Server that implements the
    common collection protocols, including enumeration. (_NewEnum)

    A COM server that can enumerate the passed in sequence will be
    created, then wrapped up for return through the COM framework.
    Optionally, a custom COM server for enumeration can be passed
    (the default is @Collection@).
    """
    return pythoncom.WrapObject(
        policy.DefaultPolicy(cls(seq)), pythoncom.IID_IDispatch, pythoncom.IID_IDispatch
    )


class FileStream:
    _public_methods_ = ["Read", "Write", "Clone", "CopyTo", "Seek"]
    _com_interfaces_ = [pythoncom.IID_IStream]

    def __init__(self, file):
        self.file = file

    def Read(self, amount):
        return self.file.read(amount)

    def Write(self, data):
        self.file.write(data)
        return len(data)

    def Clone(self):
        return self._wrap(self.__class__(self.file))

    def CopyTo(self, dest, cb):
        data = self.file.read(cb)
        cbread = len(data)
        dest.Write(data)  ## ??? Write does not currently return the length ???
        return cbread, cbread

    def Seek(self, offset, origin):
        # how convient that the 'origin' values are the same as the CRT :)
        self.file.seek(offset, origin)
        return self.file.tell()

    def _wrap(self, ob):
        return wrap(ob)
