# This is part of the Python test suite.
# The object is registered when you first run the test suite.
# (and hopefully unregistered once done ;-)

import pythoncom
import winerror

# Ensure the vtables in the tlb are known.
from win32com import universal
from win32com.client import constants, gencache
from win32com.server.exception import COMException
from win32com.server.util import wrap

pythoncom.__future_currency__ = True
# We use the constants from the module, so must insist on a gencache.
# Otherwise, use of gencache is not necessary (tho still advised)
gencache.EnsureModule("{6BCDCB60-5605-11D0-AE5F-CADD4C000000}", 0, 1, 1)


class PyCOMTest:
    _typelib_guid_ = "{6BCDCB60-5605-11D0-AE5F-CADD4C000000}"
    _typelib_version = 1, 0
    _com_interfaces_ = ["IPyCOMTest"]
    _reg_clsid_ = "{e743d9cd-cb03-4b04-b516-11d3a81c1597}"
    _reg_progid_ = "Python.Test.PyCOMTest"

    def DoubleString(self, str):
        return str * 2

    def DoubleInOutString(self, str):
        return str * 2

    def Fire(self, nID):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetLastVarArgs(self):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetMultipleInterfaces(self, outinterface1, outinterface2):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetSafeArrays(self, attrs, attrs2, ints):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetSetDispatch(self, indisp):
        raise COMException(hresult=winerror.E_NOTIMPL)

    # Result is of type IPyCOMTest
    def GetSetInterface(self, ininterface):
        return wrap(self)

    def GetSetVariant(self, indisp):
        return indisp

    def TestByRefVariant(self, v):
        return v * 2

    def TestByRefString(self, v):
        return v * 2

    # Result is of type IPyCOMTest
    def GetSetInterfaceArray(self, ininterface):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetSetUnknown(self, inunk):
        raise COMException(hresult=winerror.E_NOTIMPL)

    # Result is of type ISimpleCounter
    def GetSimpleCounter(self):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetSimpleSafeArray(self, ints):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def GetStruct(self):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def SetIntSafeArray(self, ints):
        return len(ints)

    def SetLongLongSafeArray(self, ints):
        return len(ints)

    def SetULongLongSafeArray(self, ints):
        return len(ints)

    def SetBinSafeArray(self, buf):
        return len(buf)

    def SetVarArgs(self, *args):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def SetVariantSafeArray(self, vars):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def Start(self):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def Stop(self, nID):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def StopAll(self):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def TakeByRefDispatch(self, inout):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def TakeByRefTypedDispatch(self, inout):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def Test(self, key, inval):
        return not inval

    def Test2(self, inval):
        return inval

    def Test3(self, inval):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def Test4(self, inval):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def Test5(self, inout):
        if inout == constants.TestAttr1:
            return constants.TestAttr1_1
        elif inout == constants.TestAttr1_1:
            return constants.TestAttr1
        else:
            return -1

    def Test6(self, inval):
        return inval

    def TestInOut(self, fval, bval, lval):
        return winerror.S_OK, fval * 2, not bval, lval * 2

    def TestOptionals(self, strArg="def", sval=0, lval=1, dval=3.1400001049041748):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def TestOptionals2(self, dval, strval="", sval=1):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def CheckVariantSafeArray(self, data):
        return 1

    def LongProp(self):
        return self.longval

    def SetLongProp(self, val):
        self.longval = val

    def ULongProp(self):
        return self.ulongval

    def SetULongProp(self, val):
        self.ulongval = val

    def IntProp(self):
        return self.intval

    def SetIntProp(self, val):
        self.intval = val


class PyCOMTestMI(PyCOMTest):
    _typelib_guid_ = "{6BCDCB60-5605-11D0-AE5F-CADD4C000000}"
    _typelib_version = 1, 0
    # Interfaces with a interface name, a real IID, and an IID as a string
    _com_interfaces_ = [
        "IPyCOMTest",
        pythoncom.IID_IStream,
        str(pythoncom.IID_IStorage),
    ]
    _reg_clsid_ = "{F506E9A1-FB46-4238-A597-FA4EB69787CA}"
    _reg_progid_ = "Python.Test.PyCOMTestMI"


if __name__ == "__main__":
    import win32com.server.register

    win32com.server.register.UseCommandLine(PyCOMTest)
    win32com.server.register.UseCommandLine(PyCOMTestMI)
