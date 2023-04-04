# testClipboard.py
import unittest

import pythoncom
import win32clipboard
import win32con
import winerror
from win32com.server.exception import COMException
from win32com.server.util import NewEnum, wrap

IDataObject_Methods = """GetData GetDataHere QueryGetData
                         GetCanonicalFormatEtc SetData EnumFormatEtc
                         DAdvise DUnadvise EnumDAdvise""".split()

# A COM object implementing IDataObject used for basic testing.
num_do_objects = 0


def WrapCOMObject(ob, iid=None):
    return wrap(ob, iid=iid, useDispatcher=0)


class TestDataObject:
    _com_interfaces_ = [pythoncom.IID_IDataObject]
    _public_methods_ = IDataObject_Methods

    def __init__(self, bytesval):
        global num_do_objects
        num_do_objects += 1
        self.bytesval = bytesval
        self.supported_fe = []
        for cf in (win32con.CF_TEXT, win32con.CF_UNICODETEXT):
            fe = cf, None, pythoncom.DVASPECT_CONTENT, -1, pythoncom.TYMED_HGLOBAL
            self.supported_fe.append(fe)

    def __del__(self):
        global num_do_objects
        num_do_objects -= 1

    def _query_interface_(self, iid):
        if iid == pythoncom.IID_IEnumFORMATETC:
            return NewEnum(self.supported_fe, iid=iid)

    def GetData(self, fe):
        ret_stg = None
        cf, target, aspect, index, tymed = fe
        if aspect & pythoncom.DVASPECT_CONTENT and tymed == pythoncom.TYMED_HGLOBAL:
            if cf == win32con.CF_TEXT:
                ret_stg = pythoncom.STGMEDIUM()
                ret_stg.set(pythoncom.TYMED_HGLOBAL, self.bytesval)
            elif cf == win32con.CF_UNICODETEXT:
                ret_stg = pythoncom.STGMEDIUM()
                ret_stg.set(pythoncom.TYMED_HGLOBAL, self.bytesval.decode("latin1"))

        if ret_stg is None:
            raise COMException(hresult=winerror.E_NOTIMPL)
        return ret_stg

    def GetDataHere(self, fe):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def QueryGetData(self, fe):
        cf, target, aspect, index, tymed = fe
        if aspect & pythoncom.DVASPECT_CONTENT == 0:
            raise COMException(hresult=winerror.DV_E_DVASPECT)
        if tymed != pythoncom.TYMED_HGLOBAL:
            raise COMException(hresult=winerror.DV_E_TYMED)
        return None  # should check better

    def GetCanonicalFormatEtc(self, fe):
        RaiseCOMException(winerror.DATA_S_SAMEFORMATETC)
        # return fe

    def SetData(self, fe, medium):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def EnumFormatEtc(self, direction):
        if direction != pythoncom.DATADIR_GET:
            raise COMException(hresult=winerror.E_NOTIMPL)
        return NewEnum(self.supported_fe, iid=pythoncom.IID_IEnumFORMATETC)

    def DAdvise(self, fe, flags, sink):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def DUnadvise(self, connection):
        raise COMException(hresult=winerror.E_NOTIMPL)

    def EnumDAdvise(self):
        raise COMException(hresult=winerror.E_NOTIMPL)


class ClipboardTester(unittest.TestCase):
    def setUp(self):
        pythoncom.OleInitialize()

    def tearDown(self):
        try:
            pythoncom.OleFlushClipboard()
        except pythoncom.com_error:
            # We never set anything!
            pass

    def testIsCurrentClipboard(self):
        do = TestDataObject(b"Hello from Python")
        do = WrapCOMObject(do, iid=pythoncom.IID_IDataObject)
        pythoncom.OleSetClipboard(do)
        self.assertTrue(pythoncom.OleIsCurrentClipboard(do))

    def testComToWin32(self):
        # Set the data via our DataObject
        do = TestDataObject(b"Hello from Python")
        do = WrapCOMObject(do, iid=pythoncom.IID_IDataObject)
        pythoncom.OleSetClipboard(do)
        # Then get it back via the standard win32 clipboard functions.
        win32clipboard.OpenClipboard()
        got = win32clipboard.GetClipboardData(win32con.CF_TEXT)
        # CF_TEXT gives bytes.
        expected = b"Hello from Python"
        self.assertEqual(got, expected)
        # Now check unicode
        got = win32clipboard.GetClipboardData(win32con.CF_UNICODETEXT)
        self.assertEqual(got, "Hello from Python")
        win32clipboard.CloseClipboard()

    def testWin32ToCom(self):
        # Set the data via the std win32 clipboard functions.
        val = b"Hello again!"  # always bytes
        win32clipboard.OpenClipboard()
        win32clipboard.SetClipboardData(win32con.CF_TEXT, val)
        win32clipboard.CloseClipboard()
        # and get it via an IDataObject provided by COM
        do = pythoncom.OleGetClipboard()
        cf = (
            win32con.CF_TEXT,
            None,
            pythoncom.DVASPECT_CONTENT,
            -1,
            pythoncom.TYMED_HGLOBAL,
        )
        stg = do.GetData(cf)
        got = stg.data
        # The data we get back has the \0, as our STGMEDIUM has no way of
        # knowing if it meant to be a string, or a binary buffer, so
        # it must return it too.
        self.assertTrue(got, b"Hello again!\0")

    def testDataObjectFlush(self):
        do = TestDataObject(b"Hello from Python")
        do = WrapCOMObject(do, iid=pythoncom.IID_IDataObject)
        pythoncom.OleSetClipboard(do)
        self.assertEqual(num_do_objects, 1)

        do = None  # clear my ref!
        pythoncom.OleFlushClipboard()
        self.assertEqual(num_do_objects, 0)

    def testDataObjectReset(self):
        do = TestDataObject(b"Hello from Python")
        do = WrapCOMObject(do)
        pythoncom.OleSetClipboard(do)
        do = None  # clear my ref!
        self.assertEqual(num_do_objects, 1)
        pythoncom.OleSetClipboard(None)
        self.assertEqual(num_do_objects, 0)


if __name__ == "__main__":
    from win32com.test import util

    util.testmain()
