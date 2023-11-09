import unittest

import pythoncom
import win32com.server.util
import win32com.test.util
from pywin32_testutil import str2bytes


class Persists:
    _public_methods_ = [
        "GetClassID",
        "IsDirty",
        "Load",
        "Save",
        "GetSizeMax",
        "InitNew",
    ]
    _com_interfaces_ = [pythoncom.IID_IPersistStreamInit]

    def __init__(self):
        self.data = str2bytes("abcdefg")
        self.dirty = 1

    def GetClassID(self):
        return pythoncom.IID_NULL

    def IsDirty(self):
        return self.dirty

    def Load(self, stream):
        self.data = stream.Read(26)

    def Save(self, stream, clearDirty):
        stream.Write(self.data)
        if clearDirty:
            self.dirty = 0

    def GetSizeMax(self):
        return 1024

    def InitNew(self):
        pass


class Stream:
    _public_methods_ = ["Read", "Write", "Seek"]
    _com_interfaces_ = [pythoncom.IID_IStream]

    def __init__(self, data):
        self.data = data
        self.index = 0

    def Read(self, amount):
        result = self.data[self.index : self.index + amount]
        self.index = self.index + amount
        return result

    def Write(self, data):
        self.data = data
        self.index = 0
        return len(data)

    def Seek(self, dist, origin):
        if origin == pythoncom.STREAM_SEEK_SET:
            self.index = dist
        elif origin == pythoncom.STREAM_SEEK_CUR:
            self.index = self.index + dist
        elif origin == pythoncom.STREAM_SEEK_END:
            self.index = len(self.data) + dist
        else:
            raise ValueError("Unknown Seek type: " + str(origin))
        if self.index < 0:
            self.index = 0
        else:
            self.index = min(self.index, len(self.data))
        return self.index


class BadStream(Stream):
    """PyGStream::Read could formerly overflow buffer if the python implementation
    returned more data than requested.
    """

    def Read(self, amount):
        return str2bytes("x") * (amount + 1)


class StreamTest(win32com.test.util.TestCase):
    def _readWrite(self, data, write_stream, read_stream=None):
        if read_stream is None:
            read_stream = write_stream
        write_stream.Write(data)
        read_stream.Seek(0, pythoncom.STREAM_SEEK_SET)
        got = read_stream.Read(len(data))
        self.assertEqual(data, got)
        read_stream.Seek(1, pythoncom.STREAM_SEEK_SET)
        got = read_stream.Read(len(data) - 2)
        self.assertEqual(data[1:-1], got)

    def testit(self):
        mydata = str2bytes("abcdefghijklmnopqrstuvwxyz")

        # First test the objects just as Python objects...
        s = Stream(mydata)
        p = Persists()

        p.Load(s)
        p.Save(s, 0)
        self.assertEqual(s.data, mydata)

        # Wrap the Python objects as COM objects, and make the calls as if
        # they were non-Python COM objects.
        s2 = win32com.server.util.wrap(s, pythoncom.IID_IStream)
        p2 = win32com.server.util.wrap(p, pythoncom.IID_IPersistStreamInit)

        self._readWrite(mydata, s, s)
        self._readWrite(mydata, s, s2)
        self._readWrite(mydata, s2, s)
        self._readWrite(mydata, s2, s2)

        self._readWrite(str2bytes("string with\0a NULL"), s2, s2)
        # reset the stream
        s.Write(mydata)
        p2.Load(s2)
        p2.Save(s2, 0)
        self.assertEqual(s.data, mydata)

    def testseek(self):
        s = Stream(str2bytes("yo"))
        s = win32com.server.util.wrap(s, pythoncom.IID_IStream)
        # we used to die in py3k passing a value > 32bits
        s.Seek(0x100000000, pythoncom.STREAM_SEEK_SET)

    def testerrors(self):
        # setup a test logger to capture tracebacks etc.
        records, old_log = win32com.test.util.setup_test_logger()
        ## check for buffer overflow in Read method
        badstream = BadStream("Check for buffer overflow")
        badstream2 = win32com.server.util.wrap(badstream, pythoncom.IID_IStream)
        self.assertRaises(pythoncom.com_error, badstream2.Read, 10)
        win32com.test.util.restore_test_logger(old_log)
        # there's 1 error here
        self.assertEqual(len(records), 1)
        self.assertTrue(records[0].msg.startswith("pythoncom error"))


if __name__ == "__main__":
    unittest.main()
