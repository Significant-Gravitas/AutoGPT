import datetime
import os
import struct
import sys

import win32timezone

try:
    sys_maxsize = sys.maxsize  # 2.6 and later - maxsize != maxint on 64bits
except AttributeError:
    sys_maxsize = sys.maxint

import pythoncom
import pywintypes
import win32com.test.util
import win32con
from pywin32_testutil import str2bytes
from win32com.shell import shell
from win32com.shell.shellcon import *
from win32com.storagecon import *


class ShellTester(win32com.test.util.TestCase):
    def testShellLink(self):
        desktop = str(shell.SHGetSpecialFolderPath(0, CSIDL_DESKTOP))
        num = 0
        shellLink = pythoncom.CoCreateInstance(
            shell.CLSID_ShellLink,
            None,
            pythoncom.CLSCTX_INPROC_SERVER,
            shell.IID_IShellLink,
        )
        persistFile = shellLink.QueryInterface(pythoncom.IID_IPersistFile)
        names = [os.path.join(desktop, n) for n in os.listdir(desktop)]
        programs = str(shell.SHGetSpecialFolderPath(0, CSIDL_PROGRAMS))
        names.extend([os.path.join(programs, n) for n in os.listdir(programs)])
        for name in names:
            try:
                persistFile.Load(name, STGM_READ)
            except pythoncom.com_error:
                continue
            # Resolve is slow - avoid it for our tests.
            # shellLink.Resolve(0, shell.SLR_ANY_MATCH | shell.SLR_NO_UI)
            fname, findData = shellLink.GetPath(0)
            unc = shellLink.GetPath(shell.SLGP_UNCPRIORITY)[0]
            num += 1
        if num == 0:
            # This isn't a fatal error, but is unlikely.
            print(
                "Could not find any links on your desktop or programs dir, which is unusual"
            )

    def testShellFolder(self):
        sf = shell.SHGetDesktopFolder()
        names_1 = []
        for i in sf:  # Magically calls EnumObjects
            name = sf.GetDisplayNameOf(i, SHGDN_NORMAL)
            names_1.append(name)

        # And get the enumerator manually
        enum = sf.EnumObjects(
            0, SHCONTF_FOLDERS | SHCONTF_NONFOLDERS | SHCONTF_INCLUDEHIDDEN
        )
        names_2 = []
        for i in enum:
            name = sf.GetDisplayNameOf(i, SHGDN_NORMAL)
            names_2.append(name)
        names_1.sort()
        names_2.sort()
        self.assertEqual(names_1, names_2)


class PIDLTester(win32com.test.util.TestCase):
    def _rtPIDL(self, pidl):
        pidl_str = shell.PIDLAsString(pidl)
        pidl_rt = shell.StringAsPIDL(pidl_str)
        self.assertEqual(pidl_rt, pidl)
        pidl_str_rt = shell.PIDLAsString(pidl_rt)
        self.assertEqual(pidl_str_rt, pidl_str)

    def _rtCIDA(self, parent, kids):
        cida = parent, kids
        cida_str = shell.CIDAAsString(cida)
        cida_rt = shell.StringAsCIDA(cida_str)
        self.assertEqual(cida, cida_rt)
        cida_str_rt = shell.CIDAAsString(cida_rt)
        self.assertEqual(cida_str_rt, cida_str)

    def testPIDL(self):
        # A PIDL of "\1" is:   cb    pidl   cb
        expect = str2bytes("\03\00" "\1" "\0\0")
        self.assertEqual(shell.PIDLAsString([str2bytes("\1")]), expect)
        self._rtPIDL([str2bytes("\0")])
        self._rtPIDL([str2bytes("\1"), str2bytes("\2"), str2bytes("\3")])
        self._rtPIDL([str2bytes("\0") * 2048] * 2048)
        # PIDL must be a list
        self.assertRaises(TypeError, shell.PIDLAsString, "foo")

    def testCIDA(self):
        self._rtCIDA([str2bytes("\0")], [[str2bytes("\0")]])
        self._rtCIDA([str2bytes("\1")], [[str2bytes("\2")]])
        self._rtCIDA(
            [str2bytes("\0")], [[str2bytes("\0")], [str2bytes("\1")], [str2bytes("\2")]]
        )

    def testBadShortPIDL(self):
        # A too-short child element:   cb    pidl   cb
        pidl = str2bytes("\01\00" "\1")
        self.assertRaises(ValueError, shell.StringAsPIDL, pidl)

        # ack - tried to test too long PIDLs, but a len of 0xFFFF may not
        # always fail.


class FILEGROUPDESCRIPTORTester(win32com.test.util.TestCase):
    def _getTestTimes(self):
        if issubclass(pywintypes.TimeType, datetime.datetime):
            ctime = win32timezone.now()
            # FILETIME only has ms precision...
            ctime = ctime.replace(microsecond=ctime.microsecond // 1000 * 1000)
            atime = ctime + datetime.timedelta(seconds=1)
            wtime = atime + datetime.timedelta(seconds=1)
        else:
            ctime = pywintypes.Time(11)
            atime = pywintypes.Time(12)
            wtime = pywintypes.Time(13)
        return ctime, atime, wtime

    def _testRT(self, fd):
        fgd_string = shell.FILEGROUPDESCRIPTORAsString([fd])
        fd2 = shell.StringAsFILEGROUPDESCRIPTOR(fgd_string)[0]

        fd = fd.copy()
        fd2 = fd2.copy()

        # The returned objects *always* have dwFlags and cFileName.
        if "dwFlags" not in fd:
            del fd2["dwFlags"]
        if "cFileName" not in fd:
            self.assertEqual(fd2["cFileName"], "")
            del fd2["cFileName"]

        self.assertEqual(fd, fd2)

    def _testSimple(self, make_unicode):
        fgd = shell.FILEGROUPDESCRIPTORAsString([], make_unicode)
        header = struct.pack("i", 0)
        self.assertEqual(header, fgd[: len(header)])
        self._testRT(dict())
        d = dict()
        fgd = shell.FILEGROUPDESCRIPTORAsString([d], make_unicode)
        header = struct.pack("i", 1)
        self.assertEqual(header, fgd[: len(header)])
        self._testRT(d)

    def testSimpleBytes(self):
        self._testSimple(False)

    def testSimpleUnicode(self):
        self._testSimple(True)

    def testComplex(self):
        clsid = pythoncom.MakeIID("{CD637886-DB8B-4b04-98B5-25731E1495BE}")
        ctime, atime, wtime = self._getTestTimes()
        d = dict(
            cFileName="foo.txt",
            clsid=clsid,
            sizel=(1, 2),
            pointl=(3, 4),
            dwFileAttributes=win32con.FILE_ATTRIBUTE_NORMAL,
            ftCreationTime=ctime,
            ftLastAccessTime=atime,
            ftLastWriteTime=wtime,
            nFileSize=sys_maxsize + 1,
        )
        self._testRT(d)

    def testUnicode(self):
        # exercise a bug fixed in build 210 - multiple unicode objects failed.
        ctime, atime, wtime = self._getTestTimes()
        d = [
            dict(
                cFileName="foo.txt",
                sizel=(1, 2),
                pointl=(3, 4),
                dwFileAttributes=win32con.FILE_ATTRIBUTE_NORMAL,
                ftCreationTime=ctime,
                ftLastAccessTime=atime,
                ftLastWriteTime=wtime,
                nFileSize=sys_maxsize + 1,
            ),
            dict(
                cFileName="foo2.txt",
                sizel=(1, 2),
                pointl=(3, 4),
                dwFileAttributes=win32con.FILE_ATTRIBUTE_NORMAL,
                ftCreationTime=ctime,
                ftLastAccessTime=atime,
                ftLastWriteTime=wtime,
                nFileSize=sys_maxsize + 1,
            ),
            dict(
                cFileName="foo\xa9.txt",
                sizel=(1, 2),
                pointl=(3, 4),
                dwFileAttributes=win32con.FILE_ATTRIBUTE_NORMAL,
                ftCreationTime=ctime,
                ftLastAccessTime=atime,
                ftLastWriteTime=wtime,
                nFileSize=sys_maxsize + 1,
            ),
        ]
        s = shell.FILEGROUPDESCRIPTORAsString(d, 1)
        d2 = shell.StringAsFILEGROUPDESCRIPTOR(s)
        # clobber 'dwFlags' - they are not expected to be identical
        for t in d2:
            del t["dwFlags"]
        self.assertEqual(d, d2)


class FileOperationTester(win32com.test.util.TestCase):
    def setUp(self):
        import tempfile

        self.src_name = os.path.join(tempfile.gettempdir(), "pywin32_testshell")
        self.dest_name = os.path.join(tempfile.gettempdir(), "pywin32_testshell_dest")
        self.test_data = str2bytes("Hello from\0Python")
        f = open(self.src_name, "wb")
        f.write(self.test_data)
        f.close()
        try:
            os.unlink(self.dest_name)
        except os.error:
            pass

    def tearDown(self):
        for fname in (self.src_name, self.dest_name):
            if os.path.isfile(fname):
                os.unlink(fname)

    def testCopy(self):
        s = (0, FO_COPY, self.src_name, self.dest_name)  # hwnd,  # operation

        rc, aborted = shell.SHFileOperation(s)
        self.assertTrue(not aborted)
        self.assertEqual(0, rc)
        self.assertTrue(os.path.isfile(self.src_name))
        self.assertTrue(os.path.isfile(self.dest_name))

    def testRename(self):
        s = (0, FO_RENAME, self.src_name, self.dest_name)  # hwnd,  # operation
        rc, aborted = shell.SHFileOperation(s)
        self.assertTrue(not aborted)
        self.assertEqual(0, rc)
        self.assertTrue(os.path.isfile(self.dest_name))
        self.assertTrue(not os.path.isfile(self.src_name))

    def testMove(self):
        s = (0, FO_MOVE, self.src_name, self.dest_name)  # hwnd,  # operation
        rc, aborted = shell.SHFileOperation(s)
        self.assertTrue(not aborted)
        self.assertEqual(0, rc)
        self.assertTrue(os.path.isfile(self.dest_name))
        self.assertTrue(not os.path.isfile(self.src_name))

    def testDelete(self):
        s = (
            0,  # hwnd,
            FO_DELETE,  # operation
            self.src_name,
            None,
            FOF_NOCONFIRMATION,
        )
        rc, aborted = shell.SHFileOperation(s)
        self.assertTrue(not aborted)
        self.assertEqual(0, rc)
        self.assertTrue(not os.path.isfile(self.src_name))


if __name__ == "__main__":
    win32com.test.util.testmain()
