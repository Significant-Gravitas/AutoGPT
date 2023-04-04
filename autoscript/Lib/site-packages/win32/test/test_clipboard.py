# General test module for win32api - please add some :)
import array
import os
import sys
import unittest

import pywintypes
import win32con
import win32gui
from pywin32_testutil import str2bytes
from win32clipboard import *

custom_format_name = "PythonClipboardTestFormat"


class CrashingTestCase(unittest.TestCase):
    def test_722082(self):
        class crasher(object):
            pass

        obj = crasher()
        OpenClipboard()
        try:
            EmptyClipboard()
            # This used to crash - now correctly raises type error.
            self.assertRaises(TypeError, SetClipboardData, 0, obj)
        finally:
            CloseClipboard()


class TestBitmap(unittest.TestCase):
    def setUp(self):
        self.bmp_handle = None
        try:
            this_file = __file__
        except NameError:
            this_file = sys.argv[0]
        this_dir = os.path.dirname(this_file)
        self.bmp_name = os.path.join(
            os.path.abspath(this_dir), "..", "Demos", "images", "smiley.bmp"
        )
        self.assertTrue(os.path.isfile(self.bmp_name), self.bmp_name)
        flags = win32con.LR_DEFAULTSIZE | win32con.LR_LOADFROMFILE
        self.bmp_handle = win32gui.LoadImage(
            0, self.bmp_name, win32con.IMAGE_BITMAP, 0, 0, flags
        )
        self.assertTrue(self.bmp_handle, "Failed to get a bitmap handle")

    def tearDown(self):
        if self.bmp_handle:
            win32gui.DeleteObject(self.bmp_handle)

    def test_bitmap_roundtrip(self):
        OpenClipboard()
        try:
            SetClipboardData(win32con.CF_BITMAP, self.bmp_handle)
            got_handle = GetClipboardDataHandle(win32con.CF_BITMAP)
            self.assertEqual(got_handle, self.bmp_handle)
        finally:
            CloseClipboard()


class TestStrings(unittest.TestCase):
    def setUp(self):
        OpenClipboard()

    def tearDown(self):
        CloseClipboard()

    def test_unicode(self):
        val = "test-\a9har"
        SetClipboardData(win32con.CF_UNICODETEXT, val)
        self.assertEqual(GetClipboardData(win32con.CF_UNICODETEXT), val)

    def test_unicode_text(self):
        val = "test-val"
        SetClipboardText(val)
        # GetClipboardData doesn't to auto string conversions - so on py3k,
        # CF_TEXT returns bytes.
        expected = str2bytes(val)
        self.assertEqual(GetClipboardData(win32con.CF_TEXT), expected)
        SetClipboardText(val, win32con.CF_UNICODETEXT)
        self.assertEqual(GetClipboardData(win32con.CF_UNICODETEXT), val)

    def test_string(self):
        val = str2bytes("test")
        SetClipboardData(win32con.CF_TEXT, val)
        self.assertEqual(GetClipboardData(win32con.CF_TEXT), val)


class TestGlobalMemory(unittest.TestCase):
    def setUp(self):
        OpenClipboard()

    def tearDown(self):
        CloseClipboard()

    def test_mem(self):
        val = str2bytes("test")
        expected = str2bytes("test\0")
        SetClipboardData(win32con.CF_TEXT, val)
        # Get the raw data - this will include the '\0'
        raw_data = GetGlobalMemory(GetClipboardDataHandle(win32con.CF_TEXT))
        self.assertEqual(expected, raw_data)

    def test_bad_mem(self):
        self.assertRaises(pywintypes.error, GetGlobalMemory, 0)
        self.assertRaises(pywintypes.error, GetGlobalMemory, -1)
        if sys.getwindowsversion()[0] <= 5:
            # For some reason, the value '1' dies from a 64bit process, but
            # "works" (ie, gives the correct exception) from a 32bit process.
            # just silently skip this value on Vista.
            self.assertRaises(pywintypes.error, GetGlobalMemory, 1)

    def test_custom_mem(self):
        test_data = str2bytes("hello\x00\xff")
        test_buffer = array.array("b", test_data)
        cf = RegisterClipboardFormat(custom_format_name)
        self.assertEqual(custom_format_name, GetClipboardFormatName(cf))
        SetClipboardData(cf, test_buffer)
        hglobal = GetClipboardDataHandle(cf)
        data = GetGlobalMemory(hglobal)
        self.assertEqual(data, test_data)


if __name__ == "__main__":
    unittest.main()
