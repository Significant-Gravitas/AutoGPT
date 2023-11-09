import os
import tempfile
import unittest

import win32con
import win32rcparser


class TestParser(unittest.TestCase):
    def setUp(self):
        rc_file = os.path.join(os.path.dirname(__file__), "win32rcparser", "test.rc")
        self.resources = win32rcparser.Parse(rc_file)

    def testStrings(self):
        for sid, expected in (
            ("IDS_TEST_STRING4", "Test 'single quoted' string"),
            ("IDS_TEST_STRING1", 'Test "quoted" string'),
            ("IDS_TEST_STRING3", 'String with single " quote'),
            ("IDS_TEST_STRING2", "Test string"),
        ):
            got = self.resources.stringTable[sid].value
            self.assertEqual(got, expected)

    def testStandardIds(self):
        for idc in "IDOK IDCANCEL".split():
            correct = getattr(win32con, idc)
            self.assertEqual(self.resources.names[correct], idc)
            self.assertEqual(self.resources.ids[idc], correct)

    def testTabStop(self):
        d = self.resources.dialogs["IDD_TEST_DIALOG2"]
        tabstop_names = ["IDC_EDIT1", "IDOK"]  # should have WS_TABSTOP
        tabstop_ids = [self.resources.ids[name] for name in tabstop_names]
        notabstop_names = ["IDC_EDIT2"]  # should have WS_TABSTOP
        notabstop_ids = [self.resources.ids[name] for name in notabstop_names]
        num_ok = 0
        for cdef in d[1:]:  # skip dlgdef
            # print cdef
            cid = cdef[2]
            style = cdef[-2]
            styleex = cdef[-1]
            if cid in tabstop_ids:
                self.assertEqual(style & win32con.WS_TABSTOP, win32con.WS_TABSTOP)
                num_ok += 1
            elif cid in notabstop_ids:
                self.assertEqual(style & win32con.WS_TABSTOP, 0)
                num_ok += 1
        self.assertEqual(num_ok, len(tabstop_ids) + len(notabstop_ids))


class TestGenerated(TestParser):
    def setUp(self):
        # don't call base!
        rc_file = os.path.join(os.path.dirname(__file__), "win32rcparser", "test.rc")
        py_file = tempfile.mktemp("test_win32rcparser.py")
        try:
            win32rcparser.GenerateFrozenResource(rc_file, py_file)
            py_source = open(py_file).read()
        finally:
            if os.path.isfile(py_file):
                os.unlink(py_file)

        # poor-man's import :)
        globs = {}
        exec(py_source, globs, globs)
        self.resources = globs["FakeParser"]()


if __name__ == "__main__":
    unittest.main()
