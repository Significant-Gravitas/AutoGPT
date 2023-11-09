# tests for win32gui
import array
import operator
import unittest

import pywin32_testutil
import win32gui


class TestPyGetString(unittest.TestCase):
    def test_get_string(self):
        # test invalid addresses cause a ValueError rather than crash!
        self.assertRaises(ValueError, win32gui.PyGetString, 0)
        self.assertRaises(ValueError, win32gui.PyGetString, 1)
        self.assertRaises(ValueError, win32gui.PyGetString, 1, 1)


class TestPyGetMemory(unittest.TestCase):
    def test_ob(self):
        # Check the PyGetMemory result and a bytes string can be compared
        test_data = b"\0\1\2\3\4\5\6"
        c = array.array("b", test_data)
        addr, buflen = c.buffer_info()
        got = win32gui.PyGetMemory(addr, buflen)
        self.assertEqual(len(got), len(test_data))
        self.assertEqual(bytes(got), test_data)

    def test_memory_index(self):
        # Check we can index into the buffer object returned by PyGetMemory
        test_data = b"\0\1\2\3\4\5\6"
        c = array.array("b", test_data)
        addr, buflen = c.buffer_info()
        got = win32gui.PyGetMemory(addr, buflen)
        self.assertEqual(got[0], 0)

    def test_memory_slice(self):
        # Check we can slice the buffer object returned by PyGetMemory
        test_data = b"\0\1\2\3\4\5\6"
        c = array.array("b", test_data)
        addr, buflen = c.buffer_info()
        got = win32gui.PyGetMemory(addr, buflen)
        self.assertEqual(list(got[0:3]), [0, 1, 2])

    def test_real_view(self):
        # Do the PyGetMemory, then change the original memory, then ensure
        # the initial object we fetched sees the new value.
        test_data = b"\0\1\2\3\4\5\6"
        c = array.array("b", test_data)
        addr, buflen = c.buffer_info()
        got = win32gui.PyGetMemory(addr, buflen)
        self.assertEqual(got[0], 0)
        c[0] = 1
        self.assertEqual(got[0], 1)

    def test_memory_not_writable(self):
        # Check the buffer object fetched by PyGetMemory isn't writable.
        test_data = b"\0\1\2\3\4\5\6"
        c = array.array("b", test_data)
        addr, buflen = c.buffer_info()
        got = win32gui.PyGetMemory(addr, buflen)
        self.assertRaises(TypeError, operator.setitem, got, 0, 1)


if __name__ == "__main__":
    unittest.main()
