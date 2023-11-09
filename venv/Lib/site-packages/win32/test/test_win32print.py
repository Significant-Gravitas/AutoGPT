# Tests (scarce) for win32print module

import unittest

import win32print as wprn


class Win32PrintTestCase(unittest.TestCase):
    def setUp(self):
        self.printer_idx = 0
        self.printer_levels_all = list(range(1, 10))
        self.local_printers = wprn.EnumPrinters(wprn.PRINTER_ENUM_LOCAL, None, 1)

    def test_printer_levels_read_dummy(self):
        if not self.local_printers:
            print("Test didn't run (no local printers)!")
            return
        ph = wprn.OpenPrinter(self.local_printers[self.printer_idx][2])
        for level in self.printer_levels_all:
            wprn.GetPrinter(ph, level)


if __name__ == "__main__":
    unittest.main()
