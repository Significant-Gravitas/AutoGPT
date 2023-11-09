# Test module for win32timezone

import doctest
import sys
import unittest

import win32timezone


class Win32TimeZoneTest(unittest.TestCase):
    def testWin32TZ(self):
        # On 3.7 and later, the repr() for datetime objects changed to use kwargs - eg,
        # eg, `datetime.timedelta(0, 10800)` is now `datetime.timedelta(seconds=10800)`.
        # So we just skip the tests on 3.5 and 3.6
        if sys.version_info < (3, 7):
            from pywin32_testutil import TestSkipped

            raise TestSkipped(
                "The repr() for datetime objects makes this test fail in 3.5 and 3.6"
            )

        failed, total = doctest.testmod(win32timezone, verbose=False)
        self.assertFalse(failed)


if __name__ == "__main__":
    unittest.main()
