import unittest
from datetime import datetime

import pywintypes
import win32com.client
import win32com.server.util
import win32com.test.util
from win32timezone import TimeZoneInfo


# A COM object so we can pass dates to and from the COM boundary.
class Tester:
    _public_methods_ = ["TestDate"]

    def TestDate(self, d):
        assert isinstance(d, datetime)
        return d


def test_ob():
    return win32com.client.Dispatch(win32com.server.util.wrap(Tester()))


class TestCase(win32com.test.util.TestCase):
    def check(self, d, expected=None):
        if not issubclass(pywintypes.TimeType, datetime):
            self.skipTest("this is testing pywintypes and datetime")
        got = test_ob().TestDate(d)
        self.assertEqual(got, expected or d)

    def testUTC(self):
        self.check(
            datetime(
                year=2000,
                month=12,
                day=25,
                microsecond=500000,
                tzinfo=TimeZoneInfo.utc(),
            )
        )

    def testLocal(self):
        self.check(
            datetime(
                year=2000,
                month=12,
                day=25,
                microsecond=500000,
                tzinfo=TimeZoneInfo.local(),
            )
        )

    def testMSTruncated(self):
        # milliseconds are kept but microseconds are lost after rounding.
        self.check(
            datetime(
                year=2000,
                month=12,
                day=25,
                microsecond=500500,
                tzinfo=TimeZoneInfo.utc(),
            ),
            datetime(
                year=2000,
                month=12,
                day=25,
                microsecond=500000,
                tzinfo=TimeZoneInfo.utc(),
            ),
        )


if __name__ == "__main__":
    unittest.main()
