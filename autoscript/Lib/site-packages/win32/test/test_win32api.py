# General test module for win32api - please add some :)

import datetime
import os
import sys
import tempfile
import unittest

import win32api
import win32con
import win32event
import winerror
from pywin32_testutil import TestSkipped, str2bytes


class CurrentUserTestCase(unittest.TestCase):
    def testGetCurrentUser(self):
        domain = win32api.GetDomainName()
        if domain == "NT AUTHORITY":
            # Running as a service account, so the comparison will fail
            raise TestSkipped("running as service account")
        name = "%s\\%s" % (domain, win32api.GetUserName())
        self.assertEqual(name, win32api.GetUserNameEx(win32api.NameSamCompatible))


class TestTime(unittest.TestCase):
    def testTimezone(self):
        # GetTimeZoneInformation
        rc, tzinfo = win32api.GetTimeZoneInformation()
        if rc == win32con.TIME_ZONE_ID_DAYLIGHT:
            tz_str = tzinfo[4]
            tz_time = tzinfo[5]
        else:
            tz_str = tzinfo[1]
            tz_time = tzinfo[2]
        # for the sake of code exercise but don't output
        tz_str.encode()
        if not isinstance(tz_time, datetime.datetime) and not isinstance(
            tz_time, tuple
        ):
            tz_time.Format()

    def TestDateFormat(self):
        DATE_LONGDATE = 2
        date_flags = DATE_LONGDATE
        win32api.GetDateFormat(0, date_flags, None)
        win32api.GetDateFormat(0, date_flags, 0)
        win32api.GetDateFormat(0, date_flags, datetime.datetime.now())
        win32api.GetDateFormat(0, date_flags, time.time())

    def TestTimeFormat(self):
        win32api.GetTimeFormat(0, 0, None)
        win32api.GetTimeFormat(0, 0, 0)
        win32api.GetTimeFormat(0, 0, datetime.datetime.now())
        win32api.GetTimeFormat(0, 0, time.time())


class Registry(unittest.TestCase):
    key_name = r"PythonTestHarness\Whatever"

    def test1(self):
        # This used to leave a stale exception behind.
        def reg_operation():
            hkey = win32api.RegCreateKey(win32con.HKEY_CURRENT_USER, self.key_name)
            x = 3 / 0  # or a statement like: raise 'error'

        # do the test
        try:
            try:
                try:
                    reg_operation()
                except:
                    1 / 0  # Force exception
            finally:
                win32api.RegDeleteKey(win32con.HKEY_CURRENT_USER, self.key_name)
        except ZeroDivisionError:
            pass

    def testValues(self):
        key_name = r"PythonTestHarness\win32api"
        ## tuples containing value name, value type, data
        values = (
            (None, win32con.REG_SZ, "This is default unnamed value"),
            ("REG_SZ", win32con.REG_SZ, "REG_SZ text data"),
            ("REG_EXPAND_SZ", win32con.REG_EXPAND_SZ, "%systemdir%"),
            ## REG_MULTI_SZ value needs to be a list since strings are returned as a list
            (
                "REG_MULTI_SZ",
                win32con.REG_MULTI_SZ,
                ["string 1", "string 2", "string 3", "string 4"],
            ),
            ("REG_MULTI_SZ_empty", win32con.REG_MULTI_SZ, []),
            ("REG_DWORD", win32con.REG_DWORD, 666),
            ("REG_QWORD_INT", win32con.REG_QWORD, 99),
            ("REG_QWORD", win32con.REG_QWORD, 2**33),
            (
                "REG_BINARY",
                win32con.REG_BINARY,
                str2bytes("\x00\x01\x02\x03\x04\x05\x06\x07\x08\x01\x00"),
            ),
        )

        hkey = win32api.RegCreateKey(win32con.HKEY_CURRENT_USER, key_name)
        for value_name, reg_type, data in values:
            win32api.RegSetValueEx(hkey, value_name, None, reg_type, data)

        for value_name, orig_type, orig_data in values:
            data, typ = win32api.RegQueryValueEx(hkey, value_name)
            self.assertEqual(typ, orig_type)
            self.assertEqual(data, orig_data)

    def testNotifyChange(self):
        def change():
            hkey = win32api.RegCreateKey(win32con.HKEY_CURRENT_USER, self.key_name)
            try:
                win32api.RegSetValue(hkey, None, win32con.REG_SZ, "foo")
            finally:
                win32api.RegDeleteKey(win32con.HKEY_CURRENT_USER, self.key_name)

        evt = win32event.CreateEvent(None, 0, 0, None)
        ## REG_NOTIFY_CHANGE_LAST_SET - values
        ## REG_CHANGE_NOTIFY_NAME - keys
        ## REG_NOTIFY_CHANGE_SECURITY - security descriptor
        ## REG_NOTIFY_CHANGE_ATTRIBUTES
        win32api.RegNotifyChangeKeyValue(
            win32con.HKEY_CURRENT_USER,
            1,
            win32api.REG_NOTIFY_CHANGE_LAST_SET,
            evt,
            True,
        )
        ret_code = win32event.WaitForSingleObject(evt, 0)
        # Should be no change.
        self.assertTrue(ret_code == win32con.WAIT_TIMEOUT)
        change()
        # Our event should now be in a signalled state.
        ret_code = win32event.WaitForSingleObject(evt, 0)
        self.assertTrue(ret_code == win32con.WAIT_OBJECT_0)


class FileNames(unittest.TestCase):
    def testShortLongPathNames(self):
        try:
            me = __file__
        except NameError:
            me = sys.argv[0]
        fname = os.path.abspath(me).lower()
        short_name = win32api.GetShortPathName(fname).lower()
        long_name = win32api.GetLongPathName(short_name).lower()
        self.assertTrue(
            long_name == fname,
            "Expected long name ('%s') to be original name ('%s')" % (long_name, fname),
        )
        self.assertEqual(long_name, win32api.GetLongPathNameW(short_name).lower())
        long_name = win32api.GetLongPathNameW(short_name).lower()
        self.assertTrue(
            type(long_name) == str,
            "GetLongPathNameW returned type '%s'" % (type(long_name),),
        )
        self.assertTrue(
            long_name == fname,
            "Expected long name ('%s') to be original name ('%s')" % (long_name, fname),
        )

    def testShortUnicodeNames(self):
        try:
            me = __file__
        except NameError:
            me = sys.argv[0]
        fname = os.path.abspath(me).lower()
        # passing unicode should cause GetShortPathNameW to be called.
        short_name = win32api.GetShortPathName(str(fname)).lower()
        self.assertTrue(isinstance(short_name, str))
        long_name = win32api.GetLongPathName(short_name).lower()
        self.assertTrue(
            long_name == fname,
            "Expected long name ('%s') to be original name ('%s')" % (long_name, fname),
        )
        self.assertEqual(long_name, win32api.GetLongPathNameW(short_name).lower())
        long_name = win32api.GetLongPathNameW(short_name).lower()
        self.assertTrue(
            type(long_name) == str,
            "GetLongPathNameW returned type '%s'" % (type(long_name),),
        )
        self.assertTrue(
            long_name == fname,
            "Expected long name ('%s') to be original name ('%s')" % (long_name, fname),
        )

    def testLongLongPathNames(self):
        # We need filename where the FQN is > 256 - simplest way is to create a
        # 250 character directory in the cwd (except - cwd may be on a drive
        # not supporting \\\\?\\ (eg, network share) - so use temp.
        import win32file

        basename = "a" * 250
        # but we need to ensure we use the 'long' version of the
        # temp dir for later comparison.
        long_temp_dir = win32api.GetLongPathNameW(tempfile.gettempdir())
        fname = "\\\\?\\" + os.path.join(long_temp_dir, basename)
        try:
            win32file.CreateDirectoryW(fname, None)
        except win32api.error as details:
            if details.winerror != winerror.ERROR_ALREADY_EXISTS:
                raise
        try:
            # GetFileAttributes automatically calls GetFileAttributesW when
            # passed unicode
            try:
                attr = win32api.GetFileAttributes(fname)
            except win32api.error as details:
                if details.winerror != winerror.ERROR_FILENAME_EXCED_RANGE:
                    raise

            attr = win32api.GetFileAttributes(str(fname))
            self.assertTrue(attr & win32con.FILE_ATTRIBUTE_DIRECTORY, attr)

            long_name = win32api.GetLongPathNameW(fname)
            self.assertEqual(long_name.lower(), fname.lower())
        finally:
            win32file.RemoveDirectory(fname)


class FormatMessage(unittest.TestCase):
    def test_FromString(self):
        msg = "Hello %1, how are you %2?"
        inserts = ["Mark", "today"]
        result = win32api.FormatMessage(
            win32con.FORMAT_MESSAGE_FROM_STRING,
            msg,  # source
            0,  # ID
            0,  # LangID
            inserts,
        )
        self.assertEqual(result, "Hello Mark, how are you today?")


class Misc(unittest.TestCase):
    def test_last_error(self):
        for x in (0, 1, -1, winerror.TRUST_E_PROVIDER_UNKNOWN):
            win32api.SetLastError(x)
            self.assertEqual(x, win32api.GetLastError())

    def testVkKeyScan(self):
        # hopefully ' ' doesn't depend on the locale!
        self.assertEqual(win32api.VkKeyScan(" "), 32)

    def testVkKeyScanEx(self):
        # hopefully ' ' doesn't depend on the locale!
        self.assertEqual(win32api.VkKeyScanEx(" ", 0), 32)

    def testGetSystemPowerStatus(self):
        # Dummy
        sps = win32api.GetSystemPowerStatus()
        self.assertIsInstance(sps, dict)
        test_keys = (
            "ACLineStatus",
            "BatteryFlag",
            "BatteryLifePercent",
            "SystemStatusFlag",
            "BatteryLifeTime",
            "BatteryFullLifeTime",
        )
        self.assertEqual(set(test_keys), set(sps.keys()))


if __name__ == "__main__":
    unittest.main()
