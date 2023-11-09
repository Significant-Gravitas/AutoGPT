# testDictionary.py
#
import sys
import unittest

import pythoncom
import pywintypes
import win32com.client
import win32com.server.util
import win32com.test.util
import win32timezone
import winerror


def MakeTestDictionary():
    return win32com.client.Dispatch("Python.Dictionary")


def TestDictAgainst(dict, check):
    for key, value in list(check.items()):
        if dict(key) != value:
            raise Exception(
                "Indexing for '%s' gave the incorrect value - %s/%s"
                % (repr(key), repr(dict[key]), repr(check[key]))
            )


# Ensure we have the correct version registered.
def Register(quiet):
    import win32com.servers.dictionary
    from win32com.test.util import RegisterPythonServer

    RegisterPythonServer(win32com.servers.dictionary.__file__, "Python.Dictionary")


def TestDict(quiet=None):
    if quiet is None:
        quiet = not "-v" in sys.argv
    Register(quiet)

    if not quiet:
        print("Simple enum test")
    dict = MakeTestDictionary()
    checkDict = {}
    TestDictAgainst(dict, checkDict)

    dict["NewKey"] = "NewValue"
    checkDict["NewKey"] = "NewValue"
    TestDictAgainst(dict, checkDict)

    dict["NewKey"] = None
    del checkDict["NewKey"]
    TestDictAgainst(dict, checkDict)

    now = win32timezone.now()
    # We want to keep the milliseconds but discard microseconds as they
    # don't survive the conversion.
    now = now.replace(microsecond=round(now.microsecond / 1000) * 1000)
    dict["Now"] = now
    checkDict["Now"] = now
    TestDictAgainst(dict, checkDict)

    if not quiet:
        print("Failure tests")
    try:
        dict()
        raise Exception("default method with no args worked when it shouldnt have!")
    except pythoncom.com_error as xxx_todo_changeme:
        (hr, desc, exc, argErr) = xxx_todo_changeme.args
        if hr != winerror.DISP_E_BADPARAMCOUNT:
            raise Exception("Expected DISP_E_BADPARAMCOUNT - got %d (%s)" % (hr, desc))

    try:
        dict("hi", "there")
        raise Exception("multiple args worked when it shouldnt have!")
    except pythoncom.com_error as xxx_todo_changeme1:
        (hr, desc, exc, argErr) = xxx_todo_changeme1.args
        if hr != winerror.DISP_E_BADPARAMCOUNT:
            raise Exception("Expected DISP_E_BADPARAMCOUNT - got %d (%s)" % (hr, desc))

    try:
        dict(0)
        raise Exception("int key worked when it shouldnt have!")
    except pythoncom.com_error as xxx_todo_changeme2:
        (hr, desc, exc, argErr) = xxx_todo_changeme2.args
        if hr != winerror.DISP_E_TYPEMISMATCH:
            raise Exception("Expected DISP_E_TYPEMISMATCH - got %d (%s)" % (hr, desc))

    if not quiet:
        print("Python.Dictionary tests complete.")


class TestCase(win32com.test.util.TestCase):
    def testDict(self):
        TestDict()


if __name__ == "__main__":
    unittest.main()
