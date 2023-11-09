import gc
import logging
import os
import sys
import tempfile
import unittest
import winreg

import pythoncom
import pywin32_testutil
import pywintypes
import win32api
import win32com
import winerror
from pythoncom import _GetGatewayCount, _GetInterfaceCount
from pywin32_testutil import LeakTestCase, TestLoader, TestResult, TestRunner


def CheckClean():
    # Ensure no lingering exceptions - Python should have zero outstanding
    # COM objects
    try:
        sys.exc_clear()
    except AttributeError:
        pass  # py3k
    c = _GetInterfaceCount()
    if c:
        print("Warning - %d com interface objects still alive" % c)
    c = _GetGatewayCount()
    if c:
        print("Warning - %d com gateway objects still alive" % c)


def RegisterPythonServer(filename, progids=None, verbose=0):
    if progids:
        if isinstance(progids, str):
            progids = [progids]
        # we know the CLSIDs we need, but we might not be an admin user
        # and otherwise unable to register them.  So as long as the progids
        # exist and the DLL points at our version, assume it already is.
        why_not = None
        for progid in progids:
            try:
                clsid = pywintypes.IID(progid)
            except pythoncom.com_error:
                # not registered.
                break
            try:
                HKCR = winreg.HKEY_CLASSES_ROOT
                hk = winreg.OpenKey(HKCR, "CLSID\\%s" % clsid)
                dll = winreg.QueryValue(hk, "InprocServer32")
            except WindowsError:
                # no CLSID or InProcServer32 - not registered
                break
            ok_files = [
                os.path.basename(pythoncom.__file__),
                "pythoncomloader%d%d.dll" % (sys.version_info[0], sys.version_info[1]),
            ]
            if os.path.basename(dll) not in ok_files:
                why_not = "%r is registered against a different Python version (%s)" % (
                    progid,
                    dll,
                )
                break
        else:
            # print "Skipping registration of '%s' - already registered" % filename
            return
    # needs registration - see if its likely!
    try:
        from win32com.shell.shell import IsUserAnAdmin
    except ImportError:
        print("Can't import win32com.shell - no idea if you are an admin or not?")
        is_admin = False
    else:
        try:
            is_admin = IsUserAnAdmin()
        except pythoncom.com_error:
            # old, less-secure OS - assume *is* admin.
            is_admin = True
    if not is_admin:
        msg = (
            "%r isn't registered, but I'm not an administrator who can register it."
            % progids[0]
        )
        if why_not:
            msg += "\n(registration check failed as %s)" % why_not
        # throw a normal "class not registered" exception - we don't report
        # them the same way as "real" errors.
        raise pythoncom.com_error(winerror.CO_E_CLASSSTRING, msg, None, -1)
    # so theoretically we are able to register it.
    cmd = '%s "%s" --unattended > nul 2>&1' % (win32api.GetModuleFileName(0), filename)
    if verbose:
        print("Registering engine", filename)
    #       print cmd
    rc = os.system(cmd)
    if rc:
        print("Registration command was:")
        print(cmd)
        raise RuntimeError("Registration of engine '%s' failed" % filename)


def ExecuteShellCommand(
    cmd,
    testcase,
    expected_output=None,  # Set to '' to check for nothing
    tracebacks_ok=0,  # OK if the output contains a t/b?
):
    output_name = tempfile.mktemp("win32com_test")
    cmd = cmd + ' > "%s" 2>&1' % output_name
    rc = os.system(cmd)
    output = open(output_name, "r").read().strip()
    os.remove(output_name)

    class Failed(Exception):
        pass

    try:
        if rc:
            raise Failed("exit code was " + str(rc))
        if expected_output is not None and output != expected_output:
            raise Failed("Expected output %r (got %r)" % (expected_output, output))
        if not tracebacks_ok and output.find("Traceback (most recent call last)") >= 0:
            raise Failed("traceback in program output")
        return output
    except Failed as why:
        print("Failed to exec command '%r'" % cmd)
        print("Failed as", why)
        print("** start of program output **")
        print(output)
        print("** end of program output **")
        testcase.fail("Executing '%s' failed as %s" % (cmd, why))


def assertRaisesCOM_HRESULT(testcase, hresult, func, *args, **kw):
    try:
        func(*args, **kw)
    except pythoncom.com_error as details:
        if details.hresult == hresult:
            return
    testcase.fail("Excepected COM exception with HRESULT 0x%x" % hresult)


class CaptureWriter:
    def __init__(self):
        self.old_err = self.old_out = None
        self.clear()

    def capture(self):
        self.clear()
        self.old_out = sys.stdout
        self.old_err = sys.stderr
        sys.stdout = sys.stderr = self

    def release(self):
        if self.old_out:
            sys.stdout = self.old_out
            self.old_out = None
        if self.old_err:
            sys.stderr = self.old_err
            self.old_err = None

    def clear(self):
        self.captured = []

    def write(self, msg):
        self.captured.append(msg)

    def get_captured(self):
        return "".join(self.captured)

    def get_num_lines_captured(self):
        return len("".join(self.captured).split("\n"))


# Utilities to set the win32com logger to something what just captures
# records written and doesn't print them.
class LogHandler(logging.Handler):
    def __init__(self):
        self.emitted = []
        logging.Handler.__init__(self)

    def emit(self, record):
        self.emitted.append(record)


_win32com_logger = None


def setup_test_logger():
    old_log = getattr(win32com, "logger", None)
    global _win32com_logger
    if _win32com_logger is None:
        _win32com_logger = logging.Logger("test")
        handler = LogHandler()
        _win32com_logger.addHandler(handler)

    win32com.logger = _win32com_logger
    handler = _win32com_logger.handlers[0]
    handler.emitted = []
    return handler.emitted, old_log


def restore_test_logger(prev_logger):
    assert prev_logger is None, "who needs this?"
    if prev_logger is None:
        del win32com.logger
    else:
        win32com.logger = prev_logger


# We used to override some of this (and may later!)
TestCase = unittest.TestCase


def CapturingFunctionTestCase(*args, **kw):
    real_test = _CapturingFunctionTestCase(*args, **kw)
    return LeakTestCase(real_test)


class _CapturingFunctionTestCase(unittest.FunctionTestCase):  # , TestCaseMixin):
    def __call__(self, result=None):
        if result is None:
            result = self.defaultTestResult()
        writer = CaptureWriter()
        # self._preTest()
        writer.capture()
        try:
            unittest.FunctionTestCase.__call__(self, result)
            if getattr(self, "do_leak_tests", 0) and hasattr(sys, "gettotalrefcount"):
                self.run_leak_tests(result)
        finally:
            writer.release()
            # self._postTest(result)
        output = writer.get_captured()
        self.checkOutput(output, result)
        if result.showAll:
            print(output)

    def checkOutput(self, output, result):
        if output.find("Traceback") >= 0:
            msg = "Test output contained a traceback\n---\n%s\n---" % output
            result.errors.append((self, msg))


class ShellTestCase(unittest.TestCase):
    def __init__(self, cmd, expected_output):
        self.__cmd = cmd
        self.__eo = expected_output
        unittest.TestCase.__init__(self)

    def runTest(self):
        ExecuteShellCommand(self.__cmd, self, self.__eo)

    def __str__(self):
        max = 30
        if len(self.__cmd) > max:
            cmd_repr = self.__cmd[:max] + "..."
        else:
            cmd_repr = self.__cmd
        return "exec: " + cmd_repr


def testmain(*args, **kw):
    pywin32_testutil.testmain(*args, **kw)
    CheckClean()
