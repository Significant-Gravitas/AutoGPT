# Utilities for the pywin32 tests
import gc
import os
import site
import sys
import unittest

import winerror

##
## General purpose utilities for the test suite.
##


# The test suite has lots of string constants containing binary data, but
# the strings are used in various "bytes" contexts.
def str2bytes(sval):
    if sys.version_info < (3, 0) and isinstance(sval, str):
        sval = sval.decode("latin1")
    return sval.encode("latin1")


# Sometimes we want to pass a string that should explicitly be treated as
# a memory blob.
def str2memory(sval):
    if sys.version_info < (3, 0):
        return buffer(sval)
    # py3k.
    return memoryview(sval.encode("latin1"))


# Sometimes we want to pass an object that exposes its memory
def ob2memory(ob):
    if sys.version_info < (3, 0):
        return buffer(ob)
    # py3k.
    return memoryview(ob)


##
## unittest related stuff
##


# This is a specialized TestCase adaptor which wraps a real test.
class LeakTestCase(unittest.TestCase):
    """An 'adaptor' which takes another test.  In debug builds we execute the
    test once to remove one-off side-effects, then capture the total
    reference count, then execute the test a few times.  If the total
    refcount at the end is greater than we first captured, we have a leak!

    In release builds the test is executed just once, as normal.

    Generally used automatically by the test runner - you can safely
    ignore this.
    """

    def __init__(self, real_test):
        unittest.TestCase.__init__(self)
        self.real_test = real_test
        self.num_test_cases = 1
        self.num_leak_iters = 2  # seems to be enough!
        if hasattr(sys, "gettotalrefcount"):
            self.num_test_cases = self.num_test_cases + self.num_leak_iters

    def countTestCases(self):
        return self.num_test_cases

    def __call__(self, result=None):
        # For the COM suite's sake, always ensure we don't leak
        # gateways/interfaces
        from pythoncom import _GetGatewayCount, _GetInterfaceCount

        gc.collect()
        ni = _GetInterfaceCount()
        ng = _GetGatewayCount()
        self.real_test(result)
        # Failed - no point checking anything else
        if result.shouldStop or not result.wasSuccessful():
            return
        self._do_leak_tests(result)
        gc.collect()
        lost_i = _GetInterfaceCount() - ni
        lost_g = _GetGatewayCount() - ng
        if lost_i or lost_g:
            msg = "%d interface objects and %d gateway objects leaked" % (
                lost_i,
                lost_g,
            )
            exc = AssertionError(msg)
            result.addFailure(self.real_test, (exc.__class__, exc, None))

    def runTest(self):
        assert 0, "not used"

    def _do_leak_tests(self, result=None):
        try:
            gtrc = sys.gettotalrefcount
        except AttributeError:
            return  # can't do leak tests in this build
        # Assume already called once, to prime any caches etc
        gc.collect()
        trc = gtrc()
        for i in range(self.num_leak_iters):
            self.real_test(result)
            if result.shouldStop:
                break
        del i  # created after we remembered the refcount!
        # int division here means one or 2 stray references won't force
        # failure, but one per loop
        gc.collect()
        lost = (gtrc() - trc) // self.num_leak_iters
        if lost < 0:
            msg = "LeakTest: %s appeared to gain %d references!!" % (
                self.real_test,
                -lost,
            )
            result.addFailure(self.real_test, (AssertionError, msg, None))
        if lost > 0:
            msg = "LeakTest: %s lost %d references" % (self.real_test, lost)
            exc = AssertionError(msg)
            result.addFailure(self.real_test, (exc.__class__, exc, None))


class TestLoader(unittest.TestLoader):
    def loadTestsFromTestCase(self, testCaseClass):
        """Return a suite of all tests cases contained in testCaseClass"""
        leak_tests = []
        for name in self.getTestCaseNames(testCaseClass):
            real_test = testCaseClass(name)
            leak_test = self._getTestWrapper(real_test)
            leak_tests.append(leak_test)
        return self.suiteClass(leak_tests)

    def fixupTestsForLeakTests(self, test):
        if isinstance(test, unittest.TestSuite):
            test._tests = [self.fixupTestsForLeakTests(t) for t in test._tests]
            return test
        else:
            # just a normal test case.
            return self._getTestWrapper(test)

    def _getTestWrapper(self, test):
        # one or 2 tests in the COM test suite set this...
        no_leak_tests = getattr(test, "no_leak_tests", False)
        if no_leak_tests:
            print("Test says it doesn't want leak tests!")
            return test
        return LeakTestCase(test)

    def loadTestsFromModule(self, mod):
        if hasattr(mod, "suite"):
            tests = mod.suite()
        else:
            tests = unittest.TestLoader.loadTestsFromModule(self, mod)
        return self.fixupTestsForLeakTests(tests)

    def loadTestsFromName(self, name, module=None):
        test = unittest.TestLoader.loadTestsFromName(self, name, module)
        if isinstance(test, unittest.TestSuite):
            pass  # hmmm? print "Don't wrap suites yet!", test._tests
        elif isinstance(test, unittest.TestCase):
            test = self._getTestWrapper(test)
        else:
            print("XXX - what is", test)
        return test


# Lots of classes necessary to support one simple feature: we want a 3rd
# test result state - "SKIPPED" - to indicate that the test wasn't able
# to be executed for various reasons.  Inspired by bzr's tests, but it
# has other concepts, such as "Expected Failure", which we don't bother
# with.

# win32 error codes that probably mean we need to be elevated (ie, if we
# aren't elevated, we treat these error codes as 'skipped')
non_admin_error_codes = [
    winerror.ERROR_ACCESS_DENIED,
    winerror.ERROR_PRIVILEGE_NOT_HELD,
]

_is_admin = None


def check_is_admin():
    global _is_admin
    if _is_admin is None:
        import pythoncom
        from win32com.shell.shell import IsUserAnAdmin

        try:
            _is_admin = IsUserAnAdmin()
        except pythoncom.com_error as exc:
            if exc.hresult != winerror.E_NOTIMPL:
                raise
            # not impl on this platform - must be old - assume is admin
            _is_admin = True
    return _is_admin


# Find a test "fixture" (eg, binary test file) expected to be very close to
# the test being run.
# If the tests are being run from the "installed" version, then these fixtures
# probably don't exist - the test is "skipped".
# But it's fatal if we think we might be running from a pywin32 source tree.
def find_test_fixture(basename, extra_dir="."):
    # look for the test file in various places
    candidates = [
        os.path.dirname(sys.argv[0]),
        extra_dir,
        ".",
    ]
    for candidate in candidates:
        fname = os.path.join(candidate, basename)
        if os.path.isfile(fname):
            return fname
    else:
        # Can't find it - see if this is expected or not.
        # This module is typically always in the installed dir, so use argv[0]
        this_file = os.path.normcase(os.path.abspath(sys.argv[0]))
        dirs_to_check = site.getsitepackages()[:]
        if site.USER_SITE:
            dirs_to_check.append(site.USER_SITE)

        for d in dirs_to_check:
            d = os.path.normcase(d)
            if os.path.commonprefix([this_file, d]) == d:
                # looks like we are in an installed Python, so skip the text.
                raise TestSkipped(f"Can't find test fixture '{fname}'")
        # Looks like we are running from source, so this is fatal.
        raise RuntimeError(f"Can't find test fixture '{fname}'")


# If this exception is raised by a test, the test is reported as a 'skip'
class TestSkipped(Exception):
    pass


# This appears to have been "upgraded" to non-private in 3.11
try:
    TextTestResult = unittest._TextTestResult
except AttributeError:
    TextTestResult = unittest.TextTestResult


# The 'TestResult' subclass that records the failures and has the special
# handling for the TestSkipped exception.
class TestResult(TextTestResult):
    def __init__(self, *args, **kw):
        super(TestResult, self).__init__(*args, **kw)
        self.skips = {}  # count of skips for each reason.

    def addError(self, test, err):
        """Called when an error has occurred. 'err' is a tuple of values as
        returned by sys.exc_info().
        """
        # translate a couple of 'well-known' exceptions into 'skipped'
        import pywintypes

        exc_val = err[1]
        # translate ERROR_ACCESS_DENIED for non-admin users to be skipped.
        # (access denied errors for an admin user aren't expected.)
        if (
            isinstance(exc_val, pywintypes.error)
            and exc_val.winerror in non_admin_error_codes
            and not check_is_admin()
        ):
            exc_val = TestSkipped(exc_val)
        # and COM errors due to objects not being registered (the com test
        # suite will attempt to catch this and handle it itself if the user
        # is admin)
        elif isinstance(exc_val, pywintypes.com_error) and exc_val.hresult in [
            winerror.CO_E_CLASSSTRING,
            winerror.REGDB_E_CLASSNOTREG,
            winerror.TYPE_E_LIBNOTREGISTERED,
        ]:
            exc_val = TestSkipped(exc_val)
        # NotImplemented generally means the platform doesn't support the
        # functionality.
        elif isinstance(exc_val, NotImplementedError):
            exc_val = TestSkipped(NotImplementedError)

        if isinstance(exc_val, TestSkipped):
            reason = exc_val.args[0]
            # if the reason itself is another exception, get its args.
            try:
                reason = tuple(reason.args)
            except (AttributeError, TypeError):
                pass
            self.skips.setdefault(reason, 0)
            self.skips[reason] += 1
            if self.showAll:
                self.stream.writeln("SKIP (%s)" % (reason,))
            elif self.dots:
                self.stream.write("S")
                self.stream.flush()
            return
        super(TestResult, self).addError(test, err)

    def printErrors(self):
        super(TestResult, self).printErrors()
        for reason, num_skipped in self.skips.items():
            self.stream.writeln("SKIPPED: %d tests - %s" % (num_skipped, reason))


# TestRunner subclass necessary just to get our TestResult hooked up.
class TestRunner(unittest.TextTestRunner):
    def _makeResult(self):
        return TestResult(self.stream, self.descriptions, self.verbosity)


# TestProgream subclass necessary just to get our TestRunner hooked up,
# which is necessary to get our TestResult hooked up *sob*
class TestProgram(unittest.TestProgram):
    def runTests(self):
        # clobber existing runner - *sob* - it shouldn't be this hard
        self.testRunner = TestRunner(verbosity=self.verbosity)
        unittest.TestProgram.runTests(self)


# A convenient entry-point - if used, 'SKIPPED' exceptions will be supressed.
def testmain(*args, **kw):
    new_kw = kw.copy()
    if "testLoader" not in new_kw:
        new_kw["testLoader"] = TestLoader()
    program_class = new_kw.get("testProgram", TestProgram)
    program_class(*args, **new_kw)
