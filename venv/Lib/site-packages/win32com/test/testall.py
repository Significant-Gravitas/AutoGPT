import getopt
import os
import re
import sys
import traceback
import unittest

try:
    this_file = __file__
except NameError:
    this_file = sys.argv[0]

win32com_src_dir = os.path.abspath(os.path.join(this_file, "../.."))

import win32com

# We'd prefer the win32com namespace to be the parent of __file__ - ie, our source-tree,
# rather than the version installed - otherwise every .py change needs a full install to
# test!
# We can't patch win32comext as most of them have a .pyd in their root :(
# This clearly ins't ideal or perfect :)
win32com.__path__[0] = win32com_src_dir

import pythoncom
import win32com.client
from win32com.test.util import (
    CapturingFunctionTestCase,
    CheckClean,
    RegisterPythonServer,
    ShellTestCase,
    TestCase,
    TestLoader,
    TestRunner,
)

verbosity = 1  # default unittest verbosity.


def GenerateAndRunOldStyle():
    from . import GenTestScripts

    GenTestScripts.GenerateAll()
    try:
        pass  #
    finally:
        GenTestScripts.CleanAll()


def CleanGenerated():
    import shutil

    import win32com

    if os.path.isdir(win32com.__gen_path__):
        if verbosity > 1:
            print("Deleting files from %s" % (win32com.__gen_path__))
        shutil.rmtree(win32com.__gen_path__)
    import win32com.client.gencache

    win32com.client.gencache.__init__()  # Reset


def RemoveRefCountOutput(data):
    while 1:
        last_line_pos = data.rfind("\n")
        if not re.match("\[\d+ refs\]", data[last_line_pos + 1 :]):
            break
        if last_line_pos < 0:
            # All the output
            return ""
        data = data[:last_line_pos]

    return data


def ExecuteSilentlyIfOK(cmd, testcase):
    f = os.popen(cmd)
    data = f.read().strip()
    rc = f.close()
    if rc:
        print(data)
        testcase.fail("Executing '%s' failed (%d)" % (cmd, rc))
    # for "_d" builds, strip the '[xxx refs]' line
    return RemoveRefCountOutput(data)


class PyCOMTest(TestCase):
    no_leak_tests = True  # done by the test itself

    def testit(self):
        # Check that the item is registered, so we get the correct
        # 'skipped' behaviour (and recorded as such) rather than either
        # error or silence due to non-registration.
        RegisterPythonServer(
            os.path.join(
                os.path.dirname(__file__), "..", "servers", "test_pycomtest.py"
            ),
            "Python.Test.PyCOMTest",
        )

        # Execute testPyComTest in its own process so it can play
        # with the Python thread state
        fname = os.path.join(os.path.dirname(this_file), "testPyComTest.py")
        cmd = '%s "%s" -q 2>&1' % (sys.executable, fname)
        data = ExecuteSilentlyIfOK(cmd, self)


class PippoTest(TestCase):
    def testit(self):
        # Check we are registered before spawning the process.
        from win32com.test import pippo_server

        RegisterPythonServer(pippo_server.__file__, "Python.Test.Pippo")

        python = sys.executable
        fname = os.path.join(os.path.dirname(this_file), "testPippo.py")
        cmd = '%s "%s" 2>&1' % (python, fname)
        ExecuteSilentlyIfOK(cmd, self)


# This is a list of "win32com.test.???" module names, optionally with a
# function in that module if the module isn't unitest based...
unittest_modules = [
    # Level 1 tests - fast and few dependencies - good for CI!
    """testIterators testvbscript_regexp testStorage
          testStreams testWMI policySemantics testShell testROT
          testxslt testCollections
          errorSemantics.test testArrays
          testClipboard
          testConversionErrors
        """.split(),
    # Level 2 tests - wants our demo COM objects registered.
    # (these are strange; on github CI they get further than expected when
    # our objects are not installed, so fail to quietly fail with "can't
    # register" like they do locally. So really just a nod to CI)
    """
        testAXScript testDictionary testServers testvb testMarshal
        """.split(),
    # Level 3 tests - Requires Office or other non-free stuff.
    """testMSOffice.TestAll testMSOfficeEvents.test testAccess.test
           testExplorer.TestAll testExchange.test
        """.split(),
    # Level 4 tests - we try and run `makepy` over every typelib installed!
    """testmakepy.TestAll
        """.split(),
]

# A list of other unittest modules we use - these are fully qualified module
# names and the module is assumed to be unittest based.
unittest_other_modules = [
    # Level 1 tests.
    """win32com.directsound.test.ds_test
        """.split(),
    # Level 2 tests.
    [],
    # Level 3 tests.
    [],
    # Level 4 tests.
    [],
]


output_checked_programs = [
    # Level 1 tests.
    [],
    # Level 2 tests.
    [
        ("cscript.exe /nologo //E:vbscript testInterp.vbs", "VBScript test worked OK"),
        (
            "cscript.exe /nologo //E:vbscript testDictionary.vbs",
            "VBScript has successfully tested Python.Dictionary",
        ),
    ],
    # Level 3 tests
    [],
    # Level 4 tests.
    [],
]

custom_test_cases = [
    # Level 1 tests.
    [],
    # Level 2 tests.
    [
        PyCOMTest,
        PippoTest,
    ],
    # Level 3 tests
    [],
    # Level 4 tests.
    [],
]


def get_test_mod_and_func(test_name, import_failures):
    if test_name.find(".") > 0:
        mod_name, func_name = test_name.split(".")
    else:
        mod_name = test_name
        func_name = None
    fq_mod_name = "win32com.test." + mod_name
    try:
        __import__(fq_mod_name)
        mod = sys.modules[fq_mod_name]
    except:
        import_failures.append((mod_name, sys.exc_info()[:2]))
        return None, None
    func = None if func_name is None else getattr(mod, func_name)
    return mod, func


# Return a test suite all loaded with the tests we want to run
def make_test_suite(test_level=1):
    suite = unittest.TestSuite()
    import_failures = []
    loader = TestLoader()
    for i in range(testLevel):
        for mod_name in unittest_modules[i]:
            mod, func = get_test_mod_and_func(mod_name, import_failures)
            if mod is None:
                raise Exception("no such module '{}'".format(mod_name))
            if func is not None:
                test = CapturingFunctionTestCase(func, description=mod_name)
            else:
                if hasattr(mod, "suite"):
                    test = mod.suite()
                else:
                    test = loader.loadTestsFromModule(mod)
            assert test.countTestCases() > 0, "No tests loaded from %r" % mod
            suite.addTest(test)
        for cmd, output in output_checked_programs[i]:
            suite.addTest(ShellTestCase(cmd, output))

        for test_class in custom_test_cases[i]:
            suite.addTest(unittest.defaultTestLoader.loadTestsFromTestCase(test_class))
    # other "normal" unittest modules.
    for i in range(testLevel):
        for mod_name in unittest_other_modules[i]:
            try:
                __import__(mod_name)
            except:
                import_failures.append((mod_name, sys.exc_info()[:2]))
                continue

            mod = sys.modules[mod_name]
            if hasattr(mod, "suite"):
                test = mod.suite()
            else:
                test = loader.loadTestsFromModule(mod)
            assert test.countTestCases() > 0, "No tests loaded from %r" % mod
            suite.addTest(test)

    return suite, import_failures


def usage(why):
    print(why)
    print()
    print("win32com test suite")
    print("usage: testall [-v] test_level")
    print("  where test_level is an integer 1-3.  Level 1 tests are quick,")
    print("  level 2 tests invoke Word, IE etc, level 3 take ages!")
    sys.exit(1)


if __name__ == "__main__":
    try:
        opts, args = getopt.getopt(sys.argv[1:], "v")
    except getopt.error as why:
        usage(why)
    for opt, val in opts:
        if opt == "-v":
            verbosity += 1
    testLevel = 2  # default to quick test with local objects
    test_names = []
    for arg in args:
        try:
            testLevel = int(arg)
            if testLevel < 0 or testLevel > 4:
                raise ValueError("Only levels 1-4 are supported")
        except ValueError:
            test_names.append(arg)
    if test_names:
        usage("Test names are not supported yet")
    CleanGenerated()

    suite, import_failures = make_test_suite(testLevel)
    if verbosity:
        if hasattr(sys, "gettotalrefcount"):
            print("This is a debug build - memory leak tests will also be run.")
            print("These tests may take *many* minutes to run - be patient!")
            print("(running from python.exe will avoid these leak tests)")
        print(
            "Executing level %d tests - %d test cases will be run"
            % (testLevel, suite.countTestCases())
        )
        if verbosity == 1 and suite.countTestCases() < 70:
            # A little row of markers so the dots show how close to finished
            print("|" * suite.countTestCases())
    testRunner = TestRunner(verbosity=verbosity)
    testResult = testRunner.run(suite)
    if import_failures:
        testResult.stream.writeln(
            "*** The following test modules could not be imported ***"
        )
        for mod_name, (exc_type, exc_val) in import_failures:
            desc = "\n".join(traceback.format_exception_only(exc_type, exc_val))
            testResult.stream.write("%s: %s" % (mod_name, desc))
        testResult.stream.writeln(
            "*** %d test(s) could not be run ***" % len(import_failures)
        )

    # re-print unit-test error here so it is noticed
    if not testResult.wasSuccessful():
        print("*" * 20, "- unittest tests FAILED")

    CheckClean()
    pythoncom.CoUninitialize()
    CleanGenerated()
    if not testResult.wasSuccessful():
        sys.exit(1)
