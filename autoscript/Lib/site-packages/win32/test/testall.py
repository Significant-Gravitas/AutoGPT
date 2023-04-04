import os
import re
import sys
import traceback
import unittest

import pywin32_testutil

# A list of demos that depend on user-interface of *any* kind.  Tests listed
# here are not suitable for unattended testing.
ui_demos = """GetSaveFileName print_desktop win32cred_demo win32gui_demo
              win32gui_dialog win32gui_menu win32gui_taskbar
              win32rcparser_demo winprocess win32console_demo
              win32clipboard_bitmapdemo
              win32gui_devicenotify
              NetValidatePasswordPolicy""".split()
# Other demos known as 'bad' (or at least highly unlikely to work)
# cerapi: no CE module is built (CE via pywin32 appears dead)
# desktopmanager: hangs (well, hangs for 60secs or so...)
# EvtSubscribe_*: must be run together:
# SystemParametersInfo: a couple of the params cause markh to hang, and there's
# no great reason to adjust (twice!) all those system settings!
bad_demos = """cerapi desktopmanager win32comport_demo
               EvtSubscribe_pull EvtSubscribe_push
               SystemParametersInfo
            """.split()

argvs = {
    "rastest": ("-l",),
}

no_user_interaction = True

# re to pull apart an exception line into the exception type and the args.
re_exception = re.compile("([a-zA-Z0-9_.]*): (.*)$")


def find_exception_in_output(data):
    have_traceback = False
    for line in data.splitlines():
        line = line.decode("ascii")  # not sure what the correct encoding is...
        if line.startswith("Traceback ("):
            have_traceback = True
            continue
        if line.startswith(" "):
            continue
        if have_traceback:
            # first line not starting with a space since the traceback.
            # must be the exception!
            m = re_exception.match(line)
            if m:
                exc_type, args = m.groups()
                # get hacky - get the *real* exception object from the name.
                bits = exc_type.split(".", 1)
                if len(bits) > 1:
                    mod = __import__(bits[0])
                    exc = getattr(mod, bits[1])
                else:
                    # probably builtin
                    exc = eval(bits[0])
            else:
                # hrm - probably just an exception with no args
                try:
                    exc = eval(line.strip())
                    args = "()"
                except:
                    return None
            # try and turn the args into real args.
            try:
                args = eval(args)
            except:
                pass
            if not isinstance(args, tuple):
                args = (args,)
            # try and instantiate the exception.
            try:
                ret = exc(*args)
            except:
                ret = None
            return ret
        # apparently not - keep looking...
        have_traceback = False


class TestRunner:
    def __init__(self, argv):
        self.argv = argv
        self.__name__ = "Test Runner for cmdline {}".format(argv)

    def __call__(self):
        import subprocess

        p = subprocess.Popen(
            self.argv, stdout=subprocess.PIPE, stderr=subprocess.STDOUT
        )
        output, _ = p.communicate()
        rc = p.returncode

        if rc:
            base = os.path.basename(self.argv[1])
            # See if we can detect and reconstruct an exception in the output.
            reconstituted = find_exception_in_output(output)
            if reconstituted is not None:
                raise reconstituted
            raise AssertionError(
                "%s failed with exit code %s.  Output is:\n%s" % (base, rc, output)
            )


def get_demo_tests():
    import win32api

    ret = []
    demo_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "Demos"))
    assert os.path.isdir(demo_dir), demo_dir
    for name in os.listdir(demo_dir):
        base, ext = os.path.splitext(name)
        if base in ui_demos and no_user_interaction:
            continue
        # Skip any other files than .py and bad tests in any case
        if ext != ".py" or base in bad_demos:
            continue
        argv = (sys.executable, os.path.join(demo_dir, base + ".py")) + argvs.get(
            base, ()
        )
        ret.append(
            unittest.FunctionTestCase(
                TestRunner(argv), description="win32/demos/" + name
            )
        )
    return ret


def import_all():
    # Some hacks for import order - dde depends on win32ui
    try:
        import win32ui
    except ImportError:
        pass  # 'what-ev-a....'

    import win32api

    dir = os.path.dirname(win32api.__file__)
    num = 0
    is_debug = os.path.basename(win32api.__file__).endswith("_d")
    for name in os.listdir(dir):
        base, ext = os.path.splitext(name)
        # handle `modname.cp310-win_amd64.pyd` etc
        base = base.split(".")[0]
        if (
            (ext == ".pyd")
            and name != "_winxptheme.pyd"
            and (
                is_debug
                and base.endswith("_d")
                or not is_debug
                and not base.endswith("_d")
            )
        ):
            try:
                __import__(base)
            except:
                print("FAILED to import", name)
                raise
            num += 1


def suite():
    # Loop over all .py files here, except me :)
    try:
        me = __file__
    except NameError:
        me = sys.argv[0]
    me = os.path.abspath(me)
    files = os.listdir(os.path.dirname(me))
    suite = unittest.TestSuite()
    suite.addTest(unittest.FunctionTestCase(import_all))
    for file in files:
        base, ext = os.path.splitext(file)
        if ext == ".py" and os.path.basename(me) != file:
            try:
                mod = __import__(base)
            except:
                print("FAILED to import test module %r" % base)
                traceback.print_exc()
                continue
            if hasattr(mod, "suite"):
                test = mod.suite()
            else:
                test = unittest.defaultTestLoader.loadTestsFromModule(mod)
            suite.addTest(test)
    for test in get_demo_tests():
        suite.addTest(test)
    return suite


class CustomLoader(pywin32_testutil.TestLoader):
    def loadTestsFromModule(self, module):
        return self.fixupTestsForLeakTests(suite())


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Test runner for PyWin32/win32")
    parser.add_argument(
        "-no-user-interaction",
        default=False,
        action="store_true",
        help="(This is now the default - use `-user-interaction` to include them)",
    )

    parser.add_argument(
        "-user-interaction",
        action="store_true",
        help="Include tests which require user interaction",
    )

    parsed_args, remains = parser.parse_known_args()

    if parsed_args.no_user_interaction:
        print(
            "Note: -no-user-interaction is now the default, run with `-user-interaction` to include them."
        )

    no_user_interaction = not parsed_args.user_interaction

    sys.argv = [sys.argv[0]] + remains

    pywin32_testutil.testmain(testLoader=CustomLoader())
