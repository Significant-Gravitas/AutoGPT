# NOTE NOTE - This module is designed to fail!
#
# The ONLY purpose for this script is testing/demoing the
# Pythonwin debugger package.

# It does nothing useful, and it even doesnt do that!

import sys
import time

import pywin.debugger


def a():
    a = 1
    try:
        b()
    except:
        # Break into the debugger with the exception information.
        pywin.debugger.post_mortem(sys.exc_info()[2])
        a = 1
        a = 2
        a = 3
        a = 4


def b():
    b = 1
    pywin.debugger.set_trace()
    # After importing or running this module, you are likely to be
    # sitting at the next line.  This is because we explicitely
    # broke into the debugger using the "set_trace() function
    # "pywin.debugger.brk()" is a shorter alias for this.
    c()


def c():
    c = 1
    d()


def d():
    d = 1
    e(d)
    raise ValueError("Hi")


def e(arg):
    e = 1
    time.sleep(1)
    return e


a()
