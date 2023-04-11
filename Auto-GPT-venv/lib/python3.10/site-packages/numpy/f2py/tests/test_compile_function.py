"""See https://github.com/numpy/numpy/pull/11937.

"""
import sys
import os
import uuid
from importlib import import_module
import pytest

import numpy.f2py

from . import util


def setup_module():
    if not util.has_c_compiler():
        pytest.skip("Needs C compiler")
    if not util.has_f77_compiler():
        pytest.skip("Needs FORTRAN 77 compiler")


# extra_args can be a list (since gh-11937) or string.
# also test absence of extra_args
@pytest.mark.parametrize("extra_args",
                         [["--noopt", "--debug"], "--noopt --debug", ""])
@pytest.mark.leaks_references(reason="Imported module seems never deleted.")
def test_f2py_init_compile(extra_args):
    # flush through the f2py __init__ compile() function code path as a
    # crude test for input handling following migration from
    # exec_command() to subprocess.check_output() in gh-11937

    # the Fortran 77 syntax requires 6 spaces before any commands, but
    # more space may be added/
    fsource = """
        integer function foo()
        foo = 10 + 5
        return
        end
    """
    # use various helper functions in util.py to enable robust build /
    # compile and reimport cycle in test suite
    moddir = util.get_module_dir()
    modname = util.get_temp_module_name()

    cwd = os.getcwd()
    target = os.path.join(moddir, str(uuid.uuid4()) + ".f")
    # try running compile() with and without a source_fn provided so
    # that the code path where a temporary file for writing Fortran
    # source is created is also explored
    for source_fn in [target, None]:
        # mimic the path changing behavior used by build_module() in
        # util.py, but don't actually use build_module() because it has
        # its own invocation of subprocess that circumvents the
        # f2py.compile code block under test
        with util.switchdir(moddir):
            ret_val = numpy.f2py.compile(fsource,
                                         modulename=modname,
                                         extra_args=extra_args,
                                         source_fn=source_fn)

            # check for compile success return value
            assert ret_val == 0

    # we are not currently able to import the Python-Fortran
    # interface module on Windows / Appveyor, even though we do get
    # successful compilation on that platform with Python 3.x
    if sys.platform != "win32":
        # check for sensible result of Fortran function; that means
        # we can import the module name in Python and retrieve the
        # result of the sum operation
        return_check = import_module(modname)
        calc_result = return_check.foo()
        assert calc_result == 15
        # Removal from sys.modules, is not as such necessary. Even with
        # removal, the module (dict) stays alive.
        del sys.modules[modname]


def test_f2py_init_compile_failure():
    # verify an appropriate integer status value returned by
    # f2py.compile() when invalid Fortran is provided
    ret_val = numpy.f2py.compile(b"invalid")
    assert ret_val == 1


def test_f2py_init_compile_bad_cmd():
    # verify that usage of invalid command in f2py.compile() returns
    # status value of 127 for historic consistency with exec_command()
    # error handling

    # patch the sys Python exe path temporarily to induce an OSError
    # downstream NOTE: how bad of an idea is this patching?
    try:
        temp = sys.executable
        sys.executable = "does not exist"

        # the OSError should take precedence over invalid Fortran
        ret_val = numpy.f2py.compile(b"invalid")
        assert ret_val == 127
    finally:
        sys.executable = temp


@pytest.mark.parametrize(
    "fsource",
    [
        "program test_f2py\nend program test_f2py",
        b"program test_f2py\nend program test_f2py",
    ],
)
def test_compile_from_strings(tmpdir, fsource):
    # Make sure we can compile str and bytes gh-12796
    with util.switchdir(tmpdir):
        ret_val = numpy.f2py.compile(fsource,
                                     modulename="test_compile_from_strings",
                                     extension=".f90")
        assert ret_val == 0
