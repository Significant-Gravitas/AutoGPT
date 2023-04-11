""" Test scripts

Test that we can run executable scripts that have been installed with numpy.
"""
import sys
import os
import pytest
from os.path import join as pathjoin, isfile, dirname
import subprocess

import numpy as np
from numpy.testing import assert_equal, IS_WASM

is_inplace = isfile(pathjoin(dirname(np.__file__),  '..', 'setup.py'))


def find_f2py_commands():
    if sys.platform == 'win32':
        exe_dir = dirname(sys.executable)
        if exe_dir.endswith('Scripts'): # virtualenv
            return [os.path.join(exe_dir, 'f2py')]
        else:
            return [os.path.join(exe_dir, "Scripts", 'f2py')]
    else:
        # Three scripts are installed in Unix-like systems:
        # 'f2py', 'f2py{major}', and 'f2py{major.minor}'. For example,
        # if installed with python3.9 the scripts would be named
        # 'f2py', 'f2py3', and 'f2py3.9'.
        version = sys.version_info
        major = str(version.major)
        minor = str(version.minor)
        return ['f2py', 'f2py' + major, 'f2py' + major + '.' + minor]


@pytest.mark.skipif(is_inplace, reason="Cannot test f2py command inplace")
@pytest.mark.xfail(reason="Test is unreliable")
@pytest.mark.parametrize('f2py_cmd', find_f2py_commands())
def test_f2py(f2py_cmd):
    # test that we can run f2py script
    stdout = subprocess.check_output([f2py_cmd, '-v'])
    assert_equal(stdout.strip(), np.__version__.encode('ascii'))


@pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
def test_pep338():
    stdout = subprocess.check_output([sys.executable, '-mnumpy.f2py', '-v'])
    assert_equal(stdout.strip(), np.__version__.encode('ascii'))
