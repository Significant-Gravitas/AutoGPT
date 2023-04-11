import os
import sys
import pytest
import textwrap

from . import util
from numpy.testing import IS_PYPY


class TestModuleDocString(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "module_data",
                     "module_data_docstring.f90")
    ]

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Fails with MinGW64 Gfortran (Issue #9673)")
    @pytest.mark.xfail(IS_PYPY,
                       reason="PyPy cannot modify tp_doc after PyType_Ready")
    def test_module_docstring(self):
        assert self.module.mod.__doc__ == textwrap.dedent("""\
                     i : 'i'-scalar
                     x : 'i'-array(4)
                     a : 'f'-array(2,3)
                     b : 'f'-array(-1,-1), not allocated\x00
                     foo()\n
                     Wrapper for ``foo``.\n\n""")
