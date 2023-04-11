import os
import textwrap
import pytest

from numpy.testing import IS_PYPY
from . import util


class TestMixed(util.F2PyTest):
    sources = [
        util.getpath("tests", "src", "mixed", "foo.f"),
        util.getpath("tests", "src", "mixed", "foo_fixed.f90"),
        util.getpath("tests", "src", "mixed", "foo_free.f90"),
    ]

    def test_all(self):
        assert self.module.bar11() == 11
        assert self.module.foo_fixed.bar12() == 12
        assert self.module.foo_free.bar13() == 13

    @pytest.mark.xfail(IS_PYPY,
                       reason="PyPy cannot modify tp_doc after PyType_Ready")
    def test_docstring(self):
        expected = textwrap.dedent("""\
        a = bar11()

        Wrapper for ``bar11``.

        Returns
        -------
        a : int
        """)
        assert self.module.bar11.__doc__ == expected
