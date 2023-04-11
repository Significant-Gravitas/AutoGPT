"""See https://github.com/numpy/numpy/pull/10676.

"""
import sys
import pytest

from . import util


class TestQuotedCharacter(util.F2PyTest):
    sources = [util.getpath("tests", "src", "quoted_character", "foo.f")]

    @pytest.mark.skipif(sys.platform == "win32",
                        reason="Fails with MinGW64 Gfortran (Issue #9673)")
    def test_quoted_character(self):
        assert self.module.foo() == (b"'", b'"', b";", b"!", b"(", b")")
