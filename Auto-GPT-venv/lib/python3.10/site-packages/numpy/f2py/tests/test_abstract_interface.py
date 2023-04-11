from pathlib import Path
import pytest
import textwrap
from . import util
from numpy.f2py import crackfortran
from numpy.testing import IS_WASM


@pytest.mark.skipif(IS_WASM, reason="Cannot start subprocess")
class TestAbstractInterface(util.F2PyTest):
    sources = [util.getpath("tests", "src", "abstract_interface", "foo.f90")]

    skip = ["add1", "add2"]

    def test_abstract_interface(self):
        assert self.module.ops_module.foo(3, 5) == (8, 13)

    def test_parse_abstract_interface(self):
        # Test gh18403
        fpath = util.getpath("tests", "src", "abstract_interface",
                             "gh18403_mod.f90")
        mod = crackfortran.crackfortran([str(fpath)])
        assert len(mod) == 1
        assert len(mod[0]["body"]) == 1
        assert mod[0]["body"][0]["block"] == "abstract interface"
