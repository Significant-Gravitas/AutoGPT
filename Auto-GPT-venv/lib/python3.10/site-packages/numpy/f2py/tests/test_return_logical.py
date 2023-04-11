import pytest

from numpy import array
from . import util


class TestReturnLogical(util.F2PyTest):
    def check_function(self, t):
        assert t(True) == 1
        assert t(False) == 0
        assert t(0) == 0
        assert t(None) == 0
        assert t(0.0) == 0
        assert t(0j) == 0
        assert t(1j) == 1
        assert t(234) == 1
        assert t(234.6) == 1
        assert t(234.6 + 3j) == 1
        assert t("234") == 1
        assert t("aaa") == 1
        assert t("") == 0
        assert t([]) == 0
        assert t(()) == 0
        assert t({}) == 0
        assert t(t) == 1
        assert t(-234) == 1
        assert t(10**100) == 1
        assert t([234]) == 1
        assert t((234, )) == 1
        assert t(array(234)) == 1
        assert t(array([234])) == 1
        assert t(array([[234]])) == 1
        assert t(array([127], "b")) == 1
        assert t(array([234], "h")) == 1
        assert t(array([234], "i")) == 1
        assert t(array([234], "l")) == 1
        assert t(array([234], "f")) == 1
        assert t(array([234], "d")) == 1
        assert t(array([234 + 3j], "F")) == 1
        assert t(array([234], "D")) == 1
        assert t(array(0)) == 0
        assert t(array([0])) == 0
        assert t(array([[0]])) == 0
        assert t(array([0j])) == 0
        assert t(array([1])) == 1
        pytest.raises(ValueError, t, array([0, 0]))


class TestFReturnLogical(TestReturnLogical):
    sources = [
        util.getpath("tests", "src", "return_logical", "foo77.f"),
        util.getpath("tests", "src", "return_logical", "foo90.f90"),
    ]

    @pytest.mark.slow
    @pytest.mark.parametrize("name", "t0,t1,t2,t4,s0,s1,s2,s4".split(","))
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name))

    @pytest.mark.slow
    @pytest.mark.parametrize("name",
                             "t0,t1,t2,t4,t8,s0,s1,s2,s4,s8".split(","))
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_logical, name))
