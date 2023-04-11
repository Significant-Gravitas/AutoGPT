import pytest

from numpy import array
from . import util


class TestReturnComplex(util.F2PyTest):
    def check_function(self, t, tname):
        if tname in ["t0", "t8", "s0", "s8"]:
            err = 1e-5
        else:
            err = 0.0
        assert abs(t(234j) - 234.0j) <= err
        assert abs(t(234.6) - 234.6) <= err
        assert abs(t(234) - 234.0) <= err
        assert abs(t(234.6 + 3j) - (234.6 + 3j)) <= err
        # assert abs(t('234')-234.)<=err
        # assert abs(t('234.6')-234.6)<=err
        assert abs(t(-234) + 234.0) <= err
        assert abs(t([234]) - 234.0) <= err
        assert abs(t((234, )) - 234.0) <= err
        assert abs(t(array(234)) - 234.0) <= err
        assert abs(t(array(23 + 4j, "F")) - (23 + 4j)) <= err
        assert abs(t(array([234])) - 234.0) <= err
        assert abs(t(array([[234]])) - 234.0) <= err
        assert abs(t(array([234]).astype("b")) + 22.0) <= err
        assert abs(t(array([234], "h")) - 234.0) <= err
        assert abs(t(array([234], "i")) - 234.0) <= err
        assert abs(t(array([234], "l")) - 234.0) <= err
        assert abs(t(array([234], "q")) - 234.0) <= err
        assert abs(t(array([234], "f")) - 234.0) <= err
        assert abs(t(array([234], "d")) - 234.0) <= err
        assert abs(t(array([234 + 3j], "F")) - (234 + 3j)) <= err
        assert abs(t(array([234], "D")) - 234.0) <= err

        # pytest.raises(TypeError, t, array([234], 'a1'))
        pytest.raises(TypeError, t, "abc")

        pytest.raises(IndexError, t, [])
        pytest.raises(IndexError, t, ())

        pytest.raises(TypeError, t, t)
        pytest.raises(TypeError, t, {})

        try:
            r = t(10**400)
            assert repr(r) in ["(inf+0j)", "(Infinity+0j)"]
        except OverflowError:
            pass


class TestFReturnComplex(TestReturnComplex):
    sources = [
        util.getpath("tests", "src", "return_complex", "foo77.f"),
        util.getpath("tests", "src", "return_complex", "foo90.f90"),
    ]

    @pytest.mark.parametrize("name", "t0,t8,t16,td,s0,s8,s16,sd".split(","))
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name), name)

    @pytest.mark.parametrize("name", "t0,t8,t16,td,s0,s8,s16,sd".split(","))
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_complex, name),
                            name)
