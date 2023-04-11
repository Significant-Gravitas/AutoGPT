import platform
import pytest
import numpy as np

from numpy import array
from . import util


class TestReturnReal(util.F2PyTest):
    def check_function(self, t, tname):
        if tname in ["t0", "t4", "s0", "s4"]:
            err = 1e-5
        else:
            err = 0.0
        assert abs(t(234) - 234.0) <= err
        assert abs(t(234.6) - 234.6) <= err
        assert abs(t("234") - 234) <= err
        assert abs(t("234.6") - 234.6) <= err
        assert abs(t(-234) + 234) <= err
        assert abs(t([234]) - 234) <= err
        assert abs(t((234, )) - 234.0) <= err
        assert abs(t(array(234)) - 234.0) <= err
        assert abs(t(array([234])) - 234.0) <= err
        assert abs(t(array([[234]])) - 234.0) <= err
        assert abs(t(array([234]).astype("b")) + 22) <= err
        assert abs(t(array([234], "h")) - 234.0) <= err
        assert abs(t(array([234], "i")) - 234.0) <= err
        assert abs(t(array([234], "l")) - 234.0) <= err
        assert abs(t(array([234], "B")) - 234.0) <= err
        assert abs(t(array([234], "f")) - 234.0) <= err
        assert abs(t(array([234], "d")) - 234.0) <= err
        if tname in ["t0", "t4", "s0", "s4"]:
            assert t(1e200) == t(1e300)  # inf

        # pytest.raises(ValueError, t, array([234], 'S1'))
        pytest.raises(ValueError, t, "abc")

        pytest.raises(IndexError, t, [])
        pytest.raises(IndexError, t, ())

        pytest.raises(Exception, t, t)
        pytest.raises(Exception, t, {})

        try:
            r = t(10**400)
            assert repr(r) in ["inf", "Infinity"]
        except OverflowError:
            pass


@pytest.mark.skipif(
    platform.system() == "Darwin",
    reason="Prone to error when run with numpy/f2py/tests on mac os, "
    "but not when run in isolation",
)
@pytest.mark.skipif(
    np.dtype(np.intp).itemsize < 8,
    reason="32-bit builds are buggy"
)
class TestCReturnReal(TestReturnReal):
    suffix = ".pyf"
    module_name = "c_ext_return_real"
    code = """
python module c_ext_return_real
usercode \'\'\'
float t4(float value) { return value; }
void s4(float *t4, float value) { *t4 = value; }
double t8(double value) { return value; }
void s8(double *t8, double value) { *t8 = value; }
\'\'\'
interface
  function t4(value)
    real*4 intent(c) :: t4,value
  end
  function t8(value)
    real*8 intent(c) :: t8,value
  end
  subroutine s4(t4,value)
    intent(c) s4
    real*4 intent(out) :: t4
    real*4 intent(c) :: value
  end
  subroutine s8(t8,value)
    intent(c) s8
    real*8 intent(out) :: t8
    real*8 intent(c) :: value
  end
end interface
end python module c_ext_return_real
    """

    @pytest.mark.parametrize("name", "t4,t8,s4,s8".split(","))
    def test_all(self, name):
        self.check_function(getattr(self.module, name), name)


class TestFReturnReal(TestReturnReal):
    sources = [
        util.getpath("tests", "src", "return_real", "foo77.f"),
        util.getpath("tests", "src", "return_real", "foo90.f90"),
    ]

    @pytest.mark.parametrize("name", "t0,t4,t8,td,s0,s4,s8,sd".split(","))
    def test_all_f77(self, name):
        self.check_function(getattr(self.module, name), name)

    @pytest.mark.parametrize("name", "t0,t4,t8,td,s0,s4,s8,sd".split(","))
    def test_all_f90(self, name):
        self.check_function(getattr(self.module.f90_return_real, name), name)
