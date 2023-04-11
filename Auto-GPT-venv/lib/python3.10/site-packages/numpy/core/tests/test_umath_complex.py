import sys
import platform
import pytest

import numpy as np
# import the c-extension module directly since _arg is not exported via umath
import numpy.core._multiarray_umath as ncu
from numpy.testing import (
    assert_raises, assert_equal, assert_array_equal, assert_almost_equal, assert_array_max_ulp
    )

# TODO: branch cuts (use Pauli code)
# TODO: conj 'symmetry'
# TODO: FPU exceptions

# At least on Windows the results of many complex functions are not conforming
# to the C99 standard. See ticket 1574.
# Ditto for Solaris (ticket 1642) and OS X on PowerPC.
#FIXME: this will probably change when we require full C99 campatibility
with np.errstate(all='ignore'):
    functions_seem_flaky = ((np.exp(complex(np.inf, 0)).imag != 0)
                            or (np.log(complex(np.NZERO, 0)).imag != np.pi))
# TODO: replace with a check on whether platform-provided C99 funcs are used
xfail_complex_tests = (not sys.platform.startswith('linux') or functions_seem_flaky)

# TODO This can be xfail when the generator functions are got rid of.
platform_skip = pytest.mark.skipif(xfail_complex_tests,
                                   reason="Inadequate C99 complex support")



class TestCexp:
    def test_simple(self):
        check = check_complex_value
        f = np.exp

        check(f, 1, 0, np.exp(1), 0, False)
        check(f, 0, 1, np.cos(1), np.sin(1), False)

        ref = np.exp(1) * complex(np.cos(1), np.sin(1))
        check(f, 1, 1, ref.real, ref.imag, False)

    @platform_skip
    def test_special_values(self):
        # C99: Section G 6.3.1

        check = check_complex_value
        f = np.exp

        # cexp(+-0 + 0i) is 1 + 0i
        check(f, np.PZERO, 0, 1, 0, False)
        check(f, np.NZERO, 0, 1, 0, False)

        # cexp(x + infi) is nan + nani for finite x and raises 'invalid' FPU
        # exception
        check(f,  1, np.inf, np.nan, np.nan)
        check(f, -1, np.inf, np.nan, np.nan)
        check(f,  0, np.inf, np.nan, np.nan)

        # cexp(inf + 0i) is inf + 0i
        check(f,  np.inf, 0, np.inf, 0)

        # cexp(-inf + yi) is +0 * (cos(y) + i sin(y)) for finite y
        check(f,  -np.inf, 1, np.PZERO, np.PZERO)
        check(f,  -np.inf, 0.75 * np.pi, np.NZERO, np.PZERO)

        # cexp(inf + yi) is +inf * (cos(y) + i sin(y)) for finite y
        check(f,  np.inf, 1, np.inf, np.inf)
        check(f,  np.inf, 0.75 * np.pi, -np.inf, np.inf)

        # cexp(-inf + inf i) is +-0 +- 0i (signs unspecified)
        def _check_ninf_inf(dummy):
            msgform = "cexp(-inf, inf) is (%f, %f), expected (+-0, +-0)"
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(-np.inf, np.inf)))
                if z.real != 0 or z.imag != 0:
                    raise AssertionError(msgform % (z.real, z.imag))

        _check_ninf_inf(None)

        # cexp(inf + inf i) is +-inf + NaNi and raised invalid FPU ex.
        def _check_inf_inf(dummy):
            msgform = "cexp(inf, inf) is (%f, %f), expected (+-inf, nan)"
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(np.inf, np.inf)))
                if not np.isinf(z.real) or not np.isnan(z.imag):
                    raise AssertionError(msgform % (z.real, z.imag))

        _check_inf_inf(None)

        # cexp(-inf + nan i) is +-0 +- 0i
        def _check_ninf_nan(dummy):
            msgform = "cexp(-inf, nan) is (%f, %f), expected (+-0, +-0)"
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(-np.inf, np.nan)))
                if z.real != 0 or z.imag != 0:
                    raise AssertionError(msgform % (z.real, z.imag))

        _check_ninf_nan(None)

        # cexp(inf + nan i) is +-inf + nan
        def _check_inf_nan(dummy):
            msgform = "cexp(-inf, nan) is (%f, %f), expected (+-inf, nan)"
            with np.errstate(invalid='ignore'):
                z = f(np.array(complex(np.inf, np.nan)))
                if not np.isinf(z.real) or not np.isnan(z.imag):
                    raise AssertionError(msgform % (z.real, z.imag))

        _check_inf_nan(None)

        # cexp(nan + yi) is nan + nani for y != 0 (optional: raises invalid FPU
        # ex)
        check(f, np.nan, 1, np.nan, np.nan)
        check(f, np.nan, -1, np.nan, np.nan)

        check(f, np.nan,  np.inf, np.nan, np.nan)
        check(f, np.nan, -np.inf, np.nan, np.nan)

        # cexp(nan + nani) is nan + nani
        check(f, np.nan, np.nan, np.nan, np.nan)

    # TODO This can be xfail when the generator functions are got rid of.
    @pytest.mark.skip(reason="cexp(nan + 0I) is wrong on most platforms")
    def test_special_values2(self):
        # XXX: most implementations get it wrong here (including glibc <= 2.10)
        # cexp(nan + 0i) is nan + 0i
        check = check_complex_value
        f = np.exp

        check(f, np.nan, 0, np.nan, 0)

class TestClog:
    def test_simple(self):
        x = np.array([1+0j, 1+2j])
        y_r = np.log(np.abs(x)) + 1j * np.angle(x)
        y = np.log(x)
        assert_almost_equal(y, y_r)

    @platform_skip
    @pytest.mark.skipif(platform.machine() == "armv5tel", reason="See gh-413.")
    def test_special_values(self):
        xl = []
        yl = []

        # From C99 std (Sec 6.3.2)
        # XXX: check exceptions raised
        # --- raise for invalid fails.

        # clog(-0 + i0) returns -inf + i pi and raises the 'divide-by-zero'
        # floating-point exception.
        with np.errstate(divide='raise'):
            x = np.array([np.NZERO], dtype=complex)
            y = complex(-np.inf, np.pi)
            assert_raises(FloatingPointError, np.log, x)
        with np.errstate(divide='ignore'):
            assert_almost_equal(np.log(x), y)

        xl.append(x)
        yl.append(y)

        # clog(+0 + i0) returns -inf + i0 and raises the 'divide-by-zero'
        # floating-point exception.
        with np.errstate(divide='raise'):
            x = np.array([0], dtype=complex)
            y = complex(-np.inf, 0)
            assert_raises(FloatingPointError, np.log, x)
        with np.errstate(divide='ignore'):
            assert_almost_equal(np.log(x), y)

        xl.append(x)
        yl.append(y)

        # clog(x + i inf returns +inf + i pi /2, for finite x.
        x = np.array([complex(1, np.inf)], dtype=complex)
        y = complex(np.inf, 0.5 * np.pi)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        x = np.array([complex(-1, np.inf)], dtype=complex)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(x + iNaN) returns NaN + iNaN and optionally raises the
        # 'invalid' floating- point exception, for finite x.
        with np.errstate(invalid='raise'):
            x = np.array([complex(1., np.nan)], dtype=complex)
            y = complex(np.nan, np.nan)
            #assert_raises(FloatingPointError, np.log, x)
        with np.errstate(invalid='ignore'):
            assert_almost_equal(np.log(x), y)

        xl.append(x)
        yl.append(y)

        with np.errstate(invalid='raise'):
            x = np.array([np.inf + 1j * np.nan], dtype=complex)
            #assert_raises(FloatingPointError, np.log, x)
        with np.errstate(invalid='ignore'):
            assert_almost_equal(np.log(x), y)

        xl.append(x)
        yl.append(y)

        # clog(- inf + iy) returns +inf + ipi , for finite positive-signed y.
        x = np.array([-np.inf + 1j], dtype=complex)
        y = complex(np.inf, np.pi)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(+ inf + iy) returns +inf + i0, for finite positive-signed y.
        x = np.array([np.inf + 1j], dtype=complex)
        y = complex(np.inf, 0)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(- inf + i inf) returns +inf + i3pi /4.
        x = np.array([complex(-np.inf, np.inf)], dtype=complex)
        y = complex(np.inf, 0.75 * np.pi)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(+ inf + i inf) returns +inf + ipi /4.
        x = np.array([complex(np.inf, np.inf)], dtype=complex)
        y = complex(np.inf, 0.25 * np.pi)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(+/- inf + iNaN) returns +inf + iNaN.
        x = np.array([complex(np.inf, np.nan)], dtype=complex)
        y = complex(np.inf, np.nan)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        x = np.array([complex(-np.inf, np.nan)], dtype=complex)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(NaN + iy) returns NaN + iNaN and optionally raises the
        # 'invalid' floating-point exception, for finite y.
        x = np.array([complex(np.nan, 1)], dtype=complex)
        y = complex(np.nan, np.nan)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(NaN + i inf) returns +inf + iNaN.
        x = np.array([complex(np.nan, np.inf)], dtype=complex)
        y = complex(np.inf, np.nan)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(NaN + iNaN) returns NaN + iNaN.
        x = np.array([complex(np.nan, np.nan)], dtype=complex)
        y = complex(np.nan, np.nan)
        assert_almost_equal(np.log(x), y)
        xl.append(x)
        yl.append(y)

        # clog(conj(z)) = conj(clog(z)).
        xa = np.array(xl, dtype=complex)
        ya = np.array(yl, dtype=complex)
        with np.errstate(divide='ignore'):
            for i in range(len(xa)):
                assert_almost_equal(np.log(xa[i].conj()), ya[i].conj())


class TestCsqrt:

    def test_simple(self):
        # sqrt(1)
        check_complex_value(np.sqrt, 1, 0, 1, 0)

        # sqrt(1i)
        rres = 0.5*np.sqrt(2)
        ires = rres
        check_complex_value(np.sqrt, 0, 1, rres, ires, False)

        # sqrt(-1)
        check_complex_value(np.sqrt, -1, 0, 0, 1)

    def test_simple_conjugate(self):
        ref = np.conj(np.sqrt(complex(1, 1)))

        def f(z):
            return np.sqrt(np.conj(z))

        check_complex_value(f, 1, 1, ref.real, ref.imag, False)

    #def test_branch_cut(self):
    #    _check_branch_cut(f, -1, 0, 1, -1)

    @platform_skip
    def test_special_values(self):
        # C99: Sec G 6.4.2

        check = check_complex_value
        f = np.sqrt

        # csqrt(+-0 + 0i) is 0 + 0i
        check(f, np.PZERO, 0, 0, 0)
        check(f, np.NZERO, 0, 0, 0)

        # csqrt(x + infi) is inf + infi for any x (including NaN)
        check(f,  1, np.inf, np.inf, np.inf)
        check(f, -1, np.inf, np.inf, np.inf)

        check(f, np.PZERO, np.inf, np.inf, np.inf)
        check(f, np.NZERO, np.inf, np.inf, np.inf)
        check(f,   np.inf, np.inf, np.inf, np.inf)
        check(f,  -np.inf, np.inf, np.inf, np.inf)
        check(f,  -np.nan, np.inf, np.inf, np.inf)

        # csqrt(x + nani) is nan + nani for any finite x
        check(f,  1, np.nan, np.nan, np.nan)
        check(f, -1, np.nan, np.nan, np.nan)
        check(f,  0, np.nan, np.nan, np.nan)

        # csqrt(-inf + yi) is +0 + infi for any finite y > 0
        check(f, -np.inf, 1, np.PZERO, np.inf)

        # csqrt(inf + yi) is +inf + 0i for any finite y > 0
        check(f, np.inf, 1, np.inf, np.PZERO)

        # csqrt(-inf + nani) is nan +- infi (both +i infi are valid)
        def _check_ninf_nan(dummy):
            msgform = "csqrt(-inf, nan) is (%f, %f), expected (nan, +-inf)"
            z = np.sqrt(np.array(complex(-np.inf, np.nan)))
            #Fixme: ugly workaround for isinf bug.
            with np.errstate(invalid='ignore'):
                if not (np.isnan(z.real) and np.isinf(z.imag)):
                    raise AssertionError(msgform % (z.real, z.imag))

        _check_ninf_nan(None)

        # csqrt(+inf + nani) is inf + nani
        check(f, np.inf, np.nan, np.inf, np.nan)

        # csqrt(nan + yi) is nan + nani for any finite y (infinite handled in x
        # + nani)
        check(f, np.nan,       0, np.nan, np.nan)
        check(f, np.nan,       1, np.nan, np.nan)
        check(f, np.nan,  np.nan, np.nan, np.nan)

        # XXX: check for conj(csqrt(z)) == csqrt(conj(z)) (need to fix branch
        # cuts first)

class TestCpow:
    def setup_method(self):
        self.olderr = np.seterr(invalid='ignore')

    def teardown_method(self):
        np.seterr(**self.olderr)

    def test_simple(self):
        x = np.array([1+1j, 0+2j, 1+2j, np.inf, np.nan])
        y_r = x ** 2
        y = np.power(x, 2)
        assert_almost_equal(y, y_r)

    def test_scalar(self):
        x = np.array([1, 1j,         2,  2.5+.37j, np.inf, np.nan])
        y = np.array([1, 1j, -0.5+1.5j, -0.5+1.5j,      2,      3])
        lx = list(range(len(x)))

        # Hardcode the expected `builtins.complex` values,
        # as complex exponentiation is broken as of bpo-44698
        p_r = [
            1+0j,
            0.20787957635076193+0j,
            0.35812203996480685+0.6097119028618724j,
            0.12659112128185032+0.48847676699581527j,
            complex(np.inf, np.nan),
            complex(np.nan, np.nan),
        ]

        n_r = [x[i] ** y[i] for i in lx]
        for i in lx:
            assert_almost_equal(n_r[i], p_r[i], err_msg='Loop %d\n' % i)

    def test_array(self):
        x = np.array([1, 1j,         2,  2.5+.37j, np.inf, np.nan])
        y = np.array([1, 1j, -0.5+1.5j, -0.5+1.5j,      2,      3])
        lx = list(range(len(x)))

        # Hardcode the expected `builtins.complex` values,
        # as complex exponentiation is broken as of bpo-44698
        p_r = [
            1+0j,
            0.20787957635076193+0j,
            0.35812203996480685+0.6097119028618724j,
            0.12659112128185032+0.48847676699581527j,
            complex(np.inf, np.nan),
            complex(np.nan, np.nan),
        ]

        n_r = x ** y
        for i in lx:
            assert_almost_equal(n_r[i], p_r[i], err_msg='Loop %d\n' % i)

class TestCabs:
    def setup_method(self):
        self.olderr = np.seterr(invalid='ignore')

    def teardown_method(self):
        np.seterr(**self.olderr)

    def test_simple(self):
        x = np.array([1+1j, 0+2j, 1+2j, np.inf, np.nan])
        y_r = np.array([np.sqrt(2.), 2, np.sqrt(5), np.inf, np.nan])
        y = np.abs(x)
        assert_almost_equal(y, y_r)

    def test_fabs(self):
        # Test that np.abs(x +- 0j) == np.abs(x) (as mandated by C99 for cabs)
        x = np.array([1+0j], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))

        x = np.array([complex(1, np.NZERO)], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))

        x = np.array([complex(np.inf, np.NZERO)], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))

        x = np.array([complex(np.nan, np.NZERO)], dtype=complex)
        assert_array_equal(np.abs(x), np.real(x))

    def test_cabs_inf_nan(self):
        x, y = [], []

        # cabs(+-nan + nani) returns nan
        x.append(np.nan)
        y.append(np.nan)
        check_real_value(np.abs,  np.nan, np.nan, np.nan)

        x.append(np.nan)
        y.append(-np.nan)
        check_real_value(np.abs, -np.nan, np.nan, np.nan)

        # According to C99 standard, if exactly one of the real/part is inf and
        # the other nan, then cabs should return inf
        x.append(np.inf)
        y.append(np.nan)
        check_real_value(np.abs,  np.inf, np.nan, np.inf)

        x.append(-np.inf)
        y.append(np.nan)
        check_real_value(np.abs, -np.inf, np.nan, np.inf)

        # cabs(conj(z)) == conj(cabs(z)) (= cabs(z))
        def f(a):
            return np.abs(np.conj(a))

        def g(a, b):
            return np.abs(complex(a, b))

        xa = np.array(x, dtype=complex)
        assert len(xa) == len(x) == len(y)
        for xi, yi in zip(x, y):
            ref = g(xi, yi)
            check_real_value(f, xi, yi, ref)

class TestCarg:
    def test_simple(self):
        check_real_value(ncu._arg, 1, 0, 0, False)
        check_real_value(ncu._arg, 0, 1, 0.5*np.pi, False)

        check_real_value(ncu._arg, 1, 1, 0.25*np.pi, False)
        check_real_value(ncu._arg, np.PZERO, np.PZERO, np.PZERO)

    # TODO This can be xfail when the generator functions are got rid of.
    @pytest.mark.skip(
        reason="Complex arithmetic with signed zero fails on most platforms")
    def test_zero(self):
        # carg(-0 +- 0i) returns +- pi
        check_real_value(ncu._arg, np.NZERO, np.PZERO,  np.pi, False)
        check_real_value(ncu._arg, np.NZERO, np.NZERO, -np.pi, False)

        # carg(+0 +- 0i) returns +- 0
        check_real_value(ncu._arg, np.PZERO, np.PZERO, np.PZERO)
        check_real_value(ncu._arg, np.PZERO, np.NZERO, np.NZERO)

        # carg(x +- 0i) returns +- 0 for x > 0
        check_real_value(ncu._arg, 1, np.PZERO, np.PZERO, False)
        check_real_value(ncu._arg, 1, np.NZERO, np.NZERO, False)

        # carg(x +- 0i) returns +- pi for x < 0
        check_real_value(ncu._arg, -1, np.PZERO,  np.pi, False)
        check_real_value(ncu._arg, -1, np.NZERO, -np.pi, False)

        # carg(+- 0 + yi) returns pi/2 for y > 0
        check_real_value(ncu._arg, np.PZERO, 1, 0.5 * np.pi, False)
        check_real_value(ncu._arg, np.NZERO, 1, 0.5 * np.pi, False)

        # carg(+- 0 + yi) returns -pi/2 for y < 0
        check_real_value(ncu._arg, np.PZERO, -1, 0.5 * np.pi, False)
        check_real_value(ncu._arg, np.NZERO, -1, -0.5 * np.pi, False)

    #def test_branch_cuts(self):
    #    _check_branch_cut(ncu._arg, -1, 1j, -1, 1)

    def test_special_values(self):
        # carg(-np.inf +- yi) returns +-pi for finite y > 0
        check_real_value(ncu._arg, -np.inf,  1,  np.pi, False)
        check_real_value(ncu._arg, -np.inf, -1, -np.pi, False)

        # carg(np.inf +- yi) returns +-0 for finite y > 0
        check_real_value(ncu._arg, np.inf,  1, np.PZERO, False)
        check_real_value(ncu._arg, np.inf, -1, np.NZERO, False)

        # carg(x +- np.infi) returns +-pi/2 for finite x
        check_real_value(ncu._arg, 1,  np.inf,  0.5 * np.pi, False)
        check_real_value(ncu._arg, 1, -np.inf, -0.5 * np.pi, False)

        # carg(-np.inf +- np.infi) returns +-3pi/4
        check_real_value(ncu._arg, -np.inf,  np.inf,  0.75 * np.pi, False)
        check_real_value(ncu._arg, -np.inf, -np.inf, -0.75 * np.pi, False)

        # carg(np.inf +- np.infi) returns +-pi/4
        check_real_value(ncu._arg, np.inf,  np.inf,  0.25 * np.pi, False)
        check_real_value(ncu._arg, np.inf, -np.inf, -0.25 * np.pi, False)

        # carg(x + yi) returns np.nan if x or y is nan
        check_real_value(ncu._arg, np.nan,      0, np.nan, False)
        check_real_value(ncu._arg,      0, np.nan, np.nan, False)

        check_real_value(ncu._arg, np.nan, np.inf, np.nan, False)
        check_real_value(ncu._arg, np.inf, np.nan, np.nan, False)


def check_real_value(f, x1, y1, x, exact=True):
    z1 = np.array([complex(x1, y1)])
    if exact:
        assert_equal(f(z1), x)
    else:
        assert_almost_equal(f(z1), x)


def check_complex_value(f, x1, y1, x2, y2, exact=True):
    z1 = np.array([complex(x1, y1)])
    z2 = complex(x2, y2)
    with np.errstate(invalid='ignore'):
        if exact:
            assert_equal(f(z1), z2)
        else:
            assert_almost_equal(f(z1), z2)

class TestSpecialComplexAVX:
    @pytest.mark.parametrize("stride", [-4,-2,-1,1,2,4])
    @pytest.mark.parametrize("astype", [np.complex64, np.complex128])
    def test_array(self, stride, astype):
        arr = np.array([complex(np.nan , np.nan),
                        complex(np.nan , np.inf),
                        complex(np.inf , np.nan),
                        complex(np.inf , np.inf),
                        complex(0.     , np.inf),
                        complex(np.inf , 0.),
                        complex(0.     , 0.),
                        complex(0.     , np.nan),
                        complex(np.nan , 0.)], dtype=astype)
        abs_true = np.array([np.nan, np.inf, np.inf, np.inf, np.inf, np.inf, 0., np.nan, np.nan], dtype=arr.real.dtype)
        sq_true = np.array([complex(np.nan,  np.nan),
                            complex(np.nan,  np.nan),
                            complex(np.nan,  np.nan),
                            complex(np.nan,  np.inf),
                            complex(-np.inf, np.nan),
                            complex(np.inf,  np.nan),
                            complex(0.,     0.),
                            complex(np.nan, np.nan),
                            complex(np.nan, np.nan)], dtype=astype)
        assert_equal(np.abs(arr[::stride]), abs_true[::stride])
        with np.errstate(invalid='ignore'):
            assert_equal(np.square(arr[::stride]), sq_true[::stride])

class TestComplexAbsoluteAVX:
    @pytest.mark.parametrize("arraysize", [1,2,3,4,5,6,7,8,9,10,11,13,15,17,18,19])
    @pytest.mark.parametrize("stride", [-4,-3,-2,-1,1,2,3,4])
    @pytest.mark.parametrize("astype", [np.complex64, np.complex128])
    # test to ensure masking and strides work as intended in the AVX implementation
    def test_array(self, arraysize, stride, astype):
        arr = np.ones(arraysize, dtype=astype)
        abs_true = np.ones(arraysize, dtype=arr.real.dtype)
        assert_equal(np.abs(arr[::stride]), abs_true[::stride])

# Testcase taken as is from https://github.com/numpy/numpy/issues/16660
class TestComplexAbsoluteMixedDTypes:
    @pytest.mark.parametrize("stride", [-4,-3,-2,-1,1,2,3,4])
    @pytest.mark.parametrize("astype", [np.complex64, np.complex128])
    @pytest.mark.parametrize("func", ['abs', 'square', 'conjugate'])

    def test_array(self, stride, astype, func):
        dtype = [('template_id', '<i8'), ('bank_chisq','<f4'),
                 ('bank_chisq_dof','<i8'), ('chisq', '<f4'), ('chisq_dof','<i8'),
                 ('cont_chisq', '<f4'), ('psd_var_val', '<f4'), ('sg_chisq','<f4'),
                 ('mycomplex', astype), ('time_index', '<i8')]
        vec = np.array([
               (0, 0., 0, -31.666483, 200, 0., 0.,  1.      ,  3.0+4.0j   ,  613090),
               (1, 0., 0, 260.91525 ,  42, 0., 0.,  1.      ,  5.0+12.0j  ,  787315),
               (1, 0., 0,  52.15155 ,  42, 0., 0.,  1.      ,  8.0+15.0j  ,  806641),
               (1, 0., 0,  52.430195,  42, 0., 0.,  1.      ,  7.0+24.0j  , 1363540),
               (2, 0., 0, 304.43646 ,  58, 0., 0.,  1.      ,  20.0+21.0j ,  787323),
               (3, 0., 0, 299.42108 ,  52, 0., 0.,  1.      ,  12.0+35.0j ,  787332),
               (4, 0., 0,  39.4836  ,  28, 0., 0.,  9.182192,  9.0+40.0j  ,  787304),
               (4, 0., 0,  76.83787 ,  28, 0., 0.,  1.      ,  28.0+45.0j, 1321869),
               (5, 0., 0, 143.26366 ,  24, 0., 0., 10.996129,  11.0+60.0j ,  787299)], dtype=dtype)
        myfunc = getattr(np, func)
        a = vec['mycomplex']
        g = myfunc(a[::stride])

        b = vec['mycomplex'].copy()
        h = myfunc(b[::stride])

        assert_array_max_ulp(h.real, g.real, 1)
        assert_array_max_ulp(h.imag, g.imag, 1)
