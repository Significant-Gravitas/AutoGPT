import platform
import warnings
import fnmatch
import itertools
import pytest
import sys
import os
import operator
from fractions import Fraction
from functools import reduce
from collections import namedtuple

import numpy.core.umath as ncu
from numpy.core import _umath_tests as ncu_tests
import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_raises_regex,
    assert_array_equal, assert_almost_equal, assert_array_almost_equal,
    assert_array_max_ulp, assert_allclose, assert_no_warnings, suppress_warnings,
    _gen_alignment_data, assert_array_almost_equal_nulp, IS_WASM
    )
from numpy.testing._private.utils import _glibc_older_than

UFUNCS = [obj for obj in np.core.umath.__dict__.values()
         if isinstance(obj, np.ufunc)]

UFUNCS_UNARY = [
    uf for uf in UFUNCS if uf.nin == 1
]
UFUNCS_UNARY_FP = [
    uf for uf in UFUNCS_UNARY if 'f->f' in uf.types
]

UFUNCS_BINARY = [
    uf for uf in UFUNCS if uf.nin == 2
]
UFUNCS_BINARY_ACC = [
    uf for uf in UFUNCS_BINARY if hasattr(uf, "accumulate") and uf.nout == 1
]

def interesting_binop_operands(val1, val2, dtype):
    """
    Helper to create "interesting" operands to cover common code paths:
    * scalar inputs
    * only first "values" is an array (e.g. scalar division fast-paths)
    * Longer array (SIMD) placing the value of interest at different positions
    * Oddly strided arrays which may not be SIMD compatible

    It does not attempt to cover unaligned access or mixed dtypes.
    These are normally handled by the casting/buffering machinery.

    This is not a fixture (currently), since I believe a fixture normally
    only yields once?
    """
    fill_value = 1  # could be a parameter, but maybe not an optional one?

    arr1 = np.full(10003, dtype=dtype, fill_value=fill_value)
    arr2 = np.full(10003, dtype=dtype, fill_value=fill_value)

    arr1[0] = val1
    arr2[0] = val2

    extractor = lambda res: res
    yield arr1[0], arr2[0], extractor, "scalars"

    extractor = lambda res: res
    yield arr1[0, ...], arr2[0, ...], extractor, "scalar-arrays"

    # reset array values to fill_value:
    arr1[0] = fill_value
    arr2[0] = fill_value

    for pos in [0, 1, 2, 3, 4, 5, -1, -2, -3, -4]:
        arr1[pos] = val1
        arr2[pos] = val2

        extractor = lambda res: res[pos]
        yield arr1, arr2, extractor, f"off-{pos}"
        yield arr1, arr2[pos], extractor, f"off-{pos}-with-scalar"

        arr1[pos] = fill_value
        arr2[pos] = fill_value

    for stride in [-1, 113]:
        op1 = arr1[::stride]
        op2 = arr2[::stride]
        op1[10] = val1
        op2[10] = val2

        extractor = lambda res: res[10]
        yield op1, op2, extractor, f"stride-{stride}"

        op1[10] = fill_value
        op2[10] = fill_value


def on_powerpc():
    """ True if we are running on a Power PC platform."""
    return platform.processor() == 'powerpc' or \
           platform.machine().startswith('ppc')


def bad_arcsinh():
    """The blocklisted trig functions are not accurate on aarch64/PPC for
    complex256. Rather than dig through the actual problem skip the
    test. This should be fixed when we can move past glibc2.17
    which is the version in manylinux2014
    """
    if platform.machine() == 'aarch64':
        x = 1.78e-10
    elif on_powerpc():
        x = 2.16e-10
    else:
        return False
    v1 = np.arcsinh(np.float128(x))
    v2 = np.arcsinh(np.complex256(x)).real
    # The eps for float128 is 1-e33, so this is way bigger
    return abs((v1 / v2) - 1.0) > 1e-23


class _FilterInvalids:
    def setup_method(self):
        self.olderr = np.seterr(invalid='ignore')

    def teardown_method(self):
        np.seterr(**self.olderr)


class TestConstants:
    def test_pi(self):
        assert_allclose(ncu.pi, 3.141592653589793, 1e-15)

    def test_e(self):
        assert_allclose(ncu.e, 2.718281828459045, 1e-15)

    def test_euler_gamma(self):
        assert_allclose(ncu.euler_gamma, 0.5772156649015329, 1e-15)


class TestOut:
    def test_out_subok(self):
        for subok in (True, False):
            a = np.array(0.5)
            o = np.empty(())

            r = np.add(a, 2, o, subok=subok)
            assert_(r is o)
            r = np.add(a, 2, out=o, subok=subok)
            assert_(r is o)
            r = np.add(a, 2, out=(o,), subok=subok)
            assert_(r is o)

            d = np.array(5.7)
            o1 = np.empty(())
            o2 = np.empty((), dtype=np.int32)

            r1, r2 = np.frexp(d, o1, None, subok=subok)
            assert_(r1 is o1)
            r1, r2 = np.frexp(d, None, o2, subok=subok)
            assert_(r2 is o2)
            r1, r2 = np.frexp(d, o1, o2, subok=subok)
            assert_(r1 is o1)
            assert_(r2 is o2)

            r1, r2 = np.frexp(d, out=(o1, None), subok=subok)
            assert_(r1 is o1)
            r1, r2 = np.frexp(d, out=(None, o2), subok=subok)
            assert_(r2 is o2)
            r1, r2 = np.frexp(d, out=(o1, o2), subok=subok)
            assert_(r1 is o1)
            assert_(r2 is o2)

            with assert_raises(TypeError):
                # Out argument must be tuple, since there are multiple outputs.
                r1, r2 = np.frexp(d, out=o1, subok=subok)

            assert_raises(TypeError, np.add, a, 2, o, o, subok=subok)
            assert_raises(TypeError, np.add, a, 2, o, out=o, subok=subok)
            assert_raises(TypeError, np.add, a, 2, None, out=o, subok=subok)
            assert_raises(ValueError, np.add, a, 2, out=(o, o), subok=subok)
            assert_raises(ValueError, np.add, a, 2, out=(), subok=subok)
            assert_raises(TypeError, np.add, a, 2, [], subok=subok)
            assert_raises(TypeError, np.add, a, 2, out=[], subok=subok)
            assert_raises(TypeError, np.add, a, 2, out=([],), subok=subok)
            o.flags.writeable = False
            assert_raises(ValueError, np.add, a, 2, o, subok=subok)
            assert_raises(ValueError, np.add, a, 2, out=o, subok=subok)
            assert_raises(ValueError, np.add, a, 2, out=(o,), subok=subok)

    def test_out_wrap_subok(self):
        class ArrayWrap(np.ndarray):
            __array_priority__ = 10

            def __new__(cls, arr):
                return np.asarray(arr).view(cls).copy()

            def __array_wrap__(self, arr, context):
                return arr.view(type(self))

        for subok in (True, False):
            a = ArrayWrap([0.5])

            r = np.add(a, 2, subok=subok)
            if subok:
                assert_(isinstance(r, ArrayWrap))
            else:
                assert_(type(r) == np.ndarray)

            r = np.add(a, 2, None, subok=subok)
            if subok:
                assert_(isinstance(r, ArrayWrap))
            else:
                assert_(type(r) == np.ndarray)

            r = np.add(a, 2, out=None, subok=subok)
            if subok:
                assert_(isinstance(r, ArrayWrap))
            else:
                assert_(type(r) == np.ndarray)

            r = np.add(a, 2, out=(None,), subok=subok)
            if subok:
                assert_(isinstance(r, ArrayWrap))
            else:
                assert_(type(r) == np.ndarray)

            d = ArrayWrap([5.7])
            o1 = np.empty((1,))
            o2 = np.empty((1,), dtype=np.int32)

            r1, r2 = np.frexp(d, o1, subok=subok)
            if subok:
                assert_(isinstance(r2, ArrayWrap))
            else:
                assert_(type(r2) == np.ndarray)

            r1, r2 = np.frexp(d, o1, None, subok=subok)
            if subok:
                assert_(isinstance(r2, ArrayWrap))
            else:
                assert_(type(r2) == np.ndarray)

            r1, r2 = np.frexp(d, None, o2, subok=subok)
            if subok:
                assert_(isinstance(r1, ArrayWrap))
            else:
                assert_(type(r1) == np.ndarray)

            r1, r2 = np.frexp(d, out=(o1, None), subok=subok)
            if subok:
                assert_(isinstance(r2, ArrayWrap))
            else:
                assert_(type(r2) == np.ndarray)

            r1, r2 = np.frexp(d, out=(None, o2), subok=subok)
            if subok:
                assert_(isinstance(r1, ArrayWrap))
            else:
                assert_(type(r1) == np.ndarray)

            with assert_raises(TypeError):
                # Out argument must be tuple, since there are multiple outputs.
                r1, r2 = np.frexp(d, out=o1, subok=subok)


class TestComparisons:
    import operator

    @pytest.mark.parametrize('dtype', np.sctypes['uint'] + np.sctypes['int'] +
                             np.sctypes['float'] + [np.bool_])
    @pytest.mark.parametrize('py_comp,np_comp', [
        (operator.lt, np.less),
        (operator.le, np.less_equal),
        (operator.gt, np.greater),
        (operator.ge, np.greater_equal),
        (operator.eq, np.equal),
        (operator.ne, np.not_equal)
    ])
    def test_comparison_functions(self, dtype, py_comp, np_comp):
        # Initialize input arrays
        if dtype == np.bool_:
            a = np.random.choice(a=[False, True], size=1000)
            b = np.random.choice(a=[False, True], size=1000)
            scalar = True
        else:
            a = np.random.randint(low=1, high=10, size=1000).astype(dtype)
            b = np.random.randint(low=1, high=10, size=1000).astype(dtype)
            scalar = 5
        np_scalar = np.dtype(dtype).type(scalar)
        a_lst = a.tolist()
        b_lst = b.tolist()

        # (Binary) Comparison (x1=array, x2=array)
        comp_b = np_comp(a, b).view(np.uint8)
        comp_b_list = [int(py_comp(x, y)) for x, y in zip(a_lst, b_lst)]

        # (Scalar1) Comparison (x1=scalar, x2=array)
        comp_s1 = np_comp(np_scalar, b).view(np.uint8)
        comp_s1_list = [int(py_comp(scalar, x)) for x in b_lst]

        # (Scalar2) Comparison (x1=array, x2=scalar)
        comp_s2 = np_comp(a, np_scalar).view(np.uint8)
        comp_s2_list = [int(py_comp(x, scalar)) for x in a_lst]

        # Sequence: Binary, Scalar1 and Scalar2
        assert_(comp_b.tolist() == comp_b_list,
            f"Failed comparison ({py_comp.__name__})")
        assert_(comp_s1.tolist() == comp_s1_list,
            f"Failed comparison ({py_comp.__name__})")
        assert_(comp_s2.tolist() == comp_s2_list,
            f"Failed comparison ({py_comp.__name__})")

    def test_ignore_object_identity_in_equal(self):
        # Check comparing identical objects whose comparison
        # is not a simple boolean, e.g., arrays that are compared elementwise.
        a = np.array([np.array([1, 2, 3]), None], dtype=object)
        assert_raises(ValueError, np.equal, a, a)

        # Check error raised when comparing identical non-comparable objects.
        class FunkyType:
            def __eq__(self, other):
                raise TypeError("I won't compare")

        a = np.array([FunkyType()])
        assert_raises(TypeError, np.equal, a, a)

        # Check identity doesn't override comparison mismatch.
        a = np.array([np.nan], dtype=object)
        assert_equal(np.equal(a, a), [False])

    def test_ignore_object_identity_in_not_equal(self):
        # Check comparing identical objects whose comparison
        # is not a simple boolean, e.g., arrays that are compared elementwise.
        a = np.array([np.array([1, 2, 3]), None], dtype=object)
        assert_raises(ValueError, np.not_equal, a, a)

        # Check error raised when comparing identical non-comparable objects.
        class FunkyType:
            def __ne__(self, other):
                raise TypeError("I won't compare")

        a = np.array([FunkyType()])
        assert_raises(TypeError, np.not_equal, a, a)

        # Check identity doesn't override comparison mismatch.
        a = np.array([np.nan], dtype=object)
        assert_equal(np.not_equal(a, a), [True])

    def test_error_in_equal_reduce(self):
        # gh-20929
        # make sure np.equal.reduce raises a TypeError if an array is passed
        # without specifying the dtype
        a = np.array([0, 0])
        assert_equal(np.equal.reduce(a, dtype=bool), True)
        assert_raises(TypeError, np.equal.reduce, a)

    def test_object_dtype(self):
        assert np.equal(1, [1], dtype=object).dtype == object
        assert np.equal(1, [1], signature=(None, None, "O")).dtype == object

    def test_object_nonbool_dtype_error(self):
        # bool output dtype is fine of course:
        assert np.equal(1, [1], dtype=bool).dtype == bool

        # but the following are examples do not have a loop:
        with pytest.raises(TypeError, match="No loop matching"):
            np.equal(1, 1, dtype=np.int64)

        with pytest.raises(TypeError, match="No loop matching"):
            np.equal(1, 1, sig=(None, None, "l"))


class TestAdd:
    def test_reduce_alignment(self):
        # gh-9876
        # make sure arrays with weird strides work with the optimizations in
        # pairwise_sum_@TYPE@. On x86, the 'b' field will count as aligned at a
        # 4 byte offset, even though its itemsize is 8.
        a = np.zeros(2, dtype=[('a', np.int32), ('b', np.float64)])
        a['a'] = -1
        assert_equal(a['b'].sum(), 0)


class TestDivision:
    def test_division_int(self):
        # int division should follow Python
        x = np.array([5, 10, 90, 100, -5, -10, -90, -100, -120])
        if 5 / 10 == 0.5:
            assert_equal(x / 100, [0.05, 0.1, 0.9, 1,
                                   -0.05, -0.1, -0.9, -1, -1.2])
        else:
            assert_equal(x / 100, [0, 0, 0, 1, -1, -1, -1, -1, -2])
        assert_equal(x // 100, [0, 0, 0, 1, -1, -1, -1, -1, -2])
        assert_equal(x % 100, [5, 10, 90, 0, 95, 90, 10, 0, 80])

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize("dtype,ex_val", itertools.product(
        np.sctypes['int'] + np.sctypes['uint'], (
            (
                # dividend
                "np.array(range(fo.max-lsize, fo.max)).astype(dtype),"
                # divisors
                "np.arange(lsize).astype(dtype),"
                # scalar divisors
                "range(15)"
            ),
            (
                # dividend
                "np.arange(fo.min, fo.min+lsize).astype(dtype),"
                # divisors
                "np.arange(lsize//-2, lsize//2).astype(dtype),"
                # scalar divisors
                "range(fo.min, fo.min + 15)"
            ), (
                # dividend
                "np.array(range(fo.max-lsize, fo.max)).astype(dtype),"
                # divisors
                "np.arange(lsize).astype(dtype),"
                # scalar divisors
                "[1,3,9,13,neg, fo.min+1, fo.min//2, fo.max//3, fo.max//4]"
            )
        )
    ))
    def test_division_int_boundary(self, dtype, ex_val):
        fo = np.iinfo(dtype)
        neg = -1 if fo.min < 0 else 1
        # Large enough to test SIMD loops and remaind elements
        lsize = 512 + 7
        a, b, divisors = eval(ex_val)
        a_lst, b_lst = a.tolist(), b.tolist()

        c_div = lambda n, d: (
            0 if d == 0 else (
                fo.min if (n and n == fo.min and d == -1) else n//d
            )
        )
        with np.errstate(divide='ignore'):
            ac = a.copy()
            ac //= b
            div_ab = a // b
        div_lst = [c_div(x, y) for x, y in zip(a_lst, b_lst)]

        msg = "Integer arrays floor division check (//)"
        assert all(div_ab == div_lst), msg
        msg_eq = "Integer arrays floor division check (//=)"
        assert all(ac == div_lst), msg_eq

        for divisor in divisors:
            ac = a.copy()
            with np.errstate(divide='ignore', over='ignore'):
                div_a = a // divisor
                ac //= divisor
            div_lst = [c_div(i, divisor) for i in a_lst]

            assert all(div_a == div_lst), msg
            assert all(ac == div_lst), msg_eq

        with np.errstate(divide='raise', over='raise'):
            if 0 in b:
                # Verify overflow case
                with pytest.raises(FloatingPointError,
                        match="divide by zero encountered in floor_divide"):
                    a // b
            else:
                a // b
            if fo.min and fo.min in a:
                with pytest.raises(FloatingPointError,
                        match='overflow encountered in floor_divide'):
                    a // -1
            elif fo.min:
                a // -1
            with pytest.raises(FloatingPointError,
                    match="divide by zero encountered in floor_divide"):
                a // 0
            with pytest.raises(FloatingPointError,
                    match="divide by zero encountered in floor_divide"):
                ac = a.copy()
                ac //= 0

            np.array([], dtype=dtype) // 0

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize("dtype,ex_val", itertools.product(
        np.sctypes['int'] + np.sctypes['uint'], (
            "np.array([fo.max, 1, 2, 1, 1, 2, 3], dtype=dtype)",
            "np.array([fo.min, 1, -2, 1, 1, 2, -3]).astype(dtype)",
            "np.arange(fo.min, fo.min+(100*10), 10, dtype=dtype)",
            "np.array(range(fo.max-(100*7), fo.max, 7)).astype(dtype)",
        )
    ))
    def test_division_int_reduce(self, dtype, ex_val):
        fo = np.iinfo(dtype)
        a = eval(ex_val)
        lst = a.tolist()
        c_div = lambda n, d: (
            0 if d == 0 or (n and n == fo.min and d == -1) else n//d
        )

        with np.errstate(divide='ignore'):
            div_a = np.floor_divide.reduce(a)
        div_lst = reduce(c_div, lst)
        msg = "Reduce floor integer division check"
        assert div_a == div_lst, msg

        with np.errstate(divide='raise', over='raise'):
            with pytest.raises(FloatingPointError,
                    match="divide by zero encountered in reduce"):
                np.floor_divide.reduce(np.arange(-100, 100).astype(dtype))
            if fo.min:
                with pytest.raises(FloatingPointError,
                        match='overflow encountered in reduce'):
                    np.floor_divide.reduce(
                        np.array([fo.min, 1, -1], dtype=dtype)
                    )

    @pytest.mark.parametrize(
            "dividend,divisor,quotient",
            [(np.timedelta64(2,'Y'), np.timedelta64(2,'M'), 12),
             (np.timedelta64(2,'Y'), np.timedelta64(-2,'M'), -12),
             (np.timedelta64(-2,'Y'), np.timedelta64(2,'M'), -12),
             (np.timedelta64(-2,'Y'), np.timedelta64(-2,'M'), 12),
             (np.timedelta64(2,'M'), np.timedelta64(-2,'Y'), -1),
             (np.timedelta64(2,'Y'), np.timedelta64(0,'M'), 0),
             (np.timedelta64(2,'Y'), 2, np.timedelta64(1,'Y')),
             (np.timedelta64(2,'Y'), -2, np.timedelta64(-1,'Y')),
             (np.timedelta64(-2,'Y'), 2, np.timedelta64(-1,'Y')),
             (np.timedelta64(-2,'Y'), -2, np.timedelta64(1,'Y')),
             (np.timedelta64(-2,'Y'), -2, np.timedelta64(1,'Y')),
             (np.timedelta64(-2,'Y'), -3, np.timedelta64(0,'Y')),
             (np.timedelta64(-2,'Y'), 0, np.timedelta64('Nat','Y')),
            ])
    def test_division_int_timedelta(self, dividend, divisor, quotient):
        # If either divisor is 0 or quotient is Nat, check for division by 0
        if divisor and (isinstance(quotient, int) or not np.isnat(quotient)):
            msg = "Timedelta floor division check"
            assert dividend // divisor == quotient, msg

            # Test for arrays as well
            msg = "Timedelta arrays floor division check"
            dividend_array = np.array([dividend]*5)
            quotient_array = np.array([quotient]*5)
            assert all(dividend_array // divisor == quotient_array), msg
        else:
            if IS_WASM:
                pytest.skip("fp errors don't work in wasm")
            with np.errstate(divide='raise', invalid='raise'):
                with pytest.raises(FloatingPointError):
                    dividend // divisor

    def test_division_complex(self):
        # check that implementation is correct
        msg = "Complex division implementation check"
        x = np.array([1. + 1.*1j, 1. + .5*1j, 1. + 2.*1j], dtype=np.complex128)
        assert_almost_equal(x**2/x, x, err_msg=msg)
        # check overflow, underflow
        msg = "Complex division overflow/underflow check"
        x = np.array([1.e+110, 1.e-110], dtype=np.complex128)
        y = x**2/x
        assert_almost_equal(y/x, [1, 1], err_msg=msg)

    def test_zero_division_complex(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            x = np.array([0.0], dtype=np.complex128)
            y = 1.0/x
            assert_(np.isinf(y)[0])
            y = complex(np.inf, np.nan)/x
            assert_(np.isinf(y)[0])
            y = complex(np.nan, np.inf)/x
            assert_(np.isinf(y)[0])
            y = complex(np.inf, np.inf)/x
            assert_(np.isinf(y)[0])
            y = 0.0/x
            assert_(np.isnan(y)[0])

    def test_floor_division_complex(self):
        # check that floor division, divmod and remainder raises type errors
        x = np.array([.9 + 1j, -.1 + 1j, .9 + .5*1j, .9 + 2.*1j], dtype=np.complex128)
        with pytest.raises(TypeError):
            x // 7
        with pytest.raises(TypeError):
            np.divmod(x, 7)
        with pytest.raises(TypeError):
            np.remainder(x, 7)

    def test_floor_division_signed_zero(self):
        # Check that the sign bit is correctly set when dividing positive and
        # negative zero by one.
        x = np.zeros(10)
        assert_equal(np.signbit(x//1), 0)
        assert_equal(np.signbit((-x)//1), 1)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    def test_floor_division_errors(self, dtype):
        fnan = np.array(np.nan, dtype=dtype)
        fone = np.array(1.0, dtype=dtype)
        fzer = np.array(0.0, dtype=dtype)
        finf = np.array(np.inf, dtype=dtype)
        # divide by zero error check
        with np.errstate(divide='raise', invalid='ignore'):
            assert_raises(FloatingPointError, np.floor_divide, fone, fzer)
        with np.errstate(divide='ignore', invalid='raise'):
            np.floor_divide(fone, fzer)

        # The following already contain a NaN and should not warn
        with np.errstate(all='raise'):
            np.floor_divide(fnan, fone)
            np.floor_divide(fone, fnan)
            np.floor_divide(fnan, fzer)
            np.floor_divide(fzer, fnan)

    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    def test_floor_division_corner_cases(self, dtype):
        # test corner cases like 1.0//0.0 for errors and return vals
        x = np.zeros(10, dtype=dtype)
        y = np.ones(10, dtype=dtype)
        fnan = np.array(np.nan, dtype=dtype)
        fone = np.array(1.0, dtype=dtype)
        fzer = np.array(0.0, dtype=dtype)
        finf = np.array(np.inf, dtype=dtype)
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in floor_divide")
            div = np.floor_divide(fnan, fone)
            assert(np.isnan(div)), "dt: %s, div: %s" % (dt, div)
            div = np.floor_divide(fone, fnan)
            assert(np.isnan(div)), "dt: %s, div: %s" % (dt, div)
            div = np.floor_divide(fnan, fzer)
            assert(np.isnan(div)), "dt: %s, div: %s" % (dt, div)
        # verify 1.0//0.0 computations return inf
        with np.errstate(divide='ignore'):
            z = np.floor_divide(y, x)
            assert_(np.isinf(z).all())

def floor_divide_and_remainder(x, y):
    return (np.floor_divide(x, y), np.remainder(x, y))


def _signs(dt):
    if dt in np.typecodes['UnsignedInteger']:
        return (+1,)
    else:
        return (+1, -1)


class TestRemainder:

    def test_remainder_basic(self):
        dt = np.typecodes['AllInteger'] + np.typecodes['Float']
        for op in [floor_divide_and_remainder, np.divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product(_signs(dt1), _signs(dt2)):
                    fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    a = np.array(sg1*71, dtype=dt1)
                    b = np.array(sg2*19, dtype=dt2)
                    div, rem = op(a, b)
                    assert_equal(div*b + rem, a, err_msg=msg)
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    def test_float_remainder_exact(self):
        # test that float results are exact for small integers. This also
        # holds for the same integers scaled by powers of two.
        nlst = list(range(-127, 0))
        plst = list(range(1, 128))
        dividend = nlst + [0] + plst
        divisor = nlst + plst
        arg = list(itertools.product(dividend, divisor))
        tgt = list(divmod(*t) for t in arg)

        a, b = np.array(arg, dtype=int).T
        # convert exact integer results from Python to float so that
        # signed zero can be used, it is checked.
        tgtdiv, tgtrem = np.array(tgt, dtype=float).T
        tgtdiv = np.where((tgtdiv == 0.0) & ((b < 0) ^ (a < 0)), -0.0, tgtdiv)
        tgtrem = np.where((tgtrem == 0.0) & (b < 0), -0.0, tgtrem)

        for op in [floor_divide_and_remainder, np.divmod]:
            for dt in np.typecodes['Float']:
                msg = 'op: %s, dtype: %s' % (op.__name__, dt)
                fa = a.astype(dt)
                fb = b.astype(dt)
                div, rem = op(fa, fb)
                assert_equal(div, tgtdiv, err_msg=msg)
                assert_equal(rem, tgtrem, err_msg=msg)

    def test_float_remainder_roundoff(self):
        # gh-6127
        dt = np.typecodes['Float']
        for op in [floor_divide_and_remainder, np.divmod]:
            for dt1, dt2 in itertools.product(dt, dt):
                for sg1, sg2 in itertools.product((+1, -1), (+1, -1)):
                    fmt = 'op: %s, dt1: %s, dt2: %s, sg1: %s, sg2: %s'
                    msg = fmt % (op.__name__, dt1, dt2, sg1, sg2)
                    a = np.array(sg1*78*6e-8, dtype=dt1)
                    b = np.array(sg2*6e-8, dtype=dt2)
                    div, rem = op(a, b)
                    # Equal assertion should hold when fmod is used
                    assert_equal(div*b + rem, a, err_msg=msg)
                    if sg2 == -1:
                        assert_(b < rem <= 0, msg)
                    else:
                        assert_(b > rem >= 0, msg)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.xfail(sys.platform.startswith("darwin"),
            reason="MacOS seems to not give the correct 'invalid' warning for "
                   "`fmod`.  Hopefully, others always do.")
    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    def test_float_divmod_errors(self, dtype):
        # Check valid errors raised for divmod and remainder
        fzero = np.array(0.0, dtype=dtype)
        fone = np.array(1.0, dtype=dtype)
        finf = np.array(np.inf, dtype=dtype)
        fnan = np.array(np.nan, dtype=dtype)
        # since divmod is combination of both remainder and divide
        # ops it will set both dividebyzero and invalid flags
        with np.errstate(divide='raise', invalid='ignore'):
            assert_raises(FloatingPointError, np.divmod, fone, fzero)
        with np.errstate(divide='ignore', invalid='raise'):
            assert_raises(FloatingPointError, np.divmod, fone, fzero)
        with np.errstate(invalid='raise'):
            assert_raises(FloatingPointError, np.divmod, fzero, fzero)
        with np.errstate(invalid='raise'):
            assert_raises(FloatingPointError, np.divmod, finf, finf)
        with np.errstate(divide='ignore', invalid='raise'):
            assert_raises(FloatingPointError, np.divmod, finf, fzero)
        with np.errstate(divide='raise', invalid='ignore'):
            # inf / 0 does not set any flags, only the modulo creates a NaN
            np.divmod(finf, fzero)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.xfail(sys.platform.startswith("darwin"),
           reason="MacOS seems to not give the correct 'invalid' warning for "
                  "`fmod`.  Hopefully, others always do.")
    @pytest.mark.parametrize('dtype', np.typecodes['Float'])
    @pytest.mark.parametrize('fn', [np.fmod, np.remainder])
    def test_float_remainder_errors(self, dtype, fn):
        fzero = np.array(0.0, dtype=dtype)
        fone = np.array(1.0, dtype=dtype)
        finf = np.array(np.inf, dtype=dtype)
        fnan = np.array(np.nan, dtype=dtype)

        # The following already contain a NaN and should not warn.
        with np.errstate(all='raise'):
            with pytest.raises(FloatingPointError,
                    match="invalid value"):
                fn(fone, fzero)
            fn(fnan, fzero)
            fn(fzero, fnan)
            fn(fone, fnan)
            fn(fnan, fone)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_float_remainder_overflow(self):
        a = np.finfo(np.float64).tiny
        with np.errstate(over='ignore', invalid='ignore'):
            div, mod = np.divmod(4, a)
            np.isinf(div)
            assert_(mod == 0)
        with np.errstate(over='raise', invalid='ignore'):
            assert_raises(FloatingPointError, np.divmod, 4, a)
        with np.errstate(invalid='raise', over='ignore'):
            assert_raises(FloatingPointError, np.divmod, 4, a)

    def test_float_divmod_corner_cases(self):
        # check nan cases
        for dt in np.typecodes['Float']:
            fnan = np.array(np.nan, dtype=dt)
            fone = np.array(1.0, dtype=dt)
            fzer = np.array(0.0, dtype=dt)
            finf = np.array(np.inf, dtype=dt)
            with suppress_warnings() as sup:
                sup.filter(RuntimeWarning, "invalid value encountered in divmod")
                sup.filter(RuntimeWarning, "divide by zero encountered in divmod")
                div, rem = np.divmod(fone, fzer)
                assert(np.isinf(div)), 'dt: %s, div: %s' % (dt, rem)
                assert(np.isnan(rem)), 'dt: %s, rem: %s' % (dt, rem)
                div, rem = np.divmod(fzer, fzer)
                assert(np.isnan(rem)), 'dt: %s, rem: %s' % (dt, rem)
                assert_(np.isnan(div)), 'dt: %s, rem: %s' % (dt, rem)
                div, rem = np.divmod(finf, finf)
                assert(np.isnan(div)), 'dt: %s, rem: %s' % (dt, rem)
                assert(np.isnan(rem)), 'dt: %s, rem: %s' % (dt, rem)
                div, rem = np.divmod(finf, fzer)
                assert(np.isinf(div)), 'dt: %s, rem: %s' % (dt, rem)
                assert(np.isnan(rem)), 'dt: %s, rem: %s' % (dt, rem)
                div, rem = np.divmod(fnan, fone)
                assert(np.isnan(rem)), "dt: %s, rem: %s" % (dt, rem)
                assert(np.isnan(div)), "dt: %s, rem: %s" % (dt, rem)
                div, rem = np.divmod(fone, fnan)
                assert(np.isnan(rem)), "dt: %s, rem: %s" % (dt, rem)
                assert(np.isnan(div)), "dt: %s, rem: %s" % (dt, rem)
                div, rem = np.divmod(fnan, fzer)
                assert(np.isnan(rem)), "dt: %s, rem: %s" % (dt, rem)
                assert(np.isnan(div)), "dt: %s, rem: %s" % (dt, rem)

    def test_float_remainder_corner_cases(self):
        # Check remainder magnitude.
        for dt in np.typecodes['Float']:
            fone = np.array(1.0, dtype=dt)
            fzer = np.array(0.0, dtype=dt)
            fnan = np.array(np.nan, dtype=dt)
            b = np.array(1.0, dtype=dt)
            a = np.nextafter(np.array(0.0, dtype=dt), -b)
            rem = np.remainder(a, b)
            assert_(rem <= b, 'dt: %s' % dt)
            rem = np.remainder(-a, -b)
            assert_(rem >= -b, 'dt: %s' % dt)

        # Check nans, inf
        with suppress_warnings() as sup:
            sup.filter(RuntimeWarning, "invalid value encountered in remainder")
            sup.filter(RuntimeWarning, "invalid value encountered in fmod")
            for dt in np.typecodes['Float']:
                fone = np.array(1.0, dtype=dt)
                fzer = np.array(0.0, dtype=dt)
                finf = np.array(np.inf, dtype=dt)
                fnan = np.array(np.nan, dtype=dt)
                rem = np.remainder(fone, fzer)
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                # MSVC 2008 returns NaN here, so disable the check.
                #rem = np.remainder(fone, finf)
                #assert_(rem == fone, 'dt: %s, rem: %s' % (dt, rem))
                rem = np.remainder(finf, fone)
                fmod = np.fmod(finf, fone)
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, fmod))
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                rem = np.remainder(finf, finf)
                fmod = np.fmod(finf, fone)
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, fmod))
                rem = np.remainder(finf, fzer)
                fmod = np.fmod(finf, fzer)
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, fmod))
                rem = np.remainder(fone, fnan)
                fmod = np.fmod(fone, fnan)
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, fmod))
                rem = np.remainder(fnan, fzer)
                fmod = np.fmod(fnan, fzer)
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, rem))
                rem = np.remainder(fnan, fone)
                fmod = np.fmod(fnan, fone)
                assert_(np.isnan(rem), 'dt: %s, rem: %s' % (dt, rem))
                assert_(np.isnan(fmod), 'dt: %s, fmod: %s' % (dt, rem))


class TestDivisionIntegerOverflowsAndDivideByZero:
    result_type = namedtuple('result_type',
            ['nocast', 'casted'])
    helper_lambdas = {
        'zero': lambda dtype: 0,
        'min': lambda dtype: np.iinfo(dtype).min,
        'neg_min': lambda dtype: -np.iinfo(dtype).min,
        'min-zero': lambda dtype: (np.iinfo(dtype).min, 0),
        'neg_min-zero': lambda dtype: (-np.iinfo(dtype).min, 0),
    }
    overflow_results = {
        np.remainder: result_type(
            helper_lambdas['zero'], helper_lambdas['zero']),
        np.fmod: result_type(
            helper_lambdas['zero'], helper_lambdas['zero']),
        operator.mod: result_type(
            helper_lambdas['zero'], helper_lambdas['zero']),
        operator.floordiv: result_type(
            helper_lambdas['min'], helper_lambdas['neg_min']),
        np.floor_divide: result_type(
            helper_lambdas['min'], helper_lambdas['neg_min']),
        np.divmod: result_type(
            helper_lambdas['min-zero'], helper_lambdas['neg_min-zero'])
    }

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize("dtype", np.typecodes["Integer"])
    def test_signed_division_overflow(self, dtype):
        to_check = interesting_binop_operands(np.iinfo(dtype).min, -1, dtype)
        for op1, op2, extractor, operand_identifier in to_check:
            with pytest.warns(RuntimeWarning, match="overflow encountered"):
                res = op1 // op2

            assert res.dtype == op1.dtype
            assert extractor(res) == np.iinfo(op1.dtype).min

            # Remainder is well defined though, and does not warn:
            res = op1 % op2
            assert res.dtype == op1.dtype
            assert extractor(res) == 0
            # Check fmod as well:
            res = np.fmod(op1, op2)
            assert extractor(res) == 0

            # Divmod warns for the division part:
            with pytest.warns(RuntimeWarning, match="overflow encountered"):
                res1, res2 = np.divmod(op1, op2)

            assert res1.dtype == res2.dtype == op1.dtype
            assert extractor(res1) == np.iinfo(op1.dtype).min
            assert extractor(res2) == 0

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
    def test_divide_by_zero(self, dtype):
        # Note that the return value cannot be well defined here, but NumPy
        # currently uses 0 consistently.  This could be changed.
        to_check = interesting_binop_operands(1, 0, dtype)
        for op1, op2, extractor, operand_identifier in to_check:
            with pytest.warns(RuntimeWarning, match="divide by zero"):
                res = op1 // op2

            assert res.dtype == op1.dtype
            assert extractor(res) == 0

            with pytest.warns(RuntimeWarning, match="divide by zero"):
                res1, res2 = np.divmod(op1, op2)

            assert res1.dtype == res2.dtype == op1.dtype
            assert extractor(res1) == 0
            assert extractor(res2) == 0

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize("dividend_dtype",
            np.sctypes['int'])
    @pytest.mark.parametrize("divisor_dtype",
            np.sctypes['int'])
    @pytest.mark.parametrize("operation",
            [np.remainder, np.fmod, np.divmod, np.floor_divide,
             operator.mod, operator.floordiv])
    @np.errstate(divide='warn', over='warn')
    def test_overflows(self, dividend_dtype, divisor_dtype, operation):
        # SIMD tries to perform the operation on as many elements as possible
        # that is a multiple of the register's size. We resort to the
        # default implementation for the leftover elements.
        # We try to cover all paths here.
        arrays = [np.array([np.iinfo(dividend_dtype).min]*i,
                           dtype=dividend_dtype) for i in range(1, 129)]
        divisor = np.array([-1], dtype=divisor_dtype)
        # If dividend is a larger type than the divisor (`else` case),
        # then, result will be a larger type than dividend and will not
        # result in an overflow for `divmod` and `floor_divide`.
        if np.dtype(dividend_dtype).itemsize >= np.dtype(
                divisor_dtype).itemsize and operation in (
                        np.divmod, np.floor_divide, operator.floordiv):
            with pytest.warns(
                    RuntimeWarning,
                    match="overflow encountered in"):
                result = operation(
                            dividend_dtype(np.iinfo(dividend_dtype).min),
                            divisor_dtype(-1)
                        )
                assert result == self.overflow_results[operation].nocast(
                        dividend_dtype)

            # Arrays
            for a in arrays:
                # In case of divmod, we need to flatten the result
                # column first as we get a column vector of quotient and
                # remainder and a normal flatten of the expected result.
                with pytest.warns(
                        RuntimeWarning,
                        match="overflow encountered in"):
                    result = np.array(operation(a, divisor)).flatten('f')
                    expected_array = np.array(
                            [self.overflow_results[operation].nocast(
                                dividend_dtype)]*len(a)).flatten()
                    assert_array_equal(result, expected_array)
        else:
            # Scalars
            result = operation(
                        dividend_dtype(np.iinfo(dividend_dtype).min),
                        divisor_dtype(-1)
                    )
            assert result == self.overflow_results[operation].casted(
                    dividend_dtype)

            # Arrays
            for a in arrays:
                # See above comment on flatten
                result = np.array(operation(a, divisor)).flatten('f')
                expected_array = np.array(
                        [self.overflow_results[operation].casted(
                            dividend_dtype)]*len(a)).flatten()
                assert_array_equal(result, expected_array)


class TestCbrt:
    def test_cbrt_scalar(self):
        assert_almost_equal((np.cbrt(np.float32(-2.5)**3)), -2.5)

    def test_cbrt(self):
        x = np.array([1., 2., -3., np.inf, -np.inf])
        assert_almost_equal(np.cbrt(x**3), x)

        assert_(np.isnan(np.cbrt(np.nan)))
        assert_equal(np.cbrt(np.inf), np.inf)
        assert_equal(np.cbrt(-np.inf), -np.inf)


class TestPower:
    def test_power_float(self):
        x = np.array([1., 2., 3.])
        assert_equal(x**0, [1., 1., 1.])
        assert_equal(x**1, x)
        assert_equal(x**2, [1., 4., 9.])
        y = x.copy()
        y **= 2
        assert_equal(y, [1., 4., 9.])
        assert_almost_equal(x**(-1), [1., 0.5, 1./3])
        assert_almost_equal(x**(0.5), [1., ncu.sqrt(2), ncu.sqrt(3)])

        for out, inp, msg in _gen_alignment_data(dtype=np.float32,
                                                 type='unary',
                                                 max_size=11):
            exp = [ncu.sqrt(i) for i in inp]
            assert_almost_equal(inp**(0.5), exp, err_msg=msg)
            np.sqrt(inp, out=out)
            assert_equal(out, exp, err_msg=msg)

        for out, inp, msg in _gen_alignment_data(dtype=np.float64,
                                                 type='unary',
                                                 max_size=7):
            exp = [ncu.sqrt(i) for i in inp]
            assert_almost_equal(inp**(0.5), exp, err_msg=msg)
            np.sqrt(inp, out=out)
            assert_equal(out, exp, err_msg=msg)

    def test_power_complex(self):
        x = np.array([1+2j, 2+3j, 3+4j])
        assert_equal(x**0, [1., 1., 1.])
        assert_equal(x**1, x)
        assert_almost_equal(x**2, [-3+4j, -5+12j, -7+24j])
        assert_almost_equal(x**3, [(1+2j)**3, (2+3j)**3, (3+4j)**3])
        assert_almost_equal(x**4, [(1+2j)**4, (2+3j)**4, (3+4j)**4])
        assert_almost_equal(x**(-1), [1/(1+2j), 1/(2+3j), 1/(3+4j)])
        assert_almost_equal(x**(-2), [1/(1+2j)**2, 1/(2+3j)**2, 1/(3+4j)**2])
        assert_almost_equal(x**(-3), [(-11+2j)/125, (-46-9j)/2197,
                                      (-117-44j)/15625])
        assert_almost_equal(x**(0.5), [ncu.sqrt(1+2j), ncu.sqrt(2+3j),
                                       ncu.sqrt(3+4j)])
        norm = 1./((x**14)[0])
        assert_almost_equal(x**14 * norm,
                [i * norm for i in [-76443+16124j, 23161315+58317492j,
                                    5583548873 + 2465133864j]])

        # Ticket #836
        def assert_complex_equal(x, y):
            assert_array_equal(x.real, y.real)
            assert_array_equal(x.imag, y.imag)

        for z in [complex(0, np.inf), complex(1, np.inf)]:
            z = np.array([z], dtype=np.complex_)
            with np.errstate(invalid="ignore"):
                assert_complex_equal(z**1, z)
                assert_complex_equal(z**2, z*z)
                assert_complex_equal(z**3, z*z*z)

    def test_power_zero(self):
        # ticket #1271
        zero = np.array([0j])
        one = np.array([1+0j])
        cnan = np.array([complex(np.nan, np.nan)])
        # FIXME cinf not tested.
        #cinf = np.array([complex(np.inf, 0)])

        def assert_complex_equal(x, y):
            x, y = np.asarray(x), np.asarray(y)
            assert_array_equal(x.real, y.real)
            assert_array_equal(x.imag, y.imag)

        # positive powers
        for p in [0.33, 0.5, 1, 1.5, 2, 3, 4, 5, 6.6]:
            assert_complex_equal(np.power(zero, p), zero)

        # zero power
        assert_complex_equal(np.power(zero, 0), one)
        with np.errstate(invalid="ignore"):
            assert_complex_equal(np.power(zero, 0+1j), cnan)

            # negative power
            for p in [0.33, 0.5, 1, 1.5, 2, 3, 4, 5, 6.6]:
                assert_complex_equal(np.power(zero, -p), cnan)
            assert_complex_equal(np.power(zero, -1+0.2j), cnan)

    def test_fast_power(self):
        x = np.array([1, 2, 3], np.int16)
        res = x**2.0
        assert_((x**2.00001).dtype is res.dtype)
        assert_array_equal(res, [1, 4, 9])
        # check the inplace operation on the casted copy doesn't mess with x
        assert_(not np.may_share_memory(res, x))
        assert_array_equal(x, [1, 2, 3])

        # Check that the fast path ignores 1-element not 0-d arrays
        res = x ** np.array([[[2]]])
        assert_equal(res.shape, (1, 1, 3))

    def test_integer_power(self):
        a = np.array([15, 15], 'i8')
        b = np.power(a, a)
        assert_equal(b, [437893890380859375, 437893890380859375])

    def test_integer_power_with_integer_zero_exponent(self):
        dtypes = np.typecodes['Integer']
        for dt in dtypes:
            arr = np.arange(-10, 10, dtype=dt)
            assert_equal(np.power(arr, 0), np.ones_like(arr))

        dtypes = np.typecodes['UnsignedInteger']
        for dt in dtypes:
            arr = np.arange(10, dtype=dt)
            assert_equal(np.power(arr, 0), np.ones_like(arr))

    def test_integer_power_of_1(self):
        dtypes = np.typecodes['AllInteger']
        for dt in dtypes:
            arr = np.arange(10, dtype=dt)
            assert_equal(np.power(1, arr), np.ones_like(arr))

    def test_integer_power_of_zero(self):
        dtypes = np.typecodes['AllInteger']
        for dt in dtypes:
            arr = np.arange(1, 10, dtype=dt)
            assert_equal(np.power(0, arr), np.zeros_like(arr))

    def test_integer_to_negative_power(self):
        dtypes = np.typecodes['Integer']
        for dt in dtypes:
            a = np.array([0, 1, 2, 3], dtype=dt)
            b = np.array([0, 1, 2, -3], dtype=dt)
            one = np.array(1, dtype=dt)
            minusone = np.array(-1, dtype=dt)
            assert_raises(ValueError, np.power, a, b)
            assert_raises(ValueError, np.power, a, minusone)
            assert_raises(ValueError, np.power, one, b)
            assert_raises(ValueError, np.power, one, minusone)

    def test_float_to_inf_power(self):
        for dt in [np.float32, np.float64]:
            a = np.array([1, 1, 2, 2, -2, -2, np.inf, -np.inf], dt)
            b = np.array([np.inf, -np.inf, np.inf, -np.inf,
                                np.inf, -np.inf, np.inf, -np.inf], dt)
            r = np.array([1, 1, np.inf, 0, np.inf, 0, np.inf, 0], dt)
            assert_equal(np.power(a, b), r)


class TestFloat_power:
    def test_type_conversion(self):
        arg_type = '?bhilBHILefdgFDG'
        res_type = 'ddddddddddddgDDG'
        for dtin, dtout in zip(arg_type, res_type):
            msg = "dtin: %s, dtout: %s" % (dtin, dtout)
            arg = np.ones(1, dtype=dtin)
            res = np.float_power(arg, arg)
            assert_(res.dtype.name == np.dtype(dtout).name, msg)


class TestLog2:
    @pytest.mark.parametrize('dt', ['f', 'd', 'g'])
    def test_log2_values(self, dt):
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        xf = np.array(x, dtype=dt)
        yf = np.array(y, dtype=dt)
        assert_almost_equal(np.log2(xf), yf)

    @pytest.mark.parametrize("i", range(1, 65))
    def test_log2_ints(self, i):
        # a good log2 implementation should provide this,
        # might fail on OS with bad libm
        v = np.log2(2.**i)
        assert_equal(v, float(i), err_msg='at exponent %d' % i)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_log2_special(self):
        assert_equal(np.log2(1.), 0.)
        assert_equal(np.log2(np.inf), np.inf)
        assert_(np.isnan(np.log2(np.nan)))

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_(np.isnan(np.log2(-1.)))
            assert_(np.isnan(np.log2(-np.inf)))
            assert_equal(np.log2(0.), -np.inf)
            assert_(w[0].category is RuntimeWarning)
            assert_(w[1].category is RuntimeWarning)
            assert_(w[2].category is RuntimeWarning)


class TestExp2:
    def test_exp2_values(self):
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for dt in ['f', 'd', 'g']:
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt)
            assert_almost_equal(np.exp2(yf), xf)


class TestLogAddExp2(_FilterInvalids):
    # Need test for intermediate precisions
    def test_logaddexp2_values(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        z = [6, 6, 6, 6, 6]
        for dt, dec_ in zip(['f', 'd', 'g'], [6, 15, 15]):
            xf = np.log2(np.array(x, dtype=dt))
            yf = np.log2(np.array(y, dtype=dt))
            zf = np.log2(np.array(z, dtype=dt))
            assert_almost_equal(np.logaddexp2(xf, yf), zf, decimal=dec_)

    def test_logaddexp2_range(self):
        x = [1000000, -1000000, 1000200, -1000200]
        y = [1000200, -1000200, 1000000, -1000000]
        z = [1000200, -1000000, 1000200, -1000000]
        for dt in ['f', 'd', 'g']:
            logxf = np.array(x, dtype=dt)
            logyf = np.array(y, dtype=dt)
            logzf = np.array(z, dtype=dt)
            assert_almost_equal(np.logaddexp2(logxf, logyf), logzf)

    def test_inf(self):
        inf = np.inf
        x = [inf, -inf,  inf, -inf, inf, 1,  -inf,  1]
        y = [inf,  inf, -inf, -inf, 1,   inf, 1,   -inf]
        z = [inf,  inf,  inf, -inf, inf, inf, 1,    1]
        with np.errstate(invalid='raise'):
            for dt in ['f', 'd', 'g']:
                logxf = np.array(x, dtype=dt)
                logyf = np.array(y, dtype=dt)
                logzf = np.array(z, dtype=dt)
                assert_equal(np.logaddexp2(logxf, logyf), logzf)

    def test_nan(self):
        assert_(np.isnan(np.logaddexp2(np.nan, np.inf)))
        assert_(np.isnan(np.logaddexp2(np.inf, np.nan)))
        assert_(np.isnan(np.logaddexp2(np.nan, 0)))
        assert_(np.isnan(np.logaddexp2(0, np.nan)))
        assert_(np.isnan(np.logaddexp2(np.nan, np.nan)))

    def test_reduce(self):
        assert_equal(np.logaddexp2.identity, -np.inf)
        assert_equal(np.logaddexp2.reduce([]), -np.inf)
        assert_equal(np.logaddexp2.reduce([-np.inf]), -np.inf)
        assert_equal(np.logaddexp2.reduce([-np.inf, 0]), 0)


class TestLog:
    def test_log_values(self):
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for dt in ['f', 'd', 'g']:
            log2_ = 0.69314718055994530943
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt)*log2_
            assert_almost_equal(np.log(xf), yf)

        # test aliasing(issue #17761)
        x = np.array([2, 0.937500, 3, 0.947500, 1.054697])
        xf = np.log(x)
        assert_almost_equal(np.log(x, out=x), xf)

        # test log() of max for dtype does not raise
        for dt in ['f', 'd', 'g']:
            with np.errstate(all='raise'):
                x = np.finfo(dt).max
                np.log(x)

    def test_log_strides(self):
        np.random.seed(42)
        strides = np.array([-4,-3,-2,-1,1,2,3,4])
        sizes = np.arange(2,100)
        for ii in sizes:
            x_f64 = np.float64(np.random.uniform(low=0.01, high=100.0,size=ii))
            x_special = x_f64.copy()
            x_special[3:-1:4] = 1.0
            y_true = np.log(x_f64)
            y_special = np.log(x_special)
            for jj in strides:
                assert_array_almost_equal_nulp(np.log(x_f64[::jj]), y_true[::jj], nulp=2)
                assert_array_almost_equal_nulp(np.log(x_special[::jj]), y_special[::jj], nulp=2)

class TestExp:
    def test_exp_values(self):
        x = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        for dt in ['f', 'd', 'g']:
            log2_ = 0.69314718055994530943
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt)*log2_
            assert_almost_equal(np.exp(yf), xf)

    def test_exp_strides(self):
        np.random.seed(42)
        strides = np.array([-4,-3,-2,-1,1,2,3,4])
        sizes = np.arange(2,100)
        for ii in sizes:
            x_f64 = np.float64(np.random.uniform(low=0.01, high=709.1,size=ii))
            y_true = np.exp(x_f64)
            for jj in strides:
                assert_array_almost_equal_nulp(np.exp(x_f64[::jj]), y_true[::jj], nulp=2)

class TestSpecialFloats:
    def test_exp_values(self):
        with np.errstate(under='raise', over='raise'):
            x = [np.nan,  np.nan, np.inf, 0.]
            y = [np.nan, -np.nan, np.inf, -np.inf]
            for dt in ['e', 'f', 'd', 'g']:
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                assert_equal(np.exp(yf), xf)

    # See: https://github.com/numpy/numpy/issues/19192
    @pytest.mark.xfail(
        _glibc_older_than("2.17"),
        reason="Older glibc versions may not raise appropriate FP exceptions"
    )
    def test_exp_exceptions(self):
        with np.errstate(over='raise'):
            assert_raises(FloatingPointError, np.exp, np.float16(11.0899))
            assert_raises(FloatingPointError, np.exp, np.float32(100.))
            assert_raises(FloatingPointError, np.exp, np.float32(1E19))
            assert_raises(FloatingPointError, np.exp, np.float64(800.))
            assert_raises(FloatingPointError, np.exp, np.float64(1E19))

        with np.errstate(under='raise'):
            assert_raises(FloatingPointError, np.exp, np.float16(-17.5))
            assert_raises(FloatingPointError, np.exp, np.float32(-1000.))
            assert_raises(FloatingPointError, np.exp, np.float32(-1E19))
            assert_raises(FloatingPointError, np.exp, np.float64(-1000.))
            assert_raises(FloatingPointError, np.exp, np.float64(-1E19))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_log_values(self):
        with np.errstate(all='ignore'):
            x = [np.nan, np.nan, np.inf, np.nan, -np.inf, np.nan]
            y = [np.nan, -np.nan, np.inf, -np.inf, 0.0, -1.0]
            y1p = [np.nan, -np.nan, np.inf, -np.inf, -1.0, -2.0]
            for dt in ['e', 'f', 'd', 'g']:
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                yf1p = np.array(y1p, dtype=dt)
                assert_equal(np.log(yf), xf)
                assert_equal(np.log2(yf), xf)
                assert_equal(np.log10(yf), xf)
                assert_equal(np.log1p(yf1p), xf)

        with np.errstate(divide='raise'):
            for dt in ['e', 'f', 'd']:
                assert_raises(FloatingPointError, np.log,
                              np.array(0.0, dtype=dt))
                assert_raises(FloatingPointError, np.log2,
                              np.array(0.0, dtype=dt))
                assert_raises(FloatingPointError, np.log10,
                              np.array(0.0, dtype=dt))
                assert_raises(FloatingPointError, np.log1p,
                              np.array(-1.0, dtype=dt))

        with np.errstate(invalid='raise'):
            for dt in ['e', 'f', 'd']:
                assert_raises(FloatingPointError, np.log,
                              np.array(-np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.log,
                              np.array(-1.0, dtype=dt))
                assert_raises(FloatingPointError, np.log2,
                              np.array(-np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.log2,
                              np.array(-1.0, dtype=dt))
                assert_raises(FloatingPointError, np.log10,
                              np.array(-np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.log10,
                              np.array(-1.0, dtype=dt))
                assert_raises(FloatingPointError, np.log1p,
                              np.array(-np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.log1p,
                              np.array(-2.0, dtype=dt))

        # See https://github.com/numpy/numpy/issues/18005
        with assert_no_warnings():
            a = np.array(1e9, dtype='float32')
            np.log(a)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_sincos_values(self):
        with np.errstate(all='ignore'):
            x = [np.nan, np.nan, np.nan, np.nan]
            y = [np.nan, -np.nan, np.inf, -np.inf]
            for dt in ['e', 'f', 'd', 'g']:
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                assert_equal(np.sin(yf), xf)
                assert_equal(np.cos(yf), xf)


        with np.errstate(invalid='raise'):
            for callable in [np.sin, np.cos]:
                for value in [np.inf, -np.inf]:
                    for dt in ['e', 'f', 'd']:
                        assert_raises(FloatingPointError, callable,
                                np.array([value], dtype=dt))

    @pytest.mark.parametrize('dt', ['e', 'f', 'd', 'g'])
    def test_sqrt_values(self, dt):
        with np.errstate(all='ignore'):
            x = [np.nan, np.nan, np.inf, np.nan, 0.]
            y = [np.nan, -np.nan, np.inf, -np.inf, 0.]
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt)
            assert_equal(np.sqrt(yf), xf)

        # with np.errstate(invalid='raise'):
        #     assert_raises(
        #         FloatingPointError, np.sqrt, np.array(-100., dtype=dt)
        #     )

    def test_abs_values(self):
        x = [np.nan,  np.nan, np.inf, np.inf, 0., 0., 1.0, 1.0]
        y = [np.nan, -np.nan, np.inf, -np.inf, 0., -0., -1.0, 1.0]
        for dt in ['e', 'f', 'd', 'g']:
            xf = np.array(x, dtype=dt)
            yf = np.array(y, dtype=dt)
            assert_equal(np.abs(yf), xf)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_square_values(self):
        x = [np.nan,  np.nan, np.inf, np.inf]
        y = [np.nan, -np.nan, np.inf, -np.inf]
        with np.errstate(all='ignore'):
            for dt in ['e', 'f', 'd', 'g']:
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                assert_equal(np.square(yf), xf)

        with np.errstate(over='raise'):
            assert_raises(FloatingPointError, np.square,
                          np.array(1E3, dtype='e'))
            assert_raises(FloatingPointError, np.square,
                          np.array(1E32, dtype='f'))
            assert_raises(FloatingPointError, np.square,
                          np.array(1E200, dtype='d'))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_reciprocal_values(self):
        with np.errstate(all='ignore'):
            x = [np.nan,  np.nan, 0.0, -0.0, np.inf, -np.inf]
            y = [np.nan, -np.nan, np.inf, -np.inf, 0., -0.]
            for dt in ['e', 'f', 'd', 'g']:
                xf = np.array(x, dtype=dt)
                yf = np.array(y, dtype=dt)
                assert_equal(np.reciprocal(yf), xf)

        with np.errstate(divide='raise'):
            for dt in ['e', 'f', 'd', 'g']:
                assert_raises(FloatingPointError, np.reciprocal,
                              np.array(-0.0, dtype=dt))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_tan(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, 0.0, -0.0, np.inf, -np.inf]
            out = [np.nan, np.nan, 0.0, -0.0, np.nan, np.nan]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.tan(in_arr), out_arr)

        with np.errstate(invalid='raise'):
            for dt in ['e', 'f', 'd']:
                assert_raises(FloatingPointError, np.tan,
                              np.array(np.inf, dtype=dt))
                assert_raises(FloatingPointError, np.tan,
                              np.array(-np.inf, dtype=dt))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_arcsincos(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, np.inf, -np.inf]
            out = [np.nan, np.nan, np.nan, np.nan]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.arcsin(in_arr), out_arr)
                assert_equal(np.arccos(in_arr), out_arr)

        for callable in [np.arcsin, np.arccos]:
            for value in [np.inf, -np.inf, 2.0, -2.0]:
                for dt in ['e', 'f', 'd']:
                    with np.errstate(invalid='raise'):
                        assert_raises(FloatingPointError, callable,
                                      np.array(value, dtype=dt))

    def test_arctan(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan]
            out = [np.nan, np.nan]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.arctan(in_arr), out_arr)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_sinh(self):
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        out = [np.nan, np.nan, np.inf, -np.inf]
        for dt in ['e', 'f', 'd']:
            in_arr = np.array(in_, dtype=dt)
            out_arr = np.array(out, dtype=dt)
            assert_equal(np.sinh(in_arr), out_arr)

        with np.errstate(over='raise'):
            assert_raises(FloatingPointError, np.sinh,
                          np.array(12.0, dtype='e'))
            assert_raises(FloatingPointError, np.sinh,
                          np.array(120.0, dtype='f'))
            assert_raises(FloatingPointError, np.sinh,
                          np.array(1200.0, dtype='d'))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_cosh(self):
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        out = [np.nan, np.nan, np.inf, np.inf]
        for dt in ['e', 'f', 'd']:
            in_arr = np.array(in_, dtype=dt)
            out_arr = np.array(out, dtype=dt)
            assert_equal(np.cosh(in_arr), out_arr)

        with np.errstate(over='raise'):
            assert_raises(FloatingPointError, np.cosh,
                          np.array(12.0, dtype='e'))
            assert_raises(FloatingPointError, np.cosh,
                          np.array(120.0, dtype='f'))
            assert_raises(FloatingPointError, np.cosh,
                          np.array(1200.0, dtype='d'))

    def test_tanh(self):
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        out = [np.nan, np.nan, 1.0, -1.0]
        for dt in ['e', 'f', 'd']:
            in_arr = np.array(in_, dtype=dt)
            out_arr = np.array(out, dtype=dt)
            assert_equal(np.tanh(in_arr), out_arr)

    def test_arcsinh(self):
        in_ = [np.nan, -np.nan, np.inf, -np.inf]
        out = [np.nan, np.nan, np.inf, -np.inf]
        for dt in ['e', 'f', 'd']:
            in_arr = np.array(in_, dtype=dt)
            out_arr = np.array(out, dtype=dt)
            assert_equal(np.arcsinh(in_arr), out_arr)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_arccosh(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, np.inf, -np.inf, 1.0, 0.0]
            out = [np.nan, np.nan, np.inf, np.nan, 0.0, np.nan]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.arccosh(in_arr), out_arr)

        for value in [0.0, -np.inf]:
            with np.errstate(invalid='raise'):
                for dt in ['e', 'f', 'd']:
                    assert_raises(FloatingPointError, np.arccosh,
                                  np.array(value, dtype=dt))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_arctanh(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, np.inf, -np.inf, 1.0, -1.0, 2.0]
            out = [np.nan, np.nan, np.nan, np.nan, np.inf, -np.inf, np.nan]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.arctanh(in_arr), out_arr)

        for value in [1.01, np.inf, -np.inf, 1.0, -1.0]:
            with np.errstate(invalid='raise', divide='raise'):
                for dt in ['e', 'f', 'd']:
                    assert_raises(FloatingPointError, np.arctanh,
                                  np.array(value, dtype=dt))

    # See: https://github.com/numpy/numpy/issues/20448
    @pytest.mark.xfail(
        _glibc_older_than("2.17"),
        reason="Older glibc versions may not raise appropriate FP exceptions"
    )
    def test_exp2(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, np.inf, -np.inf]
            out = [np.nan, np.nan, np.inf, 0.0]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.exp2(in_arr), out_arr)

        for value in [2000.0, -2000.0]:
            with np.errstate(over='raise', under='raise'):
                for dt in ['e', 'f', 'd']:
                    assert_raises(FloatingPointError, np.exp2,
                                  np.array(value, dtype=dt))

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_expm1(self):
        with np.errstate(all='ignore'):
            in_ = [np.nan, -np.nan, np.inf, -np.inf]
            out = [np.nan, np.nan, np.inf, -1.0]
            for dt in ['e', 'f', 'd']:
                in_arr = np.array(in_, dtype=dt)
                out_arr = np.array(out, dtype=dt)
                assert_equal(np.expm1(in_arr), out_arr)

        for value in [200.0, 2000.0]:
            with np.errstate(over='raise'):
                for dt in ['e', 'f']:
                    assert_raises(FloatingPointError, np.expm1,
                                  np.array(value, dtype=dt))

    # test to ensure no spurious FP exceptions are raised due to SIMD
    INF_INVALID_ERR = [
        np.cos, np.sin, np.tan, np.arccos, np.arcsin, np.spacing, np.arctanh
    ]
    NEG_INVALID_ERR = [
        np.log, np.log2, np.log10, np.log1p, np.sqrt, np.arccosh,
        np.arctanh
    ]
    ONE_INVALID_ERR = [
        np.arctanh,
    ]
    LTONE_INVALID_ERR = [
        np.arccosh,
    ]
    BYZERO_ERR = [
        np.log, np.log2, np.log10, np.reciprocal, np.arccosh
    ]

    @pytest.mark.parametrize("ufunc", UFUNCS_UNARY_FP)
    @pytest.mark.parametrize("dtype", ('e', 'f', 'd'))
    @pytest.mark.parametrize("data, escape", (
        ([0.03], LTONE_INVALID_ERR),
        ([0.03]*32, LTONE_INVALID_ERR),
        # neg
        ([-1.0], NEG_INVALID_ERR),
        ([-1.0]*32, NEG_INVALID_ERR),
        # flat
        ([1.0], ONE_INVALID_ERR),
        ([1.0]*32, ONE_INVALID_ERR),
        # zero
        ([0.0], BYZERO_ERR),
        ([0.0]*32, BYZERO_ERR),
        ([-0.0], BYZERO_ERR),
        ([-0.0]*32, BYZERO_ERR),
        # nan
        ([0.5, 0.5, 0.5, np.nan], LTONE_INVALID_ERR),
        ([0.5, 0.5, 0.5, np.nan]*32, LTONE_INVALID_ERR),
        ([np.nan, 1.0, 1.0, 1.0], ONE_INVALID_ERR),
        ([np.nan, 1.0, 1.0, 1.0]*32, ONE_INVALID_ERR),
        ([np.nan], []),
        ([np.nan]*32, []),
        # inf
        ([0.5, 0.5, 0.5, np.inf], INF_INVALID_ERR + LTONE_INVALID_ERR),
        ([0.5, 0.5, 0.5, np.inf]*32, INF_INVALID_ERR + LTONE_INVALID_ERR),
        ([np.inf, 1.0, 1.0, 1.0], INF_INVALID_ERR),
        ([np.inf, 1.0, 1.0, 1.0]*32, INF_INVALID_ERR),
        ([np.inf], INF_INVALID_ERR),
        ([np.inf]*32, INF_INVALID_ERR),
        # ninf
        ([0.5, 0.5, 0.5, -np.inf],
         NEG_INVALID_ERR + INF_INVALID_ERR + LTONE_INVALID_ERR),
        ([0.5, 0.5, 0.5, -np.inf]*32,
         NEG_INVALID_ERR + INF_INVALID_ERR + LTONE_INVALID_ERR),
        ([-np.inf, 1.0, 1.0, 1.0], NEG_INVALID_ERR + INF_INVALID_ERR),
        ([-np.inf, 1.0, 1.0, 1.0]*32, NEG_INVALID_ERR + INF_INVALID_ERR),
        ([-np.inf], NEG_INVALID_ERR + INF_INVALID_ERR),
        ([-np.inf]*32, NEG_INVALID_ERR + INF_INVALID_ERR),
    ))
    def test_unary_spurious_fpexception(self, ufunc, dtype, data, escape):
        if escape and ufunc in escape:
            return
        # FIXME: NAN raises FP invalid exception:
        #  - ceil/float16 on MSVC:32-bit
        #  - spacing/float16 on almost all platforms
        if ufunc in (np.spacing, np.ceil) and dtype == 'e':
            return
        array = np.array(data, dtype=dtype)
        with assert_no_warnings():
            ufunc(array)

class TestFPClass:
    @pytest.mark.parametrize("stride", [-4,-2,-1,1,2,4])
    def test_fpclass(self, stride):
        arr_f64 = np.array([np.nan, -np.nan, np.inf, -np.inf, -1.0, 1.0, -0.0, 0.0, 2.2251e-308, -2.2251e-308], dtype='d')
        arr_f32 = np.array([np.nan, -np.nan, np.inf, -np.inf, -1.0, 1.0, -0.0, 0.0, 1.4013e-045, -1.4013e-045], dtype='f')
        nan     = np.array([True, True, False, False, False, False, False, False, False, False])
        inf     = np.array([False, False, True, True, False, False, False, False, False, False])
        sign    = np.array([False, True, False, True, True, False, True, False, False, True])
        finite  = np.array([False, False, False, False, True, True, True, True, True, True])
        assert_equal(np.isnan(arr_f32[::stride]), nan[::stride])
        assert_equal(np.isnan(arr_f64[::stride]), nan[::stride])
        assert_equal(np.isinf(arr_f32[::stride]), inf[::stride])
        assert_equal(np.isinf(arr_f64[::stride]), inf[::stride])
        assert_equal(np.signbit(arr_f32[::stride]), sign[::stride])
        assert_equal(np.signbit(arr_f64[::stride]), sign[::stride])
        assert_equal(np.isfinite(arr_f32[::stride]), finite[::stride])
        assert_equal(np.isfinite(arr_f64[::stride]), finite[::stride])

class TestLDExp:
    @pytest.mark.parametrize("stride", [-4,-2,-1,1,2,4])
    @pytest.mark.parametrize("dtype", ['f', 'd'])
    def test_ldexp(self, dtype, stride):
        mant = np.array([0.125, 0.25, 0.5, 1., 1., 2., 4., 8.], dtype=dtype)
        exp  = np.array([3, 2, 1, 0, 0, -1, -2, -3], dtype='i')
        out  = np.zeros(8, dtype=dtype)
        assert_equal(np.ldexp(mant[::stride], exp[::stride], out=out[::stride]), np.ones(8, dtype=dtype)[::stride])
        assert_equal(out[::stride], np.ones(8, dtype=dtype)[::stride])

class TestFRExp:
    @pytest.mark.parametrize("stride", [-4,-2,-1,1,2,4])
    @pytest.mark.parametrize("dtype", ['f', 'd'])
    @pytest.mark.skipif(not sys.platform.startswith('linux'),
                        reason="np.frexp gives different answers for NAN/INF on windows and linux")
    def test_frexp(self, dtype, stride):
        arr = np.array([np.nan, np.nan, np.inf, -np.inf, 0.0, -0.0, 1.0, -1.0], dtype=dtype)
        mant_true = np.array([np.nan, np.nan, np.inf, -np.inf, 0.0, -0.0, 0.5, -0.5], dtype=dtype)
        exp_true  = np.array([0, 0, 0, 0, 0, 0, 1, 1], dtype='i')
        out_mant  = np.ones(8, dtype=dtype)
        out_exp   = 2*np.ones(8, dtype='i')
        mant, exp = np.frexp(arr[::stride], out=(out_mant[::stride], out_exp[::stride]))
        assert_equal(mant_true[::stride], mant)
        assert_equal(exp_true[::stride], exp)
        assert_equal(out_mant[::stride], mant_true[::stride])
        assert_equal(out_exp[::stride], exp_true[::stride])

# func : [maxulperror, low, high]
avx_ufuncs = {'sqrt'        :[1,  0.,   100.],
              'absolute'    :[0, -100., 100.],
              'reciprocal'  :[1,  1.,   100.],
              'square'      :[1, -100., 100.],
              'rint'        :[0, -100., 100.],
              'floor'       :[0, -100., 100.],
              'ceil'        :[0, -100., 100.],
              'trunc'       :[0, -100., 100.]}

class TestAVXUfuncs:
    def test_avx_based_ufunc(self):
        strides = np.array([-4,-3,-2,-1,1,2,3,4])
        np.random.seed(42)
        for func, prop in avx_ufuncs.items():
            maxulperr = prop[0]
            minval = prop[1]
            maxval = prop[2]
            # various array sizes to ensure masking in AVX is tested
            for size in range(1,32):
                myfunc = getattr(np, func)
                x_f32 = np.float32(np.random.uniform(low=minval, high=maxval,
                    size=size))
                x_f64 = np.float64(x_f32)
                x_f128 = np.longdouble(x_f32)
                y_true128 = myfunc(x_f128)
                if maxulperr == 0:
                    assert_equal(myfunc(x_f32), np.float32(y_true128))
                    assert_equal(myfunc(x_f64), np.float64(y_true128))
                else:
                    assert_array_max_ulp(myfunc(x_f32), np.float32(y_true128),
                            maxulp=maxulperr)
                    assert_array_max_ulp(myfunc(x_f64), np.float64(y_true128),
                            maxulp=maxulperr)
                # various strides to test gather instruction
                if size > 1:
                    y_true32 = myfunc(x_f32)
                    y_true64 = myfunc(x_f64)
                    for jj in strides:
                        assert_equal(myfunc(x_f64[::jj]), y_true64[::jj])
                        assert_equal(myfunc(x_f32[::jj]), y_true32[::jj])

class TestAVXFloat32Transcendental:
    def test_exp_float32(self):
        np.random.seed(42)
        x_f32 = np.float32(np.random.uniform(low=0.0,high=88.1,size=1000000))
        x_f64 = np.float64(x_f32)
        assert_array_max_ulp(np.exp(x_f32), np.float32(np.exp(x_f64)), maxulp=3)

    def test_log_float32(self):
        np.random.seed(42)
        x_f32 = np.float32(np.random.uniform(low=0.0,high=1000,size=1000000))
        x_f64 = np.float64(x_f32)
        assert_array_max_ulp(np.log(x_f32), np.float32(np.log(x_f64)), maxulp=4)

    def test_sincos_float32(self):
        np.random.seed(42)
        N = 1000000
        M = np.int_(N/20)
        index = np.random.randint(low=0, high=N, size=M)
        x_f32 = np.float32(np.random.uniform(low=-100.,high=100.,size=N))
        if not _glibc_older_than("2.17"):
            # test coverage for elements > 117435.992f for which glibc is used
            # this is known to be problematic on old glibc, so skip it there
            x_f32[index] = np.float32(10E+10*np.random.rand(M))
        x_f64 = np.float64(x_f32)
        assert_array_max_ulp(np.sin(x_f32), np.float32(np.sin(x_f64)), maxulp=2)
        assert_array_max_ulp(np.cos(x_f32), np.float32(np.cos(x_f64)), maxulp=2)
        # test aliasing(issue #17761)
        tx_f32 = x_f32.copy()
        assert_array_max_ulp(np.sin(x_f32, out=x_f32), np.float32(np.sin(x_f64)), maxulp=2)
        assert_array_max_ulp(np.cos(tx_f32, out=tx_f32), np.float32(np.cos(x_f64)), maxulp=2)

    def test_strided_float32(self):
        np.random.seed(42)
        strides = np.array([-4,-3,-2,-1,1,2,3,4])
        sizes = np.arange(2,100)
        for ii in sizes:
            x_f32 = np.float32(np.random.uniform(low=0.01,high=88.1,size=ii))
            x_f32_large = x_f32.copy()
            x_f32_large[3:-1:4] = 120000.0
            exp_true = np.exp(x_f32)
            log_true = np.log(x_f32)
            sin_true = np.sin(x_f32_large)
            cos_true = np.cos(x_f32_large)
            for jj in strides:
                assert_array_almost_equal_nulp(np.exp(x_f32[::jj]), exp_true[::jj], nulp=2)
                assert_array_almost_equal_nulp(np.log(x_f32[::jj]), log_true[::jj], nulp=2)
                assert_array_almost_equal_nulp(np.sin(x_f32_large[::jj]), sin_true[::jj], nulp=2)
                assert_array_almost_equal_nulp(np.cos(x_f32_large[::jj]), cos_true[::jj], nulp=2)

class TestLogAddExp(_FilterInvalids):
    def test_logaddexp_values(self):
        x = [1, 2, 3, 4, 5]
        y = [5, 4, 3, 2, 1]
        z = [6, 6, 6, 6, 6]
        for dt, dec_ in zip(['f', 'd', 'g'], [6, 15, 15]):
            xf = np.log(np.array(x, dtype=dt))
            yf = np.log(np.array(y, dtype=dt))
            zf = np.log(np.array(z, dtype=dt))
            assert_almost_equal(np.logaddexp(xf, yf), zf, decimal=dec_)

    def test_logaddexp_range(self):
        x = [1000000, -1000000, 1000200, -1000200]
        y = [1000200, -1000200, 1000000, -1000000]
        z = [1000200, -1000000, 1000200, -1000000]
        for dt in ['f', 'd', 'g']:
            logxf = np.array(x, dtype=dt)
            logyf = np.array(y, dtype=dt)
            logzf = np.array(z, dtype=dt)
            assert_almost_equal(np.logaddexp(logxf, logyf), logzf)

    def test_inf(self):
        inf = np.inf
        x = [inf, -inf,  inf, -inf, inf, 1,  -inf,  1]
        y = [inf,  inf, -inf, -inf, 1,   inf, 1,   -inf]
        z = [inf,  inf,  inf, -inf, inf, inf, 1,    1]
        with np.errstate(invalid='raise'):
            for dt in ['f', 'd', 'g']:
                logxf = np.array(x, dtype=dt)
                logyf = np.array(y, dtype=dt)
                logzf = np.array(z, dtype=dt)
                assert_equal(np.logaddexp(logxf, logyf), logzf)

    def test_nan(self):
        assert_(np.isnan(np.logaddexp(np.nan, np.inf)))
        assert_(np.isnan(np.logaddexp(np.inf, np.nan)))
        assert_(np.isnan(np.logaddexp(np.nan, 0)))
        assert_(np.isnan(np.logaddexp(0, np.nan)))
        assert_(np.isnan(np.logaddexp(np.nan, np.nan)))

    def test_reduce(self):
        assert_equal(np.logaddexp.identity, -np.inf)
        assert_equal(np.logaddexp.reduce([]), -np.inf)


class TestLog1p:
    def test_log1p(self):
        assert_almost_equal(ncu.log1p(0.2), ncu.log(1.2))
        assert_almost_equal(ncu.log1p(1e-6), ncu.log(1+1e-6))

    def test_special(self):
        with np.errstate(invalid="ignore", divide="ignore"):
            assert_equal(ncu.log1p(np.nan), np.nan)
            assert_equal(ncu.log1p(np.inf), np.inf)
            assert_equal(ncu.log1p(-1.), -np.inf)
            assert_equal(ncu.log1p(-2.), np.nan)
            assert_equal(ncu.log1p(-np.inf), np.nan)


class TestExpm1:
    def test_expm1(self):
        assert_almost_equal(ncu.expm1(0.2), ncu.exp(0.2)-1)
        assert_almost_equal(ncu.expm1(1e-6), ncu.exp(1e-6)-1)

    def test_special(self):
        assert_equal(ncu.expm1(np.inf), np.inf)
        assert_equal(ncu.expm1(0.), 0.)
        assert_equal(ncu.expm1(-0.), -0.)
        assert_equal(ncu.expm1(np.inf), np.inf)
        assert_equal(ncu.expm1(-np.inf), -1.)

    def test_complex(self):
        x = np.asarray(1e-12)
        assert_allclose(x, ncu.expm1(x))
        x = x.astype(np.complex128)
        assert_allclose(x, ncu.expm1(x))


class TestHypot:
    def test_simple(self):
        assert_almost_equal(ncu.hypot(1, 1), ncu.sqrt(2))
        assert_almost_equal(ncu.hypot(0, 0), 0)

    def test_reduce(self):
        assert_almost_equal(ncu.hypot.reduce([3.0, 4.0]), 5.0)
        assert_almost_equal(ncu.hypot.reduce([3.0, 4.0, 0]), 5.0)
        assert_almost_equal(ncu.hypot.reduce([9.0, 12.0, 20.0]), 25.0)
        assert_equal(ncu.hypot.reduce([]), 0.0)


def assert_hypot_isnan(x, y):
    with np.errstate(invalid='ignore'):
        assert_(np.isnan(ncu.hypot(x, y)),
                "hypot(%s, %s) is %s, not nan" % (x, y, ncu.hypot(x, y)))


def assert_hypot_isinf(x, y):
    with np.errstate(invalid='ignore'):
        assert_(np.isinf(ncu.hypot(x, y)),
                "hypot(%s, %s) is %s, not inf" % (x, y, ncu.hypot(x, y)))


class TestHypotSpecialValues:
    def test_nan_outputs(self):
        assert_hypot_isnan(np.nan, np.nan)
        assert_hypot_isnan(np.nan, 1)

    def test_nan_outputs2(self):
        assert_hypot_isinf(np.nan, np.inf)
        assert_hypot_isinf(np.inf, np.nan)
        assert_hypot_isinf(np.inf, 0)
        assert_hypot_isinf(0, np.inf)
        assert_hypot_isinf(np.inf, np.inf)
        assert_hypot_isinf(np.inf, 23.0)

    def test_no_fpe(self):
        assert_no_warnings(ncu.hypot, np.inf, 0)


def assert_arctan2_isnan(x, y):
    assert_(np.isnan(ncu.arctan2(x, y)), "arctan(%s, %s) is %s, not nan" % (x, y, ncu.arctan2(x, y)))


def assert_arctan2_ispinf(x, y):
    assert_((np.isinf(ncu.arctan2(x, y)) and ncu.arctan2(x, y) > 0), "arctan(%s, %s) is %s, not +inf" % (x, y, ncu.arctan2(x, y)))


def assert_arctan2_isninf(x, y):
    assert_((np.isinf(ncu.arctan2(x, y)) and ncu.arctan2(x, y) < 0), "arctan(%s, %s) is %s, not -inf" % (x, y, ncu.arctan2(x, y)))


def assert_arctan2_ispzero(x, y):
    assert_((ncu.arctan2(x, y) == 0 and not np.signbit(ncu.arctan2(x, y))), "arctan(%s, %s) is %s, not +0" % (x, y, ncu.arctan2(x, y)))


def assert_arctan2_isnzero(x, y):
    assert_((ncu.arctan2(x, y) == 0 and np.signbit(ncu.arctan2(x, y))), "arctan(%s, %s) is %s, not -0" % (x, y, ncu.arctan2(x, y)))


class TestArctan2SpecialValues:
    def test_one_one(self):
        # atan2(1, 1) returns pi/4.
        assert_almost_equal(ncu.arctan2(1, 1), 0.25 * np.pi)
        assert_almost_equal(ncu.arctan2(-1, 1), -0.25 * np.pi)
        assert_almost_equal(ncu.arctan2(1, -1), 0.75 * np.pi)

    def test_zero_nzero(self):
        # atan2(+-0, -0) returns +-pi.
        assert_almost_equal(ncu.arctan2(np.PZERO, np.NZERO), np.pi)
        assert_almost_equal(ncu.arctan2(np.NZERO, np.NZERO), -np.pi)

    def test_zero_pzero(self):
        # atan2(+-0, +0) returns +-0.
        assert_arctan2_ispzero(np.PZERO, np.PZERO)
        assert_arctan2_isnzero(np.NZERO, np.PZERO)

    def test_zero_negative(self):
        # atan2(+-0, x) returns +-pi for x < 0.
        assert_almost_equal(ncu.arctan2(np.PZERO, -1), np.pi)
        assert_almost_equal(ncu.arctan2(np.NZERO, -1), -np.pi)

    def test_zero_positive(self):
        # atan2(+-0, x) returns +-0 for x > 0.
        assert_arctan2_ispzero(np.PZERO, 1)
        assert_arctan2_isnzero(np.NZERO, 1)

    def test_positive_zero(self):
        # atan2(y, +-0) returns +pi/2 for y > 0.
        assert_almost_equal(ncu.arctan2(1, np.PZERO), 0.5 * np.pi)
        assert_almost_equal(ncu.arctan2(1, np.NZERO), 0.5 * np.pi)

    def test_negative_zero(self):
        # atan2(y, +-0) returns -pi/2 for y < 0.
        assert_almost_equal(ncu.arctan2(-1, np.PZERO), -0.5 * np.pi)
        assert_almost_equal(ncu.arctan2(-1, np.NZERO), -0.5 * np.pi)

    def test_any_ninf(self):
        # atan2(+-y, -infinity) returns +-pi for finite y > 0.
        assert_almost_equal(ncu.arctan2(1, np.NINF),  np.pi)
        assert_almost_equal(ncu.arctan2(-1, np.NINF), -np.pi)

    def test_any_pinf(self):
        # atan2(+-y, +infinity) returns +-0 for finite y > 0.
        assert_arctan2_ispzero(1, np.inf)
        assert_arctan2_isnzero(-1, np.inf)

    def test_inf_any(self):
        # atan2(+-infinity, x) returns +-pi/2 for finite x.
        assert_almost_equal(ncu.arctan2( np.inf, 1),  0.5 * np.pi)
        assert_almost_equal(ncu.arctan2(-np.inf, 1), -0.5 * np.pi)

    def test_inf_ninf(self):
        # atan2(+-infinity, -infinity) returns +-3*pi/4.
        assert_almost_equal(ncu.arctan2( np.inf, -np.inf),  0.75 * np.pi)
        assert_almost_equal(ncu.arctan2(-np.inf, -np.inf), -0.75 * np.pi)

    def test_inf_pinf(self):
        # atan2(+-infinity, +infinity) returns +-pi/4.
        assert_almost_equal(ncu.arctan2( np.inf, np.inf),  0.25 * np.pi)
        assert_almost_equal(ncu.arctan2(-np.inf, np.inf), -0.25 * np.pi)

    def test_nan_any(self):
        # atan2(nan, x) returns nan for any x, including inf
        assert_arctan2_isnan(np.nan, np.inf)
        assert_arctan2_isnan(np.inf, np.nan)
        assert_arctan2_isnan(np.nan, np.nan)


class TestLdexp:
    def _check_ldexp(self, tp):
        assert_almost_equal(ncu.ldexp(np.array(2., np.float32),
                                      np.array(3, tp)), 16.)
        assert_almost_equal(ncu.ldexp(np.array(2., np.float64),
                                      np.array(3, tp)), 16.)
        assert_almost_equal(ncu.ldexp(np.array(2., np.longdouble),
                                      np.array(3, tp)), 16.)

    def test_ldexp(self):
        # The default Python int type should work
        assert_almost_equal(ncu.ldexp(2., 3),  16.)
        # The following int types should all be accepted
        self._check_ldexp(np.int8)
        self._check_ldexp(np.int16)
        self._check_ldexp(np.int32)
        self._check_ldexp('i')
        self._check_ldexp('l')

    def test_ldexp_overflow(self):
        # silence warning emitted on overflow
        with np.errstate(over="ignore"):
            imax = np.iinfo(np.dtype('l')).max
            imin = np.iinfo(np.dtype('l')).min
            assert_equal(ncu.ldexp(2., imax), np.inf)
            assert_equal(ncu.ldexp(2., imin), 0)


class TestMaximum(_FilterInvalids):
    def test_reduce(self):
        dflt = np.typecodes['AllFloat']
        dint = np.typecodes['AllInteger']
        seq1 = np.arange(11)
        seq2 = seq1[::-1]
        func = np.maximum.reduce
        for dt in dint:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 10)
            assert_equal(func(tmp2), 10)
        for dt in dflt:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 10)
            assert_equal(func(tmp2), 10)
            tmp1[::2] = np.nan
            tmp2[::2] = np.nan
            assert_equal(func(tmp1), np.nan)
            assert_equal(func(tmp2), np.nan)

    def test_reduce_complex(self):
        assert_equal(np.maximum.reduce([1, 2j]), 1)
        assert_equal(np.maximum.reduce([1+3j, 2j]), 1+3j)

    def test_float_nans(self):
        nan = np.nan
        arg1 = np.array([0,   nan, nan])
        arg2 = np.array([nan, 0,   nan])
        out = np.array([nan, nan, nan])
        assert_equal(np.maximum(arg1, arg2), out)

    def test_object_nans(self):
        # Multiple checks to give this a chance to
        # fail if cmp is used instead of rich compare.
        # Failure cannot be guaranteed.
        for i in range(1):
            x = np.array(float('nan'), object)
            y = 1.0
            z = np.array(float('nan'), object)
            assert_(np.maximum(x, y) == 1.0)
            assert_(np.maximum(z, y) == 1.0)

    def test_complex_nans(self):
        nan = np.nan
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            arg1 = np.array([0, cnan, cnan], dtype=complex)
            arg2 = np.array([cnan, 0, cnan], dtype=complex)
            out = np.array([nan, nan, nan], dtype=complex)
            assert_equal(np.maximum(arg1, arg2), out)

    def test_object_array(self):
        arg1 = np.arange(5, dtype=object)
        arg2 = arg1 + 1
        assert_equal(np.maximum(arg1, arg2), arg2)

    def test_strided_array(self):
        arr1 = np.array([-4.0, 1.0, 10.0,  0.0, np.nan, -np.nan, np.inf, -np.inf])
        arr2 = np.array([-2.0,-1.0, np.nan, 1.0, 0.0,    np.nan, 1.0,    -3.0])
        maxtrue  = np.array([-2.0, 1.0, np.nan, 1.0, np.nan, np.nan, np.inf, -3.0])
        out = np.ones(8)
        out_maxtrue = np.array([-2.0, 1.0, 1.0, 10.0, 1.0, 1.0, np.nan, 1.0])
        assert_equal(np.maximum(arr1,arr2), maxtrue)
        assert_equal(np.maximum(arr1[::2],arr2[::2]), maxtrue[::2])
        assert_equal(np.maximum(arr1[:4:], arr2[::2]), np.array([-2.0, np.nan, 10.0, 1.0]))
        assert_equal(np.maximum(arr1[::3], arr2[:3:]), np.array([-2.0, 0.0, np.nan]))
        assert_equal(np.maximum(arr1[:6:2], arr2[::3], out=out[::3]), np.array([-2.0, 10., np.nan]))
        assert_equal(out, out_maxtrue)

    def test_precision(self):
        dtypes = [np.float16, np.float32, np.float64, np.longdouble]

        for dt in dtypes:
            dtmin = np.finfo(dt).min
            dtmax = np.finfo(dt).max
            d1 = dt(0.1)
            d1_next = np.nextafter(d1, np.inf)

            test_cases = [
                # v1    v2          expected
                (dtmin, -np.inf,    dtmin),
                (dtmax, -np.inf,    dtmax),
                (d1,    d1_next,    d1_next),
                (dtmax, np.nan,     np.nan),
            ]

            for v1, v2, expected in test_cases:
                assert_equal(np.maximum([v1], [v2]), [expected])
                assert_equal(np.maximum.reduce([v1, v2]), expected)


class TestMinimum(_FilterInvalids):
    def test_reduce(self):
        dflt = np.typecodes['AllFloat']
        dint = np.typecodes['AllInteger']
        seq1 = np.arange(11)
        seq2 = seq1[::-1]
        func = np.minimum.reduce
        for dt in dint:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 0)
            assert_equal(func(tmp2), 0)
        for dt in dflt:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 0)
            assert_equal(func(tmp2), 0)
            tmp1[::2] = np.nan
            tmp2[::2] = np.nan
            assert_equal(func(tmp1), np.nan)
            assert_equal(func(tmp2), np.nan)

    def test_reduce_complex(self):
        assert_equal(np.minimum.reduce([1, 2j]), 2j)
        assert_equal(np.minimum.reduce([1+3j, 2j]), 2j)

    def test_float_nans(self):
        nan = np.nan
        arg1 = np.array([0,   nan, nan])
        arg2 = np.array([nan, 0,   nan])
        out = np.array([nan, nan, nan])
        assert_equal(np.minimum(arg1, arg2), out)

    def test_object_nans(self):
        # Multiple checks to give this a chance to
        # fail if cmp is used instead of rich compare.
        # Failure cannot be guaranteed.
        for i in range(1):
            x = np.array(float('nan'), object)
            y = 1.0
            z = np.array(float('nan'), object)
            assert_(np.minimum(x, y) == 1.0)
            assert_(np.minimum(z, y) == 1.0)

    def test_complex_nans(self):
        nan = np.nan
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            arg1 = np.array([0, cnan, cnan], dtype=complex)
            arg2 = np.array([cnan, 0, cnan], dtype=complex)
            out = np.array([nan, nan, nan], dtype=complex)
            assert_equal(np.minimum(arg1, arg2), out)

    def test_object_array(self):
        arg1 = np.arange(5, dtype=object)
        arg2 = arg1 + 1
        assert_equal(np.minimum(arg1, arg2), arg1)

    def test_strided_array(self):
        arr1 = np.array([-4.0, 1.0, 10.0,  0.0, np.nan, -np.nan, np.inf, -np.inf])
        arr2 = np.array([-2.0,-1.0, np.nan, 1.0, 0.0,    np.nan, 1.0,    -3.0])
        mintrue  = np.array([-4.0, -1.0, np.nan, 0.0, np.nan, np.nan, 1.0, -np.inf])
        out = np.ones(8)
        out_mintrue = np.array([-4.0, 1.0, 1.0, 1.0, 1.0, 1.0, np.nan, 1.0])
        assert_equal(np.minimum(arr1,arr2), mintrue)
        assert_equal(np.minimum(arr1[::2],arr2[::2]), mintrue[::2])
        assert_equal(np.minimum(arr1[:4:], arr2[::2]), np.array([-4.0, np.nan, 0.0, 0.0]))
        assert_equal(np.minimum(arr1[::3], arr2[:3:]), np.array([-4.0, -1.0, np.nan]))
        assert_equal(np.minimum(arr1[:6:2], arr2[::3], out=out[::3]), np.array([-4.0, 1.0, np.nan]))
        assert_equal(out, out_mintrue)

    def test_precision(self):
        dtypes = [np.float16, np.float32, np.float64, np.longdouble]

        for dt in dtypes:
            dtmin = np.finfo(dt).min
            dtmax = np.finfo(dt).max
            d1 = dt(0.1)
            d1_next = np.nextafter(d1, np.inf)

            test_cases = [
                # v1    v2          expected
                (dtmin, np.inf,     dtmin),
                (dtmax, np.inf,     dtmax),
                (d1,    d1_next,    d1),
                (dtmin, np.nan,     np.nan),
            ]

            for v1, v2, expected in test_cases:
                assert_equal(np.minimum([v1], [v2]), [expected])
                assert_equal(np.minimum.reduce([v1, v2]), expected)


class TestFmax(_FilterInvalids):
    def test_reduce(self):
        dflt = np.typecodes['AllFloat']
        dint = np.typecodes['AllInteger']
        seq1 = np.arange(11)
        seq2 = seq1[::-1]
        func = np.fmax.reduce
        for dt in dint:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 10)
            assert_equal(func(tmp2), 10)
        for dt in dflt:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 10)
            assert_equal(func(tmp2), 10)
            tmp1[::2] = np.nan
            tmp2[::2] = np.nan
            assert_equal(func(tmp1), 9)
            assert_equal(func(tmp2), 9)

    def test_reduce_complex(self):
        assert_equal(np.fmax.reduce([1, 2j]), 1)
        assert_equal(np.fmax.reduce([1+3j, 2j]), 1+3j)

    def test_float_nans(self):
        nan = np.nan
        arg1 = np.array([0,   nan, nan])
        arg2 = np.array([nan, 0,   nan])
        out = np.array([0,   0,   nan])
        assert_equal(np.fmax(arg1, arg2), out)

    def test_complex_nans(self):
        nan = np.nan
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            arg1 = np.array([0, cnan, cnan], dtype=complex)
            arg2 = np.array([cnan, 0, cnan], dtype=complex)
            out = np.array([0,    0, nan], dtype=complex)
            assert_equal(np.fmax(arg1, arg2), out)

    def test_precision(self):
        dtypes = [np.float16, np.float32, np.float64, np.longdouble]

        for dt in dtypes:
            dtmin = np.finfo(dt).min
            dtmax = np.finfo(dt).max
            d1 = dt(0.1)
            d1_next = np.nextafter(d1, np.inf)

            test_cases = [
                # v1    v2          expected
                (dtmin, -np.inf,    dtmin),
                (dtmax, -np.inf,    dtmax),
                (d1,    d1_next,    d1_next),
                (dtmax, np.nan,     dtmax),
            ]

            for v1, v2, expected in test_cases:
                assert_equal(np.fmax([v1], [v2]), [expected])
                assert_equal(np.fmax.reduce([v1, v2]), expected)


class TestFmin(_FilterInvalids):
    def test_reduce(self):
        dflt = np.typecodes['AllFloat']
        dint = np.typecodes['AllInteger']
        seq1 = np.arange(11)
        seq2 = seq1[::-1]
        func = np.fmin.reduce
        for dt in dint:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 0)
            assert_equal(func(tmp2), 0)
        for dt in dflt:
            tmp1 = seq1.astype(dt)
            tmp2 = seq2.astype(dt)
            assert_equal(func(tmp1), 0)
            assert_equal(func(tmp2), 0)
            tmp1[::2] = np.nan
            tmp2[::2] = np.nan
            assert_equal(func(tmp1), 1)
            assert_equal(func(tmp2), 1)

    def test_reduce_complex(self):
        assert_equal(np.fmin.reduce([1, 2j]), 2j)
        assert_equal(np.fmin.reduce([1+3j, 2j]), 2j)

    def test_float_nans(self):
        nan = np.nan
        arg1 = np.array([0,   nan, nan])
        arg2 = np.array([nan, 0,   nan])
        out = np.array([0,   0,   nan])
        assert_equal(np.fmin(arg1, arg2), out)

    def test_complex_nans(self):
        nan = np.nan
        for cnan in [complex(nan, 0), complex(0, nan), complex(nan, nan)]:
            arg1 = np.array([0, cnan, cnan], dtype=complex)
            arg2 = np.array([cnan, 0, cnan], dtype=complex)
            out = np.array([0,    0, nan], dtype=complex)
            assert_equal(np.fmin(arg1, arg2), out)

    def test_precision(self):
        dtypes = [np.float16, np.float32, np.float64, np.longdouble]

        for dt in dtypes:
            dtmin = np.finfo(dt).min
            dtmax = np.finfo(dt).max
            d1 = dt(0.1)
            d1_next = np.nextafter(d1, np.inf)

            test_cases = [
                # v1    v2          expected
                (dtmin, np.inf,     dtmin),
                (dtmax, np.inf,     dtmax),
                (d1,    d1_next,    d1),
                (dtmin, np.nan,     dtmin),
            ]

            for v1, v2, expected in test_cases:
                assert_equal(np.fmin([v1], [v2]), [expected])
                assert_equal(np.fmin.reduce([v1, v2]), expected)


class TestBool:
    def test_exceptions(self):
        a = np.ones(1, dtype=np.bool_)
        assert_raises(TypeError, np.negative, a)
        assert_raises(TypeError, np.positive, a)
        assert_raises(TypeError, np.subtract, a, a)

    def test_truth_table_logical(self):
        # 2, 3 and 4 serves as true values
        input1 = [0, 0, 3, 2]
        input2 = [0, 4, 0, 2]

        typecodes = (np.typecodes['AllFloat']
                     + np.typecodes['AllInteger']
                     + '?')     # boolean
        for dtype in map(np.dtype, typecodes):
            arg1 = np.asarray(input1, dtype=dtype)
            arg2 = np.asarray(input2, dtype=dtype)

            # OR
            out = [False, True, True, True]
            for func in (np.logical_or, np.maximum):
                assert_equal(func(arg1, arg2).astype(bool), out)
            # AND
            out = [False, False, False, True]
            for func in (np.logical_and, np.minimum):
                assert_equal(func(arg1, arg2).astype(bool), out)
            # XOR
            out = [False, True, True, False]
            for func in (np.logical_xor, np.not_equal):
                assert_equal(func(arg1, arg2).astype(bool), out)

    def test_truth_table_bitwise(self):
        arg1 = [False, False, True, True]
        arg2 = [False, True, False, True]

        out = [False, True, True, True]
        assert_equal(np.bitwise_or(arg1, arg2), out)

        out = [False, False, False, True]
        assert_equal(np.bitwise_and(arg1, arg2), out)

        out = [False, True, True, False]
        assert_equal(np.bitwise_xor(arg1, arg2), out)

    def test_reduce(self):
        none = np.array([0, 0, 0, 0], bool)
        some = np.array([1, 0, 1, 1], bool)
        every = np.array([1, 1, 1, 1], bool)
        empty = np.array([], bool)

        arrs = [none, some, every, empty]

        for arr in arrs:
            assert_equal(np.logical_and.reduce(arr), all(arr))

        for arr in arrs:
            assert_equal(np.logical_or.reduce(arr), any(arr))

        for arr in arrs:
            assert_equal(np.logical_xor.reduce(arr), arr.sum() % 2 == 1)


class TestBitwiseUFuncs:

    bitwise_types = [np.dtype(c) for c in '?' + 'bBhHiIlLqQ' + 'O']

    def test_values(self):
        for dt in self.bitwise_types:
            zeros = np.array([0], dtype=dt)
            ones = np.array([-1]).astype(dt)
            msg = "dt = '%s'" % dt.char

            assert_equal(np.bitwise_not(zeros), ones, err_msg=msg)
            assert_equal(np.bitwise_not(ones), zeros, err_msg=msg)

            assert_equal(np.bitwise_or(zeros, zeros), zeros, err_msg=msg)
            assert_equal(np.bitwise_or(zeros, ones), ones, err_msg=msg)
            assert_equal(np.bitwise_or(ones, zeros), ones, err_msg=msg)
            assert_equal(np.bitwise_or(ones, ones), ones, err_msg=msg)

            assert_equal(np.bitwise_xor(zeros, zeros), zeros, err_msg=msg)
            assert_equal(np.bitwise_xor(zeros, ones), ones, err_msg=msg)
            assert_equal(np.bitwise_xor(ones, zeros), ones, err_msg=msg)
            assert_equal(np.bitwise_xor(ones, ones), zeros, err_msg=msg)

            assert_equal(np.bitwise_and(zeros, zeros), zeros, err_msg=msg)
            assert_equal(np.bitwise_and(zeros, ones), zeros, err_msg=msg)
            assert_equal(np.bitwise_and(ones, zeros), zeros, err_msg=msg)
            assert_equal(np.bitwise_and(ones, ones), ones, err_msg=msg)

    def test_types(self):
        for dt in self.bitwise_types:
            zeros = np.array([0], dtype=dt)
            ones = np.array([-1]).astype(dt)
            msg = "dt = '%s'" % dt.char

            assert_(np.bitwise_not(zeros).dtype == dt, msg)
            assert_(np.bitwise_or(zeros, zeros).dtype == dt, msg)
            assert_(np.bitwise_xor(zeros, zeros).dtype == dt, msg)
            assert_(np.bitwise_and(zeros, zeros).dtype == dt, msg)

    def test_identity(self):
        assert_(np.bitwise_or.identity == 0, 'bitwise_or')
        assert_(np.bitwise_xor.identity == 0, 'bitwise_xor')
        assert_(np.bitwise_and.identity == -1, 'bitwise_and')

    def test_reduction(self):
        binary_funcs = (np.bitwise_or, np.bitwise_xor, np.bitwise_and)

        for dt in self.bitwise_types:
            zeros = np.array([0], dtype=dt)
            ones = np.array([-1]).astype(dt)
            for f in binary_funcs:
                msg = "dt: '%s', f: '%s'" % (dt, f)
                assert_equal(f.reduce(zeros), zeros, err_msg=msg)
                assert_equal(f.reduce(ones), ones, err_msg=msg)

        # Test empty reduction, no object dtype
        for dt in self.bitwise_types[:-1]:
            # No object array types
            empty = np.array([], dtype=dt)
            for f in binary_funcs:
                msg = "dt: '%s', f: '%s'" % (dt, f)
                tgt = np.array(f.identity).astype(dt)
                res = f.reduce(empty)
                assert_equal(res, tgt, err_msg=msg)
                assert_(res.dtype == tgt.dtype, msg)

        # Empty object arrays use the identity.  Note that the types may
        # differ, the actual type used is determined by the assign_identity
        # function and is not the same as the type returned by the identity
        # method.
        for f in binary_funcs:
            msg = "dt: '%s'" % (f,)
            empty = np.array([], dtype=object)
            tgt = f.identity
            res = f.reduce(empty)
            assert_equal(res, tgt, err_msg=msg)

        # Non-empty object arrays do not use the identity
        for f in binary_funcs:
            msg = "dt: '%s'" % (f,)
            btype = np.array([True], dtype=object)
            assert_(type(f.reduce(btype)) is bool, msg)


class TestInt:
    def test_logical_not(self):
        x = np.ones(10, dtype=np.int16)
        o = np.ones(10 * 2, dtype=bool)
        tgt = o.copy()
        tgt[::2] = False
        os = o[::2]
        assert_array_equal(np.logical_not(x, out=os), False)
        assert_array_equal(o, tgt)


class TestFloatingPoint:
    def test_floating_point(self):
        assert_equal(ncu.FLOATING_POINT_SUPPORT, 1)


class TestDegrees:
    def test_degrees(self):
        assert_almost_equal(ncu.degrees(np.pi), 180.0)
        assert_almost_equal(ncu.degrees(-0.5*np.pi), -90.0)


class TestRadians:
    def test_radians(self):
        assert_almost_equal(ncu.radians(180.0), np.pi)
        assert_almost_equal(ncu.radians(-90.0), -0.5*np.pi)


class TestHeavside:
    def test_heaviside(self):
        x = np.array([[-30.0, -0.1, 0.0, 0.2], [7.5, np.nan, np.inf, -np.inf]])
        expectedhalf = np.array([[0.0, 0.0, 0.5, 1.0], [1.0, np.nan, 1.0, 0.0]])
        expected1 = expectedhalf.copy()
        expected1[0, 2] = 1

        h = ncu.heaviside(x, 0.5)
        assert_equal(h, expectedhalf)

        h = ncu.heaviside(x, 1.0)
        assert_equal(h, expected1)

        x = x.astype(np.float32)

        h = ncu.heaviside(x, np.float32(0.5))
        assert_equal(h, expectedhalf.astype(np.float32))

        h = ncu.heaviside(x, np.float32(1.0))
        assert_equal(h, expected1.astype(np.float32))


class TestSign:
    def test_sign(self):
        a = np.array([np.inf, -np.inf, np.nan, 0.0, 3.0, -3.0])
        out = np.zeros(a.shape)
        tgt = np.array([1., -1., np.nan, 0.0, 1.0, -1.0])

        with np.errstate(invalid='ignore'):
            res = ncu.sign(a)
            assert_equal(res, tgt)
            res = ncu.sign(a, out)
            assert_equal(res, tgt)
            assert_equal(out, tgt)

    def test_sign_dtype_object(self):
        # In reference to github issue #6229

        foo = np.array([-.1, 0, .1])
        a = np.sign(foo.astype(object))
        b = np.sign(foo)

        assert_array_equal(a, b)

    def test_sign_dtype_nan_object(self):
        # In reference to github issue #6229
        def test_nan():
            foo = np.array([np.nan])
            # FIXME: a not used
            a = np.sign(foo.astype(object))

        assert_raises(TypeError, test_nan)

class TestMinMax:
    def test_minmax_blocked(self):
        # simd tests on max/min, test all alignments, slow but important
        # for 2 * vz + 2 * (vs - 1) + 1 (unrolled once)
        for dt, sz in [(np.float32, 15), (np.float64, 7)]:
            for out, inp, msg in _gen_alignment_data(dtype=dt, type='unary',
                                                     max_size=sz):
                for i in range(inp.size):
                    inp[:] = np.arange(inp.size, dtype=dt)
                    inp[i] = np.nan
                    emsg = lambda: '%r\n%s' % (inp, msg)
                    with suppress_warnings() as sup:
                        sup.filter(RuntimeWarning,
                                   "invalid value encountered in reduce")
                        assert_(np.isnan(inp.max()), msg=emsg)
                        assert_(np.isnan(inp.min()), msg=emsg)

                    inp[i] = 1e10
                    assert_equal(inp.max(), 1e10, err_msg=msg)
                    inp[i] = -1e10
                    assert_equal(inp.min(), -1e10, err_msg=msg)

    def test_lower_align(self):
        # check data that is not aligned to element size
        # i.e doubles are aligned to 4 bytes on i386
        d = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
        assert_equal(d.max(), d[0])
        assert_equal(d.min(), d[0])

    def test_reduce_reorder(self):
        # gh 10370, 11029 Some compilers reorder the call to npy_getfloatstatus
        # and put it before the call to an intrisic function that causes
        # invalid status to be set. Also make sure warnings are not emitted
        for n in (2, 4, 8, 16, 32):
            for dt in (np.float32, np.float16, np.complex64):
                for r in np.diagflat(np.array([np.nan] * n, dtype=dt)):
                    assert_equal(np.min(r), np.nan)

    def test_minimize_no_warns(self):
        a = np.minimum(np.nan, 1)
        assert_equal(a, np.nan)


class TestAbsoluteNegative:
    def test_abs_neg_blocked(self):
        # simd tests on abs, test all alignments for vz + 2 * (vs - 1) + 1
        for dt, sz in [(np.float32, 11), (np.float64, 5)]:
            for out, inp, msg in _gen_alignment_data(dtype=dt, type='unary',
                                                     max_size=sz):
                tgt = [ncu.absolute(i) for i in inp]
                np.absolute(inp, out=out)
                assert_equal(out, tgt, err_msg=msg)
                assert_((out >= 0).all())

                tgt = [-1*(i) for i in inp]
                np.negative(inp, out=out)
                assert_equal(out, tgt, err_msg=msg)

                for v in [np.nan, -np.inf, np.inf]:
                    for i in range(inp.size):
                        d = np.arange(inp.size, dtype=dt)
                        inp[:] = -d
                        inp[i] = v
                        d[i] = -v if v == -np.inf else v
                        assert_array_equal(np.abs(inp), d, err_msg=msg)
                        np.abs(inp, out=out)
                        assert_array_equal(out, d, err_msg=msg)

                        assert_array_equal(-inp, -1*inp, err_msg=msg)
                        d = -1 * inp
                        np.negative(inp, out=out)
                        assert_array_equal(out, d, err_msg=msg)

    def test_lower_align(self):
        # check data that is not aligned to element size
        # i.e doubles are aligned to 4 bytes on i386
        d = np.zeros(23 * 8, dtype=np.int8)[4:-4].view(np.float64)
        assert_equal(np.abs(d), d)
        assert_equal(np.negative(d), -d)
        np.negative(d, out=d)
        np.negative(np.ones_like(d), out=d)
        np.abs(d, out=d)
        np.abs(np.ones_like(d), out=d)


class TestPositive:
    def test_valid(self):
        valid_dtypes = [int, float, complex, object]
        for dtype in valid_dtypes:
            x = np.arange(5, dtype=dtype)
            result = np.positive(x)
            assert_equal(x, result, err_msg=str(dtype))

    def test_invalid(self):
        with assert_raises(TypeError):
            np.positive(True)
        with assert_raises(TypeError):
            np.positive(np.datetime64('2000-01-01'))
        with assert_raises(TypeError):
            np.positive(np.array(['foo'], dtype=str))
        with assert_raises(TypeError):
            np.positive(np.array(['bar'], dtype=object))


class TestSpecialMethods:
    def test_wrap(self):

        class with_wrap:
            def __array__(self):
                return np.zeros(1)

            def __array_wrap__(self, arr, context):
                r = with_wrap()
                r.arr = arr
                r.context = context
                return r

        a = with_wrap()
        x = ncu.minimum(a, a)
        assert_equal(x.arr, np.zeros(1))
        func, args, i = x.context
        assert_(func is ncu.minimum)
        assert_equal(len(args), 2)
        assert_equal(args[0], a)
        assert_equal(args[1], a)
        assert_equal(i, 0)

    def test_wrap_and_prepare_out(self):
        # Calling convention for out should not affect how special methods are
        # called

        class StoreArrayPrepareWrap(np.ndarray):
            _wrap_args = None
            _prepare_args = None
            def __new__(cls):
                return np.zeros(()).view(cls)
            def __array_wrap__(self, obj, context):
                self._wrap_args = context[1]
                return obj
            def __array_prepare__(self, obj, context):
                self._prepare_args = context[1]
                return obj
            @property
            def args(self):
                # We need to ensure these are fetched at the same time, before
                # any other ufuncs are called by the assertions
                return (self._prepare_args, self._wrap_args)
            def __repr__(self):
                return "a"  # for short test output

        def do_test(f_call, f_expected):
            a = StoreArrayPrepareWrap()
            f_call(a)
            p, w = a.args
            expected = f_expected(a)
            try:
                assert_equal(p, expected)
                assert_equal(w, expected)
            except AssertionError as e:
                # assert_equal produces truly useless error messages
                raise AssertionError("\n".join([
                    "Bad arguments passed in ufunc call",
                    " expected:              {}".format(expected),
                    " __array_prepare__ got: {}".format(p),
                    " __array_wrap__ got:    {}".format(w)
                ]))

        # method not on the out argument
        do_test(lambda a: np.add(a, 0),              lambda a: (a, 0))
        do_test(lambda a: np.add(a, 0, None),        lambda a: (a, 0))
        do_test(lambda a: np.add(a, 0, out=None),    lambda a: (a, 0))
        do_test(lambda a: np.add(a, 0, out=(None,)), lambda a: (a, 0))

        # method on the out argument
        do_test(lambda a: np.add(0, 0, a),           lambda a: (0, 0, a))
        do_test(lambda a: np.add(0, 0, out=a),       lambda a: (0, 0, a))
        do_test(lambda a: np.add(0, 0, out=(a,)),    lambda a: (0, 0, a))

        # Also check the where mask handling:
        do_test(lambda a: np.add(a, 0, where=False), lambda a: (a, 0))
        do_test(lambda a: np.add(0, 0, a, where=False), lambda a: (0, 0, a))

    def test_wrap_with_iterable(self):
        # test fix for bug #1026:

        class with_wrap(np.ndarray):
            __array_priority__ = 10

            def __new__(cls):
                return np.asarray(1).view(cls).copy()

            def __array_wrap__(self, arr, context):
                return arr.view(type(self))

        a = with_wrap()
        x = ncu.multiply(a, (1, 2, 3))
        assert_(isinstance(x, with_wrap))
        assert_array_equal(x, np.array((1, 2, 3)))

    def test_priority_with_scalar(self):
        # test fix for bug #826:

        class A(np.ndarray):
            __array_priority__ = 10

            def __new__(cls):
                return np.asarray(1.0, 'float64').view(cls).copy()

        a = A()
        x = np.float64(1)*a
        assert_(isinstance(x, A))
        assert_array_equal(x, np.array(1))

    def test_old_wrap(self):

        class with_wrap:
            def __array__(self):
                return np.zeros(1)

            def __array_wrap__(self, arr):
                r = with_wrap()
                r.arr = arr
                return r

        a = with_wrap()
        x = ncu.minimum(a, a)
        assert_equal(x.arr, np.zeros(1))

    def test_priority(self):

        class A:
            def __array__(self):
                return np.zeros(1)

            def __array_wrap__(self, arr, context):
                r = type(self)()
                r.arr = arr
                r.context = context
                return r

        class B(A):
            __array_priority__ = 20.

        class C(A):
            __array_priority__ = 40.

        x = np.zeros(1)
        a = A()
        b = B()
        c = C()
        f = ncu.minimum
        assert_(type(f(x, x)) is np.ndarray)
        assert_(type(f(x, a)) is A)
        assert_(type(f(x, b)) is B)
        assert_(type(f(x, c)) is C)
        assert_(type(f(a, x)) is A)
        assert_(type(f(b, x)) is B)
        assert_(type(f(c, x)) is C)

        assert_(type(f(a, a)) is A)
        assert_(type(f(a, b)) is B)
        assert_(type(f(b, a)) is B)
        assert_(type(f(b, b)) is B)
        assert_(type(f(b, c)) is C)
        assert_(type(f(c, b)) is C)
        assert_(type(f(c, c)) is C)

        assert_(type(ncu.exp(a) is A))
        assert_(type(ncu.exp(b) is B))
        assert_(type(ncu.exp(c) is C))

    def test_failing_wrap(self):

        class A:
            def __array__(self):
                return np.zeros(2)

            def __array_wrap__(self, arr, context):
                raise RuntimeError

        a = A()
        assert_raises(RuntimeError, ncu.maximum, a, a)
        assert_raises(RuntimeError, ncu.maximum.reduce, a)

    def test_failing_out_wrap(self):

        singleton = np.array([1.0])

        class Ok(np.ndarray):
            def __array_wrap__(self, obj):
                return singleton

        class Bad(np.ndarray):
            def __array_wrap__(self, obj):
                raise RuntimeError

        ok = np.empty(1).view(Ok)
        bad = np.empty(1).view(Bad)
        # double-free (segfault) of "ok" if "bad" raises an exception
        for i in range(10):
            assert_raises(RuntimeError, ncu.frexp, 1, ok, bad)

    def test_none_wrap(self):
        # Tests that issue #8507 is resolved. Previously, this would segfault

        class A:
            def __array__(self):
                return np.zeros(1)

            def __array_wrap__(self, arr, context=None):
                return None

        a = A()
        assert_equal(ncu.maximum(a, a), None)

    def test_default_prepare(self):

        class with_wrap:
            __array_priority__ = 10

            def __array__(self):
                return np.zeros(1)

            def __array_wrap__(self, arr, context):
                return arr

        a = with_wrap()
        x = ncu.minimum(a, a)
        assert_equal(x, np.zeros(1))
        assert_equal(type(x), np.ndarray)

    @pytest.mark.parametrize("use_where", [True, False])
    def test_prepare(self, use_where):

        class with_prepare(np.ndarray):
            __array_priority__ = 10

            def __array_prepare__(self, arr, context):
                # make sure we can return a new
                return np.array(arr).view(type=with_prepare)

        a = np.array(1).view(type=with_prepare)
        if use_where:
            x = np.add(a, a, where=np.array(True))
        else:
            x = np.add(a, a)
        assert_equal(x, np.array(2))
        assert_equal(type(x), with_prepare)

    @pytest.mark.parametrize("use_where", [True, False])
    def test_prepare_out(self, use_where):

        class with_prepare(np.ndarray):
            __array_priority__ = 10

            def __array_prepare__(self, arr, context):
                return np.array(arr).view(type=with_prepare)

        a = np.array([1]).view(type=with_prepare)
        if use_where:
            x = np.add(a, a, a, where=[True])
        else:
            x = np.add(a, a, a)
        # Returned array is new, because of the strange
        # __array_prepare__ above
        assert_(not np.shares_memory(x, a))
        assert_equal(x, np.array([2]))
        assert_equal(type(x), with_prepare)

    def test_failing_prepare(self):

        class A:
            def __array__(self):
                return np.zeros(1)

            def __array_prepare__(self, arr, context=None):
                raise RuntimeError

        a = A()
        assert_raises(RuntimeError, ncu.maximum, a, a)
        assert_raises(RuntimeError, ncu.maximum, a, a, where=False)

    def test_array_too_many_args(self):

        class A:
            def __array__(self, dtype, context):
                return np.zeros(1)

        a = A()
        assert_raises_regex(TypeError, '2 required positional', np.sum, a)

    def test_ufunc_override(self):
        # check override works even with instance with high priority.
        class A:
            def __array_ufunc__(self, func, method, *inputs, **kwargs):
                return self, func, method, inputs, kwargs

        class MyNDArray(np.ndarray):
            __array_priority__ = 100

        a = A()
        b = np.array([1]).view(MyNDArray)
        res0 = np.multiply(a, b)
        res1 = np.multiply(b, b, out=a)

        # self
        assert_equal(res0[0], a)
        assert_equal(res1[0], a)
        assert_equal(res0[1], np.multiply)
        assert_equal(res1[1], np.multiply)
        assert_equal(res0[2], '__call__')
        assert_equal(res1[2], '__call__')
        assert_equal(res0[3], (a, b))
        assert_equal(res1[3], (b, b))
        assert_equal(res0[4], {})
        assert_equal(res1[4], {'out': (a,)})

    def test_ufunc_override_mro(self):

        # Some multi arg functions for testing.
        def tres_mul(a, b, c):
            return a * b * c

        def quatro_mul(a, b, c, d):
            return a * b * c * d

        # Make these into ufuncs.
        three_mul_ufunc = np.frompyfunc(tres_mul, 3, 1)
        four_mul_ufunc = np.frompyfunc(quatro_mul, 4, 1)

        class A:
            def __array_ufunc__(self, func, method, *inputs, **kwargs):
                return "A"

        class ASub(A):
            def __array_ufunc__(self, func, method, *inputs, **kwargs):
                return "ASub"

        class B:
            def __array_ufunc__(self, func, method, *inputs, **kwargs):
                return "B"

        class C:
            def __init__(self):
                self.count = 0

            def __array_ufunc__(self, func, method, *inputs, **kwargs):
                self.count += 1
                return NotImplemented

        class CSub(C):
            def __array_ufunc__(self, func, method, *inputs, **kwargs):
                self.count += 1
                return NotImplemented

        a = A()
        a_sub = ASub()
        b = B()
        c = C()

        # Standard
        res = np.multiply(a, a_sub)
        assert_equal(res, "ASub")
        res = np.multiply(a_sub, b)
        assert_equal(res, "ASub")

        # With 1 NotImplemented
        res = np.multiply(c, a)
        assert_equal(res, "A")
        assert_equal(c.count, 1)
        # Check our counter works, so we can trust tests below.
        res = np.multiply(c, a)
        assert_equal(c.count, 2)

        # Both NotImplemented.
        c = C()
        c_sub = CSub()
        assert_raises(TypeError, np.multiply, c, c_sub)
        assert_equal(c.count, 1)
        assert_equal(c_sub.count, 1)
        c.count = c_sub.count = 0
        assert_raises(TypeError, np.multiply, c_sub, c)
        assert_equal(c.count, 1)
        assert_equal(c_sub.count, 1)
        c.count = 0
        assert_raises(TypeError, np.multiply, c, c)
        assert_equal(c.count, 1)
        c.count = 0
        assert_raises(TypeError, np.multiply, 2, c)
        assert_equal(c.count, 1)

        # Ternary testing.
        assert_equal(three_mul_ufunc(a, 1, 2), "A")
        assert_equal(three_mul_ufunc(1, a, 2), "A")
        assert_equal(three_mul_ufunc(1, 2, a), "A")

        assert_equal(three_mul_ufunc(a, a, 6), "A")
        assert_equal(three_mul_ufunc(a, 2, a), "A")
        assert_equal(three_mul_ufunc(a, 2, b), "A")
        assert_equal(three_mul_ufunc(a, 2, a_sub), "ASub")
        assert_equal(three_mul_ufunc(a, a_sub, 3), "ASub")
        c.count = 0
        assert_equal(three_mul_ufunc(c, a_sub, 3), "ASub")
        assert_equal(c.count, 1)
        c.count = 0
        assert_equal(three_mul_ufunc(1, a_sub, c), "ASub")
        assert_equal(c.count, 0)

        c.count = 0
        assert_equal(three_mul_ufunc(a, b, c), "A")
        assert_equal(c.count, 0)
        c_sub.count = 0
        assert_equal(three_mul_ufunc(a, b, c_sub), "A")
        assert_equal(c_sub.count, 0)
        assert_equal(three_mul_ufunc(1, 2, b), "B")

        assert_raises(TypeError, three_mul_ufunc, 1, 2, c)
        assert_raises(TypeError, three_mul_ufunc, c_sub, 2, c)
        assert_raises(TypeError, three_mul_ufunc, c_sub, 2, 3)

        # Quaternary testing.
        assert_equal(four_mul_ufunc(a, 1, 2, 3), "A")
        assert_equal(four_mul_ufunc(1, a, 2, 3), "A")
        assert_equal(four_mul_ufunc(1, 1, a, 3), "A")
        assert_equal(four_mul_ufunc(1, 1, 2, a), "A")

        assert_equal(four_mul_ufunc(a, b, 2, 3), "A")
        assert_equal(four_mul_ufunc(1, a, 2, b), "A")
        assert_equal(four_mul_ufunc(b, 1, a, 3), "B")
        assert_equal(four_mul_ufunc(a_sub, 1, 2, a), "ASub")
        assert_equal(four_mul_ufunc(a, 1, 2, a_sub), "ASub")

        c = C()
        c_sub = CSub()
        assert_raises(TypeError, four_mul_ufunc, 1, 2, 3, c)
        assert_equal(c.count, 1)
        c.count = 0
        assert_raises(TypeError, four_mul_ufunc, 1, 2, c_sub, c)
        assert_equal(c_sub.count, 1)
        assert_equal(c.count, 1)
        c2 = C()
        c.count = c_sub.count = 0
        assert_raises(TypeError, four_mul_ufunc, 1, c, c_sub, c2)
        assert_equal(c_sub.count, 1)
        assert_equal(c.count, 1)
        assert_equal(c2.count, 0)
        c.count = c2.count = c_sub.count = 0
        assert_raises(TypeError, four_mul_ufunc, c2, c, c_sub, c)
        assert_equal(c_sub.count, 1)
        assert_equal(c.count, 0)
        assert_equal(c2.count, 1)

    def test_ufunc_override_methods(self):

        class A:
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return self, ufunc, method, inputs, kwargs

        # __call__
        a = A()
        with assert_raises(TypeError):
            np.multiply.__call__(1, a, foo='bar', answer=42)
        res = np.multiply.__call__(1, a, subok='bar', where=42)
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], '__call__')
        assert_equal(res[3], (1, a))
        assert_equal(res[4], {'subok': 'bar', 'where': 42})

        # __call__, wrong args
        assert_raises(TypeError, np.multiply, a)
        assert_raises(TypeError, np.multiply, a, a, a, a)
        assert_raises(TypeError, np.multiply, a, a, sig='a', signature='a')
        assert_raises(TypeError, ncu_tests.inner1d, a, a, axis=0, axes=[0, 0])

        # reduce, positional args
        res = np.multiply.reduce(a, 'axis0', 'dtype0', 'out0', 'keep0')
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], 'reduce')
        assert_equal(res[3], (a,))
        assert_equal(res[4], {'dtype':'dtype0',
                              'out': ('out0',),
                              'keepdims': 'keep0',
                              'axis': 'axis0'})

        # reduce, kwargs
        res = np.multiply.reduce(a, axis='axis0', dtype='dtype0', out='out0',
                                 keepdims='keep0', initial='init0',
                                 where='where0')
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], 'reduce')
        assert_equal(res[3], (a,))
        assert_equal(res[4], {'dtype':'dtype0',
                              'out': ('out0',),
                              'keepdims': 'keep0',
                              'axis': 'axis0',
                              'initial': 'init0',
                              'where': 'where0'})

        # reduce, output equal to None removed, but not other explicit ones,
        # even if they are at their default value.
        res = np.multiply.reduce(a, 0, None, None, False)
        assert_equal(res[4], {'axis': 0, 'dtype': None, 'keepdims': False})
        res = np.multiply.reduce(a, out=None, axis=0, keepdims=True)
        assert_equal(res[4], {'axis': 0, 'keepdims': True})
        res = np.multiply.reduce(a, None, out=(None,), dtype=None)
        assert_equal(res[4], {'axis': None, 'dtype': None})
        res = np.multiply.reduce(a, 0, None, None, False, 2, True)
        assert_equal(res[4], {'axis': 0, 'dtype': None, 'keepdims': False,
                              'initial': 2, 'where': True})
        # np._NoValue ignored for initial
        res = np.multiply.reduce(a, 0, None, None, False,
                                 np._NoValue, True)
        assert_equal(res[4], {'axis': 0, 'dtype': None, 'keepdims': False,
                              'where': True})
        # None kept for initial, True for where.
        res = np.multiply.reduce(a, 0, None, None, False, None, True)
        assert_equal(res[4], {'axis': 0, 'dtype': None, 'keepdims': False,
                              'initial': None, 'where': True})

        # reduce, wrong args
        assert_raises(ValueError, np.multiply.reduce, a, out=())
        assert_raises(ValueError, np.multiply.reduce, a, out=('out0', 'out1'))
        assert_raises(TypeError, np.multiply.reduce, a, 'axis0', axis='axis0')

        # accumulate, pos args
        res = np.multiply.accumulate(a, 'axis0', 'dtype0', 'out0')
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], 'accumulate')
        assert_equal(res[3], (a,))
        assert_equal(res[4], {'dtype':'dtype0',
                              'out': ('out0',),
                              'axis': 'axis0'})

        # accumulate, kwargs
        res = np.multiply.accumulate(a, axis='axis0', dtype='dtype0',
                                     out='out0')
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], 'accumulate')
        assert_equal(res[3], (a,))
        assert_equal(res[4], {'dtype':'dtype0',
                              'out': ('out0',),
                              'axis': 'axis0'})

        # accumulate, output equal to None removed.
        res = np.multiply.accumulate(a, 0, None, None)
        assert_equal(res[4], {'axis': 0, 'dtype': None})
        res = np.multiply.accumulate(a, out=None, axis=0, dtype='dtype1')
        assert_equal(res[4], {'axis': 0, 'dtype': 'dtype1'})
        res = np.multiply.accumulate(a, None, out=(None,), dtype=None)
        assert_equal(res[4], {'axis': None, 'dtype': None})

        # accumulate, wrong args
        assert_raises(ValueError, np.multiply.accumulate, a, out=())
        assert_raises(ValueError, np.multiply.accumulate, a,
                      out=('out0', 'out1'))
        assert_raises(TypeError, np.multiply.accumulate, a,
                      'axis0', axis='axis0')

        # reduceat, pos args
        res = np.multiply.reduceat(a, [4, 2], 'axis0', 'dtype0', 'out0')
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], 'reduceat')
        assert_equal(res[3], (a, [4, 2]))
        assert_equal(res[4], {'dtype':'dtype0',
                              'out': ('out0',),
                              'axis': 'axis0'})

        # reduceat, kwargs
        res = np.multiply.reduceat(a, [4, 2], axis='axis0', dtype='dtype0',
                                   out='out0')
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], 'reduceat')
        assert_equal(res[3], (a, [4, 2]))
        assert_equal(res[4], {'dtype':'dtype0',
                              'out': ('out0',),
                              'axis': 'axis0'})

        # reduceat, output equal to None removed.
        res = np.multiply.reduceat(a, [4, 2], 0, None, None)
        assert_equal(res[4], {'axis': 0, 'dtype': None})
        res = np.multiply.reduceat(a, [4, 2], axis=None, out=None, dtype='dt')
        assert_equal(res[4], {'axis': None, 'dtype': 'dt'})
        res = np.multiply.reduceat(a, [4, 2], None, None, out=(None,))
        assert_equal(res[4], {'axis': None, 'dtype': None})

        # reduceat, wrong args
        assert_raises(ValueError, np.multiply.reduce, a, [4, 2], out=())
        assert_raises(ValueError, np.multiply.reduce, a, [4, 2],
                      out=('out0', 'out1'))
        assert_raises(TypeError, np.multiply.reduce, a, [4, 2],
                      'axis0', axis='axis0')

        # outer
        res = np.multiply.outer(a, 42)
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], 'outer')
        assert_equal(res[3], (a, 42))
        assert_equal(res[4], {})

        # outer, wrong args
        assert_raises(TypeError, np.multiply.outer, a)
        assert_raises(TypeError, np.multiply.outer, a, a, a, a)
        assert_raises(TypeError, np.multiply.outer, a, a, sig='a', signature='a')

        # at
        res = np.multiply.at(a, [4, 2], 'b0')
        assert_equal(res[0], a)
        assert_equal(res[1], np.multiply)
        assert_equal(res[2], 'at')
        assert_equal(res[3], (a, [4, 2], 'b0'))

        # at, wrong args
        assert_raises(TypeError, np.multiply.at, a)
        assert_raises(TypeError, np.multiply.at, a, a, a, a)

    def test_ufunc_override_out(self):

        class A:
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return kwargs

        class B:
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return kwargs

        a = A()
        b = B()
        res0 = np.multiply(a, b, 'out_arg')
        res1 = np.multiply(a, b, out='out_arg')
        res2 = np.multiply(2, b, 'out_arg')
        res3 = np.multiply(3, b, out='out_arg')
        res4 = np.multiply(a, 4, 'out_arg')
        res5 = np.multiply(a, 5, out='out_arg')

        assert_equal(res0['out'][0], 'out_arg')
        assert_equal(res1['out'][0], 'out_arg')
        assert_equal(res2['out'][0], 'out_arg')
        assert_equal(res3['out'][0], 'out_arg')
        assert_equal(res4['out'][0], 'out_arg')
        assert_equal(res5['out'][0], 'out_arg')

        # ufuncs with multiple output modf and frexp.
        res6 = np.modf(a, 'out0', 'out1')
        res7 = np.frexp(a, 'out0', 'out1')
        assert_equal(res6['out'][0], 'out0')
        assert_equal(res6['out'][1], 'out1')
        assert_equal(res7['out'][0], 'out0')
        assert_equal(res7['out'][1], 'out1')

        # While we're at it, check that default output is never passed on.
        assert_(np.sin(a, None) == {})
        assert_(np.sin(a, out=None) == {})
        assert_(np.sin(a, out=(None,)) == {})
        assert_(np.modf(a, None) == {})
        assert_(np.modf(a, None, None) == {})
        assert_(np.modf(a, out=(None, None)) == {})
        with assert_raises(TypeError):
            # Out argument must be tuple, since there are multiple outputs.
            np.modf(a, out=None)

        # don't give positional and output argument, or too many arguments.
        # wrong number of arguments in the tuple is an error too.
        assert_raises(TypeError, np.multiply, a, b, 'one', out='two')
        assert_raises(TypeError, np.multiply, a, b, 'one', 'two')
        assert_raises(ValueError, np.multiply, a, b, out=('one', 'two'))
        assert_raises(TypeError, np.multiply, a, out=())
        assert_raises(TypeError, np.modf, a, 'one', out=('two', 'three'))
        assert_raises(TypeError, np.modf, a, 'one', 'two', 'three')
        assert_raises(ValueError, np.modf, a, out=('one', 'two', 'three'))
        assert_raises(ValueError, np.modf, a, out=('one',))

    def test_ufunc_override_exception(self):

        class A:
            def __array_ufunc__(self, *a, **kwargs):
                raise ValueError("oops")

        a = A()
        assert_raises(ValueError, np.negative, 1, out=a)
        assert_raises(ValueError, np.negative, a)
        assert_raises(ValueError, np.divide, 1., a)

    def test_ufunc_override_not_implemented(self):

        class A:
            def __array_ufunc__(self, *args, **kwargs):
                return NotImplemented

        msg = ("operand type(s) all returned NotImplemented from "
               "__array_ufunc__(<ufunc 'negative'>, '__call__', <*>): 'A'")
        with assert_raises_regex(TypeError, fnmatch.translate(msg)):
            np.negative(A())

        msg = ("operand type(s) all returned NotImplemented from "
               "__array_ufunc__(<ufunc 'add'>, '__call__', <*>, <object *>, "
               "out=(1,)): 'A', 'object', 'int'")
        with assert_raises_regex(TypeError, fnmatch.translate(msg)):
            np.add(A(), object(), out=1)

    def test_ufunc_override_disabled(self):

        class OptOut:
            __array_ufunc__ = None

        opt_out = OptOut()

        # ufuncs always raise
        msg = "operand 'OptOut' does not support ufuncs"
        with assert_raises_regex(TypeError, msg):
            np.add(opt_out, 1)
        with assert_raises_regex(TypeError, msg):
            np.add(1, opt_out)
        with assert_raises_regex(TypeError, msg):
            np.negative(opt_out)

        # opt-outs still hold even when other arguments have pathological
        # __array_ufunc__ implementations

        class GreedyArray:
            def __array_ufunc__(self, *args, **kwargs):
                return self

        greedy = GreedyArray()
        assert_(np.negative(greedy) is greedy)
        with assert_raises_regex(TypeError, msg):
            np.add(greedy, opt_out)
        with assert_raises_regex(TypeError, msg):
            np.add(greedy, 1, out=opt_out)

    def test_gufunc_override(self):
        # gufunc are just ufunc instances, but follow a different path,
        # so check __array_ufunc__ overrides them properly.
        class A:
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                return self, ufunc, method, inputs, kwargs

        inner1d = ncu_tests.inner1d
        a = A()
        res = inner1d(a, a)
        assert_equal(res[0], a)
        assert_equal(res[1], inner1d)
        assert_equal(res[2], '__call__')
        assert_equal(res[3], (a, a))
        assert_equal(res[4], {})

        res = inner1d(1, 1, out=a)
        assert_equal(res[0], a)
        assert_equal(res[1], inner1d)
        assert_equal(res[2], '__call__')
        assert_equal(res[3], (1, 1))
        assert_equal(res[4], {'out': (a,)})

        # wrong number of arguments in the tuple is an error too.
        assert_raises(TypeError, inner1d, a, out='two')
        assert_raises(TypeError, inner1d, a, a, 'one', out='two')
        assert_raises(TypeError, inner1d, a, a, 'one', 'two')
        assert_raises(ValueError, inner1d, a, a, out=('one', 'two'))
        assert_raises(ValueError, inner1d, a, a, out=())

    def test_ufunc_override_with_super(self):
        # NOTE: this class is used in doc/source/user/basics.subclassing.rst
        # if you make any changes here, do update it there too.
        class A(np.ndarray):
            def __array_ufunc__(self, ufunc, method, *inputs, out=None, **kwargs):
                args = []
                in_no = []
                for i, input_ in enumerate(inputs):
                    if isinstance(input_, A):
                        in_no.append(i)
                        args.append(input_.view(np.ndarray))
                    else:
                        args.append(input_)

                outputs = out
                out_no = []
                if outputs:
                    out_args = []
                    for j, output in enumerate(outputs):
                        if isinstance(output, A):
                            out_no.append(j)
                            out_args.append(output.view(np.ndarray))
                        else:
                            out_args.append(output)
                    kwargs['out'] = tuple(out_args)
                else:
                    outputs = (None,) * ufunc.nout

                info = {}
                if in_no:
                    info['inputs'] = in_no
                if out_no:
                    info['outputs'] = out_no

                results = super().__array_ufunc__(ufunc, method,
                                                  *args, **kwargs)
                if results is NotImplemented:
                    return NotImplemented

                if method == 'at':
                    if isinstance(inputs[0], A):
                        inputs[0].info = info
                    return

                if ufunc.nout == 1:
                    results = (results,)

                results = tuple((np.asarray(result).view(A)
                                 if output is None else output)
                                for result, output in zip(results, outputs))
                if results and isinstance(results[0], A):
                    results[0].info = info

                return results[0] if len(results) == 1 else results

        class B:
            def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
                if any(isinstance(input_, A) for input_ in inputs):
                    return "A!"
                else:
                    return NotImplemented

        d = np.arange(5.)
        # 1 input, 1 output
        a = np.arange(5.).view(A)
        b = np.sin(a)
        check = np.sin(d)
        assert_(np.all(check == b))
        assert_equal(b.info, {'inputs': [0]})
        b = np.sin(d, out=(a,))
        assert_(np.all(check == b))
        assert_equal(b.info, {'outputs': [0]})
        assert_(b is a)
        a = np.arange(5.).view(A)
        b = np.sin(a, out=a)
        assert_(np.all(check == b))
        assert_equal(b.info, {'inputs': [0], 'outputs': [0]})

        # 1 input, 2 outputs
        a = np.arange(5.).view(A)
        b1, b2 = np.modf(a)
        assert_equal(b1.info, {'inputs': [0]})
        b1, b2 = np.modf(d, out=(None, a))
        assert_(b2 is a)
        assert_equal(b1.info, {'outputs': [1]})
        a = np.arange(5.).view(A)
        b = np.arange(5.).view(A)
        c1, c2 = np.modf(a, out=(a, b))
        assert_(c1 is a)
        assert_(c2 is b)
        assert_equal(c1.info, {'inputs': [0], 'outputs': [0, 1]})

        # 2 input, 1 output
        a = np.arange(5.).view(A)
        b = np.arange(5.).view(A)
        c = np.add(a, b, out=a)
        assert_(c is a)
        assert_equal(c.info, {'inputs': [0, 1], 'outputs': [0]})
        # some tests with a non-ndarray subclass
        a = np.arange(5.)
        b = B()
        assert_(a.__array_ufunc__(np.add, '__call__', a, b) is NotImplemented)
        assert_(b.__array_ufunc__(np.add, '__call__', a, b) is NotImplemented)
        assert_raises(TypeError, np.add, a, b)
        a = a.view(A)
        assert_(a.__array_ufunc__(np.add, '__call__', a, b) is NotImplemented)
        assert_(b.__array_ufunc__(np.add, '__call__', a, b) == "A!")
        assert_(np.add(a, b) == "A!")
        # regression check for gh-9102 -- tests ufunc.reduce implicitly.
        d = np.array([[1, 2, 3], [1, 2, 3]])
        a = d.view(A)
        c = a.any()
        check = d.any()
        assert_equal(c, check)
        assert_(c.info, {'inputs': [0]})
        c = a.max()
        check = d.max()
        assert_equal(c, check)
        assert_(c.info, {'inputs': [0]})
        b = np.array(0).view(A)
        c = a.max(out=b)
        assert_equal(c, check)
        assert_(c is b)
        assert_(c.info, {'inputs': [0], 'outputs': [0]})
        check = a.max(axis=0)
        b = np.zeros_like(check).view(A)
        c = a.max(axis=0, out=b)
        assert_equal(c, check)
        assert_(c is b)
        assert_(c.info, {'inputs': [0], 'outputs': [0]})
        # simple explicit tests of reduce, accumulate, reduceat
        check = np.add.reduce(d, axis=1)
        c = np.add.reduce(a, axis=1)
        assert_equal(c, check)
        assert_(c.info, {'inputs': [0]})
        b = np.zeros_like(c)
        c = np.add.reduce(a, 1, None, b)
        assert_equal(c, check)
        assert_(c is b)
        assert_(c.info, {'inputs': [0], 'outputs': [0]})
        check = np.add.accumulate(d, axis=0)
        c = np.add.accumulate(a, axis=0)
        assert_equal(c, check)
        assert_(c.info, {'inputs': [0]})
        b = np.zeros_like(c)
        c = np.add.accumulate(a, 0, None, b)
        assert_equal(c, check)
        assert_(c is b)
        assert_(c.info, {'inputs': [0], 'outputs': [0]})
        indices = [0, 2, 1]
        check = np.add.reduceat(d, indices, axis=1)
        c = np.add.reduceat(a, indices, axis=1)
        assert_equal(c, check)
        assert_(c.info, {'inputs': [0]})
        b = np.zeros_like(c)
        c = np.add.reduceat(a, indices, 1, None, b)
        assert_equal(c, check)
        assert_(c is b)
        assert_(c.info, {'inputs': [0], 'outputs': [0]})
        # and a few tests for at
        d = np.array([[1, 2, 3], [1, 2, 3]])
        check = d.copy()
        a = d.copy().view(A)
        np.add.at(check, ([0, 1], [0, 2]), 1.)
        np.add.at(a, ([0, 1], [0, 2]), 1.)
        assert_equal(a, check)
        assert_(a.info, {'inputs': [0]})
        b = np.array(1.).view(A)
        a = d.copy().view(A)
        np.add.at(a, ([0, 1], [0, 2]), b)
        assert_equal(a, check)
        assert_(a.info, {'inputs': [0, 2]})


class TestChoose:
    def test_mixed(self):
        c = np.array([True, True])
        a = np.array([True, True])
        assert_equal(np.choose(c, (a, 1)), np.array([1, 1]))


class TestRationalFunctions:
    def test_lcm(self):
        self._test_lcm_inner(np.int16)
        self._test_lcm_inner(np.uint16)

    def test_lcm_object(self):
        self._test_lcm_inner(np.object_)

    def test_gcd(self):
        self._test_gcd_inner(np.int16)
        self._test_lcm_inner(np.uint16)

    def test_gcd_object(self):
        self._test_gcd_inner(np.object_)

    def _test_lcm_inner(self, dtype):
        # basic use
        a = np.array([12, 120], dtype=dtype)
        b = np.array([20, 200], dtype=dtype)
        assert_equal(np.lcm(a, b), [60, 600])

        if not issubclass(dtype, np.unsignedinteger):
            # negatives are ignored
            a = np.array([12, -12,  12, -12], dtype=dtype)
            b = np.array([20,  20, -20, -20], dtype=dtype)
            assert_equal(np.lcm(a, b), [60]*4)

        # reduce
        a = np.array([3, 12, 20], dtype=dtype)
        assert_equal(np.lcm.reduce([3, 12, 20]), 60)

        # broadcasting, and a test including 0
        a = np.arange(6).astype(dtype)
        b = 20
        assert_equal(np.lcm(a, b), [0, 20, 20, 60, 20, 20])

    def _test_gcd_inner(self, dtype):
        # basic use
        a = np.array([12, 120], dtype=dtype)
        b = np.array([20, 200], dtype=dtype)
        assert_equal(np.gcd(a, b), [4, 40])

        if not issubclass(dtype, np.unsignedinteger):
            # negatives are ignored
            a = np.array([12, -12,  12, -12], dtype=dtype)
            b = np.array([20,  20, -20, -20], dtype=dtype)
            assert_equal(np.gcd(a, b), [4]*4)

        # reduce
        a = np.array([15, 25, 35], dtype=dtype)
        assert_equal(np.gcd.reduce(a), 5)

        # broadcasting, and a test including 0
        a = np.arange(6).astype(dtype)
        b = 20
        assert_equal(np.gcd(a, b), [20,  1,  2,  1,  4,  5])

    def test_lcm_overflow(self):
        # verify that we don't overflow when a*b does overflow
        big = np.int32(np.iinfo(np.int32).max // 11)
        a = 2*big
        b = 5*big
        assert_equal(np.lcm(a, b), 10*big)

    def test_gcd_overflow(self):
        for dtype in (np.int32, np.int64):
            # verify that we don't overflow when taking abs(x)
            # not relevant for lcm, where the result is unrepresentable anyway
            a = dtype(np.iinfo(dtype).min)  # negative power of two
            q = -(a // 4)
            assert_equal(np.gcd(a,  q*3), q)
            assert_equal(np.gcd(a, -q*3), q)

    def test_decimal(self):
        from decimal import Decimal
        a = np.array([1,  1, -1, -1]) * Decimal('0.20')
        b = np.array([1, -1,  1, -1]) * Decimal('0.12')

        assert_equal(np.gcd(a, b), 4*[Decimal('0.04')])
        assert_equal(np.lcm(a, b), 4*[Decimal('0.60')])

    def test_float(self):
        # not well-defined on float due to rounding errors
        assert_raises(TypeError, np.gcd, 0.3, 0.4)
        assert_raises(TypeError, np.lcm, 0.3, 0.4)

    def test_builtin_long(self):
        # sanity check that array coercion is alright for builtin longs
        assert_equal(np.array(2**200).item(), 2**200)

        # expressed as prime factors
        a = np.array(2**100 * 3**5)
        b = np.array([2**100 * 5**7, 2**50 * 3**10])
        assert_equal(np.gcd(a, b), [2**100,               2**50 * 3**5])
        assert_equal(np.lcm(a, b), [2**100 * 3**5 * 5**7, 2**100 * 3**10])

        assert_equal(np.gcd(2**100, 3**100), 1)


class TestRoundingFunctions:

    def test_object_direct(self):
        """ test direct implementation of these magic methods """
        class C:
            def __floor__(self):
                return 1
            def __ceil__(self):
                return 2
            def __trunc__(self):
                return 3

        arr = np.array([C(), C()])
        assert_equal(np.floor(arr), [1, 1])
        assert_equal(np.ceil(arr),  [2, 2])
        assert_equal(np.trunc(arr), [3, 3])

    def test_object_indirect(self):
        """ test implementations via __float__ """
        class C:
            def __float__(self):
                return -2.5

        arr = np.array([C(), C()])
        assert_equal(np.floor(arr), [-3, -3])
        assert_equal(np.ceil(arr),  [-2, -2])
        with pytest.raises(TypeError):
            np.trunc(arr)  # consistent with math.trunc

    def test_fraction(self):
        f = Fraction(-4, 3)
        assert_equal(np.floor(f), -2)
        assert_equal(np.ceil(f), -1)
        assert_equal(np.trunc(f), -1)


class TestComplexFunctions:
    funcs = [np.arcsin,  np.arccos,  np.arctan, np.arcsinh, np.arccosh,
             np.arctanh, np.sin,     np.cos,    np.tan,     np.exp,
             np.exp2,    np.log,     np.sqrt,   np.log10,   np.log2,
             np.log1p]

    def test_it(self):
        for f in self.funcs:
            if f is np.arccosh:
                x = 1.5
            else:
                x = .5
            fr = f(x)
            fz = f(complex(x))
            assert_almost_equal(fz.real, fr, err_msg='real part %s' % f)
            assert_almost_equal(fz.imag, 0., err_msg='imag part %s' % f)

    @pytest.mark.xfail(IS_WASM, reason="doesn't work")
    def test_precisions_consistent(self):
        z = 1 + 1j
        for f in self.funcs:
            fcf = f(np.csingle(z))
            fcd = f(np.cdouble(z))
            fcl = f(np.clongdouble(z))
            assert_almost_equal(fcf, fcd, decimal=6, err_msg='fch-fcd %s' % f)
            assert_almost_equal(fcl, fcd, decimal=15, err_msg='fch-fcl %s' % f)

    @pytest.mark.xfail(IS_WASM, reason="doesn't work")
    def test_branch_cuts(self):
        # check branch cuts and continuity on them
        _check_branch_cut(np.log,   -0.5, 1j, 1, -1, True)
        _check_branch_cut(np.log2,  -0.5, 1j, 1, -1, True)
        _check_branch_cut(np.log10, -0.5, 1j, 1, -1, True)
        _check_branch_cut(np.log1p, -1.5, 1j, 1, -1, True)
        _check_branch_cut(np.sqrt,  -0.5, 1j, 1, -1, True)

        _check_branch_cut(np.arcsin, [ -2, 2],   [1j, 1j], 1, -1, True)
        _check_branch_cut(np.arccos, [ -2, 2],   [1j, 1j], 1, -1, True)
        _check_branch_cut(np.arctan, [0-2j, 2j],  [1,  1], -1, 1, True)

        _check_branch_cut(np.arcsinh, [0-2j,  2j], [1,   1], -1, 1, True)
        _check_branch_cut(np.arccosh, [ -1, 0.5], [1j,  1j], 1, -1, True)
        _check_branch_cut(np.arctanh, [ -2,   2], [1j, 1j], 1, -1, True)

        # check against bogus branch cuts: assert continuity between quadrants
        _check_branch_cut(np.arcsin, [0-2j, 2j], [ 1,  1], 1, 1)
        _check_branch_cut(np.arccos, [0-2j, 2j], [ 1,  1], 1, 1)
        _check_branch_cut(np.arctan, [ -2,  2], [1j, 1j], 1, 1)

        _check_branch_cut(np.arcsinh, [ -2,  2, 0], [1j, 1j, 1], 1, 1)
        _check_branch_cut(np.arccosh, [0-2j, 2j, 2], [1,  1,  1j], 1, 1)
        _check_branch_cut(np.arctanh, [0-2j, 2j, 0], [1,  1,  1j], 1, 1)

    @pytest.mark.xfail(IS_WASM, reason="doesn't work")
    def test_branch_cuts_complex64(self):
        # check branch cuts and continuity on them
        _check_branch_cut(np.log,   -0.5, 1j, 1, -1, True, np.complex64)
        _check_branch_cut(np.log2,  -0.5, 1j, 1, -1, True, np.complex64)
        _check_branch_cut(np.log10, -0.5, 1j, 1, -1, True, np.complex64)
        _check_branch_cut(np.log1p, -1.5, 1j, 1, -1, True, np.complex64)
        _check_branch_cut(np.sqrt,  -0.5, 1j, 1, -1, True, np.complex64)

        _check_branch_cut(np.arcsin, [ -2, 2],   [1j, 1j], 1, -1, True, np.complex64)
        _check_branch_cut(np.arccos, [ -2, 2],   [1j, 1j], 1, -1, True, np.complex64)
        _check_branch_cut(np.arctan, [0-2j, 2j],  [1,  1], -1, 1, True, np.complex64)

        _check_branch_cut(np.arcsinh, [0-2j,  2j], [1,   1], -1, 1, True, np.complex64)
        _check_branch_cut(np.arccosh, [ -1, 0.5], [1j,  1j], 1, -1, True, np.complex64)
        _check_branch_cut(np.arctanh, [ -2,   2], [1j, 1j], 1, -1, True, np.complex64)

        # check against bogus branch cuts: assert continuity between quadrants
        _check_branch_cut(np.arcsin, [0-2j, 2j], [ 1,  1], 1, 1, False, np.complex64)
        _check_branch_cut(np.arccos, [0-2j, 2j], [ 1,  1], 1, 1, False, np.complex64)
        _check_branch_cut(np.arctan, [ -2,  2], [1j, 1j], 1, 1, False, np.complex64)

        _check_branch_cut(np.arcsinh, [ -2,  2, 0], [1j, 1j, 1], 1, 1, False, np.complex64)
        _check_branch_cut(np.arccosh, [0-2j, 2j, 2], [1,  1,  1j], 1, 1, False, np.complex64)
        _check_branch_cut(np.arctanh, [0-2j, 2j, 0], [1,  1,  1j], 1, 1, False, np.complex64)

    def test_against_cmath(self):
        import cmath

        points = [-1-1j, -1+1j, +1-1j, +1+1j]
        name_map = {'arcsin': 'asin', 'arccos': 'acos', 'arctan': 'atan',
                    'arcsinh': 'asinh', 'arccosh': 'acosh', 'arctanh': 'atanh'}
        atol = 4*np.finfo(complex).eps
        for func in self.funcs:
            fname = func.__name__.split('.')[-1]
            cname = name_map.get(fname, fname)
            try:
                cfunc = getattr(cmath, cname)
            except AttributeError:
                continue
            for p in points:
                a = complex(func(np.complex_(p)))
                b = cfunc(p)
                assert_(abs(a - b) < atol, "%s %s: %s; cmath: %s" % (fname, p, a, b))

    @pytest.mark.xfail(IS_WASM, reason="doesn't work")
    @pytest.mark.parametrize('dtype', [np.complex64, np.complex_, np.longcomplex])
    def test_loss_of_precision(self, dtype):
        """Check loss of precision in complex arc* functions"""

        # Check against known-good functions

        info = np.finfo(dtype)
        real_dtype = dtype(0.).real.dtype
        eps = info.eps

        def check(x, rtol):
            x = x.astype(real_dtype)

            z = x.astype(dtype)
            d = np.absolute(np.arcsinh(x)/np.arcsinh(z).real - 1)
            assert_(np.all(d < rtol), (np.argmax(d), x[np.argmax(d)], d.max(),
                                      'arcsinh'))

            z = (1j*x).astype(dtype)
            d = np.absolute(np.arcsinh(x)/np.arcsin(z).imag - 1)
            assert_(np.all(d < rtol), (np.argmax(d), x[np.argmax(d)], d.max(),
                                      'arcsin'))

            z = x.astype(dtype)
            d = np.absolute(np.arctanh(x)/np.arctanh(z).real - 1)
            assert_(np.all(d < rtol), (np.argmax(d), x[np.argmax(d)], d.max(),
                                      'arctanh'))

            z = (1j*x).astype(dtype)
            d = np.absolute(np.arctanh(x)/np.arctan(z).imag - 1)
            assert_(np.all(d < rtol), (np.argmax(d), x[np.argmax(d)], d.max(),
                                      'arctan'))

        # The switchover was chosen as 1e-3; hence there can be up to
        # ~eps/1e-3 of relative cancellation error before it

        x_series = np.logspace(-20, -3.001, 200)
        x_basic = np.logspace(-2.999, 0, 10, endpoint=False)

        if dtype is np.longcomplex:
            if bad_arcsinh():
                pytest.skip("Trig functions of np.longcomplex values known "
                            "to be inaccurate on aarch64 and PPC for some "
                            "compilation configurations.")
            # It's not guaranteed that the system-provided arc functions
            # are accurate down to a few epsilons. (Eg. on Linux 64-bit)
            # So, give more leeway for long complex tests here:
            check(x_series, 50.0*eps)
        else:
            check(x_series, 2.1*eps)
        check(x_basic, 2.0*eps/1e-3)

        # Check a few points

        z = np.array([1e-5*(1+1j)], dtype=dtype)
        p = 9.999999999333333333e-6 + 1.000000000066666666e-5j
        d = np.absolute(1-np.arctanh(z)/p)
        assert_(np.all(d < 1e-15))

        p = 1.0000000000333333333e-5 + 9.999999999666666667e-6j
        d = np.absolute(1-np.arcsinh(z)/p)
        assert_(np.all(d < 1e-15))

        p = 9.999999999333333333e-6j + 1.000000000066666666e-5
        d = np.absolute(1-np.arctan(z)/p)
        assert_(np.all(d < 1e-15))

        p = 1.0000000000333333333e-5j + 9.999999999666666667e-6
        d = np.absolute(1-np.arcsin(z)/p)
        assert_(np.all(d < 1e-15))

        # Check continuity across switchover points

        def check(func, z0, d=1):
            z0 = np.asarray(z0, dtype=dtype)
            zp = z0 + abs(z0) * d * eps * 2
            zm = z0 - abs(z0) * d * eps * 2
            assert_(np.all(zp != zm), (zp, zm))

            # NB: the cancellation error at the switchover is at least eps
            good = (abs(func(zp) - func(zm)) < 2*eps)
            assert_(np.all(good), (func, z0[~good]))

        for func in (np.arcsinh, np.arcsinh, np.arcsin, np.arctanh, np.arctan):
            pts = [rp+1j*ip for rp in (-1e-3, 0, 1e-3) for ip in(-1e-3, 0, 1e-3)
                   if rp != 0 or ip != 0]
            check(func, pts, 1)
            check(func, pts, 1j)
            check(func, pts, 1+1j)

    @np.errstate(all="ignore")
    def test_promotion_corner_cases(self):
        for func in self.funcs:
            assert func(np.float16(1)).dtype == np.float16
            # Integer to low precision float promotion is a dubious choice:
            assert func(np.uint8(1)).dtype == np.float16
            assert func(np.int16(1)).dtype == np.float32


class TestAttributes:
    def test_attributes(self):
        add = ncu.add
        assert_equal(add.__name__, 'add')
        assert_(add.ntypes >= 18)  # don't fail if types added
        assert_('ii->i' in add.types)
        assert_equal(add.nin, 2)
        assert_equal(add.nout, 1)
        assert_equal(add.identity, 0)

    def test_doc(self):
        # don't bother checking the long list of kwargs, which are likely to
        # change
        assert_(ncu.add.__doc__.startswith(
            "add(x1, x2, /, out=None, *, where=True"))
        assert_(ncu.frexp.__doc__.startswith(
            "frexp(x[, out1, out2], / [, out=(None, None)], *, where=True"))


class TestSubclass:

    def test_subclass_op(self):

        class simple(np.ndarray):
            def __new__(subtype, shape):
                self = np.ndarray.__new__(subtype, shape, dtype=object)
                self.fill(0)
                return self

        a = simple((3, 4))
        assert_equal(a+a, a)


class TestFrompyfunc:

    def test_identity(self):
        def mul(a, b):
            return a * b

        # with identity=value
        mul_ufunc = np.frompyfunc(mul, nin=2, nout=1, identity=1)
        assert_equal(mul_ufunc.reduce([2, 3, 4]), 24)
        assert_equal(mul_ufunc.reduce(np.ones((2, 2)), axis=(0, 1)), 1)
        assert_equal(mul_ufunc.reduce([]), 1)

        # with identity=None (reorderable)
        mul_ufunc = np.frompyfunc(mul, nin=2, nout=1, identity=None)
        assert_equal(mul_ufunc.reduce([2, 3, 4]), 24)
        assert_equal(mul_ufunc.reduce(np.ones((2, 2)), axis=(0, 1)), 1)
        assert_raises(ValueError, lambda: mul_ufunc.reduce([]))

        # with no identity (not reorderable)
        mul_ufunc = np.frompyfunc(mul, nin=2, nout=1)
        assert_equal(mul_ufunc.reduce([2, 3, 4]), 24)
        assert_raises(ValueError, lambda: mul_ufunc.reduce(np.ones((2, 2)), axis=(0, 1)))
        assert_raises(ValueError, lambda: mul_ufunc.reduce([]))


def _check_branch_cut(f, x0, dx, re_sign=1, im_sign=-1, sig_zero_ok=False,
                      dtype=complex):
    """
    Check for a branch cut in a function.

    Assert that `x0` lies on a branch cut of function `f` and `f` is
    continuous from the direction `dx`.

    Parameters
    ----------
    f : func
        Function to check
    x0 : array-like
        Point on branch cut
    dx : array-like
        Direction to check continuity in
    re_sign, im_sign : {1, -1}
        Change of sign of the real or imaginary part expected
    sig_zero_ok : bool
        Whether to check if the branch cut respects signed zero (if applicable)
    dtype : dtype
        Dtype to check (should be complex)

    """
    x0 = np.atleast_1d(x0).astype(dtype)
    dx = np.atleast_1d(dx).astype(dtype)

    if np.dtype(dtype).char == 'F':
        scale = np.finfo(dtype).eps * 1e2
        atol = np.float32(1e-2)
    else:
        scale = np.finfo(dtype).eps * 1e3
        atol = 1e-4

    y0 = f(x0)
    yp = f(x0 + dx*scale*np.absolute(x0)/np.absolute(dx))
    ym = f(x0 - dx*scale*np.absolute(x0)/np.absolute(dx))

    assert_(np.all(np.absolute(y0.real - yp.real) < atol), (y0, yp))
    assert_(np.all(np.absolute(y0.imag - yp.imag) < atol), (y0, yp))
    assert_(np.all(np.absolute(y0.real - ym.real*re_sign) < atol), (y0, ym))
    assert_(np.all(np.absolute(y0.imag - ym.imag*im_sign) < atol), (y0, ym))

    if sig_zero_ok:
        # check that signed zeros also work as a displacement
        jr = (x0.real == 0) & (dx.real != 0)
        ji = (x0.imag == 0) & (dx.imag != 0)
        if np.any(jr):
            x = x0[jr]
            x.real = np.NZERO
            ym = f(x)
            assert_(np.all(np.absolute(y0[jr].real - ym.real*re_sign) < atol), (y0[jr], ym))
            assert_(np.all(np.absolute(y0[jr].imag - ym.imag*im_sign) < atol), (y0[jr], ym))

        if np.any(ji):
            x = x0[ji]
            x.imag = np.NZERO
            ym = f(x)
            assert_(np.all(np.absolute(y0[ji].real - ym.real*re_sign) < atol), (y0[ji], ym))
            assert_(np.all(np.absolute(y0[ji].imag - ym.imag*im_sign) < atol), (y0[ji], ym))

def test_copysign():
    assert_(np.copysign(1, -1) == -1)
    with np.errstate(divide="ignore"):
        assert_(1 / np.copysign(0, -1) < 0)
        assert_(1 / np.copysign(0, 1) > 0)
    assert_(np.signbit(np.copysign(np.nan, -1)))
    assert_(not np.signbit(np.copysign(np.nan, 1)))

def _test_nextafter(t):
    one = t(1)
    two = t(2)
    zero = t(0)
    eps = np.finfo(t).eps
    assert_(np.nextafter(one, two) - one == eps)
    assert_(np.nextafter(one, zero) - one < 0)
    assert_(np.isnan(np.nextafter(np.nan, one)))
    assert_(np.isnan(np.nextafter(one, np.nan)))
    assert_(np.nextafter(one, one) == one)

def test_nextafter():
    return _test_nextafter(np.float64)


def test_nextafterf():
    return _test_nextafter(np.float32)


@pytest.mark.skipif(np.finfo(np.double) == np.finfo(np.longdouble),
                    reason="long double is same as double")
@pytest.mark.xfail(condition=platform.machine().startswith("ppc64"),
                    reason="IBM double double")
def test_nextafterl():
    return _test_nextafter(np.longdouble)


def test_nextafter_0():
    for t, direction in itertools.product(np.sctypes['float'], (1, -1)):
        # The value of tiny for double double is NaN, so we need to pass the
        # assert
        with suppress_warnings() as sup:
            sup.filter(UserWarning)
            if not np.isnan(np.finfo(t).tiny):
                tiny = np.finfo(t).tiny
                assert_(
                    0. < direction * np.nextafter(t(0), t(direction)) < tiny)
        assert_equal(np.nextafter(t(0), t(direction)) / t(2.1), direction * 0.0)

def _test_spacing(t):
    one = t(1)
    eps = np.finfo(t).eps
    nan = t(np.nan)
    inf = t(np.inf)
    with np.errstate(invalid='ignore'):
        assert_(np.spacing(one) == eps)
        assert_(np.isnan(np.spacing(nan)))
        assert_(np.isnan(np.spacing(inf)))
        assert_(np.isnan(np.spacing(-inf)))
        assert_(np.spacing(t(1e30)) != 0)

def test_spacing():
    return _test_spacing(np.float64)

def test_spacingf():
    return _test_spacing(np.float32)


@pytest.mark.skipif(np.finfo(np.double) == np.finfo(np.longdouble),
                    reason="long double is same as double")
@pytest.mark.xfail(condition=platform.machine().startswith("ppc64"),
                    reason="IBM double double")
def test_spacingl():
    return _test_spacing(np.longdouble)

def test_spacing_gfortran():
    # Reference from this fortran file, built with gfortran 4.3.3 on linux
    # 32bits:
    #       PROGRAM test_spacing
    #        INTEGER, PARAMETER :: SGL = SELECTED_REAL_KIND(p=6, r=37)
    #        INTEGER, PARAMETER :: DBL = SELECTED_REAL_KIND(p=13, r=200)
    #
    #        WRITE(*,*) spacing(0.00001_DBL)
    #        WRITE(*,*) spacing(1.0_DBL)
    #        WRITE(*,*) spacing(1000._DBL)
    #        WRITE(*,*) spacing(10500._DBL)
    #
    #        WRITE(*,*) spacing(0.00001_SGL)
    #        WRITE(*,*) spacing(1.0_SGL)
    #        WRITE(*,*) spacing(1000._SGL)
    #        WRITE(*,*) spacing(10500._SGL)
    #       END PROGRAM
    ref = {np.float64: [1.69406589450860068E-021,
                        2.22044604925031308E-016,
                        1.13686837721616030E-013,
                        1.81898940354585648E-012],
           np.float32: [9.09494702E-13,
                        1.19209290E-07,
                        6.10351563E-05,
                        9.76562500E-04]}

    for dt, dec_ in zip([np.float32, np.float64], (10, 20)):
        x = np.array([1e-5, 1, 1000, 10500], dtype=dt)
        assert_array_almost_equal(np.spacing(x), ref[dt], decimal=dec_)

def test_nextafter_vs_spacing():
    # XXX: spacing does not handle long double yet
    for t in [np.float32, np.float64]:
        for _f in [1, 1e-5, 1000]:
            f = t(_f)
            f1 = t(_f + 1)
            assert_(np.nextafter(f, f1) - f == np.spacing(f))

def test_pos_nan():
    """Check np.nan is a positive nan."""
    assert_(np.signbit(np.nan) == 0)

def test_reduceat():
    """Test bug in reduceat when structured arrays are not copied."""
    db = np.dtype([('name', 'S11'), ('time', np.int64), ('value', np.float32)])
    a = np.empty([100], dtype=db)
    a['name'] = 'Simple'
    a['time'] = 10
    a['value'] = 100
    indx = [0, 7, 15, 25]

    h2 = []
    val1 = indx[0]
    for val2 in indx[1:]:
        h2.append(np.add.reduce(a['value'][val1:val2]))
        val1 = val2
    h2.append(np.add.reduce(a['value'][val1:]))
    h2 = np.array(h2)

    # test buffered -- this should work
    h1 = np.add.reduceat(a['value'], indx)
    assert_array_almost_equal(h1, h2)

    # This is when the error occurs.
    # test no buffer
    np.setbufsize(32)
    h1 = np.add.reduceat(a['value'], indx)
    np.setbufsize(np.UFUNC_BUFSIZE_DEFAULT)
    assert_array_almost_equal(h1, h2)

def test_reduceat_empty():
    """Reduceat should work with empty arrays"""
    indices = np.array([], 'i4')
    x = np.array([], 'f8')
    result = np.add.reduceat(x, indices)
    assert_equal(result.dtype, x.dtype)
    assert_equal(result.shape, (0,))
    # Another case with a slightly different zero-sized shape
    x = np.ones((5, 2))
    result = np.add.reduceat(x, [], axis=0)
    assert_equal(result.dtype, x.dtype)
    assert_equal(result.shape, (0, 2))
    result = np.add.reduceat(x, [], axis=1)
    assert_equal(result.dtype, x.dtype)
    assert_equal(result.shape, (5, 0))

def test_complex_nan_comparisons():
    nans = [complex(np.nan, 0), complex(0, np.nan), complex(np.nan, np.nan)]
    fins = [complex(1, 0), complex(-1, 0), complex(0, 1), complex(0, -1),
            complex(1, 1), complex(-1, -1), complex(0, 0)]

    with np.errstate(invalid='ignore'):
        for x in nans + fins:
            x = np.array([x])
            for y in nans + fins:
                y = np.array([y])

                if np.isfinite(x) and np.isfinite(y):
                    continue

                assert_equal(x < y, False, err_msg="%r < %r" % (x, y))
                assert_equal(x > y, False, err_msg="%r > %r" % (x, y))
                assert_equal(x <= y, False, err_msg="%r <= %r" % (x, y))
                assert_equal(x >= y, False, err_msg="%r >= %r" % (x, y))
                assert_equal(x == y, False, err_msg="%r == %r" % (x, y))


def test_rint_big_int():
    # np.rint bug for large integer values on Windows 32-bit and MKL
    # https://github.com/numpy/numpy/issues/6685
    val = 4607998452777363968
    # This is exactly representable in floating point
    assert_equal(val, int(float(val)))
    # Rint should not change the value
    assert_equal(val, np.rint(val))


@pytest.mark.parametrize('ftype', [np.float32, np.float64])
def test_memoverlap_accumulate(ftype):
    # Reproduces bug https://github.com/numpy/numpy/issues/15597
    arr = np.array([0.61, 0.60, 0.77, 0.41, 0.19], dtype=ftype)
    out_max = np.array([0.61, 0.61, 0.77, 0.77, 0.77], dtype=ftype)
    out_min = np.array([0.61, 0.60, 0.60, 0.41, 0.19], dtype=ftype)
    assert_equal(np.maximum.accumulate(arr), out_max)
    assert_equal(np.minimum.accumulate(arr), out_min)

@pytest.mark.parametrize("ufunc, dtype", [
    (ufunc, t[0])
    for ufunc in UFUNCS_BINARY_ACC
    for t in ufunc.types
    if t[-1] == '?' and t[0] not in 'DFGMmO'
])
def test_memoverlap_accumulate_cmp(ufunc, dtype):
    if ufunc.signature:
        pytest.skip('For generic signatures only')
    for size in (2, 8, 32, 64, 128, 256):
        arr = np.array([0, 1, 1]*size, dtype=dtype)
        acc = ufunc.accumulate(arr, dtype='?')
        acc_u8 = acc.view(np.uint8)
        exp = np.array(list(itertools.accumulate(arr, ufunc)), dtype=np.uint8)
        assert_equal(exp, acc_u8)

@pytest.mark.parametrize("ufunc, dtype", [
    (ufunc, t[0])
    for ufunc in UFUNCS_BINARY_ACC
    for t in ufunc.types
    if t[0] == t[1] and t[0] == t[-1] and t[0] not in 'DFGMmO?'
])
def test_memoverlap_accumulate_symmetric(ufunc, dtype):
    if ufunc.signature:
        pytest.skip('For generic signatures only')
    with np.errstate(all='ignore'):
        for size in (2, 8, 32, 64, 128, 256):
            arr = np.array([0, 1, 2]*size).astype(dtype)
            acc = ufunc.accumulate(arr, dtype=dtype)
            exp = np.array(list(itertools.accumulate(arr, ufunc)), dtype=dtype)
            assert_equal(exp, acc)

def test_signaling_nan_exceptions():
    with assert_no_warnings():
        a = np.ndarray(shape=(), dtype='float32', buffer=b'\x00\xe0\xbf\xff')
        np.isnan(a)

@pytest.mark.parametrize("arr", [
    np.arange(2),
    np.matrix([0, 1]),
    np.matrix([[0, 1], [2, 5]]),
    ])
def test_outer_subclass_preserve(arr):
    # for gh-8661
    class foo(np.ndarray): pass
    actual = np.multiply.outer(arr.view(foo), arr.view(foo))
    assert actual.__class__.__name__ == 'foo'

def test_outer_bad_subclass():
    class BadArr1(np.ndarray):
        def __array_finalize__(self, obj):
            # The outer call reshapes to 3 dims, try to do a bad reshape.
            if self.ndim == 3:
                self.shape = self.shape + (1,)

        def __array_prepare__(self, obj, context=None):
            return obj

    class BadArr2(np.ndarray):
        def __array_finalize__(self, obj):
            if isinstance(obj, BadArr2):
                # outer inserts 1-sized dims. In that case disturb them.
                if self.shape[-1] == 1:
                    self.shape = self.shape[::-1]

        def __array_prepare__(self, obj, context=None):
            return obj

    for cls in [BadArr1, BadArr2]:
        arr = np.ones((2, 3)).view(cls)
        with assert_raises(TypeError) as a:
            # The first array gets reshaped (not the second one)
            np.add.outer(arr, [1, 2])

        # This actually works, since we only see the reshaping error:
        arr = np.ones((2, 3)).view(cls)
        assert type(np.add.outer([1, 2], arr)) is cls

def test_outer_exceeds_maxdims():
    deep = np.ones((1,) * 17)
    with assert_raises(ValueError):
        np.add.outer(deep, deep)

def test_bad_legacy_ufunc_silent_errors():
    # legacy ufuncs can't report errors and NumPy can't check if the GIL
    # is released.  So NumPy has to check after the GIL is released just to
    # cover all bases.  `np.power` uses/used to use this.
    arr = np.arange(3).astype(np.float64)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        ncu_tests.always_error(arr, arr)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        # not contiguous means the fast-path cannot be taken
        non_contig = arr.repeat(20).reshape(-1, 6)[:, ::2]
        ncu_tests.always_error(non_contig, arr)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        ncu_tests.always_error.outer(arr, arr)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        ncu_tests.always_error.reduce(arr)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        ncu_tests.always_error.reduceat(arr, [0, 1])

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        ncu_tests.always_error.accumulate(arr)

    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        ncu_tests.always_error.at(arr, [0, 1, 2], arr)


@pytest.mark.parametrize('x1', [np.arange(3.0), [0.0, 1.0, 2.0]])
def test_bad_legacy_gufunc_silent_errors(x1):
    # Verify that an exception raised in a gufunc loop propagates correctly.
    # The signature of always_error_gufunc is '(i),()->()'.
    with pytest.raises(RuntimeError, match=r"How unexpected :\)!"):
        ncu_tests.always_error_gufunc(x1, 0.0)
