import sys
import warnings
import itertools
import platform
import pytest
import math
from decimal import Decimal

import numpy as np
from numpy.core import umath
from numpy.random import rand, randint, randn
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_raises_regex,
    assert_array_equal, assert_almost_equal, assert_array_almost_equal,
    assert_warns, assert_array_max_ulp, HAS_REFCOUNT, IS_WASM
    )
from numpy.core._rational_tests import rational

from hypothesis import given, strategies as st
from hypothesis.extra import numpy as hynp


class TestResize:
    def test_copies(self):
        A = np.array([[1, 2], [3, 4]])
        Ar1 = np.array([[1, 2, 3, 4], [1, 2, 3, 4]])
        assert_equal(np.resize(A, (2, 4)), Ar1)

        Ar2 = np.array([[1, 2], [3, 4], [1, 2], [3, 4]])
        assert_equal(np.resize(A, (4, 2)), Ar2)

        Ar3 = np.array([[1, 2, 3], [4, 1, 2], [3, 4, 1], [2, 3, 4]])
        assert_equal(np.resize(A, (4, 3)), Ar3)

    def test_repeats(self):
        A = np.array([1, 2, 3])
        Ar1 = np.array([[1, 2, 3, 1], [2, 3, 1, 2]])
        assert_equal(np.resize(A, (2, 4)), Ar1)

        Ar2 = np.array([[1, 2], [3, 1], [2, 3], [1, 2]])
        assert_equal(np.resize(A, (4, 2)), Ar2)

        Ar3 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])
        assert_equal(np.resize(A, (4, 3)), Ar3)

    def test_zeroresize(self):
        A = np.array([[1, 2], [3, 4]])
        Ar = np.resize(A, (0,))
        assert_array_equal(Ar, np.array([]))
        assert_equal(A.dtype, Ar.dtype)

        Ar = np.resize(A, (0, 2))
        assert_equal(Ar.shape, (0, 2))

        Ar = np.resize(A, (2, 0))
        assert_equal(Ar.shape, (2, 0))

    def test_reshape_from_zero(self):
        # See also gh-6740
        A = np.zeros(0, dtype=[('a', np.float32)])
        Ar = np.resize(A, (2, 1))
        assert_array_equal(Ar, np.zeros((2, 1), Ar.dtype))
        assert_equal(A.dtype, Ar.dtype)

    def test_negative_resize(self):
        A = np.arange(0, 10, dtype=np.float32)
        new_shape = (-10, -1)
        with pytest.raises(ValueError, match=r"negative"):
            np.resize(A, new_shape=new_shape)

    def test_subclass(self):
        class MyArray(np.ndarray):
            __array_priority__ = 1.

        my_arr = np.array([1]).view(MyArray)
        assert type(np.resize(my_arr, 5)) is MyArray
        assert type(np.resize(my_arr, 0)) is MyArray

        my_arr = np.array([]).view(MyArray)
        assert type(np.resize(my_arr, 5)) is MyArray


class TestNonarrayArgs:
    # check that non-array arguments to functions wrap them in arrays
    def test_choose(self):
        choices = [[0, 1, 2],
                   [3, 4, 5],
                   [5, 6, 7]]
        tgt = [5, 1, 5]
        a = [2, 0, 1]

        out = np.choose(a, choices)
        assert_equal(out, tgt)

    def test_clip(self):
        arr = [-1, 5, 2, 3, 10, -4, -9]
        out = np.clip(arr, 2, 7)
        tgt = [2, 5, 2, 3, 7, 2, 2]
        assert_equal(out, tgt)

    def test_compress(self):
        arr = [[0, 1, 2, 3, 4],
               [5, 6, 7, 8, 9]]
        tgt = [[5, 6, 7, 8, 9]]
        out = np.compress([0, 1], arr, axis=0)
        assert_equal(out, tgt)

    def test_count_nonzero(self):
        arr = [[0, 1, 7, 0, 0],
               [3, 0, 0, 2, 19]]
        tgt = np.array([2, 3])
        out = np.count_nonzero(arr, axis=1)
        assert_equal(out, tgt)

    def test_cumproduct(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_(np.all(np.cumproduct(A) == np.array([1, 2, 6, 24, 120, 720])))

    def test_diagonal(self):
        a = [[0, 1, 2, 3],
             [4, 5, 6, 7],
             [8, 9, 10, 11]]
        out = np.diagonal(a)
        tgt = [0, 5, 10]

        assert_equal(out, tgt)

    def test_mean(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_(np.mean(A) == 3.5)
        assert_(np.all(np.mean(A, 0) == np.array([2.5, 3.5, 4.5])))
        assert_(np.all(np.mean(A, 1) == np.array([2., 5.])))

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_(np.isnan(np.mean([])))
            assert_(w[0].category is RuntimeWarning)

    def test_ptp(self):
        a = [3, 4, 5, 10, -3, -5, 6.0]
        assert_equal(np.ptp(a, axis=0), 15.0)

    def test_prod(self):
        arr = [[1, 2, 3, 4],
               [5, 6, 7, 9],
               [10, 3, 4, 5]]
        tgt = [24, 1890, 600]

        assert_equal(np.prod(arr, axis=-1), tgt)

    def test_ravel(self):
        a = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tgt = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        assert_equal(np.ravel(a), tgt)

    def test_repeat(self):
        a = [1, 2, 3]
        tgt = [1, 1, 2, 2, 3, 3]

        out = np.repeat(a, 2)
        assert_equal(out, tgt)

    def test_reshape(self):
        arr = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12]]
        tgt = [[1, 2, 3, 4, 5, 6], [7, 8, 9, 10, 11, 12]]
        assert_equal(np.reshape(arr, (2, 6)), tgt)

    def test_round(self):
        arr = [1.56, 72.54, 6.35, 3.25]
        tgt = [1.6, 72.5, 6.4, 3.2]
        assert_equal(np.around(arr, decimals=1), tgt)
        s = np.float64(1.)
        assert_(isinstance(s.round(), np.float64))
        assert_equal(s.round(), 1.)

    @pytest.mark.parametrize('dtype', [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
    ])
    def test_dunder_round(self, dtype):
        s = dtype(1)
        assert_(isinstance(round(s), int))
        assert_(isinstance(round(s, None), int))
        assert_(isinstance(round(s, ndigits=None), int))
        assert_equal(round(s), 1)
        assert_equal(round(s, None), 1)
        assert_equal(round(s, ndigits=None), 1)

    @pytest.mark.parametrize('val, ndigits', [
        pytest.param(2**31 - 1, -1,
            marks=pytest.mark.xfail(reason="Out of range of int32")
        ),
        (2**31 - 1, 1-math.ceil(math.log10(2**31 - 1))),
        (2**31 - 1, -math.ceil(math.log10(2**31 - 1)))
    ])
    def test_dunder_round_edgecases(self, val, ndigits):
        assert_equal(round(val, ndigits), round(np.int32(val), ndigits))

    def test_dunder_round_accuracy(self):
        f = np.float64(5.1 * 10**73)
        assert_(isinstance(round(f, -73), np.float64))
        assert_array_max_ulp(round(f, -73), 5.0 * 10**73)
        assert_(isinstance(round(f, ndigits=-73), np.float64))
        assert_array_max_ulp(round(f, ndigits=-73), 5.0 * 10**73)

        i = np.int64(501)
        assert_(isinstance(round(i, -2), np.int64))
        assert_array_max_ulp(round(i, -2), 500)
        assert_(isinstance(round(i, ndigits=-2), np.int64))
        assert_array_max_ulp(round(i, ndigits=-2), 500)

    @pytest.mark.xfail(raises=AssertionError, reason="gh-15896")
    def test_round_py_consistency(self):
        f = 5.1 * 10**73
        assert_equal(round(np.float64(f), -73), round(f, -73))

    def test_searchsorted(self):
        arr = [-8, -5, -1, 3, 6, 10]
        out = np.searchsorted(arr, 0)
        assert_equal(out, 3)

    def test_size(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_(np.size(A) == 6)
        assert_(np.size(A, 0) == 2)
        assert_(np.size(A, 1) == 3)

    def test_squeeze(self):
        A = [[[1, 1, 1], [2, 2, 2], [3, 3, 3]]]
        assert_equal(np.squeeze(A).shape, (3, 3))
        assert_equal(np.squeeze(np.zeros((1, 3, 1))).shape, (3,))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=0).shape, (3, 1))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=-1).shape, (1, 3))
        assert_equal(np.squeeze(np.zeros((1, 3, 1)), axis=2).shape, (1, 3))
        assert_equal(np.squeeze([np.zeros((3, 1))]).shape, (3,))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=0).shape, (3, 1))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=2).shape, (1, 3))
        assert_equal(np.squeeze([np.zeros((3, 1))], axis=-1).shape, (1, 3))

    def test_std(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_almost_equal(np.std(A), 1.707825127659933)
        assert_almost_equal(np.std(A, 0), np.array([1.5, 1.5, 1.5]))
        assert_almost_equal(np.std(A, 1), np.array([0.81649658, 0.81649658]))

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_(np.isnan(np.std([])))
            assert_(w[0].category is RuntimeWarning)

    def test_swapaxes(self):
        tgt = [[[0, 4], [2, 6]], [[1, 5], [3, 7]]]
        a = [[[0, 1], [2, 3]], [[4, 5], [6, 7]]]
        out = np.swapaxes(a, 0, 2)
        assert_equal(out, tgt)

    def test_sum(self):
        m = [[1, 2, 3],
             [4, 5, 6],
             [7, 8, 9]]
        tgt = [[6], [15], [24]]
        out = np.sum(m, axis=1, keepdims=True)

        assert_equal(tgt, out)

    def test_take(self):
        tgt = [2, 3, 5]
        indices = [1, 2, 4]
        a = [1, 2, 3, 4, 5]

        out = np.take(a, indices)
        assert_equal(out, tgt)

    def test_trace(self):
        c = [[1, 2], [3, 4], [5, 6]]
        assert_equal(np.trace(c), 5)

    def test_transpose(self):
        arr = [[1, 2], [3, 4], [5, 6]]
        tgt = [[1, 3, 5], [2, 4, 6]]
        assert_equal(np.transpose(arr, (1, 0)), tgt)

    def test_var(self):
        A = [[1, 2, 3], [4, 5, 6]]
        assert_almost_equal(np.var(A), 2.9166666666666665)
        assert_almost_equal(np.var(A, 0), np.array([2.25, 2.25, 2.25]))
        assert_almost_equal(np.var(A, 1), np.array([0.66666667, 0.66666667]))

        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_(np.isnan(np.var([])))
            assert_(w[0].category is RuntimeWarning)

        B = np.array([None, 0])
        B[0] = 1j
        assert_almost_equal(np.var(B), 0.25)


class TestIsscalar:
    def test_isscalar(self):
        assert_(np.isscalar(3.1))
        assert_(np.isscalar(np.int16(12345)))
        assert_(np.isscalar(False))
        assert_(np.isscalar('numpy'))
        assert_(not np.isscalar([3.1]))
        assert_(not np.isscalar(None))

        # PEP 3141
        from fractions import Fraction
        assert_(np.isscalar(Fraction(5, 17)))
        from numbers import Number
        assert_(np.isscalar(Number()))


class TestBoolScalar:
    def test_logical(self):
        f = np.False_
        t = np.True_
        s = "xyz"
        assert_((t and s) is s)
        assert_((f and s) is f)

    def test_bitwise_or(self):
        f = np.False_
        t = np.True_
        assert_((t | t) is t)
        assert_((f | t) is t)
        assert_((t | f) is t)
        assert_((f | f) is f)

    def test_bitwise_and(self):
        f = np.False_
        t = np.True_
        assert_((t & t) is t)
        assert_((f & t) is f)
        assert_((t & f) is f)
        assert_((f & f) is f)

    def test_bitwise_xor(self):
        f = np.False_
        t = np.True_
        assert_((t ^ t) is f)
        assert_((f ^ t) is t)
        assert_((t ^ f) is t)
        assert_((f ^ f) is f)


class TestBoolArray:
    def setup_method(self):
        # offset for simd tests
        self.t = np.array([True] * 41, dtype=bool)[1::]
        self.f = np.array([False] * 41, dtype=bool)[1::]
        self.o = np.array([False] * 42, dtype=bool)[2::]
        self.nm = self.f.copy()
        self.im = self.t.copy()
        self.nm[3] = True
        self.nm[-2] = True
        self.im[3] = False
        self.im[-2] = False

    def test_all_any(self):
        assert_(self.t.all())
        assert_(self.t.any())
        assert_(not self.f.all())
        assert_(not self.f.any())
        assert_(self.nm.any())
        assert_(self.im.any())
        assert_(not self.nm.all())
        assert_(not self.im.all())
        # check bad element in all positions
        for i in range(256 - 7):
            d = np.array([False] * 256, dtype=bool)[7::]
            d[i] = True
            assert_(np.any(d))
            e = np.array([True] * 256, dtype=bool)[7::]
            e[i] = False
            assert_(not np.all(e))
            assert_array_equal(e, ~d)
        # big array test for blocked libc loops
        for i in list(range(9, 6000, 507)) + [7764, 90021, -10]:
            d = np.array([False] * 100043, dtype=bool)
            d[i] = True
            assert_(np.any(d), msg="%r" % i)
            e = np.array([True] * 100043, dtype=bool)
            e[i] = False
            assert_(not np.all(e), msg="%r" % i)

    def test_logical_not_abs(self):
        assert_array_equal(~self.t, self.f)
        assert_array_equal(np.abs(~self.t), self.f)
        assert_array_equal(np.abs(~self.f), self.t)
        assert_array_equal(np.abs(self.f), self.f)
        assert_array_equal(~np.abs(self.f), self.t)
        assert_array_equal(~np.abs(self.t), self.f)
        assert_array_equal(np.abs(~self.nm), self.im)
        np.logical_not(self.t, out=self.o)
        assert_array_equal(self.o, self.f)
        np.abs(self.t, out=self.o)
        assert_array_equal(self.o, self.t)

    def test_logical_and_or_xor(self):
        assert_array_equal(self.t | self.t, self.t)
        assert_array_equal(self.f | self.f, self.f)
        assert_array_equal(self.t | self.f, self.t)
        assert_array_equal(self.f | self.t, self.t)
        np.logical_or(self.t, self.t, out=self.o)
        assert_array_equal(self.o, self.t)
        assert_array_equal(self.t & self.t, self.t)
        assert_array_equal(self.f & self.f, self.f)
        assert_array_equal(self.t & self.f, self.f)
        assert_array_equal(self.f & self.t, self.f)
        np.logical_and(self.t, self.t, out=self.o)
        assert_array_equal(self.o, self.t)
        assert_array_equal(self.t ^ self.t, self.f)
        assert_array_equal(self.f ^ self.f, self.f)
        assert_array_equal(self.t ^ self.f, self.t)
        assert_array_equal(self.f ^ self.t, self.t)
        np.logical_xor(self.t, self.t, out=self.o)
        assert_array_equal(self.o, self.f)

        assert_array_equal(self.nm & self.t, self.nm)
        assert_array_equal(self.im & self.f, False)
        assert_array_equal(self.nm & True, self.nm)
        assert_array_equal(self.im & False, self.f)
        assert_array_equal(self.nm | self.t, self.t)
        assert_array_equal(self.im | self.f, self.im)
        assert_array_equal(self.nm | True, self.t)
        assert_array_equal(self.im | False, self.im)
        assert_array_equal(self.nm ^ self.t, self.im)
        assert_array_equal(self.im ^ self.f, self.im)
        assert_array_equal(self.nm ^ True, self.im)
        assert_array_equal(self.im ^ False, self.im)


class TestBoolCmp:
    def setup_method(self):
        self.f = np.ones(256, dtype=np.float32)
        self.ef = np.ones(self.f.size, dtype=bool)
        self.d = np.ones(128, dtype=np.float64)
        self.ed = np.ones(self.d.size, dtype=bool)
        # generate values for all permutation of 256bit simd vectors
        s = 0
        for i in range(32):
            self.f[s:s+8] = [i & 2**x for x in range(8)]
            self.ef[s:s+8] = [(i & 2**x) != 0 for x in range(8)]
            s += 8
        s = 0
        for i in range(16):
            self.d[s:s+4] = [i & 2**x for x in range(4)]
            self.ed[s:s+4] = [(i & 2**x) != 0 for x in range(4)]
            s += 4

        self.nf = self.f.copy()
        self.nd = self.d.copy()
        self.nf[self.ef] = np.nan
        self.nd[self.ed] = np.nan

        self.inff = self.f.copy()
        self.infd = self.d.copy()
        self.inff[::3][self.ef[::3]] = np.inf
        self.infd[::3][self.ed[::3]] = np.inf
        self.inff[1::3][self.ef[1::3]] = -np.inf
        self.infd[1::3][self.ed[1::3]] = -np.inf
        self.inff[2::3][self.ef[2::3]] = np.nan
        self.infd[2::3][self.ed[2::3]] = np.nan
        self.efnonan = self.ef.copy()
        self.efnonan[2::3] = False
        self.ednonan = self.ed.copy()
        self.ednonan[2::3] = False

        self.signf = self.f.copy()
        self.signd = self.d.copy()
        self.signf[self.ef] *= -1.
        self.signd[self.ed] *= -1.
        self.signf[1::6][self.ef[1::6]] = -np.inf
        self.signd[1::6][self.ed[1::6]] = -np.inf
        self.signf[3::6][self.ef[3::6]] = -np.nan
        self.signd[3::6][self.ed[3::6]] = -np.nan
        self.signf[4::6][self.ef[4::6]] = -0.
        self.signd[4::6][self.ed[4::6]] = -0.

    def test_float(self):
        # offset for alignment test
        for i in range(4):
            assert_array_equal(self.f[i:] > 0, self.ef[i:])
            assert_array_equal(self.f[i:] - 1 >= 0, self.ef[i:])
            assert_array_equal(self.f[i:] == 0, ~self.ef[i:])
            assert_array_equal(-self.f[i:] < 0, self.ef[i:])
            assert_array_equal(-self.f[i:] + 1 <= 0, self.ef[i:])
            r = self.f[i:] != 0
            assert_array_equal(r, self.ef[i:])
            r2 = self.f[i:] != np.zeros_like(self.f[i:])
            r3 = 0 != self.f[i:]
            assert_array_equal(r, r2)
            assert_array_equal(r, r3)
            # check bool == 0x1
            assert_array_equal(r.view(np.int8), r.astype(np.int8))
            assert_array_equal(r2.view(np.int8), r2.astype(np.int8))
            assert_array_equal(r3.view(np.int8), r3.astype(np.int8))

            # isnan on amd64 takes the same code path
            assert_array_equal(np.isnan(self.nf[i:]), self.ef[i:])
            assert_array_equal(np.isfinite(self.nf[i:]), ~self.ef[i:])
            assert_array_equal(np.isfinite(self.inff[i:]), ~self.ef[i:])
            assert_array_equal(np.isinf(self.inff[i:]), self.efnonan[i:])
            assert_array_equal(np.signbit(self.signf[i:]), self.ef[i:])

    def test_double(self):
        # offset for alignment test
        for i in range(2):
            assert_array_equal(self.d[i:] > 0, self.ed[i:])
            assert_array_equal(self.d[i:] - 1 >= 0, self.ed[i:])
            assert_array_equal(self.d[i:] == 0, ~self.ed[i:])
            assert_array_equal(-self.d[i:] < 0, self.ed[i:])
            assert_array_equal(-self.d[i:] + 1 <= 0, self.ed[i:])
            r = self.d[i:] != 0
            assert_array_equal(r, self.ed[i:])
            r2 = self.d[i:] != np.zeros_like(self.d[i:])
            r3 = 0 != self.d[i:]
            assert_array_equal(r, r2)
            assert_array_equal(r, r3)
            # check bool == 0x1
            assert_array_equal(r.view(np.int8), r.astype(np.int8))
            assert_array_equal(r2.view(np.int8), r2.astype(np.int8))
            assert_array_equal(r3.view(np.int8), r3.astype(np.int8))

            # isnan on amd64 takes the same code path
            assert_array_equal(np.isnan(self.nd[i:]), self.ed[i:])
            assert_array_equal(np.isfinite(self.nd[i:]), ~self.ed[i:])
            assert_array_equal(np.isfinite(self.infd[i:]), ~self.ed[i:])
            assert_array_equal(np.isinf(self.infd[i:]), self.ednonan[i:])
            assert_array_equal(np.signbit(self.signd[i:]), self.ed[i:])


class TestSeterr:
    def test_default(self):
        err = np.geterr()
        assert_equal(err,
                     dict(divide='warn',
                          invalid='warn',
                          over='warn',
                          under='ignore')
                     )

    def test_set(self):
        with np.errstate():
            err = np.seterr()
            old = np.seterr(divide='print')
            assert_(err == old)
            new = np.seterr()
            assert_(new['divide'] == 'print')
            np.seterr(over='raise')
            assert_(np.geterr()['over'] == 'raise')
            assert_(new['divide'] == 'print')
            np.seterr(**old)
            assert_(np.geterr() == old)

    @pytest.mark.skipif(IS_WASM, reason="no wasm fp exception support")
    @pytest.mark.skipif(platform.machine() == "armv5tel", reason="See gh-413.")
    def test_divide_err(self):
        with np.errstate(divide='raise'):
            with assert_raises(FloatingPointError):
                np.array([1.]) / np.array([0.])

            np.seterr(divide='ignore')
            np.array([1.]) / np.array([0.])

    @pytest.mark.skipif(IS_WASM, reason="no wasm fp exception support")
    def test_errobj(self):
        olderrobj = np.geterrobj()
        self.called = 0
        try:
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                with np.errstate(divide='warn'):
                    np.seterrobj([20000, 1, None])
                    np.array([1.]) / np.array([0.])
                    assert_equal(len(w), 1)

            def log_err(*args):
                self.called += 1
                extobj_err = args
                assert_(len(extobj_err) == 2)
                assert_("divide" in extobj_err[0])

            with np.errstate(divide='ignore'):
                np.seterrobj([20000, 3, log_err])
                np.array([1.]) / np.array([0.])
            assert_equal(self.called, 1)

            np.seterrobj(olderrobj)
            with np.errstate(divide='ignore'):
                np.divide(1., 0., extobj=[20000, 3, log_err])
            assert_equal(self.called, 2)
        finally:
            np.seterrobj(olderrobj)
            del self.called

    def test_errobj_noerrmask(self):
        # errmask = 0 has a special code path for the default
        olderrobj = np.geterrobj()
        try:
            # set errobj to something non default
            np.seterrobj([umath.UFUNC_BUFSIZE_DEFAULT,
                         umath.ERR_DEFAULT + 1, None])
            # call a ufunc
            np.isnan(np.array([6]))
            # same with the default, lots of times to get rid of possible
            # pre-existing stack in the code
            for i in range(10000):
                np.seterrobj([umath.UFUNC_BUFSIZE_DEFAULT, umath.ERR_DEFAULT,
                             None])
            np.isnan(np.array([6]))
        finally:
            np.seterrobj(olderrobj)


class TestFloatExceptions:
    def assert_raises_fpe(self, fpeerr, flop, x, y):
        ftype = type(x)
        try:
            flop(x, y)
            assert_(False,
                    "Type %s did not raise fpe error '%s'." % (ftype, fpeerr))
        except FloatingPointError as exc:
            assert_(str(exc).find(fpeerr) >= 0,
                    "Type %s raised wrong fpe error '%s'." % (ftype, exc))

    def assert_op_raises_fpe(self, fpeerr, flop, sc1, sc2):
        # Check that fpe exception is raised.
        #
        # Given a floating operation `flop` and two scalar values, check that
        # the operation raises the floating point exception specified by
        # `fpeerr`. Tests all variants with 0-d array scalars as well.

        self.assert_raises_fpe(fpeerr, flop, sc1, sc2)
        self.assert_raises_fpe(fpeerr, flop, sc1[()], sc2)
        self.assert_raises_fpe(fpeerr, flop, sc1, sc2[()])
        self.assert_raises_fpe(fpeerr, flop, sc1[()], sc2[()])

    # Test for all real and complex float types
    @pytest.mark.skipif(IS_WASM, reason="no wasm fp exception support")
    @pytest.mark.parametrize("typecode", np.typecodes["AllFloat"])
    def test_floating_exceptions(self, typecode):
        # Test basic arithmetic function errors
        with np.errstate(all='raise'):
            ftype = np.obj2sctype(typecode)
            if np.dtype(ftype).kind == 'f':
                # Get some extreme values for the type
                fi = np.finfo(ftype)
                ft_tiny = fi._machar.tiny
                ft_max = fi.max
                ft_eps = fi.eps
                underflow = 'underflow'
                divbyzero = 'divide by zero'
            else:
                # 'c', complex, corresponding real dtype
                rtype = type(ftype(0).real)
                fi = np.finfo(rtype)
                ft_tiny = ftype(fi._machar.tiny)
                ft_max = ftype(fi.max)
                ft_eps = ftype(fi.eps)
                # The complex types raise different exceptions
                underflow = ''
                divbyzero = ''
            overflow = 'overflow'
            invalid = 'invalid'

            # The value of tiny for double double is NaN, so we need to
            # pass the assert
            if not np.isnan(ft_tiny):
                self.assert_raises_fpe(underflow,
                                    lambda a, b: a/b, ft_tiny, ft_max)
                self.assert_raises_fpe(underflow,
                                    lambda a, b: a*b, ft_tiny, ft_tiny)
            self.assert_raises_fpe(overflow,
                                   lambda a, b: a*b, ft_max, ftype(2))
            self.assert_raises_fpe(overflow,
                                   lambda a, b: a/b, ft_max, ftype(0.5))
            self.assert_raises_fpe(overflow,
                                   lambda a, b: a+b, ft_max, ft_max*ft_eps)
            self.assert_raises_fpe(overflow,
                                   lambda a, b: a-b, -ft_max, ft_max*ft_eps)
            self.assert_raises_fpe(overflow,
                                   np.power, ftype(2), ftype(2**fi.nexp))
            self.assert_raises_fpe(divbyzero,
                                   lambda a, b: a/b, ftype(1), ftype(0))
            self.assert_raises_fpe(
                invalid, lambda a, b: a/b, ftype(np.inf), ftype(np.inf)
            )
            self.assert_raises_fpe(invalid,
                                   lambda a, b: a/b, ftype(0), ftype(0))
            self.assert_raises_fpe(
                invalid, lambda a, b: a-b, ftype(np.inf), ftype(np.inf)
            )
            self.assert_raises_fpe(
                invalid, lambda a, b: a+b, ftype(np.inf), ftype(-np.inf)
            )
            self.assert_raises_fpe(invalid,
                                   lambda a, b: a*b, ftype(0), ftype(np.inf))

    @pytest.mark.skipif(IS_WASM, reason="no wasm fp exception support")
    def test_warnings(self):
        # test warning code path
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            with np.errstate(all="warn"):
                np.divide(1, 0.)
                assert_equal(len(w), 1)
                assert_("divide by zero" in str(w[0].message))
                np.array(1e300) * np.array(1e300)
                assert_equal(len(w), 2)
                assert_("overflow" in str(w[-1].message))
                np.array(np.inf) - np.array(np.inf)
                assert_equal(len(w), 3)
                assert_("invalid value" in str(w[-1].message))
                np.array(1e-300) * np.array(1e-300)
                assert_equal(len(w), 4)
                assert_("underflow" in str(w[-1].message))


class TestTypes:
    def check_promotion_cases(self, promote_func):
        # tests that the scalars get coerced correctly.
        b = np.bool_(0)
        i8, i16, i32, i64 = np.int8(0), np.int16(0), np.int32(0), np.int64(0)
        u8, u16, u32, u64 = np.uint8(0), np.uint16(0), np.uint32(0), np.uint64(0)
        f32, f64, fld = np.float32(0), np.float64(0), np.longdouble(0)
        c64, c128, cld = np.complex64(0), np.complex128(0), np.clongdouble(0)

        # coercion within the same kind
        assert_equal(promote_func(i8, i16), np.dtype(np.int16))
        assert_equal(promote_func(i32, i8), np.dtype(np.int32))
        assert_equal(promote_func(i16, i64), np.dtype(np.int64))
        assert_equal(promote_func(u8, u32), np.dtype(np.uint32))
        assert_equal(promote_func(f32, f64), np.dtype(np.float64))
        assert_equal(promote_func(fld, f32), np.dtype(np.longdouble))
        assert_equal(promote_func(f64, fld), np.dtype(np.longdouble))
        assert_equal(promote_func(c128, c64), np.dtype(np.complex128))
        assert_equal(promote_func(cld, c128), np.dtype(np.clongdouble))
        assert_equal(promote_func(c64, fld), np.dtype(np.clongdouble))

        # coercion between kinds
        assert_equal(promote_func(b, i32), np.dtype(np.int32))
        assert_equal(promote_func(b, u8), np.dtype(np.uint8))
        assert_equal(promote_func(i8, u8), np.dtype(np.int16))
        assert_equal(promote_func(u8, i32), np.dtype(np.int32))
        assert_equal(promote_func(i64, u32), np.dtype(np.int64))
        assert_equal(promote_func(u64, i32), np.dtype(np.float64))
        assert_equal(promote_func(i32, f32), np.dtype(np.float64))
        assert_equal(promote_func(i64, f32), np.dtype(np.float64))
        assert_equal(promote_func(f32, i16), np.dtype(np.float32))
        assert_equal(promote_func(f32, u32), np.dtype(np.float64))
        assert_equal(promote_func(f32, c64), np.dtype(np.complex64))
        assert_equal(promote_func(c128, f32), np.dtype(np.complex128))
        assert_equal(promote_func(cld, f64), np.dtype(np.clongdouble))

        # coercion between scalars and 1-D arrays
        assert_equal(promote_func(np.array([b]), i8), np.dtype(np.int8))
        assert_equal(promote_func(np.array([b]), u8), np.dtype(np.uint8))
        assert_equal(promote_func(np.array([b]), i32), np.dtype(np.int32))
        assert_equal(promote_func(np.array([b]), u32), np.dtype(np.uint32))
        assert_equal(promote_func(np.array([i8]), i64), np.dtype(np.int8))
        assert_equal(promote_func(u64, np.array([i32])), np.dtype(np.int32))
        assert_equal(promote_func(i64, np.array([u32])), np.dtype(np.uint32))
        assert_equal(promote_func(np.int32(-1), np.array([u64])),
                     np.dtype(np.float64))
        assert_equal(promote_func(f64, np.array([f32])), np.dtype(np.float32))
        assert_equal(promote_func(fld, np.array([f32])), np.dtype(np.float32))
        assert_equal(promote_func(np.array([f64]), fld), np.dtype(np.float64))
        assert_equal(promote_func(fld, np.array([c64])),
                     np.dtype(np.complex64))
        assert_equal(promote_func(c64, np.array([f64])),
                     np.dtype(np.complex128))
        assert_equal(promote_func(np.complex64(3j), np.array([f64])),
                     np.dtype(np.complex128))

        # coercion between scalars and 1-D arrays, where
        # the scalar has greater kind than the array
        assert_equal(promote_func(np.array([b]), f64), np.dtype(np.float64))
        assert_equal(promote_func(np.array([b]), i64), np.dtype(np.int64))
        assert_equal(promote_func(np.array([b]), u64), np.dtype(np.uint64))
        assert_equal(promote_func(np.array([i8]), f64), np.dtype(np.float64))
        assert_equal(promote_func(np.array([u16]), f64), np.dtype(np.float64))

        # uint and int are treated as the same "kind" for
        # the purposes of array-scalar promotion.
        assert_equal(promote_func(np.array([u16]), i32), np.dtype(np.uint16))

        # float and complex are treated as the same "kind" for
        # the purposes of array-scalar promotion, so that you can do
        # (0j + float32array) to get a complex64 array instead of
        # a complex128 array.
        assert_equal(promote_func(np.array([f32]), c128),
                     np.dtype(np.complex64))

    def test_coercion(self):
        def res_type(a, b):
            return np.add(a, b).dtype

        self.check_promotion_cases(res_type)

        # Use-case: float/complex scalar * bool/int8 array
        #           shouldn't narrow the float/complex type
        for a in [np.array([True, False]), np.array([-3, 12], dtype=np.int8)]:
            b = 1.234 * a
            assert_equal(b.dtype, np.dtype('f8'), "array type %s" % a.dtype)
            b = np.longdouble(1.234) * a
            assert_equal(b.dtype, np.dtype(np.longdouble),
                         "array type %s" % a.dtype)
            b = np.float64(1.234) * a
            assert_equal(b.dtype, np.dtype('f8'), "array type %s" % a.dtype)
            b = np.float32(1.234) * a
            assert_equal(b.dtype, np.dtype('f4'), "array type %s" % a.dtype)
            b = np.float16(1.234) * a
            assert_equal(b.dtype, np.dtype('f2'), "array type %s" % a.dtype)

            b = 1.234j * a
            assert_equal(b.dtype, np.dtype('c16'), "array type %s" % a.dtype)
            b = np.clongdouble(1.234j) * a
            assert_equal(b.dtype, np.dtype(np.clongdouble),
                         "array type %s" % a.dtype)
            b = np.complex128(1.234j) * a
            assert_equal(b.dtype, np.dtype('c16'), "array type %s" % a.dtype)
            b = np.complex64(1.234j) * a
            assert_equal(b.dtype, np.dtype('c8'), "array type %s" % a.dtype)

        # The following use-case is problematic, and to resolve its
        # tricky side-effects requires more changes.
        #
        # Use-case: (1-t)*a, where 't' is a boolean array and 'a' is
        #            a float32, shouldn't promote to float64
        #
        # a = np.array([1.0, 1.5], dtype=np.float32)
        # t = np.array([True, False])
        # b = t*a
        # assert_equal(b, [1.0, 0.0])
        # assert_equal(b.dtype, np.dtype('f4'))
        # b = (1-t)*a
        # assert_equal(b, [0.0, 1.5])
        # assert_equal(b.dtype, np.dtype('f4'))
        #
        # Probably ~t (bitwise negation) is more proper to use here,
        # but this is arguably less intuitive to understand at a glance, and
        # would fail if 't' is actually an integer array instead of boolean:
        #
        # b = (~t)*a
        # assert_equal(b, [0.0, 1.5])
        # assert_equal(b.dtype, np.dtype('f4'))

    def test_result_type(self):
        self.check_promotion_cases(np.result_type)
        assert_(np.result_type(None) == np.dtype(None))

    def test_promote_types_endian(self):
        # promote_types should always return native-endian types
        assert_equal(np.promote_types('<i8', '<i8'), np.dtype('i8'))
        assert_equal(np.promote_types('>i8', '>i8'), np.dtype('i8'))

        assert_equal(np.promote_types('>i8', '>U16'), np.dtype('U21'))
        assert_equal(np.promote_types('<i8', '<U16'), np.dtype('U21'))
        assert_equal(np.promote_types('>U16', '>i8'), np.dtype('U21'))
        assert_equal(np.promote_types('<U16', '<i8'), np.dtype('U21'))

        assert_equal(np.promote_types('<S5', '<U8'), np.dtype('U8'))
        assert_equal(np.promote_types('>S5', '>U8'), np.dtype('U8'))
        assert_equal(np.promote_types('<U8', '<S5'), np.dtype('U8'))
        assert_equal(np.promote_types('>U8', '>S5'), np.dtype('U8'))
        assert_equal(np.promote_types('<U5', '<U8'), np.dtype('U8'))
        assert_equal(np.promote_types('>U8', '>U5'), np.dtype('U8'))

        assert_equal(np.promote_types('<M8', '<M8'), np.dtype('M8'))
        assert_equal(np.promote_types('>M8', '>M8'), np.dtype('M8'))
        assert_equal(np.promote_types('<m8', '<m8'), np.dtype('m8'))
        assert_equal(np.promote_types('>m8', '>m8'), np.dtype('m8'))

    def test_can_cast_and_promote_usertypes(self):
        # The rational type defines safe casting for signed integers,
        # boolean. Rational itself *does* cast safely to double.
        # (rational does not actually cast to all signed integers, e.g.
        # int64 can be both long and longlong and it registers only the first)
        valid_types = ["int8", "int16", "int32", "int64", "bool"]
        invalid_types = "BHILQP" + "FDG" + "mM" + "f" + "V"

        rational_dt = np.dtype(rational)
        for numpy_dtype in valid_types:
            numpy_dtype = np.dtype(numpy_dtype)
            assert np.can_cast(numpy_dtype, rational_dt)
            assert np.promote_types(numpy_dtype, rational_dt) is rational_dt

        for numpy_dtype in invalid_types:
            numpy_dtype = np.dtype(numpy_dtype)
            assert not np.can_cast(numpy_dtype, rational_dt)
            with pytest.raises(TypeError):
                np.promote_types(numpy_dtype, rational_dt)

        double_dt = np.dtype("double")
        assert np.can_cast(rational_dt, double_dt)
        assert np.promote_types(double_dt, rational_dt) is double_dt

    @pytest.mark.parametrize("swap", ["", "swap"])
    @pytest.mark.parametrize("string_dtype", ["U", "S"])
    def test_promote_types_strings(self, swap, string_dtype):
        if swap == "swap":
            promote_types = lambda a, b: np.promote_types(b, a)
        else:
            promote_types = np.promote_types

        S = string_dtype

        # Promote numeric with unsized string:
        assert_equal(promote_types('bool', S), np.dtype(S+'5'))
        assert_equal(promote_types('b', S), np.dtype(S+'4'))
        assert_equal(promote_types('u1', S), np.dtype(S+'3'))
        assert_equal(promote_types('u2', S), np.dtype(S+'5'))
        assert_equal(promote_types('u4', S), np.dtype(S+'10'))
        assert_equal(promote_types('u8', S), np.dtype(S+'20'))
        assert_equal(promote_types('i1', S), np.dtype(S+'4'))
        assert_equal(promote_types('i2', S), np.dtype(S+'6'))
        assert_equal(promote_types('i4', S), np.dtype(S+'11'))
        assert_equal(promote_types('i8', S), np.dtype(S+'21'))
        # Promote numeric with sized string:
        assert_equal(promote_types('bool', S+'1'), np.dtype(S+'5'))
        assert_equal(promote_types('bool', S+'30'), np.dtype(S+'30'))
        assert_equal(promote_types('b', S+'1'), np.dtype(S+'4'))
        assert_equal(promote_types('b', S+'30'), np.dtype(S+'30'))
        assert_equal(promote_types('u1', S+'1'), np.dtype(S+'3'))
        assert_equal(promote_types('u1', S+'30'), np.dtype(S+'30'))
        assert_equal(promote_types('u2', S+'1'), np.dtype(S+'5'))
        assert_equal(promote_types('u2', S+'30'), np.dtype(S+'30'))
        assert_equal(promote_types('u4', S+'1'), np.dtype(S+'10'))
        assert_equal(promote_types('u4', S+'30'), np.dtype(S+'30'))
        assert_equal(promote_types('u8', S+'1'), np.dtype(S+'20'))
        assert_equal(promote_types('u8', S+'30'), np.dtype(S+'30'))
        # Promote with object:
        assert_equal(promote_types('O', S+'30'), np.dtype('O'))

    @pytest.mark.parametrize(["dtype1", "dtype2"],
            [[np.dtype("V6"), np.dtype("V10")],  # mismatch shape
             # Mismatching names:
             [np.dtype([("name1", "i8")]), np.dtype([("name2", "i8")])],
            ])
    def test_invalid_void_promotion(self, dtype1, dtype2):
        with pytest.raises(TypeError):
            np.promote_types(dtype1, dtype2)

    @pytest.mark.parametrize(["dtype1", "dtype2"],
            [[np.dtype("V10"), np.dtype("V10")],
             [np.dtype([("name1", "i8")]),
              np.dtype([("name1", np.dtype("i8").newbyteorder())])],
             [np.dtype("i8,i8"), np.dtype("i8,>i8")],
             [np.dtype("i8,i8"), np.dtype("i4,i4")],
            ])
    def test_valid_void_promotion(self, dtype1, dtype2):
        assert np.promote_types(dtype1, dtype2) == dtype1

    @pytest.mark.parametrize("dtype",
            list(np.typecodes["All"]) +
            ["i,i", "10i", "S3", "S100", "U3", "U100", rational])
    def test_promote_identical_types_metadata(self, dtype):
        # The same type passed in twice to promote types always
        # preserves metadata
        metadata = {1: 1}
        dtype = np.dtype(dtype, metadata=metadata)

        res = np.promote_types(dtype, dtype)
        assert res.metadata == dtype.metadata

        # byte-swapping preserves and makes the dtype native:
        dtype = dtype.newbyteorder()
        if dtype.isnative:
            # The type does not have byte swapping
            return

        res = np.promote_types(dtype, dtype)

        # Metadata is (currently) generally lost on byte-swapping (except for
        # unicode.
        if dtype.char != "U":
            assert res.metadata is None
        else:
            assert res.metadata == metadata
        assert res.isnative

    @pytest.mark.slow
    @pytest.mark.filterwarnings('ignore:Promotion of numbers:FutureWarning')
    @pytest.mark.parametrize(["dtype1", "dtype2"],
            itertools.product(
                list(np.typecodes["All"]) +
                ["i,i", "S3", "S100", "U3", "U100", rational],
                repeat=2))
    def test_promote_types_metadata(self, dtype1, dtype2):
        """Metadata handling in promotion does not appear formalized
        right now in NumPy. This test should thus be considered to
        document behaviour, rather than test the correct definition of it.

        This test is very ugly, it was useful for rewriting part of the
        promotion, but probably should eventually be replaced/deleted
        (i.e. when metadata handling in promotion is better defined).
        """
        metadata1 = {1: 1}
        metadata2 = {2: 2}
        dtype1 = np.dtype(dtype1, metadata=metadata1)
        dtype2 = np.dtype(dtype2, metadata=metadata2)

        try:
            res = np.promote_types(dtype1, dtype2)
        except TypeError:
            # Promotion failed, this test only checks metadata
            return

        if res.char not in "USV" or res.names is not None or res.shape != ():
            # All except string dtypes (and unstructured void) lose metadata
            # on promotion (unless both dtypes are identical).
            # At some point structured ones did not, but were restrictive.
            assert res.metadata is None
        elif res == dtype1:
            # If one result is the result, it is usually returned unchanged:
            assert res is dtype1
        elif res == dtype2:
            # dtype1 may have been cast to the same type/kind as dtype2.
            # If the resulting dtype is identical we currently pick the cast
            # version of dtype1, which lost the metadata:
            if np.promote_types(dtype1, dtype2.kind) == dtype2:
                res.metadata is None
            else:
                res.metadata == metadata2
        else:
            assert res.metadata is None

        # Try again for byteswapped version
        dtype1 = dtype1.newbyteorder()
        assert dtype1.metadata == metadata1
        res_bs = np.promote_types(dtype1, dtype2)
        assert res_bs == res
        assert res_bs.metadata == res.metadata

    def test_can_cast(self):
        assert_(np.can_cast(np.int32, np.int64))
        assert_(np.can_cast(np.float64, complex))
        assert_(not np.can_cast(complex, float))

        assert_(np.can_cast('i8', 'f8'))
        assert_(not np.can_cast('i8', 'f4'))
        assert_(np.can_cast('i4', 'S11'))

        assert_(np.can_cast('i8', 'i8', 'no'))
        assert_(not np.can_cast('<i8', '>i8', 'no'))

        assert_(np.can_cast('<i8', '>i8', 'equiv'))
        assert_(not np.can_cast('<i4', '>i8', 'equiv'))

        assert_(np.can_cast('<i4', '>i8', 'safe'))
        assert_(not np.can_cast('<i8', '>i4', 'safe'))

        assert_(np.can_cast('<i8', '>i4', 'same_kind'))
        assert_(not np.can_cast('<i8', '>u4', 'same_kind'))

        assert_(np.can_cast('<i8', '>u4', 'unsafe'))

        assert_(np.can_cast('bool', 'S5'))
        assert_(not np.can_cast('bool', 'S4'))

        assert_(np.can_cast('b', 'S4'))
        assert_(not np.can_cast('b', 'S3'))

        assert_(np.can_cast('u1', 'S3'))
        assert_(not np.can_cast('u1', 'S2'))
        assert_(np.can_cast('u2', 'S5'))
        assert_(not np.can_cast('u2', 'S4'))
        assert_(np.can_cast('u4', 'S10'))
        assert_(not np.can_cast('u4', 'S9'))
        assert_(np.can_cast('u8', 'S20'))
        assert_(not np.can_cast('u8', 'S19'))

        assert_(np.can_cast('i1', 'S4'))
        assert_(not np.can_cast('i1', 'S3'))
        assert_(np.can_cast('i2', 'S6'))
        assert_(not np.can_cast('i2', 'S5'))
        assert_(np.can_cast('i4', 'S11'))
        assert_(not np.can_cast('i4', 'S10'))
        assert_(np.can_cast('i8', 'S21'))
        assert_(not np.can_cast('i8', 'S20'))

        assert_(np.can_cast('bool', 'S5'))
        assert_(not np.can_cast('bool', 'S4'))

        assert_(np.can_cast('b', 'U4'))
        assert_(not np.can_cast('b', 'U3'))

        assert_(np.can_cast('u1', 'U3'))
        assert_(not np.can_cast('u1', 'U2'))
        assert_(np.can_cast('u2', 'U5'))
        assert_(not np.can_cast('u2', 'U4'))
        assert_(np.can_cast('u4', 'U10'))
        assert_(not np.can_cast('u4', 'U9'))
        assert_(np.can_cast('u8', 'U20'))
        assert_(not np.can_cast('u8', 'U19'))

        assert_(np.can_cast('i1', 'U4'))
        assert_(not np.can_cast('i1', 'U3'))
        assert_(np.can_cast('i2', 'U6'))
        assert_(not np.can_cast('i2', 'U5'))
        assert_(np.can_cast('i4', 'U11'))
        assert_(not np.can_cast('i4', 'U10'))
        assert_(np.can_cast('i8', 'U21'))
        assert_(not np.can_cast('i8', 'U20'))

        assert_raises(TypeError, np.can_cast, 'i4', None)
        assert_raises(TypeError, np.can_cast, None, 'i4')

        # Also test keyword arguments
        assert_(np.can_cast(from_=np.int32, to=np.int64))

    def test_can_cast_simple_to_structured(self):
        # Non-structured can only be cast to structured in 'unsafe' mode.
        assert_(not np.can_cast('i4', 'i4,i4'))
        assert_(not np.can_cast('i4', 'i4,i2'))
        assert_(np.can_cast('i4', 'i4,i4', casting='unsafe'))
        assert_(np.can_cast('i4', 'i4,i2', casting='unsafe'))
        # Even if there is just a single field which is OK.
        assert_(not np.can_cast('i2', [('f1', 'i4')]))
        assert_(not np.can_cast('i2', [('f1', 'i4')], casting='same_kind'))
        assert_(np.can_cast('i2', [('f1', 'i4')], casting='unsafe'))
        # It should be the same for recursive structured or subarrays.
        assert_(not np.can_cast('i2', [('f1', 'i4,i4')]))
        assert_(np.can_cast('i2', [('f1', 'i4,i4')], casting='unsafe'))
        assert_(not np.can_cast('i2', [('f1', '(2,3)i4')]))
        assert_(np.can_cast('i2', [('f1', '(2,3)i4')], casting='unsafe'))

    def test_can_cast_structured_to_simple(self):
        # Need unsafe casting for structured to simple.
        assert_(not np.can_cast([('f1', 'i4')], 'i4'))
        assert_(np.can_cast([('f1', 'i4')], 'i4', casting='unsafe'))
        assert_(np.can_cast([('f1', 'i4')], 'i2', casting='unsafe'))
        # Since it is unclear what is being cast, multiple fields to
        # single should not work even for unsafe casting.
        assert_(not np.can_cast('i4,i4', 'i4', casting='unsafe'))
        # But a single field inside a single field is OK.
        assert_(not np.can_cast([('f1', [('x', 'i4')])], 'i4'))
        assert_(np.can_cast([('f1', [('x', 'i4')])], 'i4', casting='unsafe'))
        # And a subarray is fine too - it will just take the first element
        # (arguably not very consistently; might also take the first field).
        assert_(not np.can_cast([('f0', '(3,)i4')], 'i4'))
        assert_(np.can_cast([('f0', '(3,)i4')], 'i4', casting='unsafe'))
        # But a structured subarray with multiple fields should fail.
        assert_(not np.can_cast([('f0', ('i4,i4'), (2,))], 'i4',
                                casting='unsafe'))

    def test_can_cast_values(self):
        # gh-5917
        for dt in np.sctypes['int'] + np.sctypes['uint']:
            ii = np.iinfo(dt)
            assert_(np.can_cast(ii.min, dt))
            assert_(np.can_cast(ii.max, dt))
            assert_(not np.can_cast(ii.min - 1, dt))
            assert_(not np.can_cast(ii.max + 1, dt))

        for dt in np.sctypes['float']:
            fi = np.finfo(dt)
            assert_(np.can_cast(fi.min, dt))
            assert_(np.can_cast(fi.max, dt))


# Custom exception class to test exception propagation in fromiter
class NIterError(Exception):
    pass


class TestFromiter:
    def makegen(self):
        return (x**2 for x in range(24))

    def test_types(self):
        ai32 = np.fromiter(self.makegen(), np.int32)
        ai64 = np.fromiter(self.makegen(), np.int64)
        af = np.fromiter(self.makegen(), float)
        assert_(ai32.dtype == np.dtype(np.int32))
        assert_(ai64.dtype == np.dtype(np.int64))
        assert_(af.dtype == np.dtype(float))

    def test_lengths(self):
        expected = np.array(list(self.makegen()))
        a = np.fromiter(self.makegen(), int)
        a20 = np.fromiter(self.makegen(), int, 20)
        assert_(len(a) == len(expected))
        assert_(len(a20) == 20)
        assert_raises(ValueError, np.fromiter,
                          self.makegen(), int, len(expected) + 10)

    def test_values(self):
        expected = np.array(list(self.makegen()))
        a = np.fromiter(self.makegen(), int)
        a20 = np.fromiter(self.makegen(), int, 20)
        assert_(np.alltrue(a == expected, axis=0))
        assert_(np.alltrue(a20 == expected[:20], axis=0))

    def load_data(self, n, eindex):
        # Utility method for the issue 2592 tests.
        # Raise an exception at the desired index in the iterator.
        for e in range(n):
            if e == eindex:
                raise NIterError('error at index %s' % eindex)
            yield e

    @pytest.mark.parametrize("dtype", [int, object])
    @pytest.mark.parametrize(["count", "error_index"], [(10, 5), (10, 9)])
    def test_2592(self, count, error_index, dtype):
        # Test iteration exceptions are correctly raised. The data/generator
        # has `count` elements but errors at `error_index`
        iterable = self.load_data(count, error_index)
        with pytest.raises(NIterError):
            np.fromiter(iterable, dtype=dtype, count=count)

    @pytest.mark.parametrize("dtype", ["S", "S0", "V0", "U0"])
    def test_empty_not_structured(self, dtype):
        # Note, "S0" could be allowed at some point, so long "S" (without
        # any length) is rejected.
        with pytest.raises(ValueError, match="Must specify length"):
            np.fromiter([], dtype=dtype)

    @pytest.mark.parametrize(["dtype", "data"],
            [("d", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
             ("O", [1, 2, 3, 4, 5, 6, 7, 8, 9]),
             ("i,O", [(1, 2), (5, 4), (2, 3), (9, 8), (6, 7)]),
             # subarray dtypes (important because their dimensions end up
             # in the result arrays dimension:
             ("2i", [(1, 2), (5, 4), (2, 3), (9, 8), (6, 7)]),
             (np.dtype(("O", (2, 3))),
              [((1, 2, 3), (3, 4, 5)), ((3, 2, 1), (5, 4, 3))])])
    @pytest.mark.parametrize("length_hint", [0, 1])
    def test_growth_and_complicated_dtypes(self, dtype, data, length_hint):
        dtype = np.dtype(dtype)

        data = data * 100  # make sure we realloc a bit

        class MyIter:
            # Class/example from gh-15789
            def __length_hint__(self):
                # only required to be an estimate, this is legal
                return length_hint  # 0 or 1

            def __iter__(self):
                return iter(data)

        res = np.fromiter(MyIter(), dtype=dtype)
        expected = np.array(data, dtype=dtype)

        assert_array_equal(res, expected)

    def test_empty_result(self):
        class MyIter:
            def __length_hint__(self):
                return 10

            def __iter__(self):
                return iter([])  # actual iterator is empty.

        res = np.fromiter(MyIter(), dtype="d")
        assert res.shape == (0,)
        assert res.dtype == "d"

    def test_too_few_items(self):
        msg = "iterator too short: Expected 10 but iterator had only 3 items."
        with pytest.raises(ValueError, match=msg):
            np.fromiter([1, 2, 3], count=10, dtype=int)

    def test_failed_itemsetting(self):
        with pytest.raises(TypeError):
            np.fromiter([1, None, 3], dtype=int)

        # The following manages to hit somewhat trickier code paths:
        iterable = ((2, 3, 4) for i in range(5))
        with pytest.raises(ValueError):
            np.fromiter(iterable, dtype=np.dtype((int, 2)))

class TestNonzero:
    def test_nonzero_trivial(self):
        assert_equal(np.count_nonzero(np.array([])), 0)
        assert_equal(np.count_nonzero(np.array([], dtype='?')), 0)
        assert_equal(np.nonzero(np.array([])), ([],))

        assert_equal(np.count_nonzero(np.array([0])), 0)
        assert_equal(np.count_nonzero(np.array([0], dtype='?')), 0)
        assert_equal(np.nonzero(np.array([0])), ([],))

        assert_equal(np.count_nonzero(np.array([1])), 1)
        assert_equal(np.count_nonzero(np.array([1], dtype='?')), 1)
        assert_equal(np.nonzero(np.array([1])), ([0],))

    def test_nonzero_zerod(self):
        assert_equal(np.count_nonzero(np.array(0)), 0)
        assert_equal(np.count_nonzero(np.array(0, dtype='?')), 0)
        with assert_warns(DeprecationWarning):
            assert_equal(np.nonzero(np.array(0)), ([],))

        assert_equal(np.count_nonzero(np.array(1)), 1)
        assert_equal(np.count_nonzero(np.array(1, dtype='?')), 1)
        with assert_warns(DeprecationWarning):
            assert_equal(np.nonzero(np.array(1)), ([0],))

    def test_nonzero_onedim(self):
        x = np.array([1, 0, 2, -1, 0, 0, 8])
        assert_equal(np.count_nonzero(x), 4)
        assert_equal(np.count_nonzero(x), 4)
        assert_equal(np.nonzero(x), ([0, 2, 3, 6],))

        # x = np.array([(1, 2), (0, 0), (1, 1), (-1, 3), (0, 7)],
        #              dtype=[('a', 'i4'), ('b', 'i2')])
        x = np.array([(1, 2, -5, -3), (0, 0, 2, 7), (1, 1, 0, 1), (-1, 3, 1, 0), (0, 7, 0, 4)],
                     dtype=[('a', 'i4'), ('b', 'i2'), ('c', 'i1'), ('d', 'i8')])
        assert_equal(np.count_nonzero(x['a']), 3)
        assert_equal(np.count_nonzero(x['b']), 4)
        assert_equal(np.count_nonzero(x['c']), 3)
        assert_equal(np.count_nonzero(x['d']), 4)
        assert_equal(np.nonzero(x['a']), ([0, 2, 3],))
        assert_equal(np.nonzero(x['b']), ([0, 2, 3, 4],))

    def test_nonzero_twodim(self):
        x = np.array([[0, 1, 0], [2, 0, 3]])
        assert_equal(np.count_nonzero(x.astype('i1')), 3)
        assert_equal(np.count_nonzero(x.astype('i2')), 3)
        assert_equal(np.count_nonzero(x.astype('i4')), 3)
        assert_equal(np.count_nonzero(x.astype('i8')), 3)
        assert_equal(np.nonzero(x), ([0, 1, 1], [1, 0, 2]))

        x = np.eye(3)
        assert_equal(np.count_nonzero(x.astype('i1')), 3)
        assert_equal(np.count_nonzero(x.astype('i2')), 3)
        assert_equal(np.count_nonzero(x.astype('i4')), 3)
        assert_equal(np.count_nonzero(x.astype('i8')), 3)
        assert_equal(np.nonzero(x), ([0, 1, 2], [0, 1, 2]))

        x = np.array([[(0, 1), (0, 0), (1, 11)],
                   [(1, 1), (1, 0), (0, 0)],
                   [(0, 0), (1, 5), (0, 1)]], dtype=[('a', 'f4'), ('b', 'u1')])
        assert_equal(np.count_nonzero(x['a']), 4)
        assert_equal(np.count_nonzero(x['b']), 5)
        assert_equal(np.nonzero(x['a']), ([0, 1, 1, 2], [2, 0, 1, 1]))
        assert_equal(np.nonzero(x['b']), ([0, 0, 1, 2, 2], [0, 2, 0, 1, 2]))

        assert_(not x['a'].T.flags.aligned)
        assert_equal(np.count_nonzero(x['a'].T), 4)
        assert_equal(np.count_nonzero(x['b'].T), 5)
        assert_equal(np.nonzero(x['a'].T), ([0, 1, 1, 2], [1, 1, 2, 0]))
        assert_equal(np.nonzero(x['b'].T), ([0, 0, 1, 2, 2], [0, 1, 2, 0, 2]))

    def test_sparse(self):
        # test special sparse condition boolean code path
        for i in range(20):
            c = np.zeros(200, dtype=bool)
            c[i::20] = True
            assert_equal(np.nonzero(c)[0], np.arange(i, 200 + i, 20))

            c = np.zeros(400, dtype=bool)
            c[10 + i:20 + i] = True
            c[20 + i*2] = True
            assert_equal(np.nonzero(c)[0],
                         np.concatenate((np.arange(10 + i, 20 + i), [20 + i*2])))

    def test_return_type(self):
        class C(np.ndarray):
            pass

        for view in (C, np.ndarray):
            for nd in range(1, 4):
                shape = tuple(range(2, 2+nd))
                x = np.arange(np.prod(shape)).reshape(shape).view(view)
                for nzx in (np.nonzero(x), x.nonzero()):
                    for nzx_i in nzx:
                        assert_(type(nzx_i) is np.ndarray)
                        assert_(nzx_i.flags.writeable)

    def test_count_nonzero_axis(self):
        # Basic check of functionality
        m = np.array([[0, 1, 7, 0, 0], [3, 0, 0, 2, 19]])

        expected = np.array([1, 1, 1, 1, 1])
        assert_equal(np.count_nonzero(m, axis=0), expected)

        expected = np.array([2, 3])
        assert_equal(np.count_nonzero(m, axis=1), expected)

        assert_raises(ValueError, np.count_nonzero, m, axis=(1, 1))
        assert_raises(TypeError, np.count_nonzero, m, axis='foo')
        assert_raises(np.AxisError, np.count_nonzero, m, axis=3)
        assert_raises(TypeError, np.count_nonzero,
                      m, axis=np.array([[1], [2]]))

    def test_count_nonzero_axis_all_dtypes(self):
        # More thorough test that the axis argument is respected
        # for all dtypes and responds correctly when presented with
        # either integer or tuple arguments for axis
        msg = "Mismatch for dtype: %s"

        def assert_equal_w_dt(a, b, err_msg):
            assert_equal(a.dtype, b.dtype, err_msg=err_msg)
            assert_equal(a, b, err_msg=err_msg)

        for dt in np.typecodes['All']:
            err_msg = msg % (np.dtype(dt).name,)

            if dt != 'V':
                if dt != 'M':
                    m = np.zeros((3, 3), dtype=dt)
                    n = np.ones(1, dtype=dt)

                    m[0, 0] = n[0]
                    m[1, 0] = n[0]

                else:  # np.zeros doesn't work for np.datetime64
                    m = np.array(['1970-01-01'] * 9)
                    m = m.reshape((3, 3))

                    m[0, 0] = '1970-01-12'
                    m[1, 0] = '1970-01-12'
                    m = m.astype(dt)

                expected = np.array([2, 0, 0], dtype=np.intp)
                assert_equal_w_dt(np.count_nonzero(m, axis=0),
                                  expected, err_msg=err_msg)

                expected = np.array([1, 1, 0], dtype=np.intp)
                assert_equal_w_dt(np.count_nonzero(m, axis=1),
                                  expected, err_msg=err_msg)

                expected = np.array(2)
                assert_equal(np.count_nonzero(m, axis=(0, 1)),
                             expected, err_msg=err_msg)
                assert_equal(np.count_nonzero(m, axis=None),
                             expected, err_msg=err_msg)
                assert_equal(np.count_nonzero(m),
                             expected, err_msg=err_msg)

            if dt == 'V':
                # There are no 'nonzero' objects for np.void, so the testing
                # setup is slightly different for this dtype
                m = np.array([np.void(1)] * 6).reshape((2, 3))

                expected = np.array([0, 0, 0], dtype=np.intp)
                assert_equal_w_dt(np.count_nonzero(m, axis=0),
                                  expected, err_msg=err_msg)

                expected = np.array([0, 0], dtype=np.intp)
                assert_equal_w_dt(np.count_nonzero(m, axis=1),
                                  expected, err_msg=err_msg)

                expected = np.array(0)
                assert_equal(np.count_nonzero(m, axis=(0, 1)),
                             expected, err_msg=err_msg)
                assert_equal(np.count_nonzero(m, axis=None),
                             expected, err_msg=err_msg)
                assert_equal(np.count_nonzero(m),
                             expected, err_msg=err_msg)

    def test_count_nonzero_axis_consistent(self):
        # Check that the axis behaviour for valid axes in
        # non-special cases is consistent (and therefore
        # correct) by checking it against an integer array
        # that is then casted to the generic object dtype
        from itertools import combinations, permutations

        axis = (0, 1, 2, 3)
        size = (5, 5, 5, 5)
        msg = "Mismatch for axis: %s"

        rng = np.random.RandomState(1234)
        m = rng.randint(-100, 100, size=size)
        n = m.astype(object)

        for length in range(len(axis)):
            for combo in combinations(axis, length):
                for perm in permutations(combo):
                    assert_equal(
                        np.count_nonzero(m, axis=perm),
                        np.count_nonzero(n, axis=perm),
                        err_msg=msg % (perm,))

    def test_countnonzero_axis_empty(self):
        a = np.array([[0, 0, 1], [1, 0, 1]])
        assert_equal(np.count_nonzero(a, axis=()), a.astype(bool))

    def test_countnonzero_keepdims(self):
        a = np.array([[0, 0, 1, 0],
                      [0, 3, 5, 0],
                      [7, 9, 2, 0]])
        assert_equal(np.count_nonzero(a, axis=0, keepdims=True),
                     [[1, 2, 3, 0]])
        assert_equal(np.count_nonzero(a, axis=1, keepdims=True),
                     [[1], [2], [3]])
        assert_equal(np.count_nonzero(a, keepdims=True),
                     [[6]])

    def test_array_method(self):
        # Tests that the array method
        # call to nonzero works
        m = np.array([[1, 0, 0], [4, 0, 6]])
        tgt = [[0, 1, 1], [0, 0, 2]]

        assert_equal(m.nonzero(), tgt)

    def test_nonzero_invalid_object(self):
        # gh-9295
        a = np.array([np.array([1, 2]), 3], dtype=object)
        assert_raises(ValueError, np.nonzero, a)

        class BoolErrors:
            def __bool__(self):
                raise ValueError("Not allowed")

        assert_raises(ValueError, np.nonzero, np.array([BoolErrors()]))

    def test_nonzero_sideeffect_safety(self):
        # gh-13631
        class FalseThenTrue:
            _val = False
            def __bool__(self):
                try:
                    return self._val
                finally:
                    self._val = True

        class TrueThenFalse:
            _val = True
            def __bool__(self):
                try:
                    return self._val
                finally:
                    self._val = False

        # result grows on the second pass
        a = np.array([True, FalseThenTrue()])
        assert_raises(RuntimeError, np.nonzero, a)

        a = np.array([[True], [FalseThenTrue()]])
        assert_raises(RuntimeError, np.nonzero, a)

        # result shrinks on the second pass
        a = np.array([False, TrueThenFalse()])
        assert_raises(RuntimeError, np.nonzero, a)

        a = np.array([[False], [TrueThenFalse()]])
        assert_raises(RuntimeError, np.nonzero, a)

    def test_nonzero_sideffects_structured_void(self):
        # Checks that structured void does not mutate alignment flag of
        # original array.
        arr = np.zeros(5, dtype="i1,i8,i8")  # `ones` may short-circuit
        assert arr.flags.aligned  # structs are considered "aligned"
        assert not arr["f2"].flags.aligned
        # make sure that nonzero/count_nonzero do not flip the flag:
        np.nonzero(arr)
        assert arr.flags.aligned
        np.count_nonzero(arr)
        assert arr.flags.aligned

    def test_nonzero_exception_safe(self):
        # gh-13930

        class ThrowsAfter:
            def __init__(self, iters):
                self.iters_left = iters

            def __bool__(self):
                if self.iters_left == 0:
                    raise ValueError("called `iters` times")

                self.iters_left -= 1
                return True

        """
        Test that a ValueError is raised instead of a SystemError

        If the __bool__ function is called after the error state is set,
        Python (cpython) will raise a SystemError.
        """

        # assert that an exception in first pass is handled correctly
        a = np.array([ThrowsAfter(5)]*10)
        assert_raises(ValueError, np.nonzero, a)

        # raise exception in second pass for 1-dimensional loop
        a = np.array([ThrowsAfter(15)]*10)
        assert_raises(ValueError, np.nonzero, a)

        # raise exception in second pass for n-dimensional loop
        a = np.array([[ThrowsAfter(15)]]*10)
        assert_raises(ValueError, np.nonzero, a)

    @pytest.mark.skipif(IS_WASM, reason="wasm doesn't have threads")
    def test_structured_threadsafety(self):
        # Nonzero (and some other functions) should be threadsafe for
        # structured datatypes, see gh-15387. This test can behave randomly.
        from concurrent.futures import ThreadPoolExecutor

        # Create a deeply nested dtype to make a failure more likely:
        dt = np.dtype([("", "f8")])
        dt = np.dtype([("", dt)])
        dt = np.dtype([("", dt)] * 2)
        # The array should be large enough to likely run into threading issues
        arr = np.random.uniform(size=(5000, 4)).view(dt)[:, 0]
        def func(arr):
            arr.nonzero()

        tpe = ThreadPoolExecutor(max_workers=8)
        futures = [tpe.submit(func, arr) for _ in range(10)]
        for f in futures:
            f.result()

        assert arr.dtype is dt


class TestIndex:
    def test_boolean(self):
        a = rand(3, 5, 8)
        V = rand(5, 8)
        g1 = randint(0, 5, size=15)
        g2 = randint(0, 8, size=15)
        V[g1, g2] = -V[g1, g2]
        assert_((np.array([a[0][V > 0], a[1][V > 0], a[2][V > 0]]) == a[:, V > 0]).all())

    def test_boolean_edgecase(self):
        a = np.array([], dtype='int32')
        b = np.array([], dtype='bool')
        c = a[b]
        assert_equal(c, [])
        assert_equal(c.dtype, np.dtype('int32'))


class TestBinaryRepr:
    def test_zero(self):
        assert_equal(np.binary_repr(0), '0')

    def test_positive(self):
        assert_equal(np.binary_repr(10), '1010')
        assert_equal(np.binary_repr(12522),
                     '11000011101010')
        assert_equal(np.binary_repr(10736848),
                     '101000111101010011010000')

    def test_negative(self):
        assert_equal(np.binary_repr(-1), '-1')
        assert_equal(np.binary_repr(-10), '-1010')
        assert_equal(np.binary_repr(-12522),
                     '-11000011101010')
        assert_equal(np.binary_repr(-10736848),
                     '-101000111101010011010000')

    def test_sufficient_width(self):
        assert_equal(np.binary_repr(0, width=5), '00000')
        assert_equal(np.binary_repr(10, width=7), '0001010')
        assert_equal(np.binary_repr(-5, width=7), '1111011')

    def test_neg_width_boundaries(self):
        # see gh-8670

        # Ensure that the example in the issue does not
        # break before proceeding to a more thorough test.
        assert_equal(np.binary_repr(-128, width=8), '10000000')

        for width in range(1, 11):
            num = -2**(width - 1)
            exp = '1' + (width - 1) * '0'
            assert_equal(np.binary_repr(num, width=width), exp)

    def test_large_neg_int64(self):
        # See gh-14289.
        assert_equal(np.binary_repr(np.int64(-2**62), width=64),
                     '11' + '0'*62)


class TestBaseRepr:
    def test_base3(self):
        assert_equal(np.base_repr(3**5, 3), '100000')

    def test_positive(self):
        assert_equal(np.base_repr(12, 10), '12')
        assert_equal(np.base_repr(12, 10, 4), '000012')
        assert_equal(np.base_repr(12, 4), '30')
        assert_equal(np.base_repr(3731624803700888, 36), '10QR0ROFCEW')

    def test_negative(self):
        assert_equal(np.base_repr(-12, 10), '-12')
        assert_equal(np.base_repr(-12, 10, 4), '-000012')
        assert_equal(np.base_repr(-12, 4), '-30')

    def test_base_range(self):
        with assert_raises(ValueError):
            np.base_repr(1, 1)
        with assert_raises(ValueError):
            np.base_repr(1, 37)


class TestArrayComparisons:
    def test_array_equal(self):
        res = np.array_equal(np.array([1, 2]), np.array([1, 2]))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equal(np.array([1, 2]), np.array([1, 2, 3]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equal(np.array([1, 2]), np.array([3, 4]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equal(np.array([1, 2]), np.array([1, 3]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equal(np.array(['a'], dtype='S1'), np.array(['a'], dtype='S1'))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equal(np.array([('a', 1)], dtype='S1,u4'),
                             np.array([('a', 1)], dtype='S1,u4'))
        assert_(res)
        assert_(type(res) is bool)

    def test_array_equal_equal_nan(self):
        # Test array_equal with equal_nan kwarg
        a1 = np.array([1, 2, np.nan])
        a2 = np.array([1, np.nan, 2])
        a3 = np.array([1, 2, np.inf])

        # equal_nan=False by default
        assert_(not np.array_equal(a1, a1))
        assert_(np.array_equal(a1, a1, equal_nan=True))
        assert_(not np.array_equal(a1, a2, equal_nan=True))
        # nan's not conflated with inf's
        assert_(not np.array_equal(a1, a3, equal_nan=True))
        # 0-D arrays
        a = np.array(np.nan)
        assert_(not np.array_equal(a, a))
        assert_(np.array_equal(a, a, equal_nan=True))
        # Non-float dtype - equal_nan should have no effect
        a = np.array([1, 2, 3], dtype=int)
        assert_(np.array_equal(a, a))
        assert_(np.array_equal(a, a, equal_nan=True))
        # Multi-dimensional array
        a = np.array([[0, 1], [np.nan, 1]])
        assert_(not np.array_equal(a, a))
        assert_(np.array_equal(a, a, equal_nan=True))
        # Complex values
        a, b = [np.array([1 + 1j])]*2
        a.real, b.imag = np.nan, np.nan
        assert_(not np.array_equal(a, b, equal_nan=False))
        assert_(np.array_equal(a, b, equal_nan=True))

    def test_none_compares_elementwise(self):
        a = np.array([None, 1, None], dtype=object)
        assert_equal(a == None, [True, False, True])
        assert_equal(a != None, [False, True, False])

        a = np.ones(3)
        assert_equal(a == None, [False, False, False])
        assert_equal(a != None, [True, True, True])

    def test_array_equiv(self):
        res = np.array_equiv(np.array([1, 2]), np.array([1, 2]))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([1, 2, 3]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([3, 4]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([1, 3]))
        assert_(not res)
        assert_(type(res) is bool)

        res = np.array_equiv(np.array([1, 1]), np.array([1]))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 1]), np.array([[1], [1]]))
        assert_(res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([2]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([[1], [2]]))
        assert_(not res)
        assert_(type(res) is bool)
        res = np.array_equiv(np.array([1, 2]), np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]))
        assert_(not res)
        assert_(type(res) is bool)

    @pytest.mark.parametrize("dtype", ["V0", "V3", "V10"])
    def test_compare_unstructured_voids(self, dtype):
        zeros = np.zeros(3, dtype=dtype)

        assert_array_equal(zeros, zeros)
        assert not (zeros != zeros).any()

        if dtype == "V0":
            # Can't test != of actually different data
            return

        nonzeros = np.array([b"1", b"2", b"3"], dtype=dtype)

        assert not (zeros == nonzeros).any()
        assert (zeros != nonzeros).all()


def assert_array_strict_equal(x, y):
    assert_array_equal(x, y)
    # Check flags, 32 bit arches typically don't provide 16 byte alignment
    if ((x.dtype.alignment <= 8 or
            np.intp().dtype.itemsize != 4) and
            sys.platform != 'win32'):
        assert_(x.flags == y.flags)
    else:
        assert_(x.flags.owndata == y.flags.owndata)
        assert_(x.flags.writeable == y.flags.writeable)
        assert_(x.flags.c_contiguous == y.flags.c_contiguous)
        assert_(x.flags.f_contiguous == y.flags.f_contiguous)
        assert_(x.flags.writebackifcopy == y.flags.writebackifcopy)
    # check endianness
    assert_(x.dtype.isnative == y.dtype.isnative)


class TestClip:
    def setup_method(self):
        self.nr = 5
        self.nc = 3

    def fastclip(self, a, m, M, out=None, casting=None):
        if out is None:
            if casting is None:
                return a.clip(m, M)
            else:
                return a.clip(m, M, casting=casting)
        else:
            if casting is None:
                return a.clip(m, M, out)
            else:
                return a.clip(m, M, out, casting=casting)

    def clip(self, a, m, M, out=None):
        # use slow-clip
        selector = np.less(a, m) + 2*np.greater(a, M)
        return selector.choose((a, m, M), out=out)

    # Handy functions
    def _generate_data(self, n, m):
        return randn(n, m)

    def _generate_data_complex(self, n, m):
        return randn(n, m) + 1.j * rand(n, m)

    def _generate_flt_data(self, n, m):
        return (randn(n, m)).astype(np.float32)

    def _neg_byteorder(self, a):
        a = np.asarray(a)
        if sys.byteorder == 'little':
            a = a.astype(a.dtype.newbyteorder('>'))
        else:
            a = a.astype(a.dtype.newbyteorder('<'))
        return a

    def _generate_non_native_data(self, n, m):
        data = randn(n, m)
        data = self._neg_byteorder(data)
        assert_(not data.dtype.isnative)
        return data

    def _generate_int_data(self, n, m):
        return (10 * rand(n, m)).astype(np.int64)

    def _generate_int32_data(self, n, m):
        return (10 * rand(n, m)).astype(np.int32)

    # Now the real test cases

    @pytest.mark.parametrize("dtype", '?bhilqpBHILQPefdgFDGO')
    def test_ones_pathological(self, dtype):
        # for preservation of behavior described in
        # gh-12519; amin > amax behavior may still change
        # in the future
        arr = np.ones(10, dtype=dtype)
        expected = np.zeros(10, dtype=dtype)
        actual = np.clip(arr, 1, 0)
        if dtype == 'O':
            assert actual.tolist() == expected.tolist()
        else:
            assert_equal(actual, expected)

    def test_simple_double(self):
        # Test native double input with scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = 0.1
        M = 0.6
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_simple_int(self):
        # Test native int input with scalar min/max.
        a = self._generate_int_data(self.nr, self.nc)
        a = a.astype(int)
        m = -2
        M = 4
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_array_double(self):
        # Test native double input with array min/max.
        a = self._generate_data(self.nr, self.nc)
        m = np.zeros(a.shape)
        M = m + 0.5
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_simple_nonnative(self):
        # Test non native double input with scalar min/max.
        # Test native double input with non native double scalar min/max.
        a = self._generate_non_native_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

        # Test native double input with non native double scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = self._neg_byteorder(0.6)
        assert_(not M.dtype.isnative)
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_equal(ac, act)

    def test_simple_complex(self):
        # Test native complex input with native double scalar min/max.
        # Test native input with complex double scalar min/max.
        a = 3 * self._generate_data_complex(self.nr, self.nc)
        m = -0.5
        M = 1.
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

        # Test native input with complex double scalar min/max.
        a = 3 * self._generate_data(self.nr, self.nc)
        m = -0.5 + 1.j
        M = 1. + 2.j
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_clip_complex(self):
        # Address Issue gh-5354 for clipping complex arrays
        # Test native complex input without explicit min/max
        # ie, either min=None or max=None
        a = np.ones(10, dtype=complex)
        m = a.min()
        M = a.max()
        am = self.fastclip(a, m, None)
        aM = self.fastclip(a, None, M)
        assert_array_strict_equal(am, a)
        assert_array_strict_equal(aM, a)

    def test_clip_non_contig(self):
        # Test clip for non contiguous native input and native scalar min/max.
        a = self._generate_data(self.nr * 2, self.nc * 3)
        a = a[::2, ::3]
        assert_(not a.flags['F_CONTIGUOUS'])
        assert_(not a.flags['C_CONTIGUOUS'])
        ac = self.fastclip(a, -1.6, 1.7)
        act = self.clip(a, -1.6, 1.7)
        assert_array_strict_equal(ac, act)

    def test_simple_out(self):
        # Test native double input with scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = np.zeros(a.shape)
        act = np.zeros(a.shape)
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    @pytest.mark.parametrize("casting", [None, "unsafe"])
    def test_simple_int32_inout(self, casting):
        # Test native int32 input with double min/max and int32 out.
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.float64(0)
        M = np.float64(2)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        if casting is None:
            with assert_warns(DeprecationWarning):
                # NumPy 1.17.0, 2018-02-24 - casting is unsafe
                self.fastclip(a, m, M, ac, casting=casting)
        else:
            # explicitly passing "unsafe" will silence warning
            self.fastclip(a, m, M, ac, casting=casting)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_int64_out(self):
        # Test native int32 input with int32 scalar min/max and int64 out.
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.int32(-1)
        M = np.int32(1)
        ac = np.zeros(a.shape, dtype=np.int64)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_int64_inout(self):
        # Test native int32 input with double array min/max and int32 out.
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.zeros(a.shape, np.float64)
        M = np.float64(1)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        with assert_warns(DeprecationWarning):
            # NumPy 1.17.0, 2018-02-24 - casting is unsafe
            self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_int32_out(self):
        # Test native double input with scalar min/max and int out.
        a = self._generate_data(self.nr, self.nc)
        m = -1.0
        M = 2.0
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        with assert_warns(DeprecationWarning):
            # NumPy 1.17.0, 2018-02-24 - casting is unsafe
            self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_simple_inplace_01(self):
        # Test native double input with array min/max in-place.
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = np.zeros(a.shape)
        M = 1.0
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a, ac)

    def test_simple_inplace_02(self):
        # Test native double input with scalar min/max in-place.
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = -0.5
        M = 0.6
        self.fastclip(a, m, M, a)
        self.clip(ac, m, M, ac)
        assert_array_strict_equal(a, ac)

    def test_noncontig_inplace(self):
        # Test non contiguous double input with double scalar min/max in-place.
        a = self._generate_data(self.nr * 2, self.nc * 3)
        a = a[::2, ::3]
        assert_(not a.flags['F_CONTIGUOUS'])
        assert_(not a.flags['C_CONTIGUOUS'])
        ac = a.copy()
        m = -0.5
        M = 0.6
        self.fastclip(a, m, M, a)
        self.clip(ac, m, M, ac)
        assert_array_equal(a, ac)

    def test_type_cast_01(self):
        # Test native double input with scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_02(self):
        # Test native int32 input with int32 scalar min/max.
        a = self._generate_int_data(self.nr, self.nc)
        a = a.astype(np.int32)
        m = -2
        M = 4
        ac = self.fastclip(a, m, M)
        act = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_03(self):
        # Test native int32 input with float64 scalar min/max.
        a = self._generate_int32_data(self.nr, self.nc)
        m = -2
        M = 4
        ac = self.fastclip(a, np.float64(m), np.float64(M))
        act = self.clip(a, np.float64(m), np.float64(M))
        assert_array_strict_equal(ac, act)

    def test_type_cast_04(self):
        # Test native int32 input with float32 scalar min/max.
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.float32(-2)
        M = np.float32(4)
        act = self.fastclip(a, m, M)
        ac = self.clip(a, m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_05(self):
        # Test native int32 with double arrays min/max.
        a = self._generate_int_data(self.nr, self.nc)
        m = -0.5
        M = 1.
        ac = self.fastclip(a, m * np.zeros(a.shape), M)
        act = self.clip(a, m * np.zeros(a.shape), M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_06(self):
        # Test native with NON native scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = 0.5
        m_s = self._neg_byteorder(m)
        M = 1.
        act = self.clip(a, m_s, M)
        ac = self.fastclip(a, m_s, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_07(self):
        # Test NON native with native array min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5 * np.ones(a.shape)
        M = 1.
        a_s = self._neg_byteorder(a)
        assert_(not a_s.dtype.isnative)
        act = a_s.clip(m, M)
        ac = self.fastclip(a_s, m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_08(self):
        # Test NON native with native scalar min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 1.
        a_s = self._neg_byteorder(a)
        assert_(not a_s.dtype.isnative)
        ac = self.fastclip(a_s, m, M)
        act = a_s.clip(m, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_09(self):
        # Test native with NON native array min/max.
        a = self._generate_data(self.nr, self.nc)
        m = -0.5 * np.ones(a.shape)
        M = 1.
        m_s = self._neg_byteorder(m)
        assert_(not m_s.dtype.isnative)
        ac = self.fastclip(a, m_s, M)
        act = self.clip(a, m_s, M)
        assert_array_strict_equal(ac, act)

    def test_type_cast_10(self):
        # Test native int32 with float min/max and float out for output argument.
        a = self._generate_int_data(self.nr, self.nc)
        b = np.zeros(a.shape, dtype=np.float32)
        m = np.float32(-0.5)
        M = np.float32(1)
        act = self.clip(a, m, M, out=b)
        ac = self.fastclip(a, m, M, out=b)
        assert_array_strict_equal(ac, act)

    def test_type_cast_11(self):
        # Test non native with native scalar, min/max, out non native
        a = self._generate_non_native_data(self.nr, self.nc)
        b = a.copy()
        b = b.astype(b.dtype.newbyteorder('>'))
        bt = b.copy()
        m = -0.5
        M = 1.
        self.fastclip(a, m, M, out=b)
        self.clip(a, m, M, out=bt)
        assert_array_strict_equal(b, bt)

    def test_type_cast_12(self):
        # Test native int32 input and min/max and float out
        a = self._generate_int_data(self.nr, self.nc)
        b = np.zeros(a.shape, dtype=np.float32)
        m = np.int32(0)
        M = np.int32(1)
        act = self.clip(a, m, M, out=b)
        ac = self.fastclip(a, m, M, out=b)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_simple(self):
        # Test native double input with scalar min/max
        a = self._generate_data(self.nr, self.nc)
        m = -0.5
        M = 0.6
        ac = np.zeros(a.shape)
        act = np.zeros(a.shape)
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_simple2(self):
        # Test native int32 input with double min/max and int32 out
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.float64(0)
        M = np.float64(2)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        with assert_warns(DeprecationWarning):
            # NumPy 1.17.0, 2018-02-24 - casting is unsafe
            self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_simple_int32(self):
        # Test native int32 input with int32 scalar min/max and int64 out
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.int32(-1)
        M = np.int32(1)
        ac = np.zeros(a.shape, dtype=np.int64)
        act = ac.copy()
        self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_array_int32(self):
        # Test native int32 input with double array min/max and int32 out
        a = self._generate_int32_data(self.nr, self.nc)
        m = np.zeros(a.shape, np.float64)
        M = np.float64(1)
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        with assert_warns(DeprecationWarning):
            # NumPy 1.17.0, 2018-02-24 - casting is unsafe
            self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_array_outint32(self):
        # Test native double input with scalar min/max and int out
        a = self._generate_data(self.nr, self.nc)
        m = -1.0
        M = 2.0
        ac = np.zeros(a.shape, dtype=np.int32)
        act = ac.copy()
        with assert_warns(DeprecationWarning):
            # NumPy 1.17.0, 2018-02-24 - casting is unsafe
            self.fastclip(a, m, M, ac)
        self.clip(a, m, M, act)
        assert_array_strict_equal(ac, act)

    def test_clip_with_out_transposed(self):
        # Test that the out argument works when transposed
        a = np.arange(16).reshape(4, 4)
        out = np.empty_like(a).T
        a.clip(4, 10, out=out)
        expected = self.clip(a, 4, 10)
        assert_array_equal(out, expected)

    def test_clip_with_out_memory_overlap(self):
        # Test that the out argument works when it has memory overlap
        a = np.arange(16).reshape(4, 4)
        ac = a.copy()
        a[:-1].clip(4, 10, out=a[1:])
        expected = self.clip(ac[:-1], 4, 10)
        assert_array_equal(a[1:], expected)

    def test_clip_inplace_array(self):
        # Test native double input with array min/max
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = np.zeros(a.shape)
        M = 1.0
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a, ac)

    def test_clip_inplace_simple(self):
        # Test native double input with scalar min/max
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = -0.5
        M = 0.6
        self.fastclip(a, m, M, a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a, ac)

    def test_clip_func_takes_out(self):
        # Ensure that the clip() function takes an out=argument.
        a = self._generate_data(self.nr, self.nc)
        ac = a.copy()
        m = -0.5
        M = 0.6
        a2 = np.clip(a, m, M, out=a)
        self.clip(a, m, M, ac)
        assert_array_strict_equal(a2, ac)
        assert_(a2 is a)

    def test_clip_nan(self):
        d = np.arange(7.)
        with assert_warns(DeprecationWarning):
            assert_equal(d.clip(min=np.nan), d)
        with assert_warns(DeprecationWarning):
            assert_equal(d.clip(max=np.nan), d)
        with assert_warns(DeprecationWarning):
            assert_equal(d.clip(min=np.nan, max=np.nan), d)
        with assert_warns(DeprecationWarning):
            assert_equal(d.clip(min=-2, max=np.nan), d)
        with assert_warns(DeprecationWarning):
            assert_equal(d.clip(min=np.nan, max=10), d)

    def test_object_clip(self):
        a = np.arange(10, dtype=object)
        actual = np.clip(a, 1, 5)
        expected = np.array([1, 1, 2, 3, 4, 5, 5, 5, 5, 5])
        assert actual.tolist() == expected.tolist()

    def test_clip_all_none(self):
        a = np.arange(10, dtype=object)
        with assert_raises_regex(ValueError, 'max or min'):
            np.clip(a, None, None)

    def test_clip_invalid_casting(self):
        a = np.arange(10, dtype=object)
        with assert_raises_regex(ValueError,
                                 'casting must be one of'):
            self.fastclip(a, 1, 8, casting="garbage")

    @pytest.mark.parametrize("amin, amax", [
        # two scalars
        (1, 0),
        # mix scalar and array
        (1, np.zeros(10)),
        # two arrays
        (np.ones(10), np.zeros(10)),
        ])
    def test_clip_value_min_max_flip(self, amin, amax):
        a = np.arange(10, dtype=np.int64)
        # requirement from ufunc_docstrings.py
        expected = np.minimum(np.maximum(a, amin), amax)
        actual = np.clip(a, amin, amax)
        assert_equal(actual, expected)

    @pytest.mark.parametrize("arr, amin, amax, exp", [
        # for a bug in npy_ObjectClip, based on a
        # case produced by hypothesis
        (np.zeros(10, dtype=np.int64),
         0,
         -2**64+1,
         np.full(10, -2**64+1, dtype=object)),
        # for bugs in NPY_TIMEDELTA_MAX, based on a case
        # produced by hypothesis
        (np.zeros(10, dtype='m8') - 1,
         0,
         0,
         np.zeros(10, dtype='m8')),
    ])
    def test_clip_problem_cases(self, arr, amin, amax, exp):
        actual = np.clip(arr, amin, amax)
        assert_equal(actual, exp)

    @pytest.mark.xfail(reason="no scalar nan propagation yet",
                       raises=AssertionError,
                       strict=True)
    @pytest.mark.parametrize("arr, amin, amax", [
        # problematic scalar nan case from hypothesis
        (np.zeros(10, dtype=np.int64),
         np.array(np.nan),
         np.zeros(10, dtype=np.int32)),
    ])
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_clip_scalar_nan_propagation(self, arr, amin, amax):
        # enforcement of scalar nan propagation for comparisons
        # called through clip()
        expected = np.minimum(np.maximum(arr, amin), amax)
        actual = np.clip(arr, amin, amax)
        assert_equal(actual, expected)

    @pytest.mark.xfail(reason="propagation doesn't match spec")
    @pytest.mark.parametrize("arr, amin, amax", [
        (np.array([1] * 10, dtype='m8'),
         np.timedelta64('NaT'),
         np.zeros(10, dtype=np.int32)),
    ])
    @pytest.mark.filterwarnings("ignore::DeprecationWarning")
    def test_NaT_propagation(self, arr, amin, amax):
        # NOTE: the expected function spec doesn't
        # propagate NaT, but clip() now does
        expected = np.minimum(np.maximum(arr, amin), amax)
        actual = np.clip(arr, amin, amax)
        assert_equal(actual, expected)

    @given(
        data=st.data(),
        arr=hynp.arrays(
            dtype=hynp.integer_dtypes() | hynp.floating_dtypes(),
            shape=hynp.array_shapes()
        )
    )
    def test_clip_property(self, data, arr):
        """A property-based test using Hypothesis.

        This aims for maximum generality: it could in principle generate *any*
        valid inputs to np.clip, and in practice generates much more varied
        inputs than human testers come up with.

        Because many of the inputs have tricky dependencies - compatible dtypes
        and mutually-broadcastable shapes - we use `st.data()` strategy draw
        values *inside* the test function, from strategies we construct based
        on previous values.  An alternative would be to define a custom strategy
        with `@st.composite`, but until we have duplicated code inline is fine.

        That accounts for most of the function; the actual test is just three
        lines to calculate and compare actual vs expected results!
        """
        numeric_dtypes = hynp.integer_dtypes() | hynp.floating_dtypes()
        # Generate shapes for the bounds which can be broadcast with each other
        # and with the base shape.  Below, we might decide to use scalar bounds,
        # but it's clearer to generate these shapes unconditionally in advance.
        in_shapes, result_shape = data.draw(
            hynp.mutually_broadcastable_shapes(
                num_shapes=2, base_shape=arr.shape
            )
        )
        # Scalar `nan` is deprecated due to the differing behaviour it shows.
        s = numeric_dtypes.flatmap(
            lambda x: hynp.from_dtype(x, allow_nan=False))
        amin = data.draw(s | hynp.arrays(dtype=numeric_dtypes,
            shape=in_shapes[0], elements={"allow_nan": False}))
        amax = data.draw(s | hynp.arrays(dtype=numeric_dtypes,
            shape=in_shapes[1], elements={"allow_nan": False}))

        # Then calculate our result and expected result and check that they're
        # equal!  See gh-12519 and gh-19457 for discussion deciding on this
        # property and the result_type argument.
        result = np.clip(arr, amin, amax)
        t = np.result_type(arr, amin, amax)
        expected = np.minimum(amax, np.maximum(arr, amin, dtype=t), dtype=t)
        assert result.dtype == t
        assert_array_equal(result, expected)


class TestAllclose:
    rtol = 1e-5
    atol = 1e-8

    def setup_method(self):
        self.olderr = np.seterr(invalid='ignore')

    def teardown_method(self):
        np.seterr(**self.olderr)

    def tst_allclose(self, x, y):
        assert_(np.allclose(x, y), "%s and %s not close" % (x, y))

    def tst_not_allclose(self, x, y):
        assert_(not np.allclose(x, y), "%s and %s shouldn't be close" % (x, y))

    def test_ip_allclose(self):
        # Parametric test factory.
        arr = np.array([100, 1000])
        aran = np.arange(125).reshape((5, 5, 5))

        atol = self.atol
        rtol = self.rtol

        data = [([1, 0], [1, 0]),
                ([atol], [0]),
                ([1], [1+rtol+atol]),
                (arr, arr + arr*rtol),
                (arr, arr + arr*rtol + atol*2),
                (aran, aran + aran*rtol),
                (np.inf, np.inf),
                (np.inf, [np.inf])]

        for (x, y) in data:
            self.tst_allclose(x, y)

    def test_ip_not_allclose(self):
        # Parametric test factory.
        aran = np.arange(125).reshape((5, 5, 5))

        atol = self.atol
        rtol = self.rtol

        data = [([np.inf, 0], [1, np.inf]),
                ([np.inf, 0], [1, 0]),
                ([np.inf, np.inf], [1, np.inf]),
                ([np.inf, np.inf], [1, 0]),
                ([-np.inf, 0], [np.inf, 0]),
                ([np.nan, 0], [np.nan, 0]),
                ([atol*2], [0]),
                ([1], [1+rtol+atol*2]),
                (aran, aran + aran*atol + atol*2),
                (np.array([np.inf, 1]), np.array([0, np.inf]))]

        for (x, y) in data:
            self.tst_not_allclose(x, y)

    def test_no_parameter_modification(self):
        x = np.array([np.inf, 1])
        y = np.array([0, np.inf])
        np.allclose(x, y)
        assert_array_equal(x, np.array([np.inf, 1]))
        assert_array_equal(y, np.array([0, np.inf]))

    def test_min_int(self):
        # Could make problems because of abs(min_int) == min_int
        min_int = np.iinfo(np.int_).min
        a = np.array([min_int], dtype=np.int_)
        assert_(np.allclose(a, a))

    def test_equalnan(self):
        x = np.array([1.0, np.nan])
        assert_(np.allclose(x, x, equal_nan=True))

    def test_return_class_is_ndarray(self):
        # Issue gh-6475
        # Check that allclose does not preserve subtypes
        class Foo(np.ndarray):
            def __new__(cls, *args, **kwargs):
                return np.array(*args, **kwargs).view(cls)

        a = Foo([1])
        assert_(type(np.allclose(a, a)) is bool)


class TestIsclose:
    rtol = 1e-5
    atol = 1e-8

    def _setup(self):
        atol = self.atol
        rtol = self.rtol
        arr = np.array([100, 1000])
        aran = np.arange(125).reshape((5, 5, 5))

        self.all_close_tests = [
                ([1, 0], [1, 0]),
                ([atol], [0]),
                ([1], [1 + rtol + atol]),
                (arr, arr + arr*rtol),
                (arr, arr + arr*rtol + atol),
                (aran, aran + aran*rtol),
                (np.inf, np.inf),
                (np.inf, [np.inf]),
                ([np.inf, -np.inf], [np.inf, -np.inf]),
                ]
        self.none_close_tests = [
                ([np.inf, 0], [1, np.inf]),
                ([np.inf, -np.inf], [1, 0]),
                ([np.inf, np.inf], [1, -np.inf]),
                ([np.inf, np.inf], [1, 0]),
                ([np.nan, 0], [np.nan, -np.inf]),
                ([atol*2], [0]),
                ([1], [1 + rtol + atol*2]),
                (aran, aran + rtol*1.1*aran + atol*1.1),
                (np.array([np.inf, 1]), np.array([0, np.inf])),
                ]
        self.some_close_tests = [
                ([np.inf, 0], [np.inf, atol*2]),
                ([atol, 1, 1e6*(1 + 2*rtol) + atol], [0, np.nan, 1e6]),
                (np.arange(3), [0, 1, 2.1]),
                (np.nan, [np.nan, np.nan, np.nan]),
                ([0], [atol, np.inf, -np.inf, np.nan]),
                (0, [atol, np.inf, -np.inf, np.nan]),
                ]
        self.some_close_results = [
                [True, False],
                [True, False, False],
                [True, True, False],
                [False, False, False],
                [True, False, False, False],
                [True, False, False, False],
                ]

    def test_ip_isclose(self):
        self._setup()
        tests = self.some_close_tests
        results = self.some_close_results
        for (x, y), result in zip(tests, results):
            assert_array_equal(np.isclose(x, y), result)

    def tst_all_isclose(self, x, y):
        assert_(np.all(np.isclose(x, y)), "%s and %s not close" % (x, y))

    def tst_none_isclose(self, x, y):
        msg = "%s and %s shouldn't be close"
        assert_(not np.any(np.isclose(x, y)), msg % (x, y))

    def tst_isclose_allclose(self, x, y):
        msg = "isclose.all() and allclose aren't same for %s and %s"
        msg2 = "isclose and allclose aren't same for %s and %s"
        if np.isscalar(x) and np.isscalar(y):
            assert_(np.isclose(x, y) == np.allclose(x, y), msg=msg2 % (x, y))
        else:
            assert_array_equal(np.isclose(x, y).all(), np.allclose(x, y), msg % (x, y))

    def test_ip_all_isclose(self):
        self._setup()
        for (x, y) in self.all_close_tests:
            self.tst_all_isclose(x, y)

    def test_ip_none_isclose(self):
        self._setup()
        for (x, y) in self.none_close_tests:
            self.tst_none_isclose(x, y)

    def test_ip_isclose_allclose(self):
        self._setup()
        tests = (self.all_close_tests + self.none_close_tests +
                 self.some_close_tests)
        for (x, y) in tests:
            self.tst_isclose_allclose(x, y)

    def test_equal_nan(self):
        assert_array_equal(np.isclose(np.nan, np.nan, equal_nan=True), [True])
        arr = np.array([1.0, np.nan])
        assert_array_equal(np.isclose(arr, arr, equal_nan=True), [True, True])

    def test_masked_arrays(self):
        # Make sure to test the output type when arguments are interchanged.

        x = np.ma.masked_where([True, True, False], np.arange(3))
        assert_(type(x) is type(np.isclose(2, x)))
        assert_(type(x) is type(np.isclose(x, 2)))

        x = np.ma.masked_where([True, True, False], [np.nan, np.inf, np.nan])
        assert_(type(x) is type(np.isclose(np.inf, x)))
        assert_(type(x) is type(np.isclose(x, np.inf)))

        x = np.ma.masked_where([True, True, False], [np.nan, np.nan, np.nan])
        y = np.isclose(np.nan, x, equal_nan=True)
        assert_(type(x) is type(y))
        # Ensure that the mask isn't modified...
        assert_array_equal([True, True, False], y.mask)
        y = np.isclose(x, np.nan, equal_nan=True)
        assert_(type(x) is type(y))
        # Ensure that the mask isn't modified...
        assert_array_equal([True, True, False], y.mask)

        x = np.ma.masked_where([True, True, False], [np.nan, np.nan, np.nan])
        y = np.isclose(x, x, equal_nan=True)
        assert_(type(x) is type(y))
        # Ensure that the mask isn't modified...
        assert_array_equal([True, True, False], y.mask)

    def test_scalar_return(self):
        assert_(np.isscalar(np.isclose(1, 1)))

    def test_no_parameter_modification(self):
        x = np.array([np.inf, 1])
        y = np.array([0, np.inf])
        np.isclose(x, y)
        assert_array_equal(x, np.array([np.inf, 1]))
        assert_array_equal(y, np.array([0, np.inf]))

    def test_non_finite_scalar(self):
        # GH7014, when two scalars are compared the output should also be a
        # scalar
        assert_(np.isclose(np.inf, -np.inf) is np.False_)
        assert_(np.isclose(0, np.inf) is np.False_)
        assert_(type(np.isclose(0, np.inf)) is np.bool_)

    def test_timedelta(self):
        # Allclose currently works for timedelta64 as long as `atol` is
        # an integer or also a timedelta64
        a = np.array([[1, 2, 3, "NaT"]], dtype="m8[ns]")
        assert np.isclose(a, a, atol=0, equal_nan=True).all()
        assert np.isclose(a, a, atol=np.timedelta64(1, "ns"), equal_nan=True).all()
        assert np.allclose(a, a, atol=0, equal_nan=True)
        assert np.allclose(a, a, atol=np.timedelta64(1, "ns"), equal_nan=True)


class TestStdVar:
    def setup_method(self):
        self.A = np.array([1, -1, 1, -1])
        self.real_var = 1

    def test_basic(self):
        assert_almost_equal(np.var(self.A), self.real_var)
        assert_almost_equal(np.std(self.A)**2, self.real_var)

    def test_scalars(self):
        assert_equal(np.var(1), 0)
        assert_equal(np.std(1), 0)

    def test_ddof1(self):
        assert_almost_equal(np.var(self.A, ddof=1),
                            self.real_var * len(self.A) / (len(self.A) - 1))
        assert_almost_equal(np.std(self.A, ddof=1)**2,
                            self.real_var*len(self.A) / (len(self.A) - 1))

    def test_ddof2(self):
        assert_almost_equal(np.var(self.A, ddof=2),
                            self.real_var * len(self.A) / (len(self.A) - 2))
        assert_almost_equal(np.std(self.A, ddof=2)**2,
                            self.real_var * len(self.A) / (len(self.A) - 2))

    def test_out_scalar(self):
        d = np.arange(10)
        out = np.array(0.)
        r = np.std(d, out=out)
        assert_(r is out)
        assert_array_equal(r, out)
        r = np.var(d, out=out)
        assert_(r is out)
        assert_array_equal(r, out)
        r = np.mean(d, out=out)
        assert_(r is out)
        assert_array_equal(r, out)


class TestStdVarComplex:
    def test_basic(self):
        A = np.array([1, 1.j, -1, -1.j])
        real_var = 1
        assert_almost_equal(np.var(A), real_var)
        assert_almost_equal(np.std(A)**2, real_var)

    def test_scalars(self):
        assert_equal(np.var(1j), 0)
        assert_equal(np.std(1j), 0)


class TestCreationFuncs:
    # Test ones, zeros, empty and full.

    def setup_method(self):
        dtypes = {np.dtype(tp) for tp in itertools.chain(*np.sctypes.values())}
        # void, bytes, str
        variable_sized = {tp for tp in dtypes if tp.str.endswith('0')}
        self.dtypes = sorted(dtypes - variable_sized |
                             {np.dtype(tp.str.replace("0", str(i)))
                              for tp in variable_sized for i in range(1, 10)},
                             key=lambda dtype: dtype.str)
        self.orders = {'C': 'c_contiguous', 'F': 'f_contiguous'}
        self.ndims = 10

    def check_function(self, func, fill_value=None):
        par = ((0, 1, 2),
               range(self.ndims),
               self.orders,
               self.dtypes)
        fill_kwarg = {}
        if fill_value is not None:
            fill_kwarg = {'fill_value': fill_value}

        for size, ndims, order, dtype in itertools.product(*par):
            shape = ndims * [size]

            # do not fill void type
            if fill_kwarg and dtype.str.startswith('|V'):
                continue

            arr = func(shape, order=order, dtype=dtype,
                       **fill_kwarg)

            assert_equal(arr.dtype, dtype)
            assert_(getattr(arr.flags, self.orders[order]))

            if fill_value is not None:
                if dtype.str.startswith('|S'):
                    val = str(fill_value)
                else:
                    val = fill_value
                assert_equal(arr, dtype.type(val))

    def test_zeros(self):
        self.check_function(np.zeros)

    def test_ones(self):
        self.check_function(np.ones)

    def test_empty(self):
        self.check_function(np.empty)

    def test_full(self):
        self.check_function(np.full, 0)
        self.check_function(np.full, 1)

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_for_reference_leak(self):
        # Make sure we have an object for reference
        dim = 1
        beg = sys.getrefcount(dim)
        np.zeros([dim]*10)
        assert_(sys.getrefcount(dim) == beg)
        np.ones([dim]*10)
        assert_(sys.getrefcount(dim) == beg)
        np.empty([dim]*10)
        assert_(sys.getrefcount(dim) == beg)
        np.full([dim]*10, 0)
        assert_(sys.getrefcount(dim) == beg)


class TestLikeFuncs:
    '''Test ones_like, zeros_like, empty_like and full_like'''

    def setup_method(self):
        self.data = [
                # Array scalars
                (np.array(3.), None),
                (np.array(3), 'f8'),
                # 1D arrays
                (np.arange(6, dtype='f4'), None),
                (np.arange(6), 'c16'),
                # 2D C-layout arrays
                (np.arange(6).reshape(2, 3), None),
                (np.arange(6).reshape(3, 2), 'i1'),
                # 2D F-layout arrays
                (np.arange(6).reshape((2, 3), order='F'), None),
                (np.arange(6).reshape((3, 2), order='F'), 'i1'),
                # 3D C-layout arrays
                (np.arange(24).reshape(2, 3, 4), None),
                (np.arange(24).reshape(4, 3, 2), 'f4'),
                # 3D F-layout arrays
                (np.arange(24).reshape((2, 3, 4), order='F'), None),
                (np.arange(24).reshape((4, 3, 2), order='F'), 'f4'),
                # 3D non-C/F-layout arrays
                (np.arange(24).reshape(2, 3, 4).swapaxes(0, 1), None),
                (np.arange(24).reshape(4, 3, 2).swapaxes(0, 1), '?'),
                     ]
        self.shapes = [(), (5,), (5,6,), (5,6,7,)]

    def compare_array_value(self, dz, value, fill_value):
        if value is not None:
            if fill_value:
                # Conversion is close to what np.full_like uses
                # but we  may want to convert directly in the future
                # which may result in errors (where this does not).
                z = np.array(value).astype(dz.dtype)
                assert_(np.all(dz == z))
            else:
                assert_(np.all(dz == value))

    def check_like_function(self, like_function, value, fill_value=False):
        if fill_value:
            fill_kwarg = {'fill_value': value}
        else:
            fill_kwarg = {}
        for d, dtype in self.data:
            # default (K) order, dtype
            dz = like_function(d, dtype=dtype, **fill_kwarg)
            assert_equal(dz.shape, d.shape)
            assert_equal(np.array(dz.strides)*d.dtype.itemsize,
                         np.array(d.strides)*dz.dtype.itemsize)
            assert_equal(d.flags.c_contiguous, dz.flags.c_contiguous)
            assert_equal(d.flags.f_contiguous, dz.flags.f_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            self.compare_array_value(dz, value, fill_value)

            # C order, default dtype
            dz = like_function(d, order='C', dtype=dtype, **fill_kwarg)
            assert_equal(dz.shape, d.shape)
            assert_(dz.flags.c_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            self.compare_array_value(dz, value, fill_value)

            # F order, default dtype
            dz = like_function(d, order='F', dtype=dtype, **fill_kwarg)
            assert_equal(dz.shape, d.shape)
            assert_(dz.flags.f_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            self.compare_array_value(dz, value, fill_value)

            # A order
            dz = like_function(d, order='A', dtype=dtype, **fill_kwarg)
            assert_equal(dz.shape, d.shape)
            if d.flags.f_contiguous:
                assert_(dz.flags.f_contiguous)
            else:
                assert_(dz.flags.c_contiguous)
            if dtype is None:
                assert_equal(dz.dtype, d.dtype)
            else:
                assert_equal(dz.dtype, np.dtype(dtype))
            self.compare_array_value(dz, value, fill_value)

            # Test the 'shape' parameter
            for s in self.shapes:
                for o in 'CFA':
                    sz = like_function(d, dtype=dtype, shape=s, order=o,
                                       **fill_kwarg)
                    assert_equal(sz.shape, s)
                    if dtype is None:
                        assert_equal(sz.dtype, d.dtype)
                    else:
                        assert_equal(sz.dtype, np.dtype(dtype))
                    if o == 'C' or (o == 'A' and d.flags.c_contiguous):
                        assert_(sz.flags.c_contiguous)
                    elif o == 'F' or (o == 'A' and d.flags.f_contiguous):
                        assert_(sz.flags.f_contiguous)
                    self.compare_array_value(sz, value, fill_value)

                if (d.ndim != len(s)):
                    assert_equal(np.argsort(like_function(d, dtype=dtype,
                                                          shape=s, order='K',
                                                          **fill_kwarg).strides),
                                 np.argsort(np.empty(s, dtype=dtype,
                                                     order='C').strides))
                else:
                    assert_equal(np.argsort(like_function(d, dtype=dtype,
                                                          shape=s, order='K',
                                                          **fill_kwarg).strides),
                                 np.argsort(d.strides))

        # Test the 'subok' parameter
        class MyNDArray(np.ndarray):
            pass

        a = np.array([[1, 2], [3, 4]]).view(MyNDArray)

        b = like_function(a, **fill_kwarg)
        assert_(type(b) is MyNDArray)

        b = like_function(a, subok=False, **fill_kwarg)
        assert_(type(b) is not MyNDArray)

    def test_ones_like(self):
        self.check_like_function(np.ones_like, 1)

    def test_zeros_like(self):
        self.check_like_function(np.zeros_like, 0)

    def test_empty_like(self):
        self.check_like_function(np.empty_like, None)

    def test_filled_like(self):
        self.check_like_function(np.full_like, 0, True)
        self.check_like_function(np.full_like, 1, True)
        self.check_like_function(np.full_like, 1000, True)
        self.check_like_function(np.full_like, 123.456, True)
        # Inf to integer casts cause invalid-value errors: ignore them.
        with np.errstate(invalid="ignore"):
            self.check_like_function(np.full_like, np.inf, True)

    @pytest.mark.parametrize('likefunc', [np.empty_like, np.full_like,
                                          np.zeros_like, np.ones_like])
    @pytest.mark.parametrize('dtype', [str, bytes])
    def test_dtype_str_bytes(self, likefunc, dtype):
        # Regression test for gh-19860
        a = np.arange(16).reshape(2, 8)
        b = a[:, ::2]  # Ensure b is not contiguous.
        kwargs = {'fill_value': ''} if likefunc == np.full_like else {}
        result = likefunc(b, dtype=dtype, **kwargs)
        if dtype == str:
            assert result.strides == (16, 4)
        else:
            # dtype is bytes
            assert result.strides == (4, 1)


class TestCorrelate:
    def _setup(self, dt):
        self.x = np.array([1, 2, 3, 4, 5], dtype=dt)
        self.xs = np.arange(1, 20)[::3]
        self.y = np.array([-1, -2, -3], dtype=dt)
        self.z1 = np.array([-3., -8., -14., -20., -26., -14., -5.], dtype=dt)
        self.z1_4 = np.array([-2., -5., -8., -11., -14., -5.], dtype=dt)
        self.z1r = np.array([-15., -22., -22., -16., -10., -4., -1.], dtype=dt)
        self.z2 = np.array([-5., -14., -26., -20., -14., -8., -3.], dtype=dt)
        self.z2r = np.array([-1., -4., -10., -16., -22., -22., -15.], dtype=dt)
        self.zs = np.array([-3., -14., -30., -48., -66., -84.,
                           -102., -54., -19.], dtype=dt)

    def test_float(self):
        self._setup(float)
        z = np.correlate(self.x, self.y, 'full')
        assert_array_almost_equal(z, self.z1)
        z = np.correlate(self.x, self.y[:-1], 'full')
        assert_array_almost_equal(z, self.z1_4)
        z = np.correlate(self.y, self.x, 'full')
        assert_array_almost_equal(z, self.z2)
        z = np.correlate(self.x[::-1], self.y, 'full')
        assert_array_almost_equal(z, self.z1r)
        z = np.correlate(self.y, self.x[::-1], 'full')
        assert_array_almost_equal(z, self.z2r)
        z = np.correlate(self.xs, self.y, 'full')
        assert_array_almost_equal(z, self.zs)

    def test_object(self):
        self._setup(Decimal)
        z = np.correlate(self.x, self.y, 'full')
        assert_array_almost_equal(z, self.z1)
        z = np.correlate(self.y, self.x, 'full')
        assert_array_almost_equal(z, self.z2)

    def test_no_overwrite(self):
        d = np.ones(100)
        k = np.ones(3)
        np.correlate(d, k)
        assert_array_equal(d, np.ones(100))
        assert_array_equal(k, np.ones(3))

    def test_complex(self):
        x = np.array([1, 2, 3, 4+1j], dtype=complex)
        y = np.array([-1, -2j, 3+1j], dtype=complex)
        r_z = np.array([3-1j, 6, 8+1j, 11+5j, -5+8j, -4-1j], dtype=complex)
        r_z = r_z[::-1].conjugate()
        z = np.correlate(y, x, mode='full')
        assert_array_almost_equal(z, r_z)

    def test_zero_size(self):
        with pytest.raises(ValueError):
            np.correlate(np.array([]), np.ones(1000), mode='full')
        with pytest.raises(ValueError):
            np.correlate(np.ones(1000), np.array([]), mode='full')

    def test_mode(self):
        d = np.ones(100)
        k = np.ones(3)
        default_mode = np.correlate(d, k, mode='valid')
        with assert_warns(DeprecationWarning):
            valid_mode = np.correlate(d, k, mode='v')
        assert_array_equal(valid_mode, default_mode)
        # integer mode
        with assert_raises(ValueError):
            np.correlate(d, k, mode=-1)
        assert_array_equal(np.correlate(d, k, mode=0), valid_mode)
        # illegal arguments
        with assert_raises(TypeError):
            np.correlate(d, k, mode=None)


class TestConvolve:
    def test_object(self):
        d = [1.] * 100
        k = [1.] * 3
        assert_array_almost_equal(np.convolve(d, k)[2:-2], np.full(98, 3))

    def test_no_overwrite(self):
        d = np.ones(100)
        k = np.ones(3)
        np.convolve(d, k)
        assert_array_equal(d, np.ones(100))
        assert_array_equal(k, np.ones(3))

    def test_mode(self):
        d = np.ones(100)
        k = np.ones(3)
        default_mode = np.convolve(d, k, mode='full')
        with assert_warns(DeprecationWarning):
            full_mode = np.convolve(d, k, mode='f')
        assert_array_equal(full_mode, default_mode)
        # integer mode
        with assert_raises(ValueError):
            np.convolve(d, k, mode=-1)
        assert_array_equal(np.convolve(d, k, mode=2), full_mode)
        # illegal arguments
        with assert_raises(TypeError):
            np.convolve(d, k, mode=None)


class TestArgwhere:

    @pytest.mark.parametrize('nd', [0, 1, 2])
    def test_nd(self, nd):
        # get an nd array with multiple elements in every dimension
        x = np.empty((2,)*nd, bool)

        # none
        x[...] = False
        assert_equal(np.argwhere(x).shape, (0, nd))

        # only one
        x[...] = False
        x.flat[0] = True
        assert_equal(np.argwhere(x).shape, (1, nd))

        # all but one
        x[...] = True
        x.flat[0] = False
        assert_equal(np.argwhere(x).shape, (x.size - 1, nd))

        # all
        x[...] = True
        assert_equal(np.argwhere(x).shape, (x.size, nd))

    def test_2D(self):
        x = np.arange(6).reshape((2, 3))
        assert_array_equal(np.argwhere(x > 1),
                           [[0, 2],
                            [1, 0],
                            [1, 1],
                            [1, 2]])

    def test_list(self):
        assert_equal(np.argwhere([4, 0, 2, 1, 3]), [[0], [2], [3], [4]])


class TestStringFunction:

    def test_set_string_function(self):
        a = np.array([1])
        np.set_string_function(lambda x: "FOO", repr=True)
        assert_equal(repr(a), "FOO")
        np.set_string_function(None, repr=True)
        assert_equal(repr(a), "array([1])")

        np.set_string_function(lambda x: "FOO", repr=False)
        assert_equal(str(a), "FOO")
        np.set_string_function(None, repr=False)
        assert_equal(str(a), "[1]")


class TestRoll:
    def test_roll1d(self):
        x = np.arange(10)
        xr = np.roll(x, 2)
        assert_equal(xr, np.array([8, 9, 0, 1, 2, 3, 4, 5, 6, 7]))

    def test_roll2d(self):
        x2 = np.reshape(np.arange(10), (2, 5))
        x2r = np.roll(x2, 1)
        assert_equal(x2r, np.array([[9, 0, 1, 2, 3], [4, 5, 6, 7, 8]]))

        x2r = np.roll(x2, 1, axis=0)
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        x2r = np.roll(x2, 1, axis=1)
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        # Roll multiple axes at once.
        x2r = np.roll(x2, 1, axis=(0, 1))
        assert_equal(x2r, np.array([[9, 5, 6, 7, 8], [4, 0, 1, 2, 3]]))

        x2r = np.roll(x2, (1, 0), axis=(0, 1))
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        x2r = np.roll(x2, (-1, 0), axis=(0, 1))
        assert_equal(x2r, np.array([[5, 6, 7, 8, 9], [0, 1, 2, 3, 4]]))

        x2r = np.roll(x2, (0, 1), axis=(0, 1))
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        x2r = np.roll(x2, (0, -1), axis=(0, 1))
        assert_equal(x2r, np.array([[1, 2, 3, 4, 0], [6, 7, 8, 9, 5]]))

        x2r = np.roll(x2, (1, 1), axis=(0, 1))
        assert_equal(x2r, np.array([[9, 5, 6, 7, 8], [4, 0, 1, 2, 3]]))

        x2r = np.roll(x2, (-1, -1), axis=(0, 1))
        assert_equal(x2r, np.array([[6, 7, 8, 9, 5], [1, 2, 3, 4, 0]]))

        # Roll the same axis multiple times.
        x2r = np.roll(x2, 1, axis=(0, 0))
        assert_equal(x2r, np.array([[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]]))

        x2r = np.roll(x2, 1, axis=(1, 1))
        assert_equal(x2r, np.array([[3, 4, 0, 1, 2], [8, 9, 5, 6, 7]]))

        # Roll more than one turn in either direction.
        x2r = np.roll(x2, 6, axis=1)
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

        x2r = np.roll(x2, -4, axis=1)
        assert_equal(x2r, np.array([[4, 0, 1, 2, 3], [9, 5, 6, 7, 8]]))

    def test_roll_empty(self):
        x = np.array([])
        assert_equal(np.roll(x, 1), np.array([]))


class TestRollaxis:

    # expected shape indexed by (axis, start) for array of
    # shape (1, 2, 3, 4)
    tgtshape = {(0, 0): (1, 2, 3, 4), (0, 1): (1, 2, 3, 4),
                (0, 2): (2, 1, 3, 4), (0, 3): (2, 3, 1, 4),
                (0, 4): (2, 3, 4, 1),
                (1, 0): (2, 1, 3, 4), (1, 1): (1, 2, 3, 4),
                (1, 2): (1, 2, 3, 4), (1, 3): (1, 3, 2, 4),
                (1, 4): (1, 3, 4, 2),
                (2, 0): (3, 1, 2, 4), (2, 1): (1, 3, 2, 4),
                (2, 2): (1, 2, 3, 4), (2, 3): (1, 2, 3, 4),
                (2, 4): (1, 2, 4, 3),
                (3, 0): (4, 1, 2, 3), (3, 1): (1, 4, 2, 3),
                (3, 2): (1, 2, 4, 3), (3, 3): (1, 2, 3, 4),
                (3, 4): (1, 2, 3, 4)}

    def test_exceptions(self):
        a = np.arange(1*2*3*4).reshape(1, 2, 3, 4)
        assert_raises(np.AxisError, np.rollaxis, a, -5, 0)
        assert_raises(np.AxisError, np.rollaxis, a, 0, -5)
        assert_raises(np.AxisError, np.rollaxis, a, 4, 0)
        assert_raises(np.AxisError, np.rollaxis, a, 0, 5)

    def test_results(self):
        a = np.arange(1*2*3*4).reshape(1, 2, 3, 4).copy()
        aind = np.indices(a.shape)
        assert_(a.flags['OWNDATA'])
        for (i, j) in self.tgtshape:
            # positive axis, positive start
            res = np.rollaxis(a, axis=i, start=j)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[(i, j)], str((i,j)))
            assert_(not res.flags['OWNDATA'])

            # negative axis, positive start
            ip = i + 1
            res = np.rollaxis(a, axis=-ip, start=j)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[(4 - ip, j)])
            assert_(not res.flags['OWNDATA'])

            # positive axis, negative start
            jp = j + 1 if j < 4 else j
            res = np.rollaxis(a, axis=i, start=-jp)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[(i, 4 - jp)])
            assert_(not res.flags['OWNDATA'])

            # negative axis, negative start
            ip = i + 1
            jp = j + 1 if j < 4 else j
            res = np.rollaxis(a, axis=-ip, start=-jp)
            i0, i1, i2, i3 = aind[np.array(res.shape) - 1]
            assert_(np.all(res[i0, i1, i2, i3] == a))
            assert_(res.shape == self.tgtshape[(4 - ip, 4 - jp)])
            assert_(not res.flags['OWNDATA'])


class TestMoveaxis:
    def test_move_to_end(self):
        x = np.random.randn(5, 6, 7)
        for source, expected in [(0, (6, 7, 5)),
                                 (1, (5, 7, 6)),
                                 (2, (5, 6, 7)),
                                 (-1, (5, 6, 7))]:
            actual = np.moveaxis(x, source, -1).shape
            assert_(actual, expected)

    def test_move_new_position(self):
        x = np.random.randn(1, 2, 3, 4)
        for source, destination, expected in [
                (0, 1, (2, 1, 3, 4)),
                (1, 2, (1, 3, 2, 4)),
                (1, -1, (1, 3, 4, 2)),
                ]:
            actual = np.moveaxis(x, source, destination).shape
            assert_(actual, expected)

    def test_preserve_order(self):
        x = np.zeros((1, 2, 3, 4))
        for source, destination in [
                (0, 0),
                (3, -1),
                (-1, 3),
                ([0, -1], [0, -1]),
                ([2, 0], [2, 0]),
                (range(4), range(4)),
                ]:
            actual = np.moveaxis(x, source, destination).shape
            assert_(actual, (1, 2, 3, 4))

    def test_move_multiples(self):
        x = np.zeros((0, 1, 2, 3))
        for source, destination, expected in [
                ([0, 1], [2, 3], (2, 3, 0, 1)),
                ([2, 3], [0, 1], (2, 3, 0, 1)),
                ([0, 1, 2], [2, 3, 0], (2, 3, 0, 1)),
                ([3, 0], [1, 0], (0, 3, 1, 2)),
                ([0, 3], [0, 1], (0, 3, 1, 2)),
                ]:
            actual = np.moveaxis(x, source, destination).shape
            assert_(actual, expected)

    def test_errors(self):
        x = np.random.randn(1, 2, 3)
        assert_raises_regex(np.AxisError, 'source.*out of bounds',
                            np.moveaxis, x, 3, 0)
        assert_raises_regex(np.AxisError, 'source.*out of bounds',
                            np.moveaxis, x, -4, 0)
        assert_raises_regex(np.AxisError, 'destination.*out of bounds',
                            np.moveaxis, x, 0, 5)
        assert_raises_regex(ValueError, 'repeated axis in `source`',
                            np.moveaxis, x, [0, 0], [0, 1])
        assert_raises_regex(ValueError, 'repeated axis in `destination`',
                            np.moveaxis, x, [0, 1], [1, 1])
        assert_raises_regex(ValueError, 'must have the same number',
                            np.moveaxis, x, 0, [0, 1])
        assert_raises_regex(ValueError, 'must have the same number',
                            np.moveaxis, x, [0, 1], [0])

    def test_array_likes(self):
        x = np.ma.zeros((1, 2, 3))
        result = np.moveaxis(x, 0, 0)
        assert_(x.shape, result.shape)
        assert_(isinstance(result, np.ma.MaskedArray))

        x = [1, 2, 3]
        result = np.moveaxis(x, 0, 0)
        assert_(x, list(result))
        assert_(isinstance(result, np.ndarray))


class TestCross:
    def test_2x2(self):
        u = [1, 2]
        v = [3, 4]
        z = -2
        cp = np.cross(u, v)
        assert_equal(cp, z)
        cp = np.cross(v, u)
        assert_equal(cp, -z)

    def test_2x3(self):
        u = [1, 2]
        v = [3, 4, 5]
        z = np.array([10, -5, -2])
        cp = np.cross(u, v)
        assert_equal(cp, z)
        cp = np.cross(v, u)
        assert_equal(cp, -z)

    def test_3x3(self):
        u = [1, 2, 3]
        v = [4, 5, 6]
        z = np.array([-3, 6, -3])
        cp = np.cross(u, v)
        assert_equal(cp, z)
        cp = np.cross(v, u)
        assert_equal(cp, -z)

    def test_broadcasting(self):
        # Ticket #2624 (Trac #2032)
        u = np.tile([1, 2], (11, 1))
        v = np.tile([3, 4], (11, 1))
        z = -2
        assert_equal(np.cross(u, v), z)
        assert_equal(np.cross(v, u), -z)
        assert_equal(np.cross(u, u), 0)

        u = np.tile([1, 2], (11, 1)).T
        v = np.tile([3, 4, 5], (11, 1))
        z = np.tile([10, -5, -2], (11, 1))
        assert_equal(np.cross(u, v, axisa=0), z)
        assert_equal(np.cross(v, u.T), -z)
        assert_equal(np.cross(v, v), 0)

        u = np.tile([1, 2, 3], (11, 1)).T
        v = np.tile([3, 4], (11, 1)).T
        z = np.tile([-12, 9, -2], (11, 1))
        assert_equal(np.cross(u, v, axisa=0, axisb=0), z)
        assert_equal(np.cross(v.T, u.T), -z)
        assert_equal(np.cross(u.T, u.T), 0)

        u = np.tile([1, 2, 3], (5, 1))
        v = np.tile([4, 5, 6], (5, 1)).T
        z = np.tile([-3, 6, -3], (5, 1))
        assert_equal(np.cross(u, v, axisb=0), z)
        assert_equal(np.cross(v.T, u), -z)
        assert_equal(np.cross(u, u), 0)

    def test_broadcasting_shapes(self):
        u = np.ones((2, 1, 3))
        v = np.ones((5, 3))
        assert_equal(np.cross(u, v).shape, (2, 5, 3))
        u = np.ones((10, 3, 5))
        v = np.ones((2, 5))
        assert_equal(np.cross(u, v, axisa=1, axisb=0).shape, (10, 5, 3))
        assert_raises(np.AxisError, np.cross, u, v, axisa=1, axisb=2)
        assert_raises(np.AxisError, np.cross, u, v, axisa=3, axisb=0)
        u = np.ones((10, 3, 5, 7))
        v = np.ones((5, 7, 2))
        assert_equal(np.cross(u, v, axisa=1, axisc=2).shape, (10, 5, 3, 7))
        assert_raises(np.AxisError, np.cross, u, v, axisa=-5, axisb=2)
        assert_raises(np.AxisError, np.cross, u, v, axisa=1, axisb=-4)
        # gh-5885
        u = np.ones((3, 4, 2))
        for axisc in range(-2, 2):
            assert_equal(np.cross(u, u, axisc=axisc).shape, (3, 4))

    def test_uint8_int32_mixed_dtypes(self):
        # regression test for gh-19138
        u = np.array([[195, 8, 9]], np.uint8)
        v = np.array([250, 166, 68], np.int32)
        z = np.array([[950, 11010, -30370]], dtype=np.int32)
        assert_equal(np.cross(v, u), z)
        assert_equal(np.cross(u, v), -z)


def test_outer_out_param():
    arr1 = np.ones((5,))
    arr2 = np.ones((2,))
    arr3 = np.linspace(-2, 2, 5)
    out1 = np.ndarray(shape=(5,5))
    out2 = np.ndarray(shape=(2, 5))
    res1 = np.outer(arr1, arr3, out1)
    assert_equal(res1, out1)
    assert_equal(np.outer(arr2, arr3, out2), out2)


class TestIndices:

    def test_simple(self):
        [x, y] = np.indices((4, 3))
        assert_array_equal(x, np.array([[0, 0, 0],
                                        [1, 1, 1],
                                        [2, 2, 2],
                                        [3, 3, 3]]))
        assert_array_equal(y, np.array([[0, 1, 2],
                                        [0, 1, 2],
                                        [0, 1, 2],
                                        [0, 1, 2]]))

    def test_single_input(self):
        [x] = np.indices((4,))
        assert_array_equal(x, np.array([0, 1, 2, 3]))

        [x] = np.indices((4,), sparse=True)
        assert_array_equal(x, np.array([0, 1, 2, 3]))

    def test_scalar_input(self):
        assert_array_equal([], np.indices(()))
        assert_array_equal([], np.indices((), sparse=True))
        assert_array_equal([[]], np.indices((0,)))
        assert_array_equal([[]], np.indices((0,), sparse=True))

    def test_sparse(self):
        [x, y] = np.indices((4,3), sparse=True)
        assert_array_equal(x, np.array([[0], [1], [2], [3]]))
        assert_array_equal(y, np.array([[0, 1, 2]]))

    @pytest.mark.parametrize("dtype", [np.int32, np.int64, np.float32, np.float64])
    @pytest.mark.parametrize("dims", [(), (0,), (4, 3)])
    def test_return_type(self, dtype, dims):
        inds = np.indices(dims, dtype=dtype)
        assert_(inds.dtype == dtype)

        for arr in np.indices(dims, dtype=dtype, sparse=True):
            assert_(arr.dtype == dtype)


class TestRequire:
    flag_names = ['C', 'C_CONTIGUOUS', 'CONTIGUOUS',
                  'F', 'F_CONTIGUOUS', 'FORTRAN',
                  'A', 'ALIGNED',
                  'W', 'WRITEABLE',
                  'O', 'OWNDATA']

    def generate_all_false(self, dtype):
        arr = np.zeros((2, 2), [('junk', 'i1'), ('a', dtype)])
        arr.setflags(write=False)
        a = arr['a']
        assert_(not a.flags['C'])
        assert_(not a.flags['F'])
        assert_(not a.flags['O'])
        assert_(not a.flags['W'])
        assert_(not a.flags['A'])
        return a

    def set_and_check_flag(self, flag, dtype, arr):
        if dtype is None:
            dtype = arr.dtype
        b = np.require(arr, dtype, [flag])
        assert_(b.flags[flag])
        assert_(b.dtype == dtype)

        # a further call to np.require ought to return the same array
        # unless OWNDATA is specified.
        c = np.require(b, None, [flag])
        if flag[0] != 'O':
            assert_(c is b)
        else:
            assert_(c.flags[flag])

    def test_require_each(self):

        id = ['f8', 'i4']
        fd = [None, 'f8', 'c16']
        for idtype, fdtype, flag in itertools.product(id, fd, self.flag_names):
            a = self.generate_all_false(idtype)
            self.set_and_check_flag(flag, fdtype,  a)

    def test_unknown_requirement(self):
        a = self.generate_all_false('f8')
        assert_raises(KeyError, np.require, a, None, 'Q')

    def test_non_array_input(self):
        a = np.require([1, 2, 3, 4], 'i4', ['C', 'A', 'O'])
        assert_(a.flags['O'])
        assert_(a.flags['C'])
        assert_(a.flags['A'])
        assert_(a.dtype == 'i4')
        assert_equal(a, [1, 2, 3, 4])

    def test_C_and_F_simul(self):
        a = self.generate_all_false('f8')
        assert_raises(ValueError, np.require, a, None, ['C', 'F'])

    def test_ensure_array(self):
        class ArraySubclass(np.ndarray):
            pass

        a = ArraySubclass((2, 2))
        b = np.require(a, None, ['E'])
        assert_(type(b) is np.ndarray)

    def test_preserve_subtype(self):
        class ArraySubclass(np.ndarray):
            pass

        for flag in self.flag_names:
            a = ArraySubclass((2, 2))
            self.set_and_check_flag(flag, None, a)


class TestBroadcast:
    def test_broadcast_in_args(self):
        # gh-5881
        arrs = [np.empty((6, 7)), np.empty((5, 6, 1)), np.empty((7,)),
                np.empty((5, 1, 7))]
        mits = [np.broadcast(*arrs),
                np.broadcast(np.broadcast(*arrs[:0]), np.broadcast(*arrs[0:])),
                np.broadcast(np.broadcast(*arrs[:1]), np.broadcast(*arrs[1:])),
                np.broadcast(np.broadcast(*arrs[:2]), np.broadcast(*arrs[2:])),
                np.broadcast(arrs[0], np.broadcast(*arrs[1:-1]), arrs[-1])]
        for mit in mits:
            assert_equal(mit.shape, (5, 6, 7))
            assert_equal(mit.ndim, 3)
            assert_equal(mit.nd, 3)
            assert_equal(mit.numiter, 4)
            for a, ia in zip(arrs, mit.iters):
                assert_(a is ia.base)

    def test_broadcast_single_arg(self):
        # gh-6899
        arrs = [np.empty((5, 6, 7))]
        mit = np.broadcast(*arrs)
        assert_equal(mit.shape, (5, 6, 7))
        assert_equal(mit.ndim, 3)
        assert_equal(mit.nd, 3)
        assert_equal(mit.numiter, 1)
        assert_(arrs[0] is mit.iters[0].base)

    def test_number_of_arguments(self):
        arr = np.empty((5,))
        for j in range(35):
            arrs = [arr] * j
            if j > 32:
                assert_raises(ValueError, np.broadcast, *arrs)
            else:
                mit = np.broadcast(*arrs)
                assert_equal(mit.numiter, j)

    def test_broadcast_error_kwargs(self):
        #gh-13455
        arrs = [np.empty((5, 6, 7))]
        mit  = np.broadcast(*arrs)
        mit2 = np.broadcast(*arrs, **{})
        assert_equal(mit.shape, mit2.shape)
        assert_equal(mit.ndim, mit2.ndim)
        assert_equal(mit.nd, mit2.nd)
        assert_equal(mit.numiter, mit2.numiter)
        assert_(mit.iters[0].base is mit2.iters[0].base)

        assert_raises(ValueError, np.broadcast, 1, **{'x': 1})

    def test_shape_mismatch_error_message(self):
        with pytest.raises(ValueError, match=r"arg 0 with shape \(1, 3\) and "
                                             r"arg 2 with shape \(2,\)"):
            np.broadcast([[1, 2, 3]], [[4], [5]], [6, 7])


class TestKeepdims:

    class sub_array(np.ndarray):
        def sum(self, axis=None, dtype=None, out=None):
            return np.ndarray.sum(self, axis, dtype, out, keepdims=True)

    def test_raise(self):
        sub_class = self.sub_array
        x = np.arange(30).view(sub_class)
        assert_raises(TypeError, np.sum, x, keepdims=True)


class TestTensordot:

    def test_zero_dimension(self):
        # Test resolution to issue #5663
        a = np.ndarray((3,0))
        b = np.ndarray((0,4))
        td = np.tensordot(a, b, (1, 0))
        assert_array_equal(td, np.dot(a, b))
        assert_array_equal(td, np.einsum('ij,jk', a, b))

    def test_zero_dimensional(self):
        # gh-12130
        arr_0d = np.array(1)
        ret = np.tensordot(arr_0d, arr_0d, ([], []))  # contracting no axes is well defined
        assert_array_equal(ret, arr_0d)
