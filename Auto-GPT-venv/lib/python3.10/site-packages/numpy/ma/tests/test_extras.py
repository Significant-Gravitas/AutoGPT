# pylint: disable-msg=W0611, W0612, W0511
"""Tests suite for MaskedArray.
Adapted from the original test_ma by Pierre Gerard-Marchant

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: test_extras.py 3473 2007-10-29 15:18:13Z jarrod.millman $

"""
import warnings
import itertools
import pytest

import numpy as np
from numpy.core.numeric import normalize_axis_tuple
from numpy.testing import (
    assert_warns, suppress_warnings
    )
from numpy.ma.testutils import (
    assert_, assert_array_equal, assert_equal, assert_almost_equal
    )
from numpy.ma.core import (
    array, arange, masked, MaskedArray, masked_array, getmaskarray, shape,
    nomask, ones, zeros, count
    )
from numpy.ma.extras import (
    atleast_1d, atleast_2d, atleast_3d, mr_, dot, polyfit, cov, corrcoef,
    median, average, unique, setxor1d, setdiff1d, union1d, intersect1d, in1d,
    ediff1d, apply_over_axes, apply_along_axis, compress_nd, compress_rowcols,
    mask_rowcols, clump_masked, clump_unmasked, flatnotmasked_contiguous,
    notmasked_contiguous, notmasked_edges, masked_all, masked_all_like, isin,
    diagflat, ndenumerate, stack, vstack
    )


class TestGeneric:
    #
    def test_masked_all(self):
        # Tests masked_all
        # Standard dtype
        test = masked_all((2,), dtype=float)
        control = array([1, 1], mask=[1, 1], dtype=float)
        assert_equal(test, control)
        # Flexible dtype
        dt = np.dtype({'names': ['a', 'b'], 'formats': ['f', 'f']})
        test = masked_all((2,), dtype=dt)
        control = array([(0, 0), (0, 0)], mask=[(1, 1), (1, 1)], dtype=dt)
        assert_equal(test, control)
        test = masked_all((2, 2), dtype=dt)
        control = array([[(0, 0), (0, 0)], [(0, 0), (0, 0)]],
                        mask=[[(1, 1), (1, 1)], [(1, 1), (1, 1)]],
                        dtype=dt)
        assert_equal(test, control)
        # Nested dtype
        dt = np.dtype([('a', 'f'), ('b', [('ba', 'f'), ('bb', 'f')])])
        test = masked_all((2,), dtype=dt)
        control = array([(1, (1, 1)), (1, (1, 1))],
                        mask=[(1, (1, 1)), (1, (1, 1))], dtype=dt)
        assert_equal(test, control)
        test = masked_all((2,), dtype=dt)
        control = array([(1, (1, 1)), (1, (1, 1))],
                        mask=[(1, (1, 1)), (1, (1, 1))], dtype=dt)
        assert_equal(test, control)
        test = masked_all((1, 1), dtype=dt)
        control = array([[(1, (1, 1))]], mask=[[(1, (1, 1))]], dtype=dt)
        assert_equal(test, control)

    def test_masked_all_with_object_nested(self):
        # Test masked_all works with nested array with dtype of an 'object'
        # refers to issue #15895
        my_dtype = np.dtype([('b', ([('c', object)], (1,)))])
        masked_arr = np.ma.masked_all((1,), my_dtype)

        assert_equal(type(masked_arr['b']), np.ma.core.MaskedArray)
        assert_equal(type(masked_arr['b']['c']), np.ma.core.MaskedArray)
        assert_equal(len(masked_arr['b']['c']), 1)
        assert_equal(masked_arr['b']['c'].shape, (1, 1))
        assert_equal(masked_arr['b']['c']._fill_value.shape, ())

    def test_masked_all_with_object(self):
        # same as above except that the array is not nested
        my_dtype = np.dtype([('b', (object, (1,)))])
        masked_arr = np.ma.masked_all((1,), my_dtype)

        assert_equal(type(masked_arr['b']), np.ma.core.MaskedArray)
        assert_equal(len(masked_arr['b']), 1)
        assert_equal(masked_arr['b'].shape, (1, 1))
        assert_equal(masked_arr['b']._fill_value.shape, ())

    def test_masked_all_like(self):
        # Tests masked_all
        # Standard dtype
        base = array([1, 2], dtype=float)
        test = masked_all_like(base)
        control = array([1, 1], mask=[1, 1], dtype=float)
        assert_equal(test, control)
        # Flexible dtype
        dt = np.dtype({'names': ['a', 'b'], 'formats': ['f', 'f']})
        base = array([(0, 0), (0, 0)], mask=[(1, 1), (1, 1)], dtype=dt)
        test = masked_all_like(base)
        control = array([(10, 10), (10, 10)], mask=[(1, 1), (1, 1)], dtype=dt)
        assert_equal(test, control)
        # Nested dtype
        dt = np.dtype([('a', 'f'), ('b', [('ba', 'f'), ('bb', 'f')])])
        control = array([(1, (1, 1)), (1, (1, 1))],
                        mask=[(1, (1, 1)), (1, (1, 1))], dtype=dt)
        test = masked_all_like(control)
        assert_equal(test, control)

    def check_clump(self, f):
        for i in range(1, 7):
            for j in range(2**i):
                k = np.arange(i, dtype=int)
                ja = np.full(i, j, dtype=int)
                a = masked_array(2**k)
                a.mask = (ja & (2**k)) != 0
                s = 0
                for sl in f(a):
                    s += a.data[sl].sum()
                if f == clump_unmasked:
                    assert_equal(a.compressed().sum(), s)
                else:
                    a.mask = ~a.mask
                    assert_equal(a.compressed().sum(), s)

    def test_clump_masked(self):
        # Test clump_masked
        a = masked_array(np.arange(10))
        a[[0, 1, 2, 6, 8, 9]] = masked
        #
        test = clump_masked(a)
        control = [slice(0, 3), slice(6, 7), slice(8, 10)]
        assert_equal(test, control)

        self.check_clump(clump_masked)

    def test_clump_unmasked(self):
        # Test clump_unmasked
        a = masked_array(np.arange(10))
        a[[0, 1, 2, 6, 8, 9]] = masked
        test = clump_unmasked(a)
        control = [slice(3, 6), slice(7, 8), ]
        assert_equal(test, control)

        self.check_clump(clump_unmasked)

    def test_flatnotmasked_contiguous(self):
        # Test flatnotmasked_contiguous
        a = arange(10)
        # No mask
        test = flatnotmasked_contiguous(a)
        assert_equal(test, [slice(0, a.size)])
        # mask of all false
        a.mask = np.zeros(10, dtype=bool)
        assert_equal(test, [slice(0, a.size)])
        # Some mask
        a[(a < 3) | (a > 8) | (a == 5)] = masked
        test = flatnotmasked_contiguous(a)
        assert_equal(test, [slice(3, 5), slice(6, 9)])
        #
        a[:] = masked
        test = flatnotmasked_contiguous(a)
        assert_equal(test, [])


class TestAverage:
    # Several tests of average. Why so many ? Good point...
    def test_testAverage1(self):
        # Test of average.
        ott = array([0., 1., 2., 3.], mask=[True, False, False, False])
        assert_equal(2.0, average(ott, axis=0))
        assert_equal(2.0, average(ott, weights=[1., 1., 2., 1.]))
        result, wts = average(ott, weights=[1., 1., 2., 1.], returned=True)
        assert_equal(2.0, result)
        assert_(wts == 4.0)
        ott[:] = masked
        assert_equal(average(ott, axis=0).mask, [True])
        ott = array([0., 1., 2., 3.], mask=[True, False, False, False])
        ott = ott.reshape(2, 2)
        ott[:, 1] = masked
        assert_equal(average(ott, axis=0), [2.0, 0.0])
        assert_equal(average(ott, axis=1).mask[0], [True])
        assert_equal([2., 0.], average(ott, axis=0))
        result, wts = average(ott, axis=0, returned=True)
        assert_equal(wts, [1., 0.])

    def test_testAverage2(self):
        # More tests of average.
        w1 = [0, 1, 1, 1, 1, 0]
        w2 = [[0, 1, 1, 1, 1, 0], [1, 0, 0, 0, 0, 1]]
        x = arange(6, dtype=np.float_)
        assert_equal(average(x, axis=0), 2.5)
        assert_equal(average(x, axis=0, weights=w1), 2.5)
        y = array([arange(6, dtype=np.float_), 2.0 * arange(6)])
        assert_equal(average(y, None), np.add.reduce(np.arange(6)) * 3. / 12.)
        assert_equal(average(y, axis=0), np.arange(6) * 3. / 2.)
        assert_equal(average(y, axis=1),
                     [average(x, axis=0), average(x, axis=0) * 2.0])
        assert_equal(average(y, None, weights=w2), 20. / 6.)
        assert_equal(average(y, axis=0, weights=w2),
                     [0., 1., 2., 3., 4., 10.])
        assert_equal(average(y, axis=1),
                     [average(x, axis=0), average(x, axis=0) * 2.0])
        m1 = zeros(6)
        m2 = [0, 0, 1, 1, 0, 0]
        m3 = [[0, 0, 1, 1, 0, 0], [0, 1, 1, 1, 1, 0]]
        m4 = ones(6)
        m5 = [0, 1, 1, 1, 1, 1]
        assert_equal(average(masked_array(x, m1), axis=0), 2.5)
        assert_equal(average(masked_array(x, m2), axis=0), 2.5)
        assert_equal(average(masked_array(x, m4), axis=0).mask, [True])
        assert_equal(average(masked_array(x, m5), axis=0), 0.0)
        assert_equal(count(average(masked_array(x, m4), axis=0)), 0)
        z = masked_array(y, m3)
        assert_equal(average(z, None), 20. / 6.)
        assert_equal(average(z, axis=0), [0., 1., 99., 99., 4.0, 7.5])
        assert_equal(average(z, axis=1), [2.5, 5.0])
        assert_equal(average(z, axis=0, weights=w2),
                     [0., 1., 99., 99., 4.0, 10.0])

    def test_testAverage3(self):
        # Yet more tests of average!
        a = arange(6)
        b = arange(6) * 3
        r1, w1 = average([[a, b], [b, a]], axis=1, returned=True)
        assert_equal(shape(r1), shape(w1))
        assert_equal(r1.shape, w1.shape)
        r2, w2 = average(ones((2, 2, 3)), axis=0, weights=[3, 1], returned=True)
        assert_equal(shape(w2), shape(r2))
        r2, w2 = average(ones((2, 2, 3)), returned=True)
        assert_equal(shape(w2), shape(r2))
        r2, w2 = average(ones((2, 2, 3)), weights=ones((2, 2, 3)), returned=True)
        assert_equal(shape(w2), shape(r2))
        a2d = array([[1, 2], [0, 4]], float)
        a2dm = masked_array(a2d, [[False, False], [True, False]])
        a2da = average(a2d, axis=0)
        assert_equal(a2da, [0.5, 3.0])
        a2dma = average(a2dm, axis=0)
        assert_equal(a2dma, [1.0, 3.0])
        a2dma = average(a2dm, axis=None)
        assert_equal(a2dma, 7. / 3.)
        a2dma = average(a2dm, axis=1)
        assert_equal(a2dma, [1.5, 4.0])

    def test_testAverage4(self):
        # Test that `keepdims` works with average
        x = np.array([2, 3, 4]).reshape(3, 1)
        b = np.ma.array(x, mask=[[False], [False], [True]])
        w = np.array([4, 5, 6]).reshape(3, 1)
        actual = average(b, weights=w, axis=1, keepdims=True)
        desired = masked_array([[2.], [3.], [4.]], [[False], [False], [True]])
        assert_equal(actual, desired)

    def test_onintegers_with_mask(self):
        # Test average on integers with mask
        a = average(array([1, 2]))
        assert_equal(a, 1.5)
        a = average(array([1, 2, 3, 4], mask=[False, False, True, True]))
        assert_equal(a, 1.5)

    def test_complex(self):
        # Test with complex data.
        # (Regression test for https://github.com/numpy/numpy/issues/2684)
        mask = np.array([[0, 0, 0, 1, 0],
                         [0, 1, 0, 0, 0]], dtype=bool)
        a = masked_array([[0, 1+2j, 3+4j, 5+6j, 7+8j],
                          [9j, 0+1j, 2+3j, 4+5j, 7+7j]],
                         mask=mask)

        av = average(a)
        expected = np.average(a.compressed())
        assert_almost_equal(av.real, expected.real)
        assert_almost_equal(av.imag, expected.imag)

        av0 = average(a, axis=0)
        expected0 = average(a.real, axis=0) + average(a.imag, axis=0)*1j
        assert_almost_equal(av0.real, expected0.real)
        assert_almost_equal(av0.imag, expected0.imag)

        av1 = average(a, axis=1)
        expected1 = average(a.real, axis=1) + average(a.imag, axis=1)*1j
        assert_almost_equal(av1.real, expected1.real)
        assert_almost_equal(av1.imag, expected1.imag)

        # Test with the 'weights' argument.
        wts = np.array([[0.5, 1.0, 2.0, 1.0, 0.5],
                        [1.0, 1.0, 1.0, 1.0, 1.0]])
        wav = average(a, weights=wts)
        expected = np.average(a.compressed(), weights=wts[~mask])
        assert_almost_equal(wav.real, expected.real)
        assert_almost_equal(wav.imag, expected.imag)

        wav0 = average(a, weights=wts, axis=0)
        expected0 = (average(a.real, weights=wts, axis=0) +
                     average(a.imag, weights=wts, axis=0)*1j)
        assert_almost_equal(wav0.real, expected0.real)
        assert_almost_equal(wav0.imag, expected0.imag)

        wav1 = average(a, weights=wts, axis=1)
        expected1 = (average(a.real, weights=wts, axis=1) +
                     average(a.imag, weights=wts, axis=1)*1j)
        assert_almost_equal(wav1.real, expected1.real)
        assert_almost_equal(wav1.imag, expected1.imag)

    @pytest.mark.parametrize(
        'x, axis, expected_avg, weights, expected_wavg, expected_wsum',
        [([1, 2, 3], None, [2.0], [3, 4, 1], [1.75], [8.0]),
         ([[1, 2, 5], [1, 6, 11]], 0, [[1.0, 4.0, 8.0]],
          [1, 3], [[1.0, 5.0, 9.5]], [[4, 4, 4]])],
    )
    def test_basic_keepdims(self, x, axis, expected_avg,
                            weights, expected_wavg, expected_wsum):
        avg = np.ma.average(x, axis=axis, keepdims=True)
        assert avg.shape == np.shape(expected_avg)
        assert_array_equal(avg, expected_avg)

        wavg = np.ma.average(x, axis=axis, weights=weights, keepdims=True)
        assert wavg.shape == np.shape(expected_wavg)
        assert_array_equal(wavg, expected_wavg)

        wavg, wsum = np.ma.average(x, axis=axis, weights=weights,
                                   returned=True, keepdims=True)
        assert wavg.shape == np.shape(expected_wavg)
        assert_array_equal(wavg, expected_wavg)
        assert wsum.shape == np.shape(expected_wsum)
        assert_array_equal(wsum, expected_wsum)

    def test_masked_weights(self):
        # Test with masked weights.
        # (Regression test for https://github.com/numpy/numpy/issues/10438)
        a = np.ma.array(np.arange(9).reshape(3, 3),
                        mask=[[1, 0, 0], [1, 0, 0], [0, 0, 0]])
        weights_unmasked = masked_array([5, 28, 31], mask=False)
        weights_masked = masked_array([5, 28, 31], mask=[1, 0, 0])

        avg_unmasked = average(a, axis=0,
                               weights=weights_unmasked, returned=False)
        expected_unmasked = np.array([6.0, 5.21875, 6.21875])
        assert_almost_equal(avg_unmasked, expected_unmasked)

        avg_masked = average(a, axis=0, weights=weights_masked, returned=False)
        expected_masked = np.array([6.0, 5.576271186440678, 6.576271186440678])
        assert_almost_equal(avg_masked, expected_masked)

        # weights should be masked if needed
        # depending on the array mask. This is to avoid summing
        # masked nan or other values that are not cancelled by a zero
        a = np.ma.array([1.0,   2.0,   3.0,  4.0],
                   mask=[False, False, True, True])
        avg_unmasked = average(a, weights=[1, 1, 1, np.nan])

        assert_almost_equal(avg_unmasked, 1.5)

        a = np.ma.array([
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 1.0, 2.0, 3.0],
        ], mask=[
            [False, True, True, False],
            [True, False, True, True],
            [True, False, True, False],
        ])

        avg_masked = np.ma.average(a, weights=[1, np.nan, 1], axis=0)
        avg_expected = np.ma.array([1.0, np.nan, np.nan, 3.5],
                              mask=[False, True, True, False])

        assert_almost_equal(avg_masked, avg_expected)
        assert_equal(avg_masked.mask, avg_expected.mask)


class TestConcatenator:
    # Tests for mr_, the equivalent of r_ for masked arrays.

    def test_1d(self):
        # Tests mr_ on 1D arrays.
        assert_array_equal(mr_[1, 2, 3, 4, 5, 6], array([1, 2, 3, 4, 5, 6]))
        b = ones(5)
        m = [1, 0, 0, 0, 0]
        d = masked_array(b, mask=m)
        c = mr_[d, 0, 0, d]
        assert_(isinstance(c, MaskedArray))
        assert_array_equal(c, [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1])
        assert_array_equal(c.mask, mr_[m, 0, 0, m])

    def test_2d(self):
        # Tests mr_ on 2D arrays.
        a_1 = np.random.rand(5, 5)
        a_2 = np.random.rand(5, 5)
        m_1 = np.round_(np.random.rand(5, 5), 0)
        m_2 = np.round_(np.random.rand(5, 5), 0)
        b_1 = masked_array(a_1, mask=m_1)
        b_2 = masked_array(a_2, mask=m_2)
        # append columns
        d = mr_['1', b_1, b_2]
        assert_(d.shape == (5, 10))
        assert_array_equal(d[:, :5], b_1)
        assert_array_equal(d[:, 5:], b_2)
        assert_array_equal(d.mask, np.r_['1', m_1, m_2])
        d = mr_[b_1, b_2]
        assert_(d.shape == (10, 5))
        assert_array_equal(d[:5,:], b_1)
        assert_array_equal(d[5:,:], b_2)
        assert_array_equal(d.mask, np.r_[m_1, m_2])

    def test_masked_constant(self):
        actual = mr_[np.ma.masked, 1]
        assert_equal(actual.mask, [True, False])
        assert_equal(actual.data[1], 1)

        actual = mr_[[1, 2], np.ma.masked]
        assert_equal(actual.mask, [False, False, True])
        assert_equal(actual.data[:2], [1, 2])


class TestNotMasked:
    # Tests notmasked_edges and notmasked_contiguous.

    def test_edges(self):
        # Tests unmasked_edges
        data = masked_array(np.arange(25).reshape(5, 5),
                            mask=[[0, 0, 1, 0, 0],
                                  [0, 0, 0, 1, 1],
                                  [1, 1, 0, 0, 0],
                                  [0, 0, 0, 0, 0],
                                  [1, 1, 1, 0, 0]],)
        test = notmasked_edges(data, None)
        assert_equal(test, [0, 24])
        test = notmasked_edges(data, 0)
        assert_equal(test[0], [(0, 0, 1, 0, 0), (0, 1, 2, 3, 4)])
        assert_equal(test[1], [(3, 3, 3, 4, 4), (0, 1, 2, 3, 4)])
        test = notmasked_edges(data, 1)
        assert_equal(test[0], [(0, 1, 2, 3, 4), (0, 0, 2, 0, 3)])
        assert_equal(test[1], [(0, 1, 2, 3, 4), (4, 2, 4, 4, 4)])
        #
        test = notmasked_edges(data.data, None)
        assert_equal(test, [0, 24])
        test = notmasked_edges(data.data, 0)
        assert_equal(test[0], [(0, 0, 0, 0, 0), (0, 1, 2, 3, 4)])
        assert_equal(test[1], [(4, 4, 4, 4, 4), (0, 1, 2, 3, 4)])
        test = notmasked_edges(data.data, -1)
        assert_equal(test[0], [(0, 1, 2, 3, 4), (0, 0, 0, 0, 0)])
        assert_equal(test[1], [(0, 1, 2, 3, 4), (4, 4, 4, 4, 4)])
        #
        data[-2] = masked
        test = notmasked_edges(data, 0)
        assert_equal(test[0], [(0, 0, 1, 0, 0), (0, 1, 2, 3, 4)])
        assert_equal(test[1], [(1, 1, 2, 4, 4), (0, 1, 2, 3, 4)])
        test = notmasked_edges(data, -1)
        assert_equal(test[0], [(0, 1, 2, 4), (0, 0, 2, 3)])
        assert_equal(test[1], [(0, 1, 2, 4), (4, 2, 4, 4)])

    def test_contiguous(self):
        # Tests notmasked_contiguous
        a = masked_array(np.arange(24).reshape(3, 8),
                         mask=[[0, 0, 0, 0, 1, 1, 1, 1],
                               [1, 1, 1, 1, 1, 1, 1, 1],
                               [0, 0, 0, 0, 0, 0, 1, 0]])
        tmp = notmasked_contiguous(a, None)
        assert_equal(tmp, [
            slice(0, 4, None),
            slice(16, 22, None),
            slice(23, 24, None)
        ])

        tmp = notmasked_contiguous(a, 0)
        assert_equal(tmp, [
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(0, 1, None), slice(2, 3, None)],
            [slice(2, 3, None)],
            [slice(2, 3, None)],
            [],
            [slice(2, 3, None)]
        ])
        #
        tmp = notmasked_contiguous(a, 1)
        assert_equal(tmp, [
            [slice(0, 4, None)],
            [],
            [slice(0, 6, None), slice(7, 8, None)]
        ])


class TestCompressFunctions:

    def test_compress_nd(self):
        # Tests compress_nd
        x = np.array(list(range(3*4*5))).reshape(3, 4, 5)
        m = np.zeros((3,4,5)).astype(bool)
        m[1,1,1] = True
        x = array(x, mask=m)

        # axis=None
        a = compress_nd(x)
        assert_equal(a, [[[ 0,  2,  3,  4],
                          [10, 12, 13, 14],
                          [15, 17, 18, 19]],
                         [[40, 42, 43, 44],
                          [50, 52, 53, 54],
                          [55, 57, 58, 59]]])

        # axis=0
        a = compress_nd(x, 0)
        assert_equal(a, [[[ 0,  1,  2,  3,  4],
                          [ 5,  6,  7,  8,  9],
                          [10, 11, 12, 13, 14],
                          [15, 16, 17, 18, 19]],
                         [[40, 41, 42, 43, 44],
                          [45, 46, 47, 48, 49],
                          [50, 51, 52, 53, 54],
                          [55, 56, 57, 58, 59]]])

        # axis=1
        a = compress_nd(x, 1)
        assert_equal(a, [[[ 0,  1,  2,  3,  4],
                          [10, 11, 12, 13, 14],
                          [15, 16, 17, 18, 19]],
                         [[20, 21, 22, 23, 24],
                          [30, 31, 32, 33, 34],
                          [35, 36, 37, 38, 39]],
                         [[40, 41, 42, 43, 44],
                          [50, 51, 52, 53, 54],
                          [55, 56, 57, 58, 59]]])

        a2 = compress_nd(x, (1,))
        a3 = compress_nd(x, -2)
        a4 = compress_nd(x, (-2,))
        assert_equal(a, a2)
        assert_equal(a, a3)
        assert_equal(a, a4)

        # axis=2
        a = compress_nd(x, 2)
        assert_equal(a, [[[ 0, 2,  3,  4],
                          [ 5, 7,  8,  9],
                          [10, 12, 13, 14],
                          [15, 17, 18, 19]],
                         [[20, 22, 23, 24],
                          [25, 27, 28, 29],
                          [30, 32, 33, 34],
                          [35, 37, 38, 39]],
                         [[40, 42, 43, 44],
                          [45, 47, 48, 49],
                          [50, 52, 53, 54],
                          [55, 57, 58, 59]]])

        a2 = compress_nd(x, (2,))
        a3 = compress_nd(x, -1)
        a4 = compress_nd(x, (-1,))
        assert_equal(a, a2)
        assert_equal(a, a3)
        assert_equal(a, a4)

        # axis=(0, 1)
        a = compress_nd(x, (0, 1))
        assert_equal(a, [[[ 0,  1,  2,  3,  4],
                          [10, 11, 12, 13, 14],
                          [15, 16, 17, 18, 19]],
                         [[40, 41, 42, 43, 44],
                          [50, 51, 52, 53, 54],
                          [55, 56, 57, 58, 59]]])
        a2 = compress_nd(x, (0, -2))
        assert_equal(a, a2)

        # axis=(1, 2)
        a = compress_nd(x, (1, 2))
        assert_equal(a, [[[ 0,  2,  3,  4],
                          [10, 12, 13, 14],
                          [15, 17, 18, 19]],
                         [[20, 22, 23, 24],
                          [30, 32, 33, 34],
                          [35, 37, 38, 39]],
                         [[40, 42, 43, 44],
                          [50, 52, 53, 54],
                          [55, 57, 58, 59]]])

        a2 = compress_nd(x, (-2, 2))
        a3 = compress_nd(x, (1, -1))
        a4 = compress_nd(x, (-2, -1))
        assert_equal(a, a2)
        assert_equal(a, a3)
        assert_equal(a, a4)

        # axis=(0, 2)
        a = compress_nd(x, (0, 2))
        assert_equal(a, [[[ 0,  2,  3,  4],
                          [ 5,  7,  8,  9],
                          [10, 12, 13, 14],
                          [15, 17, 18, 19]],
                         [[40, 42, 43, 44],
                          [45, 47, 48, 49],
                          [50, 52, 53, 54],
                          [55, 57, 58, 59]]])

        a2 = compress_nd(x, (0, -1))
        assert_equal(a, a2)

    def test_compress_rowcols(self):
        # Tests compress_rowcols
        x = array(np.arange(9).reshape(3, 3),
                  mask=[[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert_equal(compress_rowcols(x), [[4, 5], [7, 8]])
        assert_equal(compress_rowcols(x, 0), [[3, 4, 5], [6, 7, 8]])
        assert_equal(compress_rowcols(x, 1), [[1, 2], [4, 5], [7, 8]])
        x = array(x._data, mask=[[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_equal(compress_rowcols(x), [[0, 2], [6, 8]])
        assert_equal(compress_rowcols(x, 0), [[0, 1, 2], [6, 7, 8]])
        assert_equal(compress_rowcols(x, 1), [[0, 2], [3, 5], [6, 8]])
        x = array(x._data, mask=[[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_equal(compress_rowcols(x), [[8]])
        assert_equal(compress_rowcols(x, 0), [[6, 7, 8]])
        assert_equal(compress_rowcols(x, 1,), [[2], [5], [8]])
        x = array(x._data, mask=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert_equal(compress_rowcols(x).size, 0)
        assert_equal(compress_rowcols(x, 0).size, 0)
        assert_equal(compress_rowcols(x, 1).size, 0)

    def test_mask_rowcols(self):
        # Tests mask_rowcols.
        x = array(np.arange(9).reshape(3, 3),
                  mask=[[1, 0, 0], [0, 0, 0], [0, 0, 0]])
        assert_equal(mask_rowcols(x).mask,
                     [[1, 1, 1], [1, 0, 0], [1, 0, 0]])
        assert_equal(mask_rowcols(x, 0).mask,
                     [[1, 1, 1], [0, 0, 0], [0, 0, 0]])
        assert_equal(mask_rowcols(x, 1).mask,
                     [[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        x = array(x._data, mask=[[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_equal(mask_rowcols(x).mask,
                     [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        assert_equal(mask_rowcols(x, 0).mask,
                     [[0, 0, 0], [1, 1, 1], [0, 0, 0]])
        assert_equal(mask_rowcols(x, 1).mask,
                     [[0, 1, 0], [0, 1, 0], [0, 1, 0]])
        x = array(x._data, mask=[[1, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_equal(mask_rowcols(x).mask,
                     [[1, 1, 1], [1, 1, 1], [1, 1, 0]])
        assert_equal(mask_rowcols(x, 0).mask,
                     [[1, 1, 1], [1, 1, 1], [0, 0, 0]])
        assert_equal(mask_rowcols(x, 1,).mask,
                     [[1, 1, 0], [1, 1, 0], [1, 1, 0]])
        x = array(x._data, mask=[[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        assert_(mask_rowcols(x).all() is masked)
        assert_(mask_rowcols(x, 0).all() is masked)
        assert_(mask_rowcols(x, 1).all() is masked)
        assert_(mask_rowcols(x).mask.all())
        assert_(mask_rowcols(x, 0).mask.all())
        assert_(mask_rowcols(x, 1).mask.all())

    @pytest.mark.parametrize("axis", [None, 0, 1])
    @pytest.mark.parametrize(["func", "rowcols_axis"],
                             [(np.ma.mask_rows, 0), (np.ma.mask_cols, 1)])
    def test_mask_row_cols_axis_deprecation(self, axis, func, rowcols_axis):
        # Test deprecation of the axis argument to `mask_rows` and `mask_cols`
        x = array(np.arange(9).reshape(3, 3),
                  mask=[[1, 0, 0], [0, 0, 0], [0, 0, 0]])

        with assert_warns(DeprecationWarning):
            res = func(x, axis=axis)
            assert_equal(res, mask_rowcols(x, rowcols_axis))

    def test_dot(self):
        # Tests dot product
        n = np.arange(1, 7)
        #
        m = [1, 0, 0, 0, 0, 0]
        a = masked_array(n, mask=m).reshape(2, 3)
        b = masked_array(n, mask=m).reshape(3, 2)
        c = dot(a, b, strict=True)
        assert_equal(c.mask, [[1, 1], [1, 0]])
        c = dot(b, a, strict=True)
        assert_equal(c.mask, [[1, 1, 1], [1, 0, 0], [1, 0, 0]])
        c = dot(a, b, strict=False)
        assert_equal(c, np.dot(a.filled(0), b.filled(0)))
        c = dot(b, a, strict=False)
        assert_equal(c, np.dot(b.filled(0), a.filled(0)))
        #
        m = [0, 0, 0, 0, 0, 1]
        a = masked_array(n, mask=m).reshape(2, 3)
        b = masked_array(n, mask=m).reshape(3, 2)
        c = dot(a, b, strict=True)
        assert_equal(c.mask, [[0, 1], [1, 1]])
        c = dot(b, a, strict=True)
        assert_equal(c.mask, [[0, 0, 1], [0, 0, 1], [1, 1, 1]])
        c = dot(a, b, strict=False)
        assert_equal(c, np.dot(a.filled(0), b.filled(0)))
        assert_equal(c, dot(a, b))
        c = dot(b, a, strict=False)
        assert_equal(c, np.dot(b.filled(0), a.filled(0)))
        #
        m = [0, 0, 0, 0, 0, 0]
        a = masked_array(n, mask=m).reshape(2, 3)
        b = masked_array(n, mask=m).reshape(3, 2)
        c = dot(a, b)
        assert_equal(c.mask, nomask)
        c = dot(b, a)
        assert_equal(c.mask, nomask)
        #
        a = masked_array(n, mask=[1, 0, 0, 0, 0, 0]).reshape(2, 3)
        b = masked_array(n, mask=[0, 0, 0, 0, 0, 0]).reshape(3, 2)
        c = dot(a, b, strict=True)
        assert_equal(c.mask, [[1, 1], [0, 0]])
        c = dot(a, b, strict=False)
        assert_equal(c, np.dot(a.filled(0), b.filled(0)))
        c = dot(b, a, strict=True)
        assert_equal(c.mask, [[1, 0, 0], [1, 0, 0], [1, 0, 0]])
        c = dot(b, a, strict=False)
        assert_equal(c, np.dot(b.filled(0), a.filled(0)))
        #
        a = masked_array(n, mask=[0, 0, 0, 0, 0, 1]).reshape(2, 3)
        b = masked_array(n, mask=[0, 0, 0, 0, 0, 0]).reshape(3, 2)
        c = dot(a, b, strict=True)
        assert_equal(c.mask, [[0, 0], [1, 1]])
        c = dot(a, b)
        assert_equal(c, np.dot(a.filled(0), b.filled(0)))
        c = dot(b, a, strict=True)
        assert_equal(c.mask, [[0, 0, 1], [0, 0, 1], [0, 0, 1]])
        c = dot(b, a, strict=False)
        assert_equal(c, np.dot(b.filled(0), a.filled(0)))
        #
        a = masked_array(n, mask=[0, 0, 0, 0, 0, 1]).reshape(2, 3)
        b = masked_array(n, mask=[0, 0, 1, 0, 0, 0]).reshape(3, 2)
        c = dot(a, b, strict=True)
        assert_equal(c.mask, [[1, 0], [1, 1]])
        c = dot(a, b, strict=False)
        assert_equal(c, np.dot(a.filled(0), b.filled(0)))
        c = dot(b, a, strict=True)
        assert_equal(c.mask, [[0, 0, 1], [1, 1, 1], [0, 0, 1]])
        c = dot(b, a, strict=False)
        assert_equal(c, np.dot(b.filled(0), a.filled(0)))

    def test_dot_returns_maskedarray(self):
        # See gh-6611
        a = np.eye(3)
        b = array(a)
        assert_(type(dot(a, a)) is MaskedArray)
        assert_(type(dot(a, b)) is MaskedArray)
        assert_(type(dot(b, a)) is MaskedArray)
        assert_(type(dot(b, b)) is MaskedArray)

    def test_dot_out(self):
        a = array(np.eye(3))
        out = array(np.zeros((3, 3)))
        res = dot(a, a, out=out)
        assert_(res is out)
        assert_equal(a, res)


class TestApplyAlongAxis:
    # Tests 2D functions
    def test_3d(self):
        a = arange(12.).reshape(2, 2, 3)

        def myfunc(b):
            return b[1]

        xa = apply_along_axis(myfunc, 2, a)
        assert_equal(xa, [[1, 4], [7, 10]])

    # Tests kwargs functions
    def test_3d_kwargs(self):
        a = arange(12).reshape(2, 2, 3)

        def myfunc(b, offset=0):
            return b[1+offset]

        xa = apply_along_axis(myfunc, 2, a, offset=1)
        assert_equal(xa, [[2, 5], [8, 11]])


class TestApplyOverAxes:
    # Tests apply_over_axes
    def test_basic(self):
        a = arange(24).reshape(2, 3, 4)
        test = apply_over_axes(np.sum, a, [0, 2])
        ctrl = np.array([[[60], [92], [124]]])
        assert_equal(test, ctrl)
        a[(a % 2).astype(bool)] = masked
        test = apply_over_axes(np.sum, a, [0, 2])
        ctrl = np.array([[[28], [44], [60]]])
        assert_equal(test, ctrl)


class TestMedian:
    def test_pytype(self):
        r = np.ma.median([[np.inf, np.inf], [np.inf, np.inf]], axis=-1)
        assert_equal(r, np.inf)

    def test_inf(self):
        # test that even which computes handles inf / x = masked
        r = np.ma.median(np.ma.masked_array([[np.inf, np.inf],
                                             [np.inf, np.inf]]), axis=-1)
        assert_equal(r, np.inf)
        r = np.ma.median(np.ma.masked_array([[np.inf, np.inf],
                                             [np.inf, np.inf]]), axis=None)
        assert_equal(r, np.inf)
        # all masked
        r = np.ma.median(np.ma.masked_array([[np.inf, np.inf],
                                             [np.inf, np.inf]], mask=True),
                         axis=-1)
        assert_equal(r.mask, True)
        r = np.ma.median(np.ma.masked_array([[np.inf, np.inf],
                                             [np.inf, np.inf]], mask=True),
                         axis=None)
        assert_equal(r.mask, True)

    def test_non_masked(self):
        x = np.arange(9)
        assert_equal(np.ma.median(x), 4.)
        assert_(type(np.ma.median(x)) is not MaskedArray)
        x = range(8)
        assert_equal(np.ma.median(x), 3.5)
        assert_(type(np.ma.median(x)) is not MaskedArray)
        x = 5
        assert_equal(np.ma.median(x), 5.)
        assert_(type(np.ma.median(x)) is not MaskedArray)
        # integer
        x = np.arange(9 * 8).reshape(9, 8)
        assert_equal(np.ma.median(x, axis=0), np.median(x, axis=0))
        assert_equal(np.ma.median(x, axis=1), np.median(x, axis=1))
        assert_(np.ma.median(x, axis=1) is not MaskedArray)
        # float
        x = np.arange(9 * 8.).reshape(9, 8)
        assert_equal(np.ma.median(x, axis=0), np.median(x, axis=0))
        assert_equal(np.ma.median(x, axis=1), np.median(x, axis=1))
        assert_(np.ma.median(x, axis=1) is not MaskedArray)

    def test_docstring_examples(self):
        "test the examples given in the docstring of ma.median"
        x = array(np.arange(8), mask=[0]*4 + [1]*4)
        assert_equal(np.ma.median(x), 1.5)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)
        x = array(np.arange(10).reshape(2, 5), mask=[0]*6 + [1]*4)
        assert_equal(np.ma.median(x), 2.5)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)
        ma_x = np.ma.median(x, axis=-1, overwrite_input=True)
        assert_equal(ma_x, [2., 5.])
        assert_equal(ma_x.shape, (2,), "shape mismatch")
        assert_(type(ma_x) is MaskedArray)

    def test_axis_argument_errors(self):
        msg = "mask = %s, ndim = %s, axis = %s, overwrite_input = %s"
        for ndmin in range(5):
            for mask in [False, True]:
                x = array(1, ndmin=ndmin, mask=mask)

                # Valid axis values should not raise exception
                args = itertools.product(range(-ndmin, ndmin), [False, True])
                for axis, over in args:
                    try:
                        np.ma.median(x, axis=axis, overwrite_input=over)
                    except Exception:
                        raise AssertionError(msg % (mask, ndmin, axis, over))

                # Invalid axis values should raise exception
                args = itertools.product([-(ndmin + 1), ndmin], [False, True])
                for axis, over in args:
                    try:
                        np.ma.median(x, axis=axis, overwrite_input=over)
                    except np.AxisError:
                        pass
                    else:
                        raise AssertionError(msg % (mask, ndmin, axis, over))

    def test_masked_0d(self):
        # Check values
        x = array(1, mask=False)
        assert_equal(np.ma.median(x), 1)
        x = array(1, mask=True)
        assert_equal(np.ma.median(x), np.ma.masked)

    def test_masked_1d(self):
        x = array(np.arange(5), mask=True)
        assert_equal(np.ma.median(x), np.ma.masked)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is np.ma.core.MaskedConstant)
        x = array(np.arange(5), mask=False)
        assert_equal(np.ma.median(x), 2.)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)
        x = array(np.arange(5), mask=[0,1,0,0,0])
        assert_equal(np.ma.median(x), 2.5)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)
        x = array(np.arange(5), mask=[0,1,1,1,1])
        assert_equal(np.ma.median(x), 0.)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)
        # integer
        x = array(np.arange(5), mask=[0,1,1,0,0])
        assert_equal(np.ma.median(x), 3.)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)
        # float
        x = array(np.arange(5.), mask=[0,1,1,0,0])
        assert_equal(np.ma.median(x), 3.)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)
        # integer
        x = array(np.arange(6), mask=[0,1,1,1,1,0])
        assert_equal(np.ma.median(x), 2.5)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)
        # float
        x = array(np.arange(6.), mask=[0,1,1,1,1,0])
        assert_equal(np.ma.median(x), 2.5)
        assert_equal(np.ma.median(x).shape, (), "shape mismatch")
        assert_(type(np.ma.median(x)) is not MaskedArray)

    def test_1d_shape_consistency(self):
        assert_equal(np.ma.median(array([1,2,3],mask=[0,0,0])).shape,
                     np.ma.median(array([1,2,3],mask=[0,1,0])).shape )

    def test_2d(self):
        # Tests median w/ 2D
        (n, p) = (101, 30)
        x = masked_array(np.linspace(-1., 1., n),)
        x[:10] = x[-10:] = masked
        z = masked_array(np.empty((n, p), dtype=float))
        z[:, 0] = x[:]
        idx = np.arange(len(x))
        for i in range(1, p):
            np.random.shuffle(idx)
            z[:, i] = x[idx]
        assert_equal(median(z[:, 0]), 0)
        assert_equal(median(z), 0)
        assert_equal(median(z, axis=0), np.zeros(p))
        assert_equal(median(z.T, axis=1), np.zeros(p))

    def test_2d_waxis(self):
        # Tests median w/ 2D arrays and different axis.
        x = masked_array(np.arange(30).reshape(10, 3))
        x[:3] = x[-3:] = masked
        assert_equal(median(x), 14.5)
        assert_(type(np.ma.median(x)) is not MaskedArray)
        assert_equal(median(x, axis=0), [13.5, 14.5, 15.5])
        assert_(type(np.ma.median(x, axis=0)) is MaskedArray)
        assert_equal(median(x, axis=1), [0, 0, 0, 10, 13, 16, 19, 0, 0, 0])
        assert_(type(np.ma.median(x, axis=1)) is MaskedArray)
        assert_equal(median(x, axis=1).mask, [1, 1, 1, 0, 0, 0, 0, 1, 1, 1])

    def test_3d(self):
        # Tests median w/ 3D
        x = np.ma.arange(24).reshape(3, 4, 2)
        x[x % 3 == 0] = masked
        assert_equal(median(x, 0), [[12, 9], [6, 15], [12, 9], [18, 15]])
        x.shape = (4, 3, 2)
        assert_equal(median(x, 0), [[99, 10], [11, 99], [13, 14]])
        x = np.ma.arange(24).reshape(4, 3, 2)
        x[x % 5 == 0] = masked
        assert_equal(median(x, 0), [[12, 10], [8, 9], [16, 17]])

    def test_neg_axis(self):
        x = masked_array(np.arange(30).reshape(10, 3))
        x[:3] = x[-3:] = masked
        assert_equal(median(x, axis=-1), median(x, axis=1))

    def test_out_1d(self):
        # integer float even odd
        for v in (30, 30., 31, 31.):
            x = masked_array(np.arange(v))
            x[:3] = x[-3:] = masked
            out = masked_array(np.ones(()))
            r = median(x, out=out)
            if v == 30:
                assert_equal(out, 14.5)
            else:
                assert_equal(out, 15.)
            assert_(r is out)
            assert_(type(r) is MaskedArray)

    def test_out(self):
        # integer float even odd
        for v in (40, 40., 30, 30.):
            x = masked_array(np.arange(v).reshape(10, -1))
            x[:3] = x[-3:] = masked
            out = masked_array(np.ones(10))
            r = median(x, axis=1, out=out)
            if v == 30:
                e = masked_array([0.]*3 + [10, 13, 16, 19] + [0.]*3,
                                 mask=[True] * 3 + [False] * 4 + [True] * 3)
            else:
                e = masked_array([0.]*3 + [13.5, 17.5, 21.5, 25.5] + [0.]*3,
                                 mask=[True]*3 + [False]*4 + [True]*3)
            assert_equal(r, e)
            assert_(r is out)
            assert_(type(r) is MaskedArray)

    @pytest.mark.parametrize(
        argnames='axis',
        argvalues=[
            None,
            1,
            (1, ),
            (0, 1),
            (-3, -1),
        ]
    )
    def test_keepdims_out(self, axis):
        mask = np.zeros((3, 5, 7, 11), dtype=bool)
        # Randomly set some elements to True:
        w = np.random.random((4, 200)) * np.array(mask.shape)[:, None]
        w = w.astype(np.intp)
        mask[tuple(w)] = np.nan
        d = masked_array(np.ones(mask.shape), mask=mask)
        if axis is None:
            shape_out = (1,) * d.ndim
        else:
            axis_norm = normalize_axis_tuple(axis, d.ndim)
            shape_out = tuple(
                1 if i in axis_norm else d.shape[i] for i in range(d.ndim))
        out = masked_array(np.empty(shape_out))
        result = median(d, axis=axis, keepdims=True, out=out)
        assert result is out
        assert_equal(result.shape, shape_out)

    def test_single_non_masked_value_on_axis(self):
        data = [[1., 0.],
                [0., 3.],
                [0., 0.]]
        masked_arr = np.ma.masked_equal(data, 0)
        expected = [1., 3.]
        assert_array_equal(np.ma.median(masked_arr, axis=0),
                           expected)

    def test_nan(self):
        for mask in (False, np.zeros(6, dtype=bool)):
            dm = np.ma.array([[1, np.nan, 3], [1, 2, 3]])
            dm.mask = mask

            # scalar result
            r = np.ma.median(dm, axis=None)
            assert_(np.isscalar(r))
            assert_array_equal(r, np.nan)
            r = np.ma.median(dm.ravel(), axis=0)
            assert_(np.isscalar(r))
            assert_array_equal(r, np.nan)

            r = np.ma.median(dm, axis=0)
            assert_equal(type(r), MaskedArray)
            assert_array_equal(r, [1, np.nan, 3])
            r = np.ma.median(dm, axis=1)
            assert_equal(type(r), MaskedArray)
            assert_array_equal(r, [np.nan, 2])
            r = np.ma.median(dm, axis=-1)
            assert_equal(type(r), MaskedArray)
            assert_array_equal(r, [np.nan, 2])

        dm = np.ma.array([[1, np.nan, 3], [1, 2, 3]])
        dm[:, 2] = np.ma.masked
        assert_array_equal(np.ma.median(dm, axis=None), np.nan)
        assert_array_equal(np.ma.median(dm, axis=0), [1, np.nan, 3])
        assert_array_equal(np.ma.median(dm, axis=1), [np.nan, 1.5])

    def test_out_nan(self):
        o = np.ma.masked_array(np.zeros((4,)))
        d = np.ma.masked_array(np.ones((3, 4)))
        d[2, 1] = np.nan
        d[2, 2] = np.ma.masked
        assert_equal(np.ma.median(d, 0, out=o), o)
        o = np.ma.masked_array(np.zeros((3,)))
        assert_equal(np.ma.median(d, 1, out=o), o)
        o = np.ma.masked_array(np.zeros(()))
        assert_equal(np.ma.median(d, out=o), o)

    def test_nan_behavior(self):
        a = np.ma.masked_array(np.arange(24, dtype=float))
        a[::3] = np.ma.masked
        a[2] = np.nan
        assert_array_equal(np.ma.median(a), np.nan)
        assert_array_equal(np.ma.median(a, axis=0), np.nan)

        a = np.ma.masked_array(np.arange(24, dtype=float).reshape(2, 3, 4))
        a.mask = np.arange(a.size) % 2 == 1
        aorig = a.copy()
        a[1, 2, 3] = np.nan
        a[1, 1, 2] = np.nan

        # no axis
        assert_array_equal(np.ma.median(a), np.nan)
        assert_(np.isscalar(np.ma.median(a)))

        # axis0
        b = np.ma.median(aorig, axis=0)
        b[2, 3] = np.nan
        b[1, 2] = np.nan
        assert_equal(np.ma.median(a, 0), b)

        # axis1
        b = np.ma.median(aorig, axis=1)
        b[1, 3] = np.nan
        b[1, 2] = np.nan
        assert_equal(np.ma.median(a, 1), b)

        # axis02
        b = np.ma.median(aorig, axis=(0, 2))
        b[1] = np.nan
        b[2] = np.nan
        assert_equal(np.ma.median(a, (0, 2)), b)

    def test_ambigous_fill(self):
        # 255 is max value, used as filler for sort
        a = np.array([[3, 3, 255], [3, 3, 255]], dtype=np.uint8)
        a = np.ma.masked_array(a, mask=a == 3)
        assert_array_equal(np.ma.median(a, axis=1), 255)
        assert_array_equal(np.ma.median(a, axis=1).mask, False)
        assert_array_equal(np.ma.median(a, axis=0), a[0])
        assert_array_equal(np.ma.median(a), 255)

    def test_special(self):
        for inf in [np.inf, -np.inf]:
            a = np.array([[inf,  np.nan], [np.nan, np.nan]])
            a = np.ma.masked_array(a, mask=np.isnan(a))
            assert_equal(np.ma.median(a, axis=0), [inf,  np.nan])
            assert_equal(np.ma.median(a, axis=1), [inf,  np.nan])
            assert_equal(np.ma.median(a), inf)

            a = np.array([[np.nan, np.nan, inf], [np.nan, np.nan, inf]])
            a = np.ma.masked_array(a, mask=np.isnan(a))
            assert_array_equal(np.ma.median(a, axis=1), inf)
            assert_array_equal(np.ma.median(a, axis=1).mask, False)
            assert_array_equal(np.ma.median(a, axis=0), a[0])
            assert_array_equal(np.ma.median(a), inf)

            # no mask
            a = np.array([[inf, inf], [inf, inf]])
            assert_equal(np.ma.median(a), inf)
            assert_equal(np.ma.median(a, axis=0), inf)
            assert_equal(np.ma.median(a, axis=1), inf)

            a = np.array([[inf, 7, -inf, -9],
                          [-10, np.nan, np.nan, 5],
                          [4, np.nan, np.nan, inf]],
                          dtype=np.float32)
            a = np.ma.masked_array(a, mask=np.isnan(a))
            if inf > 0:
                assert_equal(np.ma.median(a, axis=0), [4., 7., -inf, 5.])
                assert_equal(np.ma.median(a), 4.5)
            else:
                assert_equal(np.ma.median(a, axis=0), [-10., 7., -inf, -9.])
                assert_equal(np.ma.median(a), -2.5)
            assert_equal(np.ma.median(a, axis=1), [-1., -2.5, inf])

            for i in range(0, 10):
                for j in range(1, 10):
                    a = np.array([([np.nan] * i) + ([inf] * j)] * 2)
                    a = np.ma.masked_array(a, mask=np.isnan(a))
                    assert_equal(np.ma.median(a), inf)
                    assert_equal(np.ma.median(a, axis=1), inf)
                    assert_equal(np.ma.median(a, axis=0),
                                 ([np.nan] * i) + [inf] * j)

    def test_empty(self):
        # empty arrays
        a = np.ma.masked_array(np.array([], dtype=float))
        with suppress_warnings() as w:
            w.record(RuntimeWarning)
            assert_array_equal(np.ma.median(a), np.nan)
            assert_(w.log[0].category is RuntimeWarning)

        # multiple dimensions
        a = np.ma.masked_array(np.array([], dtype=float, ndmin=3))
        # no axis
        with suppress_warnings() as w:
            w.record(RuntimeWarning)
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_array_equal(np.ma.median(a), np.nan)
            assert_(w.log[0].category is RuntimeWarning)

        # axis 0 and 1
        b = np.ma.masked_array(np.array([], dtype=float, ndmin=2))
        assert_equal(np.ma.median(a, axis=0), b)
        assert_equal(np.ma.median(a, axis=1), b)

        # axis 2
        b = np.ma.masked_array(np.array(np.nan, dtype=float, ndmin=2))
        with warnings.catch_warnings(record=True) as w:
            warnings.filterwarnings('always', '', RuntimeWarning)
            assert_equal(np.ma.median(a, axis=2), b)
            assert_(w[0].category is RuntimeWarning)

    def test_object(self):
        o = np.ma.masked_array(np.arange(7.))
        assert_(type(np.ma.median(o.astype(object))), float)
        o[2] = np.nan
        assert_(type(np.ma.median(o.astype(object))), float)


class TestCov:

    def setup_method(self):
        self.data = array(np.random.rand(12))

    def test_1d_without_missing(self):
        # Test cov on 1D variable w/o missing values
        x = self.data
        assert_almost_equal(np.cov(x), cov(x))
        assert_almost_equal(np.cov(x, rowvar=False), cov(x, rowvar=False))
        assert_almost_equal(np.cov(x, rowvar=False, bias=True),
                            cov(x, rowvar=False, bias=True))

    def test_2d_without_missing(self):
        # Test cov on 1 2D variable w/o missing values
        x = self.data.reshape(3, 4)
        assert_almost_equal(np.cov(x), cov(x))
        assert_almost_equal(np.cov(x, rowvar=False), cov(x, rowvar=False))
        assert_almost_equal(np.cov(x, rowvar=False, bias=True),
                            cov(x, rowvar=False, bias=True))

    def test_1d_with_missing(self):
        # Test cov 1 1D variable w/missing values
        x = self.data
        x[-1] = masked
        x -= x.mean()
        nx = x.compressed()
        assert_almost_equal(np.cov(nx), cov(x))
        assert_almost_equal(np.cov(nx, rowvar=False), cov(x, rowvar=False))
        assert_almost_equal(np.cov(nx, rowvar=False, bias=True),
                            cov(x, rowvar=False, bias=True))
        #
        try:
            cov(x, allow_masked=False)
        except ValueError:
            pass
        #
        # 2 1D variables w/ missing values
        nx = x[1:-1]
        assert_almost_equal(np.cov(nx, nx[::-1]), cov(x, x[::-1]))
        assert_almost_equal(np.cov(nx, nx[::-1], rowvar=False),
                            cov(x, x[::-1], rowvar=False))
        assert_almost_equal(np.cov(nx, nx[::-1], rowvar=False, bias=True),
                            cov(x, x[::-1], rowvar=False, bias=True))

    def test_2d_with_missing(self):
        # Test cov on 2D variable w/ missing value
        x = self.data
        x[-1] = masked
        x = x.reshape(3, 4)
        valid = np.logical_not(getmaskarray(x)).astype(int)
        frac = np.dot(valid, valid.T)
        xf = (x - x.mean(1)[:, None]).filled(0)
        assert_almost_equal(cov(x),
                            np.cov(xf) * (x.shape[1] - 1) / (frac - 1.))
        assert_almost_equal(cov(x, bias=True),
                            np.cov(xf, bias=True) * x.shape[1] / frac)
        frac = np.dot(valid.T, valid)
        xf = (x - x.mean(0)).filled(0)
        assert_almost_equal(cov(x, rowvar=False),
                            (np.cov(xf, rowvar=False) *
                             (x.shape[0] - 1) / (frac - 1.)))
        assert_almost_equal(cov(x, rowvar=False, bias=True),
                            (np.cov(xf, rowvar=False, bias=True) *
                             x.shape[0] / frac))


class TestCorrcoef:

    def setup_method(self):
        self.data = array(np.random.rand(12))
        self.data2 = array(np.random.rand(12))

    def test_ddof(self):
        # ddof raises DeprecationWarning
        x, y = self.data, self.data2
        expected = np.corrcoef(x)
        expected2 = np.corrcoef(x, y)
        with suppress_warnings() as sup:
            warnings.simplefilter("always")
            assert_warns(DeprecationWarning, corrcoef, x, ddof=-1)
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            # ddof has no or negligible effect on the function
            assert_almost_equal(np.corrcoef(x, ddof=0), corrcoef(x, ddof=0))
            assert_almost_equal(corrcoef(x, ddof=-1), expected)
            assert_almost_equal(corrcoef(x, y, ddof=-1), expected2)
            assert_almost_equal(corrcoef(x, ddof=3), expected)
            assert_almost_equal(corrcoef(x, y, ddof=3), expected2)

    def test_bias(self):
        x, y = self.data, self.data2
        expected = np.corrcoef(x)
        # bias raises DeprecationWarning
        with suppress_warnings() as sup:
            warnings.simplefilter("always")
            assert_warns(DeprecationWarning, corrcoef, x, y, True, False)
            assert_warns(DeprecationWarning, corrcoef, x, y, True, True)
            assert_warns(DeprecationWarning, corrcoef, x, bias=False)
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            # bias has no or negligible effect on the function
            assert_almost_equal(corrcoef(x, bias=1), expected)

    def test_1d_without_missing(self):
        # Test cov on 1D variable w/o missing values
        x = self.data
        assert_almost_equal(np.corrcoef(x), corrcoef(x))
        assert_almost_equal(np.corrcoef(x, rowvar=False),
                            corrcoef(x, rowvar=False))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            assert_almost_equal(np.corrcoef(x, rowvar=False, bias=True),
                                corrcoef(x, rowvar=False, bias=True))

    def test_2d_without_missing(self):
        # Test corrcoef on 1 2D variable w/o missing values
        x = self.data.reshape(3, 4)
        assert_almost_equal(np.corrcoef(x), corrcoef(x))
        assert_almost_equal(np.corrcoef(x, rowvar=False),
                            corrcoef(x, rowvar=False))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            assert_almost_equal(np.corrcoef(x, rowvar=False, bias=True),
                                corrcoef(x, rowvar=False, bias=True))

    def test_1d_with_missing(self):
        # Test corrcoef 1 1D variable w/missing values
        x = self.data
        x[-1] = masked
        x -= x.mean()
        nx = x.compressed()
        assert_almost_equal(np.corrcoef(nx), corrcoef(x))
        assert_almost_equal(np.corrcoef(nx, rowvar=False),
                            corrcoef(x, rowvar=False))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            assert_almost_equal(np.corrcoef(nx, rowvar=False, bias=True),
                                corrcoef(x, rowvar=False, bias=True))
        try:
            corrcoef(x, allow_masked=False)
        except ValueError:
            pass
        # 2 1D variables w/ missing values
        nx = x[1:-1]
        assert_almost_equal(np.corrcoef(nx, nx[::-1]), corrcoef(x, x[::-1]))
        assert_almost_equal(np.corrcoef(nx, nx[::-1], rowvar=False),
                            corrcoef(x, x[::-1], rowvar=False))
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            # ddof and bias have no or negligible effect on the function
            assert_almost_equal(np.corrcoef(nx, nx[::-1]),
                                corrcoef(x, x[::-1], bias=1))
            assert_almost_equal(np.corrcoef(nx, nx[::-1]),
                                corrcoef(x, x[::-1], ddof=2))

    def test_2d_with_missing(self):
        # Test corrcoef on 2D variable w/ missing value
        x = self.data
        x[-1] = masked
        x = x.reshape(3, 4)

        test = corrcoef(x)
        control = np.corrcoef(x)
        assert_almost_equal(test[:-1, :-1], control[:-1, :-1])
        with suppress_warnings() as sup:
            sup.filter(DeprecationWarning, "bias and ddof have no effect")
            # ddof and bias have no or negligible effect on the function
            assert_almost_equal(corrcoef(x, ddof=-2)[:-1, :-1],
                                control[:-1, :-1])
            assert_almost_equal(corrcoef(x, ddof=3)[:-1, :-1],
                                control[:-1, :-1])
            assert_almost_equal(corrcoef(x, bias=1)[:-1, :-1],
                                control[:-1, :-1])


class TestPolynomial:
    #
    def test_polyfit(self):
        # Tests polyfit
        # On ndarrays
        x = np.random.rand(10)
        y = np.random.rand(20).reshape(-1, 2)
        assert_almost_equal(polyfit(x, y, 3), np.polyfit(x, y, 3))
        # ON 1D maskedarrays
        x = x.view(MaskedArray)
        x[0] = masked
        y = y.view(MaskedArray)
        y[0, 0] = y[-1, -1] = masked
        #
        (C, R, K, S, D) = polyfit(x, y[:, 0], 3, full=True)
        (c, r, k, s, d) = np.polyfit(x[1:], y[1:, 0].compressed(), 3,
                                     full=True)
        for (a, a_) in zip((C, R, K, S, D), (c, r, k, s, d)):
            assert_almost_equal(a, a_)
        #
        (C, R, K, S, D) = polyfit(x, y[:, -1], 3, full=True)
        (c, r, k, s, d) = np.polyfit(x[1:-1], y[1:-1, -1], 3, full=True)
        for (a, a_) in zip((C, R, K, S, D), (c, r, k, s, d)):
            assert_almost_equal(a, a_)
        #
        (C, R, K, S, D) = polyfit(x, y, 3, full=True)
        (c, r, k, s, d) = np.polyfit(x[1:-1], y[1:-1,:], 3, full=True)
        for (a, a_) in zip((C, R, K, S, D), (c, r, k, s, d)):
            assert_almost_equal(a, a_)
        #
        w = np.random.rand(10) + 1
        wo = w.copy()
        xs = x[1:-1]
        ys = y[1:-1]
        ws = w[1:-1]
        (C, R, K, S, D) = polyfit(x, y, 3, full=True, w=w)
        (c, r, k, s, d) = np.polyfit(xs, ys, 3, full=True, w=ws)
        assert_equal(w, wo)
        for (a, a_) in zip((C, R, K, S, D), (c, r, k, s, d)):
            assert_almost_equal(a, a_)

    def test_polyfit_with_masked_NaNs(self):
        x = np.random.rand(10)
        y = np.random.rand(20).reshape(-1, 2)

        x[0] = np.nan
        y[-1,-1] = np.nan
        x = x.view(MaskedArray)
        y = y.view(MaskedArray)
        x[0] = masked
        y[-1,-1] = masked

        (C, R, K, S, D) = polyfit(x, y, 3, full=True)
        (c, r, k, s, d) = np.polyfit(x[1:-1], y[1:-1,:], 3, full=True)
        for (a, a_) in zip((C, R, K, S, D), (c, r, k, s, d)):
            assert_almost_equal(a, a_)


class TestArraySetOps:

    def test_unique_onlist(self):
        # Test unique on list
        data = [1, 1, 1, 2, 2, 3]
        test = unique(data, return_index=True, return_inverse=True)
        assert_(isinstance(test[0], MaskedArray))
        assert_equal(test[0], masked_array([1, 2, 3], mask=[0, 0, 0]))
        assert_equal(test[1], [0, 3, 5])
        assert_equal(test[2], [0, 0, 0, 1, 1, 2])

    def test_unique_onmaskedarray(self):
        # Test unique on masked data w/use_mask=True
        data = masked_array([1, 1, 1, 2, 2, 3], mask=[0, 0, 1, 0, 1, 0])
        test = unique(data, return_index=True, return_inverse=True)
        assert_equal(test[0], masked_array([1, 2, 3, -1], mask=[0, 0, 0, 1]))
        assert_equal(test[1], [0, 3, 5, 2])
        assert_equal(test[2], [0, 0, 3, 1, 3, 2])
        #
        data.fill_value = 3
        data = masked_array(data=[1, 1, 1, 2, 2, 3],
                            mask=[0, 0, 1, 0, 1, 0], fill_value=3)
        test = unique(data, return_index=True, return_inverse=True)
        assert_equal(test[0], masked_array([1, 2, 3, -1], mask=[0, 0, 0, 1]))
        assert_equal(test[1], [0, 3, 5, 2])
        assert_equal(test[2], [0, 0, 3, 1, 3, 2])

    def test_unique_allmasked(self):
        # Test all masked
        data = masked_array([1, 1, 1], mask=True)
        test = unique(data, return_index=True, return_inverse=True)
        assert_equal(test[0], masked_array([1, ], mask=[True]))
        assert_equal(test[1], [0])
        assert_equal(test[2], [0, 0, 0])
        #
        # Test masked
        data = masked
        test = unique(data, return_index=True, return_inverse=True)
        assert_equal(test[0], masked_array(masked))
        assert_equal(test[1], [0])
        assert_equal(test[2], [0])

    def test_ediff1d(self):
        # Tests mediff1d
        x = masked_array(np.arange(5), mask=[1, 0, 0, 0, 1])
        control = array([1, 1, 1, 4], mask=[1, 0, 0, 1])
        test = ediff1d(x)
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_ediff1d_tobegin(self):
        # Test ediff1d w/ to_begin
        x = masked_array(np.arange(5), mask=[1, 0, 0, 0, 1])
        test = ediff1d(x, to_begin=masked)
        control = array([0, 1, 1, 1, 4], mask=[1, 1, 0, 0, 1])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)
        #
        test = ediff1d(x, to_begin=[1, 2, 3])
        control = array([1, 2, 3, 1, 1, 1, 4], mask=[0, 0, 0, 1, 0, 0, 1])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_ediff1d_toend(self):
        # Test ediff1d w/ to_end
        x = masked_array(np.arange(5), mask=[1, 0, 0, 0, 1])
        test = ediff1d(x, to_end=masked)
        control = array([1, 1, 1, 4, 0], mask=[1, 0, 0, 1, 1])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)
        #
        test = ediff1d(x, to_end=[1, 2, 3])
        control = array([1, 1, 1, 4, 1, 2, 3], mask=[1, 0, 0, 1, 0, 0, 0])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_ediff1d_tobegin_toend(self):
        # Test ediff1d w/ to_begin and to_end
        x = masked_array(np.arange(5), mask=[1, 0, 0, 0, 1])
        test = ediff1d(x, to_end=masked, to_begin=masked)
        control = array([0, 1, 1, 1, 4, 0], mask=[1, 1, 0, 0, 1, 1])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)
        #
        test = ediff1d(x, to_end=[1, 2, 3], to_begin=masked)
        control = array([0, 1, 1, 1, 4, 1, 2, 3],
                        mask=[1, 1, 0, 0, 1, 0, 0, 0])
        assert_equal(test, control)
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_ediff1d_ndarray(self):
        # Test ediff1d w/ a ndarray
        x = np.arange(5)
        test = ediff1d(x)
        control = array([1, 1, 1, 1], mask=[0, 0, 0, 0])
        assert_equal(test, control)
        assert_(isinstance(test, MaskedArray))
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)
        #
        test = ediff1d(x, to_end=masked, to_begin=masked)
        control = array([0, 1, 1, 1, 1, 0], mask=[1, 0, 0, 0, 0, 1])
        assert_(isinstance(test, MaskedArray))
        assert_equal(test.filled(0), control.filled(0))
        assert_equal(test.mask, control.mask)

    def test_intersect1d(self):
        # Test intersect1d
        x = array([1, 3, 3, 3], mask=[0, 0, 0, 1])
        y = array([3, 1, 1, 1], mask=[0, 0, 0, 1])
        test = intersect1d(x, y)
        control = array([1, 3, -1], mask=[0, 0, 1])
        assert_equal(test, control)

    def test_setxor1d(self):
        # Test setxor1d
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 2, 3, 4, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        test = setxor1d(a, b)
        assert_equal(test, array([3, 4, 7]))
        #
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = [1, 2, 3, 4, 5]
        test = setxor1d(a, b)
        assert_equal(test, array([3, 4, 7, -1], mask=[0, 0, 0, 1]))
        #
        a = array([1, 2, 3])
        b = array([6, 5, 4])
        test = setxor1d(a, b)
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, [1, 2, 3, 4, 5, 6])
        #
        a = array([1, 8, 2, 3], mask=[0, 1, 0, 0])
        b = array([6, 5, 4, 8], mask=[0, 0, 0, 1])
        test = setxor1d(a, b)
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, [1, 2, 3, 4, 5, 6])
        #
        assert_array_equal([], setxor1d([], []))

    def test_isin(self):
        # the tests for in1d cover most of isin's behavior
        # if in1d is removed, would need to change those tests to test
        # isin instead.
        a = np.arange(24).reshape([2, 3, 4])
        mask = np.zeros([2, 3, 4])
        mask[1, 2, 0] = 1
        a = array(a, mask=mask)
        b = array(data=[0, 10, 20, 30,  1,  3, 11, 22, 33],
                  mask=[0,  1,  0,  1,  0,  1,  0,  1,  0])
        ec = zeros((2, 3, 4), dtype=bool)
        ec[0, 0, 0] = True
        ec[0, 0, 1] = True
        ec[0, 2, 3] = True
        c = isin(a, b)
        assert_(isinstance(c, MaskedArray))
        assert_array_equal(c, ec)
        #compare results of np.isin to ma.isin
        d = np.isin(a, b[~b.mask]) & ~a.mask
        assert_array_equal(c, d)

    def test_in1d(self):
        # Test in1d
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 2, 3, 4, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        test = in1d(a, b)
        assert_equal(test, [True, True, True, False, True])
        #
        a = array([5, 5, 2, 1, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 5, -1], mask=[0, 0, 1])
        test = in1d(a, b)
        assert_equal(test, [True, True, False, True, True])
        #
        assert_array_equal([], in1d([], []))

    def test_in1d_invert(self):
        # Test in1d's invert parameter
        a = array([1, 2, 5, 7, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 2, 3, 4, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        assert_equal(np.invert(in1d(a, b)), in1d(a, b, invert=True))

        a = array([5, 5, 2, 1, -1], mask=[0, 0, 0, 0, 1])
        b = array([1, 5, -1], mask=[0, 0, 1])
        assert_equal(np.invert(in1d(a, b)), in1d(a, b, invert=True))

        assert_array_equal([], in1d([], [], invert=True))

    def test_union1d(self):
        # Test union1d
        a = array([1, 2, 5, 7, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        b = array([1, 2, 3, 4, 5, -1], mask=[0, 0, 0, 0, 0, 1])
        test = union1d(a, b)
        control = array([1, 2, 3, 4, 5, 7, -1], mask=[0, 0, 0, 0, 0, 0, 1])
        assert_equal(test, control)

        # Tests gh-10340, arguments to union1d should be
        # flattened if they are not already 1D
        x = array([[0, 1, 2], [3, 4, 5]], mask=[[0, 0, 0], [0, 0, 1]])
        y = array([0, 1, 2, 3, 4], mask=[0, 0, 0, 0, 1])
        ez = array([0, 1, 2, 3, 4, 5], mask=[0, 0, 0, 0, 0, 1])
        z = union1d(x, y)
        assert_equal(z, ez)
        #
        assert_array_equal([], union1d([], []))

    def test_setdiff1d(self):
        # Test setdiff1d
        a = array([6, 5, 4, 7, 7, 1, 2, 1], mask=[0, 0, 0, 0, 0, 0, 0, 1])
        b = array([2, 4, 3, 3, 2, 1, 5])
        test = setdiff1d(a, b)
        assert_equal(test, array([6, 7, -1], mask=[0, 0, 1]))
        #
        a = arange(10)
        b = arange(8)
        assert_equal(setdiff1d(a, b), array([8, 9]))
        a = array([], np.uint32, mask=[])
        assert_equal(setdiff1d(a, []).dtype, np.uint32)

    def test_setdiff1d_char_array(self):
        # Test setdiff1d_charray
        a = np.array(['a', 'b', 'c'])
        b = np.array(['a', 'b', 's'])
        assert_array_equal(setdiff1d(a, b), np.array(['c']))


class TestShapeBase:

    def test_atleast_2d(self):
        # Test atleast_2d
        a = masked_array([0, 1, 2], mask=[0, 1, 0])
        b = atleast_2d(a)
        assert_equal(b.shape, (1, 3))
        assert_equal(b.mask.shape, b.data.shape)
        assert_equal(a.shape, (3,))
        assert_equal(a.mask.shape, a.data.shape)
        assert_equal(b.mask.shape, b.data.shape)

    def test_shape_scalar(self):
        # the atleast and diagflat function should work with scalars
        # GitHub issue #3367
        # Additionally, the atleast functions should accept multiple scalars
        # correctly
        b = atleast_1d(1.0)
        assert_equal(b.shape, (1,))
        assert_equal(b.mask.shape, b.shape)
        assert_equal(b.data.shape, b.shape)

        b = atleast_1d(1.0, 2.0)
        for a in b:
            assert_equal(a.shape, (1,))
            assert_equal(a.mask.shape, a.shape)
            assert_equal(a.data.shape, a.shape)

        b = atleast_2d(1.0)
        assert_equal(b.shape, (1, 1))
        assert_equal(b.mask.shape, b.shape)
        assert_equal(b.data.shape, b.shape)

        b = atleast_2d(1.0, 2.0)
        for a in b:
            assert_equal(a.shape, (1, 1))
            assert_equal(a.mask.shape, a.shape)
            assert_equal(a.data.shape, a.shape)

        b = atleast_3d(1.0)
        assert_equal(b.shape, (1, 1, 1))
        assert_equal(b.mask.shape, b.shape)
        assert_equal(b.data.shape, b.shape)

        b = atleast_3d(1.0, 2.0)
        for a in b:
            assert_equal(a.shape, (1, 1, 1))
            assert_equal(a.mask.shape, a.shape)
            assert_equal(a.data.shape, a.shape)

        b = diagflat(1.0)
        assert_equal(b.shape, (1, 1))
        assert_equal(b.mask.shape, b.data.shape)


class TestNDEnumerate:

    def test_ndenumerate_nomasked(self):
        ordinary = np.arange(6.).reshape((1, 3, 2))
        empty_mask = np.zeros_like(ordinary, dtype=bool)
        with_mask = masked_array(ordinary, mask=empty_mask)
        assert_equal(list(np.ndenumerate(ordinary)),
                     list(ndenumerate(ordinary)))
        assert_equal(list(ndenumerate(ordinary)),
                     list(ndenumerate(with_mask)))
        assert_equal(list(ndenumerate(with_mask)),
                     list(ndenumerate(with_mask, compressed=False)))

    def test_ndenumerate_allmasked(self):
        a = masked_all(())
        b = masked_all((100,))
        c = masked_all((2, 3, 4))
        assert_equal(list(ndenumerate(a)), [])
        assert_equal(list(ndenumerate(b)), [])
        assert_equal(list(ndenumerate(b, compressed=False)),
                     list(zip(np.ndindex((100,)), 100 * [masked])))
        assert_equal(list(ndenumerate(c)), [])
        assert_equal(list(ndenumerate(c, compressed=False)),
                     list(zip(np.ndindex((2, 3, 4)), 2 * 3 * 4 * [masked])))

    def test_ndenumerate_mixedmasked(self):
        a = masked_array(np.arange(12).reshape((3, 4)),
                         mask=[[1, 1, 1, 1],
                               [1, 1, 0, 1],
                               [0, 0, 0, 0]])
        items = [((1, 2), 6),
                 ((2, 0), 8), ((2, 1), 9), ((2, 2), 10), ((2, 3), 11)]
        assert_equal(list(ndenumerate(a)), items)
        assert_equal(len(list(ndenumerate(a, compressed=False))), a.size)
        for coordinate, value in ndenumerate(a, compressed=False):
            assert_equal(a[coordinate], value)


class TestStack:

    def test_stack_1d(self):
        a = masked_array([0, 1, 2], mask=[0, 1, 0])
        b = masked_array([9, 8, 7], mask=[1, 0, 0])

        c = stack([a, b], axis=0)
        assert_equal(c.shape, (2, 3))
        assert_array_equal(a.mask, c[0].mask)
        assert_array_equal(b.mask, c[1].mask)

        d = vstack([a, b])
        assert_array_equal(c.data, d.data)
        assert_array_equal(c.mask, d.mask)

        c = stack([a, b], axis=1)
        assert_equal(c.shape, (3, 2))
        assert_array_equal(a.mask, c[:, 0].mask)
        assert_array_equal(b.mask, c[:, 1].mask)

    def test_stack_masks(self):
        a = masked_array([0, 1, 2], mask=True)
        b = masked_array([9, 8, 7], mask=False)

        c = stack([a, b], axis=0)
        assert_equal(c.shape, (2, 3))
        assert_array_equal(a.mask, c[0].mask)
        assert_array_equal(b.mask, c[1].mask)

        d = vstack([a, b])
        assert_array_equal(c.data, d.data)
        assert_array_equal(c.mask, d.mask)

        c = stack([a, b], axis=1)
        assert_equal(c.shape, (3, 2))
        assert_array_equal(a.mask, c[:, 0].mask)
        assert_array_equal(b.mask, c[:, 1].mask)

    def test_stack_nd(self):
        # 2D
        shp = (3, 2)
        d1 = np.random.randint(0, 10, shp)
        d2 = np.random.randint(0, 10, shp)
        m1 = np.random.randint(0, 2, shp).astype(bool)
        m2 = np.random.randint(0, 2, shp).astype(bool)
        a1 = masked_array(d1, mask=m1)
        a2 = masked_array(d2, mask=m2)

        c = stack([a1, a2], axis=0)
        c_shp = (2,) + shp
        assert_equal(c.shape, c_shp)
        assert_array_equal(a1.mask, c[0].mask)
        assert_array_equal(a2.mask, c[1].mask)

        c = stack([a1, a2], axis=-1)
        c_shp = shp + (2,)
        assert_equal(c.shape, c_shp)
        assert_array_equal(a1.mask, c[..., 0].mask)
        assert_array_equal(a2.mask, c[..., 1].mask)

        # 4D
        shp = (3, 2, 4, 5,)
        d1 = np.random.randint(0, 10, shp)
        d2 = np.random.randint(0, 10, shp)
        m1 = np.random.randint(0, 2, shp).astype(bool)
        m2 = np.random.randint(0, 2, shp).astype(bool)
        a1 = masked_array(d1, mask=m1)
        a2 = masked_array(d2, mask=m2)

        c = stack([a1, a2], axis=0)
        c_shp = (2,) + shp
        assert_equal(c.shape, c_shp)
        assert_array_equal(a1.mask, c[0].mask)
        assert_array_equal(a2.mask, c[1].mask)

        c = stack([a1, a2], axis=-1)
        c_shp = shp + (2,)
        assert_equal(c.shape, c_shp)
        assert_array_equal(a1.mask, c[..., 0].mask)
        assert_array_equal(a2.mask, c[..., 1].mask)
