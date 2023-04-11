"""Tests of interaction of matrix with other parts of numpy.

Note that tests with MaskedArray and linalg are done in separate files.
"""
import pytest

import textwrap
import warnings

import numpy as np
from numpy.testing import (assert_, assert_equal, assert_raises,
                           assert_raises_regex, assert_array_equal,
                           assert_almost_equal, assert_array_almost_equal)


def test_fancy_indexing():
    # The matrix class messes with the shape. While this is always
    # weird (getitem is not used, it does not have setitem nor knows
    # about fancy indexing), this tests gh-3110
    # 2018-04-29: moved here from core.tests.test_index.
    m = np.matrix([[1, 2], [3, 4]])

    assert_(isinstance(m[[0, 1, 0], :], np.matrix))

    # gh-3110. Note the transpose currently because matrices do *not*
    # support dimension fixing for fancy indexing correctly.
    x = np.asmatrix(np.arange(50).reshape(5, 10))
    assert_equal(x[:2, np.array(-1)], x[:2, -1].T)


def test_polynomial_mapdomain():
    # test that polynomial preserved matrix subtype.
    # 2018-04-29: moved here from polynomial.tests.polyutils.
    dom1 = [0, 4]
    dom2 = [1, 3]
    x = np.matrix([dom1, dom1])
    res = np.polynomial.polyutils.mapdomain(x, dom1, dom2)
    assert_(isinstance(res, np.matrix))


def test_sort_matrix_none():
    # 2018-04-29: moved here from core.tests.test_multiarray
    a = np.matrix([[2, 1, 0]])
    actual = np.sort(a, axis=None)
    expected = np.matrix([[0, 1, 2]])
    assert_equal(actual, expected)
    assert_(type(expected) is np.matrix)


def test_partition_matrix_none():
    # gh-4301
    # 2018-04-29: moved here from core.tests.test_multiarray
    a = np.matrix([[2, 1, 0]])
    actual = np.partition(a, 1, axis=None)
    expected = np.matrix([[0, 1, 2]])
    assert_equal(actual, expected)
    assert_(type(expected) is np.matrix)


def test_dot_scalar_and_matrix_of_objects():
    # Ticket #2469
    # 2018-04-29: moved here from core.tests.test_multiarray
    arr = np.matrix([1, 2], dtype=object)
    desired = np.matrix([[3, 6]], dtype=object)
    assert_equal(np.dot(arr, 3), desired)
    assert_equal(np.dot(3, arr), desired)


def test_inner_scalar_and_matrix():
    # 2018-04-29: moved here from core.tests.test_multiarray
    for dt in np.typecodes['AllInteger'] + np.typecodes['AllFloat'] + '?':
        sca = np.array(3, dtype=dt)[()]
        arr = np.matrix([[1, 2], [3, 4]], dtype=dt)
        desired = np.matrix([[3, 6], [9, 12]], dtype=dt)
        assert_equal(np.inner(arr, sca), desired)
        assert_equal(np.inner(sca, arr), desired)


def test_inner_scalar_and_matrix_of_objects():
    # Ticket #4482
    # 2018-04-29: moved here from core.tests.test_multiarray
    arr = np.matrix([1, 2], dtype=object)
    desired = np.matrix([[3, 6]], dtype=object)
    assert_equal(np.inner(arr, 3), desired)
    assert_equal(np.inner(3, arr), desired)


def test_iter_allocate_output_subtype():
    # Make sure that the subtype with priority wins
    # 2018-04-29: moved here from core.tests.test_nditer, given the
    # matrix specific shape test.

    # matrix vs ndarray
    a = np.matrix([[1, 2], [3, 4]])
    b = np.arange(4).reshape(2, 2).T
    i = np.nditer([a, b, None], [],
                  [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    assert_(type(i.operands[2]) is np.matrix)
    assert_(type(i.operands[2]) is not np.ndarray)
    assert_equal(i.operands[2].shape, (2, 2))

    # matrix always wants things to be 2D
    b = np.arange(4).reshape(1, 2, 2)
    assert_raises(RuntimeError, np.nditer, [a, b, None], [],
                  [['readonly'], ['readonly'], ['writeonly', 'allocate']])
    # but if subtypes are disabled, the result can still work
    i = np.nditer([a, b, None], [],
                  [['readonly'], ['readonly'],
                   ['writeonly', 'allocate', 'no_subtype']])
    assert_(type(i.operands[2]) is np.ndarray)
    assert_(type(i.operands[2]) is not np.matrix)
    assert_equal(i.operands[2].shape, (1, 2, 2))


def like_function():
    # 2018-04-29: moved here from core.tests.test_numeric
    a = np.matrix([[1, 2], [3, 4]])
    for like_function in np.zeros_like, np.ones_like, np.empty_like:
        b = like_function(a)
        assert_(type(b) is np.matrix)

        c = like_function(a, subok=False)
        assert_(type(c) is not np.matrix)


def test_array_astype():
    # 2018-04-29: copied here from core.tests.test_api
    # subok=True passes through a matrix
    a = np.matrix([[0, 1, 2], [3, 4, 5]], dtype='f4')
    b = a.astype('f4', subok=True, copy=False)
    assert_(a is b)

    # subok=True is default, and creates a subtype on a cast
    b = a.astype('i4', copy=False)
    assert_equal(a, b)
    assert_equal(type(b), np.matrix)

    # subok=False never returns a matrix
    b = a.astype('f4', subok=False, copy=False)
    assert_equal(a, b)
    assert_(not (a is b))
    assert_(type(b) is not np.matrix)


def test_stack():
    # 2018-04-29: copied here from core.tests.test_shape_base
    # check np.matrix cannot be stacked
    m = np.matrix([[1, 2], [3, 4]])
    assert_raises_regex(ValueError, 'shape too large to be a matrix',
                        np.stack, [m, m])


def test_object_scalar_multiply():
    # Tickets #2469 and #4482
    # 2018-04-29: moved here from core.tests.test_ufunc
    arr = np.matrix([1, 2], dtype=object)
    desired = np.matrix([[3, 6]], dtype=object)
    assert_equal(np.multiply(arr, 3), desired)
    assert_equal(np.multiply(3, arr), desired)


def test_nanfunctions_matrices():
    # Check that it works and that type and
    # shape are preserved
    # 2018-04-29: moved here from core.tests.test_nanfunctions
    mat = np.matrix(np.eye(3))
    for f in [np.nanmin, np.nanmax]:
        res = f(mat, axis=0)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (1, 3))
        res = f(mat, axis=1)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (3, 1))
        res = f(mat)
        assert_(np.isscalar(res))
    # check that rows of nan are dealt with for subclasses (#4628)
    mat[1] = np.nan
    for f in [np.nanmin, np.nanmax]:
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            res = f(mat, axis=0)
            assert_(isinstance(res, np.matrix))
            assert_(not np.any(np.isnan(res)))
            assert_(len(w) == 0)

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            res = f(mat, axis=1)
            assert_(isinstance(res, np.matrix))
            assert_(np.isnan(res[1, 0]) and not np.isnan(res[0, 0])
                    and not np.isnan(res[2, 0]))
            assert_(len(w) == 1, 'no warning raised')
            assert_(issubclass(w[0].category, RuntimeWarning))

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter('always')
            res = f(mat)
            assert_(np.isscalar(res))
            assert_(res != np.nan)
            assert_(len(w) == 0)


def test_nanfunctions_matrices_general():
    # Check that it works and that type and
    # shape are preserved
    # 2018-04-29: moved here from core.tests.test_nanfunctions
    mat = np.matrix(np.eye(3))
    for f in (np.nanargmin, np.nanargmax, np.nansum, np.nanprod,
              np.nanmean, np.nanvar, np.nanstd):
        res = f(mat, axis=0)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (1, 3))
        res = f(mat, axis=1)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (3, 1))
        res = f(mat)
        assert_(np.isscalar(res))

    for f in np.nancumsum, np.nancumprod:
        res = f(mat, axis=0)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (3, 3))
        res = f(mat, axis=1)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (3, 3))
        res = f(mat)
        assert_(isinstance(res, np.matrix))
        assert_(res.shape == (1, 3*3))


def test_average_matrix():
    # 2018-04-29: moved here from core.tests.test_function_base.
    y = np.matrix(np.random.rand(5, 5))
    assert_array_equal(y.mean(0), np.average(y, 0))

    a = np.matrix([[1, 2], [3, 4]])
    w = np.matrix([[1, 2], [3, 4]])

    r = np.average(a, axis=0, weights=w)
    assert_equal(type(r), np.matrix)
    assert_equal(r, [[2.5, 10.0/3]])


def test_trapz_matrix():
    # Test to make sure matrices give the same answer as ndarrays
    # 2018-04-29: moved here from core.tests.test_function_base.
    x = np.linspace(0, 5)
    y = x * x
    r = np.trapz(y, x)
    mx = np.matrix(x)
    my = np.matrix(y)
    mr = np.trapz(my, mx)
    assert_almost_equal(mr, r)


def test_ediff1d_matrix():
    # 2018-04-29: moved here from core.tests.test_arraysetops.
    assert(isinstance(np.ediff1d(np.matrix(1)), np.matrix))
    assert(isinstance(np.ediff1d(np.matrix(1), to_begin=1), np.matrix))


def test_apply_along_axis_matrix():
    # this test is particularly malicious because matrix
    # refuses to become 1d
    # 2018-04-29: moved here from core.tests.test_shape_base.
    def double(row):
        return row * 2

    m = np.matrix([[0, 1], [2, 3]])
    expected = np.matrix([[0, 2], [4, 6]])

    result = np.apply_along_axis(double, 0, m)
    assert_(isinstance(result, np.matrix))
    assert_array_equal(result, expected)

    result = np.apply_along_axis(double, 1, m)
    assert_(isinstance(result, np.matrix))
    assert_array_equal(result, expected)


def test_kron_matrix():
    # 2018-04-29: moved here from core.tests.test_shape_base.
    a = np.ones([2, 2])
    m = np.asmatrix(a)
    assert_equal(type(np.kron(a, a)), np.ndarray)
    assert_equal(type(np.kron(m, m)), np.matrix)
    assert_equal(type(np.kron(a, m)), np.matrix)
    assert_equal(type(np.kron(m, a)), np.matrix)


class TestConcatenatorMatrix:
    # 2018-04-29: moved here from core.tests.test_index_tricks.
    def test_matrix(self):
        a = [1, 2]
        b = [3, 4]

        ab_r = np.r_['r', a, b]
        ab_c = np.r_['c', a, b]

        assert_equal(type(ab_r), np.matrix)
        assert_equal(type(ab_c), np.matrix)

        assert_equal(np.array(ab_r), [[1, 2, 3, 4]])
        assert_equal(np.array(ab_c), [[1], [2], [3], [4]])

        assert_raises(ValueError, lambda: np.r_['rc', a, b])

    def test_matrix_scalar(self):
        r = np.r_['r', [1, 2], 3]
        assert_equal(type(r), np.matrix)
        assert_equal(np.array(r), [[1, 2, 3]])

    def test_matrix_builder(self):
        a = np.array([1])
        b = np.array([2])
        c = np.array([3])
        d = np.array([4])
        actual = np.r_['a, b; c, d']
        expected = np.bmat([[a, b], [c, d]])

        assert_equal(actual, expected)
        assert_equal(type(actual), type(expected))


def test_array_equal_error_message_matrix():
    # 2018-04-29: moved here from testing.tests.test_utils.
    with pytest.raises(AssertionError) as exc_info:
        assert_equal(np.array([1, 2]), np.matrix([1, 2]))
    msg = str(exc_info.value)
    msg_reference = textwrap.dedent("""\

    Arrays are not equal

    (shapes (2,), (1, 2) mismatch)
     x: array([1, 2])
     y: matrix([[1, 2]])""")
    assert_equal(msg, msg_reference)


def test_array_almost_equal_matrix():
    # Matrix slicing keeps things 2-D, while array does not necessarily.
    # See gh-8452.
    # 2018-04-29: moved here from testing.tests.test_utils.
    m1 = np.matrix([[1., 2.]])
    m2 = np.matrix([[1., np.nan]])
    m3 = np.matrix([[1., -np.inf]])
    m4 = np.matrix([[np.nan, np.inf]])
    m5 = np.matrix([[1., 2.], [np.nan, np.inf]])
    for assert_func in assert_array_almost_equal, assert_almost_equal:
        for m in m1, m2, m3, m4, m5:
            assert_func(m, m)
            a = np.array(m)
            assert_func(a, m)
            assert_func(m, a)
