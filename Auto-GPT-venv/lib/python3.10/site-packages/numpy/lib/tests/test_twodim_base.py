"""Test functions for matrix module

"""
from numpy.testing import (
    assert_equal, assert_array_equal, assert_array_max_ulp,
    assert_array_almost_equal, assert_raises, assert_
    )

from numpy import (
    arange, add, fliplr, flipud, zeros, ones, eye, array, diag, histogram2d,
    tri, mask_indices, triu_indices, triu_indices_from, tril_indices,
    tril_indices_from, vander,
    )

import numpy as np


from numpy.core.tests.test_overrides import requires_array_function


import pytest


def get_mat(n):
    data = arange(n)
    data = add.outer(data, data)
    return data


class TestEye:
    def test_basic(self):
        assert_equal(eye(4),
                     array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]]))

        assert_equal(eye(4, dtype='f'),
                     array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1]], 'f'))

        assert_equal(eye(3) == 1,
                     eye(3, dtype=bool))

    def test_uint64(self):
        # Regression test for gh-9982
        assert_equal(eye(np.uint64(2), dtype=int), array([[1, 0], [0, 1]]))
        assert_equal(eye(np.uint64(2), M=np.uint64(4), k=np.uint64(1)),
                     array([[0, 1, 0, 0], [0, 0, 1, 0]]))

    def test_diag(self):
        assert_equal(eye(4, k=1),
                     array([[0, 1, 0, 0],
                            [0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0]]))

        assert_equal(eye(4, k=-1),
                     array([[0, 0, 0, 0],
                            [1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0]]))

    def test_2d(self):
        assert_equal(eye(4, 3),
                     array([[1, 0, 0],
                            [0, 1, 0],
                            [0, 0, 1],
                            [0, 0, 0]]))

        assert_equal(eye(3, 4),
                     array([[1, 0, 0, 0],
                            [0, 1, 0, 0],
                            [0, 0, 1, 0]]))

    def test_diag2d(self):
        assert_equal(eye(3, 4, k=2),
                     array([[0, 0, 1, 0],
                            [0, 0, 0, 1],
                            [0, 0, 0, 0]]))

        assert_equal(eye(4, 3, k=-2),
                     array([[0, 0, 0],
                            [0, 0, 0],
                            [1, 0, 0],
                            [0, 1, 0]]))

    def test_eye_bounds(self):
        assert_equal(eye(2, 2, 1), [[0, 1], [0, 0]])
        assert_equal(eye(2, 2, -1), [[0, 0], [1, 0]])
        assert_equal(eye(2, 2, 2), [[0, 0], [0, 0]])
        assert_equal(eye(2, 2, -2), [[0, 0], [0, 0]])
        assert_equal(eye(3, 2, 2), [[0, 0], [0, 0], [0, 0]])
        assert_equal(eye(3, 2, 1), [[0, 1], [0, 0], [0, 0]])
        assert_equal(eye(3, 2, -1), [[0, 0], [1, 0], [0, 1]])
        assert_equal(eye(3, 2, -2), [[0, 0], [0, 0], [1, 0]])
        assert_equal(eye(3, 2, -3), [[0, 0], [0, 0], [0, 0]])

    def test_strings(self):
        assert_equal(eye(2, 2, dtype='S3'),
                     [[b'1', b''], [b'', b'1']])

    def test_bool(self):
        assert_equal(eye(2, 2, dtype=bool), [[True, False], [False, True]])

    def test_order(self):
        mat_c = eye(4, 3, k=-1)
        mat_f = eye(4, 3, k=-1, order='F')
        assert_equal(mat_c, mat_f)
        assert mat_c.flags.c_contiguous
        assert not mat_c.flags.f_contiguous
        assert not mat_f.flags.c_contiguous
        assert mat_f.flags.f_contiguous


class TestDiag:
    def test_vector(self):
        vals = (100 * arange(5)).astype('l')
        b = zeros((5, 5))
        for k in range(5):
            b[k, k] = vals[k]
        assert_equal(diag(vals), b)
        b = zeros((7, 7))
        c = b.copy()
        for k in range(5):
            b[k, k + 2] = vals[k]
            c[k + 2, k] = vals[k]
        assert_equal(diag(vals, k=2), b)
        assert_equal(diag(vals, k=-2), c)

    def test_matrix(self, vals=None):
        if vals is None:
            vals = (100 * get_mat(5) + 1).astype('l')
        b = zeros((5,))
        for k in range(5):
            b[k] = vals[k, k]
        assert_equal(diag(vals), b)
        b = b * 0
        for k in range(3):
            b[k] = vals[k, k + 2]
        assert_equal(diag(vals, 2), b[:3])
        for k in range(3):
            b[k] = vals[k + 2, k]
        assert_equal(diag(vals, -2), b[:3])

    def test_fortran_order(self):
        vals = array((100 * get_mat(5) + 1), order='F', dtype='l')
        self.test_matrix(vals)

    def test_diag_bounds(self):
        A = [[1, 2], [3, 4], [5, 6]]
        assert_equal(diag(A, k=2), [])
        assert_equal(diag(A, k=1), [2])
        assert_equal(diag(A, k=0), [1, 4])
        assert_equal(diag(A, k=-1), [3, 6])
        assert_equal(diag(A, k=-2), [5])
        assert_equal(diag(A, k=-3), [])

    def test_failure(self):
        assert_raises(ValueError, diag, [[[1]]])


class TestFliplr:
    def test_basic(self):
        assert_raises(ValueError, fliplr, ones(4))
        a = get_mat(4)
        b = a[:, ::-1]
        assert_equal(fliplr(a), b)
        a = [[0, 1, 2],
             [3, 4, 5]]
        b = [[2, 1, 0],
             [5, 4, 3]]
        assert_equal(fliplr(a), b)


class TestFlipud:
    def test_basic(self):
        a = get_mat(4)
        b = a[::-1, :]
        assert_equal(flipud(a), b)
        a = [[0, 1, 2],
             [3, 4, 5]]
        b = [[3, 4, 5],
             [0, 1, 2]]
        assert_equal(flipud(a), b)


class TestHistogram2d:
    def test_simple(self):
        x = array(
            [0.41702200, 0.72032449, 1.1437481e-4, 0.302332573, 0.146755891])
        y = array(
            [0.09233859, 0.18626021, 0.34556073, 0.39676747, 0.53881673])
        xedges = np.linspace(0, 1, 10)
        yedges = np.linspace(0, 1, 10)
        H = histogram2d(x, y, (xedges, yedges))[0]
        answer = array(
            [[0, 0, 0, 1, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 1, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [1, 0, 1, 0, 0, 0, 0, 0, 0],
             [0, 1, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0],
             [0, 0, 0, 0, 0, 0, 0, 0, 0]])
        assert_array_equal(H.T, answer)
        H = histogram2d(x, y, xedges)[0]
        assert_array_equal(H.T, answer)
        H, xedges, yedges = histogram2d(list(range(10)), list(range(10)))
        assert_array_equal(H, eye(10, 10))
        assert_array_equal(xedges, np.linspace(0, 9, 11))
        assert_array_equal(yedges, np.linspace(0, 9, 11))

    def test_asym(self):
        x = array([1, 1, 2, 3, 4, 4, 4, 5])
        y = array([1, 3, 2, 0, 1, 2, 3, 4])
        H, xed, yed = histogram2d(
            x, y, (6, 5), range=[[0, 6], [0, 5]], density=True)
        answer = array(
            [[0., 0, 0, 0, 0],
             [0, 1, 0, 1, 0],
             [0, 0, 1, 0, 0],
             [1, 0, 0, 0, 0],
             [0, 1, 1, 1, 0],
             [0, 0, 0, 0, 1]])
        assert_array_almost_equal(H, answer/8., 3)
        assert_array_equal(xed, np.linspace(0, 6, 7))
        assert_array_equal(yed, np.linspace(0, 5, 6))

    def test_density(self):
        x = array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        y = array([1, 1, 1, 2, 2, 2, 3, 3, 3])
        H, xed, yed = histogram2d(
            x, y, [[1, 2, 3, 5], [1, 2, 3, 5]], density=True)
        answer = array([[1, 1, .5],
                        [1, 1, .5],
                        [.5, .5, .25]])/9.
        assert_array_almost_equal(H, answer, 3)

    def test_all_outliers(self):
        r = np.random.rand(100) + 1. + 1e6  # histogramdd rounds by decimal=6
        H, xed, yed = histogram2d(r, r, (4, 5), range=([0, 1], [0, 1]))
        assert_array_equal(H, 0)

    def test_empty(self):
        a, edge1, edge2 = histogram2d([], [], bins=([0, 1], [0, 1]))
        assert_array_max_ulp(a, array([[0.]]))

        a, edge1, edge2 = histogram2d([], [], bins=4)
        assert_array_max_ulp(a, np.zeros((4, 4)))

    def test_binparameter_combination(self):
        x = array(
            [0, 0.09207008, 0.64575234, 0.12875982, 0.47390599,
             0.59944483, 1])
        y = array(
            [0, 0.14344267, 0.48988575, 0.30558665, 0.44700682,
             0.15886423, 1])
        edges = (0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1)
        H, xe, ye = histogram2d(x, y, (edges, 4))
        answer = array(
            [[2., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 1., 0., 0.],
             [1., 0., 0., 0.],
             [0., 1., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 0.],
             [0., 0., 0., 1.]])
        assert_array_equal(H, answer)
        assert_array_equal(ye, array([0., 0.25, 0.5, 0.75, 1]))
        H, xe, ye = histogram2d(x, y, (4, edges))
        answer = array(
            [[1., 1., 0., 1., 0., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 1., 0., 0., 0., 0., 0.],
             [0., 1., 0., 0., 1., 0., 0., 0., 0., 0.],
             [0., 0., 0., 0., 0., 0., 0., 0., 0., 1.]])
        assert_array_equal(H, answer)
        assert_array_equal(xe, array([0., 0.25, 0.5, 0.75, 1]))

    @requires_array_function
    def test_dispatch(self):
        class ShouldDispatch:
            def __array_function__(self, function, types, args, kwargs):
                return types, args, kwargs

        xy = [1, 2]
        s_d = ShouldDispatch()
        r = histogram2d(s_d, xy)
        # Cannot use assert_equal since that dispatches...
        assert_(r == ((ShouldDispatch,), (s_d, xy), {}))
        r = histogram2d(xy, s_d)
        assert_(r == ((ShouldDispatch,), (xy, s_d), {}))
        r = histogram2d(xy, xy, bins=s_d)
        assert_(r, ((ShouldDispatch,), (xy, xy), dict(bins=s_d)))
        r = histogram2d(xy, xy, bins=[s_d, 5])
        assert_(r, ((ShouldDispatch,), (xy, xy), dict(bins=[s_d, 5])))
        assert_raises(Exception, histogram2d, xy, xy, bins=[s_d])
        r = histogram2d(xy, xy, weights=s_d)
        assert_(r, ((ShouldDispatch,), (xy, xy), dict(weights=s_d)))

    @pytest.mark.parametrize(("x_len", "y_len"), [(10, 11), (20, 19)])
    def test_bad_length(self, x_len, y_len):
        x, y = np.ones(x_len), np.ones(y_len)
        with pytest.raises(ValueError,
                           match='x and y must have the same length.'):
            histogram2d(x, y)


class TestTri:
    def test_dtype(self):
        out = array([[1, 0, 0],
                     [1, 1, 0],
                     [1, 1, 1]])
        assert_array_equal(tri(3), out)
        assert_array_equal(tri(3, dtype=bool), out.astype(bool))


def test_tril_triu_ndim2():
    for dtype in np.typecodes['AllFloat'] + np.typecodes['AllInteger']:
        a = np.ones((2, 2), dtype=dtype)
        b = np.tril(a)
        c = np.triu(a)
        assert_array_equal(b, [[1, 0], [1, 1]])
        assert_array_equal(c, b.T)
        # should return the same dtype as the original array
        assert_equal(b.dtype, a.dtype)
        assert_equal(c.dtype, a.dtype)


def test_tril_triu_ndim3():
    for dtype in np.typecodes['AllFloat'] + np.typecodes['AllInteger']:
        a = np.array([
            [[1, 1], [1, 1]],
            [[1, 1], [1, 0]],
            [[1, 1], [0, 0]],
            ], dtype=dtype)
        a_tril_desired = np.array([
            [[1, 0], [1, 1]],
            [[1, 0], [1, 0]],
            [[1, 0], [0, 0]],
            ], dtype=dtype)
        a_triu_desired = np.array([
            [[1, 1], [0, 1]],
            [[1, 1], [0, 0]],
            [[1, 1], [0, 0]],
            ], dtype=dtype)
        a_triu_observed = np.triu(a)
        a_tril_observed = np.tril(a)
        assert_array_equal(a_triu_observed, a_triu_desired)
        assert_array_equal(a_tril_observed, a_tril_desired)
        assert_equal(a_triu_observed.dtype, a.dtype)
        assert_equal(a_tril_observed.dtype, a.dtype)


def test_tril_triu_with_inf():
    # Issue 4859
    arr = np.array([[1, 1, np.inf],
                    [1, 1, 1],
                    [np.inf, 1, 1]])
    out_tril = np.array([[1, 0, 0],
                         [1, 1, 0],
                         [np.inf, 1, 1]])
    out_triu = out_tril.T
    assert_array_equal(np.triu(arr), out_triu)
    assert_array_equal(np.tril(arr), out_tril)


def test_tril_triu_dtype():
    # Issue 4916
    # tril and triu should return the same dtype as input
    for c in np.typecodes['All']:
        if c == 'V':
            continue
        arr = np.zeros((3, 3), dtype=c)
        assert_equal(np.triu(arr).dtype, arr.dtype)
        assert_equal(np.tril(arr).dtype, arr.dtype)

    # check special cases
    arr = np.array([['2001-01-01T12:00', '2002-02-03T13:56'],
                    ['2004-01-01T12:00', '2003-01-03T13:45']],
                   dtype='datetime64')
    assert_equal(np.triu(arr).dtype, arr.dtype)
    assert_equal(np.tril(arr).dtype, arr.dtype)

    arr = np.zeros((3, 3), dtype='f4,f4')
    assert_equal(np.triu(arr).dtype, arr.dtype)
    assert_equal(np.tril(arr).dtype, arr.dtype)


def test_mask_indices():
    # simple test without offset
    iu = mask_indices(3, np.triu)
    a = np.arange(9).reshape(3, 3)
    assert_array_equal(a[iu], array([0, 1, 2, 4, 5, 8]))
    # Now with an offset
    iu1 = mask_indices(3, np.triu, 1)
    assert_array_equal(a[iu1], array([1, 2, 5]))


def test_tril_indices():
    # indices without and with offset
    il1 = tril_indices(4)
    il2 = tril_indices(4, k=2)
    il3 = tril_indices(4, m=5)
    il4 = tril_indices(4, k=2, m=5)

    a = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    b = np.arange(1, 21).reshape(4, 5)

    # indexing:
    assert_array_equal(a[il1],
                       array([1, 5, 6, 9, 10, 11, 13, 14, 15, 16]))
    assert_array_equal(b[il3],
                       array([1, 6, 7, 11, 12, 13, 16, 17, 18, 19]))

    # And for assigning values:
    a[il1] = -1
    assert_array_equal(a,
                       array([[-1, 2, 3, 4],
                              [-1, -1, 7, 8],
                              [-1, -1, -1, 12],
                              [-1, -1, -1, -1]]))
    b[il3] = -1
    assert_array_equal(b,
                       array([[-1, 2, 3, 4, 5],
                              [-1, -1, 8, 9, 10],
                              [-1, -1, -1, 14, 15],
                              [-1, -1, -1, -1, 20]]))
    # These cover almost the whole array (two diagonals right of the main one):
    a[il2] = -10
    assert_array_equal(a,
                       array([[-10, -10, -10, 4],
                              [-10, -10, -10, -10],
                              [-10, -10, -10, -10],
                              [-10, -10, -10, -10]]))
    b[il4] = -10
    assert_array_equal(b,
                       array([[-10, -10, -10, 4, 5],
                              [-10, -10, -10, -10, 10],
                              [-10, -10, -10, -10, -10],
                              [-10, -10, -10, -10, -10]]))


class TestTriuIndices:
    def test_triu_indices(self):
        iu1 = triu_indices(4)
        iu2 = triu_indices(4, k=2)
        iu3 = triu_indices(4, m=5)
        iu4 = triu_indices(4, k=2, m=5)

        a = np.array([[1, 2, 3, 4],
                      [5, 6, 7, 8],
                      [9, 10, 11, 12],
                      [13, 14, 15, 16]])
        b = np.arange(1, 21).reshape(4, 5)

        # Both for indexing:
        assert_array_equal(a[iu1],
                           array([1, 2, 3, 4, 6, 7, 8, 11, 12, 16]))
        assert_array_equal(b[iu3],
                           array([1, 2, 3, 4, 5, 7, 8, 9,
                                  10, 13, 14, 15, 19, 20]))

        # And for assigning values:
        a[iu1] = -1
        assert_array_equal(a,
                           array([[-1, -1, -1, -1],
                                  [5, -1, -1, -1],
                                  [9, 10, -1, -1],
                                  [13, 14, 15, -1]]))
        b[iu3] = -1
        assert_array_equal(b,
                           array([[-1, -1, -1, -1, -1],
                                  [6, -1, -1, -1, -1],
                                  [11, 12, -1, -1, -1],
                                  [16, 17, 18, -1, -1]]))

        # These cover almost the whole array (two diagonals right of the
        # main one):
        a[iu2] = -10
        assert_array_equal(a,
                           array([[-1, -1, -10, -10],
                                  [5, -1, -1, -10],
                                  [9, 10, -1, -1],
                                  [13, 14, 15, -1]]))
        b[iu4] = -10
        assert_array_equal(b,
                           array([[-1, -1, -10, -10, -10],
                                  [6, -1, -1, -10, -10],
                                  [11, 12, -1, -1, -10],
                                  [16, 17, 18, -1, -1]]))


class TestTrilIndicesFrom:
    def test_exceptions(self):
        assert_raises(ValueError, tril_indices_from, np.ones((2,)))
        assert_raises(ValueError, tril_indices_from, np.ones((2, 2, 2)))
        # assert_raises(ValueError, tril_indices_from, np.ones((2, 3)))


class TestTriuIndicesFrom:
    def test_exceptions(self):
        assert_raises(ValueError, triu_indices_from, np.ones((2,)))
        assert_raises(ValueError, triu_indices_from, np.ones((2, 2, 2)))
        # assert_raises(ValueError, triu_indices_from, np.ones((2, 3)))


class TestVander:
    def test_basic(self):
        c = np.array([0, 1, -2, 3])
        v = vander(c)
        powers = np.array([[0, 0, 0, 0, 1],
                           [1, 1, 1, 1, 1],
                           [16, -8, 4, -2, 1],
                           [81, 27, 9, 3, 1]])
        # Check default value of N:
        assert_array_equal(v, powers[:, 1:])
        # Check a range of N values, including 0 and 5 (greater than default)
        m = powers.shape[1]
        for n in range(6):
            v = vander(c, N=n)
            assert_array_equal(v, powers[:, m-n:m])

    def test_dtypes(self):
        c = array([11, -12, 13], dtype=np.int8)
        v = vander(c)
        expected = np.array([[121, 11, 1],
                             [144, -12, 1],
                             [169, 13, 1]])
        assert_array_equal(v, expected)

        c = array([1.0+1j, 1.0-1j])
        v = vander(c, N=3)
        expected = np.array([[2j, 1+1j, 1],
                             [-2j, 1-1j, 1]])
        # The data is floating point, but the values are small integers,
        # so assert_array_equal *should* be safe here (rather than, say,
        # assert_array_almost_equal).
        assert_array_equal(v, expected)
