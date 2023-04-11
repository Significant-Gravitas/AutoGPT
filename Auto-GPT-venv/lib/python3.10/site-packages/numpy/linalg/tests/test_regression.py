""" Test functions for linalg module
"""
import warnings

import numpy as np
from numpy import linalg, arange, float64, array, dot, transpose
from numpy.testing import (
    assert_, assert_raises, assert_equal, assert_array_equal,
    assert_array_almost_equal, assert_array_less
)


class TestRegression:

    def test_eig_build(self):
        # Ticket #652
        rva = array([1.03221168e+02 + 0.j,
                     -1.91843603e+01 + 0.j,
                     -6.04004526e-01 + 15.84422474j,
                     -6.04004526e-01 - 15.84422474j,
                     -1.13692929e+01 + 0.j,
                     -6.57612485e-01 + 10.41755503j,
                     -6.57612485e-01 - 10.41755503j,
                     1.82126812e+01 + 0.j,
                     1.06011014e+01 + 0.j,
                     7.80732773e+00 + 0.j,
                     -7.65390898e-01 + 0.j,
                     1.51971555e-15 + 0.j,
                     -1.51308713e-15 + 0.j])
        a = arange(13 * 13, dtype=float64)
        a.shape = (13, 13)
        a = a % 17
        va, ve = linalg.eig(a)
        va.sort()
        rva.sort()
        assert_array_almost_equal(va, rva)

    def test_eigh_build(self):
        # Ticket 662.
        rvals = [68.60568999, 89.57756725, 106.67185574]

        cov = array([[77.70273908,   3.51489954,  15.64602427],
                     [3.51489954,  88.97013878,  -1.07431931],
                     [15.64602427,  -1.07431931,  98.18223512]])

        vals, vecs = linalg.eigh(cov)
        assert_array_almost_equal(vals, rvals)

    def test_svd_build(self):
        # Ticket 627.
        a = array([[0., 1.], [1., 1.], [2., 1.], [3., 1.]])
        m, n = a.shape
        u, s, vh = linalg.svd(a)

        b = dot(transpose(u[:, n:]), a)

        assert_array_almost_equal(b, np.zeros((2, 2)))

    def test_norm_vector_badarg(self):
        # Regression for #786: Frobenius norm for vectors raises
        # ValueError.
        assert_raises(ValueError, linalg.norm, array([1., 2., 3.]), 'fro')

    def test_lapack_endian(self):
        # For bug #1482
        a = array([[5.7998084,  -2.1825367],
                   [-2.1825367,   9.85910595]], dtype='>f8')
        b = array(a, dtype='<f8')

        ap = linalg.cholesky(a)
        bp = linalg.cholesky(b)
        assert_array_equal(ap, bp)

    def test_large_svd_32bit(self):
        # See gh-4442, 64bit would require very large/slow matrices.
        x = np.eye(1000, 66)
        np.linalg.svd(x)

    def test_svd_no_uv(self):
        # gh-4733
        for shape in (3, 4), (4, 4), (4, 3):
            for t in float, complex:
                a = np.ones(shape, dtype=t)
                w = linalg.svd(a, compute_uv=False)
                c = np.count_nonzero(np.absolute(w) > 0.5)
                assert_equal(c, 1)
                assert_equal(np.linalg.matrix_rank(a), 1)
                assert_array_less(1, np.linalg.norm(a, ord=2))

    def test_norm_object_array(self):
        # gh-7575
        testvector = np.array([np.array([0, 1]), 0, 0], dtype=object)

        norm = linalg.norm(testvector)
        assert_array_equal(norm, [0, 1])
        assert_(norm.dtype == np.dtype('float64'))

        norm = linalg.norm(testvector, ord=1)
        assert_array_equal(norm, [0, 1])
        assert_(norm.dtype != np.dtype('float64'))

        norm = linalg.norm(testvector, ord=2)
        assert_array_equal(norm, [0, 1])
        assert_(norm.dtype == np.dtype('float64'))

        assert_raises(ValueError, linalg.norm, testvector, ord='fro')
        assert_raises(ValueError, linalg.norm, testvector, ord='nuc')
        assert_raises(ValueError, linalg.norm, testvector, ord=np.inf)
        assert_raises(ValueError, linalg.norm, testvector, ord=-np.inf)
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            assert_raises((AttributeError, DeprecationWarning),
                              linalg.norm, testvector, ord=0)
        assert_raises(ValueError, linalg.norm, testvector, ord=-1)
        assert_raises(ValueError, linalg.norm, testvector, ord=-2)

        testmatrix = np.array([[np.array([0, 1]), 0, 0],
                               [0,                0, 0]], dtype=object)

        norm = linalg.norm(testmatrix)
        assert_array_equal(norm, [0, 1])
        assert_(norm.dtype == np.dtype('float64'))

        norm = linalg.norm(testmatrix, ord='fro')
        assert_array_equal(norm, [0, 1])
        assert_(norm.dtype == np.dtype('float64'))

        assert_raises(TypeError, linalg.norm, testmatrix, ord='nuc')
        assert_raises(ValueError, linalg.norm, testmatrix, ord=np.inf)
        assert_raises(ValueError, linalg.norm, testmatrix, ord=-np.inf)
        assert_raises(ValueError, linalg.norm, testmatrix, ord=0)
        assert_raises(ValueError, linalg.norm, testmatrix, ord=1)
        assert_raises(ValueError, linalg.norm, testmatrix, ord=-1)
        assert_raises(TypeError, linalg.norm, testmatrix, ord=2)
        assert_raises(TypeError, linalg.norm, testmatrix, ord=-2)
        assert_raises(ValueError, linalg.norm, testmatrix, ord=3)

    def test_lstsq_complex_larger_rhs(self):
        # gh-9891
        size = 20
        n_rhs = 70
        G = np.random.randn(size, size) + 1j * np.random.randn(size, size)
        u = np.random.randn(size, n_rhs) + 1j * np.random.randn(size, n_rhs)
        b = G.dot(u)
        # This should work without segmentation fault.
        u_lstsq, res, rank, sv = linalg.lstsq(G, b, rcond=None)
        # check results just in case
        assert_array_almost_equal(u_lstsq, u)
