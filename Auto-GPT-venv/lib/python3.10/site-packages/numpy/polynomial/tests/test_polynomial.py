"""Tests for polynomial module.

"""
from functools import reduce

import numpy as np
import numpy.polynomial.polynomial as poly
import pickle
from copy import deepcopy
from numpy.testing import (
    assert_almost_equal, assert_raises, assert_equal, assert_,
    assert_warns, assert_array_equal, assert_raises_regex)


def trim(x):
    return poly.polytrim(x, tol=1e-6)

T0 = [1]
T1 = [0, 1]
T2 = [-1, 0, 2]
T3 = [0, -3, 0, 4]
T4 = [1, 0, -8, 0, 8]
T5 = [0, 5, 0, -20, 0, 16]
T6 = [-1, 0, 18, 0, -48, 0, 32]
T7 = [0, -7, 0, 56, 0, -112, 0, 64]
T8 = [1, 0, -32, 0, 160, 0, -256, 0, 128]
T9 = [0, 9, 0, -120, 0, 432, 0, -576, 0, 256]

Tlist = [T0, T1, T2, T3, T4, T5, T6, T7, T8, T9]


class TestConstants:

    def test_polydomain(self):
        assert_equal(poly.polydomain, [-1, 1])

    def test_polyzero(self):
        assert_equal(poly.polyzero, [0])

    def test_polyone(self):
        assert_equal(poly.polyone, [1])

    def test_polyx(self):
        assert_equal(poly.polyx, [0, 1])

    def test_copy(self):
        x = poly.Polynomial([1, 2, 3])
        y = deepcopy(x)
        assert_equal(x, y)

    def test_pickle(self):
        x = poly.Polynomial([1, 2, 3])
        y = pickle.loads(pickle.dumps(x))
        assert_equal(x, y)

class TestArithmetic:

    def test_polyadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = poly.polyadd([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_polysub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = poly.polysub([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_polymulx(self):
        assert_equal(poly.polymulx([0]), [0])
        assert_equal(poly.polymulx([1]), [0, 1])
        for i in range(1, 5):
            ser = [0]*i + [1]
            tgt = [0]*(i + 1) + [1]
            assert_equal(poly.polymulx(ser), tgt)

    def test_polymul(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(i + j + 1)
                tgt[i + j] += 1
                res = poly.polymul([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_polydiv(self):
        # check zero division
        assert_raises(ZeroDivisionError, poly.polydiv, [1], [0])

        # check scalar division
        quo, rem = poly.polydiv([2], [2])
        assert_equal((quo, rem), (1, 0))
        quo, rem = poly.polydiv([2, 2], [2])
        assert_equal((quo, rem), ((1, 1), 0))

        # check rest.
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0]*i + [1, 2]
                cj = [0]*j + [1, 2]
                tgt = poly.polyadd(ci, cj)
                quo, rem = poly.polydiv(tgt, ci)
                res = poly.polyadd(poly.polymul(quo, ci), rem)
                assert_equal(res, tgt, err_msg=msg)

    def test_polypow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = np.arange(i + 1)
                tgt = reduce(poly.polymul, [c]*j, np.array([1]))
                res = poly.polypow(c, j) 
                assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = np.array([1., 2., 3.])
    c2d = np.einsum('i,j->ij', c1d, c1d)
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = np.random.random((3, 5))*2 - 1
    y = poly.polyval(x, [1., 2., 3.])

    def test_polyval(self):
        #check empty input
        assert_equal(poly.polyval([], [1]).size, 0)

        #check normal input)
        x = np.linspace(-1, 1)
        y = [x**i for i in range(5)]
        for i in range(5):
            tgt = y[i]
            res = poly.polyval(x, [0]*i + [1])
            assert_almost_equal(res, tgt)
        tgt = x*(x**2 - 1)
        res = poly.polyval(x, [0, -1, 0, 1])
        assert_almost_equal(res, tgt)

        #check that shape is preserved
        for i in range(3):
            dims = [2]*i
            x = np.zeros(dims)
            assert_equal(poly.polyval(x, [1]).shape, dims)
            assert_equal(poly.polyval(x, [1, 0]).shape, dims)
            assert_equal(poly.polyval(x, [1, 0, 0]).shape, dims)

        #check masked arrays are processed correctly
        mask = [False, True, False]
        mx = np.ma.array([1, 2, 3], mask=mask)
        res = np.polyval([7, 5, 3], mx)
        assert_array_equal(res.mask, mask)

        #check subtypes of ndarray are preserved
        class C(np.ndarray):
            pass

        cx = np.array([1, 2, 3]).view(C)
        assert_equal(type(np.polyval([2, 3, 4], cx)), C)

    def test_polyvalfromroots(self):
        # check exception for broadcasting x values over root array with
        # too few dimensions
        assert_raises(ValueError, poly.polyvalfromroots,
                      [1], [1], tensor=False)

        # check empty input
        assert_equal(poly.polyvalfromroots([], [1]).size, 0)
        assert_(poly.polyvalfromroots([], [1]).shape == (0,))

        # check empty input + multidimensional roots
        assert_equal(poly.polyvalfromroots([], [[1] * 5]).size, 0)
        assert_(poly.polyvalfromroots([], [[1] * 5]).shape == (5, 0))

        # check scalar input
        assert_equal(poly.polyvalfromroots(1, 1), 0)
        assert_(poly.polyvalfromroots(1, np.ones((3, 3))).shape == (3,))

        # check normal input)
        x = np.linspace(-1, 1)
        y = [x**i for i in range(5)]
        for i in range(1, 5):
            tgt = y[i]
            res = poly.polyvalfromroots(x, [0]*i)
            assert_almost_equal(res, tgt)
        tgt = x*(x - 1)*(x + 1)
        res = poly.polyvalfromroots(x, [-1, 0, 1])
        assert_almost_equal(res, tgt)

        # check that shape is preserved
        for i in range(3):
            dims = [2]*i
            x = np.zeros(dims)
            assert_equal(poly.polyvalfromroots(x, [1]).shape, dims)
            assert_equal(poly.polyvalfromroots(x, [1, 0]).shape, dims)
            assert_equal(poly.polyvalfromroots(x, [1, 0, 0]).shape, dims)

        # check compatibility with factorization
        ptest = [15, 2, -16, -2, 1]
        r = poly.polyroots(ptest)
        x = np.linspace(-1, 1)
        assert_almost_equal(poly.polyval(x, ptest),
                            poly.polyvalfromroots(x, r))

        # check multidimensional arrays of roots and values
        # check tensor=False
        rshape = (3, 5)
        x = np.arange(-3, 2)
        r = np.random.randint(-5, 5, size=rshape)
        res = poly.polyvalfromroots(x, r, tensor=False)
        tgt = np.empty(r.shape[1:])
        for ii in range(tgt.size):
            tgt[ii] = poly.polyvalfromroots(x[ii], r[:, ii])
        assert_equal(res, tgt)

        # check tensor=True
        x = np.vstack([x, 2*x])
        res = poly.polyvalfromroots(x, r, tensor=True)
        tgt = np.empty(r.shape[1:] + x.shape)
        for ii in range(r.shape[1]):
            for jj in range(x.shape[0]):
                tgt[ii, jj, :] = poly.polyvalfromroots(x[jj], r[:, ii])
        assert_equal(res, tgt)

    def test_polyval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test exceptions
        assert_raises_regex(ValueError, 'incompatible',
                            poly.polyval2d, x1, x2[:2], self.c2d)

        #test values
        tgt = y1*y2
        res = poly.polyval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = poly.polyval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))

    def test_polyval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test exceptions
        assert_raises_regex(ValueError, 'incompatible',
                      poly.polyval3d, x1, x2, x3[:2], self.c3d)

        #test values
        tgt = y1*y2*y3
        res = poly.polyval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = poly.polyval3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3))

    def test_polygrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test values
        tgt = np.einsum('i,j->ij', y1, y2)
        res = poly.polygrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = poly.polygrid2d(z, z, self.c2d)
        assert_(res.shape == (2, 3)*2)

    def test_polygrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test values
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        res = poly.polygrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = poly.polygrid3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3)*3)


class TestIntegral:

    def test_polyint(self):
        # check exceptions
        assert_raises(TypeError, poly.polyint, [0], .5)
        assert_raises(ValueError, poly.polyint, [0], -1)
        assert_raises(ValueError, poly.polyint, [0], 1, [0, 0])
        assert_raises(ValueError, poly.polyint, [0], lbnd=[0])
        assert_raises(ValueError, poly.polyint, [0], scl=[0])
        assert_raises(TypeError, poly.polyint, [0], axis=.5)
        with assert_warns(DeprecationWarning):
            poly.polyint([1, 1], 1.)

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0]*(i - 2) + [1]
            res = poly.polyint([0], m=i, k=k)
            assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [1/scl]
            res = poly.polyint(pol, m=1, k=[i])
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0]*i + [1]
            res = poly.polyint(pol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(poly.polyval(-1, res), i)

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [2/scl]
            res = poly.polyint(pol, m=1, k=[i], scl=2)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = poly.polyint(tgt, m=1)
                res = poly.polyint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = poly.polyint(tgt, m=1, k=[k])
                res = poly.polyint(pol, m=j, k=list(range(j)))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = poly.polyint(tgt, m=1, k=[k], lbnd=-1)
                res = poly.polyint(pol, m=j, k=list(range(j)), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = poly.polyint(tgt, m=1, k=[k], scl=2)
                res = poly.polyint(pol, m=j, k=list(range(j)), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_polyint_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([poly.polyint(c) for c in c2d.T]).T
        res = poly.polyint(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([poly.polyint(c) for c in c2d])
        res = poly.polyint(c2d, axis=1)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([poly.polyint(c, k=3) for c in c2d])
        res = poly.polyint(c2d, k=3, axis=1)
        assert_almost_equal(res, tgt)


class TestDerivative:

    def test_polyder(self):
        # check exceptions
        assert_raises(TypeError, poly.polyder, [0], .5)
        assert_raises(ValueError, poly.polyder, [0], -1)

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0]*i + [1]
            res = poly.polyder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = poly.polyder(poly.polyint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = poly.polyder(poly.polyint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_polyder_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([poly.polyder(c) for c in c2d.T]).T
        res = poly.polyder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([poly.polyder(c) for c in c2d])
        res = poly.polyder(c2d, axis=1)
        assert_almost_equal(res, tgt)


class TestVander:
    # some random values in [-1, 1)
    x = np.random.random((3, 5))*2 - 1

    def test_polyvander(self):
        # check for 1d x
        x = np.arange(3)
        v = poly.polyvander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], poly.polyval(x, coef))

        # check for 2d x
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = poly.polyvander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], poly.polyval(x, coef))

    def test_polyvander2d(self):
        # also tests polyval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = poly.polyvander2d(x1, x2, [1, 2])
        tgt = poly.polyval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = poly.polyvander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_polyvander3d(self):
        # also tests polyval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3, 4))
        van = poly.polyvander3d(x1, x2, x3, [1, 2, 3])
        tgt = poly.polyval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = poly.polyvander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))

    def test_polyvandernegdeg(self):
        x = np.arange(3)
        assert_raises(ValueError, poly.polyvander, x, -1)


class TestCompanion:

    def test_raises(self):
        assert_raises(ValueError, poly.polycompanion, [])
        assert_raises(ValueError, poly.polycompanion, [1])

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0]*i + [1]
            assert_(poly.polycompanion(coef).shape == (i, i))

    def test_linear_root(self):
        assert_(poly.polycompanion([1, 2])[0, 0] == -.5)


class TestMisc:

    def test_polyfromroots(self):
        res = poly.polyfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            tgt = Tlist[i]
            res = poly.polyfromroots(roots)*2**(i-1)
            assert_almost_equal(trim(res), trim(tgt))

    def test_polyroots(self):
        assert_almost_equal(poly.polyroots([1]), [])
        assert_almost_equal(poly.polyroots([1, 2]), [-.5])
        for i in range(2, 5):
            tgt = np.linspace(-1, 1, i)
            res = poly.polyroots(poly.polyfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_polyfit(self):
        def f(x):
            return x*(x - 1)*(x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        # Test exceptions
        assert_raises(ValueError, poly.polyfit, [1], [1], -1)
        assert_raises(TypeError, poly.polyfit, [[1]], [1], 0)
        assert_raises(TypeError, poly.polyfit, [], [1], 0)
        assert_raises(TypeError, poly.polyfit, [1], [[[1]]], 0)
        assert_raises(TypeError, poly.polyfit, [1, 2], [1], 0)
        assert_raises(TypeError, poly.polyfit, [1], [1, 2], 0)
        assert_raises(TypeError, poly.polyfit, [1], [1], 0, w=[[1]])
        assert_raises(TypeError, poly.polyfit, [1], [1], 0, w=[1, 1])
        assert_raises(ValueError, poly.polyfit, [1], [1], [-1,])
        assert_raises(ValueError, poly.polyfit, [1], [1], [2, -1, 6])
        assert_raises(TypeError, poly.polyfit, [1], [1], [])

        # Test fit
        x = np.linspace(0, 2)
        y = f(x)
        #
        coef3 = poly.polyfit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(poly.polyval(x, coef3), y)
        coef3 = poly.polyfit(x, y, [0, 1, 2, 3])
        assert_equal(len(coef3), 4)
        assert_almost_equal(poly.polyval(x, coef3), y)
        #
        coef4 = poly.polyfit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(poly.polyval(x, coef4), y)
        coef4 = poly.polyfit(x, y, [0, 1, 2, 3, 4])
        assert_equal(len(coef4), 5)
        assert_almost_equal(poly.polyval(x, coef4), y)
        #
        coef2d = poly.polyfit(x, np.array([y, y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        coef2d = poly.polyfit(x, np.array([y, y]).T, [0, 1, 2, 3])
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        yw[0::2] = 0
        wcoef3 = poly.polyfit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        wcoef3 = poly.polyfit(x, yw, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = poly.polyfit(x, np.array([yw, yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        wcoef2d = poly.polyfit(x, np.array([yw, yw]).T, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        assert_almost_equal(poly.polyfit(x, x, 1), [0, 1])
        assert_almost_equal(poly.polyfit(x, x, [0, 1]), [0, 1])
        # test fitting only even Polyendre polynomials
        x = np.linspace(-1, 1)
        y = f2(x)
        coef1 = poly.polyfit(x, y, 4)
        assert_almost_equal(poly.polyval(x, coef1), y)
        coef2 = poly.polyfit(x, y, [0, 2, 4])
        assert_almost_equal(poly.polyval(x, coef2), y)
        assert_almost_equal(coef1, coef2)

    def test_polytrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, poly.polytrim, coef, -1)

        # Test results
        assert_equal(poly.polytrim(coef), coef[:-1])
        assert_equal(poly.polytrim(coef, 1), coef[:-3])
        assert_equal(poly.polytrim(coef, 2), [0])

    def test_polyline(self):
        assert_equal(poly.polyline(3, 4), [3, 4])

    def test_polyline_zero(self):
        assert_equal(poly.polyline(3, 0), [3])
