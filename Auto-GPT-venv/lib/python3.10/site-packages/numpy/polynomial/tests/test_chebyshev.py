"""Tests for chebyshev module.

"""
from functools import reduce

import numpy as np
import numpy.polynomial.chebyshev as cheb
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
    assert_almost_equal, assert_raises, assert_equal, assert_,
    )


def trim(x):
    return cheb.chebtrim(x, tol=1e-6)

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


class TestPrivate:

    def test__cseries_to_zseries(self):
        for i in range(5):
            inp = np.array([2] + [1]*i, np.double)
            tgt = np.array([.5]*i + [2] + [.5]*i, np.double)
            res = cheb._cseries_to_zseries(inp)
            assert_equal(res, tgt)

    def test__zseries_to_cseries(self):
        for i in range(5):
            inp = np.array([.5]*i + [2] + [.5]*i, np.double)
            tgt = np.array([2] + [1]*i, np.double)
            res = cheb._zseries_to_cseries(inp)
            assert_equal(res, tgt)


class TestConstants:

    def test_chebdomain(self):
        assert_equal(cheb.chebdomain, [-1, 1])

    def test_chebzero(self):
        assert_equal(cheb.chebzero, [0])

    def test_chebone(self):
        assert_equal(cheb.chebone, [1])

    def test_chebx(self):
        assert_equal(cheb.chebx, [0, 1])


class TestArithmetic:

    def test_chebadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = cheb.chebadd([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = cheb.chebsub([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebmulx(self):
        assert_equal(cheb.chebmulx([0]), [0])
        assert_equal(cheb.chebmulx([1]), [0, 1])
        for i in range(1, 5):
            ser = [0]*i + [1]
            tgt = [0]*(i - 1) + [.5, 0, .5]
            assert_equal(cheb.chebmulx(ser), tgt)

    def test_chebmul(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(i + j + 1)
                tgt[i + j] += .5
                tgt[abs(i - j)] += .5
                res = cheb.chebmul([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0]*i + [1]
                cj = [0]*j + [1]
                tgt = cheb.chebadd(ci, cj)
                quo, rem = cheb.chebdiv(tgt, ci)
                res = cheb.chebadd(cheb.chebmul(quo, ci), rem)
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_chebpow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = np.arange(i + 1)
                tgt = reduce(cheb.chebmul, [c]*j, np.array([1]))
                res = cheb.chebpow(c, j)
                assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = np.array([2.5, 2., 1.5])
    c2d = np.einsum('i,j->ij', c1d, c1d)
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = np.random.random((3, 5))*2 - 1
    y = polyval(x, [1., 2., 3.])

    def test_chebval(self):
        #check empty input
        assert_equal(cheb.chebval([], [1]).size, 0)

        #check normal input)
        x = np.linspace(-1, 1)
        y = [polyval(x, c) for c in Tlist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = cheb.chebval(x, [0]*i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        #check that shape is preserved
        for i in range(3):
            dims = [2]*i
            x = np.zeros(dims)
            assert_equal(cheb.chebval(x, [1]).shape, dims)
            assert_equal(cheb.chebval(x, [1, 0]).shape, dims)
            assert_equal(cheb.chebval(x, [1, 0, 0]).shape, dims)

    def test_chebval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test exceptions
        assert_raises(ValueError, cheb.chebval2d, x1, x2[:2], self.c2d)

        #test values
        tgt = y1*y2
        res = cheb.chebval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = cheb.chebval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))

    def test_chebval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test exceptions
        assert_raises(ValueError, cheb.chebval3d, x1, x2, x3[:2], self.c3d)

        #test values
        tgt = y1*y2*y3
        res = cheb.chebval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = cheb.chebval3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3))

    def test_chebgrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test values
        tgt = np.einsum('i,j->ij', y1, y2)
        res = cheb.chebgrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = cheb.chebgrid2d(z, z, self.c2d)
        assert_(res.shape == (2, 3)*2)

    def test_chebgrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test values
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        res = cheb.chebgrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = cheb.chebgrid3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3)*3)


class TestIntegral:

    def test_chebint(self):
        # check exceptions
        assert_raises(TypeError, cheb.chebint, [0], .5)
        assert_raises(ValueError, cheb.chebint, [0], -1)
        assert_raises(ValueError, cheb.chebint, [0], 1, [0, 0])
        assert_raises(ValueError, cheb.chebint, [0], lbnd=[0])
        assert_raises(ValueError, cheb.chebint, [0], scl=[0])
        assert_raises(TypeError, cheb.chebint, [0], axis=.5)

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0]*(i - 2) + [1]
            res = cheb.chebint([0], m=i, k=k)
            assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [1/scl]
            chebpol = cheb.poly2cheb(pol)
            chebint = cheb.chebint(chebpol, m=1, k=[i])
            res = cheb.cheb2poly(chebint)
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0]*i + [1]
            chebpol = cheb.poly2cheb(pol)
            chebint = cheb.chebint(chebpol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(cheb.chebval(-1, chebint), i)

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [2/scl]
            chebpol = cheb.poly2cheb(pol)
            chebint = cheb.chebint(chebpol, m=1, k=[i], scl=2)
            res = cheb.cheb2poly(chebint)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = cheb.chebint(tgt, m=1)
                res = cheb.chebint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = cheb.chebint(tgt, m=1, k=[k])
                res = cheb.chebint(pol, m=j, k=list(range(j)))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = cheb.chebint(tgt, m=1, k=[k], lbnd=-1)
                res = cheb.chebint(pol, m=j, k=list(range(j)), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = cheb.chebint(tgt, m=1, k=[k], scl=2)
                res = cheb.chebint(pol, m=j, k=list(range(j)), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_chebint_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([cheb.chebint(c) for c in c2d.T]).T
        res = cheb.chebint(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([cheb.chebint(c) for c in c2d])
        res = cheb.chebint(c2d, axis=1)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([cheb.chebint(c, k=3) for c in c2d])
        res = cheb.chebint(c2d, k=3, axis=1)
        assert_almost_equal(res, tgt)


class TestDerivative:

    def test_chebder(self):
        # check exceptions
        assert_raises(TypeError, cheb.chebder, [0], .5)
        assert_raises(ValueError, cheb.chebder, [0], -1)

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0]*i + [1]
            res = cheb.chebder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = cheb.chebder(cheb.chebint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = cheb.chebder(cheb.chebint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_chebder_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([cheb.chebder(c) for c in c2d.T]).T
        res = cheb.chebder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([cheb.chebder(c) for c in c2d])
        res = cheb.chebder(c2d, axis=1)
        assert_almost_equal(res, tgt)


class TestVander:
    # some random values in [-1, 1)
    x = np.random.random((3, 5))*2 - 1

    def test_chebvander(self):
        # check for 1d x
        x = np.arange(3)
        v = cheb.chebvander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], cheb.chebval(x, coef))

        # check for 2d x
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = cheb.chebvander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], cheb.chebval(x, coef))

    def test_chebvander2d(self):
        # also tests chebval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = cheb.chebvander2d(x1, x2, [1, 2])
        tgt = cheb.chebval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = cheb.chebvander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_chebvander3d(self):
        # also tests chebval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3, 4))
        van = cheb.chebvander3d(x1, x2, x3, [1, 2, 3])
        tgt = cheb.chebval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = cheb.chebvander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))


class TestFitting:

    def test_chebfit(self):
        def f(x):
            return x*(x - 1)*(x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        # Test exceptions
        assert_raises(ValueError, cheb.chebfit, [1], [1], -1)
        assert_raises(TypeError, cheb.chebfit, [[1]], [1], 0)
        assert_raises(TypeError, cheb.chebfit, [], [1], 0)
        assert_raises(TypeError, cheb.chebfit, [1], [[[1]]], 0)
        assert_raises(TypeError, cheb.chebfit, [1, 2], [1], 0)
        assert_raises(TypeError, cheb.chebfit, [1], [1, 2], 0)
        assert_raises(TypeError, cheb.chebfit, [1], [1], 0, w=[[1]])
        assert_raises(TypeError, cheb.chebfit, [1], [1], 0, w=[1, 1])
        assert_raises(ValueError, cheb.chebfit, [1], [1], [-1,])
        assert_raises(ValueError, cheb.chebfit, [1], [1], [2, -1, 6])
        assert_raises(TypeError, cheb.chebfit, [1], [1], [])

        # Test fit
        x = np.linspace(0, 2)
        y = f(x)
        #
        coef3 = cheb.chebfit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(cheb.chebval(x, coef3), y)
        coef3 = cheb.chebfit(x, y, [0, 1, 2, 3])
        assert_equal(len(coef3), 4)
        assert_almost_equal(cheb.chebval(x, coef3), y)
        #
        coef4 = cheb.chebfit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(cheb.chebval(x, coef4), y)
        coef4 = cheb.chebfit(x, y, [0, 1, 2, 3, 4])
        assert_equal(len(coef4), 5)
        assert_almost_equal(cheb.chebval(x, coef4), y)
        # check things still work if deg is not in strict increasing
        coef4 = cheb.chebfit(x, y, [2, 3, 4, 1, 0])
        assert_equal(len(coef4), 5)
        assert_almost_equal(cheb.chebval(x, coef4), y)
        #
        coef2d = cheb.chebfit(x, np.array([y, y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        coef2d = cheb.chebfit(x, np.array([y, y]).T, [0, 1, 2, 3])
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = cheb.chebfit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        wcoef3 = cheb.chebfit(x, yw, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = cheb.chebfit(x, np.array([yw, yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        wcoef2d = cheb.chebfit(x, np.array([yw, yw]).T, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        assert_almost_equal(cheb.chebfit(x, x, 1), [0, 1])
        assert_almost_equal(cheb.chebfit(x, x, [0, 1]), [0, 1])
        # test fitting only even polynomials
        x = np.linspace(-1, 1)
        y = f2(x)
        coef1 = cheb.chebfit(x, y, 4)
        assert_almost_equal(cheb.chebval(x, coef1), y)
        coef2 = cheb.chebfit(x, y, [0, 2, 4])
        assert_almost_equal(cheb.chebval(x, coef2), y)
        assert_almost_equal(coef1, coef2)


class TestInterpolate:

    def f(self, x):
        return x * (x - 1) * (x - 2)

    def test_raises(self):
        assert_raises(ValueError, cheb.chebinterpolate, self.f, -1)
        assert_raises(TypeError, cheb.chebinterpolate, self.f, 10.)

    def test_dimensions(self):
        for deg in range(1, 5):
            assert_(cheb.chebinterpolate(self.f, deg).shape == (deg + 1,))

    def test_approximation(self):

        def powx(x, p):
            return x**p

        x = np.linspace(-1, 1, 10)
        for deg in range(0, 10):
            for p in range(0, deg + 1):
                c = cheb.chebinterpolate(powx, deg, (p,))
                assert_almost_equal(cheb.chebval(x, c), powx(x, p), decimal=12)


class TestCompanion:

    def test_raises(self):
        assert_raises(ValueError, cheb.chebcompanion, [])
        assert_raises(ValueError, cheb.chebcompanion, [1])

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0]*i + [1]
            assert_(cheb.chebcompanion(coef).shape == (i, i))

    def test_linear_root(self):
        assert_(cheb.chebcompanion([1, 2])[0, 0] == -.5)


class TestGauss:

    def test_100(self):
        x, w = cheb.chebgauss(100)

        # test orthogonality. Note that the results need to be normalized,
        # otherwise the huge values that can arise from fast growing
        # functions like Laguerre can be very confusing.
        v = cheb.chebvander(x, 99)
        vv = np.dot(v.T * w, v)
        vd = 1/np.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        assert_almost_equal(vv, np.eye(100))

        # check that the integral of 1 is correct
        tgt = np.pi
        assert_almost_equal(w.sum(), tgt)


class TestMisc:

    def test_chebfromroots(self):
        res = cheb.chebfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            tgt = [0]*i + [1]
            res = cheb.chebfromroots(roots)*2**(i-1)
            assert_almost_equal(trim(res), trim(tgt))

    def test_chebroots(self):
        assert_almost_equal(cheb.chebroots([1]), [])
        assert_almost_equal(cheb.chebroots([1, 2]), [-.5])
        for i in range(2, 5):
            tgt = np.linspace(-1, 1, i)
            res = cheb.chebroots(cheb.chebfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_chebtrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, cheb.chebtrim, coef, -1)

        # Test results
        assert_equal(cheb.chebtrim(coef), coef[:-1])
        assert_equal(cheb.chebtrim(coef, 1), coef[:-3])
        assert_equal(cheb.chebtrim(coef, 2), [0])

    def test_chebline(self):
        assert_equal(cheb.chebline(3, 4), [3, 4])

    def test_cheb2poly(self):
        for i in range(10):
            assert_almost_equal(cheb.cheb2poly([0]*i + [1]), Tlist[i])

    def test_poly2cheb(self):
        for i in range(10):
            assert_almost_equal(cheb.poly2cheb(Tlist[i]), [0]*i + [1])

    def test_weight(self):
        x = np.linspace(-1, 1, 11)[1:-1]
        tgt = 1./(np.sqrt(1 + x) * np.sqrt(1 - x))
        res = cheb.chebweight(x)
        assert_almost_equal(res, tgt)

    def test_chebpts1(self):
        #test exceptions
        assert_raises(ValueError, cheb.chebpts1, 1.5)
        assert_raises(ValueError, cheb.chebpts1, 0)

        #test points
        tgt = [0]
        assert_almost_equal(cheb.chebpts1(1), tgt)
        tgt = [-0.70710678118654746, 0.70710678118654746]
        assert_almost_equal(cheb.chebpts1(2), tgt)
        tgt = [-0.86602540378443871, 0, 0.86602540378443871]
        assert_almost_equal(cheb.chebpts1(3), tgt)
        tgt = [-0.9238795325, -0.3826834323, 0.3826834323, 0.9238795325]
        assert_almost_equal(cheb.chebpts1(4), tgt)

    def test_chebpts2(self):
        #test exceptions
        assert_raises(ValueError, cheb.chebpts2, 1.5)
        assert_raises(ValueError, cheb.chebpts2, 1)

        #test points
        tgt = [-1, 1]
        assert_almost_equal(cheb.chebpts2(2), tgt)
        tgt = [-1, 0, 1]
        assert_almost_equal(cheb.chebpts2(3), tgt)
        tgt = [-1, -0.5, .5, 1]
        assert_almost_equal(cheb.chebpts2(4), tgt)
        tgt = [-1.0, -0.707106781187, 0, 0.707106781187, 1.0]
        assert_almost_equal(cheb.chebpts2(5), tgt)
