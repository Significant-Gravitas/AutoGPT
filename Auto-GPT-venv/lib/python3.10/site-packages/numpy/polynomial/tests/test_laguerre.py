"""Tests for laguerre module.

"""
from functools import reduce

import numpy as np
import numpy.polynomial.laguerre as lag
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
    assert_almost_equal, assert_raises, assert_equal, assert_,
    )

L0 = np.array([1])/1
L1 = np.array([1, -1])/1
L2 = np.array([2, -4, 1])/2
L3 = np.array([6, -18, 9, -1])/6
L4 = np.array([24, -96, 72, -16, 1])/24
L5 = np.array([120, -600, 600, -200, 25, -1])/120
L6 = np.array([720, -4320, 5400, -2400, 450, -36, 1])/720

Llist = [L0, L1, L2, L3, L4, L5, L6]


def trim(x):
    return lag.lagtrim(x, tol=1e-6)


class TestConstants:

    def test_lagdomain(self):
        assert_equal(lag.lagdomain, [0, 1])

    def test_lagzero(self):
        assert_equal(lag.lagzero, [0])

    def test_lagone(self):
        assert_equal(lag.lagone, [1])

    def test_lagx(self):
        assert_equal(lag.lagx, [1, -1])


class TestArithmetic:
    x = np.linspace(-3, 3, 100)

    def test_lagadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = lag.lagadd([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_lagsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = lag.lagsub([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_lagmulx(self):
        assert_equal(lag.lagmulx([0]), [0])
        assert_equal(lag.lagmulx([1]), [1, -1])
        for i in range(1, 5):
            ser = [0]*i + [1]
            tgt = [0]*(i - 1) + [-i, 2*i + 1, -(i + 1)]
            assert_almost_equal(lag.lagmulx(ser), tgt)

    def test_lagmul(self):
        # check values of result
        for i in range(5):
            pol1 = [0]*i + [1]
            val1 = lag.lagval(self.x, pol1)
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0]*j + [1]
                val2 = lag.lagval(self.x, pol2)
                pol3 = lag.lagmul(pol1, pol2)
                val3 = lag.lagval(self.x, pol3)
                assert_(len(pol3) == i + j + 1, msg)
                assert_almost_equal(val3, val1*val2, err_msg=msg)

    def test_lagdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0]*i + [1]
                cj = [0]*j + [1]
                tgt = lag.lagadd(ci, cj)
                quo, rem = lag.lagdiv(tgt, ci)
                res = lag.lagadd(lag.lagmul(quo, ci), rem)
                assert_almost_equal(trim(res), trim(tgt), err_msg=msg)

    def test_lagpow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = np.arange(i + 1)
                tgt = reduce(lag.lagmul, [c]*j, np.array([1]))
                res = lag.lagpow(c, j) 
                assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = np.array([9., -14., 6.])
    c2d = np.einsum('i,j->ij', c1d, c1d)
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = np.random.random((3, 5))*2 - 1
    y = polyval(x, [1., 2., 3.])

    def test_lagval(self):
        #check empty input
        assert_equal(lag.lagval([], [1]).size, 0)

        #check normal input)
        x = np.linspace(-1, 1)
        y = [polyval(x, c) for c in Llist]
        for i in range(7):
            msg = f"At i={i}"
            tgt = y[i]
            res = lag.lagval(x, [0]*i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        #check that shape is preserved
        for i in range(3):
            dims = [2]*i
            x = np.zeros(dims)
            assert_equal(lag.lagval(x, [1]).shape, dims)
            assert_equal(lag.lagval(x, [1, 0]).shape, dims)
            assert_equal(lag.lagval(x, [1, 0, 0]).shape, dims)

    def test_lagval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test exceptions
        assert_raises(ValueError, lag.lagval2d, x1, x2[:2], self.c2d)

        #test values
        tgt = y1*y2
        res = lag.lagval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = lag.lagval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))

    def test_lagval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test exceptions
        assert_raises(ValueError, lag.lagval3d, x1, x2, x3[:2], self.c3d)

        #test values
        tgt = y1*y2*y3
        res = lag.lagval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = lag.lagval3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3))

    def test_laggrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test values
        tgt = np.einsum('i,j->ij', y1, y2)
        res = lag.laggrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = lag.laggrid2d(z, z, self.c2d)
        assert_(res.shape == (2, 3)*2)

    def test_laggrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test values
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        res = lag.laggrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = lag.laggrid3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3)*3)


class TestIntegral:

    def test_lagint(self):
        # check exceptions
        assert_raises(TypeError, lag.lagint, [0], .5)
        assert_raises(ValueError, lag.lagint, [0], -1)
        assert_raises(ValueError, lag.lagint, [0], 1, [0, 0])
        assert_raises(ValueError, lag.lagint, [0], lbnd=[0])
        assert_raises(ValueError, lag.lagint, [0], scl=[0])
        assert_raises(TypeError, lag.lagint, [0], axis=.5)

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0]*(i - 2) + [1]
            res = lag.lagint([0], m=i, k=k)
            assert_almost_equal(res, [1, -1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [1/scl]
            lagpol = lag.poly2lag(pol)
            lagint = lag.lagint(lagpol, m=1, k=[i])
            res = lag.lag2poly(lagint)
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0]*i + [1]
            lagpol = lag.poly2lag(pol)
            lagint = lag.lagint(lagpol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(lag.lagval(-1, lagint), i)

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [2/scl]
            lagpol = lag.poly2lag(pol)
            lagint = lag.lagint(lagpol, m=1, k=[i], scl=2)
            res = lag.lag2poly(lagint)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = lag.lagint(tgt, m=1)
                res = lag.lagint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = lag.lagint(tgt, m=1, k=[k])
                res = lag.lagint(pol, m=j, k=list(range(j)))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = lag.lagint(tgt, m=1, k=[k], lbnd=-1)
                res = lag.lagint(pol, m=j, k=list(range(j)), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = lag.lagint(tgt, m=1, k=[k], scl=2)
                res = lag.lagint(pol, m=j, k=list(range(j)), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_lagint_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([lag.lagint(c) for c in c2d.T]).T
        res = lag.lagint(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([lag.lagint(c) for c in c2d])
        res = lag.lagint(c2d, axis=1)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([lag.lagint(c, k=3) for c in c2d])
        res = lag.lagint(c2d, k=3, axis=1)
        assert_almost_equal(res, tgt)


class TestDerivative:

    def test_lagder(self):
        # check exceptions
        assert_raises(TypeError, lag.lagder, [0], .5)
        assert_raises(ValueError, lag.lagder, [0], -1)

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0]*i + [1]
            res = lag.lagder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = lag.lagder(lag.lagint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = lag.lagder(lag.lagint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_lagder_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([lag.lagder(c) for c in c2d.T]).T
        res = lag.lagder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([lag.lagder(c) for c in c2d])
        res = lag.lagder(c2d, axis=1)
        assert_almost_equal(res, tgt)


class TestVander:
    # some random values in [-1, 1)
    x = np.random.random((3, 5))*2 - 1

    def test_lagvander(self):
        # check for 1d x
        x = np.arange(3)
        v = lag.lagvander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], lag.lagval(x, coef))

        # check for 2d x
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = lag.lagvander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], lag.lagval(x, coef))

    def test_lagvander2d(self):
        # also tests lagval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = lag.lagvander2d(x1, x2, [1, 2])
        tgt = lag.lagval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = lag.lagvander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_lagvander3d(self):
        # also tests lagval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3, 4))
        van = lag.lagvander3d(x1, x2, x3, [1, 2, 3])
        tgt = lag.lagval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = lag.lagvander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))


class TestFitting:

    def test_lagfit(self):
        def f(x):
            return x*(x - 1)*(x - 2)

        # Test exceptions
        assert_raises(ValueError, lag.lagfit, [1], [1], -1)
        assert_raises(TypeError, lag.lagfit, [[1]], [1], 0)
        assert_raises(TypeError, lag.lagfit, [], [1], 0)
        assert_raises(TypeError, lag.lagfit, [1], [[[1]]], 0)
        assert_raises(TypeError, lag.lagfit, [1, 2], [1], 0)
        assert_raises(TypeError, lag.lagfit, [1], [1, 2], 0)
        assert_raises(TypeError, lag.lagfit, [1], [1], 0, w=[[1]])
        assert_raises(TypeError, lag.lagfit, [1], [1], 0, w=[1, 1])
        assert_raises(ValueError, lag.lagfit, [1], [1], [-1,])
        assert_raises(ValueError, lag.lagfit, [1], [1], [2, -1, 6])
        assert_raises(TypeError, lag.lagfit, [1], [1], [])

        # Test fit
        x = np.linspace(0, 2)
        y = f(x)
        #
        coef3 = lag.lagfit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(lag.lagval(x, coef3), y)
        coef3 = lag.lagfit(x, y, [0, 1, 2, 3])
        assert_equal(len(coef3), 4)
        assert_almost_equal(lag.lagval(x, coef3), y)
        #
        coef4 = lag.lagfit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(lag.lagval(x, coef4), y)
        coef4 = lag.lagfit(x, y, [0, 1, 2, 3, 4])
        assert_equal(len(coef4), 5)
        assert_almost_equal(lag.lagval(x, coef4), y)
        #
        coef2d = lag.lagfit(x, np.array([y, y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        coef2d = lag.lagfit(x, np.array([y, y]).T, [0, 1, 2, 3])
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = lag.lagfit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        wcoef3 = lag.lagfit(x, yw, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = lag.lagfit(x, np.array([yw, yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        wcoef2d = lag.lagfit(x, np.array([yw, yw]).T, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        assert_almost_equal(lag.lagfit(x, x, 1), [1, -1])
        assert_almost_equal(lag.lagfit(x, x, [0, 1]), [1, -1])


class TestCompanion:

    def test_raises(self):
        assert_raises(ValueError, lag.lagcompanion, [])
        assert_raises(ValueError, lag.lagcompanion, [1])

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0]*i + [1]
            assert_(lag.lagcompanion(coef).shape == (i, i))

    def test_linear_root(self):
        assert_(lag.lagcompanion([1, 2])[0, 0] == 1.5)


class TestGauss:

    def test_100(self):
        x, w = lag.laggauss(100)

        # test orthogonality. Note that the results need to be normalized,
        # otherwise the huge values that can arise from fast growing
        # functions like Laguerre can be very confusing.
        v = lag.lagvander(x, 99)
        vv = np.dot(v.T * w, v)
        vd = 1/np.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        assert_almost_equal(vv, np.eye(100))

        # check that the integral of 1 is correct
        tgt = 1.0
        assert_almost_equal(w.sum(), tgt)


class TestMisc:

    def test_lagfromroots(self):
        res = lag.lagfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            pol = lag.lagfromroots(roots)
            res = lag.lagval(roots, pol)
            tgt = 0
            assert_(len(pol) == i + 1)
            assert_almost_equal(lag.lag2poly(pol)[-1], 1)
            assert_almost_equal(res, tgt)

    def test_lagroots(self):
        assert_almost_equal(lag.lagroots([1]), [])
        assert_almost_equal(lag.lagroots([0, 1]), [1])
        for i in range(2, 5):
            tgt = np.linspace(0, 3, i)
            res = lag.lagroots(lag.lagfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_lagtrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, lag.lagtrim, coef, -1)

        # Test results
        assert_equal(lag.lagtrim(coef), coef[:-1])
        assert_equal(lag.lagtrim(coef, 1), coef[:-3])
        assert_equal(lag.lagtrim(coef, 2), [0])

    def test_lagline(self):
        assert_equal(lag.lagline(3, 4), [7, -4])

    def test_lag2poly(self):
        for i in range(7):
            assert_almost_equal(lag.lag2poly([0]*i + [1]), Llist[i])

    def test_poly2lag(self):
        for i in range(7):
            assert_almost_equal(lag.poly2lag(Llist[i]), [0]*i + [1])

    def test_weight(self):
        x = np.linspace(0, 10, 11)
        tgt = np.exp(-x)
        res = lag.lagweight(x)
        assert_almost_equal(res, tgt)
