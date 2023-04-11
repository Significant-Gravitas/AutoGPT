"""Tests for legendre module.

"""
from functools import reduce

import numpy as np
import numpy.polynomial.legendre as leg
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
    assert_almost_equal, assert_raises, assert_equal, assert_,
    )

L0 = np.array([1])
L1 = np.array([0, 1])
L2 = np.array([-1, 0, 3])/2
L3 = np.array([0, -3, 0, 5])/2
L4 = np.array([3, 0, -30, 0, 35])/8
L5 = np.array([0, 15, 0, -70, 0, 63])/8
L6 = np.array([-5, 0, 105, 0, -315, 0, 231])/16
L7 = np.array([0, -35, 0, 315, 0, -693, 0, 429])/16
L8 = np.array([35, 0, -1260, 0, 6930, 0, -12012, 0, 6435])/128
L9 = np.array([0, 315, 0, -4620, 0, 18018, 0, -25740, 0, 12155])/128

Llist = [L0, L1, L2, L3, L4, L5, L6, L7, L8, L9]


def trim(x):
    return leg.legtrim(x, tol=1e-6)


class TestConstants:

    def test_legdomain(self):
        assert_equal(leg.legdomain, [-1, 1])

    def test_legzero(self):
        assert_equal(leg.legzero, [0])

    def test_legone(self):
        assert_equal(leg.legone, [1])

    def test_legx(self):
        assert_equal(leg.legx, [0, 1])


class TestArithmetic:
    x = np.linspace(-1, 1, 100)

    def test_legadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = leg.legadd([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legsub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = leg.legsub([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legmulx(self):
        assert_equal(leg.legmulx([0]), [0])
        assert_equal(leg.legmulx([1]), [0, 1])
        for i in range(1, 5):
            tmp = 2*i + 1
            ser = [0]*i + [1]
            tgt = [0]*(i - 1) + [i/tmp, 0, (i + 1)/tmp]
            assert_equal(leg.legmulx(ser), tgt)

    def test_legmul(self):
        # check values of result
        for i in range(5):
            pol1 = [0]*i + [1]
            val1 = leg.legval(self.x, pol1)
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0]*j + [1]
                val2 = leg.legval(self.x, pol2)
                pol3 = leg.legmul(pol1, pol2)
                val3 = leg.legval(self.x, pol3)
                assert_(len(pol3) == i + j + 1, msg)
                assert_almost_equal(val3, val1*val2, err_msg=msg)

    def test_legdiv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0]*i + [1]
                cj = [0]*j + [1]
                tgt = leg.legadd(ci, cj)
                quo, rem = leg.legdiv(tgt, ci)
                res = leg.legadd(leg.legmul(quo, ci), rem)
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_legpow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = np.arange(i + 1)
                tgt = reduce(leg.legmul, [c]*j, np.array([1]))
                res = leg.legpow(c, j) 
                assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = np.array([2., 2., 2.])
    c2d = np.einsum('i,j->ij', c1d, c1d)
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = np.random.random((3, 5))*2 - 1
    y = polyval(x, [1., 2., 3.])

    def test_legval(self):
        #check empty input
        assert_equal(leg.legval([], [1]).size, 0)

        #check normal input)
        x = np.linspace(-1, 1)
        y = [polyval(x, c) for c in Llist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = leg.legval(x, [0]*i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        #check that shape is preserved
        for i in range(3):
            dims = [2]*i
            x = np.zeros(dims)
            assert_equal(leg.legval(x, [1]).shape, dims)
            assert_equal(leg.legval(x, [1, 0]).shape, dims)
            assert_equal(leg.legval(x, [1, 0, 0]).shape, dims)

    def test_legval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test exceptions
        assert_raises(ValueError, leg.legval2d, x1, x2[:2], self.c2d)

        #test values
        tgt = y1*y2
        res = leg.legval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = leg.legval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))

    def test_legval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test exceptions
        assert_raises(ValueError, leg.legval3d, x1, x2, x3[:2], self.c3d)

        #test values
        tgt = y1*y2*y3
        res = leg.legval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = leg.legval3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3))

    def test_leggrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test values
        tgt = np.einsum('i,j->ij', y1, y2)
        res = leg.leggrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = leg.leggrid2d(z, z, self.c2d)
        assert_(res.shape == (2, 3)*2)

    def test_leggrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test values
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        res = leg.leggrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = leg.leggrid3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3)*3)


class TestIntegral:

    def test_legint(self):
        # check exceptions
        assert_raises(TypeError, leg.legint, [0], .5)
        assert_raises(ValueError, leg.legint, [0], -1)
        assert_raises(ValueError, leg.legint, [0], 1, [0, 0])
        assert_raises(ValueError, leg.legint, [0], lbnd=[0])
        assert_raises(ValueError, leg.legint, [0], scl=[0])
        assert_raises(TypeError, leg.legint, [0], axis=.5)

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0]*(i - 2) + [1]
            res = leg.legint([0], m=i, k=k)
            assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [1/scl]
            legpol = leg.poly2leg(pol)
            legint = leg.legint(legpol, m=1, k=[i])
            res = leg.leg2poly(legint)
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0]*i + [1]
            legpol = leg.poly2leg(pol)
            legint = leg.legint(legpol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(leg.legval(-1, legint), i)

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [2/scl]
            legpol = leg.poly2leg(pol)
            legint = leg.legint(legpol, m=1, k=[i], scl=2)
            res = leg.leg2poly(legint)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = leg.legint(tgt, m=1)
                res = leg.legint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = leg.legint(tgt, m=1, k=[k])
                res = leg.legint(pol, m=j, k=list(range(j)))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = leg.legint(tgt, m=1, k=[k], lbnd=-1)
                res = leg.legint(pol, m=j, k=list(range(j)), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = leg.legint(tgt, m=1, k=[k], scl=2)
                res = leg.legint(pol, m=j, k=list(range(j)), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_legint_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([leg.legint(c) for c in c2d.T]).T
        res = leg.legint(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([leg.legint(c) for c in c2d])
        res = leg.legint(c2d, axis=1)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([leg.legint(c, k=3) for c in c2d])
        res = leg.legint(c2d, k=3, axis=1)
        assert_almost_equal(res, tgt)

    def test_legint_zerointord(self):
        assert_equal(leg.legint((1, 2, 3), 0), (1, 2, 3))


class TestDerivative:

    def test_legder(self):
        # check exceptions
        assert_raises(TypeError, leg.legder, [0], .5)
        assert_raises(ValueError, leg.legder, [0], -1)

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0]*i + [1]
            res = leg.legder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = leg.legder(leg.legint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = leg.legder(leg.legint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_legder_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([leg.legder(c) for c in c2d.T]).T
        res = leg.legder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([leg.legder(c) for c in c2d])
        res = leg.legder(c2d, axis=1)
        assert_almost_equal(res, tgt)

    def test_legder_orderhigherthancoeff(self):
        c = (1, 2, 3, 4)
        assert_equal(leg.legder(c, 4), [0])

class TestVander:
    # some random values in [-1, 1)
    x = np.random.random((3, 5))*2 - 1

    def test_legvander(self):
        # check for 1d x
        x = np.arange(3)
        v = leg.legvander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], leg.legval(x, coef))

        # check for 2d x
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = leg.legvander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], leg.legval(x, coef))

    def test_legvander2d(self):
        # also tests polyval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = leg.legvander2d(x1, x2, [1, 2])
        tgt = leg.legval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = leg.legvander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_legvander3d(self):
        # also tests polyval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3, 4))
        van = leg.legvander3d(x1, x2, x3, [1, 2, 3])
        tgt = leg.legval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = leg.legvander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))

    def test_legvander_negdeg(self):
        assert_raises(ValueError, leg.legvander, (1, 2, 3), -1)


class TestFitting:

    def test_legfit(self):
        def f(x):
            return x*(x - 1)*(x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        # Test exceptions
        assert_raises(ValueError, leg.legfit, [1], [1], -1)
        assert_raises(TypeError, leg.legfit, [[1]], [1], 0)
        assert_raises(TypeError, leg.legfit, [], [1], 0)
        assert_raises(TypeError, leg.legfit, [1], [[[1]]], 0)
        assert_raises(TypeError, leg.legfit, [1, 2], [1], 0)
        assert_raises(TypeError, leg.legfit, [1], [1, 2], 0)
        assert_raises(TypeError, leg.legfit, [1], [1], 0, w=[[1]])
        assert_raises(TypeError, leg.legfit, [1], [1], 0, w=[1, 1])
        assert_raises(ValueError, leg.legfit, [1], [1], [-1,])
        assert_raises(ValueError, leg.legfit, [1], [1], [2, -1, 6])
        assert_raises(TypeError, leg.legfit, [1], [1], [])

        # Test fit
        x = np.linspace(0, 2)
        y = f(x)
        #
        coef3 = leg.legfit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(leg.legval(x, coef3), y)
        coef3 = leg.legfit(x, y, [0, 1, 2, 3])
        assert_equal(len(coef3), 4)
        assert_almost_equal(leg.legval(x, coef3), y)
        #
        coef4 = leg.legfit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(leg.legval(x, coef4), y)
        coef4 = leg.legfit(x, y, [0, 1, 2, 3, 4])
        assert_equal(len(coef4), 5)
        assert_almost_equal(leg.legval(x, coef4), y)
        # check things still work if deg is not in strict increasing
        coef4 = leg.legfit(x, y, [2, 3, 4, 1, 0])
        assert_equal(len(coef4), 5)
        assert_almost_equal(leg.legval(x, coef4), y)
        #
        coef2d = leg.legfit(x, np.array([y, y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        coef2d = leg.legfit(x, np.array([y, y]).T, [0, 1, 2, 3])
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = leg.legfit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        wcoef3 = leg.legfit(x, yw, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = leg.legfit(x, np.array([yw, yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        wcoef2d = leg.legfit(x, np.array([yw, yw]).T, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        assert_almost_equal(leg.legfit(x, x, 1), [0, 1])
        assert_almost_equal(leg.legfit(x, x, [0, 1]), [0, 1])
        # test fitting only even Legendre polynomials
        x = np.linspace(-1, 1)
        y = f2(x)
        coef1 = leg.legfit(x, y, 4)
        assert_almost_equal(leg.legval(x, coef1), y)
        coef2 = leg.legfit(x, y, [0, 2, 4])
        assert_almost_equal(leg.legval(x, coef2), y)
        assert_almost_equal(coef1, coef2)


class TestCompanion:

    def test_raises(self):
        assert_raises(ValueError, leg.legcompanion, [])
        assert_raises(ValueError, leg.legcompanion, [1])

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0]*i + [1]
            assert_(leg.legcompanion(coef).shape == (i, i))

    def test_linear_root(self):
        assert_(leg.legcompanion([1, 2])[0, 0] == -.5)


class TestGauss:

    def test_100(self):
        x, w = leg.leggauss(100)

        # test orthogonality. Note that the results need to be normalized,
        # otherwise the huge values that can arise from fast growing
        # functions like Laguerre can be very confusing.
        v = leg.legvander(x, 99)
        vv = np.dot(v.T * w, v)
        vd = 1/np.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        assert_almost_equal(vv, np.eye(100))

        # check that the integral of 1 is correct
        tgt = 2.0
        assert_almost_equal(w.sum(), tgt)


class TestMisc:

    def test_legfromroots(self):
        res = leg.legfromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            pol = leg.legfromroots(roots)
            res = leg.legval(roots, pol)
            tgt = 0
            assert_(len(pol) == i + 1)
            assert_almost_equal(leg.leg2poly(pol)[-1], 1)
            assert_almost_equal(res, tgt)

    def test_legroots(self):
        assert_almost_equal(leg.legroots([1]), [])
        assert_almost_equal(leg.legroots([1, 2]), [-.5])
        for i in range(2, 5):
            tgt = np.linspace(-1, 1, i)
            res = leg.legroots(leg.legfromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_legtrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, leg.legtrim, coef, -1)

        # Test results
        assert_equal(leg.legtrim(coef), coef[:-1])
        assert_equal(leg.legtrim(coef, 1), coef[:-3])
        assert_equal(leg.legtrim(coef, 2), [0])

    def test_legline(self):
        assert_equal(leg.legline(3, 4), [3, 4])

    def test_legline_zeroscl(self):
        assert_equal(leg.legline(3, 0), [3])

    def test_leg2poly(self):
        for i in range(10):
            assert_almost_equal(leg.leg2poly([0]*i + [1]), Llist[i])

    def test_poly2leg(self):
        for i in range(10):
            assert_almost_equal(leg.poly2leg(Llist[i]), [0]*i + [1])

    def test_weight(self):
        x = np.linspace(-1, 1, 11)
        tgt = 1.
        res = leg.legweight(x)
        assert_almost_equal(res, tgt)
