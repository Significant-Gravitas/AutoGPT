"""Tests for hermite_e module.

"""
from functools import reduce

import numpy as np
import numpy.polynomial.hermite_e as herme
from numpy.polynomial.polynomial import polyval
from numpy.testing import (
    assert_almost_equal, assert_raises, assert_equal, assert_,
    )

He0 = np.array([1])
He1 = np.array([0, 1])
He2 = np.array([-1, 0, 1])
He3 = np.array([0, -3, 0, 1])
He4 = np.array([3, 0, -6, 0, 1])
He5 = np.array([0, 15, 0, -10, 0, 1])
He6 = np.array([-15, 0, 45, 0, -15, 0, 1])
He7 = np.array([0, -105, 0, 105, 0, -21, 0, 1])
He8 = np.array([105, 0, -420, 0, 210, 0, -28, 0, 1])
He9 = np.array([0, 945, 0, -1260, 0, 378, 0, -36, 0, 1])

Helist = [He0, He1, He2, He3, He4, He5, He6, He7, He8, He9]


def trim(x):
    return herme.hermetrim(x, tol=1e-6)


class TestConstants:

    def test_hermedomain(self):
        assert_equal(herme.hermedomain, [-1, 1])

    def test_hermezero(self):
        assert_equal(herme.hermezero, [0])

    def test_hermeone(self):
        assert_equal(herme.hermeone, [1])

    def test_hermex(self):
        assert_equal(herme.hermex, [0, 1])


class TestArithmetic:
    x = np.linspace(-3, 3, 100)

    def test_hermeadd(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] += 1
                res = herme.hermeadd([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermesub(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                tgt = np.zeros(max(i, j) + 1)
                tgt[i] += 1
                tgt[j] -= 1
                res = herme.hermesub([0]*i + [1], [0]*j + [1])
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermemulx(self):
        assert_equal(herme.hermemulx([0]), [0])
        assert_equal(herme.hermemulx([1]), [0, 1])
        for i in range(1, 5):
            ser = [0]*i + [1]
            tgt = [0]*(i - 1) + [i, 0, 1]
            assert_equal(herme.hermemulx(ser), tgt)

    def test_hermemul(self):
        # check values of result
        for i in range(5):
            pol1 = [0]*i + [1]
            val1 = herme.hermeval(self.x, pol1)
            for j in range(5):
                msg = f"At i={i}, j={j}"
                pol2 = [0]*j + [1]
                val2 = herme.hermeval(self.x, pol2)
                pol3 = herme.hermemul(pol1, pol2)
                val3 = herme.hermeval(self.x, pol3)
                assert_(len(pol3) == i + j + 1, msg)
                assert_almost_equal(val3, val1*val2, err_msg=msg)

    def test_hermediv(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                ci = [0]*i + [1]
                cj = [0]*j + [1]
                tgt = herme.hermeadd(ci, cj)
                quo, rem = herme.hermediv(tgt, ci)
                res = herme.hermeadd(herme.hermemul(quo, ci), rem)
                assert_equal(trim(res), trim(tgt), err_msg=msg)

    def test_hermepow(self):
        for i in range(5):
            for j in range(5):
                msg = f"At i={i}, j={j}"
                c = np.arange(i + 1)
                tgt = reduce(herme.hermemul, [c]*j, np.array([1]))
                res = herme.hermepow(c, j)
                assert_equal(trim(res), trim(tgt), err_msg=msg)


class TestEvaluation:
    # coefficients of 1 + 2*x + 3*x**2
    c1d = np.array([4., 2., 3.])
    c2d = np.einsum('i,j->ij', c1d, c1d)
    c3d = np.einsum('i,j,k->ijk', c1d, c1d, c1d)

    # some random values in [-1, 1)
    x = np.random.random((3, 5))*2 - 1
    y = polyval(x, [1., 2., 3.])

    def test_hermeval(self):
        #check empty input
        assert_equal(herme.hermeval([], [1]).size, 0)

        #check normal input)
        x = np.linspace(-1, 1)
        y = [polyval(x, c) for c in Helist]
        for i in range(10):
            msg = f"At i={i}"
            tgt = y[i]
            res = herme.hermeval(x, [0]*i + [1])
            assert_almost_equal(res, tgt, err_msg=msg)

        #check that shape is preserved
        for i in range(3):
            dims = [2]*i
            x = np.zeros(dims)
            assert_equal(herme.hermeval(x, [1]).shape, dims)
            assert_equal(herme.hermeval(x, [1, 0]).shape, dims)
            assert_equal(herme.hermeval(x, [1, 0, 0]).shape, dims)

    def test_hermeval2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test exceptions
        assert_raises(ValueError, herme.hermeval2d, x1, x2[:2], self.c2d)

        #test values
        tgt = y1*y2
        res = herme.hermeval2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = herme.hermeval2d(z, z, self.c2d)
        assert_(res.shape == (2, 3))

    def test_hermeval3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test exceptions
        assert_raises(ValueError, herme.hermeval3d, x1, x2, x3[:2], self.c3d)

        #test values
        tgt = y1*y2*y3
        res = herme.hermeval3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = herme.hermeval3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3))

    def test_hermegrid2d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test values
        tgt = np.einsum('i,j->ij', y1, y2)
        res = herme.hermegrid2d(x1, x2, self.c2d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = herme.hermegrid2d(z, z, self.c2d)
        assert_(res.shape == (2, 3)*2)

    def test_hermegrid3d(self):
        x1, x2, x3 = self.x
        y1, y2, y3 = self.y

        #test values
        tgt = np.einsum('i,j,k->ijk', y1, y2, y3)
        res = herme.hermegrid3d(x1, x2, x3, self.c3d)
        assert_almost_equal(res, tgt)

        #test shape
        z = np.ones((2, 3))
        res = herme.hermegrid3d(z, z, z, self.c3d)
        assert_(res.shape == (2, 3)*3)


class TestIntegral:

    def test_hermeint(self):
        # check exceptions
        assert_raises(TypeError, herme.hermeint, [0], .5)
        assert_raises(ValueError, herme.hermeint, [0], -1)
        assert_raises(ValueError, herme.hermeint, [0], 1, [0, 0])
        assert_raises(ValueError, herme.hermeint, [0], lbnd=[0])
        assert_raises(ValueError, herme.hermeint, [0], scl=[0])
        assert_raises(TypeError, herme.hermeint, [0], axis=.5)

        # test integration of zero polynomial
        for i in range(2, 5):
            k = [0]*(i - 2) + [1]
            res = herme.hermeint([0], m=i, k=k)
            assert_almost_equal(res, [0, 1])

        # check single integration with integration constant
        for i in range(5):
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [1/scl]
            hermepol = herme.poly2herme(pol)
            hermeint = herme.hermeint(hermepol, m=1, k=[i])
            res = herme.herme2poly(hermeint)
            assert_almost_equal(trim(res), trim(tgt))

        # check single integration with integration constant and lbnd
        for i in range(5):
            scl = i + 1
            pol = [0]*i + [1]
            hermepol = herme.poly2herme(pol)
            hermeint = herme.hermeint(hermepol, m=1, k=[i], lbnd=-1)
            assert_almost_equal(herme.hermeval(-1, hermeint), i)

        # check single integration with integration constant and scaling
        for i in range(5):
            scl = i + 1
            pol = [0]*i + [1]
            tgt = [i] + [0]*i + [2/scl]
            hermepol = herme.poly2herme(pol)
            hermeint = herme.hermeint(hermepol, m=1, k=[i], scl=2)
            res = herme.herme2poly(hermeint)
            assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with default k
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = herme.hermeint(tgt, m=1)
                res = herme.hermeint(pol, m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with defined k
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = herme.hermeint(tgt, m=1, k=[k])
                res = herme.hermeint(pol, m=j, k=list(range(j)))
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with lbnd
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = herme.hermeint(tgt, m=1, k=[k], lbnd=-1)
                res = herme.hermeint(pol, m=j, k=list(range(j)), lbnd=-1)
                assert_almost_equal(trim(res), trim(tgt))

        # check multiple integrations with scaling
        for i in range(5):
            for j in range(2, 5):
                pol = [0]*i + [1]
                tgt = pol[:]
                for k in range(j):
                    tgt = herme.hermeint(tgt, m=1, k=[k], scl=2)
                res = herme.hermeint(pol, m=j, k=list(range(j)), scl=2)
                assert_almost_equal(trim(res), trim(tgt))

    def test_hermeint_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([herme.hermeint(c) for c in c2d.T]).T
        res = herme.hermeint(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([herme.hermeint(c) for c in c2d])
        res = herme.hermeint(c2d, axis=1)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([herme.hermeint(c, k=3) for c in c2d])
        res = herme.hermeint(c2d, k=3, axis=1)
        assert_almost_equal(res, tgt)


class TestDerivative:

    def test_hermeder(self):
        # check exceptions
        assert_raises(TypeError, herme.hermeder, [0], .5)
        assert_raises(ValueError, herme.hermeder, [0], -1)

        # check that zeroth derivative does nothing
        for i in range(5):
            tgt = [0]*i + [1]
            res = herme.hermeder(tgt, m=0)
            assert_equal(trim(res), trim(tgt))

        # check that derivation is the inverse of integration
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = herme.hermeder(herme.hermeint(tgt, m=j), m=j)
                assert_almost_equal(trim(res), trim(tgt))

        # check derivation with scaling
        for i in range(5):
            for j in range(2, 5):
                tgt = [0]*i + [1]
                res = herme.hermeder(
                    herme.hermeint(tgt, m=j, scl=2), m=j, scl=.5)
                assert_almost_equal(trim(res), trim(tgt))

    def test_hermeder_axis(self):
        # check that axis keyword works
        c2d = np.random.random((3, 4))

        tgt = np.vstack([herme.hermeder(c) for c in c2d.T]).T
        res = herme.hermeder(c2d, axis=0)
        assert_almost_equal(res, tgt)

        tgt = np.vstack([herme.hermeder(c) for c in c2d])
        res = herme.hermeder(c2d, axis=1)
        assert_almost_equal(res, tgt)


class TestVander:
    # some random values in [-1, 1)
    x = np.random.random((3, 5))*2 - 1

    def test_hermevander(self):
        # check for 1d x
        x = np.arange(3)
        v = herme.hermevander(x, 3)
        assert_(v.shape == (3, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], herme.hermeval(x, coef))

        # check for 2d x
        x = np.array([[1, 2], [3, 4], [5, 6]])
        v = herme.hermevander(x, 3)
        assert_(v.shape == (3, 2, 4))
        for i in range(4):
            coef = [0]*i + [1]
            assert_almost_equal(v[..., i], herme.hermeval(x, coef))

    def test_hermevander2d(self):
        # also tests hermeval2d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3))
        van = herme.hermevander2d(x1, x2, [1, 2])
        tgt = herme.hermeval2d(x1, x2, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = herme.hermevander2d([x1], [x2], [1, 2])
        assert_(van.shape == (1, 5, 6))

    def test_hermevander3d(self):
        # also tests hermeval3d for non-square coefficient array
        x1, x2, x3 = self.x
        c = np.random.random((2, 3, 4))
        van = herme.hermevander3d(x1, x2, x3, [1, 2, 3])
        tgt = herme.hermeval3d(x1, x2, x3, c)
        res = np.dot(van, c.flat)
        assert_almost_equal(res, tgt)

        # check shape
        van = herme.hermevander3d([x1], [x2], [x3], [1, 2, 3])
        assert_(van.shape == (1, 5, 24))


class TestFitting:

    def test_hermefit(self):
        def f(x):
            return x*(x - 1)*(x - 2)

        def f2(x):
            return x**4 + x**2 + 1

        # Test exceptions
        assert_raises(ValueError, herme.hermefit, [1], [1], -1)
        assert_raises(TypeError, herme.hermefit, [[1]], [1], 0)
        assert_raises(TypeError, herme.hermefit, [], [1], 0)
        assert_raises(TypeError, herme.hermefit, [1], [[[1]]], 0)
        assert_raises(TypeError, herme.hermefit, [1, 2], [1], 0)
        assert_raises(TypeError, herme.hermefit, [1], [1, 2], 0)
        assert_raises(TypeError, herme.hermefit, [1], [1], 0, w=[[1]])
        assert_raises(TypeError, herme.hermefit, [1], [1], 0, w=[1, 1])
        assert_raises(ValueError, herme.hermefit, [1], [1], [-1,])
        assert_raises(ValueError, herme.hermefit, [1], [1], [2, -1, 6])
        assert_raises(TypeError, herme.hermefit, [1], [1], [])

        # Test fit
        x = np.linspace(0, 2)
        y = f(x)
        #
        coef3 = herme.hermefit(x, y, 3)
        assert_equal(len(coef3), 4)
        assert_almost_equal(herme.hermeval(x, coef3), y)
        coef3 = herme.hermefit(x, y, [0, 1, 2, 3])
        assert_equal(len(coef3), 4)
        assert_almost_equal(herme.hermeval(x, coef3), y)
        #
        coef4 = herme.hermefit(x, y, 4)
        assert_equal(len(coef4), 5)
        assert_almost_equal(herme.hermeval(x, coef4), y)
        coef4 = herme.hermefit(x, y, [0, 1, 2, 3, 4])
        assert_equal(len(coef4), 5)
        assert_almost_equal(herme.hermeval(x, coef4), y)
        # check things still work if deg is not in strict increasing
        coef4 = herme.hermefit(x, y, [2, 3, 4, 1, 0])
        assert_equal(len(coef4), 5)
        assert_almost_equal(herme.hermeval(x, coef4), y)
        #
        coef2d = herme.hermefit(x, np.array([y, y]).T, 3)
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        coef2d = herme.hermefit(x, np.array([y, y]).T, [0, 1, 2, 3])
        assert_almost_equal(coef2d, np.array([coef3, coef3]).T)
        # test weighting
        w = np.zeros_like(x)
        yw = y.copy()
        w[1::2] = 1
        y[0::2] = 0
        wcoef3 = herme.hermefit(x, yw, 3, w=w)
        assert_almost_equal(wcoef3, coef3)
        wcoef3 = herme.hermefit(x, yw, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef3, coef3)
        #
        wcoef2d = herme.hermefit(x, np.array([yw, yw]).T, 3, w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        wcoef2d = herme.hermefit(x, np.array([yw, yw]).T, [0, 1, 2, 3], w=w)
        assert_almost_equal(wcoef2d, np.array([coef3, coef3]).T)
        # test scaling with complex values x points whose square
        # is zero when summed.
        x = [1, 1j, -1, -1j]
        assert_almost_equal(herme.hermefit(x, x, 1), [0, 1])
        assert_almost_equal(herme.hermefit(x, x, [0, 1]), [0, 1])
        # test fitting only even Legendre polynomials
        x = np.linspace(-1, 1)
        y = f2(x)
        coef1 = herme.hermefit(x, y, 4)
        assert_almost_equal(herme.hermeval(x, coef1), y)
        coef2 = herme.hermefit(x, y, [0, 2, 4])
        assert_almost_equal(herme.hermeval(x, coef2), y)
        assert_almost_equal(coef1, coef2)


class TestCompanion:

    def test_raises(self):
        assert_raises(ValueError, herme.hermecompanion, [])
        assert_raises(ValueError, herme.hermecompanion, [1])

    def test_dimensions(self):
        for i in range(1, 5):
            coef = [0]*i + [1]
            assert_(herme.hermecompanion(coef).shape == (i, i))

    def test_linear_root(self):
        assert_(herme.hermecompanion([1, 2])[0, 0] == -.5)


class TestGauss:

    def test_100(self):
        x, w = herme.hermegauss(100)

        # test orthogonality. Note that the results need to be normalized,
        # otherwise the huge values that can arise from fast growing
        # functions like Laguerre can be very confusing.
        v = herme.hermevander(x, 99)
        vv = np.dot(v.T * w, v)
        vd = 1/np.sqrt(vv.diagonal())
        vv = vd[:, None] * vv * vd
        assert_almost_equal(vv, np.eye(100))

        # check that the integral of 1 is correct
        tgt = np.sqrt(2*np.pi)
        assert_almost_equal(w.sum(), tgt)


class TestMisc:

    def test_hermefromroots(self):
        res = herme.hermefromroots([])
        assert_almost_equal(trim(res), [1])
        for i in range(1, 5):
            roots = np.cos(np.linspace(-np.pi, 0, 2*i + 1)[1::2])
            pol = herme.hermefromroots(roots)
            res = herme.hermeval(roots, pol)
            tgt = 0
            assert_(len(pol) == i + 1)
            assert_almost_equal(herme.herme2poly(pol)[-1], 1)
            assert_almost_equal(res, tgt)

    def test_hermeroots(self):
        assert_almost_equal(herme.hermeroots([1]), [])
        assert_almost_equal(herme.hermeroots([1, 1]), [-1])
        for i in range(2, 5):
            tgt = np.linspace(-1, 1, i)
            res = herme.hermeroots(herme.hermefromroots(tgt))
            assert_almost_equal(trim(res), trim(tgt))

    def test_hermetrim(self):
        coef = [2, -1, 1, 0]

        # Test exceptions
        assert_raises(ValueError, herme.hermetrim, coef, -1)

        # Test results
        assert_equal(herme.hermetrim(coef), coef[:-1])
        assert_equal(herme.hermetrim(coef, 1), coef[:-3])
        assert_equal(herme.hermetrim(coef, 2), [0])

    def test_hermeline(self):
        assert_equal(herme.hermeline(3, 4), [3, 4])

    def test_herme2poly(self):
        for i in range(10):
            assert_almost_equal(herme.herme2poly([0]*i + [1]), Helist[i])

    def test_poly2herme(self):
        for i in range(10):
            assert_almost_equal(herme.poly2herme(Helist[i]), [0]*i + [1])

    def test_weight(self):
        x = np.linspace(-5, 5, 11)
        tgt = np.exp(-.5*x**2)
        res = herme.hermeweight(x)
        assert_almost_equal(res, tgt)
