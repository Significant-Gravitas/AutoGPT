"""Test inter-conversion of different polynomial classes.

This tests the convert and cast methods of all the polynomial classes.

"""
import operator as op
from numbers import Number

import pytest
import numpy as np
from numpy.polynomial import (
    Polynomial, Legendre, Chebyshev, Laguerre, Hermite, HermiteE)
from numpy.testing import (
    assert_almost_equal, assert_raises, assert_equal, assert_,
    )
from numpy.polynomial.polyutils import RankWarning

#
# fixtures
#

classes = (
    Polynomial, Legendre, Chebyshev, Laguerre,
    Hermite, HermiteE
    )
classids = tuple(cls.__name__ for cls in classes)

@pytest.fixture(params=classes, ids=classids)
def Poly(request):
    return request.param

#
# helper functions
#
random = np.random.random


def assert_poly_almost_equal(p1, p2, msg=""):
    try:
        assert_(np.all(p1.domain == p2.domain))
        assert_(np.all(p1.window == p2.window))
        assert_almost_equal(p1.coef, p2.coef)
    except AssertionError:
        msg = f"Result: {p1}\nTarget: {p2}"
        raise AssertionError(msg)


#
# Test conversion methods that depend on combinations of two classes.
#

Poly1 = Poly
Poly2 = Poly


def test_conversion(Poly1, Poly2):
    x = np.linspace(0, 1, 10)
    coef = random((3,))

    d1 = Poly1.domain + random((2,))*.25
    w1 = Poly1.window + random((2,))*.25
    p1 = Poly1(coef, domain=d1, window=w1)

    d2 = Poly2.domain + random((2,))*.25
    w2 = Poly2.window + random((2,))*.25
    p2 = p1.convert(kind=Poly2, domain=d2, window=w2)

    assert_almost_equal(p2.domain, d2)
    assert_almost_equal(p2.window, w2)
    assert_almost_equal(p2(x), p1(x))


def test_cast(Poly1, Poly2):
    x = np.linspace(0, 1, 10)
    coef = random((3,))

    d1 = Poly1.domain + random((2,))*.25
    w1 = Poly1.window + random((2,))*.25
    p1 = Poly1(coef, domain=d1, window=w1)

    d2 = Poly2.domain + random((2,))*.25
    w2 = Poly2.window + random((2,))*.25
    p2 = Poly2.cast(p1, domain=d2, window=w2)

    assert_almost_equal(p2.domain, d2)
    assert_almost_equal(p2.window, w2)
    assert_almost_equal(p2(x), p1(x))


#
# test methods that depend on one class
#


def test_identity(Poly):
    d = Poly.domain + random((2,))*.25
    w = Poly.window + random((2,))*.25
    x = np.linspace(d[0], d[1], 11)
    p = Poly.identity(domain=d, window=w)
    assert_equal(p.domain, d)
    assert_equal(p.window, w)
    assert_almost_equal(p(x), x)


def test_basis(Poly):
    d = Poly.domain + random((2,))*.25
    w = Poly.window + random((2,))*.25
    p = Poly.basis(5, domain=d, window=w)
    assert_equal(p.domain, d)
    assert_equal(p.window, w)
    assert_equal(p.coef, [0]*5 + [1])


def test_fromroots(Poly):
    # check that requested roots are zeros of a polynomial
    # of correct degree, domain, and window.
    d = Poly.domain + random((2,))*.25
    w = Poly.window + random((2,))*.25
    r = random((5,))
    p1 = Poly.fromroots(r, domain=d, window=w)
    assert_equal(p1.degree(), len(r))
    assert_equal(p1.domain, d)
    assert_equal(p1.window, w)
    assert_almost_equal(p1(r), 0)

    # check that polynomial is monic
    pdom = Polynomial.domain
    pwin = Polynomial.window
    p2 = Polynomial.cast(p1, domain=pdom, window=pwin)
    assert_almost_equal(p2.coef[-1], 1)


def test_bad_conditioned_fit(Poly):

    x = [0., 0., 1.]
    y = [1., 2., 3.]

    # check RankWarning is raised
    with pytest.warns(RankWarning) as record:
        Poly.fit(x, y, 2)
    assert record[0].message.args[0] == "The fit may be poorly conditioned"


def test_fit(Poly):

    def f(x):
        return x*(x - 1)*(x - 2)
    x = np.linspace(0, 3)
    y = f(x)

    # check default value of domain and window
    p = Poly.fit(x, y, 3)
    assert_almost_equal(p.domain, [0, 3])
    assert_almost_equal(p(x), y)
    assert_equal(p.degree(), 3)

    # check with given domains and window
    d = Poly.domain + random((2,))*.25
    w = Poly.window + random((2,))*.25
    p = Poly.fit(x, y, 3, domain=d, window=w)
    assert_almost_equal(p(x), y)
    assert_almost_equal(p.domain, d)
    assert_almost_equal(p.window, w)
    p = Poly.fit(x, y, [0, 1, 2, 3], domain=d, window=w)
    assert_almost_equal(p(x), y)
    assert_almost_equal(p.domain, d)
    assert_almost_equal(p.window, w)

    # check with class domain default
    p = Poly.fit(x, y, 3, [])
    assert_equal(p.domain, Poly.domain)
    assert_equal(p.window, Poly.window)
    p = Poly.fit(x, y, [0, 1, 2, 3], [])
    assert_equal(p.domain, Poly.domain)
    assert_equal(p.window, Poly.window)

    # check that fit accepts weights.
    w = np.zeros_like(x)
    z = y + random(y.shape)*.25
    w[::2] = 1
    p1 = Poly.fit(x[::2], z[::2], 3)
    p2 = Poly.fit(x, z, 3, w=w)
    p3 = Poly.fit(x, z, [0, 1, 2, 3], w=w)
    assert_almost_equal(p1(x), p2(x))
    assert_almost_equal(p2(x), p3(x))


def test_equal(Poly):
    p1 = Poly([1, 2, 3], domain=[0, 1], window=[2, 3])
    p2 = Poly([1, 1, 1], domain=[0, 1], window=[2, 3])
    p3 = Poly([1, 2, 3], domain=[1, 2], window=[2, 3])
    p4 = Poly([1, 2, 3], domain=[0, 1], window=[1, 2])
    assert_(p1 == p1)
    assert_(not p1 == p2)
    assert_(not p1 == p3)
    assert_(not p1 == p4)


def test_not_equal(Poly):
    p1 = Poly([1, 2, 3], domain=[0, 1], window=[2, 3])
    p2 = Poly([1, 1, 1], domain=[0, 1], window=[2, 3])
    p3 = Poly([1, 2, 3], domain=[1, 2], window=[2, 3])
    p4 = Poly([1, 2, 3], domain=[0, 1], window=[1, 2])
    assert_(not p1 != p1)
    assert_(p1 != p2)
    assert_(p1 != p3)
    assert_(p1 != p4)


def test_add(Poly):
    # This checks commutation, not numerical correctness
    c1 = list(random((4,)) + .5)
    c2 = list(random((3,)) + .5)
    p1 = Poly(c1)
    p2 = Poly(c2)
    p3 = p1 + p2
    assert_poly_almost_equal(p2 + p1, p3)
    assert_poly_almost_equal(p1 + c2, p3)
    assert_poly_almost_equal(c2 + p1, p3)
    assert_poly_almost_equal(p1 + tuple(c2), p3)
    assert_poly_almost_equal(tuple(c2) + p1, p3)
    assert_poly_almost_equal(p1 + np.array(c2), p3)
    assert_poly_almost_equal(np.array(c2) + p1, p3)
    assert_raises(TypeError, op.add, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(TypeError, op.add, p1, Poly([0], window=Poly.window + 1))
    if Poly is Polynomial:
        assert_raises(TypeError, op.add, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, op.add, p1, Polynomial([0]))


def test_sub(Poly):
    # This checks commutation, not numerical correctness
    c1 = list(random((4,)) + .5)
    c2 = list(random((3,)) + .5)
    p1 = Poly(c1)
    p2 = Poly(c2)
    p3 = p1 - p2
    assert_poly_almost_equal(p2 - p1, -p3)
    assert_poly_almost_equal(p1 - c2, p3)
    assert_poly_almost_equal(c2 - p1, -p3)
    assert_poly_almost_equal(p1 - tuple(c2), p3)
    assert_poly_almost_equal(tuple(c2) - p1, -p3)
    assert_poly_almost_equal(p1 - np.array(c2), p3)
    assert_poly_almost_equal(np.array(c2) - p1, -p3)
    assert_raises(TypeError, op.sub, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(TypeError, op.sub, p1, Poly([0], window=Poly.window + 1))
    if Poly is Polynomial:
        assert_raises(TypeError, op.sub, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, op.sub, p1, Polynomial([0]))


def test_mul(Poly):
    c1 = list(random((4,)) + .5)
    c2 = list(random((3,)) + .5)
    p1 = Poly(c1)
    p2 = Poly(c2)
    p3 = p1 * p2
    assert_poly_almost_equal(p2 * p1, p3)
    assert_poly_almost_equal(p1 * c2, p3)
    assert_poly_almost_equal(c2 * p1, p3)
    assert_poly_almost_equal(p1 * tuple(c2), p3)
    assert_poly_almost_equal(tuple(c2) * p1, p3)
    assert_poly_almost_equal(p1 * np.array(c2), p3)
    assert_poly_almost_equal(np.array(c2) * p1, p3)
    assert_poly_almost_equal(p1 * 2, p1 * Poly([2]))
    assert_poly_almost_equal(2 * p1, p1 * Poly([2]))
    assert_raises(TypeError, op.mul, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(TypeError, op.mul, p1, Poly([0], window=Poly.window + 1))
    if Poly is Polynomial:
        assert_raises(TypeError, op.mul, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, op.mul, p1, Polynomial([0]))


def test_floordiv(Poly):
    c1 = list(random((4,)) + .5)
    c2 = list(random((3,)) + .5)
    c3 = list(random((2,)) + .5)
    p1 = Poly(c1)
    p2 = Poly(c2)
    p3 = Poly(c3)
    p4 = p1 * p2 + p3
    c4 = list(p4.coef)
    assert_poly_almost_equal(p4 // p2, p1)
    assert_poly_almost_equal(p4 // c2, p1)
    assert_poly_almost_equal(c4 // p2, p1)
    assert_poly_almost_equal(p4 // tuple(c2), p1)
    assert_poly_almost_equal(tuple(c4) // p2, p1)
    assert_poly_almost_equal(p4 // np.array(c2), p1)
    assert_poly_almost_equal(np.array(c4) // p2, p1)
    assert_poly_almost_equal(2 // p2, Poly([0]))
    assert_poly_almost_equal(p2 // 2, 0.5*p2)
    assert_raises(
        TypeError, op.floordiv, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(
        TypeError, op.floordiv, p1, Poly([0], window=Poly.window + 1))
    if Poly is Polynomial:
        assert_raises(TypeError, op.floordiv, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, op.floordiv, p1, Polynomial([0]))


def test_truediv(Poly):
    # true division is valid only if the denominator is a Number and
    # not a python bool.
    p1 = Poly([1,2,3])
    p2 = p1 * 5

    for stype in np.ScalarType:
        if not issubclass(stype, Number) or issubclass(stype, bool):
            continue
        s = stype(5)
        assert_poly_almost_equal(op.truediv(p2, s), p1)
        assert_raises(TypeError, op.truediv, s, p2)
    for stype in (int, float):
        s = stype(5)
        assert_poly_almost_equal(op.truediv(p2, s), p1)
        assert_raises(TypeError, op.truediv, s, p2)
    for stype in [complex]:
        s = stype(5, 0)
        assert_poly_almost_equal(op.truediv(p2, s), p1)
        assert_raises(TypeError, op.truediv, s, p2)
    for s in [tuple(), list(), dict(), bool(), np.array([1])]:
        assert_raises(TypeError, op.truediv, p2, s)
        assert_raises(TypeError, op.truediv, s, p2)
    for ptype in classes:
        assert_raises(TypeError, op.truediv, p2, ptype(1))


def test_mod(Poly):
    # This checks commutation, not numerical correctness
    c1 = list(random((4,)) + .5)
    c2 = list(random((3,)) + .5)
    c3 = list(random((2,)) + .5)
    p1 = Poly(c1)
    p2 = Poly(c2)
    p3 = Poly(c3)
    p4 = p1 * p2 + p3
    c4 = list(p4.coef)
    assert_poly_almost_equal(p4 % p2, p3)
    assert_poly_almost_equal(p4 % c2, p3)
    assert_poly_almost_equal(c4 % p2, p3)
    assert_poly_almost_equal(p4 % tuple(c2), p3)
    assert_poly_almost_equal(tuple(c4) % p2, p3)
    assert_poly_almost_equal(p4 % np.array(c2), p3)
    assert_poly_almost_equal(np.array(c4) % p2, p3)
    assert_poly_almost_equal(2 % p2, Poly([2]))
    assert_poly_almost_equal(p2 % 2, Poly([0]))
    assert_raises(TypeError, op.mod, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(TypeError, op.mod, p1, Poly([0], window=Poly.window + 1))
    if Poly is Polynomial:
        assert_raises(TypeError, op.mod, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, op.mod, p1, Polynomial([0]))


def test_divmod(Poly):
    # This checks commutation, not numerical correctness
    c1 = list(random((4,)) + .5)
    c2 = list(random((3,)) + .5)
    c3 = list(random((2,)) + .5)
    p1 = Poly(c1)
    p2 = Poly(c2)
    p3 = Poly(c3)
    p4 = p1 * p2 + p3
    c4 = list(p4.coef)
    quo, rem = divmod(p4, p2)
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(p4, c2)
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(c4, p2)
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(p4, tuple(c2))
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(tuple(c4), p2)
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(p4, np.array(c2))
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(np.array(c4), p2)
    assert_poly_almost_equal(quo, p1)
    assert_poly_almost_equal(rem, p3)
    quo, rem = divmod(p2, 2)
    assert_poly_almost_equal(quo, 0.5*p2)
    assert_poly_almost_equal(rem, Poly([0]))
    quo, rem = divmod(2, p2)
    assert_poly_almost_equal(quo, Poly([0]))
    assert_poly_almost_equal(rem, Poly([2]))
    assert_raises(TypeError, divmod, p1, Poly([0], domain=Poly.domain + 1))
    assert_raises(TypeError, divmod, p1, Poly([0], window=Poly.window + 1))
    if Poly is Polynomial:
        assert_raises(TypeError, divmod, p1, Chebyshev([0]))
    else:
        assert_raises(TypeError, divmod, p1, Polynomial([0]))


def test_roots(Poly):
    d = Poly.domain * 1.25 + .25
    w = Poly.window
    tgt = np.linspace(d[0], d[1], 5)
    res = np.sort(Poly.fromroots(tgt, domain=d, window=w).roots())
    assert_almost_equal(res, tgt)
    # default domain and window
    res = np.sort(Poly.fromroots(tgt).roots())
    assert_almost_equal(res, tgt)


def test_degree(Poly):
    p = Poly.basis(5)
    assert_equal(p.degree(), 5)


def test_copy(Poly):
    p1 = Poly.basis(5)
    p2 = p1.copy()
    assert_(p1 == p2)
    assert_(p1 is not p2)
    assert_(p1.coef is not p2.coef)
    assert_(p1.domain is not p2.domain)
    assert_(p1.window is not p2.window)


def test_integ(Poly):
    P = Polynomial
    # Check defaults
    p0 = Poly.cast(P([1*2, 2*3, 3*4]))
    p1 = P.cast(p0.integ())
    p2 = P.cast(p0.integ(2))
    assert_poly_almost_equal(p1, P([0, 2, 3, 4]))
    assert_poly_almost_equal(p2, P([0, 0, 1, 1, 1]))
    # Check with k
    p0 = Poly.cast(P([1*2, 2*3, 3*4]))
    p1 = P.cast(p0.integ(k=1))
    p2 = P.cast(p0.integ(2, k=[1, 1]))
    assert_poly_almost_equal(p1, P([1, 2, 3, 4]))
    assert_poly_almost_equal(p2, P([1, 1, 1, 1, 1]))
    # Check with lbnd
    p0 = Poly.cast(P([1*2, 2*3, 3*4]))
    p1 = P.cast(p0.integ(lbnd=1))
    p2 = P.cast(p0.integ(2, lbnd=1))
    assert_poly_almost_equal(p1, P([-9, 2, 3, 4]))
    assert_poly_almost_equal(p2, P([6, -9, 1, 1, 1]))
    # Check scaling
    d = 2*Poly.domain
    p0 = Poly.cast(P([1*2, 2*3, 3*4]), domain=d)
    p1 = P.cast(p0.integ())
    p2 = P.cast(p0.integ(2))
    assert_poly_almost_equal(p1, P([0, 2, 3, 4]))
    assert_poly_almost_equal(p2, P([0, 0, 1, 1, 1]))


def test_deriv(Poly):
    # Check that the derivative is the inverse of integration. It is
    # assumes that the integration has been checked elsewhere.
    d = Poly.domain + random((2,))*.25
    w = Poly.window + random((2,))*.25
    p1 = Poly([1, 2, 3], domain=d, window=w)
    p2 = p1.integ(2, k=[1, 2])
    p3 = p1.integ(1, k=[1])
    assert_almost_equal(p2.deriv(1).coef, p3.coef)
    assert_almost_equal(p2.deriv(2).coef, p1.coef)
    # default domain and window
    p1 = Poly([1, 2, 3])
    p2 = p1.integ(2, k=[1, 2])
    p3 = p1.integ(1, k=[1])
    assert_almost_equal(p2.deriv(1).coef, p3.coef)
    assert_almost_equal(p2.deriv(2).coef, p1.coef)


def test_linspace(Poly):
    d = Poly.domain + random((2,))*.25
    w = Poly.window + random((2,))*.25
    p = Poly([1, 2, 3], domain=d, window=w)
    # check default domain
    xtgt = np.linspace(d[0], d[1], 20)
    ytgt = p(xtgt)
    xres, yres = p.linspace(20)
    assert_almost_equal(xres, xtgt)
    assert_almost_equal(yres, ytgt)
    # check specified domain
    xtgt = np.linspace(0, 2, 20)
    ytgt = p(xtgt)
    xres, yres = p.linspace(20, domain=[0, 2])
    assert_almost_equal(xres, xtgt)
    assert_almost_equal(yres, ytgt)


def test_pow(Poly):
    d = Poly.domain + random((2,))*.25
    w = Poly.window + random((2,))*.25
    tgt = Poly([1], domain=d, window=w)
    tst = Poly([1, 2, 3], domain=d, window=w)
    for i in range(5):
        assert_poly_almost_equal(tst**i, tgt)
        tgt = tgt * tst
    # default domain and window
    tgt = Poly([1])
    tst = Poly([1, 2, 3])
    for i in range(5):
        assert_poly_almost_equal(tst**i, tgt)
        tgt = tgt * tst
    # check error for invalid powers
    assert_raises(ValueError, op.pow, tgt, 1.5)
    assert_raises(ValueError, op.pow, tgt, -1)


def test_call(Poly):
    P = Polynomial
    d = Poly.domain
    x = np.linspace(d[0], d[1], 11)

    # Check defaults
    p = Poly.cast(P([1, 2, 3]))
    tgt = 1 + x*(2 + 3*x)
    res = p(x)
    assert_almost_equal(res, tgt)


def test_cutdeg(Poly):
    p = Poly([1, 2, 3])
    assert_raises(ValueError, p.cutdeg, .5)
    assert_raises(ValueError, p.cutdeg, -1)
    assert_equal(len(p.cutdeg(3)), 3)
    assert_equal(len(p.cutdeg(2)), 3)
    assert_equal(len(p.cutdeg(1)), 2)
    assert_equal(len(p.cutdeg(0)), 1)


def test_truncate(Poly):
    p = Poly([1, 2, 3])
    assert_raises(ValueError, p.truncate, .5)
    assert_raises(ValueError, p.truncate, 0)
    assert_equal(len(p.truncate(4)), 3)
    assert_equal(len(p.truncate(3)), 3)
    assert_equal(len(p.truncate(2)), 2)
    assert_equal(len(p.truncate(1)), 1)


def test_trim(Poly):
    c = [1, 1e-6, 1e-12, 0]
    p = Poly(c)
    assert_equal(p.trim().coef, c[:3])
    assert_equal(p.trim(1e-10).coef, c[:2])
    assert_equal(p.trim(1e-5).coef, c[:1])


def test_mapparms(Poly):
    # check with defaults. Should be identity.
    d = Poly.domain
    w = Poly.window
    p = Poly([1], domain=d, window=w)
    assert_almost_equal([0, 1], p.mapparms())
    #
    w = 2*d + 1
    p = Poly([1], domain=d, window=w)
    assert_almost_equal([1, 2], p.mapparms())


def test_ufunc_override(Poly):
    p = Poly([1, 2, 3])
    x = np.ones(3)
    assert_raises(TypeError, np.add, p, x)
    assert_raises(TypeError, np.add, x, p)


#
# Test class method that only exists for some classes
#


class TestInterpolate:

    def f(self, x):
        return x * (x - 1) * (x - 2)

    def test_raises(self):
        assert_raises(ValueError, Chebyshev.interpolate, self.f, -1)
        assert_raises(TypeError, Chebyshev.interpolate, self.f, 10.)

    def test_dimensions(self):
        for deg in range(1, 5):
            assert_(Chebyshev.interpolate(self.f, deg).degree() == deg)

    def test_approximation(self):

        def powx(x, p):
            return x**p

        x = np.linspace(0, 2, 10)
        for deg in range(0, 10):
            for t in range(0, deg + 1):
                p = Chebyshev.interpolate(powx, deg, domain=[0, 2], args=(t,))
                assert_almost_equal(p(x), powx(x, t), decimal=11)
