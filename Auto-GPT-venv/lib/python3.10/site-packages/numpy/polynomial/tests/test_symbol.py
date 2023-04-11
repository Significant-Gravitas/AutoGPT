"""
Tests related to the ``symbol`` attribute of the ABCPolyBase class.
"""

import pytest
import numpy.polynomial as poly
from numpy.core import array
from numpy.testing import assert_equal, assert_raises, assert_


class TestInit:
    """
    Test polynomial creation with symbol kwarg.
    """
    c = [1, 2, 3]

    def test_default_symbol(self):
        p = poly.Polynomial(self.c)
        assert_equal(p.symbol, 'x')

    @pytest.mark.parametrize(('bad_input', 'exception'), (
        ('', ValueError),
        ('3', ValueError),
        (None, TypeError),
        (1, TypeError),
    ))
    def test_symbol_bad_input(self, bad_input, exception):
        with pytest.raises(exception):
            p = poly.Polynomial(self.c, symbol=bad_input)

    @pytest.mark.parametrize('symbol', (
        'x',
        'x_1',
        'A',
        'xyz',
        'β',
    ))
    def test_valid_symbols(self, symbol):
        """
        Values for symbol that should pass input validation.
        """
        p = poly.Polynomial(self.c, symbol=symbol)
        assert_equal(p.symbol, symbol)

    def test_property(self):
        """
        'symbol' attribute is read only.
        """
        p = poly.Polynomial(self.c, symbol='x')
        with pytest.raises(AttributeError):
            p.symbol = 'z'

    def test_change_symbol(self):
        p = poly.Polynomial(self.c, symbol='y')
        # Create new polynomial from p with different symbol
        pt = poly.Polynomial(p.coef, symbol='t')
        assert_equal(pt.symbol, 't')


class TestUnaryOperators:
    p = poly.Polynomial([1, 2, 3], symbol='z')

    def test_neg(self):
        n = -self.p
        assert_equal(n.symbol, 'z')

    def test_scalarmul(self):
        out = self.p * 10
        assert_equal(out.symbol, 'z')

    def test_rscalarmul(self):
        out = 10 * self.p
        assert_equal(out.symbol, 'z')

    def test_pow(self):
        out = self.p ** 3
        assert_equal(out.symbol, 'z')


@pytest.mark.parametrize(
    'rhs',
    (
        poly.Polynomial([4, 5, 6], symbol='z'),
        array([4, 5, 6]),
    ),
)
class TestBinaryOperatorsSameSymbol:
    """
    Ensure symbol is preserved for numeric operations on polynomials with
    the same symbol
    """
    p = poly.Polynomial([1, 2, 3], symbol='z')

    def test_add(self, rhs):
        out = self.p + rhs
        assert_equal(out.symbol, 'z')

    def test_sub(self, rhs):
        out = self.p - rhs
        assert_equal(out.symbol, 'z')

    def test_polymul(self, rhs):
        out = self.p * rhs
        assert_equal(out.symbol, 'z')

    def test_divmod(self, rhs):
        for out in divmod(self.p, rhs):
            assert_equal(out.symbol, 'z')

    def test_radd(self, rhs):
        out = rhs + self.p
        assert_equal(out.symbol, 'z')

    def test_rsub(self, rhs):
        out = rhs - self.p
        assert_equal(out.symbol, 'z')

    def test_rmul(self, rhs):
        out = rhs * self.p
        assert_equal(out.symbol, 'z')

    def test_rdivmod(self, rhs):
        for out in divmod(rhs, self.p):
            assert_equal(out.symbol, 'z')


class TestBinaryOperatorsDifferentSymbol:
    p = poly.Polynomial([1, 2, 3], symbol='x')
    other = poly.Polynomial([4, 5, 6], symbol='y')
    ops = (p.__add__, p.__sub__, p.__mul__, p.__floordiv__, p.__mod__)

    @pytest.mark.parametrize('f', ops)
    def test_binops_fails(self, f):
        assert_raises(ValueError, f, self.other)


class TestEquality:
    p = poly.Polynomial([1, 2, 3], symbol='x')

    def test_eq(self):
        other = poly.Polynomial([1, 2, 3], symbol='x')
        assert_(self.p == other)

    def test_neq(self):
        other = poly.Polynomial([1, 2, 3], symbol='y')
        assert_(not self.p == other)


class TestExtraMethods:
    """
    Test other methods for manipulating/creating polynomial objects.
    """
    p = poly.Polynomial([1, 2, 3, 0], symbol='z')

    def test_copy(self):
        other = self.p.copy()
        assert_equal(other.symbol, 'z')

    def test_trim(self):
        other = self.p.trim()
        assert_equal(other.symbol, 'z')

    def test_truncate(self):
        other = self.p.truncate(2)
        assert_equal(other.symbol, 'z')

    @pytest.mark.parametrize('kwarg', (
        {'domain': [-10, 10]},
        {'window': [-10, 10]},
        {'kind': poly.Chebyshev},
    ))
    def test_convert(self, kwarg):
        other = self.p.convert(**kwarg)
        assert_equal(other.symbol, 'z')

    def test_integ(self):
        other = self.p.integ()
        assert_equal(other.symbol, 'z')

    def test_deriv(self):
        other = self.p.deriv()
        assert_equal(other.symbol, 'z')


def test_composition():
    p = poly.Polynomial([3, 2, 1], symbol="t")
    q = poly.Polynomial([5, 1, 0, -1], symbol="λ_1")
    r = p(q)
    assert r.symbol == "λ_1"


#
# Class methods that result in new polynomial class instances
#


def test_fit():
    x, y = (range(10),)*2
    p = poly.Polynomial.fit(x, y, deg=1, symbol='z')
    assert_equal(p.symbol, 'z')


def test_froomroots():
    roots = [-2, 2]
    p = poly.Polynomial.fromroots(roots, symbol='z')
    assert_equal(p.symbol, 'z')


def test_identity():
    p = poly.Polynomial.identity(domain=[-1, 1], window=[5, 20], symbol='z')
    assert_equal(p.symbol, 'z')


def test_basis():
    p = poly.Polynomial.basis(3, symbol='z')
    assert_equal(p.symbol, 'z')
