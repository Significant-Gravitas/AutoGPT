import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_almost_equal,
    assert_array_almost_equal, assert_raises, assert_allclose
    )

import pytest

# `poly1d` has some support for `bool_` and `timedelta64`,
# but it is limited and they are therefore excluded here
TYPE_CODES = np.typecodes["AllInteger"] + np.typecodes["AllFloat"] + "O"


class TestPolynomial:
    def test_poly1d_str_and_repr(self):
        p = np.poly1d([1., 2, 3])
        assert_equal(repr(p), 'poly1d([1., 2., 3.])')
        assert_equal(str(p),
                     '   2\n'
                     '1 x + 2 x + 3')

        q = np.poly1d([3., 2, 1])
        assert_equal(repr(q), 'poly1d([3., 2., 1.])')
        assert_equal(str(q),
                     '   2\n'
                     '3 x + 2 x + 1')

        r = np.poly1d([1.89999 + 2j, -3j, -5.12345678, 2 + 1j])
        assert_equal(str(r),
                     '            3      2\n'
                     '(1.9 + 2j) x - 3j x - 5.123 x + (2 + 1j)')

        assert_equal(str(np.poly1d([-3, -2, -1])),
                     '    2\n'
                     '-3 x - 2 x - 1')

    def test_poly1d_resolution(self):
        p = np.poly1d([1., 2, 3])
        q = np.poly1d([3., 2, 1])
        assert_equal(p(0), 3.0)
        assert_equal(p(5), 38.0)
        assert_equal(q(0), 1.0)
        assert_equal(q(5), 86.0)

    def test_poly1d_math(self):
        # here we use some simple coeffs to make calculations easier
        p = np.poly1d([1., 2, 4])
        q = np.poly1d([4., 2, 1])
        assert_equal(p/q, (np.poly1d([0.25]), np.poly1d([1.5, 3.75])))
        assert_equal(p.integ(), np.poly1d([1/3, 1., 4., 0.]))
        assert_equal(p.integ(1), np.poly1d([1/3, 1., 4., 0.]))

        p = np.poly1d([1., 2, 3])
        q = np.poly1d([3., 2, 1])
        assert_equal(p * q, np.poly1d([3., 8., 14., 8., 3.]))
        assert_equal(p + q, np.poly1d([4., 4., 4.]))
        assert_equal(p - q, np.poly1d([-2., 0., 2.]))
        assert_equal(p ** 4, np.poly1d([1., 8., 36., 104., 214., 312., 324., 216., 81.]))
        assert_equal(p(q), np.poly1d([9., 12., 16., 8., 6.]))
        assert_equal(q(p), np.poly1d([3., 12., 32., 40., 34.]))
        assert_equal(p.deriv(), np.poly1d([2., 2.]))
        assert_equal(p.deriv(2), np.poly1d([2.]))
        assert_equal(np.polydiv(np.poly1d([1, 0, -1]), np.poly1d([1, 1])),
                     (np.poly1d([1., -1.]), np.poly1d([0.])))

    @pytest.mark.parametrize("type_code", TYPE_CODES)
    def test_poly1d_misc(self, type_code: str) -> None:
        dtype = np.dtype(type_code)
        ar = np.array([1, 2, 3], dtype=dtype)
        p = np.poly1d(ar)

        # `__eq__`
        assert_equal(np.asarray(p), ar)
        assert_equal(np.asarray(p).dtype, dtype)
        assert_equal(len(p), 2)

        # `__getitem__`
        comparison_dct = {-1: 0, 0: 3, 1: 2, 2: 1, 3: 0}
        for index, ref in comparison_dct.items():
            scalar = p[index]
            assert_equal(scalar, ref)
            if dtype == np.object_:
                assert isinstance(scalar, int)
            else:
                assert_equal(scalar.dtype, dtype)

    def test_poly1d_variable_arg(self):
        q = np.poly1d([1., 2, 3], variable='y')
        assert_equal(str(q),
                     '   2\n'
                     '1 y + 2 y + 3')
        q = np.poly1d([1., 2, 3], variable='lambda')
        assert_equal(str(q),
                     '        2\n'
                     '1 lambda + 2 lambda + 3')

    def test_poly(self):
        assert_array_almost_equal(np.poly([3, -np.sqrt(2), np.sqrt(2)]),
                                  [1, -3, -2, 6])

        # From matlab docs
        A = [[1, 2, 3], [4, 5, 6], [7, 8, 0]]
        assert_array_almost_equal(np.poly(A), [1, -6, -72, -27])

        # Should produce real output for perfect conjugates
        assert_(np.isrealobj(np.poly([+1.082j, +2.613j, -2.613j, -1.082j])))
        assert_(np.isrealobj(np.poly([0+1j, -0+-1j, 1+2j,
                                      1-2j, 1.+3.5j, 1-3.5j])))
        assert_(np.isrealobj(np.poly([1j, -1j, 1+2j, 1-2j, 1+3j, 1-3.j])))
        assert_(np.isrealobj(np.poly([1j, -1j, 1+2j, 1-2j])))
        assert_(np.isrealobj(np.poly([1j, -1j, 2j, -2j])))
        assert_(np.isrealobj(np.poly([1j, -1j])))
        assert_(np.isrealobj(np.poly([1, -1])))

        assert_(np.iscomplexobj(np.poly([1j, -1.0000001j])))

        np.random.seed(42)
        a = np.random.randn(100) + 1j*np.random.randn(100)
        assert_(np.isrealobj(np.poly(np.concatenate((a, np.conjugate(a))))))

    def test_roots(self):
        assert_array_equal(np.roots([1, 0, 0]), [0, 0])

    def test_str_leading_zeros(self):
        p = np.poly1d([4, 3, 2, 1])
        p[3] = 0
        assert_equal(str(p),
                     "   2\n"
                     "3 x + 2 x + 1")

        p = np.poly1d([1, 2])
        p[0] = 0
        p[1] = 0
        assert_equal(str(p), " \n0")

    def test_polyfit(self):
        c = np.array([3., 2., 1.])
        x = np.linspace(0, 2, 7)
        y = np.polyval(c, x)
        err = [1, -1, 1, -1, 1, -1, 1]
        weights = np.arange(8, 1, -1)**2/7.0

        # Check exception when too few points for variance estimate. Note that
        # the estimate requires the number of data points to exceed
        # degree + 1
        assert_raises(ValueError, np.polyfit,
                      [1], [1], deg=0, cov=True)

        # check 1D case
        m, cov = np.polyfit(x, y+err, 2, cov=True)
        est = [3.8571, 0.2857, 1.619]
        assert_almost_equal(est, m, decimal=4)
        val0 = [[ 1.4694, -2.9388,  0.8163],
                [-2.9388,  6.3673, -2.1224],
                [ 0.8163, -2.1224,  1.161 ]]
        assert_almost_equal(val0, cov, decimal=4)

        m2, cov2 = np.polyfit(x, y+err, 2, w=weights, cov=True)
        assert_almost_equal([4.8927, -1.0177, 1.7768], m2, decimal=4)
        val = [[ 4.3964, -5.0052,  0.4878],
               [-5.0052,  6.8067, -0.9089],
               [ 0.4878, -0.9089,  0.3337]]
        assert_almost_equal(val, cov2, decimal=4)

        m3, cov3 = np.polyfit(x, y+err, 2, w=weights, cov="unscaled")
        assert_almost_equal([4.8927, -1.0177, 1.7768], m3, decimal=4)
        val = [[ 0.1473, -0.1677,  0.0163],
               [-0.1677,  0.228 , -0.0304],
               [ 0.0163, -0.0304,  0.0112]]
        assert_almost_equal(val, cov3, decimal=4)

        # check 2D (n,1) case
        y = y[:, np.newaxis]
        c = c[:, np.newaxis]
        assert_almost_equal(c, np.polyfit(x, y, 2))
        # check 2D (n,2) case
        yy = np.concatenate((y, y), axis=1)
        cc = np.concatenate((c, c), axis=1)
        assert_almost_equal(cc, np.polyfit(x, yy, 2))

        m, cov = np.polyfit(x, yy + np.array(err)[:, np.newaxis], 2, cov=True)
        assert_almost_equal(est, m[:, 0], decimal=4)
        assert_almost_equal(est, m[:, 1], decimal=4)
        assert_almost_equal(val0, cov[:, :, 0], decimal=4)
        assert_almost_equal(val0, cov[:, :, 1], decimal=4)

        # check order 1 (deg=0) case, were the analytic results are simple
        np.random.seed(123)
        y = np.random.normal(size=(4, 10000))
        mean, cov = np.polyfit(np.zeros(y.shape[0]), y, deg=0, cov=True)
        # Should get sigma_mean = sigma/sqrt(N) = 1./sqrt(4) = 0.5.
        assert_allclose(mean.std(), 0.5, atol=0.01)
        assert_allclose(np.sqrt(cov.mean()), 0.5, atol=0.01)
        # Without scaling, since reduced chi2 is 1, the result should be the same.
        mean, cov = np.polyfit(np.zeros(y.shape[0]), y, w=np.ones(y.shape[0]),
                               deg=0, cov="unscaled")
        assert_allclose(mean.std(), 0.5, atol=0.01)
        assert_almost_equal(np.sqrt(cov.mean()), 0.5)
        # If we estimate our errors wrong, no change with scaling:
        w = np.full(y.shape[0], 1./0.5)
        mean, cov = np.polyfit(np.zeros(y.shape[0]), y, w=w, deg=0, cov=True)
        assert_allclose(mean.std(), 0.5, atol=0.01)
        assert_allclose(np.sqrt(cov.mean()), 0.5, atol=0.01)
        # But if we do not scale, our estimate for the error in the mean will
        # differ.
        mean, cov = np.polyfit(np.zeros(y.shape[0]), y, w=w, deg=0, cov="unscaled")
        assert_allclose(mean.std(), 0.5, atol=0.01)
        assert_almost_equal(np.sqrt(cov.mean()), 0.25)

    def test_objects(self):
        from decimal import Decimal
        p = np.poly1d([Decimal('4.0'), Decimal('3.0'), Decimal('2.0')])
        p2 = p * Decimal('1.333333333333333')
        assert_(p2[1] == Decimal("3.9999999999999990"))
        p2 = p.deriv()
        assert_(p2[1] == Decimal('8.0'))
        p2 = p.integ()
        assert_(p2[3] == Decimal("1.333333333333333333333333333"))
        assert_(p2[2] == Decimal('1.5'))
        assert_(np.issubdtype(p2.coeffs.dtype, np.object_))
        p = np.poly([Decimal(1), Decimal(2)])
        assert_equal(np.poly([Decimal(1), Decimal(2)]),
                     [1, Decimal(-3), Decimal(2)])

    def test_complex(self):
        p = np.poly1d([3j, 2j, 1j])
        p2 = p.integ()
        assert_((p2.coeffs == [1j, 1j, 1j, 0]).all())
        p2 = p.deriv()
        assert_((p2.coeffs == [6j, 2j]).all())

    def test_integ_coeffs(self):
        p = np.poly1d([3, 2, 1])
        p2 = p.integ(3, k=[9, 7, 6])
        assert_(
            (p2.coeffs == [1/4./5., 1/3./4., 1/2./3., 9/1./2., 7, 6]).all())

    def test_zero_dims(self):
        try:
            np.poly(np.zeros((0, 0)))
        except ValueError:
            pass

    def test_poly_int_overflow(self):
        """
        Regression test for gh-5096.
        """
        v = np.arange(1, 21)
        assert_almost_equal(np.poly(v), np.poly(np.diag(v)))

    def test_zero_poly_dtype(self):
        """
        Regression test for gh-16354.
        """
        z = np.array([0, 0, 0])
        p = np.poly1d(z.astype(np.int64))
        assert_equal(p.coeffs.dtype, np.int64)

        p = np.poly1d(z.astype(np.float32))
        assert_equal(p.coeffs.dtype, np.float32)

        p = np.poly1d(z.astype(np.complex64))
        assert_equal(p.coeffs.dtype, np.complex64)

    def test_poly_eq(self):
        p = np.poly1d([1, 2, 3])
        p2 = np.poly1d([1, 2, 4])
        assert_equal(p == None, False)
        assert_equal(p != None, True)
        assert_equal(p == p, True)
        assert_equal(p == p2, False)
        assert_equal(p != p2, True)

    def test_polydiv(self):
        b = np.poly1d([2, 6, 6, 1])
        a = np.poly1d([-1j, (1+2j), -(2+1j), 1])
        q, r = np.polydiv(b, a)
        assert_equal(q.coeffs.dtype, np.complex128)
        assert_equal(r.coeffs.dtype, np.complex128)
        assert_equal(q*a + r, b)

        c = [1, 2, 3]
        d = np.poly1d([1, 2, 3])
        s, t = np.polydiv(c, d)
        assert isinstance(s, np.poly1d)
        assert isinstance(t, np.poly1d)
        u, v = np.polydiv(d, c)
        assert isinstance(u, np.poly1d)
        assert isinstance(v, np.poly1d)

    def test_poly_coeffs_mutable(self):
        """ Coefficients should be modifiable """
        p = np.poly1d([1, 2, 3])

        p.coeffs += 1
        assert_equal(p.coeffs, [2, 3, 4])

        p.coeffs[2] += 10
        assert_equal(p.coeffs, [2, 3, 14])

        # this never used to be allowed - let's not add features to deprecated
        # APIs
        assert_raises(AttributeError, setattr, p, 'coeffs', np.array(1))
