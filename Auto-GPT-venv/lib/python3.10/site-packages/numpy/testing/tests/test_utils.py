import warnings
import sys
import os
import itertools
import pytest
import weakref

import numpy as np
from numpy.testing import (
    assert_equal, assert_array_equal, assert_almost_equal,
    assert_array_almost_equal, assert_array_less, build_err_msg, raises,
    assert_raises, assert_warns, assert_no_warnings, assert_allclose,
    assert_approx_equal, assert_array_almost_equal_nulp, assert_array_max_ulp,
    clear_and_catch_warnings, suppress_warnings, assert_string_equal, assert_,
    tempdir, temppath, assert_no_gc_cycles, HAS_REFCOUNT
    )
from numpy.core.overrides import ARRAY_FUNCTION_ENABLED


class _GenericTest:

    def _test_equal(self, a, b):
        self._assert_func(a, b)

    def _test_not_equal(self, a, b):
        with assert_raises(AssertionError):
            self._assert_func(a, b)

    def test_array_rank1_eq(self):
        """Test two equal array of rank 1 are found equal."""
        a = np.array([1, 2])
        b = np.array([1, 2])

        self._test_equal(a, b)

    def test_array_rank1_noteq(self):
        """Test two different array of rank 1 are found not equal."""
        a = np.array([1, 2])
        b = np.array([2, 2])

        self._test_not_equal(a, b)

    def test_array_rank2_eq(self):
        """Test two equal array of rank 2 are found equal."""
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[1, 2], [3, 4]])

        self._test_equal(a, b)

    def test_array_diffshape(self):
        """Test two arrays with different shapes are found not equal."""
        a = np.array([1, 2])
        b = np.array([[1, 2], [1, 2]])

        self._test_not_equal(a, b)

    def test_objarray(self):
        """Test object arrays."""
        a = np.array([1, 1], dtype=object)
        self._test_equal(a, 1)

    def test_array_likes(self):
        self._test_equal([1, 2, 3], (1, 2, 3))


class TestArrayEqual(_GenericTest):

    def setup_method(self):
        self._assert_func = assert_array_equal

    def test_generic_rank1(self):
        """Test rank 1 array for all dtypes."""
        def foo(t):
            a = np.empty(2, t)
            a.fill(1)
            b = a.copy()
            c = a.copy()
            c.fill(0)
            self._test_equal(a, b)
            self._test_not_equal(c, b)

        # Test numeric types and object
        for t in '?bhilqpBHILQPfdgFDG':
            foo(t)

        # Test strings
        for t in ['S1', 'U1']:
            foo(t)

    def test_0_ndim_array(self):
        x = np.array(473963742225900817127911193656584771)
        y = np.array(18535119325151578301457182298393896)
        assert_raises(AssertionError, self._assert_func, x, y)

        y = x
        self._assert_func(x, y)

        x = np.array(43)
        y = np.array(10)
        assert_raises(AssertionError, self._assert_func, x, y)

        y = x
        self._assert_func(x, y)

    def test_generic_rank3(self):
        """Test rank 3 array for all dtypes."""
        def foo(t):
            a = np.empty((4, 2, 3), t)
            a.fill(1)
            b = a.copy()
            c = a.copy()
            c.fill(0)
            self._test_equal(a, b)
            self._test_not_equal(c, b)

        # Test numeric types and object
        for t in '?bhilqpBHILQPfdgFDG':
            foo(t)

        # Test strings
        for t in ['S1', 'U1']:
            foo(t)

    def test_nan_array(self):
        """Test arrays with nan values in them."""
        a = np.array([1, 2, np.nan])
        b = np.array([1, 2, np.nan])

        self._test_equal(a, b)

        c = np.array([1, 2, 3])
        self._test_not_equal(c, b)

    def test_string_arrays(self):
        """Test two arrays with different shapes are found not equal."""
        a = np.array(['floupi', 'floupa'])
        b = np.array(['floupi', 'floupa'])

        self._test_equal(a, b)

        c = np.array(['floupipi', 'floupa'])

        self._test_not_equal(c, b)

    def test_recarrays(self):
        """Test record arrays."""
        a = np.empty(2, [('floupi', float), ('floupa', float)])
        a['floupi'] = [1, 2]
        a['floupa'] = [1, 2]
        b = a.copy()

        self._test_equal(a, b)

        c = np.empty(2, [('floupipi', float),
                         ('floupi', float), ('floupa', float)])
        c['floupipi'] = a['floupi'].copy()
        c['floupa'] = a['floupa'].copy()

        with pytest.raises(TypeError):
            self._test_not_equal(c, b)

    def test_masked_nan_inf(self):
        # Regression test for gh-11121
        a = np.ma.MaskedArray([3., 4., 6.5], mask=[False, True, False])
        b = np.array([3., np.nan, 6.5])
        self._test_equal(a, b)
        self._test_equal(b, a)
        a = np.ma.MaskedArray([3., 4., 6.5], mask=[True, False, False])
        b = np.array([np.inf, 4., 6.5])
        self._test_equal(a, b)
        self._test_equal(b, a)

    def test_subclass_that_overrides_eq(self):
        # While we cannot guarantee testing functions will always work for
        # subclasses, the tests should ideally rely only on subclasses having
        # comparison operators, not on them being able to store booleans
        # (which, e.g., astropy Quantity cannot usefully do). See gh-8452.
        class MyArray(np.ndarray):
            def __eq__(self, other):
                return bool(np.equal(self, other).all())

            def __ne__(self, other):
                return not self == other

        a = np.array([1., 2.]).view(MyArray)
        b = np.array([2., 3.]).view(MyArray)
        assert_(type(a == a), bool)
        assert_(a == a)
        assert_(a != b)
        self._test_equal(a, a)
        self._test_not_equal(a, b)
        self._test_not_equal(b, a)

    @pytest.mark.skipif(
        not ARRAY_FUNCTION_ENABLED, reason='requires __array_function__')
    def test_subclass_that_does_not_implement_npall(self):
        class MyArray(np.ndarray):
            def __array_function__(self, *args, **kwargs):
                return NotImplemented

        a = np.array([1., 2.]).view(MyArray)
        b = np.array([2., 3.]).view(MyArray)
        with assert_raises(TypeError):
            np.all(a)
        self._test_equal(a, a)
        self._test_not_equal(a, b)
        self._test_not_equal(b, a)

    def test_suppress_overflow_warnings(self):
        # Based on issue #18992
        with pytest.raises(AssertionError):
            with np.errstate(all="raise"):
                np.testing.assert_array_equal(
                    np.array([1, 2, 3], np.float32),
                    np.array([1, 1e-40, 3], np.float32))

    def test_array_vs_scalar_is_equal(self):
        """Test comparing an array with a scalar when all values are equal."""
        a = np.array([1., 1., 1.])
        b = 1.

        self._test_equal(a, b)

    def test_array_vs_scalar_not_equal(self):
        """Test comparing an array with a scalar when not all values equal."""
        a = np.array([1., 2., 3.])
        b = 1.

        self._test_not_equal(a, b)

    def test_array_vs_scalar_strict(self):
        """Test comparing an array with a scalar with strict option."""
        a = np.array([1., 1., 1.])
        b = 1.

        with pytest.raises(AssertionError):
            assert_array_equal(a, b, strict=True)

    def test_array_vs_array_strict(self):
        """Test comparing two arrays with strict option."""
        a = np.array([1., 1., 1.])
        b = np.array([1., 1., 1.])

        assert_array_equal(a, b, strict=True)

    def test_array_vs_float_array_strict(self):
        """Test comparing two arrays with strict option."""
        a = np.array([1, 1, 1])
        b = np.array([1., 1., 1.])

        with pytest.raises(AssertionError):
            assert_array_equal(a, b, strict=True)


class TestBuildErrorMessage:

    def test_build_err_msg_defaults(self):
        x = np.array([1.00001, 2.00002, 3.00003])
        y = np.array([1.00002, 2.00003, 3.00004])
        err_msg = 'There is a mismatch'

        a = build_err_msg([x, y], err_msg)
        b = ('\nItems are not equal: There is a mismatch\n ACTUAL: array(['
             '1.00001, 2.00002, 3.00003])\n DESIRED: array([1.00002, '
             '2.00003, 3.00004])')
        assert_equal(a, b)

    def test_build_err_msg_no_verbose(self):
        x = np.array([1.00001, 2.00002, 3.00003])
        y = np.array([1.00002, 2.00003, 3.00004])
        err_msg = 'There is a mismatch'

        a = build_err_msg([x, y], err_msg, verbose=False)
        b = '\nItems are not equal: There is a mismatch'
        assert_equal(a, b)

    def test_build_err_msg_custom_names(self):
        x = np.array([1.00001, 2.00002, 3.00003])
        y = np.array([1.00002, 2.00003, 3.00004])
        err_msg = 'There is a mismatch'

        a = build_err_msg([x, y], err_msg, names=('FOO', 'BAR'))
        b = ('\nItems are not equal: There is a mismatch\n FOO: array(['
             '1.00001, 2.00002, 3.00003])\n BAR: array([1.00002, 2.00003, '
             '3.00004])')
        assert_equal(a, b)

    def test_build_err_msg_custom_precision(self):
        x = np.array([1.000000001, 2.00002, 3.00003])
        y = np.array([1.000000002, 2.00003, 3.00004])
        err_msg = 'There is a mismatch'

        a = build_err_msg([x, y], err_msg, precision=10)
        b = ('\nItems are not equal: There is a mismatch\n ACTUAL: array(['
             '1.000000001, 2.00002    , 3.00003    ])\n DESIRED: array(['
             '1.000000002, 2.00003    , 3.00004    ])')
        assert_equal(a, b)


class TestEqual(TestArrayEqual):

    def setup_method(self):
        self._assert_func = assert_equal

    def test_nan_items(self):
        self._assert_func(np.nan, np.nan)
        self._assert_func([np.nan], [np.nan])
        self._test_not_equal(np.nan, [np.nan])
        self._test_not_equal(np.nan, 1)

    def test_inf_items(self):
        self._assert_func(np.inf, np.inf)
        self._assert_func([np.inf], [np.inf])
        self._test_not_equal(np.inf, [np.inf])

    def test_datetime(self):
        self._test_equal(
            np.datetime64("2017-01-01", "s"),
            np.datetime64("2017-01-01", "s")
        )
        self._test_equal(
            np.datetime64("2017-01-01", "s"),
            np.datetime64("2017-01-01", "m")
        )

        # gh-10081
        self._test_not_equal(
            np.datetime64("2017-01-01", "s"),
            np.datetime64("2017-01-02", "s")
        )
        self._test_not_equal(
            np.datetime64("2017-01-01", "s"),
            np.datetime64("2017-01-02", "m")
        )

    def test_nat_items(self):
        # not a datetime
        nadt_no_unit = np.datetime64("NaT")
        nadt_s = np.datetime64("NaT", "s")
        nadt_d = np.datetime64("NaT", "ns")
        # not a timedelta
        natd_no_unit = np.timedelta64("NaT")
        natd_s = np.timedelta64("NaT", "s")
        natd_d = np.timedelta64("NaT", "ns")

        dts = [nadt_no_unit, nadt_s, nadt_d]
        tds = [natd_no_unit, natd_s, natd_d]
        for a, b in itertools.product(dts, dts):
            self._assert_func(a, b)
            self._assert_func([a], [b])
            self._test_not_equal([a], b)

        for a, b in itertools.product(tds, tds):
            self._assert_func(a, b)
            self._assert_func([a], [b])
            self._test_not_equal([a], b)

        for a, b in itertools.product(tds, dts):
            self._test_not_equal(a, b)
            self._test_not_equal(a, [b])
            self._test_not_equal([a], [b])
            self._test_not_equal([a], np.datetime64("2017-01-01", "s"))
            self._test_not_equal([b], np.datetime64("2017-01-01", "s"))
            self._test_not_equal([a], np.timedelta64(123, "s"))
            self._test_not_equal([b], np.timedelta64(123, "s"))

    def test_non_numeric(self):
        self._assert_func('ab', 'ab')
        self._test_not_equal('ab', 'abb')

    def test_complex_item(self):
        self._assert_func(complex(1, 2), complex(1, 2))
        self._assert_func(complex(1, np.nan), complex(1, np.nan))
        self._test_not_equal(complex(1, np.nan), complex(1, 2))
        self._test_not_equal(complex(np.nan, 1), complex(1, np.nan))
        self._test_not_equal(complex(np.nan, np.inf), complex(np.nan, 2))

    def test_negative_zero(self):
        self._test_not_equal(np.PZERO, np.NZERO)

    def test_complex(self):
        x = np.array([complex(1, 2), complex(1, np.nan)])
        y = np.array([complex(1, 2), complex(1, 2)])
        self._assert_func(x, x)
        self._test_not_equal(x, y)

    def test_object(self):
        #gh-12942
        import datetime
        a = np.array([datetime.datetime(2000, 1, 1),
                      datetime.datetime(2000, 1, 2)])
        self._test_not_equal(a, a[::-1])


class TestArrayAlmostEqual(_GenericTest):

    def setup_method(self):
        self._assert_func = assert_array_almost_equal

    def test_closeness(self):
        # Note that in the course of time we ended up with
        #     `abs(x - y) < 1.5 * 10**(-decimal)`
        # instead of the previously documented
        #     `abs(x - y) < 0.5 * 10**(-decimal)`
        # so this check serves to preserve the wrongness.

        # test scalars
        self._assert_func(1.499999, 0.0, decimal=0)
        assert_raises(AssertionError,
                          lambda: self._assert_func(1.5, 0.0, decimal=0))

        # test arrays
        self._assert_func([1.499999], [0.0], decimal=0)
        assert_raises(AssertionError,
                          lambda: self._assert_func([1.5], [0.0], decimal=0))

    def test_simple(self):
        x = np.array([1234.2222])
        y = np.array([1234.2223])

        self._assert_func(x, y, decimal=3)
        self._assert_func(x, y, decimal=4)
        assert_raises(AssertionError,
                lambda: self._assert_func(x, y, decimal=5))

    def test_nan(self):
        anan = np.array([np.nan])
        aone = np.array([1])
        ainf = np.array([np.inf])
        self._assert_func(anan, anan)
        assert_raises(AssertionError,
                lambda: self._assert_func(anan, aone))
        assert_raises(AssertionError,
                lambda: self._assert_func(anan, ainf))
        assert_raises(AssertionError,
                lambda: self._assert_func(ainf, anan))

    def test_inf(self):
        a = np.array([[1., 2.], [3., 4.]])
        b = a.copy()
        a[0, 0] = np.inf
        assert_raises(AssertionError,
                lambda: self._assert_func(a, b))
        b[0, 0] = -np.inf
        assert_raises(AssertionError,
                lambda: self._assert_func(a, b))

    def test_subclass(self):
        a = np.array([[1., 2.], [3., 4.]])
        b = np.ma.masked_array([[1., 2.], [0., 4.]],
                               [[False, False], [True, False]])
        self._assert_func(a, b)
        self._assert_func(b, a)
        self._assert_func(b, b)

        # Test fully masked as well (see gh-11123).
        a = np.ma.MaskedArray(3.5, mask=True)
        b = np.array([3., 4., 6.5])
        self._test_equal(a, b)
        self._test_equal(b, a)
        a = np.ma.masked
        b = np.array([3., 4., 6.5])
        self._test_equal(a, b)
        self._test_equal(b, a)
        a = np.ma.MaskedArray([3., 4., 6.5], mask=[True, True, True])
        b = np.array([1., 2., 3.])
        self._test_equal(a, b)
        self._test_equal(b, a)
        a = np.ma.MaskedArray([3., 4., 6.5], mask=[True, True, True])
        b = np.array(1.)
        self._test_equal(a, b)
        self._test_equal(b, a)

    def test_subclass_that_cannot_be_bool(self):
        # While we cannot guarantee testing functions will always work for
        # subclasses, the tests should ideally rely only on subclasses having
        # comparison operators, not on them being able to store booleans
        # (which, e.g., astropy Quantity cannot usefully do). See gh-8452.
        class MyArray(np.ndarray):
            def __eq__(self, other):
                return super().__eq__(other).view(np.ndarray)

            def __lt__(self, other):
                return super().__lt__(other).view(np.ndarray)

            def all(self, *args, **kwargs):
                raise NotImplementedError

        a = np.array([1., 2.]).view(MyArray)
        self._assert_func(a, a)


class TestAlmostEqual(_GenericTest):

    def setup_method(self):
        self._assert_func = assert_almost_equal

    def test_closeness(self):
        # Note that in the course of time we ended up with
        #     `abs(x - y) < 1.5 * 10**(-decimal)`
        # instead of the previously documented
        #     `abs(x - y) < 0.5 * 10**(-decimal)`
        # so this check serves to preserve the wrongness.

        # test scalars
        self._assert_func(1.499999, 0.0, decimal=0)
        assert_raises(AssertionError,
                      lambda: self._assert_func(1.5, 0.0, decimal=0))

        # test arrays
        self._assert_func([1.499999], [0.0], decimal=0)
        assert_raises(AssertionError,
                      lambda: self._assert_func([1.5], [0.0], decimal=0))

    def test_nan_item(self):
        self._assert_func(np.nan, np.nan)
        assert_raises(AssertionError,
                      lambda: self._assert_func(np.nan, 1))
        assert_raises(AssertionError,
                      lambda: self._assert_func(np.nan, np.inf))
        assert_raises(AssertionError,
                      lambda: self._assert_func(np.inf, np.nan))

    def test_inf_item(self):
        self._assert_func(np.inf, np.inf)
        self._assert_func(-np.inf, -np.inf)
        assert_raises(AssertionError,
                      lambda: self._assert_func(np.inf, 1))
        assert_raises(AssertionError,
                      lambda: self._assert_func(-np.inf, np.inf))

    def test_simple_item(self):
        self._test_not_equal(1, 2)

    def test_complex_item(self):
        self._assert_func(complex(1, 2), complex(1, 2))
        self._assert_func(complex(1, np.nan), complex(1, np.nan))
        self._assert_func(complex(np.inf, np.nan), complex(np.inf, np.nan))
        self._test_not_equal(complex(1, np.nan), complex(1, 2))
        self._test_not_equal(complex(np.nan, 1), complex(1, np.nan))
        self._test_not_equal(complex(np.nan, np.inf), complex(np.nan, 2))

    def test_complex(self):
        x = np.array([complex(1, 2), complex(1, np.nan)])
        z = np.array([complex(1, 2), complex(np.nan, 1)])
        y = np.array([complex(1, 2), complex(1, 2)])
        self._assert_func(x, x)
        self._test_not_equal(x, y)
        self._test_not_equal(x, z)

    def test_error_message(self):
        """Check the message is formatted correctly for the decimal value.
           Also check the message when input includes inf or nan (gh12200)"""
        x = np.array([1.00000000001, 2.00000000002, 3.00003])
        y = np.array([1.00000000002, 2.00000000003, 3.00004])

        # Test with a different amount of decimal digits
        with pytest.raises(AssertionError) as exc_info:
            self._assert_func(x, y, decimal=12)
        msgs = str(exc_info.value).split('\n')
        assert_equal(msgs[3], 'Mismatched elements: 3 / 3 (100%)')
        assert_equal(msgs[4], 'Max absolute difference: 1.e-05')
        assert_equal(msgs[5], 'Max relative difference: 3.33328889e-06')
        assert_equal(
            msgs[6],
            ' x: array([1.00000000001, 2.00000000002, 3.00003      ])')
        assert_equal(
            msgs[7],
            ' y: array([1.00000000002, 2.00000000003, 3.00004      ])')

        # With the default value of decimal digits, only the 3rd element
        # differs. Note that we only check for the formatting of the arrays
        # themselves.
        with pytest.raises(AssertionError) as exc_info:
            self._assert_func(x, y)
        msgs = str(exc_info.value).split('\n')
        assert_equal(msgs[3], 'Mismatched elements: 1 / 3 (33.3%)')
        assert_equal(msgs[4], 'Max absolute difference: 1.e-05')
        assert_equal(msgs[5], 'Max relative difference: 3.33328889e-06')
        assert_equal(msgs[6], ' x: array([1.     , 2.     , 3.00003])')
        assert_equal(msgs[7], ' y: array([1.     , 2.     , 3.00004])')

        # Check the error message when input includes inf
        x = np.array([np.inf, 0])
        y = np.array([np.inf, 1])
        with pytest.raises(AssertionError) as exc_info:
            self._assert_func(x, y)
        msgs = str(exc_info.value).split('\n')
        assert_equal(msgs[3], 'Mismatched elements: 1 / 2 (50%)')
        assert_equal(msgs[4], 'Max absolute difference: 1.')
        assert_equal(msgs[5], 'Max relative difference: 1.')
        assert_equal(msgs[6], ' x: array([inf,  0.])')
        assert_equal(msgs[7], ' y: array([inf,  1.])')

        # Check the error message when dividing by zero
        x = np.array([1, 2])
        y = np.array([0, 0])
        with pytest.raises(AssertionError) as exc_info:
            self._assert_func(x, y)
        msgs = str(exc_info.value).split('\n')
        assert_equal(msgs[3], 'Mismatched elements: 2 / 2 (100%)')
        assert_equal(msgs[4], 'Max absolute difference: 2')
        assert_equal(msgs[5], 'Max relative difference: inf')

    def test_error_message_2(self):
        """Check the message is formatted correctly when either x or y is a scalar."""
        x = 2
        y = np.ones(20)
        with pytest.raises(AssertionError) as exc_info:
            self._assert_func(x, y)
        msgs = str(exc_info.value).split('\n')
        assert_equal(msgs[3], 'Mismatched elements: 20 / 20 (100%)')
        assert_equal(msgs[4], 'Max absolute difference: 1.')
        assert_equal(msgs[5], 'Max relative difference: 1.')

        y = 2
        x = np.ones(20)
        with pytest.raises(AssertionError) as exc_info:
            self._assert_func(x, y)
        msgs = str(exc_info.value).split('\n')
        assert_equal(msgs[3], 'Mismatched elements: 20 / 20 (100%)')
        assert_equal(msgs[4], 'Max absolute difference: 1.')
        assert_equal(msgs[5], 'Max relative difference: 0.5')

    def test_subclass_that_cannot_be_bool(self):
        # While we cannot guarantee testing functions will always work for
        # subclasses, the tests should ideally rely only on subclasses having
        # comparison operators, not on them being able to store booleans
        # (which, e.g., astropy Quantity cannot usefully do). See gh-8452.
        class MyArray(np.ndarray):
            def __eq__(self, other):
                return super().__eq__(other).view(np.ndarray)

            def __lt__(self, other):
                return super().__lt__(other).view(np.ndarray)

            def all(self, *args, **kwargs):
                raise NotImplementedError

        a = np.array([1., 2.]).view(MyArray)
        self._assert_func(a, a)


class TestApproxEqual:

    def setup_method(self):
        self._assert_func = assert_approx_equal

    def test_simple_0d_arrays(self):
        x = np.array(1234.22)
        y = np.array(1234.23)

        self._assert_func(x, y, significant=5)
        self._assert_func(x, y, significant=6)
        assert_raises(AssertionError,
                      lambda: self._assert_func(x, y, significant=7))

    def test_simple_items(self):
        x = 1234.22
        y = 1234.23

        self._assert_func(x, y, significant=4)
        self._assert_func(x, y, significant=5)
        self._assert_func(x, y, significant=6)
        assert_raises(AssertionError,
                      lambda: self._assert_func(x, y, significant=7))

    def test_nan_array(self):
        anan = np.array(np.nan)
        aone = np.array(1)
        ainf = np.array(np.inf)
        self._assert_func(anan, anan)
        assert_raises(AssertionError, lambda: self._assert_func(anan, aone))
        assert_raises(AssertionError, lambda: self._assert_func(anan, ainf))
        assert_raises(AssertionError, lambda: self._assert_func(ainf, anan))

    def test_nan_items(self):
        anan = np.array(np.nan)
        aone = np.array(1)
        ainf = np.array(np.inf)
        self._assert_func(anan, anan)
        assert_raises(AssertionError, lambda: self._assert_func(anan, aone))
        assert_raises(AssertionError, lambda: self._assert_func(anan, ainf))
        assert_raises(AssertionError, lambda: self._assert_func(ainf, anan))


class TestArrayAssertLess:

    def setup_method(self):
        self._assert_func = assert_array_less

    def test_simple_arrays(self):
        x = np.array([1.1, 2.2])
        y = np.array([1.2, 2.3])

        self._assert_func(x, y)
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

        y = np.array([1.0, 2.3])

        assert_raises(AssertionError, lambda: self._assert_func(x, y))
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

    def test_rank2(self):
        x = np.array([[1.1, 2.2], [3.3, 4.4]])
        y = np.array([[1.2, 2.3], [3.4, 4.5]])

        self._assert_func(x, y)
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

        y = np.array([[1.0, 2.3], [3.4, 4.5]])

        assert_raises(AssertionError, lambda: self._assert_func(x, y))
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

    def test_rank3(self):
        x = np.ones(shape=(2, 2, 2))
        y = np.ones(shape=(2, 2, 2))+1

        self._assert_func(x, y)
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

        y[0, 0, 0] = 0

        assert_raises(AssertionError, lambda: self._assert_func(x, y))
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

    def test_simple_items(self):
        x = 1.1
        y = 2.2

        self._assert_func(x, y)
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

        y = np.array([2.2, 3.3])

        self._assert_func(x, y)
        assert_raises(AssertionError, lambda: self._assert_func(y, x))

        y = np.array([1.0, 3.3])

        assert_raises(AssertionError, lambda: self._assert_func(x, y))

    def test_nan_noncompare(self):
        anan = np.array(np.nan)
        aone = np.array(1)
        ainf = np.array(np.inf)
        self._assert_func(anan, anan)
        assert_raises(AssertionError, lambda: self._assert_func(aone, anan))
        assert_raises(AssertionError, lambda: self._assert_func(anan, aone))
        assert_raises(AssertionError, lambda: self._assert_func(anan, ainf))
        assert_raises(AssertionError, lambda: self._assert_func(ainf, anan))

    def test_nan_noncompare_array(self):
        x = np.array([1.1, 2.2, 3.3])
        anan = np.array(np.nan)

        assert_raises(AssertionError, lambda: self._assert_func(x, anan))
        assert_raises(AssertionError, lambda: self._assert_func(anan, x))

        x = np.array([1.1, 2.2, np.nan])

        assert_raises(AssertionError, lambda: self._assert_func(x, anan))
        assert_raises(AssertionError, lambda: self._assert_func(anan, x))

        y = np.array([1.0, 2.0, np.nan])

        self._assert_func(y, x)
        assert_raises(AssertionError, lambda: self._assert_func(x, y))

    def test_inf_compare(self):
        aone = np.array(1)
        ainf = np.array(np.inf)

        self._assert_func(aone, ainf)
        self._assert_func(-ainf, aone)
        self._assert_func(-ainf, ainf)
        assert_raises(AssertionError, lambda: self._assert_func(ainf, aone))
        assert_raises(AssertionError, lambda: self._assert_func(aone, -ainf))
        assert_raises(AssertionError, lambda: self._assert_func(ainf, ainf))
        assert_raises(AssertionError, lambda: self._assert_func(ainf, -ainf))
        assert_raises(AssertionError, lambda: self._assert_func(-ainf, -ainf))

    def test_inf_compare_array(self):
        x = np.array([1.1, 2.2, np.inf])
        ainf = np.array(np.inf)

        assert_raises(AssertionError, lambda: self._assert_func(x, ainf))
        assert_raises(AssertionError, lambda: self._assert_func(ainf, x))
        assert_raises(AssertionError, lambda: self._assert_func(x, -ainf))
        assert_raises(AssertionError, lambda: self._assert_func(-x, -ainf))
        assert_raises(AssertionError, lambda: self._assert_func(-ainf, -x))
        self._assert_func(-ainf, x)


@pytest.mark.skip(reason="The raises decorator depends on Nose")
class TestRaises:

    def setup_method(self):
        class MyException(Exception):
            pass

        self.e = MyException

    def raises_exception(self, e):
        raise e

    def does_not_raise_exception(self):
        pass

    def test_correct_catch(self):
        raises(self.e)(self.raises_exception)(self.e)  # raises?

    def test_wrong_exception(self):
        try:
            raises(self.e)(self.raises_exception)(RuntimeError)  # raises?
        except RuntimeError:
            return
        else:
            raise AssertionError("should have caught RuntimeError")

    def test_catch_no_raise(self):
        try:
            raises(self.e)(self.does_not_raise_exception)()  # raises?
        except AssertionError:
            return
        else:
            raise AssertionError("should have raised an AssertionError")


class TestWarns:

    def test_warn(self):
        def f():
            warnings.warn("yo")
            return 3

        before_filters = sys.modules['warnings'].filters[:]
        assert_equal(assert_warns(UserWarning, f), 3)
        after_filters = sys.modules['warnings'].filters

        assert_raises(AssertionError, assert_no_warnings, f)
        assert_equal(assert_no_warnings(lambda x: x, 1), 1)

        # Check that the warnings state is unchanged
        assert_equal(before_filters, after_filters,
                     "assert_warns does not preserver warnings state")

    def test_context_manager(self):

        before_filters = sys.modules['warnings'].filters[:]
        with assert_warns(UserWarning):
            warnings.warn("yo")
        after_filters = sys.modules['warnings'].filters

        def no_warnings():
            with assert_no_warnings():
                warnings.warn("yo")

        assert_raises(AssertionError, no_warnings)
        assert_equal(before_filters, after_filters,
                     "assert_warns does not preserver warnings state")

    def test_warn_wrong_warning(self):
        def f():
            warnings.warn("yo", DeprecationWarning)

        failed = False
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            try:
                # Should raise a DeprecationWarning
                assert_warns(UserWarning, f)
                failed = True
            except DeprecationWarning:
                pass

        if failed:
            raise AssertionError("wrong warning caught by assert_warn")


class TestAssertAllclose:

    def test_simple(self):
        x = 1e-3
        y = 1e-9

        assert_allclose(x, y, atol=1)
        assert_raises(AssertionError, assert_allclose, x, y)

        a = np.array([x, y, x, y])
        b = np.array([x, y, x, x])

        assert_allclose(a, b, atol=1)
        assert_raises(AssertionError, assert_allclose, a, b)

        b[-1] = y * (1 + 1e-8)
        assert_allclose(a, b)
        assert_raises(AssertionError, assert_allclose, a, b, rtol=1e-9)

        assert_allclose(6, 10, rtol=0.5)
        assert_raises(AssertionError, assert_allclose, 10, 6, rtol=0.5)

    def test_min_int(self):
        a = np.array([np.iinfo(np.int_).min], dtype=np.int_)
        # Should not raise:
        assert_allclose(a, a)

    def test_report_fail_percentage(self):
        a = np.array([1, 1, 1, 1])
        b = np.array([1, 1, 1, 2])

        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(a, b)
        msg = str(exc_info.value)
        assert_('Mismatched elements: 1 / 4 (25%)\n'
                'Max absolute difference: 1\n'
                'Max relative difference: 0.5' in msg)

    def test_equal_nan(self):
        a = np.array([np.nan])
        b = np.array([np.nan])
        # Should not raise:
        assert_allclose(a, b, equal_nan=True)

    def test_not_equal_nan(self):
        a = np.array([np.nan])
        b = np.array([np.nan])
        assert_raises(AssertionError, assert_allclose, a, b, equal_nan=False)

    def test_equal_nan_default(self):
        # Make sure equal_nan default behavior remains unchanged. (All
        # of these functions use assert_array_compare under the hood.)
        # None of these should raise.
        a = np.array([np.nan])
        b = np.array([np.nan])
        assert_array_equal(a, b)
        assert_array_almost_equal(a, b)
        assert_array_less(a, b)
        assert_allclose(a, b)

    def test_report_max_relative_error(self):
        a = np.array([0, 1])
        b = np.array([0, 2])

        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(a, b)
        msg = str(exc_info.value)
        assert_('Max relative difference: 0.5' in msg)

    def test_timedelta(self):
        # see gh-18286
        a = np.array([[1, 2, 3, "NaT"]], dtype="m8[ns]")
        assert_allclose(a, a)

    def test_error_message_unsigned(self):
        """Check the the message is formatted correctly when overflow can occur
           (gh21768)"""
        # Ensure to test for potential overflow in the case of:
        #        x - y
        # and
        #        y - x
        x = np.asarray([0, 1, 8], dtype='uint8')
        y = np.asarray([4, 4, 4], dtype='uint8')
        with pytest.raises(AssertionError) as exc_info:
            assert_allclose(x, y, atol=3)
        msgs = str(exc_info.value).split('\n')
        assert_equal(msgs[4], 'Max absolute difference: 4')


class TestArrayAlmostEqualNulp:

    def test_float64_pass(self):
        # The number of units of least precision
        # In this case, use a few places above the lowest level (ie nulp=1)
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float64)
        x = 10**x
        x = np.r_[-x, x]

        # Addition
        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp/2.
        assert_array_almost_equal_nulp(x, y, nulp)

        # Subtraction
        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp/2.
        assert_array_almost_equal_nulp(x, y, nulp)

    def test_float64_fail(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float64)
        x = 10**x
        x = np.r_[-x, x]

        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp*2.
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      x, y, nulp)

        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp*2.
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      x, y, nulp)

    def test_float64_ignore_nan(self):
        # Ignore ULP differences between various NAN's
        # Note that MIPS may reverse quiet and signaling nans
        # so we use the builtin version as a base.
        offset = np.uint64(0xffffffff)
        nan1_i64 = np.array(np.nan, dtype=np.float64).view(np.uint64)
        nan2_i64 = nan1_i64 ^ offset  # nan payload on MIPS is all ones.
        nan1_f64 = nan1_i64.view(np.float64)
        nan2_f64 = nan2_i64.view(np.float64)
        assert_array_max_ulp(nan1_f64, nan2_f64, 0)

    def test_float32_pass(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float32)
        x = 10**x
        x = np.r_[-x, x]

        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp/2.
        assert_array_almost_equal_nulp(x, y, nulp)

        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp/2.
        assert_array_almost_equal_nulp(x, y, nulp)

    def test_float32_fail(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float32)
        x = 10**x
        x = np.r_[-x, x]

        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp*2.
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      x, y, nulp)

        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp*2.
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      x, y, nulp)

    def test_float32_ignore_nan(self):
        # Ignore ULP differences between various NAN's
        # Note that MIPS may reverse quiet and signaling nans
        # so we use the builtin version as a base.
        offset = np.uint32(0xffff)
        nan1_i32 = np.array(np.nan, dtype=np.float32).view(np.uint32)
        nan2_i32 = nan1_i32 ^ offset  # nan payload on MIPS is all ones.
        nan1_f32 = nan1_i32.view(np.float32)
        nan2_f32 = nan2_i32.view(np.float32)
        assert_array_max_ulp(nan1_f32, nan2_f32, 0)

    def test_float16_pass(self):
        nulp = 5
        x = np.linspace(-4, 4, 10, dtype=np.float16)
        x = 10**x
        x = np.r_[-x, x]

        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp/2.
        assert_array_almost_equal_nulp(x, y, nulp)

        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp/2.
        assert_array_almost_equal_nulp(x, y, nulp)

    def test_float16_fail(self):
        nulp = 5
        x = np.linspace(-4, 4, 10, dtype=np.float16)
        x = 10**x
        x = np.r_[-x, x]

        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp*2.
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      x, y, nulp)

        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp*2.
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      x, y, nulp)

    def test_float16_ignore_nan(self):
        # Ignore ULP differences between various NAN's
        # Note that MIPS may reverse quiet and signaling nans
        # so we use the builtin version as a base.
        offset = np.uint16(0xff)
        nan1_i16 = np.array(np.nan, dtype=np.float16).view(np.uint16)
        nan2_i16 = nan1_i16 ^ offset  # nan payload on MIPS is all ones.
        nan1_f16 = nan1_i16.view(np.float16)
        nan2_f16 = nan2_i16.view(np.float16)
        assert_array_max_ulp(nan1_f16, nan2_f16, 0)

    def test_complex128_pass(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float64)
        x = 10**x
        x = np.r_[-x, x]
        xi = x + x*1j

        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp/2.
        assert_array_almost_equal_nulp(xi, x + y*1j, nulp)
        assert_array_almost_equal_nulp(xi, y + x*1j, nulp)
        # The test condition needs to be at least a factor of sqrt(2) smaller
        # because the real and imaginary parts both change
        y = x + x*eps*nulp/4.
        assert_array_almost_equal_nulp(xi, y + y*1j, nulp)

        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp/2.
        assert_array_almost_equal_nulp(xi, x + y*1j, nulp)
        assert_array_almost_equal_nulp(xi, y + x*1j, nulp)
        y = x - x*epsneg*nulp/4.
        assert_array_almost_equal_nulp(xi, y + y*1j, nulp)

    def test_complex128_fail(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float64)
        x = 10**x
        x = np.r_[-x, x]
        xi = x + x*1j

        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp*2.
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, x + y*1j, nulp)
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, y + x*1j, nulp)
        # The test condition needs to be at least a factor of sqrt(2) smaller
        # because the real and imaginary parts both change
        y = x + x*eps*nulp
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, y + y*1j, nulp)

        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp*2.
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, x + y*1j, nulp)
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, y + x*1j, nulp)
        y = x - x*epsneg*nulp
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, y + y*1j, nulp)

    def test_complex64_pass(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float32)
        x = 10**x
        x = np.r_[-x, x]
        xi = x + x*1j

        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp/2.
        assert_array_almost_equal_nulp(xi, x + y*1j, nulp)
        assert_array_almost_equal_nulp(xi, y + x*1j, nulp)
        y = x + x*eps*nulp/4.
        assert_array_almost_equal_nulp(xi, y + y*1j, nulp)

        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp/2.
        assert_array_almost_equal_nulp(xi, x + y*1j, nulp)
        assert_array_almost_equal_nulp(xi, y + x*1j, nulp)
        y = x - x*epsneg*nulp/4.
        assert_array_almost_equal_nulp(xi, y + y*1j, nulp)

    def test_complex64_fail(self):
        nulp = 5
        x = np.linspace(-20, 20, 50, dtype=np.float32)
        x = 10**x
        x = np.r_[-x, x]
        xi = x + x*1j

        eps = np.finfo(x.dtype).eps
        y = x + x*eps*nulp*2.
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, x + y*1j, nulp)
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, y + x*1j, nulp)
        y = x + x*eps*nulp
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, y + y*1j, nulp)

        epsneg = np.finfo(x.dtype).epsneg
        y = x - x*epsneg*nulp*2.
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, x + y*1j, nulp)
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, y + x*1j, nulp)
        y = x - x*epsneg*nulp
        assert_raises(AssertionError, assert_array_almost_equal_nulp,
                      xi, y + y*1j, nulp)


class TestULP:

    def test_equal(self):
        x = np.random.randn(10)
        assert_array_max_ulp(x, x, maxulp=0)

    def test_single(self):
        # Generate 1 + small deviation, check that adding eps gives a few UNL
        x = np.ones(10).astype(np.float32)
        x += 0.01 * np.random.randn(10).astype(np.float32)
        eps = np.finfo(np.float32).eps
        assert_array_max_ulp(x, x+eps, maxulp=20)

    def test_double(self):
        # Generate 1 + small deviation, check that adding eps gives a few UNL
        x = np.ones(10).astype(np.float64)
        x += 0.01 * np.random.randn(10).astype(np.float64)
        eps = np.finfo(np.float64).eps
        assert_array_max_ulp(x, x+eps, maxulp=200)

    def test_inf(self):
        for dt in [np.float32, np.float64]:
            inf = np.array([np.inf]).astype(dt)
            big = np.array([np.finfo(dt).max])
            assert_array_max_ulp(inf, big, maxulp=200)

    def test_nan(self):
        # Test that nan is 'far' from small, tiny, inf, max and min
        for dt in [np.float32, np.float64]:
            if dt == np.float32:
                maxulp = 1e6
            else:
                maxulp = 1e12
            inf = np.array([np.inf]).astype(dt)
            nan = np.array([np.nan]).astype(dt)
            big = np.array([np.finfo(dt).max])
            tiny = np.array([np.finfo(dt).tiny])
            zero = np.array([np.PZERO]).astype(dt)
            nzero = np.array([np.NZERO]).astype(dt)
            assert_raises(AssertionError,
                          lambda: assert_array_max_ulp(nan, inf,
                          maxulp=maxulp))
            assert_raises(AssertionError,
                          lambda: assert_array_max_ulp(nan, big,
                          maxulp=maxulp))
            assert_raises(AssertionError,
                          lambda: assert_array_max_ulp(nan, tiny,
                          maxulp=maxulp))
            assert_raises(AssertionError,
                          lambda: assert_array_max_ulp(nan, zero,
                          maxulp=maxulp))
            assert_raises(AssertionError,
                          lambda: assert_array_max_ulp(nan, nzero,
                          maxulp=maxulp))


class TestStringEqual:
    def test_simple(self):
        assert_string_equal("hello", "hello")
        assert_string_equal("hello\nmultiline", "hello\nmultiline")

        with pytest.raises(AssertionError) as exc_info:
            assert_string_equal("foo\nbar", "hello\nbar")
        msg = str(exc_info.value)
        assert_equal(msg, "Differences in strings:\n- foo\n+ hello")

        assert_raises(AssertionError,
                      lambda: assert_string_equal("foo", "hello"))

    def test_regex(self):
        assert_string_equal("a+*b", "a+*b")

        assert_raises(AssertionError,
                      lambda: assert_string_equal("aaa", "a+b"))


def assert_warn_len_equal(mod, n_in_context):
    try:
        mod_warns = mod.__warningregistry__
    except AttributeError:
        # the lack of a __warningregistry__
        # attribute means that no warning has
        # occurred; this can be triggered in
        # a parallel test scenario, while in
        # a serial test scenario an initial
        # warning (and therefore the attribute)
        # are always created first
        mod_warns = {}

    num_warns = len(mod_warns)

    if 'version' in mod_warns:
        # Python 3 adds a 'version' entry to the registry,
        # do not count it.
        num_warns -= 1

    assert_equal(num_warns, n_in_context)


def test_warn_len_equal_call_scenarios():
    # assert_warn_len_equal is called under
    # varying circumstances depending on serial
    # vs. parallel test scenarios; this test
    # simply aims to probe both code paths and
    # check that no assertion is uncaught

    # parallel scenario -- no warning issued yet
    class mod:
        pass

    mod_inst = mod()

    assert_warn_len_equal(mod=mod_inst,
                          n_in_context=0)

    # serial test scenario -- the __warningregistry__
    # attribute should be present
    class mod:
        def __init__(self):
            self.__warningregistry__ = {'warning1':1,
                                        'warning2':2}

    mod_inst = mod()
    assert_warn_len_equal(mod=mod_inst,
                          n_in_context=2)


def _get_fresh_mod():
    # Get this module, with warning registry empty
    my_mod = sys.modules[__name__]
    try:
        my_mod.__warningregistry__.clear()
    except AttributeError:
        # will not have a __warningregistry__ unless warning has been
        # raised in the module at some point
        pass
    return my_mod


def test_clear_and_catch_warnings():
    # Initial state of module, no warnings
    my_mod = _get_fresh_mod()
    assert_equal(getattr(my_mod, '__warningregistry__', {}), {})
    with clear_and_catch_warnings(modules=[my_mod]):
        warnings.simplefilter('ignore')
        warnings.warn('Some warning')
    assert_equal(my_mod.__warningregistry__, {})
    # Without specified modules, don't clear warnings during context.
    # catch_warnings doesn't make an entry for 'ignore'.
    with clear_and_catch_warnings():
        warnings.simplefilter('ignore')
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)

    # Manually adding two warnings to the registry:
    my_mod.__warningregistry__ = {'warning1': 1,
                                  'warning2': 2}

    # Confirm that specifying module keeps old warning, does not add new
    with clear_and_catch_warnings(modules=[my_mod]):
        warnings.simplefilter('ignore')
        warnings.warn('Another warning')
    assert_warn_len_equal(my_mod, 2)

    # Another warning, no module spec it clears up registry
    with clear_and_catch_warnings():
        warnings.simplefilter('ignore')
        warnings.warn('Another warning')
    assert_warn_len_equal(my_mod, 0)


def test_suppress_warnings_module():
    # Initial state of module, no warnings
    my_mod = _get_fresh_mod()
    assert_equal(getattr(my_mod, '__warningregistry__', {}), {})

    def warn_other_module():
        # Apply along axis is implemented in python; stacklevel=2 means
        # we end up inside its module, not ours.
        def warn(arr):
            warnings.warn("Some warning 2", stacklevel=2)
            return arr
        np.apply_along_axis(warn, 0, [0])

    # Test module based warning suppression:
    assert_warn_len_equal(my_mod, 0)
    with suppress_warnings() as sup:
        sup.record(UserWarning)
        # suppress warning from other module (may have .pyc ending),
        # if apply_along_axis is moved, had to be changed.
        sup.filter(module=np.lib.shape_base)
        warnings.warn("Some warning")
        warn_other_module()
    # Check that the suppression did test the file correctly (this module
    # got filtered)
    assert_equal(len(sup.log), 1)
    assert_equal(sup.log[0].message.args[0], "Some warning")
    assert_warn_len_equal(my_mod, 0)
    sup = suppress_warnings()
    # Will have to be changed if apply_along_axis is moved:
    sup.filter(module=my_mod)
    with sup:
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)
    # And test repeat works:
    sup.filter(module=my_mod)
    with sup:
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)

    # Without specified modules
    with suppress_warnings():
        warnings.simplefilter('ignore')
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)


def test_suppress_warnings_type():
    # Initial state of module, no warnings
    my_mod = _get_fresh_mod()
    assert_equal(getattr(my_mod, '__warningregistry__', {}), {})

    # Test module based warning suppression:
    with suppress_warnings() as sup:
        sup.filter(UserWarning)
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)
    sup = suppress_warnings()
    sup.filter(UserWarning)
    with sup:
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)
    # And test repeat works:
    sup.filter(module=my_mod)
    with sup:
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)

    # Without specified modules
    with suppress_warnings():
        warnings.simplefilter('ignore')
        warnings.warn('Some warning')
    assert_warn_len_equal(my_mod, 0)


def test_suppress_warnings_decorate_no_record():
    sup = suppress_warnings()
    sup.filter(UserWarning)

    @sup
    def warn(category):
        warnings.warn('Some warning', category)

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        warn(UserWarning)  # should be supppressed
        warn(RuntimeWarning)
        assert_equal(len(w), 1)


def test_suppress_warnings_record():
    sup = suppress_warnings()
    log1 = sup.record()

    with sup:
        log2 = sup.record(message='Some other warning 2')
        sup.filter(message='Some warning')
        warnings.warn('Some warning')
        warnings.warn('Some other warning')
        warnings.warn('Some other warning 2')

        assert_equal(len(sup.log), 2)
        assert_equal(len(log1), 1)
        assert_equal(len(log2),1)
        assert_equal(log2[0].message.args[0], 'Some other warning 2')

    # Do it again, with the same context to see if some warnings survived:
    with sup:
        log2 = sup.record(message='Some other warning 2')
        sup.filter(message='Some warning')
        warnings.warn('Some warning')
        warnings.warn('Some other warning')
        warnings.warn('Some other warning 2')

        assert_equal(len(sup.log), 2)
        assert_equal(len(log1), 1)
        assert_equal(len(log2), 1)
        assert_equal(log2[0].message.args[0], 'Some other warning 2')

    # Test nested:
    with suppress_warnings() as sup:
        sup.record()
        with suppress_warnings() as sup2:
            sup2.record(message='Some warning')
            warnings.warn('Some warning')
            warnings.warn('Some other warning')
            assert_equal(len(sup2.log), 1)
        assert_equal(len(sup.log), 1)


def test_suppress_warnings_forwarding():
    def warn_other_module():
        # Apply along axis is implemented in python; stacklevel=2 means
        # we end up inside its module, not ours.
        def warn(arr):
            warnings.warn("Some warning", stacklevel=2)
            return arr
        np.apply_along_axis(warn, 0, [0])

    with suppress_warnings() as sup:
        sup.record()
        with suppress_warnings("always"):
            for i in range(2):
                warnings.warn("Some warning")

        assert_equal(len(sup.log), 2)

    with suppress_warnings() as sup:
        sup.record()
        with suppress_warnings("location"):
            for i in range(2):
                warnings.warn("Some warning")
                warnings.warn("Some warning")

        assert_equal(len(sup.log), 2)

    with suppress_warnings() as sup:
        sup.record()
        with suppress_warnings("module"):
            for i in range(2):
                warnings.warn("Some warning")
                warnings.warn("Some warning")
                warn_other_module()

        assert_equal(len(sup.log), 2)

    with suppress_warnings() as sup:
        sup.record()
        with suppress_warnings("once"):
            for i in range(2):
                warnings.warn("Some warning")
                warnings.warn("Some other warning")
                warn_other_module()

        assert_equal(len(sup.log), 2)


def test_tempdir():
    with tempdir() as tdir:
        fpath = os.path.join(tdir, 'tmp')
        with open(fpath, 'w'):
            pass
    assert_(not os.path.isdir(tdir))

    raised = False
    try:
        with tempdir() as tdir:
            raise ValueError()
    except ValueError:
        raised = True
    assert_(raised)
    assert_(not os.path.isdir(tdir))


def test_temppath():
    with temppath() as fpath:
        with open(fpath, 'w'):
            pass
    assert_(not os.path.isfile(fpath))

    raised = False
    try:
        with temppath() as fpath:
            raise ValueError()
    except ValueError:
        raised = True
    assert_(raised)
    assert_(not os.path.isfile(fpath))


class my_cacw(clear_and_catch_warnings):

    class_modules = (sys.modules[__name__],)


def test_clear_and_catch_warnings_inherit():
    # Test can subclass and add default modules
    my_mod = _get_fresh_mod()
    with my_cacw():
        warnings.simplefilter('ignore')
        warnings.warn('Some warning')
    assert_equal(my_mod.__warningregistry__, {})


@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
class TestAssertNoGcCycles:
    """ Test assert_no_gc_cycles """
    def test_passes(self):
        def no_cycle():
            b = []
            b.append([])
            return b

        with assert_no_gc_cycles():
            no_cycle()

        assert_no_gc_cycles(no_cycle)

    def test_asserts(self):
        def make_cycle():
            a = []
            a.append(a)
            a.append(a)
            return a

        with assert_raises(AssertionError):
            with assert_no_gc_cycles():
                make_cycle()

        with assert_raises(AssertionError):
            assert_no_gc_cycles(make_cycle)

    @pytest.mark.slow
    def test_fails(self):
        """
        Test that in cases where the garbage cannot be collected, we raise an
        error, instead of hanging forever trying to clear it.
        """

        class ReferenceCycleInDel:
            """
            An object that not only contains a reference cycle, but creates new
            cycles whenever it's garbage-collected and its __del__ runs
            """
            make_cycle = True

            def __init__(self):
                self.cycle = self

            def __del__(self):
                # break the current cycle so that `self` can be freed
                self.cycle = None

                if ReferenceCycleInDel.make_cycle:
                    # but create a new one so that the garbage collector has more
                    # work to do.
                    ReferenceCycleInDel()

        try:
            w = weakref.ref(ReferenceCycleInDel())
            try:
                with assert_raises(RuntimeError):
                    # this will be unable to get a baseline empty garbage
                    assert_no_gc_cycles(lambda: None)
            except AssertionError:
                # the above test is only necessary if the GC actually tried to free
                # our object anyway, which python 2.7 does not.
                if w() is not None:
                    pytest.skip("GC does not call __del__ on cyclic objects")
                    raise

        finally:
            # make sure that we stop creating reference cycles
            ReferenceCycleInDel.make_cycle = False
