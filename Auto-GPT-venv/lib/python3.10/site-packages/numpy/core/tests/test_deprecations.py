"""
Tests related to deprecation warnings. Also a convenient place
to document how deprecations should eventually be turned into errors.

"""
import datetime
import operator
import warnings
import pytest
import tempfile
import re
import sys

import numpy as np
from numpy.testing import (
    assert_raises, assert_warns, assert_, assert_array_equal, SkipTest,
    KnownFailureException, break_cycles,
    )

from numpy.core._multiarray_tests import fromstring_null_term_c_api

try:
    import pytz
    _has_pytz = True
except ImportError:
    _has_pytz = False


class _DeprecationTestCase:
    # Just as warning: warnings uses re.match, so the start of this message
    # must match.
    message = ''
    warning_cls = DeprecationWarning

    def setup_method(self):
        self.warn_ctx = warnings.catch_warnings(record=True)
        self.log = self.warn_ctx.__enter__()

        # Do *not* ignore other DeprecationWarnings. Ignoring warnings
        # can give very confusing results because of
        # https://bugs.python.org/issue4180 and it is probably simplest to
        # try to keep the tests cleanly giving only the right warning type.
        # (While checking them set to "error" those are ignored anyway)
        # We still have them show up, because otherwise they would be raised
        warnings.filterwarnings("always", category=self.warning_cls)
        warnings.filterwarnings("always", message=self.message,
                                category=self.warning_cls)

    def teardown_method(self):
        self.warn_ctx.__exit__()

    def assert_deprecated(self, function, num=1, ignore_others=False,
                          function_fails=False,
                          exceptions=np._NoValue,
                          args=(), kwargs={}):
        """Test if DeprecationWarnings are given and raised.

        This first checks if the function when called gives `num`
        DeprecationWarnings, after that it tries to raise these
        DeprecationWarnings and compares them with `exceptions`.
        The exceptions can be different for cases where this code path
        is simply not anticipated and the exception is replaced.

        Parameters
        ----------
        function : callable
            The function to test
        num : int
            Number of DeprecationWarnings to expect. This should normally be 1.
        ignore_others : bool
            Whether warnings of the wrong type should be ignored (note that
            the message is not checked)
        function_fails : bool
            If the function would normally fail, setting this will check for
            warnings inside a try/except block.
        exceptions : Exception or tuple of Exceptions
            Exception to expect when turning the warnings into an error.
            The default checks for DeprecationWarnings. If exceptions is
            empty the function is expected to run successfully.
        args : tuple
            Arguments for `function`
        kwargs : dict
            Keyword arguments for `function`
        """
        __tracebackhide__ = True  # Hide traceback for py.test

        # reset the log
        self.log[:] = []

        if exceptions is np._NoValue:
            exceptions = (self.warning_cls,)

        try:
            function(*args, **kwargs)
        except (Exception if function_fails else tuple()):
            pass

        # just in case, clear the registry
        num_found = 0
        for warning in self.log:
            if warning.category is self.warning_cls:
                num_found += 1
            elif not ignore_others:
                raise AssertionError(
                        "expected %s but got: %s" %
                        (self.warning_cls.__name__, warning.category))
        if num is not None and num_found != num:
            msg = "%i warnings found but %i expected." % (len(self.log), num)
            lst = [str(w) for w in self.log]
            raise AssertionError("\n".join([msg] + lst))

        with warnings.catch_warnings():
            warnings.filterwarnings("error", message=self.message,
                                    category=self.warning_cls)
            try:
                function(*args, **kwargs)
                if exceptions != tuple():
                    raise AssertionError(
                            "No error raised during function call")
            except exceptions:
                if exceptions == tuple():
                    raise AssertionError(
                            "Error raised during function call")

    def assert_not_deprecated(self, function, args=(), kwargs={}):
        """Test that warnings are not raised.

        This is just a shorthand for:

        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)
        """
        self.assert_deprecated(function, num=0, ignore_others=True,
                        exceptions=tuple(), args=args, kwargs=kwargs)


class _VisibleDeprecationTestCase(_DeprecationTestCase):
    warning_cls = np.VisibleDeprecationWarning


class TestComparisonDeprecations(_DeprecationTestCase):
    """This tests the deprecation, for non-element-wise comparison logic.
    This used to mean that when an error occurred during element-wise comparison
    (i.e. broadcasting) NotImplemented was returned, but also in the comparison
    itself, False was given instead of the error.

    Also test FutureWarning for the None comparison.
    """

    message = "elementwise.* comparison failed; .*"

    def test_normal_types(self):
        for op in (operator.eq, operator.ne):
            # Broadcasting errors:
            self.assert_deprecated(op, args=(np.zeros(3), []))
            a = np.zeros(3, dtype='i,i')
            # (warning is issued a couple of times here)
            self.assert_deprecated(op, args=(a, a[:-1]), num=None)

            # ragged array comparison returns True/False
            a = np.array([1, np.array([1,2,3])], dtype=object)
            b = np.array([1, np.array([1,2,3])], dtype=object)
            self.assert_deprecated(op, args=(a, b), num=None)

    def test_string(self):
        # For two string arrays, strings always raised the broadcasting error:
        a = np.array(['a', 'b'])
        b = np.array(['a', 'b', 'c'])
        assert_warns(FutureWarning, lambda x, y: x == y, a, b)

        # The empty list is not cast to string, and this used to pass due
        # to dtype mismatch; now (2018-06-21) it correctly leads to a
        # FutureWarning.
        assert_warns(FutureWarning, lambda: a == [])

    def test_void_dtype_equality_failures(self):
        class NotArray:
            def __array__(self):
                raise TypeError

            # Needed so Python 3 does not raise DeprecationWarning twice.
            def __ne__(self, other):
                return NotImplemented

        self.assert_deprecated(lambda: np.arange(2) == NotArray())
        self.assert_deprecated(lambda: np.arange(2) != NotArray())

    def test_array_richcompare_legacy_weirdness(self):
        # It doesn't really work to use assert_deprecated here, b/c part of
        # the point of assert_deprecated is to check that when warnings are
        # set to "error" mode then the error is propagated -- which is good!
        # But here we are testing a bunch of code that is deprecated *because*
        # it has the habit of swallowing up errors and converting them into
        # different warnings. So assert_warns will have to be sufficient.
        assert_warns(FutureWarning, lambda: np.arange(2) == "a")
        assert_warns(FutureWarning, lambda: np.arange(2) != "a")
        # No warning for scalar comparisons
        with warnings.catch_warnings():
            warnings.filterwarnings("error")
            assert_(not (np.array(0) == "a"))
            assert_(np.array(0) != "a")
            assert_(not (np.int16(0) == "a"))
            assert_(np.int16(0) != "a")

        for arg1 in [np.asarray(0), np.int16(0)]:
            struct = np.zeros(2, dtype="i4,i4")
            for arg2 in [struct, "a"]:
                for f in [operator.lt, operator.le, operator.gt, operator.ge]:
                    with warnings.catch_warnings() as l:
                        warnings.filterwarnings("always")
                        assert_raises(TypeError, f, arg1, arg2)
                        assert_(not l)


class TestDatetime64Timezone(_DeprecationTestCase):
    """Parsing of datetime64 with timezones deprecated in 1.11.0, because
    datetime64 is now timezone naive rather than UTC only.

    It will be quite a while before we can remove this, because, at the very
    least, a lot of existing code uses the 'Z' modifier to avoid conversion
    from local time to UTC, even if otherwise it handles time in a timezone
    naive fashion.
    """
    def test_string(self):
        self.assert_deprecated(np.datetime64, args=('2000-01-01T00+01',))
        self.assert_deprecated(np.datetime64, args=('2000-01-01T00Z',))

    @pytest.mark.skipif(not _has_pytz,
                        reason="The pytz module is not available.")
    def test_datetime(self):
        tz = pytz.timezone('US/Eastern')
        dt = datetime.datetime(2000, 1, 1, 0, 0, tzinfo=tz)
        self.assert_deprecated(np.datetime64, args=(dt,))


class TestArrayDataAttributeAssignmentDeprecation(_DeprecationTestCase):
    """Assigning the 'data' attribute of an ndarray is unsafe as pointed
     out in gh-7093. Eventually, such assignment should NOT be allowed, but
     in the interests of maintaining backwards compatibility, only a Deprecation-
     Warning will be raised instead for the time being to give developers time to
     refactor relevant code.
    """

    def test_data_attr_assignment(self):
        a = np.arange(10)
        b = np.linspace(0, 1, 10)

        self.message = ("Assigning the 'data' attribute is an "
                        "inherently unsafe operation and will "
                        "be removed in the future.")
        self.assert_deprecated(a.__setattr__, args=('data', b.data))


class TestBinaryReprInsufficientWidthParameterForRepresentation(_DeprecationTestCase):
    """
    If a 'width' parameter is passed into ``binary_repr`` that is insufficient to
    represent the number in base 2 (positive) or 2's complement (negative) form,
    the function used to silently ignore the parameter and return a representation
    using the minimal number of bits needed for the form in question. Such behavior
    is now considered unsafe from a user perspective and will raise an error in the future.
    """

    def test_insufficient_width_positive(self):
        args = (10,)
        kwargs = {'width': 2}

        self.message = ("Insufficient bit width provided. This behavior "
                        "will raise an error in the future.")
        self.assert_deprecated(np.binary_repr, args=args, kwargs=kwargs)

    def test_insufficient_width_negative(self):
        args = (-5,)
        kwargs = {'width': 2}

        self.message = ("Insufficient bit width provided. This behavior "
                        "will raise an error in the future.")
        self.assert_deprecated(np.binary_repr, args=args, kwargs=kwargs)


class TestDTypeAttributeIsDTypeDeprecation(_DeprecationTestCase):
    # Deprecated 2021-01-05, NumPy 1.21
    message = r".*`.dtype` attribute"

    def test_deprecation_dtype_attribute_is_dtype(self):
        class dt:
            dtype = "f8"

        class vdt(np.void):
            dtype = "f,f"

        self.assert_deprecated(lambda: np.dtype(dt))
        self.assert_deprecated(lambda: np.dtype(dt()))
        self.assert_deprecated(lambda: np.dtype(vdt))
        self.assert_deprecated(lambda: np.dtype(vdt(1)))


class TestTestDeprecated:
    def test_assert_deprecated(self):
        test_case_instance = _DeprecationTestCase()
        test_case_instance.setup_method()
        assert_raises(AssertionError,
                      test_case_instance.assert_deprecated,
                      lambda: None)

        def foo():
            warnings.warn("foo", category=DeprecationWarning, stacklevel=2)

        test_case_instance.assert_deprecated(foo)
        test_case_instance.teardown_method()


class TestNonNumericConjugate(_DeprecationTestCase):
    """
    Deprecate no-op behavior of ndarray.conjugate on non-numeric dtypes,
    which conflicts with the error behavior of np.conjugate.
    """
    def test_conjugate(self):
        for a in np.array(5), np.array(5j):
            self.assert_not_deprecated(a.conjugate)
        for a in (np.array('s'), np.array('2016', 'M'),
                np.array((1, 2), [('a', int), ('b', int)])):
            self.assert_deprecated(a.conjugate)


class TestNPY_CHAR(_DeprecationTestCase):
    # 2017-05-03, 1.13.0
    def test_npy_char_deprecation(self):
        from numpy.core._multiarray_tests import npy_char_deprecation
        self.assert_deprecated(npy_char_deprecation)
        assert_(npy_char_deprecation() == 'S1')


class TestPyArray_AS1D(_DeprecationTestCase):
    def test_npy_pyarrayas1d_deprecation(self):
        from numpy.core._multiarray_tests import npy_pyarrayas1d_deprecation
        assert_raises(NotImplementedError, npy_pyarrayas1d_deprecation)


class TestPyArray_AS2D(_DeprecationTestCase):
    def test_npy_pyarrayas2d_deprecation(self):
        from numpy.core._multiarray_tests import npy_pyarrayas2d_deprecation
        assert_raises(NotImplementedError, npy_pyarrayas2d_deprecation)


class TestDatetimeEvent(_DeprecationTestCase):
    # 2017-08-11, 1.14.0
    def test_3_tuple(self):
        for cls in (np.datetime64, np.timedelta64):
            # two valid uses - (unit, num) and (unit, num, den, None)
            self.assert_not_deprecated(cls, args=(1, ('ms', 2)))
            self.assert_not_deprecated(cls, args=(1, ('ms', 2, 1, None)))

            # trying to use the event argument, removed in 1.7.0, is deprecated
            # it used to be a uint8
            self.assert_deprecated(cls, args=(1, ('ms', 2, 'event')))
            self.assert_deprecated(cls, args=(1, ('ms', 2, 63)))
            self.assert_deprecated(cls, args=(1, ('ms', 2, 1, 'event')))
            self.assert_deprecated(cls, args=(1, ('ms', 2, 1, 63)))


class TestTruthTestingEmptyArrays(_DeprecationTestCase):
    # 2017-09-25, 1.14.0
    message = '.*truth value of an empty array is ambiguous.*'

    def test_1d(self):
        self.assert_deprecated(bool, args=(np.array([]),))

    def test_2d(self):
        self.assert_deprecated(bool, args=(np.zeros((1, 0)),))
        self.assert_deprecated(bool, args=(np.zeros((0, 1)),))
        self.assert_deprecated(bool, args=(np.zeros((0, 0)),))


class TestBincount(_DeprecationTestCase):
    # 2017-06-01, 1.14.0
    def test_bincount_minlength(self):
        self.assert_deprecated(lambda: np.bincount([1, 2, 3], minlength=None))



class TestGeneratorSum(_DeprecationTestCase):
    # 2018-02-25, 1.15.0
    def test_generator_sum(self):
        self.assert_deprecated(np.sum, args=((i for i in range(5)),))


class TestPositiveOnNonNumerical(_DeprecationTestCase):
    # 2018-06-28, 1.16.0
    def test_positive_on_non_number(self):
        self.assert_deprecated(operator.pos, args=(np.array('foo'),))


class TestFromstring(_DeprecationTestCase):
    # 2017-10-19, 1.14
    def test_fromstring(self):
        self.assert_deprecated(np.fromstring, args=('\x00'*80,))


class TestFromStringAndFileInvalidData(_DeprecationTestCase):
    # 2019-06-08, 1.17.0
    # Tests should be moved to real tests when deprecation is done.
    message = "string or file could not be read to its end"

    @pytest.mark.parametrize("invalid_str", [",invalid_data", "invalid_sep"])
    def test_deprecate_unparsable_data_file(self, invalid_str):
        x = np.array([1.51, 2, 3.51, 4], dtype=float)

        with tempfile.TemporaryFile(mode="w") as f:
            x.tofile(f, sep=',', format='%.2f')
            f.write(invalid_str)

            f.seek(0)
            self.assert_deprecated(lambda: np.fromfile(f, sep=","))
            f.seek(0)
            self.assert_deprecated(lambda: np.fromfile(f, sep=",", count=5))
            # Should not raise:
            with warnings.catch_warnings():
                warnings.simplefilter("error", DeprecationWarning)
                f.seek(0)
                res = np.fromfile(f, sep=",", count=4)
                assert_array_equal(res, x)

    @pytest.mark.parametrize("invalid_str", [",invalid_data", "invalid_sep"])
    def test_deprecate_unparsable_string(self, invalid_str):
        x = np.array([1.51, 2, 3.51, 4], dtype=float)
        x_str = "1.51,2,3.51,4{}".format(invalid_str)

        self.assert_deprecated(lambda: np.fromstring(x_str, sep=","))
        self.assert_deprecated(lambda: np.fromstring(x_str, sep=",", count=5))

        # The C-level API can use not fixed size, but 0 terminated strings,
        # so test that as well:
        bytestr = x_str.encode("ascii")
        self.assert_deprecated(lambda: fromstring_null_term_c_api(bytestr))

        with assert_warns(DeprecationWarning):
            # this is slightly strange, in that fromstring leaves data
            # potentially uninitialized (would be good to error when all is
            # read, but count is larger then actual data maybe).
            res = np.fromstring(x_str, sep=",", count=5)
            assert_array_equal(res[:-1], x)

        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)

            # Should not raise:
            res = np.fromstring(x_str, sep=",", count=4)
            assert_array_equal(res, x)


class Test_GetSet_NumericOps(_DeprecationTestCase):
    # 2018-09-20, 1.16.0
    def test_get_numeric_ops(self):
        from numpy.core._multiarray_tests import getset_numericops
        self.assert_deprecated(getset_numericops, num=2)

        # empty kwargs prevents any state actually changing which would break
        # other tests.
        self.assert_deprecated(np.set_numeric_ops, kwargs={})
        assert_raises(ValueError, np.set_numeric_ops, add='abc')


class TestShape1Fields(_DeprecationTestCase):
    warning_cls = FutureWarning

    # 2019-05-20, 1.17.0
    def test_shape_1_fields(self):
        self.assert_deprecated(np.dtype, args=([('a', int, 1)],))


class TestNonZero(_DeprecationTestCase):
    # 2019-05-26, 1.17.0
    def test_zerod(self):
        self.assert_deprecated(lambda: np.nonzero(np.array(0)))
        self.assert_deprecated(lambda: np.nonzero(np.array(1)))


class TestToString(_DeprecationTestCase):
    # 2020-03-06 1.19.0
    message = re.escape("tostring() is deprecated. Use tobytes() instead.")

    def test_tostring(self):
        arr = np.array(list(b"test\xFF"), dtype=np.uint8)
        self.assert_deprecated(arr.tostring)

    def test_tostring_matches_tobytes(self):
        arr = np.array(list(b"test\xFF"), dtype=np.uint8)
        b = arr.tobytes()
        with assert_warns(DeprecationWarning):
            s = arr.tostring()
        assert s == b


class TestDTypeCoercion(_DeprecationTestCase):
    # 2020-02-06 1.19.0
    message = "Converting .* to a dtype .*is deprecated"
    deprecated_types = [
        # The builtin scalar super types:
        np.generic, np.flexible, np.number,
        np.inexact, np.floating, np.complexfloating,
        np.integer, np.unsignedinteger, np.signedinteger,
        # character is a deprecated S1 special case:
        np.character,
    ]

    def test_dtype_coercion(self):
        for scalar_type in self.deprecated_types:
            self.assert_deprecated(np.dtype, args=(scalar_type,))

    def test_array_construction(self):
        for scalar_type in self.deprecated_types:
            self.assert_deprecated(np.array, args=([], scalar_type,))

    def test_not_deprecated(self):
        # All specific types are not deprecated:
        for group in np.sctypes.values():
            for scalar_type in group:
                self.assert_not_deprecated(np.dtype, args=(scalar_type,))

        for scalar_type in [type, dict, list, tuple]:
            # Typical python types are coerced to object currently:
            self.assert_not_deprecated(np.dtype, args=(scalar_type,))


class BuiltInRoundComplexDType(_DeprecationTestCase):
    # 2020-03-31 1.19.0
    deprecated_types = [np.csingle, np.cdouble, np.clongdouble]
    not_deprecated_types = [
        np.int8, np.int16, np.int32, np.int64,
        np.uint8, np.uint16, np.uint32, np.uint64,
        np.float16, np.float32, np.float64,
    ]

    def test_deprecated(self):
        for scalar_type in self.deprecated_types:
            scalar = scalar_type(0)
            self.assert_deprecated(round, args=(scalar,))
            self.assert_deprecated(round, args=(scalar, 0))
            self.assert_deprecated(round, args=(scalar,), kwargs={'ndigits': 0})

    def test_not_deprecated(self):
        for scalar_type in self.not_deprecated_types:
            scalar = scalar_type(0)
            self.assert_not_deprecated(round, args=(scalar,))
            self.assert_not_deprecated(round, args=(scalar, 0))
            self.assert_not_deprecated(round, args=(scalar,), kwargs={'ndigits': 0})


class TestIncorrectAdvancedIndexWithEmptyResult(_DeprecationTestCase):
    # 2020-05-27, NumPy 1.20.0
    message = "Out of bound index found. This was previously ignored.*"

    @pytest.mark.parametrize("index", [([3, 0],), ([0, 0], [3, 0])])
    def test_empty_subspace(self, index):
        # Test for both a single and two/multiple advanced indices. These
        # This will raise an IndexError in the future.
        arr = np.ones((2, 2, 0))
        self.assert_deprecated(arr.__getitem__, args=(index,))
        self.assert_deprecated(arr.__setitem__, args=(index, 0.))

        # for this array, the subspace is only empty after applying the slice
        arr2 = np.ones((2, 2, 1))
        index2 = (slice(0, 0),) + index
        self.assert_deprecated(arr2.__getitem__, args=(index2,))
        self.assert_deprecated(arr2.__setitem__, args=(index2, 0.))

    def test_empty_index_broadcast_not_deprecated(self):
        arr = np.ones((2, 2, 2))

        index = ([[3], [2]], [])  # broadcast to an empty result.
        self.assert_not_deprecated(arr.__getitem__, args=(index,))
        self.assert_not_deprecated(arr.__setitem__,
                                   args=(index, np.empty((2, 0, 2))))


class TestNonExactMatchDeprecation(_DeprecationTestCase):
    # 2020-04-22
    def test_non_exact_match(self):
        arr = np.array([[3, 6, 6], [4, 5, 1]])
        # misspelt mode check
        self.assert_deprecated(lambda: np.ravel_multi_index(arr, (7, 6), mode='Cilp'))
        # using completely different word with first character as R
        self.assert_deprecated(lambda: np.searchsorted(arr[0], 4, side='Random'))


class TestMatrixInOuter(_DeprecationTestCase):
    # 2020-05-13 NumPy 1.20.0
    message = (r"add.outer\(\) was passed a numpy matrix as "
               r"(first|second) argument.")

    def test_deprecated(self):
        arr = np.array([1, 2, 3])
        m = np.array([1, 2, 3]).view(np.matrix)
        self.assert_deprecated(np.add.outer, args=(m, m), num=2)
        self.assert_deprecated(np.add.outer, args=(arr, m))
        self.assert_deprecated(np.add.outer, args=(m, arr))
        self.assert_not_deprecated(np.add.outer, args=(arr, arr))


class FlatteningConcatenateUnsafeCast(_DeprecationTestCase):
    # NumPy 1.20, 2020-09-03
    message = "concatenate with `axis=None` will use same-kind casting"

    def test_deprecated(self):
        self.assert_deprecated(np.concatenate,
                args=(([0.], [1.]),),
                kwargs=dict(axis=None, out=np.empty(2, dtype=np.int64)))

    def test_not_deprecated(self):
        self.assert_not_deprecated(np.concatenate,
                args=(([0.], [1.]),),
                kwargs={'axis': None, 'out': np.empty(2, dtype=np.int64),
                        'casting': "unsafe"})

        with assert_raises(TypeError):
            # Tests should notice if the deprecation warning is given first...
            np.concatenate(([0.], [1.]), out=np.empty(2, dtype=np.int64),
                           casting="same_kind")


class TestDeprecateSubarrayDTypeDuringArrayCoercion(_DeprecationTestCase):
    warning_cls = FutureWarning
    message = "(creating|casting) an array (with|to) a subarray dtype"

    def test_deprecated_array(self):
        # Arrays are more complex, since they "broadcast" on success:
        arr = np.array([1, 2])

        self.assert_deprecated(lambda: arr.astype("(2)i,"))
        with pytest.warns(FutureWarning):
            res = arr.astype("(2)i,")

        assert_array_equal(res, [[1, 2], [1, 2]])

        self.assert_deprecated(lambda: np.array(arr, dtype="(2)i,"))
        with pytest.warns(FutureWarning):
            res = np.array(arr, dtype="(2)i,")

        assert_array_equal(res, [[1, 2], [1, 2]])

        with pytest.warns(FutureWarning):
            res = np.array([[(1,), (2,)], arr], dtype="(2)i,")

        assert_array_equal(res, [[[1, 1], [2, 2]], [[1, 2], [1, 2]]])

    def test_deprecated_and_error(self):
        # These error paths do not give a warning, but will succeed in the
        # future.
        arr = np.arange(5 * 2).reshape(5, 2)
        def check():
            with pytest.raises(ValueError):
                arr.astype("(2,2)f")

        self.assert_deprecated(check)

        def check():
            with pytest.raises(ValueError):
                np.array(arr, dtype="(2,2)f")

        self.assert_deprecated(check)


class TestFutureWarningArrayLikeNotIterable(_DeprecationTestCase):
    # Deprecated 2020-12-09, NumPy 1.20
    warning_cls = FutureWarning
    message = "The input object of type.*but not a sequence"

    @pytest.mark.parametrize("protocol",
            ["__array__", "__array_interface__", "__array_struct__"])
    def test_deprecated(self, protocol):
        """Test that these objects give a warning since they are not 0-D,
        not coerced at the top level `np.array(obj)`, but nested, and do
        *not* define the sequence protocol.

        NOTE: Tests for the versions including __len__ and __getitem__ exist
              in `test_array_coercion.py` and they can be modified or amended
              when this deprecation expired.
        """
        blueprint = np.arange(10)
        MyArr = type("MyArr", (), {protocol: getattr(blueprint, protocol)})
        self.assert_deprecated(lambda: np.array([MyArr()], dtype=object))

    @pytest.mark.parametrize("protocol",
             ["__array__", "__array_interface__", "__array_struct__"])
    def test_0d_not_deprecated(self, protocol):
        # 0-D always worked (albeit it would use __float__ or similar for the
        # conversion, which may not happen anymore)
        blueprint = np.array(1.)
        MyArr = type("MyArr", (), {protocol: getattr(blueprint, protocol)})
        myarr = MyArr()

        self.assert_not_deprecated(lambda: np.array([myarr], dtype=object))
        res = np.array([myarr], dtype=object)
        expected = np.empty(1, dtype=object)
        expected[0] = myarr
        assert_array_equal(res, expected)

    @pytest.mark.parametrize("protocol",
             ["__array__", "__array_interface__", "__array_struct__"])
    def test_unnested_not_deprecated(self, protocol):
        blueprint = np.arange(10)
        MyArr = type("MyArr", (), {protocol: getattr(blueprint, protocol)})
        myarr = MyArr()

        self.assert_not_deprecated(lambda: np.array(myarr))
        res = np.array(myarr)
        assert_array_equal(res, blueprint)

    @pytest.mark.parametrize("protocol",
             ["__array__", "__array_interface__", "__array_struct__"])
    def test_strange_dtype_handling(self, protocol):
        """The old code would actually use the dtype from the array, but
        then end up not using the array (for dimension discovery)
        """
        blueprint = np.arange(10).astype("f4")
        MyArr = type("MyArr", (), {protocol: getattr(blueprint, protocol),
                                   "__float__": lambda _: 0.5})
        myarr = MyArr()

        # Make sure we warn (and capture the FutureWarning)
        with pytest.warns(FutureWarning, match=self.message):
            res = np.array([[myarr]])

        assert res.shape == (1, 1)
        assert res.dtype == "f4"
        assert res[0, 0] == 0.5

    @pytest.mark.parametrize("protocol",
             ["__array__", "__array_interface__", "__array_struct__"])
    def test_assignment_not_deprecated(self, protocol):
        # If the result is dtype=object we do not unpack a nested array or
        # array-like, if it is nested at exactly the right depth.
        # NOTE: We actually do still call __array__, etc. but ignore the result
        #       in the end. For `dtype=object` we could optimize that away.
        blueprint = np.arange(10).astype("f4")
        MyArr = type("MyArr", (), {protocol: getattr(blueprint, protocol),
                                   "__float__": lambda _: 0.5})
        myarr = MyArr()

        res = np.empty(3, dtype=object)
        def set():
            res[:] = [myarr, myarr, myarr]
        self.assert_not_deprecated(set)
        assert res[0] is myarr
        assert res[1] is myarr
        assert res[2] is myarr


class TestDeprecatedUnpickleObjectScalar(_DeprecationTestCase):
    # Deprecated 2020-11-24, NumPy 1.20
    """
    Technically, it should be impossible to create numpy object scalars,
    but there was an unpickle path that would in theory allow it. That
    path is invalid and must lead to the warning.
    """
    message = "Unpickling a scalar with object dtype is deprecated."

    def test_deprecated(self):
        ctor = np.core.multiarray.scalar
        self.assert_deprecated(lambda: ctor(np.dtype("O"), 1))

try:
    with warnings.catch_warnings():
        warnings.simplefilter("always")
        import nose  # noqa: F401
except ImportError:
    HAVE_NOSE = False
else:
    HAVE_NOSE = True


@pytest.mark.skipif(not HAVE_NOSE, reason="Needs nose")
class TestNoseDecoratorsDeprecated(_DeprecationTestCase):
    class DidntSkipException(Exception):
        pass

    def test_slow(self):
        def _test_slow():
            @np.testing.dec.slow
            def slow_func(x, y, z):
                pass

            assert_(slow_func.slow)
        self.assert_deprecated(_test_slow)

    def test_setastest(self):
        def _test_setastest():
            @np.testing.dec.setastest()
            def f_default(a):
                pass

            @np.testing.dec.setastest(True)
            def f_istest(a):
                pass

            @np.testing.dec.setastest(False)
            def f_isnottest(a):
                pass

            assert_(f_default.__test__)
            assert_(f_istest.__test__)
            assert_(not f_isnottest.__test__)
        self.assert_deprecated(_test_setastest, num=3)

    def test_skip_functions_hardcoded(self):
        def _test_skip_functions_hardcoded():
            @np.testing.dec.skipif(True)
            def f1(x):
                raise self.DidntSkipException

            try:
                f1('a')
            except self.DidntSkipException:
                raise Exception('Failed to skip')
            except SkipTest().__class__:
                pass

            @np.testing.dec.skipif(False)
            def f2(x):
                raise self.DidntSkipException

            try:
                f2('a')
            except self.DidntSkipException:
                pass
            except SkipTest().__class__:
                raise Exception('Skipped when not expected to')
        self.assert_deprecated(_test_skip_functions_hardcoded, num=2)

    def test_skip_functions_callable(self):
        def _test_skip_functions_callable():
            def skip_tester():
                return skip_flag == 'skip me!'

            @np.testing.dec.skipif(skip_tester)
            def f1(x):
                raise self.DidntSkipException

            try:
                skip_flag = 'skip me!'
                f1('a')
            except self.DidntSkipException:
                raise Exception('Failed to skip')
            except SkipTest().__class__:
                pass

            @np.testing.dec.skipif(skip_tester)
            def f2(x):
                raise self.DidntSkipException

            try:
                skip_flag = 'five is right out!'
                f2('a')
            except self.DidntSkipException:
                pass
            except SkipTest().__class__:
                raise Exception('Skipped when not expected to')
        self.assert_deprecated(_test_skip_functions_callable, num=2)

    def test_skip_generators_hardcoded(self):
        def _test_skip_generators_hardcoded():
            @np.testing.dec.knownfailureif(True, "This test is known to fail")
            def g1(x):
                yield from range(x)

            try:
                for j in g1(10):
                    pass
            except KnownFailureException().__class__:
                pass
            else:
                raise Exception('Failed to mark as known failure')

            @np.testing.dec.knownfailureif(False, "This test is NOT known to fail")
            def g2(x):
                yield from range(x)
                raise self.DidntSkipException('FAIL')

            try:
                for j in g2(10):
                    pass
            except KnownFailureException().__class__:
                raise Exception('Marked incorrectly as known failure')
            except self.DidntSkipException:
                pass
        self.assert_deprecated(_test_skip_generators_hardcoded, num=2)

    def test_skip_generators_callable(self):
        def _test_skip_generators_callable():
            def skip_tester():
                return skip_flag == 'skip me!'

            @np.testing.dec.knownfailureif(skip_tester, "This test is known to fail")
            def g1(x):
                yield from range(x)

            try:
                skip_flag = 'skip me!'
                for j in g1(10):
                    pass
            except KnownFailureException().__class__:
                pass
            else:
                raise Exception('Failed to mark as known failure')

            @np.testing.dec.knownfailureif(skip_tester, "This test is NOT known to fail")
            def g2(x):
                yield from range(x)
                raise self.DidntSkipException('FAIL')

            try:
                skip_flag = 'do not skip'
                for j in g2(10):
                    pass
            except KnownFailureException().__class__:
                raise Exception('Marked incorrectly as known failure')
            except self.DidntSkipException:
                pass
        self.assert_deprecated(_test_skip_generators_callable, num=2)

    def test_deprecated(self):
        def _test_deprecated():
            @np.testing.dec.deprecated(True)
            def non_deprecated_func():
                pass

            @np.testing.dec.deprecated()
            def deprecated_func():
                import warnings
                warnings.warn("TEST: deprecated func", DeprecationWarning, stacklevel=1)

            @np.testing.dec.deprecated()
            def deprecated_func2():
                import warnings
                warnings.warn("AHHHH", stacklevel=1)
                raise ValueError

            @np.testing.dec.deprecated()
            def deprecated_func3():
                import warnings
                warnings.warn("AHHHH", stacklevel=1)

            # marked as deprecated, but does not raise DeprecationWarning
            assert_raises(AssertionError, non_deprecated_func)
            # should be silent
            deprecated_func()
            with warnings.catch_warnings(record=True):
                warnings.simplefilter("always")  # do not propagate unrelated warnings
                # fails if deprecated decorator just disables test. See #1453.
                assert_raises(ValueError, deprecated_func2)
                # warning is not a DeprecationWarning
                assert_raises(AssertionError, deprecated_func3)
        self.assert_deprecated(_test_deprecated, num=4)

    def test_parametrize(self):
        def _test_parametrize():
            # dec.parametrize assumes that it is being run by nose. Because
            # we are running under pytest, we need to explicitly check the
            # results.
            @np.testing.dec.parametrize('base, power, expected',
                    [(1, 1, 1),
                    (2, 1, 2),
                    (2, 2, 4)])
            def check_parametrize(base, power, expected):
                assert_(base**power == expected)

            count = 0
            for test in check_parametrize():
                test[0](*test[1:])
                count += 1
            assert_(count == 3)
        self.assert_deprecated(_test_parametrize)


class TestSingleElementSignature(_DeprecationTestCase):
    # Deprecated 2021-04-01, NumPy 1.21
    message = r"The use of a length 1"

    def test_deprecated(self):
        self.assert_deprecated(lambda: np.add(1, 2, signature="d"))
        self.assert_deprecated(lambda: np.add(1, 2, sig=(np.dtype("l"),)))


class TestCtypesGetter(_DeprecationTestCase):
    # Deprecated 2021-05-18, Numpy 1.21.0
    warning_cls = DeprecationWarning
    ctypes = np.array([1]).ctypes

    @pytest.mark.parametrize(
        "name", ["get_data", "get_shape", "get_strides", "get_as_parameter"]
    )
    def test_deprecated(self, name: str) -> None:
        func = getattr(self.ctypes, name)
        self.assert_deprecated(lambda: func())

    @pytest.mark.parametrize(
        "name", ["data", "shape", "strides", "_as_parameter_"]
    )
    def test_not_deprecated(self, name: str) -> None:
        self.assert_not_deprecated(lambda: getattr(self.ctypes, name))


PARTITION_DICT = {
    "partition method": np.arange(10).partition,
    "argpartition method": np.arange(10).argpartition,
    "partition function": lambda kth: np.partition(np.arange(10), kth),
    "argpartition function": lambda kth: np.argpartition(np.arange(10), kth),
}


@pytest.mark.parametrize("func", PARTITION_DICT.values(), ids=PARTITION_DICT)
class TestPartitionBoolIndex(_DeprecationTestCase):
    # Deprecated 2021-09-29, NumPy 1.22
    warning_cls = DeprecationWarning
    message = "Passing booleans as partition index is deprecated"

    def test_deprecated(self, func):
        self.assert_deprecated(lambda: func(True))
        self.assert_deprecated(lambda: func([False, True]))

    def test_not_deprecated(self, func):
        self.assert_not_deprecated(lambda: func(1))
        self.assert_not_deprecated(lambda: func([0, 1]))


class TestMachAr(_DeprecationTestCase):
    # Deprecated 2021-10-19, NumPy 1.22
    warning_cls = DeprecationWarning

    def test_deprecated_module(self):
        self.assert_deprecated(lambda: getattr(np.core, "machar"))

    def test_deprecated_attr(self):
        finfo = np.finfo(float)
        self.assert_deprecated(lambda: getattr(finfo, "machar"))


class TestQuantileInterpolationDeprecation(_DeprecationTestCase):
    # Deprecated 2021-11-08, NumPy 1.22
    @pytest.mark.parametrize("func",
        [np.percentile, np.quantile, np.nanpercentile, np.nanquantile])
    def test_deprecated(self, func):
        self.assert_deprecated(
            lambda: func([0., 1.], 0., interpolation="linear"))
        self.assert_deprecated(
            lambda: func([0., 1.], 0., interpolation="nearest"))

    @pytest.mark.parametrize("func",
            [np.percentile, np.quantile, np.nanpercentile, np.nanquantile])
    def test_both_passed(self, func):
        with warnings.catch_warnings():
            # catch the DeprecationWarning so that it does not raise:
            warnings.simplefilter("always", DeprecationWarning)
            with pytest.raises(TypeError):
                func([0., 1.], 0., interpolation="nearest", method="nearest")


class TestMemEventHook(_DeprecationTestCase):
    # Deprecated 2021-11-18, NumPy 1.23
    def test_mem_seteventhook(self):
        # The actual tests are within the C code in
        # multiarray/_multiarray_tests.c.src
        import numpy.core._multiarray_tests as ma_tests
        with pytest.warns(DeprecationWarning,
                          match='PyDataMem_SetEventHook is deprecated'):
            ma_tests.test_pydatamem_seteventhook_start()
        # force an allocation and free of a numpy array
        # needs to be larger then limit of small memory cacher in ctors.c
        a = np.zeros(1000)
        del a
        break_cycles()
        with pytest.warns(DeprecationWarning,
                          match='PyDataMem_SetEventHook is deprecated'):
            ma_tests.test_pydatamem_seteventhook_end()


class TestArrayFinalizeNone(_DeprecationTestCase):
    message = "Setting __array_finalize__ = None"

    def test_use_none_is_deprecated(self):
        # Deprecated way that ndarray itself showed nothing needs finalizing.
        class NoFinalize(np.ndarray):
            __array_finalize__ = None

        self.assert_deprecated(lambda: np.array(1).view(NoFinalize))

class TestAxisNotMAXDIMS(_DeprecationTestCase):
    # Deprecated 2022-01-08, NumPy 1.23
    message = r"Using `axis=32` \(MAXDIMS\) is deprecated"

    def test_deprecated(self):
        a = np.zeros((1,)*32)
        self.assert_deprecated(lambda: np.repeat(a, 1, axis=np.MAXDIMS))


class TestLoadtxtParseIntsViaFloat(_DeprecationTestCase):
    # Deprecated 2022-07-03, NumPy 1.23
    # This test can be removed without replacement after the deprecation.
    # The tests:
    #   * numpy/lib/tests/test_loadtxt.py::test_integer_signs
    #   * lib/tests/test_loadtxt.py::test_implicit_cast_float_to_int_fails
    # Have a warning filter that needs to be removed.
    message = r"loadtxt\(\): Parsing an integer via a float is deprecated.*"

    @pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
    def test_deprecated_warning(self, dtype):
        with pytest.warns(DeprecationWarning, match=self.message):
            np.loadtxt(["10.5"], dtype=dtype)

    @pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
    def test_deprecated_raised(self, dtype):
        # The DeprecationWarning is chained when raised, so test manually:
        with warnings.catch_warnings():
            warnings.simplefilter("error", DeprecationWarning)
            try:
                np.loadtxt(["10.5"], dtype=dtype)
            except ValueError as e:
                assert isinstance(e.__cause__, DeprecationWarning)


class TestPyIntConversion(_DeprecationTestCase):
    message = r".*stop allowing conversion of out-of-bound.*"

    @pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
    def test_deprecated_scalar(self, dtype):
        dtype = np.dtype(dtype)
        info = np.iinfo(dtype)

        # Cover the most common creation paths (all end up in the
        # same place):
        def scalar(value, dtype):
            dtype.type(value)

        def assign(value, dtype):
            arr = np.array([0, 0, 0], dtype=dtype)
            arr[2] = value

        def create(value, dtype):
            np.array([value], dtype=dtype)

        for creation_func in [scalar, assign, create]:
            try:
                self.assert_deprecated(
                        lambda: creation_func(info.min - 1, dtype))
            except OverflowError:
                pass  # OverflowErrors always happened also before and are OK.

            try:
                self.assert_deprecated(
                        lambda: creation_func(info.max + 1, dtype))
            except OverflowError:
                pass  # OverflowErrors always happened also before and are OK.


class TestDeprecatedGlobals(_DeprecationTestCase):
    # Deprecated 2022-11-17, NumPy 1.24
    def test_type_aliases(self):
        # from builtins
        self.assert_deprecated(lambda: np.bool8)
        self.assert_deprecated(lambda: np.int0)
        self.assert_deprecated(lambda: np.uint0)
        self.assert_deprecated(lambda: np.bytes0)
        self.assert_deprecated(lambda: np.str0)
        self.assert_deprecated(lambda: np.object0)


@pytest.mark.parametrize("name",
        ["bool", "long", "ulong", "str", "bytes", "object"])
def test_future_scalar_attributes(name):
    # FutureWarning added 2022-11-17, NumPy 1.24,
    assert name not in dir(np)  # we may want to not add them
    with pytest.warns(FutureWarning,
            match=f"In the future .*{name}"):
        assert not hasattr(np, name)

    # Unfortunately, they are currently still valid via `np.dtype()`
    np.dtype(name)
    name in np.sctypeDict


# Ignore the above future attribute warning for this test.
@pytest.mark.filterwarnings("ignore:In the future:FutureWarning")
class TestRemovedGlobals:
    # Removed 2023-01-12, NumPy 1.24.0
    # Not a deprecation, but the large error was added to aid those who missed
    # the previous deprecation, and should be removed similarly to one
    # (or faster).
    @pytest.mark.parametrize("name",
            ["object", "bool", "float", "complex", "str", "int"])
    def test_attributeerror_includes_info(self, name):
        msg = f".*\n`np.{name}` was a deprecated alias for the builtin"
        with pytest.raises(AttributeError, match=msg):
            getattr(np, name)
