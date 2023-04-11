"""Test functions for 1D array set operations.

"""
import numpy as np

from numpy.testing import (assert_array_equal, assert_equal,
                           assert_raises, assert_raises_regex)
from numpy.lib.arraysetops import (
    ediff1d, intersect1d, setxor1d, union1d, setdiff1d, unique, in1d, isin
    )
import pytest


class TestSetOps:

    def test_intersect1d(self):
        # unique inputs
        a = np.array([5, 7, 1, 2])
        b = np.array([2, 4, 3, 1, 5])

        ec = np.array([1, 2, 5])
        c = intersect1d(a, b, assume_unique=True)
        assert_array_equal(c, ec)

        # non-unique inputs
        a = np.array([5, 5, 7, 1, 2])
        b = np.array([2, 1, 4, 3, 3, 1, 5])

        ed = np.array([1, 2, 5])
        c = intersect1d(a, b)
        assert_array_equal(c, ed)
        assert_array_equal([], intersect1d([], []))

    def test_intersect1d_array_like(self):
        # See gh-11772
        class Test:
            def __array__(self):
                return np.arange(3)

        a = Test()
        res = intersect1d(a, a)
        assert_array_equal(res, a)
        res = intersect1d([1, 2, 3], [1, 2, 3])
        assert_array_equal(res, [1, 2, 3])

    def test_intersect1d_indices(self):
        # unique inputs
        a = np.array([1, 2, 3, 4])
        b = np.array([2, 1, 4, 6])
        c, i1, i2 = intersect1d(a, b, assume_unique=True, return_indices=True)
        ee = np.array([1, 2, 4])
        assert_array_equal(c, ee)
        assert_array_equal(a[i1], ee)
        assert_array_equal(b[i2], ee)

        # non-unique inputs
        a = np.array([1, 2, 2, 3, 4, 3, 2])
        b = np.array([1, 8, 4, 2, 2, 3, 2, 3])
        c, i1, i2 = intersect1d(a, b, return_indices=True)
        ef = np.array([1, 2, 3, 4])
        assert_array_equal(c, ef)
        assert_array_equal(a[i1], ef)
        assert_array_equal(b[i2], ef)

        # non1d, unique inputs
        a = np.array([[2, 4, 5, 6], [7, 8, 1, 15]])
        b = np.array([[3, 2, 7, 6], [10, 12, 8, 9]])
        c, i1, i2 = intersect1d(a, b, assume_unique=True, return_indices=True)
        ui1 = np.unravel_index(i1, a.shape)
        ui2 = np.unravel_index(i2, b.shape)
        ea = np.array([2, 6, 7, 8])
        assert_array_equal(ea, a[ui1])
        assert_array_equal(ea, b[ui2])

        # non1d, not assumed to be uniqueinputs
        a = np.array([[2, 4, 5, 6, 6], [4, 7, 8, 7, 2]])
        b = np.array([[3, 2, 7, 7], [10, 12, 8, 7]])
        c, i1, i2 = intersect1d(a, b, return_indices=True)
        ui1 = np.unravel_index(i1, a.shape)
        ui2 = np.unravel_index(i2, b.shape)
        ea = np.array([2, 7, 8])
        assert_array_equal(ea, a[ui1])
        assert_array_equal(ea, b[ui2])

    def test_setxor1d(self):
        a = np.array([5, 7, 1, 2])
        b = np.array([2, 4, 3, 1, 5])

        ec = np.array([3, 4, 7])
        c = setxor1d(a, b)
        assert_array_equal(c, ec)

        a = np.array([1, 2, 3])
        b = np.array([6, 5, 4])

        ec = np.array([1, 2, 3, 4, 5, 6])
        c = setxor1d(a, b)
        assert_array_equal(c, ec)

        a = np.array([1, 8, 2, 3])
        b = np.array([6, 5, 4, 8])

        ec = np.array([1, 2, 3, 4, 5, 6])
        c = setxor1d(a, b)
        assert_array_equal(c, ec)

        assert_array_equal([], setxor1d([], []))

    def test_ediff1d(self):
        zero_elem = np.array([])
        one_elem = np.array([1])
        two_elem = np.array([1, 2])

        assert_array_equal([], ediff1d(zero_elem))
        assert_array_equal([0], ediff1d(zero_elem, to_begin=0))
        assert_array_equal([0], ediff1d(zero_elem, to_end=0))
        assert_array_equal([-1, 0], ediff1d(zero_elem, to_begin=-1, to_end=0))
        assert_array_equal([], ediff1d(one_elem))
        assert_array_equal([1], ediff1d(two_elem))
        assert_array_equal([7, 1, 9], ediff1d(two_elem, to_begin=7, to_end=9))
        assert_array_equal([5, 6, 1, 7, 8],
                           ediff1d(two_elem, to_begin=[5, 6], to_end=[7, 8]))
        assert_array_equal([1, 9], ediff1d(two_elem, to_end=9))
        assert_array_equal([1, 7, 8], ediff1d(two_elem, to_end=[7, 8]))
        assert_array_equal([7, 1], ediff1d(two_elem, to_begin=7))
        assert_array_equal([5, 6, 1], ediff1d(two_elem, to_begin=[5, 6]))

    @pytest.mark.parametrize("ary, prepend, append, expected", [
        # should fail because trying to cast
        # np.nan standard floating point value
        # into an integer array:
        (np.array([1, 2, 3], dtype=np.int64),
         None,
         np.nan,
         'to_end'),
        # should fail because attempting
        # to downcast to int type:
        (np.array([1, 2, 3], dtype=np.int64),
         np.array([5, 7, 2], dtype=np.float32),
         None,
         'to_begin'),
        # should fail because attempting to cast
        # two special floating point values
        # to integers (on both sides of ary),
        # `to_begin` is in the error message as the impl checks this first:
        (np.array([1., 3., 9.], dtype=np.int8),
         np.nan,
         np.nan,
         'to_begin'),
         ])
    def test_ediff1d_forbidden_type_casts(self, ary, prepend, append, expected):
        # verify resolution of gh-11490

        # specifically, raise an appropriate
        # Exception when attempting to append or
        # prepend with an incompatible type
        msg = 'dtype of `{}` must be compatible'.format(expected)
        with assert_raises_regex(TypeError, msg):
            ediff1d(ary=ary,
                    to_end=append,
                    to_begin=prepend)

    @pytest.mark.parametrize(
        "ary,prepend,append,expected",
        [
         (np.array([1, 2, 3], dtype=np.int16),
          2**16,  # will be cast to int16 under same kind rule.
          2**16 + 4,
          np.array([0, 1, 1, 4], dtype=np.int16)),
         (np.array([1, 2, 3], dtype=np.float32),
          np.array([5], dtype=np.float64),
          None,
          np.array([5, 1, 1], dtype=np.float32)),
         (np.array([1, 2, 3], dtype=np.int32),
          0,
          0,
          np.array([0, 1, 1, 0], dtype=np.int32)),
         (np.array([1, 2, 3], dtype=np.int64),
          3,
          -9,
          np.array([3, 1, 1, -9], dtype=np.int64)),
        ]
    )
    def test_ediff1d_scalar_handling(self,
                                     ary,
                                     prepend,
                                     append,
                                     expected):
        # maintain backwards-compatibility
        # of scalar prepend / append behavior
        # in ediff1d following fix for gh-11490
        actual = np.ediff1d(ary=ary,
                            to_end=append,
                            to_begin=prepend)
        assert_equal(actual, expected)
        assert actual.dtype == expected.dtype

    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_isin(self, kind):
        # the tests for in1d cover most of isin's behavior
        # if in1d is removed, would need to change those tests to test
        # isin instead.
        def _isin_slow(a, b):
            b = np.asarray(b).flatten().tolist()
            return a in b
        isin_slow = np.vectorize(_isin_slow, otypes=[bool], excluded={1})

        def assert_isin_equal(a, b):
            x = isin(a, b, kind=kind)
            y = isin_slow(a, b)
            assert_array_equal(x, y)

        # multidimensional arrays in both arguments
        a = np.arange(24).reshape([2, 3, 4])
        b = np.array([[10, 20, 30], [0, 1, 3], [11, 22, 33]])
        assert_isin_equal(a, b)

        # array-likes as both arguments
        c = [(9, 8), (7, 6)]
        d = (9, 7)
        assert_isin_equal(c, d)

        # zero-d array:
        f = np.array(3)
        assert_isin_equal(f, b)
        assert_isin_equal(a, f)
        assert_isin_equal(f, f)

        # scalar:
        assert_isin_equal(5, b)
        assert_isin_equal(a, 6)
        assert_isin_equal(5, 6)

        # empty array-like:
        if kind != "table":
            # An empty list will become float64,
            # which is invalid for kind="table"
            x = []
            assert_isin_equal(x, b)
            assert_isin_equal(a, x)
            assert_isin_equal(x, x)

        # empty array with various types:
        for dtype in [bool, np.int64, np.float64]:
            if kind == "table" and dtype == np.float64:
                continue

            if dtype in {np.int64, np.float64}:
                ar = np.array([10, 20, 30], dtype=dtype)
            elif dtype in {bool}:
                ar = np.array([True, False, False])

            empty_array = np.array([], dtype=dtype)

            assert_isin_equal(empty_array, ar)
            assert_isin_equal(ar, empty_array)
            assert_isin_equal(empty_array, empty_array)

    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_in1d(self, kind):
        # we use two different sizes for the b array here to test the
        # two different paths in in1d().
        for mult in (1, 10):
            # One check without np.array to make sure lists are handled correct
            a = [5, 7, 1, 2]
            b = [2, 4, 3, 1, 5] * mult
            ec = np.array([True, False, True, True])
            c = in1d(a, b, assume_unique=True, kind=kind)
            assert_array_equal(c, ec)

            a[0] = 8
            ec = np.array([False, False, True, True])
            c = in1d(a, b, assume_unique=True, kind=kind)
            assert_array_equal(c, ec)

            a[0], a[3] = 4, 8
            ec = np.array([True, False, True, False])
            c = in1d(a, b, assume_unique=True, kind=kind)
            assert_array_equal(c, ec)

            a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5])
            b = [2, 3, 4] * mult
            ec = [False, True, False, True, True, True, True, True, True,
                  False, True, False, False, False]
            c = in1d(a, b, kind=kind)
            assert_array_equal(c, ec)

            b = b + [5, 5, 4] * mult
            ec = [True, True, True, True, True, True, True, True, True, True,
                  True, False, True, True]
            c = in1d(a, b, kind=kind)
            assert_array_equal(c, ec)

            a = np.array([5, 7, 1, 2])
            b = np.array([2, 4, 3, 1, 5] * mult)
            ec = np.array([True, False, True, True])
            c = in1d(a, b, kind=kind)
            assert_array_equal(c, ec)

            a = np.array([5, 7, 1, 1, 2])
            b = np.array([2, 4, 3, 3, 1, 5] * mult)
            ec = np.array([True, False, True, True, True])
            c = in1d(a, b, kind=kind)
            assert_array_equal(c, ec)

            a = np.array([5, 5])
            b = np.array([2, 2] * mult)
            ec = np.array([False, False])
            c = in1d(a, b, kind=kind)
            assert_array_equal(c, ec)

        a = np.array([5])
        b = np.array([2])
        ec = np.array([False])
        c = in1d(a, b, kind=kind)
        assert_array_equal(c, ec)

        if kind in {None, "sort"}:
            assert_array_equal(in1d([], [], kind=kind), [])

    def test_in1d_char_array(self):
        a = np.array(['a', 'b', 'c', 'd', 'e', 'c', 'e', 'b'])
        b = np.array(['a', 'c'])

        ec = np.array([True, False, True, False, False, True, False, False])
        c = in1d(a, b)

        assert_array_equal(c, ec)

    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_in1d_invert(self, kind):
        "Test in1d's invert parameter"
        # We use two different sizes for the b array here to test the
        # two different paths in in1d().
        for mult in (1, 10):
            a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5])
            b = [2, 3, 4] * mult
            assert_array_equal(np.invert(in1d(a, b, kind=kind)),
                               in1d(a, b, invert=True, kind=kind))

        # float:
        if kind in {None, "sort"}:
            for mult in (1, 10):
                a = np.array([5, 4, 5, 3, 4, 4, 3, 4, 3, 5, 2, 1, 5, 5],
                            dtype=np.float32)
                b = [2, 3, 4] * mult
                b = np.array(b, dtype=np.float32)
                assert_array_equal(np.invert(in1d(a, b, kind=kind)),
                                   in1d(a, b, invert=True, kind=kind))

    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_in1d_ravel(self, kind):
        # Test that in1d ravels its input arrays. This is not documented
        # behavior however. The test is to ensure consistentency.
        a = np.arange(6).reshape(2, 3)
        b = np.arange(3, 9).reshape(3, 2)
        long_b = np.arange(3, 63).reshape(30, 2)
        ec = np.array([False, False, False, True, True, True])

        assert_array_equal(in1d(a, b, assume_unique=True, kind=kind),
                           ec)
        assert_array_equal(in1d(a, b, assume_unique=False,
                                kind=kind),
                           ec)
        assert_array_equal(in1d(a, long_b, assume_unique=True,
                                kind=kind),
                           ec)
        assert_array_equal(in1d(a, long_b, assume_unique=False,
                                kind=kind),
                           ec)

    def test_in1d_hit_alternate_algorithm(self):
        """Hit the standard isin code with integers"""
        # Need extreme range to hit standard code
        # This hits it without the use of kind='table'
        a = np.array([5, 4, 5, 3, 4, 4, 1e9], dtype=np.int64)
        b = np.array([2, 3, 4, 1e9], dtype=np.int64)
        expected = np.array([0, 1, 0, 1, 1, 1, 1], dtype=bool)
        assert_array_equal(expected, in1d(a, b))
        assert_array_equal(np.invert(expected), in1d(a, b, invert=True))

        a = np.array([5, 7, 1, 2], dtype=np.int64)
        b = np.array([2, 4, 3, 1, 5, 1e9], dtype=np.int64)
        ec = np.array([True, False, True, True])
        c = in1d(a, b, assume_unique=True)
        assert_array_equal(c, ec)

    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_in1d_boolean(self, kind):
        """Test that in1d works for boolean input"""
        a = np.array([True, False])
        b = np.array([False, False, False])
        expected = np.array([False, True])
        assert_array_equal(expected,
                           in1d(a, b, kind=kind))
        assert_array_equal(np.invert(expected),
                           in1d(a, b, invert=True, kind=kind))

    @pytest.mark.parametrize("kind", [None, "sort"])
    def test_in1d_timedelta(self, kind):
        """Test that in1d works for timedelta input"""
        rstate = np.random.RandomState(0)
        a = rstate.randint(0, 100, size=10)
        b = rstate.randint(0, 100, size=10)
        truth = in1d(a, b)
        a_timedelta = a.astype("timedelta64[s]")
        b_timedelta = b.astype("timedelta64[s]")
        assert_array_equal(truth, in1d(a_timedelta, b_timedelta, kind=kind))

    def test_in1d_table_timedelta_fails(self):
        a = np.array([0, 1, 2], dtype="timedelta64[s]")
        b = a
        # Make sure it raises a value error:
        with pytest.raises(ValueError):
            in1d(a, b, kind="table")

    @pytest.mark.parametrize(
        "dtype1,dtype2",
        [
            (np.int8, np.int16),
            (np.int16, np.int8),
            (np.uint8, np.uint16),
            (np.uint16, np.uint8),
            (np.uint8, np.int16),
            (np.int16, np.uint8),
        ]
    )
    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_in1d_mixed_dtype(self, dtype1, dtype2, kind):
        """Test that in1d works as expected for mixed dtype input."""
        is_dtype2_signed = np.issubdtype(dtype2, np.signedinteger)
        ar1 = np.array([0, 0, 1, 1], dtype=dtype1)

        if is_dtype2_signed:
            ar2 = np.array([-128, 0, 127], dtype=dtype2)
        else:
            ar2 = np.array([127, 0, 255], dtype=dtype2)

        expected = np.array([True, True, False, False])

        expect_failure = kind == "table" and any((
            dtype1 == np.int8 and dtype2 == np.int16,
            dtype1 == np.int16 and dtype2 == np.int8
        ))

        if expect_failure:
            with pytest.raises(RuntimeError, match="exceed the maximum"):
                in1d(ar1, ar2, kind=kind)
        else:
            assert_array_equal(in1d(ar1, ar2, kind=kind), expected)

    @pytest.mark.parametrize("kind", [None, "sort", "table"])
    def test_in1d_mixed_boolean(self, kind):
        """Test that in1d works as expected for bool/int input."""
        for dtype in np.typecodes["AllInteger"]:
            a = np.array([True, False, False], dtype=bool)
            b = np.array([0, 0, 0, 0], dtype=dtype)
            expected = np.array([False, True, True], dtype=bool)
            assert_array_equal(in1d(a, b, kind=kind), expected)

            a, b = b, a
            expected = np.array([True, True, True, True], dtype=bool)
            assert_array_equal(in1d(a, b, kind=kind), expected)

    def test_in1d_first_array_is_object(self):
        ar1 = [None]
        ar2 = np.array([1]*10)
        expected = np.array([False])
        result = np.in1d(ar1, ar2)
        assert_array_equal(result, expected)

    def test_in1d_second_array_is_object(self):
        ar1 = 1
        ar2 = np.array([None]*10)
        expected = np.array([False])
        result = np.in1d(ar1, ar2)
        assert_array_equal(result, expected)

    def test_in1d_both_arrays_are_object(self):
        ar1 = [None]
        ar2 = np.array([None]*10)
        expected = np.array([True])
        result = np.in1d(ar1, ar2)
        assert_array_equal(result, expected)

    def test_in1d_both_arrays_have_structured_dtype(self):
        # Test arrays of a structured data type containing an integer field
        # and a field of dtype `object` allowing for arbitrary Python objects
        dt = np.dtype([('field1', int), ('field2', object)])
        ar1 = np.array([(1, None)], dtype=dt)
        ar2 = np.array([(1, None)]*10, dtype=dt)
        expected = np.array([True])
        result = np.in1d(ar1, ar2)
        assert_array_equal(result, expected)

    def test_in1d_with_arrays_containing_tuples(self):
        ar1 = np.array([(1,), 2], dtype=object)
        ar2 = np.array([(1,), 2], dtype=object)
        expected = np.array([True, True])
        result = np.in1d(ar1, ar2)
        assert_array_equal(result, expected)
        result = np.in1d(ar1, ar2, invert=True)
        assert_array_equal(result, np.invert(expected))

        # An integer is added at the end of the array to make sure
        # that the array builder will create the array with tuples
        # and after it's created the integer is removed.
        # There's a bug in the array constructor that doesn't handle
        # tuples properly and adding the integer fixes that.
        ar1 = np.array([(1,), (2, 1), 1], dtype=object)
        ar1 = ar1[:-1]
        ar2 = np.array([(1,), (2, 1), 1], dtype=object)
        ar2 = ar2[:-1]
        expected = np.array([True, True])
        result = np.in1d(ar1, ar2)
        assert_array_equal(result, expected)
        result = np.in1d(ar1, ar2, invert=True)
        assert_array_equal(result, np.invert(expected))

        ar1 = np.array([(1,), (2, 3), 1], dtype=object)
        ar1 = ar1[:-1]
        ar2 = np.array([(1,), 2], dtype=object)
        expected = np.array([True, False])
        result = np.in1d(ar1, ar2)
        assert_array_equal(result, expected)
        result = np.in1d(ar1, ar2, invert=True)
        assert_array_equal(result, np.invert(expected))

    def test_in1d_errors(self):
        """Test that in1d raises expected errors."""

        # Error 1: `kind` is not one of 'sort' 'table' or None.
        ar1 = np.array([1, 2, 3, 4, 5])
        ar2 = np.array([2, 4, 6, 8, 10])
        assert_raises(ValueError, in1d, ar1, ar2, kind='quicksort')

        # Error 2: `kind="table"` does not work for non-integral arrays.
        obj_ar1 = np.array([1, 'a', 3, 'b', 5], dtype=object)
        obj_ar2 = np.array([1, 'a', 3, 'b', 5], dtype=object)
        assert_raises(ValueError, in1d, obj_ar1, obj_ar2, kind='table')

        for dtype in [np.int32, np.int64]:
            ar1 = np.array([-1, 2, 3, 4, 5], dtype=dtype)
            # The range of this array will overflow:
            overflow_ar2 = np.array([-1, np.iinfo(dtype).max], dtype=dtype)

            # Error 3: `kind="table"` will trigger a runtime error
            #  if there is an integer overflow expected when computing the
            #  range of ar2
            assert_raises(
                RuntimeError,
                in1d, ar1, overflow_ar2, kind='table'
            )

            # Non-error: `kind=None` will *not* trigger a runtime error
            #  if there is an integer overflow, it will switch to
            #  the `sort` algorithm.
            result = np.in1d(ar1, overflow_ar2, kind=None)
            assert_array_equal(result, [True] + [False] * 4)
            result = np.in1d(ar1, overflow_ar2, kind='sort')
            assert_array_equal(result, [True] + [False] * 4)

    def test_union1d(self):
        a = np.array([5, 4, 7, 1, 2])
        b = np.array([2, 4, 3, 3, 2, 1, 5])

        ec = np.array([1, 2, 3, 4, 5, 7])
        c = union1d(a, b)
        assert_array_equal(c, ec)

        # Tests gh-10340, arguments to union1d should be
        # flattened if they are not already 1D
        x = np.array([[0, 1, 2], [3, 4, 5]])
        y = np.array([0, 1, 2, 3, 4])
        ez = np.array([0, 1, 2, 3, 4, 5])
        z = union1d(x, y)
        assert_array_equal(z, ez)

        assert_array_equal([], union1d([], []))

    def test_setdiff1d(self):
        a = np.array([6, 5, 4, 7, 1, 2, 7, 4])
        b = np.array([2, 4, 3, 3, 2, 1, 5])

        ec = np.array([6, 7])
        c = setdiff1d(a, b)
        assert_array_equal(c, ec)

        a = np.arange(21)
        b = np.arange(19)
        ec = np.array([19, 20])
        c = setdiff1d(a, b)
        assert_array_equal(c, ec)

        assert_array_equal([], setdiff1d([], []))
        a = np.array((), np.uint32)
        assert_equal(setdiff1d(a, []).dtype, np.uint32)

    def test_setdiff1d_unique(self):
        a = np.array([3, 2, 1])
        b = np.array([7, 5, 2])
        expected = np.array([3, 1])
        actual = setdiff1d(a, b, assume_unique=True)
        assert_equal(actual, expected)

    def test_setdiff1d_char_array(self):
        a = np.array(['a', 'b', 'c'])
        b = np.array(['a', 'b', 's'])
        assert_array_equal(setdiff1d(a, b), np.array(['c']))

    def test_manyways(self):
        a = np.array([5, 7, 1, 2, 8])
        b = np.array([9, 8, 2, 4, 3, 1, 5])

        c1 = setxor1d(a, b)
        aux1 = intersect1d(a, b)
        aux2 = union1d(a, b)
        c2 = setdiff1d(aux2, aux1)
        assert_array_equal(c1, c2)


class TestUnique:

    def test_unique_1d(self):

        def check_all(a, b, i1, i2, c, dt):
            base_msg = 'check {0} failed for type {1}'

            msg = base_msg.format('values', dt)
            v = unique(a)
            assert_array_equal(v, b, msg)

            msg = base_msg.format('return_index', dt)
            v, j = unique(a, True, False, False)
            assert_array_equal(v, b, msg)
            assert_array_equal(j, i1, msg)

            msg = base_msg.format('return_inverse', dt)
            v, j = unique(a, False, True, False)
            assert_array_equal(v, b, msg)
            assert_array_equal(j, i2, msg)

            msg = base_msg.format('return_counts', dt)
            v, j = unique(a, False, False, True)
            assert_array_equal(v, b, msg)
            assert_array_equal(j, c, msg)

            msg = base_msg.format('return_index and return_inverse', dt)
            v, j1, j2 = unique(a, True, True, False)
            assert_array_equal(v, b, msg)
            assert_array_equal(j1, i1, msg)
            assert_array_equal(j2, i2, msg)

            msg = base_msg.format('return_index and return_counts', dt)
            v, j1, j2 = unique(a, True, False, True)
            assert_array_equal(v, b, msg)
            assert_array_equal(j1, i1, msg)
            assert_array_equal(j2, c, msg)

            msg = base_msg.format('return_inverse and return_counts', dt)
            v, j1, j2 = unique(a, False, True, True)
            assert_array_equal(v, b, msg)
            assert_array_equal(j1, i2, msg)
            assert_array_equal(j2, c, msg)

            msg = base_msg.format(('return_index, return_inverse '
                                   'and return_counts'), dt)
            v, j1, j2, j3 = unique(a, True, True, True)
            assert_array_equal(v, b, msg)
            assert_array_equal(j1, i1, msg)
            assert_array_equal(j2, i2, msg)
            assert_array_equal(j3, c, msg)

        a = [5, 7, 1, 2, 1, 5, 7]*10
        b = [1, 2, 5, 7]
        i1 = [2, 3, 0, 1]
        i2 = [2, 3, 0, 1, 0, 2, 3]*10
        c = np.multiply([2, 1, 2, 2], 10)

        # test for numeric arrays
        types = []
        types.extend(np.typecodes['AllInteger'])
        types.extend(np.typecodes['AllFloat'])
        types.append('datetime64[D]')
        types.append('timedelta64[D]')
        for dt in types:
            aa = np.array(a, dt)
            bb = np.array(b, dt)
            check_all(aa, bb, i1, i2, c, dt)

        # test for object arrays
        dt = 'O'
        aa = np.empty(len(a), dt)
        aa[:] = a
        bb = np.empty(len(b), dt)
        bb[:] = b
        check_all(aa, bb, i1, i2, c, dt)

        # test for structured arrays
        dt = [('', 'i'), ('', 'i')]
        aa = np.array(list(zip(a, a)), dt)
        bb = np.array(list(zip(b, b)), dt)
        check_all(aa, bb, i1, i2, c, dt)

        # test for ticket #2799
        aa = [1. + 0.j, 1 - 1.j, 1]
        assert_array_equal(np.unique(aa), [1. - 1.j, 1. + 0.j])

        # test for ticket #4785
        a = [(1, 2), (1, 2), (2, 3)]
        unq = [1, 2, 3]
        inv = [0, 1, 0, 1, 1, 2]
        a1 = unique(a)
        assert_array_equal(a1, unq)
        a2, a2_inv = unique(a, return_inverse=True)
        assert_array_equal(a2, unq)
        assert_array_equal(a2_inv, inv)

        # test for chararrays with return_inverse (gh-5099)
        a = np.chararray(5)
        a[...] = ''
        a2, a2_inv = np.unique(a, return_inverse=True)
        assert_array_equal(a2_inv, np.zeros(5))

        # test for ticket #9137
        a = []
        a1_idx = np.unique(a, return_index=True)[1]
        a2_inv = np.unique(a, return_inverse=True)[1]
        a3_idx, a3_inv = np.unique(a, return_index=True,
                                   return_inverse=True)[1:]
        assert_equal(a1_idx.dtype, np.intp)
        assert_equal(a2_inv.dtype, np.intp)
        assert_equal(a3_idx.dtype, np.intp)
        assert_equal(a3_inv.dtype, np.intp)

        # test for ticket 2111 - float
        a = [2.0, np.nan, 1.0, np.nan]
        ua = [1.0, 2.0, np.nan]
        ua_idx = [2, 0, 1]
        ua_inv = [1, 2, 0, 2]
        ua_cnt = [1, 1, 2]
        assert_equal(np.unique(a), ua)
        assert_equal(np.unique(a, return_index=True), (ua, ua_idx))
        assert_equal(np.unique(a, return_inverse=True), (ua, ua_inv))
        assert_equal(np.unique(a, return_counts=True), (ua, ua_cnt))

        # test for ticket 2111 - complex
        a = [2.0-1j, np.nan, 1.0+1j, complex(0.0, np.nan), complex(1.0, np.nan)]
        ua = [1.0+1j, 2.0-1j, complex(0.0, np.nan)]
        ua_idx = [2, 0, 3]
        ua_inv = [1, 2, 0, 2, 2]
        ua_cnt = [1, 1, 3]
        assert_equal(np.unique(a), ua)
        assert_equal(np.unique(a, return_index=True), (ua, ua_idx))
        assert_equal(np.unique(a, return_inverse=True), (ua, ua_inv))
        assert_equal(np.unique(a, return_counts=True), (ua, ua_cnt))

        # test for ticket 2111 - datetime64
        nat = np.datetime64('nat')
        a = [np.datetime64('2020-12-26'), nat, np.datetime64('2020-12-24'), nat]
        ua = [np.datetime64('2020-12-24'), np.datetime64('2020-12-26'), nat]
        ua_idx = [2, 0, 1]
        ua_inv = [1, 2, 0, 2]
        ua_cnt = [1, 1, 2]
        assert_equal(np.unique(a), ua)
        assert_equal(np.unique(a, return_index=True), (ua, ua_idx))
        assert_equal(np.unique(a, return_inverse=True), (ua, ua_inv))
        assert_equal(np.unique(a, return_counts=True), (ua, ua_cnt))

        # test for ticket 2111 - timedelta
        nat = np.timedelta64('nat')
        a = [np.timedelta64(1, 'D'), nat, np.timedelta64(1, 'h'), nat]
        ua = [np.timedelta64(1, 'h'), np.timedelta64(1, 'D'), nat]
        ua_idx = [2, 0, 1]
        ua_inv = [1, 2, 0, 2]
        ua_cnt = [1, 1, 2]
        assert_equal(np.unique(a), ua)
        assert_equal(np.unique(a, return_index=True), (ua, ua_idx))
        assert_equal(np.unique(a, return_inverse=True), (ua, ua_inv))
        assert_equal(np.unique(a, return_counts=True), (ua, ua_cnt))

        # test for gh-19300
        all_nans = [np.nan] * 4
        ua = [np.nan]
        ua_idx = [0]
        ua_inv = [0, 0, 0, 0]
        ua_cnt = [4]
        assert_equal(np.unique(all_nans), ua)
        assert_equal(np.unique(all_nans, return_index=True), (ua, ua_idx))
        assert_equal(np.unique(all_nans, return_inverse=True), (ua, ua_inv))
        assert_equal(np.unique(all_nans, return_counts=True), (ua, ua_cnt))

    def test_unique_axis_errors(self):
        assert_raises(TypeError, self._run_axis_tests, object)
        assert_raises(TypeError, self._run_axis_tests,
                      [('a', int), ('b', object)])

        assert_raises(np.AxisError, unique, np.arange(10), axis=2)
        assert_raises(np.AxisError, unique, np.arange(10), axis=-2)

    def test_unique_axis_list(self):
        msg = "Unique failed on list of lists"
        inp = [[0, 1, 0], [0, 1, 0]]
        inp_arr = np.asarray(inp)
        assert_array_equal(unique(inp, axis=0), unique(inp_arr, axis=0), msg)
        assert_array_equal(unique(inp, axis=1), unique(inp_arr, axis=1), msg)

    def test_unique_axis(self):
        types = []
        types.extend(np.typecodes['AllInteger'])
        types.extend(np.typecodes['AllFloat'])
        types.append('datetime64[D]')
        types.append('timedelta64[D]')
        types.append([('a', int), ('b', int)])
        types.append([('a', int), ('b', float)])

        for dtype in types:
            self._run_axis_tests(dtype)

        msg = 'Non-bitwise-equal booleans test failed'
        data = np.arange(10, dtype=np.uint8).reshape(-1, 2).view(bool)
        result = np.array([[False, True], [True, True]], dtype=bool)
        assert_array_equal(unique(data, axis=0), result, msg)

        msg = 'Negative zero equality test failed'
        data = np.array([[-0.0, 0.0], [0.0, -0.0], [-0.0, 0.0], [0.0, -0.0]])
        result = np.array([[-0.0, 0.0]])
        assert_array_equal(unique(data, axis=0), result, msg)

    @pytest.mark.parametrize("axis", [0, -1])
    def test_unique_1d_with_axis(self, axis):
        x = np.array([4, 3, 2, 3, 2, 1, 2, 2])
        uniq = unique(x, axis=axis)
        assert_array_equal(uniq, [1, 2, 3, 4])

    def test_unique_axis_zeros(self):
        # issue 15559
        single_zero = np.empty(shape=(2, 0), dtype=np.int8)
        uniq, idx, inv, cnt = unique(single_zero, axis=0, return_index=True,
                                     return_inverse=True, return_counts=True)

        # there's 1 element of shape (0,) along axis 0
        assert_equal(uniq.dtype, single_zero.dtype)
        assert_array_equal(uniq, np.empty(shape=(1, 0)))
        assert_array_equal(idx, np.array([0]))
        assert_array_equal(inv, np.array([0, 0]))
        assert_array_equal(cnt, np.array([2]))

        # there's 0 elements of shape (2,) along axis 1
        uniq, idx, inv, cnt = unique(single_zero, axis=1, return_index=True,
                                     return_inverse=True, return_counts=True)

        assert_equal(uniq.dtype, single_zero.dtype)
        assert_array_equal(uniq, np.empty(shape=(2, 0)))
        assert_array_equal(idx, np.array([]))
        assert_array_equal(inv, np.array([]))
        assert_array_equal(cnt, np.array([]))

        # test a "complicated" shape
        shape = (0, 2, 0, 3, 0, 4, 0)
        multiple_zeros = np.empty(shape=shape)
        for axis in range(len(shape)):
            expected_shape = list(shape)
            if shape[axis] == 0:
                expected_shape[axis] = 0
            else:
                expected_shape[axis] = 1

            assert_array_equal(unique(multiple_zeros, axis=axis),
                               np.empty(shape=expected_shape))

    def test_unique_masked(self):
        # issue 8664
        x = np.array([64, 0, 1, 2, 3, 63, 63, 0, 0, 0, 1, 2, 0, 63, 0],
                     dtype='uint8')
        y = np.ma.masked_equal(x, 0)

        v = np.unique(y)
        v2, i, c = np.unique(y, return_index=True, return_counts=True)

        msg = 'Unique returned different results when asked for index'
        assert_array_equal(v.data, v2.data, msg)
        assert_array_equal(v.mask, v2.mask, msg)

    def test_unique_sort_order_with_axis(self):
        # These tests fail if sorting along axis is done by treating subarrays
        # as unsigned byte strings.  See gh-10495.
        fmt = "sort order incorrect for integer type '%s'"
        for dt in 'bhilq':
            a = np.array([[-1], [0]], dt)
            b = np.unique(a, axis=0)
            assert_array_equal(a, b, fmt % dt)

    def _run_axis_tests(self, dtype):
        data = np.array([[0, 1, 0, 0],
                         [1, 0, 0, 0],
                         [0, 1, 0, 0],
                         [1, 0, 0, 0]]).astype(dtype)

        msg = 'Unique with 1d array and axis=0 failed'
        result = np.array([0, 1])
        assert_array_equal(unique(data), result.astype(dtype), msg)

        msg = 'Unique with 2d array and axis=0 failed'
        result = np.array([[0, 1, 0, 0], [1, 0, 0, 0]])
        assert_array_equal(unique(data, axis=0), result.astype(dtype), msg)

        msg = 'Unique with 2d array and axis=1 failed'
        result = np.array([[0, 0, 1], [0, 1, 0], [0, 0, 1], [0, 1, 0]])
        assert_array_equal(unique(data, axis=1), result.astype(dtype), msg)

        msg = 'Unique with 3d array and axis=2 failed'
        data3d = np.array([[[1, 1],
                            [1, 0]],
                           [[0, 1],
                            [0, 0]]]).astype(dtype)
        result = np.take(data3d, [1, 0], axis=2)
        assert_array_equal(unique(data3d, axis=2), result, msg)

        uniq, idx, inv, cnt = unique(data, axis=0, return_index=True,
                                     return_inverse=True, return_counts=True)
        msg = "Unique's return_index=True failed with axis=0"
        assert_array_equal(data[idx], uniq, msg)
        msg = "Unique's return_inverse=True failed with axis=0"
        assert_array_equal(uniq[inv], data)
        msg = "Unique's return_counts=True failed with axis=0"
        assert_array_equal(cnt, np.array([2, 2]), msg)

        uniq, idx, inv, cnt = unique(data, axis=1, return_index=True,
                                     return_inverse=True, return_counts=True)
        msg = "Unique's return_index=True failed with axis=1"
        assert_array_equal(data[:, idx], uniq)
        msg = "Unique's return_inverse=True failed with axis=1"
        assert_array_equal(uniq[:, inv], data)
        msg = "Unique's return_counts=True failed with axis=1"
        assert_array_equal(cnt, np.array([2, 1, 1]), msg)

    def test_unique_nanequals(self):
        # issue 20326
        a = np.array([1, 1, np.nan, np.nan, np.nan])
        unq = np.unique(a)
        not_unq = np.unique(a, equal_nan=False)
        assert_array_equal(unq, np.array([1, np.nan]))
        assert_array_equal(not_unq, np.array([1, np.nan, np.nan, np.nan]))
