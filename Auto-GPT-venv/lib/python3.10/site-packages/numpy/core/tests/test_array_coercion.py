"""
Tests for array coercion, mainly through testing `np.array` results directly.
Note that other such tests exist e.g. in `test_api.py` and many corner-cases
are tested (sometimes indirectly) elsewhere.
"""

from itertools import permutations, product

import pytest
from pytest import param

import numpy as np
from numpy.core._rational_tests import rational
from numpy.core._multiarray_umath import _discover_array_parameters

from numpy.testing import (
    assert_array_equal, assert_warns, IS_PYPY)


def arraylikes():
    """
    Generator for functions converting an array into various array-likes.
    If full is True (default) includes array-likes not capable of handling
    all dtypes
    """
    # base array:
    def ndarray(a):
        return a

    yield param(ndarray, id="ndarray")

    # subclass:
    class MyArr(np.ndarray):
        pass

    def subclass(a):
        return a.view(MyArr)

    yield subclass

    class _SequenceLike():
        # We are giving a warning that array-like's were also expected to be
        # sequence-like in `np.array([array_like])`, this can be removed
        # when the deprecation exired (started NumPy 1.20)
        def __len__(self):
            raise TypeError

        def __getitem__(self):
            raise TypeError

    # Array-interface
    class ArrayDunder(_SequenceLike):
        def __init__(self, a):
            self.a = a

        def __array__(self, dtype=None):
            return self.a

    yield param(ArrayDunder, id="__array__")

    # memory-view
    yield param(memoryview, id="memoryview")

    # Array-interface
    class ArrayInterface(_SequenceLike):
        def __init__(self, a):
            self.a = a  # need to hold on to keep interface valid
            self.__array_interface__ = a.__array_interface__

    yield param(ArrayInterface, id="__array_interface__")

    # Array-Struct
    class ArrayStruct(_SequenceLike):
        def __init__(self, a):
            self.a = a  # need to hold on to keep struct valid
            self.__array_struct__ = a.__array_struct__

    yield param(ArrayStruct, id="__array_struct__")


def scalar_instances(times=True, extended_precision=True, user_dtype=True):
    # Hard-coded list of scalar instances.
    # Floats:
    yield param(np.sqrt(np.float16(5)), id="float16")
    yield param(np.sqrt(np.float32(5)), id="float32")
    yield param(np.sqrt(np.float64(5)), id="float64")
    if extended_precision:
        yield param(np.sqrt(np.longdouble(5)), id="longdouble")

    # Complex:
    yield param(np.sqrt(np.complex64(2+3j)), id="complex64")
    yield param(np.sqrt(np.complex128(2+3j)), id="complex128")
    if extended_precision:
        yield param(np.sqrt(np.longcomplex(2+3j)), id="clongdouble")

    # Bool:
    # XFAIL: Bool should be added, but has some bad properties when it
    # comes to strings, see also gh-9875
    # yield param(np.bool_(0), id="bool")

    # Integers:
    yield param(np.int8(2), id="int8")
    yield param(np.int16(2), id="int16")
    yield param(np.int32(2), id="int32")
    yield param(np.int64(2), id="int64")

    yield param(np.uint8(2), id="uint8")
    yield param(np.uint16(2), id="uint16")
    yield param(np.uint32(2), id="uint32")
    yield param(np.uint64(2), id="uint64")

    # Rational:
    if user_dtype:
        yield param(rational(1, 2), id="rational")

    # Cannot create a structured void scalar directly:
    structured = np.array([(1, 3)], "i,i")[0]
    assert isinstance(structured, np.void)
    assert structured.dtype == np.dtype("i,i")
    yield param(structured, id="structured")

    if times:
        # Datetimes and timedelta
        yield param(np.timedelta64(2), id="timedelta64[generic]")
        yield param(np.timedelta64(23, "s"), id="timedelta64[s]")
        yield param(np.timedelta64("NaT", "s"), id="timedelta64[s](NaT)")

        yield param(np.datetime64("NaT"), id="datetime64[generic](NaT)")
        yield param(np.datetime64("2020-06-07 12:43", "ms"), id="datetime64[ms]")

    # Strings and unstructured void:
    yield param(np.bytes_(b"1234"), id="bytes")
    yield param(np.unicode_("2345"), id="unicode")
    yield param(np.void(b"4321"), id="unstructured_void")


def is_parametric_dtype(dtype):
    """Returns True if the dtype is a parametric legacy dtype (itemsize
    is 0, or a datetime without units)
    """
    if dtype.itemsize == 0:
        return True
    if issubclass(dtype.type, (np.datetime64, np.timedelta64)):
        if dtype.name.endswith("64"):
            # Generic time units
            return True
    return False


class TestStringDiscovery:
    @pytest.mark.parametrize("obj",
            [object(), 1.2, 10**43, None, "string"],
            ids=["object", "1.2", "10**43", "None", "string"])
    def test_basic_stringlength(self, obj):
        length = len(str(obj))
        expected = np.dtype(f"S{length}")

        assert np.array(obj, dtype="S").dtype == expected
        assert np.array([obj], dtype="S").dtype == expected

        # A nested array is also discovered correctly
        arr = np.array(obj, dtype="O")
        assert np.array(arr, dtype="S").dtype == expected
        # Check that .astype() behaves identical
        assert arr.astype("S").dtype == expected

    @pytest.mark.parametrize("obj",
            [object(), 1.2, 10**43, None, "string"],
            ids=["object", "1.2", "10**43", "None", "string"])
    def test_nested_arrays_stringlength(self, obj):
        length = len(str(obj))
        expected = np.dtype(f"S{length}")
        arr = np.array(obj, dtype="O")
        assert np.array([arr, arr], dtype="S").dtype == expected

    @pytest.mark.parametrize("arraylike", arraylikes())
    def test_unpack_first_level(self, arraylike):
        # We unpack exactly one level of array likes
        obj = np.array([None])
        obj[0] = np.array(1.2)
        # the length of the included item, not of the float dtype
        length = len(str(obj[0]))
        expected = np.dtype(f"S{length}")

        obj = arraylike(obj)
        # casting to string usually calls str(obj)
        arr = np.array([obj], dtype="S")
        assert arr.shape == (1, 1)
        assert arr.dtype == expected


class TestScalarDiscovery:
    def test_void_special_case(self):
        # Void dtypes with structures discover tuples as elements
        arr = np.array((1, 2, 3), dtype="i,i,i")
        assert arr.shape == ()
        arr = np.array([(1, 2, 3)], dtype="i,i,i")
        assert arr.shape == (1,)

    def test_char_special_case(self):
        arr = np.array("string", dtype="c")
        assert arr.shape == (6,)
        assert arr.dtype.char == "c"
        arr = np.array(["string"], dtype="c")
        assert arr.shape == (1, 6)
        assert arr.dtype.char == "c"

    def test_char_special_case_deep(self):
        # Check that the character special case errors correctly if the
        # array is too deep:
        nested = ["string"]  # 2 dimensions (due to string being sequence)
        for i in range(np.MAXDIMS - 2):
            nested = [nested]

        arr = np.array(nested, dtype='c')
        assert arr.shape == (1,) * (np.MAXDIMS - 1) + (6,)
        with pytest.raises(ValueError):
            np.array([nested], dtype="c")

    def test_unknown_object(self):
        arr = np.array(object())
        assert arr.shape == ()
        assert arr.dtype == np.dtype("O")

    @pytest.mark.parametrize("scalar", scalar_instances())
    def test_scalar(self, scalar):
        arr = np.array(scalar)
        assert arr.shape == ()
        assert arr.dtype == scalar.dtype

        arr = np.array([[scalar, scalar]])
        assert arr.shape == (1, 2)
        assert arr.dtype == scalar.dtype

    # Additionally to string this test also runs into a corner case
    # with datetime promotion (the difference is the promotion order).
    @pytest.mark.filterwarnings("ignore:Promotion of numbers:FutureWarning")
    def test_scalar_promotion(self):
        for sc1, sc2 in product(scalar_instances(), scalar_instances()):
            sc1, sc2 = sc1.values[0], sc2.values[0]
            # test all combinations:
            try:
                arr = np.array([sc1, sc2])
            except (TypeError, ValueError):
                # The promotion between two times can fail
                # XFAIL (ValueError): Some object casts are currently undefined
                continue
            assert arr.shape == (2,)
            try:
                dt1, dt2 = sc1.dtype, sc2.dtype
                expected_dtype = np.promote_types(dt1, dt2)
                assert arr.dtype == expected_dtype
            except TypeError as e:
                # Will currently always go to object dtype
                assert arr.dtype == np.dtype("O")

    @pytest.mark.parametrize("scalar", scalar_instances())
    def test_scalar_coercion(self, scalar):
        # This tests various scalar coercion paths, mainly for the numerical
        # types.  It includes some paths not directly related to `np.array`
        if isinstance(scalar, np.inexact):
            # Ensure we have a full-precision number if available
            scalar = type(scalar)((scalar * 2)**0.5)

        if type(scalar) is rational:
            # Rational generally fails due to a missing cast. In the future
            # object casts should automatically be defined based on `setitem`.
            pytest.xfail("Rational to object cast is undefined currently.")

        # Use casting from object:
        arr = np.array(scalar, dtype=object).astype(scalar.dtype)

        # Test various ways to create an array containing this scalar:
        arr1 = np.array(scalar).reshape(1)
        arr2 = np.array([scalar])
        arr3 = np.empty(1, dtype=scalar.dtype)
        arr3[0] = scalar
        arr4 = np.empty(1, dtype=scalar.dtype)
        arr4[:] = [scalar]
        # All of these methods should yield the same results
        assert_array_equal(arr, arr1)
        assert_array_equal(arr, arr2)
        assert_array_equal(arr, arr3)
        assert_array_equal(arr, arr4)

    @pytest.mark.xfail(IS_PYPY, reason="`int(np.complex128(3))` fails on PyPy")
    @pytest.mark.filterwarnings("ignore::numpy.ComplexWarning")
    @pytest.mark.parametrize("cast_to", scalar_instances())
    def test_scalar_coercion_same_as_cast_and_assignment(self, cast_to):
        """
        Test that in most cases:
           * `np.array(scalar, dtype=dtype)`
           * `np.empty((), dtype=dtype)[()] = scalar`
           * `np.array(scalar).astype(dtype)`
        should behave the same.  The only exceptions are paramteric dtypes
        (mainly datetime/timedelta without unit) and void without fields.
        """
        dtype = cast_to.dtype  # use to parametrize only the target dtype

        for scalar in scalar_instances(times=False):
            scalar = scalar.values[0]

            if dtype.type == np.void:
               if scalar.dtype.fields is not None and dtype.fields is None:
                    # Here, coercion to "V6" works, but the cast fails.
                    # Since the types are identical, SETITEM takes care of
                    # this, but has different rules than the cast.
                    with pytest.raises(TypeError):
                        np.array(scalar).astype(dtype)
                    np.array(scalar, dtype=dtype)
                    np.array([scalar], dtype=dtype)
                    continue

            # The main test, we first try to use casting and if it succeeds
            # continue below testing that things are the same, otherwise
            # test that the alternative paths at least also fail.
            try:
                cast = np.array(scalar).astype(dtype)
            except (TypeError, ValueError, RuntimeError):
                # coercion should also raise (error type may change)
                with pytest.raises(Exception):
                    np.array(scalar, dtype=dtype)

                if (isinstance(scalar, rational) and
                        np.issubdtype(dtype, np.signedinteger)):
                    return

                with pytest.raises(Exception):
                    np.array([scalar], dtype=dtype)
                # assignment should also raise
                res = np.zeros((), dtype=dtype)
                with pytest.raises(Exception):
                    res[()] = scalar

                return

            # Non error path:
            arr = np.array(scalar, dtype=dtype)
            assert_array_equal(arr, cast)
            # assignment behaves the same
            ass = np.zeros((), dtype=dtype)
            ass[()] = scalar
            assert_array_equal(ass, cast)

    @pytest.mark.parametrize("pyscalar", [10, 10.32, 10.14j, 10**100])
    def test_pyscalar_subclasses(self, pyscalar):
        """NumPy arrays are read/write which means that anything but invariant
        behaviour is on thin ice.  However, we currently are happy to discover
        subclasses of Python float, int, complex the same as the base classes.
        This should potentially be deprecated.
        """
        class MyScalar(type(pyscalar)):
            pass

        res = np.array(MyScalar(pyscalar))
        expected = np.array(pyscalar)
        assert_array_equal(res, expected)

    @pytest.mark.parametrize("dtype_char", np.typecodes["All"])
    def test_default_dtype_instance(self, dtype_char):
        if dtype_char in "SU":
            dtype = np.dtype(dtype_char + "1")
        elif dtype_char == "V":
            # Legacy behaviour was to use V8. The reason was float64 being the
            # default dtype and that having 8 bytes.
            dtype = np.dtype("V8")
        else:
            dtype = np.dtype(dtype_char)

        discovered_dtype, _ = _discover_array_parameters([], type(dtype))

        assert discovered_dtype == dtype
        assert discovered_dtype.itemsize == dtype.itemsize

    @pytest.mark.parametrize("dtype", np.typecodes["Integer"])
    @pytest.mark.parametrize(["scalar", "error"],
            [(np.float64(np.nan), ValueError),
             (np.array(-1).astype(np.ulonglong)[()], OverflowError)])
    def test_scalar_to_int_coerce_does_not_cast(self, dtype, scalar, error):
        """
        Signed integers are currently different in that they do not cast other
        NumPy scalar, but instead use scalar.__int__(). The hardcoded
        exception to this rule is `np.array(scalar, dtype=integer)`.
        """
        dtype = np.dtype(dtype)

        # This is a special case using casting logic.  It warns for the NaN
        # but allows the cast (giving undefined behaviour).
        with np.errstate(invalid="ignore"):
            coerced = np.array(scalar, dtype=dtype)
            cast = np.array(scalar).astype(dtype)
        assert_array_equal(coerced, cast)

        # However these fail:
        with pytest.raises(error):
            np.array([scalar], dtype=dtype)
        with pytest.raises(error):
            cast[()] = scalar


class TestTimeScalars:
    @pytest.mark.parametrize("dtype", [np.int64, np.float32])
    @pytest.mark.parametrize("scalar",
            [param(np.timedelta64("NaT", "s"), id="timedelta64[s](NaT)"),
             param(np.timedelta64(123, "s"), id="timedelta64[s]"),
             param(np.datetime64("NaT", "generic"), id="datetime64[generic](NaT)"),
             param(np.datetime64(1, "D"), id="datetime64[D]")],)
    def test_coercion_basic(self, dtype, scalar):
        # Note the `[scalar]` is there because np.array(scalar) uses stricter
        # `scalar.__int__()` rules for backward compatibility right now.
        arr = np.array(scalar, dtype=dtype)
        cast = np.array(scalar).astype(dtype)
        assert_array_equal(arr, cast)

        ass = np.ones((), dtype=dtype)
        if issubclass(dtype, np.integer):
            with pytest.raises(TypeError):
                # raises, as would np.array([scalar], dtype=dtype), this is
                # conversion from times, but behaviour of integers.
                ass[()] = scalar
        else:
            ass[()] = scalar
            assert_array_equal(ass, cast)

    @pytest.mark.parametrize("dtype", [np.int64, np.float32])
    @pytest.mark.parametrize("scalar",
            [param(np.timedelta64(123, "ns"), id="timedelta64[ns]"),
             param(np.timedelta64(12, "generic"), id="timedelta64[generic]")])
    def test_coercion_timedelta_convert_to_number(self, dtype, scalar):
        # Only "ns" and "generic" timedeltas can be converted to numbers
        # so these are slightly special.
        arr = np.array(scalar, dtype=dtype)
        cast = np.array(scalar).astype(dtype)
        ass = np.ones((), dtype=dtype)
        ass[()] = scalar  # raises, as would np.array([scalar], dtype=dtype)

        assert_array_equal(arr, cast)
        assert_array_equal(cast, cast)

    @pytest.mark.parametrize("dtype", ["S6", "U6"])
    @pytest.mark.parametrize(["val", "unit"],
            [param(123, "s", id="[s]"), param(123, "D", id="[D]")])
    def test_coercion_assignment_datetime(self, val, unit, dtype):
        # String from datetime64 assignment is currently special cased to
        # never use casting.  This is because casting will error in this
        # case, and traditionally in most cases the behaviour is maintained
        # like this.  (`np.array(scalar, dtype="U6")` would have failed before)
        # TODO: This discrepancy _should_ be resolved, either by relaxing the
        #       cast, or by deprecating the first part.
        scalar = np.datetime64(val, unit)
        dtype = np.dtype(dtype)
        cut_string = dtype.type(str(scalar)[:6])

        arr = np.array(scalar, dtype=dtype)
        assert arr[()] == cut_string
        ass = np.ones((), dtype=dtype)
        ass[()] = scalar
        assert ass[()] == cut_string

        with pytest.raises(RuntimeError):
            # However, unlike the above assignment using `str(scalar)[:6]`
            # due to being handled by the string DType and not be casting
            # the explicit cast fails:
            np.array(scalar).astype(dtype)


    @pytest.mark.parametrize(["val", "unit"],
            [param(123, "s", id="[s]"), param(123, "D", id="[D]")])
    def test_coercion_assignment_timedelta(self, val, unit):
        scalar = np.timedelta64(val, unit)

        # Unlike datetime64, timedelta allows the unsafe cast:
        np.array(scalar, dtype="S6")
        cast = np.array(scalar).astype("S6")
        ass = np.ones((), dtype="S6")
        ass[()] = scalar
        expected = scalar.astype("S")[:6]
        assert cast[()] == expected
        assert ass[()] == expected

class TestNested:
    def test_nested_simple(self):
        initial = [1.2]
        nested = initial
        for i in range(np.MAXDIMS - 1):
            nested = [nested]

        arr = np.array(nested, dtype="float64")
        assert arr.shape == (1,) * np.MAXDIMS
        with pytest.raises(ValueError):
            np.array([nested], dtype="float64")

        with pytest.raises(ValueError, match=".*would exceed the maximum"):
            np.array([nested])  # user must ask for `object` explicitly

        arr = np.array([nested], dtype=object)
        assert arr.dtype == np.dtype("O")
        assert arr.shape == (1,) * np.MAXDIMS
        assert arr.item() is initial

    def test_pathological_self_containing(self):
        # Test that this also works for two nested sequences
        l = []
        l.append(l)
        arr = np.array([l, l, l], dtype=object)
        assert arr.shape == (3,) + (1,) * (np.MAXDIMS - 1)

        # Also check a ragged case:
        arr = np.array([l, [None], l], dtype=object)
        assert arr.shape == (3, 1)

    @pytest.mark.parametrize("arraylike", arraylikes())
    def test_nested_arraylikes(self, arraylike):
        # We try storing an array like into an array, but the array-like
        # will have too many dimensions.  This means the shape discovery
        # decides that the array-like must be treated as an object (a special
        # case of ragged discovery).  The result will be an array with one
        # dimension less than the maximum dimensions, and the array being
        # assigned to it (which does work for object or if `float(arraylike)`
        # works).
        initial = arraylike(np.ones((1, 1)))

        nested = initial
        for i in range(np.MAXDIMS - 1):
            nested = [nested]

        with pytest.raises(ValueError, match=".*would exceed the maximum"):
            # It will refuse to assign the array into
            np.array(nested, dtype="float64")

        # If this is object, we end up assigning a (1, 1) array into (1,)
        # (due to running out of dimensions), this is currently supported but
        # a special case which is not ideal.
        arr = np.array(nested, dtype=object)
        assert arr.shape == (1,) * np.MAXDIMS
        assert arr.item() == np.array(initial).item()

    @pytest.mark.parametrize("arraylike", arraylikes())
    def test_uneven_depth_ragged(self, arraylike):
        arr = np.arange(4).reshape((2, 2))
        arr = arraylike(arr)

        # Array is ragged in the second dimension already:
        out = np.array([arr, [arr]], dtype=object)
        assert out.shape == (2,)
        assert out[0] is arr
        assert type(out[1]) is list

        # Array is ragged in the third dimension:
        with pytest.raises(ValueError):
            # This is a broadcast error during assignment, because
            # the array shape would be (2, 2, 2) but `arr[0, 0] = arr` fails.
            np.array([arr, [arr, arr]], dtype=object)

    def test_empty_sequence(self):
        arr = np.array([[], [1], [[1]]], dtype=object)
        assert arr.shape == (3,)

        # The empty sequence stops further dimension discovery, so the
        # result shape will be (0,) which leads to an error during:
        with pytest.raises(ValueError):
            np.array([[], np.empty((0, 1))], dtype=object)

    def test_array_of_different_depths(self):
        # When multiple arrays (or array-likes) are included in a
        # sequences and have different depth, we currently discover
        # as many dimensions as they share. (see also gh-17224)
        arr = np.zeros((3, 2))
        mismatch_first_dim = np.zeros((1, 2))
        mismatch_second_dim = np.zeros((3, 3))

        dtype, shape = _discover_array_parameters(
            [arr, mismatch_second_dim], dtype=np.dtype("O"))
        assert shape == (2, 3)

        dtype, shape = _discover_array_parameters(
            [arr, mismatch_first_dim], dtype=np.dtype("O"))
        assert shape == (2,)
        # The second case is currently supported because the arrays
        # can be stored as objects:
        res = np.asarray([arr, mismatch_first_dim], dtype=np.dtype("O"))
        assert res[0] is arr
        assert res[1] is mismatch_first_dim


class TestBadSequences:
    # These are tests for bad objects passed into `np.array`, in general
    # these have undefined behaviour.  In the old code they partially worked
    # when now they will fail.  We could (and maybe should) create a copy
    # of all sequences to be safe against bad-actors.

    def test_growing_list(self):
        # List to coerce, `mylist` will append to it during coercion
        obj = []
        class mylist(list):
            def __len__(self):
                obj.append([1, 2])
                return super().__len__()

        obj.append(mylist([1, 2]))

        with pytest.raises(RuntimeError):
            np.array(obj)

    # Note: We do not test a shrinking list.  These do very evil things
    #       and the only way to fix them would be to copy all sequences.
    #       (which may be a real option in the future).

    def test_mutated_list(self):
        # List to coerce, `mylist` will mutate the first element
        obj = []
        class mylist(list):
            def __len__(self):
                obj[0] = [2, 3]  # replace with a different list.
                return super().__len__()

        obj.append([2, 3])
        obj.append(mylist([1, 2]))
        # Does not crash:
        np.array(obj)

    def test_replace_0d_array(self):
        # List to coerce, `mylist` will mutate the first element
        obj = []
        class baditem:
            def __len__(self):
                obj[0][0] = 2  # replace with a different list.
                raise ValueError("not actually a sequence!")

            def __getitem__(self):
                pass

        # Runs into a corner case in the new code, the `array(2)` is cached
        # so replacing it invalidates the cache.
        obj.append([np.array(2), baditem()])
        with pytest.raises(RuntimeError):
            np.array(obj)


class TestArrayLikes:
    @pytest.mark.parametrize("arraylike", arraylikes())
    def test_0d_object_special_case(self, arraylike):
        arr = np.array(0.)
        obj = arraylike(arr)
        # A single array-like is always converted:
        res = np.array(obj, dtype=object)
        assert_array_equal(arr, res)

        # But a single 0-D nested array-like never:
        res = np.array([obj], dtype=object)
        assert res[0] is obj

    def test_0d_generic_special_case(self):
        class ArraySubclass(np.ndarray):
            def __float__(self):
                raise TypeError("e.g. quantities raise on this")

        arr = np.array(0.)
        obj = arr.view(ArraySubclass)
        res = np.array(obj)
        # The subclass is simply cast:
        assert_array_equal(arr, res)

        # If the 0-D array-like is included, __float__ is currently
        # guaranteed to be used.  We may want to change that, quantities
        # and masked arrays half make use of this.
        with pytest.raises(TypeError):
            np.array([obj])

        # The same holds for memoryview:
        obj = memoryview(arr)
        res = np.array(obj)
        assert_array_equal(arr, res)
        with pytest.raises(ValueError):
            # The error type does not matter much here.
            np.array([obj])

    def test_arraylike_classes(self):
        # The classes of array-likes should generally be acceptable to be
        # stored inside a numpy (object) array.  This tests all of the
        # special attributes (since all are checked during coercion).
        arr = np.array(np.int64)
        assert arr[()] is np.int64
        arr = np.array([np.int64])
        assert arr[0] is np.int64

        # This also works for properties/unbound methods:
        class ArrayLike:
            @property
            def __array_interface__(self):
                pass

            @property
            def __array_struct__(self):
                pass

            def __array__(self):
                pass

        arr = np.array(ArrayLike)
        assert arr[()] is ArrayLike
        arr = np.array([ArrayLike])
        assert arr[0] is ArrayLike

    @pytest.mark.skipif(
            np.dtype(np.intp).itemsize < 8, reason="Needs 64bit platform")
    def test_too_large_array_error_paths(self):
        """Test the error paths, including for memory leaks"""
        arr = np.array(0, dtype="uint8")
        # Guarantees that a contiguous copy won't work:
        arr = np.broadcast_to(arr, 2**62)

        for i in range(5):
            # repeat, to ensure caching cannot have an effect:
            with pytest.raises(MemoryError):
                np.array(arr)
            with pytest.raises(MemoryError):
                np.array([arr])

    @pytest.mark.parametrize("attribute",
        ["__array_interface__", "__array__", "__array_struct__"])
    @pytest.mark.parametrize("error", [RecursionError, MemoryError])
    def test_bad_array_like_attributes(self, attribute, error):
        # RecursionError and MemoryError are considered fatal. All errors
        # (except AttributeError) should probably be raised in the future,
        # but shapely made use of it, so it will require a deprecation.

        class BadInterface:
            def __getattr__(self, attr):
                if attr == attribute:
                    raise error
                super().__getattr__(attr)

        with pytest.raises(error):
            np.array(BadInterface())

    @pytest.mark.parametrize("error", [RecursionError, MemoryError])
    def test_bad_array_like_bad_length(self, error):
        # RecursionError and MemoryError are considered "critical" in
        # sequences. We could expand this more generally though. (NumPy 1.20)
        class BadSequence:
            def __len__(self):
                raise error
            def __getitem__(self):
                # must have getitem to be a Sequence
                return 1

        with pytest.raises(error):
            np.array(BadSequence())


class TestAsArray:
    """Test expected behaviors of ``asarray``."""

    def test_dtype_identity(self):
        """Confirm the intended behavior for *dtype* kwarg.

        The result of ``asarray()`` should have the dtype provided through the
        keyword argument, when used. This forces unique array handles to be
        produced for unique np.dtype objects, but (for equivalent dtypes), the
        underlying data (the base object) is shared with the original array
        object.

        Ref https://github.com/numpy/numpy/issues/1468
        """
        int_array = np.array([1, 2, 3], dtype='i')
        assert np.asarray(int_array) is int_array

        # The character code resolves to the singleton dtype object provided
        # by the numpy package.
        assert np.asarray(int_array, dtype='i') is int_array

        # Derive a dtype from n.dtype('i'), but add a metadata object to force
        # the dtype to be distinct.
        unequal_type = np.dtype('i', metadata={'spam': True})
        annotated_int_array = np.asarray(int_array, dtype=unequal_type)
        assert annotated_int_array is not int_array
        assert annotated_int_array.base is int_array
        # Create an equivalent descriptor with a new and distinct dtype
        # instance.
        equivalent_requirement = np.dtype('i', metadata={'spam': True})
        annotated_int_array_alt = np.asarray(annotated_int_array,
                                             dtype=equivalent_requirement)
        assert unequal_type == equivalent_requirement
        assert unequal_type is not equivalent_requirement
        assert annotated_int_array_alt is not annotated_int_array
        assert annotated_int_array_alt.dtype is equivalent_requirement

        # Check the same logic for a pair of C types whose equivalence may vary
        # between computing environments.
        # Find an equivalent pair.
        integer_type_codes = ('i', 'l', 'q')
        integer_dtypes = [np.dtype(code) for code in integer_type_codes]
        typeA = None
        typeB = None
        for typeA, typeB in permutations(integer_dtypes, r=2):
            if typeA == typeB:
                assert typeA is not typeB
                break
        assert isinstance(typeA, np.dtype) and isinstance(typeB, np.dtype)

        # These ``asarray()`` calls may produce a new view or a copy,
        # but never the same object.
        long_int_array = np.asarray(int_array, dtype='l')
        long_long_int_array = np.asarray(int_array, dtype='q')
        assert long_int_array is not int_array
        assert long_long_int_array is not int_array
        assert np.asarray(long_int_array, dtype='q') is not long_int_array
        array_a = np.asarray(int_array, dtype=typeA)
        assert typeA == typeB
        assert typeA is not typeB
        assert array_a.dtype is typeA
        assert array_a is not np.asarray(array_a, dtype=typeB)
        assert np.asarray(array_a, dtype=typeB).dtype is typeB
        assert array_a is np.asarray(array_a, dtype=typeB).base


class TestSpecialAttributeLookupFailure:
    # An exception was raised while fetching the attribute

    class WeirdArrayLike:
        @property
        def __array__(self):
            raise RuntimeError("oops!")

    class WeirdArrayInterface:
        @property
        def __array_interface__(self):
            raise RuntimeError("oops!")

    def test_deprecated(self):
        with pytest.raises(RuntimeError):
            np.array(self.WeirdArrayLike())
        with pytest.raises(RuntimeError):
            np.array(self.WeirdArrayInterface())
