"""
The tests exercise the casting machinery in a more low-level manner.
The reason is mostly to test a new implementation of the casting machinery.

Unlike most tests in NumPy, these are closer to unit-tests rather
than integration tests.
"""

import pytest
import textwrap
import enum
import random
import ctypes

import numpy as np
from numpy.lib.stride_tricks import as_strided

from numpy.testing import assert_array_equal
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl


# Simple skips object, parametric and long double (unsupported by struct)
simple_dtypes = "?bhilqBHILQefdFD"
if np.dtype("l").itemsize != np.dtype("q").itemsize:
    # Remove l and L, the table was generated with 64bit linux in mind.
    simple_dtypes = simple_dtypes.replace("l", "").replace("L", "")
simple_dtypes = [type(np.dtype(c)) for c in simple_dtypes]


def simple_dtype_instances():
    for dtype_class in simple_dtypes:
        dt = dtype_class()
        yield pytest.param(dt, id=str(dt))
        if dt.byteorder != "|":
            dt = dt.newbyteorder()
            yield pytest.param(dt, id=str(dt))


def get_expected_stringlength(dtype):
    """Returns the string length when casting the basic dtypes to strings.
    """
    if dtype == np.bool_:
        return 5
    if dtype.kind in "iu":
        if dtype.itemsize == 1:
            length = 3
        elif dtype.itemsize == 2:
            length = 5
        elif dtype.itemsize == 4:
            length = 10
        elif dtype.itemsize == 8:
            length = 20
        else:
            raise AssertionError(f"did not find expected length for {dtype}")

        if dtype.kind == "i":
            length += 1  # adds one character for the sign

        return length

    # Note: Can't do dtype comparison for longdouble on windows
    if dtype.char == "g":
        return 48
    elif dtype.char == "G":
        return 48 * 2
    elif dtype.kind == "f":
        return 32  # also for half apparently.
    elif dtype.kind == "c":
        return 32 * 2

    raise AssertionError(f"did not find expected length for {dtype}")


class Casting(enum.IntEnum):
    no = 0
    equiv = 1
    safe = 2
    same_kind = 3
    unsafe = 4


def _get_cancast_table():
    table = textwrap.dedent("""
        X ? b h i l q B H I L Q e f d g F D G S U V O M m
        ? # = = = = = = = = = = = = = = = = = = = = = . =
        b . # = = = = . . . . . = = = = = = = = = = = . =
        h . ~ # = = = . . . . . ~ = = = = = = = = = = . =
        i . ~ ~ # = = . . . . . ~ ~ = = ~ = = = = = = . =
        l . ~ ~ ~ # # . . . . . ~ ~ = = ~ = = = = = = . =
        q . ~ ~ ~ # # . . . . . ~ ~ = = ~ = = = = = = . =
        B . ~ = = = = # = = = = = = = = = = = = = = = . =
        H . ~ ~ = = = ~ # = = = ~ = = = = = = = = = = . =
        I . ~ ~ ~ = = ~ ~ # = = ~ ~ = = ~ = = = = = = . =
        L . ~ ~ ~ ~ ~ ~ ~ ~ # # ~ ~ = = ~ = = = = = = . ~
        Q . ~ ~ ~ ~ ~ ~ ~ ~ # # ~ ~ = = ~ = = = = = = . ~
        e . . . . . . . . . . . # = = = = = = = = = = . .
        f . . . . . . . . . . . ~ # = = = = = = = = = . .
        d . . . . . . . . . . . ~ ~ # = ~ = = = = = = . .
        g . . . . . . . . . . . ~ ~ ~ # ~ ~ = = = = = . .
        F . . . . . . . . . . . . . . . # = = = = = = . .
        D . . . . . . . . . . . . . . . ~ # = = = = = . .
        G . . . . . . . . . . . . . . . ~ ~ # = = = = . .
        S . . . . . . . . . . . . . . . . . . # = = = . .
        U . . . . . . . . . . . . . . . . . . . # = = . .
        V . . . . . . . . . . . . . . . . . . . . # = . .
        O . . . . . . . . . . . . . . . . . . . . = # . .
        M . . . . . . . . . . . . . . . . . . . . = = # .
        m . . . . . . . . . . . . . . . . . . . . = = . #
        """).strip().split("\n")
    dtypes = [type(np.dtype(c)) for c in table[0][2::2]]

    convert_cast = {".": Casting.unsafe, "~": Casting.same_kind,
                    "=": Casting.safe, "#": Casting.equiv,
                    " ": -1}

    cancast = {}
    for from_dt, row in zip(dtypes, table[1:]):
        cancast[from_dt] = {}
        for to_dt, c in zip(dtypes, row[2::2]):
            cancast[from_dt][to_dt] = convert_cast[c]

    return cancast

CAST_TABLE = _get_cancast_table()


class TestChanges:
    """
    These test cases exercise some behaviour changes
    """
    @pytest.mark.parametrize("string", ["S", "U"])
    @pytest.mark.parametrize("floating", ["e", "f", "d", "g"])
    def test_float_to_string(self, floating, string):
        assert np.can_cast(floating, string)
        # 100 is long enough to hold any formatted floating
        assert np.can_cast(floating, f"{string}100")

    def test_to_void(self):
        # But in general, we do consider these safe:
        assert np.can_cast("d", "V")
        assert np.can_cast("S20", "V")

        # Do not consider it a safe cast if the void is too smaller:
        assert not np.can_cast("d", "V1")
        assert not np.can_cast("S20", "V1")
        assert not np.can_cast("U1", "V1")
        # Structured to unstructured is just like any other:
        assert np.can_cast("d,i", "V", casting="same_kind")
        # Unstructured void to unstructured is actually no cast at all:
        assert np.can_cast("V3", "V", casting="no")
        assert np.can_cast("V0", "V", casting="no")


class TestCasting:
    size = 1500  # Best larger than NPY_LOWLEVEL_BUFFER_BLOCKSIZE * itemsize

    def get_data(self, dtype1, dtype2):
        if dtype2 is None or dtype1.itemsize >= dtype2.itemsize:
            length = self.size // dtype1.itemsize
        else:
            length = self.size // dtype2.itemsize

        # Assume that the base array is well enough aligned for all inputs.
        arr1 = np.empty(length, dtype=dtype1)
        assert arr1.flags.c_contiguous
        assert arr1.flags.aligned

        values = [random.randrange(-128, 128) for _ in range(length)]

        for i, value in enumerate(values):
            # Use item assignment to ensure this is not using casting:
            if value < 0 and dtype1.kind == "u":
                # Manually rollover unsigned integers (-1 -> int.max)
                value = value + np.iinfo(dtype1).max + 1
            arr1[i] = value

        if dtype2 is None:
            if dtype1.char == "?":
                values = [bool(v) for v in values]
            return arr1, values

        if dtype2.char == "?":
            values = [bool(v) for v in values]

        arr2 = np.empty(length, dtype=dtype2)
        assert arr2.flags.c_contiguous
        assert arr2.flags.aligned

        for i, value in enumerate(values):
            # Use item assignment to ensure this is not using casting:
            if value < 0 and dtype2.kind == "u":
                # Manually rollover unsigned integers (-1 -> int.max)
                value = value + np.iinfo(dtype2).max + 1
            arr2[i] = value

        return arr1, arr2, values

    def get_data_variation(self, arr1, arr2, aligned=True, contig=True):
        """
        Returns a copy of arr1 that may be non-contiguous or unaligned, and a
        matching array for arr2 (although not a copy).
        """
        if contig:
            stride1 = arr1.dtype.itemsize
            stride2 = arr2.dtype.itemsize
        elif aligned:
            stride1 = 2 * arr1.dtype.itemsize
            stride2 = 2 * arr2.dtype.itemsize
        else:
            stride1 = arr1.dtype.itemsize + 1
            stride2 = arr2.dtype.itemsize + 1

        max_size1 = len(arr1) * 3 * arr1.dtype.itemsize + 1
        max_size2 = len(arr2) * 3 * arr2.dtype.itemsize + 1
        from_bytes = np.zeros(max_size1, dtype=np.uint8)
        to_bytes = np.zeros(max_size2, dtype=np.uint8)

        # Sanity check that the above is large enough:
        assert stride1 * len(arr1) <= from_bytes.nbytes
        assert stride2 * len(arr2) <= to_bytes.nbytes

        if aligned:
            new1 = as_strided(from_bytes[:-1].view(arr1.dtype),
                              arr1.shape, (stride1,))
            new2 = as_strided(to_bytes[:-1].view(arr2.dtype),
                              arr2.shape, (stride2,))
        else:
            new1 = as_strided(from_bytes[1:].view(arr1.dtype),
                              arr1.shape, (stride1,))
            new2 = as_strided(to_bytes[1:].view(arr2.dtype),
                              arr2.shape, (stride2,))

        new1[...] = arr1

        if not contig:
            # Ensure we did not overwrite bytes that should not be written:
            offset = arr1.dtype.itemsize if aligned else 0
            buf = from_bytes[offset::stride1].tobytes()
            assert buf.count(b"\0") == len(buf)

        if contig:
            assert new1.flags.c_contiguous
            assert new2.flags.c_contiguous
        else:
            assert not new1.flags.c_contiguous
            assert not new2.flags.c_contiguous

        if aligned:
            assert new1.flags.aligned
            assert new2.flags.aligned
        else:
            assert not new1.flags.aligned or new1.dtype.alignment == 1
            assert not new2.flags.aligned or new2.dtype.alignment == 1

        return new1, new2

    @pytest.mark.parametrize("from_Dt", simple_dtypes)
    def test_simple_cancast(self, from_Dt):
        for to_Dt in simple_dtypes:
            cast = get_castingimpl(from_Dt, to_Dt)

            for from_dt in [from_Dt(), from_Dt().newbyteorder()]:
                default = cast._resolve_descriptors((from_dt, None))[1][1]
                assert default == to_Dt()
                del default

                for to_dt in [to_Dt(), to_Dt().newbyteorder()]:
                    casting, (from_res, to_res), view_off = (
                            cast._resolve_descriptors((from_dt, to_dt)))
                    assert(type(from_res) == from_Dt)
                    assert(type(to_res) == to_Dt)
                    if view_off is not None:
                        # If a view is acceptable, this is "no" casting
                        # and byte order must be matching.
                        assert casting == Casting.no
                        # The above table lists this as "equivalent"
                        assert Casting.equiv == CAST_TABLE[from_Dt][to_Dt]
                        # Note that to_res may not be the same as from_dt
                        assert from_res.isnative == to_res.isnative
                    else:
                        if from_Dt == to_Dt:
                            # Note that to_res may not be the same as from_dt
                            assert from_res.isnative != to_res.isnative
                        assert casting == CAST_TABLE[from_Dt][to_Dt]

                    if from_Dt is to_Dt:
                        assert(from_dt is from_res)
                        assert(to_dt is to_res)


    @pytest.mark.filterwarnings("ignore::numpy.ComplexWarning")
    @pytest.mark.parametrize("from_dt", simple_dtype_instances())
    def test_simple_direct_casts(self, from_dt):
        """
        This test checks numeric direct casts for dtypes supported also by the
        struct module (plus complex).  It tries to be test a wide range of
        inputs, but skips over possibly undefined behaviour (e.g. int rollover).
        Longdouble and CLongdouble are tested, but only using double precision.

        If this test creates issues, it should possibly just be simplified
        or even removed (checking whether unaligned/non-contiguous casts give
        the same results is useful, though).
        """
        for to_dt in simple_dtype_instances():
            to_dt = to_dt.values[0]
            cast = get_castingimpl(type(from_dt), type(to_dt))

            casting, (from_res, to_res), view_off = cast._resolve_descriptors(
                (from_dt, to_dt))

            if from_res is not from_dt or to_res is not to_dt:
                # Do not test this case, it is handled in multiple steps,
                # each of which should is tested individually.
                return

            safe = casting <= Casting.safe
            del from_res, to_res, casting

            arr1, arr2, values = self.get_data(from_dt, to_dt)

            cast._simple_strided_call((arr1, arr2))

            # Check via python list
            assert arr2.tolist() == values

            # Check that the same results are achieved for strided loops
            arr1_o, arr2_o = self.get_data_variation(arr1, arr2, True, False)
            cast._simple_strided_call((arr1_o, arr2_o))

            assert_array_equal(arr2_o, arr2)
            assert arr2_o.tobytes() == arr2.tobytes()

            # Check if alignment makes a difference, but only if supported
            # and only if the alignment can be wrong
            if ((from_dt.alignment == 1 and to_dt.alignment == 1) or
                    not cast._supports_unaligned):
                return

            arr1_o, arr2_o = self.get_data_variation(arr1, arr2, False, True)
            cast._simple_strided_call((arr1_o, arr2_o))

            assert_array_equal(arr2_o, arr2)
            assert arr2_o.tobytes() == arr2.tobytes()

            arr1_o, arr2_o = self.get_data_variation(arr1, arr2, False, False)
            cast._simple_strided_call((arr1_o, arr2_o))

            assert_array_equal(arr2_o, arr2)
            assert arr2_o.tobytes() == arr2.tobytes()

            del arr1_o, arr2_o, cast

    @pytest.mark.parametrize("from_Dt", simple_dtypes)
    def test_numeric_to_times(self, from_Dt):
        # We currently only implement contiguous loops, so only need to
        # test those.
        from_dt = from_Dt()

        time_dtypes = [np.dtype("M8"), np.dtype("M8[ms]"), np.dtype("M8[4D]"),
                       np.dtype("m8"), np.dtype("m8[ms]"), np.dtype("m8[4D]")]
        for time_dt in time_dtypes:
            cast = get_castingimpl(type(from_dt), type(time_dt))

            casting, (from_res, to_res), view_off = cast._resolve_descriptors(
                (from_dt, time_dt))

            assert from_res is from_dt
            assert to_res is time_dt
            del from_res, to_res

            assert casting & CAST_TABLE[from_Dt][type(time_dt)]
            assert view_off is None

            int64_dt = np.dtype(np.int64)
            arr1, arr2, values = self.get_data(from_dt, int64_dt)
            arr2 = arr2.view(time_dt)
            arr2[...] = np.datetime64("NaT")

            if time_dt == np.dtype("M8"):
                # This is a bit of a strange path, and could probably be removed
                arr1[-1] = 0  # ensure at least one value is not NaT

                # The cast currently succeeds, but the values are invalid:
                cast._simple_strided_call((arr1, arr2))
                with pytest.raises(ValueError):
                    str(arr2[-1])  # e.g. conversion to string fails
                return

            cast._simple_strided_call((arr1, arr2))

            assert [int(v) for v in arr2.tolist()] == values

            # Check that the same results are achieved for strided loops
            arr1_o, arr2_o = self.get_data_variation(arr1, arr2, True, False)
            cast._simple_strided_call((arr1_o, arr2_o))

            assert_array_equal(arr2_o, arr2)
            assert arr2_o.tobytes() == arr2.tobytes()

    @pytest.mark.parametrize(
            ["from_dt", "to_dt", "expected_casting", "expected_view_off",
             "nom", "denom"],
            [("M8[ns]", None, Casting.no, 0, 1, 1),
             (str(np.dtype("M8[ns]").newbyteorder()), None,
                  Casting.equiv, None, 1, 1),
             ("M8", "M8[ms]", Casting.safe, 0, 1, 1),
             # should be invalid cast:
             ("M8[ms]", "M8", Casting.unsafe, None, 1, 1),
             ("M8[5ms]", "M8[5ms]", Casting.no, 0, 1, 1),
             ("M8[ns]", "M8[ms]", Casting.same_kind, None, 1, 10**6),
             ("M8[ms]", "M8[ns]", Casting.safe, None, 10**6, 1),
             ("M8[ms]", "M8[7ms]", Casting.same_kind, None, 1, 7),
             ("M8[4D]", "M8[1M]", Casting.same_kind, None, None,
                  # give full values based on NumPy 1.19.x
                  [-2**63, 0, -1, 1314, -1315, 564442610]),
             ("m8[ns]", None, Casting.no, 0, 1, 1),
             (str(np.dtype("m8[ns]").newbyteorder()), None,
                  Casting.equiv, None, 1, 1),
             ("m8", "m8[ms]", Casting.safe, 0, 1, 1),
             # should be invalid cast:
             ("m8[ms]", "m8", Casting.unsafe, None, 1, 1),
             ("m8[5ms]", "m8[5ms]", Casting.no, 0, 1, 1),
             ("m8[ns]", "m8[ms]", Casting.same_kind, None, 1, 10**6),
             ("m8[ms]", "m8[ns]", Casting.safe, None, 10**6, 1),
             ("m8[ms]", "m8[7ms]", Casting.same_kind, None, 1, 7),
             ("m8[4D]", "m8[1M]", Casting.unsafe, None, None,
                  # give full values based on NumPy 1.19.x
                  [-2**63, 0, 0, 1314, -1315, 564442610])])
    def test_time_to_time(self, from_dt, to_dt,
                          expected_casting, expected_view_off,
                          nom, denom):
        from_dt = np.dtype(from_dt)
        if to_dt is not None:
            to_dt = np.dtype(to_dt)

        # Test a few values for casting (results generated with NumPy 1.19)
        values = np.array([-2**63, 1, 2**63-1, 10000, -10000, 2**32])
        values = values.astype(np.dtype("int64").newbyteorder(from_dt.byteorder))
        assert values.dtype.byteorder == from_dt.byteorder
        assert np.isnat(values.view(from_dt)[0])

        DType = type(from_dt)
        cast = get_castingimpl(DType, DType)
        casting, (from_res, to_res), view_off = cast._resolve_descriptors(
                (from_dt, to_dt))
        assert from_res is from_dt
        assert to_res is to_dt or to_dt is None
        assert casting == expected_casting
        assert view_off == expected_view_off

        if nom is not None:
            expected_out = (values * nom // denom).view(to_res)
            expected_out[0] = "NaT"
        else:
            expected_out = np.empty_like(values)
            expected_out[...] = denom
            expected_out = expected_out.view(to_dt)

        orig_arr = values.view(from_dt)
        orig_out = np.empty_like(expected_out)

        if casting == Casting.unsafe and (to_dt == "m8" or to_dt == "M8"):
            # Casting from non-generic to generic units is an error and should
            # probably be reported as an invalid cast earlier.
            with pytest.raises(ValueError):
                cast._simple_strided_call((orig_arr, orig_out))
            return

        for aligned in [True, True]:
            for contig in [True, True]:
                arr, out = self.get_data_variation(
                        orig_arr, orig_out, aligned, contig)
                out[...] = 0
                cast._simple_strided_call((arr, out))
                assert_array_equal(out.view("int64"), expected_out.view("int64"))

    def string_with_modified_length(self, dtype, change_length):
        fact = 1 if dtype.char == "S" else 4
        length = dtype.itemsize // fact + change_length
        return np.dtype(f"{dtype.byteorder}{dtype.char}{length}")

    @pytest.mark.parametrize("other_DT", simple_dtypes)
    @pytest.mark.parametrize("string_char", ["S", "U"])
    def test_string_cancast(self, other_DT, string_char):
        fact = 1 if string_char == "S" else 4

        string_DT = type(np.dtype(string_char))
        cast = get_castingimpl(other_DT, string_DT)

        other_dt = other_DT()
        expected_length = get_expected_stringlength(other_dt)
        string_dt = np.dtype(f"{string_char}{expected_length}")

        safety, (res_other_dt, res_dt), view_off = cast._resolve_descriptors(
                (other_dt, None))
        assert res_dt.itemsize == expected_length * fact
        assert safety == Casting.safe  # we consider to string casts "safe"
        assert view_off is None
        assert isinstance(res_dt, string_DT)

        # These casts currently implement changing the string length, so
        # check the cast-safety for too long/fixed string lengths:
        for change_length in [-1, 0, 1]:
            if change_length >= 0:
                expected_safety = Casting.safe
            else:
                expected_safety = Casting.same_kind

            to_dt = self.string_with_modified_length(string_dt, change_length)
            safety, (_, res_dt), view_off = cast._resolve_descriptors(
                    (other_dt, to_dt))
            assert res_dt is to_dt
            assert safety == expected_safety
            assert view_off is None

        # The opposite direction is always considered unsafe:
        cast = get_castingimpl(string_DT, other_DT)

        safety, _, view_off = cast._resolve_descriptors((string_dt, other_dt))
        assert safety == Casting.unsafe
        assert view_off is None

        cast = get_castingimpl(string_DT, other_DT)
        safety, (_, res_dt), view_off = cast._resolve_descriptors(
            (string_dt, None))
        assert safety == Casting.unsafe
        assert view_off is None
        assert other_dt is res_dt  # returns the singleton for simple dtypes

    @pytest.mark.parametrize("string_char", ["S", "U"])
    @pytest.mark.parametrize("other_dt", simple_dtype_instances())
    def test_simple_string_casts_roundtrip(self, other_dt, string_char):
        """
        Tests casts from and to string by checking the roundtripping property.

        The test also covers some string to string casts (but not all).

        If this test creates issues, it should possibly just be simplified
        or even removed (checking whether unaligned/non-contiguous casts give
        the same results is useful, though).
        """
        string_DT = type(np.dtype(string_char))

        cast = get_castingimpl(type(other_dt), string_DT)
        cast_back = get_castingimpl(string_DT, type(other_dt))
        _, (res_other_dt, string_dt), _ = cast._resolve_descriptors(
                (other_dt, None))

        if res_other_dt is not other_dt:
            # do not support non-native byteorder, skip test in that case
            assert other_dt.byteorder != res_other_dt.byteorder
            return

        orig_arr, values = self.get_data(other_dt, None)
        str_arr = np.zeros(len(orig_arr), dtype=string_dt)
        string_dt_short = self.string_with_modified_length(string_dt, -1)
        str_arr_short = np.zeros(len(orig_arr), dtype=string_dt_short)
        string_dt_long = self.string_with_modified_length(string_dt, 1)
        str_arr_long = np.zeros(len(orig_arr), dtype=string_dt_long)

        assert not cast._supports_unaligned  # if support is added, should test
        assert not cast_back._supports_unaligned

        for contig in [True, False]:
            other_arr, str_arr = self.get_data_variation(
                orig_arr, str_arr, True, contig)
            _, str_arr_short = self.get_data_variation(
                orig_arr, str_arr_short.copy(), True, contig)
            _, str_arr_long = self.get_data_variation(
                orig_arr, str_arr_long, True, contig)

            cast._simple_strided_call((other_arr, str_arr))

            cast._simple_strided_call((other_arr, str_arr_short))
            assert_array_equal(str_arr.astype(string_dt_short), str_arr_short)

            cast._simple_strided_call((other_arr, str_arr_long))
            assert_array_equal(str_arr, str_arr_long)

            if other_dt.kind == "b":
                # Booleans do not roundtrip
                continue

            other_arr[...] = 0
            cast_back._simple_strided_call((str_arr, other_arr))
            assert_array_equal(orig_arr, other_arr)

            other_arr[...] = 0
            cast_back._simple_strided_call((str_arr_long, other_arr))
            assert_array_equal(orig_arr, other_arr)

    @pytest.mark.parametrize("other_dt", ["S8", "<U8", ">U8"])
    @pytest.mark.parametrize("string_char", ["S", "U"])
    def test_string_to_string_cancast(self, other_dt, string_char):
        other_dt = np.dtype(other_dt)

        fact = 1 if string_char == "S" else 4
        div = 1 if other_dt.char == "S" else 4

        string_DT = type(np.dtype(string_char))
        cast = get_castingimpl(type(other_dt), string_DT)

        expected_length = other_dt.itemsize // div
        string_dt = np.dtype(f"{string_char}{expected_length}")

        safety, (res_other_dt, res_dt), view_off = cast._resolve_descriptors(
                (other_dt, None))
        assert res_dt.itemsize == expected_length * fact
        assert isinstance(res_dt, string_DT)

        expected_view_off = None
        if other_dt.char == string_char:
            if other_dt.isnative:
                expected_safety = Casting.no
                expected_view_off = 0
            else:
                expected_safety = Casting.equiv
        elif string_char == "U":
            expected_safety = Casting.safe
        else:
            expected_safety = Casting.unsafe

        assert view_off == expected_view_off
        assert expected_safety == safety

        for change_length in [-1, 0, 1]:
            to_dt = self.string_with_modified_length(string_dt, change_length)
            safety, (_, res_dt), view_off = cast._resolve_descriptors(
                    (other_dt, to_dt))

            assert res_dt is to_dt
            if change_length <= 0:
                assert view_off == expected_view_off
            else:
                assert view_off is None
            if expected_safety == Casting.unsafe:
                assert safety == expected_safety
            elif change_length < 0:
                assert safety == Casting.same_kind
            elif change_length == 0:
                assert safety == expected_safety
            elif change_length > 0:
                assert safety == Casting.safe

    @pytest.mark.parametrize("order1", [">", "<"])
    @pytest.mark.parametrize("order2", [">", "<"])
    def test_unicode_byteswapped_cast(self, order1, order2):
        # Very specific tests (not using the castingimpl directly)
        # that tests unicode bytedwaps including for unaligned array data.
        dtype1 = np.dtype(f"{order1}U30")
        dtype2 = np.dtype(f"{order2}U30")
        data1 = np.empty(30 * 4 + 1, dtype=np.uint8)[1:].view(dtype1)
        data2 = np.empty(30 * 4 + 1, dtype=np.uint8)[1:].view(dtype2)
        if dtype1.alignment != 1:
            # alignment should always be >1, but skip the check if not
            assert not data1.flags.aligned
            assert not data2.flags.aligned

        element = "this is a ünicode string‽"
        data1[()] = element
        # Test both `data1` and `data1.copy()`  (which should be aligned)
        for data in [data1, data1.copy()]:
            data2[...] = data1
            assert data2[()] == element
            assert data2.copy()[()] == element

    def test_void_to_string_special_case(self):
        # Cover a small special case in void to string casting that could
        # probably just as well be turned into an error (compare
        # `test_object_to_parametric_internal_error` below).
        assert np.array([], dtype="V5").astype("S").dtype.itemsize == 5
        assert np.array([], dtype="V5").astype("U").dtype.itemsize == 4 * 5

    def test_object_to_parametric_internal_error(self):
        # We reject casting from object to a parametric type, without
        # figuring out the correct instance first.
        object_dtype = type(np.dtype(object))
        other_dtype = type(np.dtype(str))
        cast = get_castingimpl(object_dtype, other_dtype)
        with pytest.raises(TypeError,
                    match="casting from object to the parametric DType"):
            cast._resolve_descriptors((np.dtype("O"), None))

    @pytest.mark.parametrize("dtype", simple_dtype_instances())
    def test_object_and_simple_resolution(self, dtype):
        # Simple test to exercise the cast when no instance is specified
        object_dtype = type(np.dtype(object))
        cast = get_castingimpl(object_dtype, type(dtype))

        safety, (_, res_dt), view_off = cast._resolve_descriptors(
                (np.dtype("O"), dtype))
        assert safety == Casting.unsafe
        assert view_off is None
        assert res_dt is dtype

        safety, (_, res_dt), view_off = cast._resolve_descriptors(
                (np.dtype("O"), None))
        assert safety == Casting.unsafe
        assert view_off is None
        assert res_dt == dtype.newbyteorder("=")

    @pytest.mark.parametrize("dtype", simple_dtype_instances())
    def test_simple_to_object_resolution(self, dtype):
        # Simple test to exercise the cast when no instance is specified
        object_dtype = type(np.dtype(object))
        cast = get_castingimpl(type(dtype), object_dtype)

        safety, (_, res_dt), view_off = cast._resolve_descriptors(
                (dtype, None))
        assert safety == Casting.safe
        assert view_off is None
        assert res_dt is np.dtype("O")

    @pytest.mark.parametrize("casting", ["no", "unsafe"])
    def test_void_and_structured_with_subarray(self, casting):
        # test case corresponding to gh-19325
        dtype = np.dtype([("foo", "<f4", (3, 2))])
        expected = casting == "unsafe"
        assert np.can_cast("V4", dtype, casting=casting) == expected
        assert np.can_cast(dtype, "V4", casting=casting) == expected

    @pytest.mark.parametrize(["to_dt", "expected_off"],
            [  # Same as `from_dt` but with both fields shifted:
             (np.dtype({"names": ["a", "b"], "formats": ["i4", "f4"],
                        "offsets": [0, 4]}), 2),
             # Additional change of the names
             (np.dtype({"names": ["b", "a"], "formats": ["i4", "f4"],
                        "offsets": [0, 4]}), 2),
             # Incompatible field offset change
             (np.dtype({"names": ["b", "a"], "formats": ["i4", "f4"],
                        "offsets": [0, 6]}), None)])
    def test_structured_field_offsets(self, to_dt, expected_off):
        # This checks the cast-safety and view offset for swapped and "shifted"
        # fields which are viewable
        from_dt = np.dtype({"names": ["a", "b"],
                            "formats": ["i4", "f4"],
                            "offsets": [2, 6]})
        cast = get_castingimpl(type(from_dt), type(to_dt))
        safety, _, view_off = cast._resolve_descriptors((from_dt, to_dt))
        if from_dt.names == to_dt.names:
            assert safety == Casting.equiv
        else:
            assert safety == Casting.safe
        # Shifting the original data pointer by -2 will align both by
        # effectively adding 2 bytes of spacing before `from_dt`.
        assert view_off == expected_off

    @pytest.mark.parametrize(("from_dt", "to_dt", "expected_off"), [
            # Subarray cases:
            ("i", "(1,1)i", 0),
            ("(1,1)i", "i", 0),
            ("(2,1)i", "(2,1)i", 0),
            # field cases (field to field is tested explicitly also):
            # Not considered viewable, because a negative offset would allow
            # may structured dtype to indirectly access invalid memory.
            ("i", dict(names=["a"], formats=["i"], offsets=[2]), None),
            (dict(names=["a"], formats=["i"], offsets=[2]), "i", 2),
            # Currently considered not viewable, due to multiple fields
            # even though they overlap (maybe we should not allow that?)
            ("i", dict(names=["a", "b"], formats=["i", "i"], offsets=[2, 2]),
             None),
            # different number of fields can't work, should probably just fail
            # so it never reports "viewable":
            ("i,i", "i,i,i", None),
            # Unstructured void cases:
            ("i4", "V3", 0),  # void smaller or equal
            ("i4", "V4", 0),  # void smaller or equal
            ("i4", "V10", None),  # void is larger (no view)
            ("O", "V4", None),  # currently reject objects for view here.
            ("O", "V8", None),  # currently reject objects for view here.
            ("V4", "V3", 0),
            ("V4", "V4", 0),
            ("V3", "V4", None),
            # Note that currently void-to-other cast goes via byte-strings
            # and is not a "view" based cast like the opposite direction:
            ("V4", "i4", None),
            # completely invalid/impossible cast:
            ("i,i", "i,i,i", None),
        ])
    def test_structured_view_offsets_paramteric(
            self, from_dt, to_dt, expected_off):
        # TODO: While this test is fairly thorough, right now, it does not
        # really test some paths that may have nonzero offsets (they don't
        # really exists).
        from_dt = np.dtype(from_dt)
        to_dt = np.dtype(to_dt)
        cast = get_castingimpl(type(from_dt), type(to_dt))
        _, _, view_off = cast._resolve_descriptors((from_dt, to_dt))
        assert view_off == expected_off

    @pytest.mark.parametrize("dtype", np.typecodes["All"])
    def test_object_casts_NULL_None_equivalence(self, dtype):
        # None to <other> casts may succeed or fail, but a NULL'ed array must
        # behave the same as one filled with None's.
        arr_normal = np.array([None] * 5)
        arr_NULLs = np.empty_like(arr_normal)
        ctypes.memset(arr_NULLs.ctypes.data, 0, arr_NULLs.nbytes)
        # If the check fails (maybe it should) the test would lose its purpose:
        assert arr_NULLs.tobytes() == b"\x00" * arr_NULLs.nbytes

        try:
            expected = arr_normal.astype(dtype)
        except TypeError:
            with pytest.raises(TypeError):
                arr_NULLs.astype(dtype),
        else:
            assert_array_equal(expected, arr_NULLs.astype(dtype))

    @pytest.mark.parametrize("dtype",
            np.typecodes["AllInteger"] + np.typecodes["AllFloat"])
    def test_nonstandard_bool_to_other(self, dtype):
        # simple test for casting bool_ to numeric types, which should not
        # expose the detail that NumPy bools can sometimes take values other
        # than 0 and 1.  See also gh-19514.
        nonstandard_bools = np.array([0, 3, -7], dtype=np.int8).view(bool)
        res = nonstandard_bools.astype(dtype)
        expected = [0, 1, 1]
        assert_array_equal(res, expected)

