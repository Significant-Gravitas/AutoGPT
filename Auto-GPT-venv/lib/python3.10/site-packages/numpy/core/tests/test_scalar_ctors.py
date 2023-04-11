"""
Test the scalar constructors, which also do type-coercion
"""
import pytest

import numpy as np
from numpy.testing import (
    assert_equal, assert_almost_equal, assert_warns,
    )

class TestFromString:
    def test_floating(self):
        # Ticket #640, floats from string
        fsingle = np.single('1.234')
        fdouble = np.double('1.234')
        flongdouble = np.longdouble('1.234')
        assert_almost_equal(fsingle, 1.234)
        assert_almost_equal(fdouble, 1.234)
        assert_almost_equal(flongdouble, 1.234)

    def test_floating_overflow(self):
        """ Strings containing an unrepresentable float overflow """
        fhalf = np.half('1e10000')
        assert_equal(fhalf, np.inf)
        fsingle = np.single('1e10000')
        assert_equal(fsingle, np.inf)
        fdouble = np.double('1e10000')
        assert_equal(fdouble, np.inf)
        flongdouble = assert_warns(RuntimeWarning, np.longdouble, '1e10000')
        assert_equal(flongdouble, np.inf)

        fhalf = np.half('-1e10000')
        assert_equal(fhalf, -np.inf)
        fsingle = np.single('-1e10000')
        assert_equal(fsingle, -np.inf)
        fdouble = np.double('-1e10000')
        assert_equal(fdouble, -np.inf)
        flongdouble = assert_warns(RuntimeWarning, np.longdouble, '-1e10000')
        assert_equal(flongdouble, -np.inf)


class TestExtraArgs:
    def test_superclass(self):
        # try both positional and keyword arguments
        s = np.str_(b'\\x61', encoding='unicode-escape')
        assert s == 'a'
        s = np.str_(b'\\x61', 'unicode-escape')
        assert s == 'a'

        # previously this would return '\\xx'
        with pytest.raises(UnicodeDecodeError):
            np.str_(b'\\xx', encoding='unicode-escape')
        with pytest.raises(UnicodeDecodeError):
            np.str_(b'\\xx', 'unicode-escape')

        # superclass fails, but numpy succeeds
        assert np.bytes_(-2) == b'-2'

    def test_datetime(self):
        dt = np.datetime64('2000-01', ('M', 2))
        assert np.datetime_data(dt) == ('M', 2)

        with pytest.raises(TypeError):
            np.datetime64('2000', garbage=True)

    def test_bool(self):
        with pytest.raises(TypeError):
            np.bool_(False, garbage=True)

    def test_void(self):
        with pytest.raises(TypeError):
            np.void(b'test', garbage=True)


class TestFromInt:
    def test_intp(self):
        # Ticket #99
        assert_equal(1024, np.intp(1024))

    def test_uint64_from_negative(self):
        with pytest.warns(DeprecationWarning):
            assert_equal(np.uint64(-2), np.uint64(18446744073709551614))


int_types = [np.byte, np.short, np.intc, np.int_, np.longlong]
uint_types = [np.ubyte, np.ushort, np.uintc, np.uint, np.ulonglong]
float_types = [np.half, np.single, np.double, np.longdouble]
cfloat_types = [np.csingle, np.cdouble, np.clongdouble]


class TestArrayFromScalar:
    """ gh-15467 """

    def _do_test(self, t1, t2):
        x = t1(2)
        arr = np.array(x, dtype=t2)
        # type should be preserved exactly
        if t2 is None:
            assert arr.dtype.type is t1
        else:
            assert arr.dtype.type is t2

    @pytest.mark.parametrize('t1', int_types + uint_types)
    @pytest.mark.parametrize('t2', int_types + uint_types + [None])
    def test_integers(self, t1, t2):
        return self._do_test(t1, t2)

    @pytest.mark.parametrize('t1', float_types)
    @pytest.mark.parametrize('t2', float_types + [None])
    def test_reals(self, t1, t2):
        return self._do_test(t1, t2)

    @pytest.mark.parametrize('t1', cfloat_types)
    @pytest.mark.parametrize('t2', cfloat_types + [None])
    def test_complex(self, t1, t2):
        return self._do_test(t1, t2)


@pytest.mark.parametrize("length",
        [5, np.int8(5), np.array(5, dtype=np.uint16)])
def test_void_via_length(length):
    res = np.void(length)
    assert type(res) is np.void
    assert res.item() == b"\0" * 5
    assert res.dtype == "V5"

@pytest.mark.parametrize("bytes_",
        [b"spam", np.array(567.)])
def test_void_from_byteslike(bytes_):
    res = np.void(bytes_)
    expected = bytes(bytes_)
    assert type(res) is np.void
    assert res.item() == expected

    # Passing dtype can extend it (this is how filling works)
    res = np.void(bytes_, dtype="V100")
    assert type(res) is np.void
    assert res.item()[:len(expected)] == expected
    assert res.item()[len(expected):] == b"\0" * (res.nbytes - len(expected))
    # As well as shorten:
    res = np.void(bytes_, dtype="V4")
    assert type(res) is np.void
    assert res.item() == expected[:4]

def test_void_arraylike_trumps_byteslike():
    # The memoryview is converted as an array-like of shape (18,)
    # rather than a single bytes-like of that length.
    m = memoryview(b"just one mintleaf?")
    res = np.void(m)
    assert type(res) is np.ndarray
    assert res.dtype == "V1"
    assert res.shape == (18,)

def test_void_dtype_arg():
    # Basic test for the dtype argument (positional and keyword)
    res = np.void((1, 2), dtype="i,i")
    assert res.item() == (1, 2)
    res = np.void((2, 3), "i,i")
    assert res.item() == (2, 3)

@pytest.mark.parametrize("data",
        [5, np.int8(5), np.array(5, dtype=np.uint16)])
def test_void_from_integer_with_dtype(data):
    # The "length" meaning is ignored, rather data is used:
    res = np.void(data, dtype="i,i")
    assert type(res) is np.void
    assert res.dtype == "i,i"
    assert res["f0"] == 5 and res["f1"] == 5

def test_void_from_structure():
    dtype = np.dtype([('s', [('f', 'f8'), ('u', 'U1')]), ('i', 'i2')])
    data = np.array(((1., 'a'), 2), dtype=dtype)
    res = np.void(data[()], dtype=dtype)
    assert type(res) is np.void
    assert res.dtype == dtype
    assert res == data[()]

def test_void_bad_dtype():
    with pytest.raises(TypeError,
            match="void: descr must be a `void.*int64"):
        np.void(4, dtype="i8")

    # Subarray dtype (with shape `(4,)` is rejected):
    with pytest.raises(TypeError,
            match=r"void: descr must be a `void.*\(4,\)"):
        np.void(4, dtype="4i")
