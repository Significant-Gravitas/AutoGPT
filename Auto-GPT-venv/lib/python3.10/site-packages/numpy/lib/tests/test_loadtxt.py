"""
Tests specific to `np.loadtxt` added during the move of loadtxt to be backed
by C code.
These tests complement those found in `test_io.py`.
"""

import sys
import os
import pytest
from tempfile import NamedTemporaryFile, mkstemp
from io import StringIO

import numpy as np
from numpy.ma.testutils import assert_equal
from numpy.testing import assert_array_equal, HAS_REFCOUNT, IS_PYPY


def test_scientific_notation():
    """Test that both 'e' and 'E' are parsed correctly."""
    data = StringIO(
        (
            "1.0e-1,2.0E1,3.0\n"
            "4.0e-2,5.0E-1,6.0\n"
            "7.0e-3,8.0E1,9.0\n"
            "0.0e-4,1.0E-1,2.0"
        )
    )
    expected = np.array(
        [[0.1, 20., 3.0], [0.04, 0.5, 6], [0.007, 80., 9], [0, 0.1, 2]]
    )
    assert_array_equal(np.loadtxt(data, delimiter=","), expected)


@pytest.mark.parametrize("comment", ["..", "//", "@-", "this is a comment:"])
def test_comment_multiple_chars(comment):
    content = "# IGNORE\n1.5, 2.5# ABC\n3.0,4.0# XXX\n5.5,6.0\n"
    txt = StringIO(content.replace("#", comment))
    a = np.loadtxt(txt, delimiter=",", comments=comment)
    assert_equal(a, [[1.5, 2.5], [3.0, 4.0], [5.5, 6.0]])


@pytest.fixture
def mixed_types_structured():
    """
    Fixture providing hetergeneous input data with a structured dtype, along
    with the associated structured array.
    """
    data = StringIO(
        (
            "1000;2.4;alpha;-34\n"
            "2000;3.1;beta;29\n"
            "3500;9.9;gamma;120\n"
            "4090;8.1;delta;0\n"
            "5001;4.4;epsilon;-99\n"
            "6543;7.8;omega;-1\n"
        )
    )
    dtype = np.dtype(
        [('f0', np.uint16), ('f1', np.float64), ('f2', 'S7'), ('f3', np.int8)]
    )
    expected = np.array(
        [
            (1000, 2.4, "alpha", -34),
            (2000, 3.1, "beta", 29),
            (3500, 9.9, "gamma", 120),
            (4090, 8.1, "delta", 0),
            (5001, 4.4, "epsilon", -99),
            (6543, 7.8, "omega", -1)
        ],
        dtype=dtype
    )
    return data, dtype, expected


@pytest.mark.parametrize('skiprows', [0, 1, 2, 3])
def test_structured_dtype_and_skiprows_no_empty_lines(
        skiprows, mixed_types_structured):
    data, dtype, expected = mixed_types_structured
    a = np.loadtxt(data, dtype=dtype, delimiter=";", skiprows=skiprows)
    assert_array_equal(a, expected[skiprows:])


def test_unpack_structured(mixed_types_structured):
    data, dtype, expected = mixed_types_structured

    a, b, c, d = np.loadtxt(data, dtype=dtype, delimiter=";", unpack=True)
    assert_array_equal(a, expected["f0"])
    assert_array_equal(b, expected["f1"])
    assert_array_equal(c, expected["f2"])
    assert_array_equal(d, expected["f3"])


def test_structured_dtype_with_shape():
    dtype = np.dtype([("a", "u1", 2), ("b", "u1", 2)])
    data = StringIO("0,1,2,3\n6,7,8,9\n")
    expected = np.array([((0, 1), (2, 3)), ((6, 7), (8, 9))], dtype=dtype)
    assert_array_equal(np.loadtxt(data, delimiter=",", dtype=dtype), expected)


def test_structured_dtype_with_multi_shape():
    dtype = np.dtype([("a", "u1", (2, 2))])
    data = StringIO("0 1 2 3\n")
    expected = np.array([(((0, 1), (2, 3)),)], dtype=dtype)
    assert_array_equal(np.loadtxt(data, dtype=dtype), expected)


def test_nested_structured_subarray():
    # Test from gh-16678
    point = np.dtype([('x', float), ('y', float)])
    dt = np.dtype([('code', int), ('points', point, (2,))])
    data = StringIO("100,1,2,3,4\n200,5,6,7,8\n")
    expected = np.array(
        [
            (100, [(1., 2.), (3., 4.)]),
            (200, [(5., 6.), (7., 8.)]),
        ],
        dtype=dt
    )
    assert_array_equal(np.loadtxt(data, dtype=dt, delimiter=","), expected)


def test_structured_dtype_offsets():
    # An aligned structured dtype will have additional padding
    dt = np.dtype("i1, i4, i1, i4, i1, i4", align=True)
    data = StringIO("1,2,3,4,5,6\n7,8,9,10,11,12\n")
    expected = np.array([(1, 2, 3, 4, 5, 6), (7, 8, 9, 10, 11, 12)], dtype=dt)
    assert_array_equal(np.loadtxt(data, delimiter=",", dtype=dt), expected)


@pytest.mark.parametrize("param", ("skiprows", "max_rows"))
def test_exception_negative_row_limits(param):
    """skiprows and max_rows should raise for negative parameters."""
    with pytest.raises(ValueError, match="argument must be nonnegative"):
        np.loadtxt("foo.bar", **{param: -3})


@pytest.mark.parametrize("param", ("skiprows", "max_rows"))
def test_exception_noninteger_row_limits(param):
    with pytest.raises(TypeError, match="argument must be an integer"):
        np.loadtxt("foo.bar", **{param: 1.0})


@pytest.mark.parametrize(
    "data, shape",
    [
        ("1 2 3 4 5\n", (1, 5)),  # Single row
        ("1\n2\n3\n4\n5\n", (5, 1)),  # Single column
    ]
)
def test_ndmin_single_row_or_col(data, shape):
    arr = np.array([1, 2, 3, 4, 5])
    arr2d = arr.reshape(shape)

    assert_array_equal(np.loadtxt(StringIO(data), dtype=int), arr)
    assert_array_equal(np.loadtxt(StringIO(data), dtype=int, ndmin=0), arr)
    assert_array_equal(np.loadtxt(StringIO(data), dtype=int, ndmin=1), arr)
    assert_array_equal(np.loadtxt(StringIO(data), dtype=int, ndmin=2), arr2d)


@pytest.mark.parametrize("badval", [-1, 3, None, "plate of shrimp"])
def test_bad_ndmin(badval):
    with pytest.raises(ValueError, match="Illegal value of ndmin keyword"):
        np.loadtxt("foo.bar", ndmin=badval)


@pytest.mark.parametrize(
    "ws",
    (
            " ",  # space
            "\t",  # tab
            "\u2003",  # em
            "\u00A0",  # non-break
            "\u3000",  # ideographic space
    )
)
def test_blank_lines_spaces_delimit(ws):
    txt = StringIO(
        f"1 2{ws}30\n\n{ws}\n"
        f"4 5 60{ws}\n  {ws}  \n"
        f"7 8 {ws} 90\n  # comment\n"
        f"3 2 1"
    )
    # NOTE: It is unclear that the `  # comment` should succeed. Except
    #       for delimiter=None, which should use any whitespace (and maybe
    #       should just be implemented closer to Python
    expected = np.array([[1, 2, 30], [4, 5, 60], [7, 8, 90], [3, 2, 1]])
    assert_equal(
        np.loadtxt(txt, dtype=int, delimiter=None, comments="#"), expected
    )


def test_blank_lines_normal_delimiter():
    txt = StringIO('1,2,30\n\n4,5,60\n\n7,8,90\n# comment\n3,2,1')
    expected = np.array([[1, 2, 30], [4, 5, 60], [7, 8, 90], [3, 2, 1]])
    assert_equal(
        np.loadtxt(txt, dtype=int, delimiter=',', comments="#"), expected
    )


@pytest.mark.parametrize("dtype", (float, object))
def test_maxrows_no_blank_lines(dtype):
    txt = StringIO("1.5,2.5\n3.0,4.0\n5.5,6.0")
    res = np.loadtxt(txt, dtype=dtype, delimiter=",", max_rows=2)
    assert_equal(res.dtype, dtype)
    assert_equal(res, np.array([["1.5", "2.5"], ["3.0", "4.0"]], dtype=dtype))


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype", (np.dtype("f8"), np.dtype("i2")))
def test_exception_message_bad_values(dtype):
    txt = StringIO("1,2\n3,XXX\n5,6")
    msg = f"could not convert string 'XXX' to {dtype} at row 1, column 2"
    with pytest.raises(ValueError, match=msg):
        np.loadtxt(txt, dtype=dtype, delimiter=",")


def test_converters_negative_indices():
    txt = StringIO('1.5,2.5\n3.0,XXX\n5.5,6.0')
    conv = {-1: lambda s: np.nan if s == 'XXX' else float(s)}
    expected = np.array([[1.5, 2.5], [3.0, np.nan], [5.5, 6.0]])
    res = np.loadtxt(
        txt, dtype=np.float64, delimiter=",", converters=conv, encoding=None
    )
    assert_equal(res, expected)


def test_converters_negative_indices_with_usecols():
    txt = StringIO('1.5,2.5,3.5\n3.0,4.0,XXX\n5.5,6.0,7.5\n')
    conv = {-1: lambda s: np.nan if s == 'XXX' else float(s)}
    expected = np.array([[1.5, 3.5], [3.0, np.nan], [5.5, 7.5]])
    res = np.loadtxt(
        txt,
        dtype=np.float64,
        delimiter=",",
        converters=conv,
        usecols=[0, -1],
        encoding=None,
    )
    assert_equal(res, expected)

    # Second test with variable number of rows:
    res = np.loadtxt(StringIO('''0,1,2\n0,1,2,3,4'''), delimiter=",",
                     usecols=[0, -1], converters={-1: (lambda x: -1)})
    assert_array_equal(res, [[0, -1], [0, -1]])

def test_ragged_usecols():
    # usecols, and negative ones, work even with varying number of columns.
    txt = StringIO("0,0,XXX\n0,XXX,0,XXX\n0,XXX,XXX,0,XXX\n")
    expected = np.array([[0, 0], [0, 0], [0, 0]])
    res = np.loadtxt(txt, dtype=float, delimiter=",", usecols=[0, -2])
    assert_equal(res, expected)

    txt = StringIO("0,0,XXX\n0\n0,XXX,XXX,0,XXX\n")
    with pytest.raises(ValueError,
                match="invalid column index -2 at row 2 with 1 columns"):
        # There is no -2 column in the second row:
        np.loadtxt(txt, dtype=float, delimiter=",", usecols=[0, -2])


def test_empty_usecols():
    txt = StringIO("0,0,XXX\n0,XXX,0,XXX\n0,XXX,XXX,0,XXX\n")
    res = np.loadtxt(txt, dtype=np.dtype([]), delimiter=",", usecols=[])
    assert res.shape == (3,)
    assert res.dtype == np.dtype([])


@pytest.mark.parametrize("c1", ["a", "の", "🫕"])
@pytest.mark.parametrize("c2", ["a", "の", "🫕"])
def test_large_unicode_characters(c1, c2):
    # c1 and c2 span ascii, 16bit and 32bit range.
    txt = StringIO(f"a,{c1},c,1.0\ne,{c2},2.0,g")
    res = np.loadtxt(txt, dtype=np.dtype('U12'), delimiter=",")
    expected = np.array(
        [f"a,{c1},c,1.0".split(","), f"e,{c2},2.0,g".split(",")],
        dtype=np.dtype('U12')
    )
    assert_equal(res, expected)


def test_unicode_with_converter():
    txt = StringIO("cat,dog\nαβγ,δεζ\nabc,def\n")
    conv = {0: lambda s: s.upper()}
    res = np.loadtxt(
        txt,
        dtype=np.dtype("U12"),
        converters=conv,
        delimiter=",",
        encoding=None
    )
    expected = np.array([['CAT', 'dog'], ['ΑΒΓ', 'δεζ'], ['ABC', 'def']])
    assert_equal(res, expected)


def test_converter_with_structured_dtype():
    txt = StringIO('1.5,2.5,Abc\n3.0,4.0,dEf\n5.5,6.0,ghI\n')
    dt = np.dtype([('m', np.int32), ('r', np.float32), ('code', 'U8')])
    conv = {0: lambda s: int(10*float(s)), -1: lambda s: s.upper()}
    res = np.loadtxt(txt, dtype=dt, delimiter=",", converters=conv)
    expected = np.array(
        [(15, 2.5, 'ABC'), (30, 4.0, 'DEF'), (55, 6.0, 'GHI')], dtype=dt
    )
    assert_equal(res, expected)


def test_converter_with_unicode_dtype():
    """
    With the default 'bytes' encoding, tokens are encoded prior to being
    passed to the converter. This means that the output of the converter may
    be bytes instead of unicode as expected by `read_rows`.

    This test checks that outputs from the above scenario are properly decoded
    prior to parsing by `read_rows`.
    """
    txt = StringIO('abc,def\nrst,xyz')
    conv = bytes.upper
    res = np.loadtxt(
            txt, dtype=np.dtype("U3"), converters=conv, delimiter=",")
    expected = np.array([['ABC', 'DEF'], ['RST', 'XYZ']])
    assert_equal(res, expected)


def test_read_huge_row():
    row = "1.5, 2.5," * 50000
    row = row[:-1] + "\n"
    txt = StringIO(row * 2)
    res = np.loadtxt(txt, delimiter=",", dtype=float)
    assert_equal(res, np.tile([1.5, 2.5], (2, 50000)))


@pytest.mark.parametrize("dtype", "edfgFDG")
def test_huge_float(dtype):
    # Covers a non-optimized path that is rarely taken:
    field = "0" * 1000 + ".123456789"
    dtype = np.dtype(dtype)
    value = np.loadtxt([field], dtype=dtype)[()]
    assert value == dtype.type("0.123456789")


@pytest.mark.parametrize(
    ("given_dtype", "expected_dtype"),
    [
        ("S", np.dtype("S5")),
        ("U", np.dtype("U5")),
    ],
)
def test_string_no_length_given(given_dtype, expected_dtype):
    """
    The given dtype is just 'S' or 'U' with no length. In these cases, the
    length of the resulting dtype is determined by the longest string found
    in the file.
    """
    txt = StringIO("AAA,5-1\nBBBBB,0-3\nC,4-9\n")
    res = np.loadtxt(txt, dtype=given_dtype, delimiter=",")
    expected = np.array(
        [['AAA', '5-1'], ['BBBBB', '0-3'], ['C', '4-9']], dtype=expected_dtype
    )
    assert_equal(res, expected)
    assert_equal(res.dtype, expected_dtype)


def test_float_conversion():
    """
    Some tests that the conversion to float64 works as accurately as the
    Python built-in `float` function. In a naive version of the float parser,
    these strings resulted in values that were off by an ULP or two.
    """
    strings = [
        '0.9999999999999999',
        '9876543210.123456',
        '5.43215432154321e+300',
        '0.901',
        '0.333',
    ]
    txt = StringIO('\n'.join(strings))
    res = np.loadtxt(txt)
    expected = np.array([float(s) for s in strings])
    assert_equal(res, expected)


def test_bool():
    # Simple test for bool via integer
    txt = StringIO("1, 0\n10, -1")
    res = np.loadtxt(txt, dtype=bool, delimiter=",")
    assert res.dtype == bool
    assert_array_equal(res, [[True, False], [True, True]])
    # Make sure we use only 1 and 0 on the byte level:
    assert_array_equal(res.view(np.uint8), [[1, 0], [1, 1]])


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
@pytest.mark.filterwarnings("error:.*integer via a float.*:DeprecationWarning")
def test_integer_signs(dtype):
    dtype = np.dtype(dtype)
    assert np.loadtxt(["+2"], dtype=dtype) == 2
    if dtype.kind == "u":
        with pytest.raises(ValueError):
            np.loadtxt(["-1\n"], dtype=dtype)
    else:
        assert np.loadtxt(["-2\n"], dtype=dtype) == -2

    for sign in ["++", "+-", "--", "-+"]:
        with pytest.raises(ValueError):
            np.loadtxt([f"{sign}2\n"], dtype=dtype)


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype", np.typecodes["AllInteger"])
@pytest.mark.filterwarnings("error:.*integer via a float.*:DeprecationWarning")
def test_implicit_cast_float_to_int_fails(dtype):
    txt = StringIO("1.0, 2.1, 3.7\n4, 5, 6")
    with pytest.raises(ValueError):
        np.loadtxt(txt, dtype=dtype, delimiter=",")

@pytest.mark.parametrize("dtype", (np.complex64, np.complex128))
@pytest.mark.parametrize("with_parens", (False, True))
def test_complex_parsing(dtype, with_parens):
    s = "(1.0-2.5j),3.75,(7+-5.0j)\n(4),(-19e2j),(0)"
    if not with_parens:
        s = s.replace("(", "").replace(")", "")

    res = np.loadtxt(StringIO(s), dtype=dtype, delimiter=",")
    expected = np.array(
        [[1.0-2.5j, 3.75, 7-5j], [4.0, -1900j, 0]], dtype=dtype
    )
    assert_equal(res, expected)


def test_read_from_generator():
    def gen():
        for i in range(4):
            yield f"{i},{2*i},{i**2}"

    res = np.loadtxt(gen(), dtype=int, delimiter=",")
    expected = np.array([[0, 0, 0], [1, 2, 1], [2, 4, 4], [3, 6, 9]])
    assert_equal(res, expected)


def test_read_from_generator_multitype():
    def gen():
        for i in range(3):
            yield f"{i} {i / 4}"

    res = np.loadtxt(gen(), dtype="i, d", delimiter=" ")
    expected = np.array([(0, 0.0), (1, 0.25), (2, 0.5)], dtype="i, d")
    assert_equal(res, expected)


def test_read_from_bad_generator():
    def gen():
        for entry in ["1,2", b"3, 5", 12738]:
            yield entry

    with pytest.raises(
            TypeError, match=r"non-string returned while reading data"):
        np.loadtxt(gen(), dtype="i, i", delimiter=",")


@pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
def test_object_cleanup_on_read_error():
    sentinel = object()
    already_read = 0

    def conv(x):
        nonlocal already_read
        if already_read > 4999:
            raise ValueError("failed half-way through!")
        already_read += 1
        return sentinel

    txt = StringIO("x\n" * 10000)

    with pytest.raises(ValueError, match="at row 5000, column 1"):
        np.loadtxt(txt, dtype=object, converters={0: conv})

    assert sys.getrefcount(sentinel) == 2


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
def test_character_not_bytes_compatible():
    """Test exception when a character cannot be encoded as 'S'."""
    data = StringIO("–")  # == \u2013
    with pytest.raises(ValueError):
        np.loadtxt(data, dtype="S5")


@pytest.mark.parametrize("conv", (0, [float], ""))
def test_invalid_converter(conv):
    msg = (
        "converters must be a dictionary mapping columns to converter "
        "functions or a single callable."
    )
    with pytest.raises(TypeError, match=msg):
        np.loadtxt(StringIO("1 2\n3 4"), converters=conv)


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
def test_converters_dict_raises_non_integer_key():
    with pytest.raises(TypeError, match="keys of the converters dict"):
        np.loadtxt(StringIO("1 2\n3 4"), converters={"a": int})
    with pytest.raises(TypeError, match="keys of the converters dict"):
        np.loadtxt(StringIO("1 2\n3 4"), converters={"a": int}, usecols=0)


@pytest.mark.parametrize("bad_col_ind", (3, -3))
def test_converters_dict_raises_non_col_key(bad_col_ind):
    data = StringIO("1 2\n3 4")
    with pytest.raises(ValueError, match="converter specified for column"):
        np.loadtxt(data, converters={bad_col_ind: int})


def test_converters_dict_raises_val_not_callable():
    with pytest.raises(TypeError,
                match="values of the converters dictionary must be callable"):
        np.loadtxt(StringIO("1 2\n3 4"), converters={0: 1})


@pytest.mark.parametrize("q", ('"', "'", "`"))
def test_quoted_field(q):
    txt = StringIO(
        f"{q}alpha, x{q}, 2.5\n{q}beta, y{q}, 4.5\n{q}gamma, z{q}, 5.0\n"
    )
    dtype = np.dtype([('f0', 'U8'), ('f1', np.float64)])
    expected = np.array(
        [("alpha, x", 2.5), ("beta, y", 4.5), ("gamma, z", 5.0)], dtype=dtype
    )

    res = np.loadtxt(txt, dtype=dtype, delimiter=",", quotechar=q)
    assert_array_equal(res, expected)


@pytest.mark.parametrize("q", ('"', "'", "`"))
def test_quoted_field_with_whitepace_delimiter(q):
    txt = StringIO(
        f"{q}alpha, x{q}     2.5\n{q}beta, y{q} 4.5\n{q}gamma, z{q}   5.0\n"
    )
    dtype = np.dtype([('f0', 'U8'), ('f1', np.float64)])
    expected = np.array(
        [("alpha, x", 2.5), ("beta, y", 4.5), ("gamma, z", 5.0)], dtype=dtype
    )

    res = np.loadtxt(txt, dtype=dtype, delimiter=None, quotechar=q)
    assert_array_equal(res, expected)


def test_quote_support_default():
    """Support for quoted fields is disabled by default."""
    txt = StringIO('"lat,long", 45, 30\n')
    dtype = np.dtype([('f0', 'U24'), ('f1', np.float64), ('f2', np.float64)])

    with pytest.raises(ValueError, match="the number of columns changed"):
        np.loadtxt(txt, dtype=dtype, delimiter=",")

    # Enable quoting support with non-None value for quotechar param
    txt.seek(0)
    expected = np.array([("lat,long", 45., 30.)], dtype=dtype)

    res = np.loadtxt(txt, dtype=dtype, delimiter=",", quotechar='"')
    assert_array_equal(res, expected)


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
def test_quotechar_multichar_error():
    txt = StringIO("1,2\n3,4")
    msg = r".*must be a single unicode character or None"
    with pytest.raises(TypeError, match=msg):
        np.loadtxt(txt, delimiter=",", quotechar="''")


def test_comment_multichar_error_with_quote():
    txt = StringIO("1,2\n3,4")
    msg = (
        "when multiple comments or a multi-character comment is given, "
        "quotes are not supported."
    )
    with pytest.raises(ValueError, match=msg):
        np.loadtxt(txt, delimiter=",", comments="123", quotechar='"')
    with pytest.raises(ValueError, match=msg):
        np.loadtxt(txt, delimiter=",", comments=["#", "%"], quotechar='"')

    # A single character string in a tuple is unpacked though:
    res = np.loadtxt(txt, delimiter=",", comments=("#",), quotechar="'")
    assert_equal(res, [[1, 2], [3, 4]])


def test_structured_dtype_with_quotes():
    data = StringIO(
        (
            "1000;2.4;'alpha';-34\n"
            "2000;3.1;'beta';29\n"
            "3500;9.9;'gamma';120\n"
            "4090;8.1;'delta';0\n"
            "5001;4.4;'epsilon';-99\n"
            "6543;7.8;'omega';-1\n"
        )
    )
    dtype = np.dtype(
        [('f0', np.uint16), ('f1', np.float64), ('f2', 'S7'), ('f3', np.int8)]
    )
    expected = np.array(
        [
            (1000, 2.4, "alpha", -34),
            (2000, 3.1, "beta", 29),
            (3500, 9.9, "gamma", 120),
            (4090, 8.1, "delta", 0),
            (5001, 4.4, "epsilon", -99),
            (6543, 7.8, "omega", -1)
        ],
        dtype=dtype
    )
    res = np.loadtxt(data, dtype=dtype, delimiter=";", quotechar="'")
    assert_array_equal(res, expected)


def test_quoted_field_is_not_empty():
    txt = StringIO('1\n\n"4"\n""')
    expected = np.array(["1", "4", ""], dtype="U1")
    res = np.loadtxt(txt, delimiter=",", dtype="U1", quotechar='"')
    assert_equal(res, expected)

def test_quoted_field_is_not_empty_nonstrict():
    # Same as test_quoted_field_is_not_empty but check that we are not strict
    # about missing closing quote (this is the `csv.reader` default also)
    txt = StringIO('1\n\n"4"\n"')
    expected = np.array(["1", "4", ""], dtype="U1")
    res = np.loadtxt(txt, delimiter=",", dtype="U1", quotechar='"')
    assert_equal(res, expected)

def test_consecutive_quotechar_escaped():
    txt = StringIO('"Hello, my name is ""Monty""!"')
    expected = np.array('Hello, my name is "Monty"!', dtype="U40")
    res = np.loadtxt(txt, dtype="U40", delimiter=",", quotechar='"')
    assert_equal(res, expected)


@pytest.mark.parametrize("data", ("", "\n\n\n", "# 1 2 3\n# 4 5 6\n"))
@pytest.mark.parametrize("ndmin", (0, 1, 2))
@pytest.mark.parametrize("usecols", [None, (1, 2, 3)])
def test_warn_on_no_data(data, ndmin, usecols):
    """Check that a UserWarning is emitted when no data is read from input."""
    if usecols is not None:
        expected_shape = (0, 3)
    elif ndmin == 2:
        expected_shape = (0, 1)  # guess a single column?!
    else:
        expected_shape = (0,)

    txt = StringIO(data)
    with pytest.warns(UserWarning, match="input contained no data"):
        res = np.loadtxt(txt, ndmin=ndmin, usecols=usecols)
    assert res.shape == expected_shape

    with NamedTemporaryFile(mode="w") as fh:
        fh.write(data)
        fh.seek(0)
        with pytest.warns(UserWarning, match="input contained no data"):
            res = np.loadtxt(txt, ndmin=ndmin, usecols=usecols)
        assert res.shape == expected_shape

@pytest.mark.parametrize("skiprows", (2, 3))
def test_warn_on_skipped_data(skiprows):
    data = "1 2 3\n4 5 6"
    txt = StringIO(data)
    with pytest.warns(UserWarning, match="input contained no data"):
        np.loadtxt(txt, skiprows=skiprows)


@pytest.mark.parametrize(["dtype", "value"], [
        ("i2", 0x0001), ("u2", 0x0001),
        ("i4", 0x00010203), ("u4", 0x00010203),
        ("i8", 0x0001020304050607), ("u8", 0x0001020304050607),
        # The following values are constructed to lead to unique bytes:
        ("float16", 3.07e-05),
        ("float32", 9.2557e-41), ("complex64", 9.2557e-41+2.8622554e-29j),
        ("float64", -1.758571353180402e-24),
        # Here and below, the repr side-steps a small loss of precision in
        # complex `str` in PyPy (which is probably fine, as repr works):
        ("complex128", repr(5.406409232372729e-29-1.758571353180402e-24j)),
        # Use integer values that fit into double.  Everything else leads to
        # problems due to longdoubles going via double and decimal strings
        # causing rounding errors.
        ("longdouble", 0x01020304050607),
        ("clongdouble", repr(0x01020304050607 + (0x00121314151617 * 1j))),
        ("U2", "\U00010203\U000a0b0c")])
@pytest.mark.parametrize("swap", [True, False])
def test_byteswapping_and_unaligned(dtype, value, swap):
    # Try to create "interesting" values within the valid unicode range:
    dtype = np.dtype(dtype)
    data = [f"x,{value}\n"]  # repr as PyPy `str` truncates some
    if swap:
        dtype = dtype.newbyteorder()
    full_dt = np.dtype([("a", "S1"), ("b", dtype)], align=False)
    # The above ensures that the interesting "b" field is unaligned:
    assert full_dt.fields["b"][1] == 1
    res = np.loadtxt(data, dtype=full_dt, delimiter=",", encoding=None,
                     max_rows=1)  # max-rows prevents over-allocation
    assert res["b"] == dtype.type(value)


@pytest.mark.parametrize("dtype",
        np.typecodes["AllInteger"] + "efdFD" + "?")
def test_unicode_whitespace_stripping(dtype):
    # Test that all numeric types (and bool) strip whitespace correctly
    # \u202F is a narrow no-break space, `\n` is just a whitespace if quoted.
    # Currently, skip float128 as it did not always support this and has no
    # "custom" parsing:
    txt = StringIO(' 3 ,"\u202F2\n"')
    res = np.loadtxt(txt, dtype=dtype, delimiter=",", quotechar='"')
    assert_array_equal(res, np.array([3, 2]).astype(dtype))


@pytest.mark.parametrize("dtype", "FD")
def test_unicode_whitespace_stripping_complex(dtype):
    # Complex has a few extra cases since it has two components and
    # parentheses
    line = " 1 , 2+3j , ( 4+5j ), ( 6+-7j )  , 8j , ( 9j ) \n"
    data = [line, line.replace(" ", "\u202F")]
    res = np.loadtxt(data, dtype=dtype, delimiter=',')
    assert_array_equal(res, np.array([[1, 2+3j, 4+5j, 6-7j, 8j, 9j]] * 2))


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype", "FD")
@pytest.mark.parametrize("field",
        ["1 +2j", "1+ 2j", "1+2 j", "1+-+3", "(1j", "(1", "(1+2j", "1+2j)"])
def test_bad_complex(dtype, field):
    with pytest.raises(ValueError):
        np.loadtxt([field + "\n"], dtype=dtype, delimiter=",")


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype",
            np.typecodes["AllInteger"] + "efgdFDG" + "?")
def test_nul_character_error(dtype):
    # Test that a \0 character is correctly recognized as an error even if
    # what comes before is valid (not everything gets parsed internally).
    if dtype.lower() == "g":
        pytest.xfail("longdouble/clongdouble assignment may misbehave.")
    with pytest.raises(ValueError):
        np.loadtxt(["1\000"], dtype=dtype, delimiter=",", quotechar='"')


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
@pytest.mark.parametrize("dtype",
        np.typecodes["AllInteger"] + "efgdFDG" + "?")
def test_no_thousands_support(dtype):
    # Mainly to document behaviour, Python supports thousands like 1_1.
    # (e and G may end up using different conversion and support it, this is
    # a bug but happens...)
    if dtype == "e":
        pytest.skip("half assignment currently uses Python float converter")
    if dtype in "eG":
        pytest.xfail("clongdouble assignment is buggy (uses `complex`?).")

    assert int("1_1") == float("1_1") == complex("1_1") == 11
    with pytest.raises(ValueError):
        np.loadtxt(["1_1\n"], dtype=dtype)


@pytest.mark.parametrize("data", [
    ["1,2\n", "2\n,3\n"],
    ["1,2\n", "2\r,3\n"]])
def test_bad_newline_in_iterator(data):
    # In NumPy <=1.22 this was accepted, because newlines were completely
    # ignored when the input was an iterable.  This could be changed, but right
    # now, we raise an error.
    msg = "Found an unquoted embedded newline within a single line"
    with pytest.raises(ValueError, match=msg):
        np.loadtxt(data, delimiter=",")


@pytest.mark.parametrize("data", [
    ["1,2\n", "2,3\r\n"],  # a universal newline
    ["1,2\n", "'2\n',3\n"],  # a quoted newline
    ["1,2\n", "'2\r',3\n"],
    ["1,2\n", "'2\r\n',3\n"],
])
def test_good_newline_in_iterator(data):
    # The quoted newlines will be untransformed here, but are just whitespace.
    res = np.loadtxt(data, delimiter=",", quotechar="'")
    assert_array_equal(res, [[1., 2.], [2., 3.]])


@pytest.mark.parametrize("newline", ["\n", "\r", "\r\n"])
def test_universal_newlines_quoted(newline):
    # Check that universal newline support within the tokenizer is not applied
    # to quoted fields.  (note that lines must end in newline or quoted
    # fields will not include a newline at all)
    data = ['1,"2\n"\n', '3,"4\n', '1"\n']
    data = [row.replace("\n", newline) for row in data]
    res = np.loadtxt(data, dtype=object, delimiter=",", quotechar='"')
    assert_array_equal(res, [['1', f'2{newline}'], ['3', f'4{newline}1']])


def test_null_character():
    # Basic tests to check that the NUL character is not special:
    res = np.loadtxt(["1\0002\0003\n", "4\0005\0006"], delimiter="\000")
    assert_array_equal(res, [[1, 2, 3], [4, 5, 6]])

    # Also not as part of a field (avoid unicode/arrays as unicode strips \0)
    res = np.loadtxt(["1\000,2\000,3\n", "4\000,5\000,6"],
                     delimiter=",", dtype=object)
    assert res.tolist() == [["1\000", "2\000", "3"], ["4\000", "5\000", "6"]]


def test_iterator_fails_getting_next_line():
    class BadSequence:
        def __len__(self):
            return 100

        def __getitem__(self, item):
            if item == 50:
                raise RuntimeError("Bad things happened!")
            return f"{item}, {item+1}"

    with pytest.raises(RuntimeError, match="Bad things happened!"):
        np.loadtxt(BadSequence(), dtype=int, delimiter=",")


class TestCReaderUnitTests:
    # These are internal tests for path that should not be possible to hit
    # unless things go very very wrong somewhere.
    def test_not_an_filelike(self):
        with pytest.raises(AttributeError, match=".*read"):
            np.core._multiarray_umath._load_from_filelike(
                object(), dtype=np.dtype("i"), filelike=True)

    def test_filelike_read_fails(self):
        # Can only be reached if loadtxt opens the file, so it is hard to do
        # via the public interface (although maybe not impossible considering
        # the current "DataClass" backing).
        class BadFileLike:
            counter = 0

            def read(self, size):
                self.counter += 1
                if self.counter > 20:
                    raise RuntimeError("Bad bad bad!")
                return "1,2,3\n"

        with pytest.raises(RuntimeError, match="Bad bad bad!"):
            np.core._multiarray_umath._load_from_filelike(
                BadFileLike(), dtype=np.dtype("i"), filelike=True)

    def test_filelike_bad_read(self):
        # Can only be reached if loadtxt opens the file, so it is hard to do
        # via the public interface (although maybe not impossible considering
        # the current "DataClass" backing).

        class BadFileLike:
            counter = 0

            def read(self, size):
                return 1234  # not a string!

        with pytest.raises(TypeError,
                    match="non-string returned while reading data"):
            np.core._multiarray_umath._load_from_filelike(
                BadFileLike(), dtype=np.dtype("i"), filelike=True)

    def test_not_an_iter(self):
        with pytest.raises(TypeError,
                    match="error reading from object, expected an iterable"):
            np.core._multiarray_umath._load_from_filelike(
                object(), dtype=np.dtype("i"), filelike=False)

    def test_bad_type(self):
        with pytest.raises(TypeError, match="internal error: dtype must"):
            np.core._multiarray_umath._load_from_filelike(
                object(), dtype="i", filelike=False)

    def test_bad_encoding(self):
        with pytest.raises(TypeError, match="encoding must be a unicode"):
            np.core._multiarray_umath._load_from_filelike(
                object(), dtype=np.dtype("i"), filelike=False, encoding=123)

    @pytest.mark.parametrize("newline", ["\r", "\n", "\r\n"])
    def test_manual_universal_newlines(self, newline):
        # This is currently not available to users, because we should always
        # open files with universal newlines enabled `newlines=None`.
        # (And reading from an iterator uses slightly different code paths.)
        # We have no real support for `newline="\r"` or `newline="\n" as the
        # user cannot specify those options.
        data = StringIO('0\n1\n"2\n"\n3\n4 #\n'.replace("\n", newline),
                        newline="")

        res = np.core._multiarray_umath._load_from_filelike(
            data, dtype=np.dtype("U10"), filelike=True,
            quote='"', comment="#", skiplines=1)
        assert_array_equal(res[:, 0], ["1", f"2{newline}", "3", "4 "])


def test_delimiter_comment_collision_raises():
    with pytest.raises(TypeError, match=".*control characters.*incompatible"):
        np.loadtxt(StringIO("1, 2, 3"), delimiter=",", comments=",")


def test_delimiter_quotechar_collision_raises():
    with pytest.raises(TypeError, match=".*control characters.*incompatible"):
        np.loadtxt(StringIO("1, 2, 3"), delimiter=",", quotechar=",")


def test_comment_quotechar_collision_raises():
    with pytest.raises(TypeError, match=".*control characters.*incompatible"):
        np.loadtxt(StringIO("1 2 3"), comments="#", quotechar="#")


def test_delimiter_and_multiple_comments_collision_raises():
    with pytest.raises(
        TypeError, match="Comment characters.*cannot include the delimiter"
    ):
        np.loadtxt(StringIO("1, 2, 3"), delimiter=",", comments=["#", ","])


@pytest.mark.parametrize(
    "ws",
    (
        " ",  # space
        "\t",  # tab
        "\u2003",  # em
        "\u00A0",  # non-break
        "\u3000",  # ideographic space
    )
)
def test_collision_with_default_delimiter_raises(ws):
    with pytest.raises(TypeError, match=".*control characters.*incompatible"):
        np.loadtxt(StringIO(f"1{ws}2{ws}3\n4{ws}5{ws}6\n"), comments=ws)
    with pytest.raises(TypeError, match=".*control characters.*incompatible"):
        np.loadtxt(StringIO(f"1{ws}2{ws}3\n4{ws}5{ws}6\n"), quotechar=ws)


@pytest.mark.parametrize("nl", ("\n", "\r"))
def test_control_character_newline_raises(nl):
    txt = StringIO(f"1{nl}2{nl}3{nl}{nl}4{nl}5{nl}6{nl}{nl}")
    msg = "control character.*cannot be a newline"
    with pytest.raises(TypeError, match=msg):
        np.loadtxt(txt, delimiter=nl)
    with pytest.raises(TypeError, match=msg):
        np.loadtxt(txt, comments=nl)
    with pytest.raises(TypeError, match=msg):
        np.loadtxt(txt, quotechar=nl)


@pytest.mark.parametrize(
    ("generic_data", "long_datum", "unitless_dtype", "expected_dtype"),
    [
        ("2012-03", "2013-01-15", "M8", "M8[D]"),  # Datetimes
        ("spam-a-lot", "tis_but_a_scratch", "U", "U17"),  # str
    ],
)
@pytest.mark.parametrize("nrows", (10, 50000, 60000))  # lt, eq, gt chunksize
def test_parametric_unit_discovery(
    generic_data, long_datum, unitless_dtype, expected_dtype, nrows
):
    """Check that the correct unit (e.g. month, day, second) is discovered from
    the data when a user specifies a unitless datetime."""
    # Unit should be "D" (days) due to last entry
    data = [generic_data] * 50000 + [long_datum]
    expected = np.array(data, dtype=expected_dtype)

    # file-like path
    txt = StringIO("\n".join(data))
    a = np.loadtxt(txt, dtype=unitless_dtype)
    assert a.dtype == expected.dtype
    assert_equal(a, expected)

    # file-obj path
    fd, fname = mkstemp()
    os.close(fd)
    with open(fname, "w") as fh:
        fh.write("\n".join(data))
    a = np.loadtxt(fname, dtype=unitless_dtype)
    os.remove(fname)
    assert a.dtype == expected.dtype
    assert_equal(a, expected)


def test_str_dtype_unit_discovery_with_converter():
    data = ["spam-a-lot"] * 60000 + ["XXXtis_but_a_scratch"]
    expected = np.array(
        ["spam-a-lot"] * 60000 + ["tis_but_a_scratch"], dtype="U17"
    )
    conv = lambda s: s.strip("XXX")

    # file-like path
    txt = StringIO("\n".join(data))
    a = np.loadtxt(txt, dtype="U", converters=conv, encoding=None)
    assert a.dtype == expected.dtype
    assert_equal(a, expected)

    # file-obj path
    fd, fname = mkstemp()
    os.close(fd)
    with open(fname, "w") as fh:
        fh.write("\n".join(data))
    a = np.loadtxt(fname, dtype="U", converters=conv, encoding=None)
    os.remove(fname)
    assert a.dtype == expected.dtype
    assert_equal(a, expected)


@pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
                    reason="PyPy bug in error formatting")
def test_control_character_empty():
    with pytest.raises(TypeError, match="Text reading control character must"):
        np.loadtxt(StringIO("1 2 3"), delimiter="")
    with pytest.raises(TypeError, match="Text reading control character must"):
        np.loadtxt(StringIO("1 2 3"), quotechar="")
    with pytest.raises(ValueError, match="comments cannot be an empty string"):
        np.loadtxt(StringIO("1 2 3"), comments="")
    with pytest.raises(ValueError, match="comments cannot be an empty string"):
        np.loadtxt(StringIO("1 2 3"), comments=["#", ""])


def test_control_characters_as_bytes():
    """Byte control characters (comments, delimiter) are supported."""
    a = np.loadtxt(StringIO("#header\n1,2,3"), comments=b"#", delimiter=b",")
    assert_equal(a, [1, 2, 3])


@pytest.mark.filterwarnings('ignore::UserWarning')
def test_field_growing_cases():
    # Test empty field appending/growing (each field still takes 1 character)
    # to see if the final field appending does not create issues.
    res = np.loadtxt([""], delimiter=",", dtype=bytes)
    assert len(res) == 0

    for i in range(1, 1024):
        res = np.loadtxt(["," * i], delimiter=",", dtype=bytes)
        assert len(res) == i+1
