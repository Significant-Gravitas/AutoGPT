import itertools
import contextlib
import operator
import pytest

import numpy as np
import numpy.core._multiarray_tests as mt

from numpy.testing import assert_raises, assert_equal


INT64_MAX = np.iinfo(np.int64).max
INT64_MIN = np.iinfo(np.int64).min
INT64_MID = 2**32

# int128 is not two's complement, the sign bit is separate
INT128_MAX = 2**128 - 1
INT128_MIN = -INT128_MAX
INT128_MID = 2**64

INT64_VALUES = (
    [INT64_MIN + j for j in range(20)] +
    [INT64_MAX - j for j in range(20)] +
    [INT64_MID + j for j in range(-20, 20)] +
    [2*INT64_MID + j for j in range(-20, 20)] +
    [INT64_MID//2 + j for j in range(-20, 20)] +
    list(range(-70, 70))
)

INT128_VALUES = (
    [INT128_MIN + j for j in range(20)] +
    [INT128_MAX - j for j in range(20)] +
    [INT128_MID + j for j in range(-20, 20)] +
    [2*INT128_MID + j for j in range(-20, 20)] +
    [INT128_MID//2 + j for j in range(-20, 20)] +
    list(range(-70, 70)) +
    [False]  # negative zero
)

INT64_POS_VALUES = [x for x in INT64_VALUES if x > 0]


@contextlib.contextmanager
def exc_iter(*args):
    """
    Iterate over Cartesian product of *args, and if an exception is raised,
    add information of the current iterate.
    """

    value = [None]

    def iterate():
        for v in itertools.product(*args):
            value[0] = v
            yield v

    try:
        yield iterate()
    except Exception:
        import traceback
        msg = "At: %r\n%s" % (repr(value[0]),
                              traceback.format_exc())
        raise AssertionError(msg)


def test_safe_binop():
    # Test checked arithmetic routines

    ops = [
        (operator.add, 1),
        (operator.sub, 2),
        (operator.mul, 3)
    ]

    with exc_iter(ops, INT64_VALUES, INT64_VALUES) as it:
        for xop, a, b in it:
            pyop, op = xop
            c = pyop(a, b)

            if not (INT64_MIN <= c <= INT64_MAX):
                assert_raises(OverflowError, mt.extint_safe_binop, a, b, op)
            else:
                d = mt.extint_safe_binop(a, b, op)
                if c != d:
                    # assert_equal is slow
                    assert_equal(d, c)


def test_to_128():
    with exc_iter(INT64_VALUES) as it:
        for a, in it:
            b = mt.extint_to_128(a)
            if a != b:
                assert_equal(b, a)


def test_to_64():
    with exc_iter(INT128_VALUES) as it:
        for a, in it:
            if not (INT64_MIN <= a <= INT64_MAX):
                assert_raises(OverflowError, mt.extint_to_64, a)
            else:
                b = mt.extint_to_64(a)
                if a != b:
                    assert_equal(b, a)


def test_mul_64_64():
    with exc_iter(INT64_VALUES, INT64_VALUES) as it:
        for a, b in it:
            c = a * b
            d = mt.extint_mul_64_64(a, b)
            if c != d:
                assert_equal(d, c)


def test_add_128():
    with exc_iter(INT128_VALUES, INT128_VALUES) as it:
        for a, b in it:
            c = a + b
            if not (INT128_MIN <= c <= INT128_MAX):
                assert_raises(OverflowError, mt.extint_add_128, a, b)
            else:
                d = mt.extint_add_128(a, b)
                if c != d:
                    assert_equal(d, c)


def test_sub_128():
    with exc_iter(INT128_VALUES, INT128_VALUES) as it:
        for a, b in it:
            c = a - b
            if not (INT128_MIN <= c <= INT128_MAX):
                assert_raises(OverflowError, mt.extint_sub_128, a, b)
            else:
                d = mt.extint_sub_128(a, b)
                if c != d:
                    assert_equal(d, c)


def test_neg_128():
    with exc_iter(INT128_VALUES) as it:
        for a, in it:
            b = -a
            c = mt.extint_neg_128(a)
            if b != c:
                assert_equal(c, b)


def test_shl_128():
    with exc_iter(INT128_VALUES) as it:
        for a, in it:
            if a < 0:
                b = -(((-a) << 1) & (2**128-1))
            else:
                b = (a << 1) & (2**128-1)
            c = mt.extint_shl_128(a)
            if b != c:
                assert_equal(c, b)


def test_shr_128():
    with exc_iter(INT128_VALUES) as it:
        for a, in it:
            if a < 0:
                b = -((-a) >> 1)
            else:
                b = a >> 1
            c = mt.extint_shr_128(a)
            if b != c:
                assert_equal(c, b)


def test_gt_128():
    with exc_iter(INT128_VALUES, INT128_VALUES) as it:
        for a, b in it:
            c = a > b
            d = mt.extint_gt_128(a, b)
            if c != d:
                assert_equal(d, c)


@pytest.mark.slow
def test_divmod_128_64():
    with exc_iter(INT128_VALUES, INT64_POS_VALUES) as it:
        for a, b in it:
            if a >= 0:
                c, cr = divmod(a, b)
            else:
                c, cr = divmod(-a, b)
                c = -c
                cr = -cr

            d, dr = mt.extint_divmod_128_64(a, b)

            if c != d or d != dr or b*d + dr != a:
                assert_equal(d, c)
                assert_equal(dr, cr)
                assert_equal(b*d + dr, a)


def test_floordiv_128_64():
    with exc_iter(INT128_VALUES, INT64_POS_VALUES) as it:
        for a, b in it:
            c = a // b
            d = mt.extint_floordiv_128_64(a, b)

            if c != d:
                assert_equal(d, c)


def test_ceildiv_128_64():
    with exc_iter(INT128_VALUES, INT64_POS_VALUES) as it:
        for a, b in it:
            c = (a + b - 1) // b
            d = mt.extint_ceildiv_128_64(a, b)

            if c != d:
                assert_equal(d, c)
