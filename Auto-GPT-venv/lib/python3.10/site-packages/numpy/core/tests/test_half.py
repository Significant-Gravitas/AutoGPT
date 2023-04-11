import platform
import pytest

import numpy as np
from numpy import uint16, float16, float32, float64
from numpy.testing import assert_, assert_equal, _OLD_PROMOTION, IS_WASM


def assert_raises_fpe(strmatch, callable, *args, **kwargs):
    try:
        callable(*args, **kwargs)
    except FloatingPointError as exc:
        assert_(str(exc).find(strmatch) >= 0,
                "Did not raise floating point %s error" % strmatch)
    else:
        assert_(False,
                "Did not raise floating point %s error" % strmatch)

class TestHalf:
    def setup_method(self):
        # An array of all possible float16 values
        self.all_f16 = np.arange(0x10000, dtype=uint16)
        self.all_f16.dtype = float16
        self.all_f32 = np.array(self.all_f16, dtype=float32)
        self.all_f64 = np.array(self.all_f16, dtype=float64)

        # An array of all non-NaN float16 values, in sorted order
        self.nonan_f16 = np.concatenate(
                                (np.arange(0xfc00, 0x7fff, -1, dtype=uint16),
                                 np.arange(0x0000, 0x7c01, 1, dtype=uint16)))
        self.nonan_f16.dtype = float16
        self.nonan_f32 = np.array(self.nonan_f16, dtype=float32)
        self.nonan_f64 = np.array(self.nonan_f16, dtype=float64)

        # An array of all finite float16 values, in sorted order
        self.finite_f16 = self.nonan_f16[1:-1]
        self.finite_f32 = self.nonan_f32[1:-1]
        self.finite_f64 = self.nonan_f64[1:-1]

    def test_half_conversions(self):
        """Checks that all 16-bit values survive conversion
           to/from 32-bit and 64-bit float"""
        # Because the underlying routines preserve the NaN bits, every
        # value is preserved when converting to/from other floats.

        # Convert from float32 back to float16
        b = np.array(self.all_f32, dtype=float16)
        assert_equal(self.all_f16.view(dtype=uint16),
                     b.view(dtype=uint16))

        # Convert from float64 back to float16
        b = np.array(self.all_f64, dtype=float16)
        assert_equal(self.all_f16.view(dtype=uint16),
                     b.view(dtype=uint16))

        # Convert float16 to longdouble and back
        # This doesn't necessarily preserve the extra NaN bits,
        # so exclude NaNs.
        a_ld = np.array(self.nonan_f16, dtype=np.longdouble)
        b = np.array(a_ld, dtype=float16)
        assert_equal(self.nonan_f16.view(dtype=uint16),
                     b.view(dtype=uint16))

        # Check the range for which all integers can be represented
        i_int = np.arange(-2048, 2049)
        i_f16 = np.array(i_int, dtype=float16)
        j = np.array(i_f16, dtype=int)
        assert_equal(i_int, j)

    @pytest.mark.parametrize("string_dt", ["S", "U"])
    def test_half_conversion_to_string(self, string_dt):
        # Currently uses S/U32 (which is sufficient for float32)
        expected_dt = np.dtype(f"{string_dt}32")
        assert np.promote_types(np.float16, string_dt) == expected_dt
        assert np.promote_types(string_dt, np.float16) == expected_dt

        arr = np.ones(3, dtype=np.float16).astype(string_dt)
        assert arr.dtype == expected_dt

    @pytest.mark.parametrize("string_dt", ["S", "U"])
    def test_half_conversion_from_string(self, string_dt):
        string = np.array("3.1416", dtype=string_dt)
        assert string.astype(np.float16) == np.array(3.1416, dtype=np.float16)

    @pytest.mark.parametrize("offset", [None, "up", "down"])
    @pytest.mark.parametrize("shift", [None, "up", "down"])
    @pytest.mark.parametrize("float_t", [np.float32, np.float64])
    @np._no_nep50_warning()
    def test_half_conversion_rounding(self, float_t, shift, offset):
        # Assumes that round to even is used during casting.
        max_pattern = np.float16(np.finfo(np.float16).max).view(np.uint16)

        # Test all (positive) finite numbers, denormals are most interesting
        # however:
        f16s_patterns = np.arange(0, max_pattern+1, dtype=np.uint16)
        f16s_float = f16s_patterns.view(np.float16).astype(float_t)

        # Shift the values by half a bit up or a down (or do not shift),
        if shift == "up":
            f16s_float = 0.5 * (f16s_float[:-1] + f16s_float[1:])[1:]
        elif shift == "down":
            f16s_float = 0.5 * (f16s_float[:-1] + f16s_float[1:])[:-1]
        else:
            f16s_float = f16s_float[1:-1]

        # Increase the float by a minimal value:
        if offset == "up":
            f16s_float = np.nextafter(f16s_float, float_t(np.inf))
        elif offset == "down":
            f16s_float = np.nextafter(f16s_float, float_t(-np.inf))

        # Convert back to float16 and its bit pattern:
        res_patterns = f16s_float.astype(np.float16).view(np.uint16)

        # The above calculations tries the original values, or the exact
        # mid points between the float16 values. It then further offsets them
        # by as little as possible. If no offset occurs, "round to even"
        # logic will be necessary, an arbitrarily small offset should cause
        # normal up/down rounding always.

        # Calculate the expected pattern:
        cmp_patterns = f16s_patterns[1:-1].copy()

        if shift == "down" and offset != "up":
            shift_pattern = -1
        elif shift == "up" and offset != "down":
            shift_pattern = 1
        else:
            # There cannot be a shift, either shift is None, so all rounding
            # will go back to original, or shift is reduced by offset too much.
            shift_pattern = 0

        # If rounding occurs, is it normal rounding or round to even?
        if offset is None:
            # Round to even occurs, modify only non-even, cast to allow + (-1)
            cmp_patterns[0::2].view(np.int16)[...] += shift_pattern
        else:
            cmp_patterns.view(np.int16)[...] += shift_pattern

        assert_equal(res_patterns, cmp_patterns)

    @pytest.mark.parametrize(["float_t", "uint_t", "bits"],
                             [(np.float32, np.uint32, 23),
                              (np.float64, np.uint64, 52)])
    def test_half_conversion_denormal_round_even(self, float_t, uint_t, bits):
        # Test specifically that all bits are considered when deciding
        # whether round to even should occur (i.e. no bits are lost at the
        # end. Compare also gh-12721. The most bits can get lost for the
        # smallest denormal:
        smallest_value = np.uint16(1).view(np.float16).astype(float_t)
        assert smallest_value == 2**-24

        # Will be rounded to zero based on round to even rule:
        rounded_to_zero = smallest_value / float_t(2)
        assert rounded_to_zero.astype(np.float16) == 0

        # The significand will be all 0 for the float_t, test that we do not
        # lose the lower ones of these:
        for i in range(bits):
            # slightly increasing the value should make it round up:
            larger_pattern = rounded_to_zero.view(uint_t) | uint_t(1 << i)
            larger_value = larger_pattern.view(float_t)
            assert larger_value.astype(np.float16) == smallest_value

    def test_nans_infs(self):
        with np.errstate(all='ignore'):
            # Check some of the ufuncs
            assert_equal(np.isnan(self.all_f16), np.isnan(self.all_f32))
            assert_equal(np.isinf(self.all_f16), np.isinf(self.all_f32))
            assert_equal(np.isfinite(self.all_f16), np.isfinite(self.all_f32))
            assert_equal(np.signbit(self.all_f16), np.signbit(self.all_f32))
            assert_equal(np.spacing(float16(65504)), np.inf)

            # Check comparisons of all values with NaN
            nan = float16(np.nan)

            assert_(not (self.all_f16 == nan).any())
            assert_(not (nan == self.all_f16).any())

            assert_((self.all_f16 != nan).all())
            assert_((nan != self.all_f16).all())

            assert_(not (self.all_f16 < nan).any())
            assert_(not (nan < self.all_f16).any())

            assert_(not (self.all_f16 <= nan).any())
            assert_(not (nan <= self.all_f16).any())

            assert_(not (self.all_f16 > nan).any())
            assert_(not (nan > self.all_f16).any())

            assert_(not (self.all_f16 >= nan).any())
            assert_(not (nan >= self.all_f16).any())

    def test_half_values(self):
        """Confirms a small number of known half values"""
        a = np.array([1.0, -1.0,
                      2.0, -2.0,
                      0.0999755859375, 0.333251953125,  # 1/10, 1/3
                      65504, -65504,           # Maximum magnitude
                      2.0**(-14), -2.0**(-14),  # Minimum normal
                      2.0**(-24), -2.0**(-24),  # Minimum subnormal
                      0, -1/1e1000,            # Signed zeros
                      np.inf, -np.inf])
        b = np.array([0x3c00, 0xbc00,
                      0x4000, 0xc000,
                      0x2e66, 0x3555,
                      0x7bff, 0xfbff,
                      0x0400, 0x8400,
                      0x0001, 0x8001,
                      0x0000, 0x8000,
                      0x7c00, 0xfc00], dtype=uint16)
        b.dtype = float16
        assert_equal(a, b)

    def test_half_rounding(self):
        """Checks that rounding when converting to half is correct"""
        a = np.array([2.0**-25 + 2.0**-35,  # Rounds to minimum subnormal
                      2.0**-25,       # Underflows to zero (nearest even mode)
                      2.0**-26,       # Underflows to zero
                      1.0+2.0**-11 + 2.0**-16,  # rounds to 1.0+2**(-10)
                      1.0+2.0**-11,   # rounds to 1.0 (nearest even mode)
                      1.0+2.0**-12,   # rounds to 1.0
                      65519,          # rounds to 65504
                      65520],         # rounds to inf
                      dtype=float64)
        rounded = [2.0**-24,
                   0.0,
                   0.0,
                   1.0+2.0**(-10),
                   1.0,
                   1.0,
                   65504,
                   np.inf]

        # Check float64->float16 rounding
        with np.errstate(over="ignore"):
            b = np.array(a, dtype=float16)
        assert_equal(b, rounded)

        # Check float32->float16 rounding
        a = np.array(a, dtype=float32)
        with np.errstate(over="ignore"):
            b = np.array(a, dtype=float16)
        assert_equal(b, rounded)

    def test_half_correctness(self):
        """Take every finite float16, and check the casting functions with
           a manual conversion."""

        # Create an array of all finite float16s
        a_bits = self.finite_f16.view(dtype=uint16)

        # Convert to 64-bit float manually
        a_sgn = (-1.0)**((a_bits & 0x8000) >> 15)
        a_exp = np.array((a_bits & 0x7c00) >> 10, dtype=np.int32) - 15
        a_man = (a_bits & 0x03ff) * 2.0**(-10)
        # Implicit bit of normalized floats
        a_man[a_exp != -15] += 1
        # Denormalized exponent is -14
        a_exp[a_exp == -15] = -14

        a_manual = a_sgn * a_man * 2.0**a_exp

        a32_fail = np.nonzero(self.finite_f32 != a_manual)[0]
        if len(a32_fail) != 0:
            bad_index = a32_fail[0]
            assert_equal(self.finite_f32, a_manual,
                 "First non-equal is half value %x -> %g != %g" %
                            (self.finite_f16[bad_index],
                             self.finite_f32[bad_index],
                             a_manual[bad_index]))

        a64_fail = np.nonzero(self.finite_f64 != a_manual)[0]
        if len(a64_fail) != 0:
            bad_index = a64_fail[0]
            assert_equal(self.finite_f64, a_manual,
                 "First non-equal is half value %x -> %g != %g" %
                            (self.finite_f16[bad_index],
                             self.finite_f64[bad_index],
                             a_manual[bad_index]))

    def test_half_ordering(self):
        """Make sure comparisons are working right"""

        # All non-NaN float16 values in reverse order
        a = self.nonan_f16[::-1].copy()

        # 32-bit float copy
        b = np.array(a, dtype=float32)

        # Should sort the same
        a.sort()
        b.sort()
        assert_equal(a, b)

        # Comparisons should work
        assert_((a[:-1] <= a[1:]).all())
        assert_(not (a[:-1] > a[1:]).any())
        assert_((a[1:] >= a[:-1]).all())
        assert_(not (a[1:] < a[:-1]).any())
        # All != except for +/-0
        assert_equal(np.nonzero(a[:-1] < a[1:])[0].size, a.size-2)
        assert_equal(np.nonzero(a[1:] > a[:-1])[0].size, a.size-2)

    def test_half_funcs(self):
        """Test the various ArrFuncs"""

        # fill
        assert_equal(np.arange(10, dtype=float16),
                     np.arange(10, dtype=float32))

        # fillwithscalar
        a = np.zeros((5,), dtype=float16)
        a.fill(1)
        assert_equal(a, np.ones((5,), dtype=float16))

        # nonzero and copyswap
        a = np.array([0, 0, -1, -1/1e20, 0, 2.0**-24, 7.629e-6], dtype=float16)
        assert_equal(a.nonzero()[0],
                     [2, 5, 6])
        a = a.byteswap().newbyteorder()
        assert_equal(a.nonzero()[0],
                     [2, 5, 6])

        # dot
        a = np.arange(0, 10, 0.5, dtype=float16)
        b = np.ones((20,), dtype=float16)
        assert_equal(np.dot(a, b),
                     95)

        # argmax
        a = np.array([0, -np.inf, -2, 0.5, 12.55, 7.3, 2.1, 12.4], dtype=float16)
        assert_equal(a.argmax(),
                     4)
        a = np.array([0, -np.inf, -2, np.inf, 12.55, np.nan, 2.1, 12.4], dtype=float16)
        assert_equal(a.argmax(),
                     5)

        # getitem
        a = np.arange(10, dtype=float16)
        for i in range(10):
            assert_equal(a.item(i), i)

    def test_spacing_nextafter(self):
        """Test np.spacing and np.nextafter"""
        # All non-negative finite #'s
        a = np.arange(0x7c00, dtype=uint16)
        hinf = np.array((np.inf,), dtype=float16)
        hnan = np.array((np.nan,), dtype=float16)
        a_f16 = a.view(dtype=float16)

        assert_equal(np.spacing(a_f16[:-1]), a_f16[1:]-a_f16[:-1])

        assert_equal(np.nextafter(a_f16[:-1], hinf), a_f16[1:])
        assert_equal(np.nextafter(a_f16[0], -hinf), -a_f16[1])
        assert_equal(np.nextafter(a_f16[1:], -hinf), a_f16[:-1])

        assert_equal(np.nextafter(hinf, a_f16), a_f16[-1])
        assert_equal(np.nextafter(-hinf, a_f16), -a_f16[-1])

        assert_equal(np.nextafter(hinf, hinf), hinf)
        assert_equal(np.nextafter(hinf, -hinf), a_f16[-1])
        assert_equal(np.nextafter(-hinf, hinf), -a_f16[-1])
        assert_equal(np.nextafter(-hinf, -hinf), -hinf)

        assert_equal(np.nextafter(a_f16, hnan), hnan[0])
        assert_equal(np.nextafter(hnan, a_f16), hnan[0])

        assert_equal(np.nextafter(hnan, hnan), hnan)
        assert_equal(np.nextafter(hinf, hnan), hnan)
        assert_equal(np.nextafter(hnan, hinf), hnan)

        # switch to negatives
        a |= 0x8000

        assert_equal(np.spacing(a_f16[0]), np.spacing(a_f16[1]))
        assert_equal(np.spacing(a_f16[1:]), a_f16[:-1]-a_f16[1:])

        assert_equal(np.nextafter(a_f16[0], hinf), -a_f16[1])
        assert_equal(np.nextafter(a_f16[1:], hinf), a_f16[:-1])
        assert_equal(np.nextafter(a_f16[:-1], -hinf), a_f16[1:])

        assert_equal(np.nextafter(hinf, a_f16), -a_f16[-1])
        assert_equal(np.nextafter(-hinf, a_f16), a_f16[-1])

        assert_equal(np.nextafter(a_f16, hnan), hnan[0])
        assert_equal(np.nextafter(hnan, a_f16), hnan[0])

    def test_half_ufuncs(self):
        """Test the various ufuncs"""

        a = np.array([0, 1, 2, 4, 2], dtype=float16)
        b = np.array([-2, 5, 1, 4, 3], dtype=float16)
        c = np.array([0, -1, -np.inf, np.nan, 6], dtype=float16)

        assert_equal(np.add(a, b), [-2, 6, 3, 8, 5])
        assert_equal(np.subtract(a, b), [2, -4, 1, 0, -1])
        assert_equal(np.multiply(a, b), [0, 5, 2, 16, 6])
        assert_equal(np.divide(a, b), [0, 0.199951171875, 2, 1, 0.66650390625])

        assert_equal(np.equal(a, b), [False, False, False, True, False])
        assert_equal(np.not_equal(a, b), [True, True, True, False, True])
        assert_equal(np.less(a, b), [False, True, False, False, True])
        assert_equal(np.less_equal(a, b), [False, True, False, True, True])
        assert_equal(np.greater(a, b), [True, False, True, False, False])
        assert_equal(np.greater_equal(a, b), [True, False, True, True, False])
        assert_equal(np.logical_and(a, b), [False, True, True, True, True])
        assert_equal(np.logical_or(a, b), [True, True, True, True, True])
        assert_equal(np.logical_xor(a, b), [True, False, False, False, False])
        assert_equal(np.logical_not(a), [True, False, False, False, False])

        assert_equal(np.isnan(c), [False, False, False, True, False])
        assert_equal(np.isinf(c), [False, False, True, False, False])
        assert_equal(np.isfinite(c), [True, True, False, False, True])
        assert_equal(np.signbit(b), [True, False, False, False, False])

        assert_equal(np.copysign(b, a), [2, 5, 1, 4, 3])

        assert_equal(np.maximum(a, b), [0, 5, 2, 4, 3])

        x = np.maximum(b, c)
        assert_(np.isnan(x[3]))
        x[3] = 0
        assert_equal(x, [0, 5, 1, 0, 6])

        assert_equal(np.minimum(a, b), [-2, 1, 1, 4, 2])

        x = np.minimum(b, c)
        assert_(np.isnan(x[3]))
        x[3] = 0
        assert_equal(x, [-2, -1, -np.inf, 0, 3])

        assert_equal(np.fmax(a, b), [0, 5, 2, 4, 3])
        assert_equal(np.fmax(b, c), [0, 5, 1, 4, 6])
        assert_equal(np.fmin(a, b), [-2, 1, 1, 4, 2])
        assert_equal(np.fmin(b, c), [-2, -1, -np.inf, 4, 3])

        assert_equal(np.floor_divide(a, b), [0, 0, 2, 1, 0])
        assert_equal(np.remainder(a, b), [0, 1, 0, 0, 2])
        assert_equal(np.divmod(a, b), ([0, 0, 2, 1, 0], [0, 1, 0, 0, 2]))
        assert_equal(np.square(b), [4, 25, 1, 16, 9])
        assert_equal(np.reciprocal(b), [-0.5, 0.199951171875, 1, 0.25, 0.333251953125])
        assert_equal(np.ones_like(b), [1, 1, 1, 1, 1])
        assert_equal(np.conjugate(b), b)
        assert_equal(np.absolute(b), [2, 5, 1, 4, 3])
        assert_equal(np.negative(b), [2, -5, -1, -4, -3])
        assert_equal(np.positive(b), b)
        assert_equal(np.sign(b), [-1, 1, 1, 1, 1])
        assert_equal(np.modf(b), ([0, 0, 0, 0, 0], b))
        assert_equal(np.frexp(b), ([-0.5, 0.625, 0.5, 0.5, 0.75], [2, 3, 1, 3, 2]))
        assert_equal(np.ldexp(b, [0, 1, 2, 4, 2]), [-2, 10, 4, 64, 12])

    @np._no_nep50_warning()
    def test_half_coercion(self, weak_promotion):
        """Test that half gets coerced properly with the other types"""
        a16 = np.array((1,), dtype=float16)
        a32 = np.array((1,), dtype=float32)
        b16 = float16(1)
        b32 = float32(1)

        assert np.power(a16, 2).dtype == float16
        assert np.power(a16, 2.0).dtype == float16
        assert np.power(a16, b16).dtype == float16
        expected_dt = float32 if weak_promotion else float16
        assert np.power(a16, b32).dtype == expected_dt
        assert np.power(a16, a16).dtype == float16
        assert np.power(a16, a32).dtype == float32

        expected_dt = float16 if weak_promotion else float64
        assert np.power(b16, 2).dtype == expected_dt
        assert np.power(b16, 2.0).dtype == expected_dt
        assert np.power(b16, b16).dtype, float16
        assert np.power(b16, b32).dtype, float32
        assert np.power(b16, a16).dtype, float16
        assert np.power(b16, a32).dtype, float32

        assert np.power(a32, a16).dtype == float32
        assert np.power(a32, b16).dtype == float32
        expected_dt = float32 if weak_promotion else float16
        assert np.power(b32, a16).dtype == expected_dt
        assert np.power(b32, b16).dtype == float32

    @pytest.mark.skipif(platform.machine() == "armv5tel",
                        reason="See gh-413.")
    @pytest.mark.skipif(IS_WASM,
                        reason="fp exceptions don't work in wasm.")
    def test_half_fpe(self):
        with np.errstate(all='raise'):
            sx16 = np.array((1e-4,), dtype=float16)
            bx16 = np.array((1e4,), dtype=float16)
            sy16 = float16(1e-4)
            by16 = float16(1e4)

            # Underflow errors
            assert_raises_fpe('underflow', lambda a, b:a*b, sx16, sx16)
            assert_raises_fpe('underflow', lambda a, b:a*b, sx16, sy16)
            assert_raises_fpe('underflow', lambda a, b:a*b, sy16, sx16)
            assert_raises_fpe('underflow', lambda a, b:a*b, sy16, sy16)
            assert_raises_fpe('underflow', lambda a, b:a/b, sx16, bx16)
            assert_raises_fpe('underflow', lambda a, b:a/b, sx16, by16)
            assert_raises_fpe('underflow', lambda a, b:a/b, sy16, bx16)
            assert_raises_fpe('underflow', lambda a, b:a/b, sy16, by16)
            assert_raises_fpe('underflow', lambda a, b:a/b,
                                             float16(2.**-14), float16(2**11))
            assert_raises_fpe('underflow', lambda a, b:a/b,
                                             float16(-2.**-14), float16(2**11))
            assert_raises_fpe('underflow', lambda a, b:a/b,
                                             float16(2.**-14+2**-24), float16(2))
            assert_raises_fpe('underflow', lambda a, b:a/b,
                                             float16(-2.**-14-2**-24), float16(2))
            assert_raises_fpe('underflow', lambda a, b:a/b,
                                             float16(2.**-14+2**-23), float16(4))

            # Overflow errors
            assert_raises_fpe('overflow', lambda a, b:a*b, bx16, bx16)
            assert_raises_fpe('overflow', lambda a, b:a*b, bx16, by16)
            assert_raises_fpe('overflow', lambda a, b:a*b, by16, bx16)
            assert_raises_fpe('overflow', lambda a, b:a*b, by16, by16)
            assert_raises_fpe('overflow', lambda a, b:a/b, bx16, sx16)
            assert_raises_fpe('overflow', lambda a, b:a/b, bx16, sy16)
            assert_raises_fpe('overflow', lambda a, b:a/b, by16, sx16)
            assert_raises_fpe('overflow', lambda a, b:a/b, by16, sy16)
            assert_raises_fpe('overflow', lambda a, b:a+b,
                                             float16(65504), float16(17))
            assert_raises_fpe('overflow', lambda a, b:a-b,
                                             float16(-65504), float16(17))
            assert_raises_fpe('overflow', np.nextafter, float16(65504), float16(np.inf))
            assert_raises_fpe('overflow', np.nextafter, float16(-65504), float16(-np.inf))
            assert_raises_fpe('overflow', np.spacing, float16(65504))

            # Invalid value errors
            assert_raises_fpe('invalid', np.divide, float16(np.inf), float16(np.inf))
            assert_raises_fpe('invalid', np.spacing, float16(np.inf))
            assert_raises_fpe('invalid', np.spacing, float16(np.nan))

            # These should not raise
            float16(65472)+float16(32)
            float16(2**-13)/float16(2)
            float16(2**-14)/float16(2**10)
            np.spacing(float16(-65504))
            np.nextafter(float16(65504), float16(-np.inf))
            np.nextafter(float16(-65504), float16(np.inf))
            np.nextafter(float16(np.inf), float16(0))
            np.nextafter(float16(-np.inf), float16(0))
            np.nextafter(float16(0), float16(np.nan))
            np.nextafter(float16(np.nan), float16(0))
            float16(2**-14)/float16(2**10)
            float16(-2**-14)/float16(2**10)
            float16(2**-14+2**-23)/float16(2)
            float16(-2**-14-2**-23)/float16(2)

    def test_half_array_interface(self):
        """Test that half is compatible with __array_interface__"""
        class Dummy:
            pass

        a = np.ones((1,), dtype=float16)
        b = Dummy()
        b.__array_interface__ = a.__array_interface__
        c = np.array(b)
        assert_(c.dtype == float16)
        assert_equal(a, c)
