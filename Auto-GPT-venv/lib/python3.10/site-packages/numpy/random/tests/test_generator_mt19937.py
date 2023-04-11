import sys
import hashlib

import pytest

import numpy as np
from numpy.linalg import LinAlgError
from numpy.testing import (
    assert_, assert_raises, assert_equal, assert_allclose,
    assert_warns, assert_no_warnings, assert_array_equal,
    assert_array_almost_equal, suppress_warnings, IS_WASM)

from numpy.random import Generator, MT19937, SeedSequence, RandomState

random = Generator(MT19937())

JUMP_TEST_DATA = [
    {
        "seed": 0,
        "steps": 10,
        "initial": {"key_sha256": "bb1636883c2707b51c5b7fc26c6927af4430f2e0785a8c7bc886337f919f9edf", "pos": 9},
        "jumped": {"key_sha256": "ff682ac12bb140f2d72fba8d3506cf4e46817a0db27aae1683867629031d8d55", "pos": 598},
    },
    {
        "seed":384908324,
        "steps":312,
        "initial": {"key_sha256": "16b791a1e04886ccbbb4d448d6ff791267dc458ae599475d08d5cced29d11614", "pos": 311},
        "jumped": {"key_sha256": "a0110a2cf23b56be0feaed8f787a7fc84bef0cb5623003d75b26bdfa1c18002c", "pos": 276},
    },
    {
        "seed": [839438204, 980239840, 859048019, 821],
        "steps": 511,
        "initial": {"key_sha256": "d306cf01314d51bd37892d874308200951a35265ede54d200f1e065004c3e9ea", "pos": 510},
        "jumped": {"key_sha256": "0e00ab449f01a5195a83b4aee0dfbc2ce8d46466a640b92e33977d2e42f777f8", "pos": 475},
    },
]

@pytest.fixture(scope='module', params=[True, False])
def endpoint(request):
    return request.param


class TestSeed:
    def test_scalar(self):
        s = Generator(MT19937(0))
        assert_equal(s.integers(1000), 479)
        s = Generator(MT19937(4294967295))
        assert_equal(s.integers(1000), 324)

    def test_array(self):
        s = Generator(MT19937(range(10)))
        assert_equal(s.integers(1000), 465)
        s = Generator(MT19937(np.arange(10)))
        assert_equal(s.integers(1000), 465)
        s = Generator(MT19937([0]))
        assert_equal(s.integers(1000), 479)
        s = Generator(MT19937([4294967295]))
        assert_equal(s.integers(1000), 324)

    def test_seedsequence(self):
        s = MT19937(SeedSequence(0))
        assert_equal(s.random_raw(1), 2058676884)

    def test_invalid_scalar(self):
        # seed must be an unsigned 32 bit integer
        assert_raises(TypeError, MT19937, -0.5)
        assert_raises(ValueError, MT19937, -1)

    def test_invalid_array(self):
        # seed must be an unsigned integer
        assert_raises(TypeError, MT19937, [-0.5])
        assert_raises(ValueError, MT19937, [-1])
        assert_raises(ValueError, MT19937, [1, -2, 4294967296])

    def test_noninstantized_bitgen(self):
        assert_raises(ValueError, Generator, MT19937)


class TestBinomial:
    def test_n_zero(self):
        # Tests the corner case of n == 0 for the binomial distribution.
        # binomial(0, p) should be zero for any p in [0, 1].
        # This test addresses issue #3480.
        zeros = np.zeros(2, dtype='int')
        for p in [0, .5, 1]:
            assert_(random.binomial(0, p) == 0)
            assert_array_equal(random.binomial(zeros, p), zeros)

    def test_p_is_nan(self):
        # Issue #4571.
        assert_raises(ValueError, random.binomial, 1, np.nan)


class TestMultinomial:
    def test_basic(self):
        random.multinomial(100, [0.2, 0.8])

    def test_zero_probability(self):
        random.multinomial(100, [0.2, 0.8, 0.0, 0.0, 0.0])

    def test_int_negative_interval(self):
        assert_(-5 <= random.integers(-5, -1) < -1)
        x = random.integers(-5, -1, 5)
        assert_(np.all(-5 <= x))
        assert_(np.all(x < -1))

    def test_size(self):
        # gh-3173
        p = [0.5, 0.5]
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.multinomial(1, p, [2, 2]).shape, (2, 2, 2))
        assert_equal(random.multinomial(1, p, (2, 2)).shape, (2, 2, 2))
        assert_equal(random.multinomial(1, p, np.array((2, 2))).shape,
                     (2, 2, 2))

        assert_raises(TypeError, random.multinomial, 1, p,
                      float(1))

    def test_invalid_prob(self):
        assert_raises(ValueError, random.multinomial, 100, [1.1, 0.2])
        assert_raises(ValueError, random.multinomial, 100, [-.1, 0.9])

    def test_invalid_n(self):
        assert_raises(ValueError, random.multinomial, -1, [0.8, 0.2])
        assert_raises(ValueError, random.multinomial, [-1] * 10, [0.8, 0.2])

    def test_p_non_contiguous(self):
        p = np.arange(15.)
        p /= np.sum(p[1::3])
        pvals = p[1::3]
        random = Generator(MT19937(1432985819))
        non_contig = random.multinomial(100, pvals=pvals)
        random = Generator(MT19937(1432985819))
        contig = random.multinomial(100, pvals=np.ascontiguousarray(pvals))
        assert_array_equal(non_contig, contig)

    def test_multinomial_pvals_float32(self):
        x = np.array([9.9e-01, 9.9e-01, 1.0e-09, 1.0e-09, 1.0e-09, 1.0e-09,
                      1.0e-09, 1.0e-09, 1.0e-09, 1.0e-09], dtype=np.float32)
        pvals = x / x.sum()
        random = Generator(MT19937(1432985819))
        match = r"[\w\s]*pvals array is cast to 64-bit floating"
        with pytest.raises(ValueError, match=match):
            random.multinomial(1, pvals)

class TestMultivariateHypergeometric:

    def setup_method(self):
        self.seed = 8675309

    def test_argument_validation(self):
        # Error cases...

        # `colors` must be a 1-d sequence
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      10, 4)

        # Negative nsample
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [2, 3, 4], -1)

        # Negative color
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [-1, 2, 3], 2)

        # nsample exceeds sum(colors)
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [2, 3, 4], 10)

        # nsample exceeds sum(colors) (edge case of empty colors)
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [], 1)

        # Validation errors associated with very large values in colors.
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [999999999, 101], 5, 1, 'marginals')

        int64_info = np.iinfo(np.int64)
        max_int64 = int64_info.max
        max_int64_index = max_int64 // int64_info.dtype.itemsize
        assert_raises(ValueError, random.multivariate_hypergeometric,
                      [max_int64_index - 100, 101], 5, 1, 'count')

    @pytest.mark.parametrize('method', ['count', 'marginals'])
    def test_edge_cases(self, method):
        # Set the seed, but in fact, all the results in this test are
        # deterministic, so we don't really need this.
        random = Generator(MT19937(self.seed))

        x = random.multivariate_hypergeometric([0, 0, 0], 0, method=method)
        assert_array_equal(x, [0, 0, 0])

        x = random.multivariate_hypergeometric([], 0, method=method)
        assert_array_equal(x, [])

        x = random.multivariate_hypergeometric([], 0, size=1, method=method)
        assert_array_equal(x, np.empty((1, 0), dtype=np.int64))

        x = random.multivariate_hypergeometric([1, 2, 3], 0, method=method)
        assert_array_equal(x, [0, 0, 0])

        x = random.multivariate_hypergeometric([9, 0, 0], 3, method=method)
        assert_array_equal(x, [3, 0, 0])

        colors = [1, 1, 0, 1, 1]
        x = random.multivariate_hypergeometric(colors, sum(colors),
                                               method=method)
        assert_array_equal(x, colors)

        x = random.multivariate_hypergeometric([3, 4, 5], 12, size=3,
                                               method=method)
        assert_array_equal(x, [[3, 4, 5]]*3)

    # Cases for nsample:
    #     nsample < 10
    #     10 <= nsample < colors.sum()/2
    #     colors.sum()/2 < nsample < colors.sum() - 10
    #     colors.sum() - 10 < nsample < colors.sum()
    @pytest.mark.parametrize('nsample', [8, 25, 45, 55])
    @pytest.mark.parametrize('method', ['count', 'marginals'])
    @pytest.mark.parametrize('size', [5, (2, 3), 150000])
    def test_typical_cases(self, nsample, method, size):
        random = Generator(MT19937(self.seed))

        colors = np.array([10, 5, 20, 25])
        sample = random.multivariate_hypergeometric(colors, nsample, size,
                                                    method=method)
        if isinstance(size, int):
            expected_shape = (size,) + colors.shape
        else:
            expected_shape = size + colors.shape
        assert_equal(sample.shape, expected_shape)
        assert_((sample >= 0).all())
        assert_((sample <= colors).all())
        assert_array_equal(sample.sum(axis=-1),
                           np.full(size, fill_value=nsample, dtype=int))
        if isinstance(size, int) and size >= 100000:
            # This sample is large enough to compare its mean to
            # the expected values.
            assert_allclose(sample.mean(axis=0),
                            nsample * colors / colors.sum(),
                            rtol=1e-3, atol=0.005)

    def test_repeatability1(self):
        random = Generator(MT19937(self.seed))
        sample = random.multivariate_hypergeometric([3, 4, 5], 5, size=5,
                                                    method='count')
        expected = np.array([[2, 1, 2],
                             [2, 1, 2],
                             [1, 1, 3],
                             [2, 0, 3],
                             [2, 1, 2]])
        assert_array_equal(sample, expected)

    def test_repeatability2(self):
        random = Generator(MT19937(self.seed))
        sample = random.multivariate_hypergeometric([20, 30, 50], 50,
                                                    size=5,
                                                    method='marginals')
        expected = np.array([[ 9, 17, 24],
                             [ 7, 13, 30],
                             [ 9, 15, 26],
                             [ 9, 17, 24],
                             [12, 14, 24]])
        assert_array_equal(sample, expected)

    def test_repeatability3(self):
        random = Generator(MT19937(self.seed))
        sample = random.multivariate_hypergeometric([20, 30, 50], 12,
                                                    size=5,
                                                    method='marginals')
        expected = np.array([[2, 3, 7],
                             [5, 3, 4],
                             [2, 5, 5],
                             [5, 3, 4],
                             [1, 5, 6]])
        assert_array_equal(sample, expected)


class TestSetState:
    def setup_method(self):
        self.seed = 1234567890
        self.rg = Generator(MT19937(self.seed))
        self.bit_generator = self.rg.bit_generator
        self.state = self.bit_generator.state
        self.legacy_state = (self.state['bit_generator'],
                             self.state['state']['key'],
                             self.state['state']['pos'])

    def test_gaussian_reset(self):
        # Make sure the cached every-other-Gaussian is reset.
        old = self.rg.standard_normal(size=3)
        self.bit_generator.state = self.state
        new = self.rg.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_gaussian_reset_in_media_res(self):
        # When the state is saved with a cached Gaussian, make sure the
        # cached Gaussian is restored.

        self.rg.standard_normal()
        state = self.bit_generator.state
        old = self.rg.standard_normal(size=3)
        self.bit_generator.state = state
        new = self.rg.standard_normal(size=3)
        assert_(np.all(old == new))

    def test_negative_binomial(self):
        # Ensure that the negative binomial results take floating point
        # arguments without truncation.
        self.rg.negative_binomial(0.5, 0.5)


class TestIntegers:
    rfunc = random.integers

    # valid integer/boolean types
    itype = [bool, np.int8, np.uint8, np.int16, np.uint16,
             np.int32, np.uint32, np.int64, np.uint64]

    def test_unsupported_type(self, endpoint):
        assert_raises(TypeError, self.rfunc, 1, endpoint=endpoint, dtype=float)

    def test_bounds_checking(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            assert_raises(ValueError, self.rfunc, lbnd - 1, ubnd,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, lbnd, ubnd + 1,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, lbnd,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, 0, endpoint=endpoint,
                          dtype=dt)

            assert_raises(ValueError, self.rfunc, [lbnd - 1], ubnd,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [lbnd], [ubnd + 1],
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [ubnd], [lbnd],
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, 1, [0],
                          endpoint=endpoint, dtype=dt)

    def test_bounds_checking_array(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + (not endpoint)

            assert_raises(ValueError, self.rfunc, [lbnd - 1] * 2, [ubnd] * 2,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [lbnd] * 2,
                          [ubnd + 1] * 2, endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, ubnd, [lbnd] * 2,
                          endpoint=endpoint, dtype=dt)
            assert_raises(ValueError, self.rfunc, [1] * 2, 0,
                          endpoint=endpoint, dtype=dt)

    def test_rng_zero_and_extremes(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            is_open = not endpoint

            tgt = ubnd - 1
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt], tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)

            tgt = lbnd
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc(tgt, [tgt + is_open], size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)

            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc(tgt, tgt + is_open, size=1000,
                                    endpoint=endpoint, dtype=dt), tgt)
            assert_equal(self.rfunc([tgt], [tgt + is_open],
                                    size=1000, endpoint=endpoint, dtype=dt),
                         tgt)

    def test_rng_zero_and_extremes_array(self, endpoint):
        size = 1000
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            tgt = ubnd - 1
            assert_equal(self.rfunc([tgt], [tgt + 1],
                                    size=size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)

            tgt = lbnd
            assert_equal(self.rfunc([tgt], [tgt + 1],
                                    size=size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)

            tgt = (lbnd + ubnd) // 2
            assert_equal(self.rfunc([tgt], [tgt + 1],
                                    size=size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, dtype=dt), tgt)
            assert_equal(self.rfunc(
                [tgt] * size, [tgt + 1] * size, size=size, dtype=dt), tgt)

    def test_full_range(self, endpoint):
        # Test for ticket #1690

        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            try:
                self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            except Exception as e:
                raise AssertionError("No error should have been raised, "
                                     "but one was with the following "
                                     "message:\n\n%s" % str(e))

    def test_full_range_array(self, endpoint):
        # Test for ticket #1690

        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            try:
                self.rfunc([lbnd] * 2, [ubnd], endpoint=endpoint, dtype=dt)
            except Exception as e:
                raise AssertionError("No error should have been raised, "
                                     "but one was with the following "
                                     "message:\n\n%s" % str(e))

    def test_in_bounds_fuzz(self, endpoint):
        # Don't use fixed seed
        random = Generator(MT19937())

        for dt in self.itype[1:]:
            for ubnd in [4, 8, 16]:
                vals = self.rfunc(2, ubnd - endpoint, size=2 ** 16,
                                  endpoint=endpoint, dtype=dt)
                assert_(vals.max() < ubnd)
                assert_(vals.min() >= 2)

        vals = self.rfunc(0, 2 - endpoint, size=2 ** 16, endpoint=endpoint,
                          dtype=bool)
        assert_(vals.max() < 2)
        assert_(vals.min() >= 0)

    def test_scalar_array_equiv(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            size = 1000
            random = Generator(MT19937(1234))
            scalar = random.integers(lbnd, ubnd, size=size, endpoint=endpoint,
                                dtype=dt)

            random = Generator(MT19937(1234))
            scalar_array = random.integers([lbnd], [ubnd], size=size,
                                      endpoint=endpoint, dtype=dt)

            random = Generator(MT19937(1234))
            array = random.integers([lbnd] * size, [ubnd] *
                               size, size=size, endpoint=endpoint, dtype=dt)
            assert_array_equal(scalar, scalar_array)
            assert_array_equal(scalar, array)

    def test_repeatability(self, endpoint):
        # We use a sha256 hash of generated sequences of 1000 samples
        # in the range [0, 6) for all but bool, where the range
        # is [0, 2). Hashes are for little endian numbers.
        tgt = {'bool':   '053594a9b82d656f967c54869bc6970aa0358cf94ad469c81478459c6a90eee3',
               'int16':  '54de9072b6ee9ff7f20b58329556a46a447a8a29d67db51201bf88baa6e4e5d4',
               'int32':  'd3a0d5efb04542b25ac712e50d21f39ac30f312a5052e9bbb1ad3baa791ac84b',
               'int64':  '14e224389ac4580bfbdccb5697d6190b496f91227cf67df60989de3d546389b1',
               'int8':   '0e203226ff3fbbd1580f15da4621e5f7164d0d8d6b51696dd42d004ece2cbec1',
               'uint16': '54de9072b6ee9ff7f20b58329556a46a447a8a29d67db51201bf88baa6e4e5d4',
               'uint32': 'd3a0d5efb04542b25ac712e50d21f39ac30f312a5052e9bbb1ad3baa791ac84b',
               'uint64': '14e224389ac4580bfbdccb5697d6190b496f91227cf67df60989de3d546389b1',
               'uint8':  '0e203226ff3fbbd1580f15da4621e5f7164d0d8d6b51696dd42d004ece2cbec1'}

        for dt in self.itype[1:]:
            random = Generator(MT19937(1234))

            # view as little endian for hash
            if sys.byteorder == 'little':
                val = random.integers(0, 6 - endpoint, size=1000, endpoint=endpoint,
                                 dtype=dt)
            else:
                val = random.integers(0, 6 - endpoint, size=1000, endpoint=endpoint,
                                 dtype=dt).byteswap()

            res = hashlib.sha256(val).hexdigest()
            assert_(tgt[np.dtype(dt).name] == res)

        # bools do not depend on endianness
        random = Generator(MT19937(1234))
        val = random.integers(0, 2 - endpoint, size=1000, endpoint=endpoint,
                         dtype=bool).view(np.int8)
        res = hashlib.sha256(val).hexdigest()
        assert_(tgt[np.dtype(bool).name] == res)

    def test_repeatability_broadcasting(self, endpoint):
        for dt in self.itype:
            lbnd = 0 if dt in (bool, np.bool_) else np.iinfo(dt).min
            ubnd = 2 if dt in (bool, np.bool_) else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            # view as little endian for hash
            random = Generator(MT19937(1234))
            val = random.integers(lbnd, ubnd, size=1000, endpoint=endpoint,
                             dtype=dt)

            random = Generator(MT19937(1234))
            val_bc = random.integers([lbnd] * 1000, ubnd, endpoint=endpoint,
                                dtype=dt)

            assert_array_equal(val, val_bc)

            random = Generator(MT19937(1234))
            val_bc = random.integers([lbnd] * 1000, [ubnd] * 1000,
                                endpoint=endpoint, dtype=dt)

            assert_array_equal(val, val_bc)

    @pytest.mark.parametrize(
        'bound, expected',
        [(2**32 - 1, np.array([517043486, 1364798665, 1733884389, 1353720612,
                               3769704066, 1170797179, 4108474671])),
         (2**32, np.array([517043487, 1364798666, 1733884390, 1353720613,
                           3769704067, 1170797180, 4108474672])),
         (2**32 + 1, np.array([517043487, 1733884390, 3769704068, 4108474673,
                               1831631863, 1215661561, 3869512430]))]
    )
    def test_repeatability_32bit_boundary(self, bound, expected):
        for size in [None, len(expected)]:
            random = Generator(MT19937(1234))
            x = random.integers(bound, size=size)
            assert_equal(x, expected if size is not None else expected[0])

    def test_repeatability_32bit_boundary_broadcasting(self):
        desired = np.array([[[1622936284, 3620788691, 1659384060],
                             [1417365545,  760222891, 1909653332],
                             [3788118662,  660249498, 4092002593]],
                            [[3625610153, 2979601262, 3844162757],
                             [ 685800658,  120261497, 2694012896],
                             [1207779440, 1586594375, 3854335050]],
                            [[3004074748, 2310761796, 3012642217],
                             [2067714190, 2786677879, 1363865881],
                             [ 791663441, 1867303284, 2169727960]],
                            [[1939603804, 1250951100,  298950036],
                             [1040128489, 3791912209, 3317053765],
                             [3155528714,   61360675, 2305155588]],
                            [[ 817688762, 1335621943, 3288952434],
                             [1770890872, 1102951817, 1957607470],
                             [3099996017,  798043451,   48334215]]])
        for size in [None, (5, 3, 3)]:
            random = Generator(MT19937(12345))
            x = random.integers([[-1], [0], [1]],
                                [2**32 - 1, 2**32, 2**32 + 1],
                                size=size)
            assert_array_equal(x, desired if size is not None else desired[0])

    def test_int64_uint64_broadcast_exceptions(self, endpoint):
        configs = {np.uint64: ((0, 2**65), (-1, 2**62), (10, 9), (0, 0)),
                   np.int64: ((0, 2**64), (-(2**64), 2**62), (10, 9), (0, 0),
                              (-2**63-1, -2**63-1))}
        for dtype in configs:
            for config in configs[dtype]:
                low, high = config
                high = high - endpoint
                low_a = np.array([[low]*10])
                high_a = np.array([high] * 10)
                assert_raises(ValueError, random.integers, low, high,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_a, high,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low, high_a,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_a, high_a,
                              endpoint=endpoint, dtype=dtype)

                low_o = np.array([[low]*10], dtype=object)
                high_o = np.array([high] * 10, dtype=object)
                assert_raises(ValueError, random.integers, low_o, high,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low, high_o,
                              endpoint=endpoint, dtype=dtype)
                assert_raises(ValueError, random.integers, low_o, high_o,
                              endpoint=endpoint, dtype=dtype)

    def test_int64_uint64_corner_case(self, endpoint):
        # When stored in Numpy arrays, `lbnd` is casted
        # as np.int64, and `ubnd` is casted as np.uint64.
        # Checking whether `lbnd` >= `ubnd` used to be
        # done solely via direct comparison, which is incorrect
        # because when Numpy tries to compare both numbers,
        # it casts both to np.float64 because there is
        # no integer superset of np.int64 and np.uint64. However,
        # `ubnd` is too large to be represented in np.float64,
        # causing it be round down to np.iinfo(np.int64).max,
        # leading to a ValueError because `lbnd` now equals
        # the new `ubnd`.

        dt = np.int64
        tgt = np.iinfo(np.int64).max
        lbnd = np.int64(np.iinfo(np.int64).max)
        ubnd = np.uint64(np.iinfo(np.int64).max + 1 - endpoint)

        # None of these function calls should
        # generate a ValueError now.
        actual = random.integers(lbnd, ubnd, endpoint=endpoint, dtype=dt)
        assert_equal(actual, tgt)

    def test_respect_dtype_singleton(self, endpoint):
        # See gh-7203
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            dt = np.bool_ if dt is bool else dt

            sample = self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            assert_equal(sample.dtype, dt)

        for dt in (bool, int, np.compat.long):
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd

            # gh-7284: Ensure that we get Python data types
            sample = self.rfunc(lbnd, ubnd, endpoint=endpoint, dtype=dt)
            assert not hasattr(sample, 'dtype')
            assert_equal(type(sample), dt)

    def test_respect_dtype_array(self, endpoint):
        # See gh-7203
        for dt in self.itype:
            lbnd = 0 if dt is bool else np.iinfo(dt).min
            ubnd = 2 if dt is bool else np.iinfo(dt).max + 1
            ubnd = ubnd - 1 if endpoint else ubnd
            dt = np.bool_ if dt is bool else dt

            sample = self.rfunc([lbnd], [ubnd], endpoint=endpoint, dtype=dt)
            assert_equal(sample.dtype, dt)
            sample = self.rfunc([lbnd] * 2, [ubnd] * 2, endpoint=endpoint,
                                dtype=dt)
            assert_equal(sample.dtype, dt)

    def test_zero_size(self, endpoint):
        # See gh-7203
        for dt in self.itype:
            sample = self.rfunc(0, 0, (3, 0, 4), endpoint=endpoint, dtype=dt)
            assert sample.shape == (3, 0, 4)
            assert sample.dtype == dt
            assert self.rfunc(0, -10, 0, endpoint=endpoint,
                              dtype=dt).shape == (0,)
            assert_equal(random.integers(0, 0, size=(3, 0, 4)).shape,
                         (3, 0, 4))
            assert_equal(random.integers(0, -10, size=0).shape, (0,))
            assert_equal(random.integers(10, 10, size=0).shape, (0,))

    def test_error_byteorder(self):
        other_byteord_dt = '<i4' if sys.byteorder == 'big' else '>i4'
        with pytest.raises(ValueError):
            random.integers(0, 200, size=10, dtype=other_byteord_dt)

    # chi2max is the maximum acceptable chi-squared value.
    @pytest.mark.slow
    @pytest.mark.parametrize('sample_size,high,dtype,chi2max',
        [(5000000, 5, np.int8, 125.0),          # p-value ~4.6e-25
         (5000000, 7, np.uint8, 150.0),         # p-value ~7.7e-30
         (10000000, 2500, np.int16, 3300.0),    # p-value ~3.0e-25
         (50000000, 5000, np.uint16, 6500.0),   # p-value ~3.5e-25
        ])
    def test_integers_small_dtype_chisquared(self, sample_size, high,
                                             dtype, chi2max):
        # Regression test for gh-14774.
        samples = random.integers(high, size=sample_size, dtype=dtype)

        values, counts = np.unique(samples, return_counts=True)
        expected = sample_size / high
        chi2 = ((counts - expected)**2 / expected).sum()
        assert chi2 < chi2max


class TestRandomDist:
    # Make sure the random distribution returns the correct value for a
    # given seed

    def setup_method(self):
        self.seed = 1234567890

    def test_integers(self):
        random = Generator(MT19937(self.seed))
        actual = random.integers(-99, 99, size=(3, 2))
        desired = np.array([[-80, -56], [41, 37], [-83, -16]])
        assert_array_equal(actual, desired)

    def test_integers_masked(self):
        # Test masked rejection sampling algorithm to generate array of
        # uint32 in an interval.
        random = Generator(MT19937(self.seed))
        actual = random.integers(0, 99, size=(3, 2), dtype=np.uint32)
        desired = np.array([[9, 21], [70, 68], [8, 41]], dtype=np.uint32)
        assert_array_equal(actual, desired)

    def test_integers_closed(self):
        random = Generator(MT19937(self.seed))
        actual = random.integers(-99, 99, size=(3, 2), endpoint=True)
        desired = np.array([[-80, -56], [ 41, 38], [-83, -15]])
        assert_array_equal(actual, desired)

    def test_integers_max_int(self):
        # Tests whether integers with closed=True can generate the
        # maximum allowed Python int that can be converted
        # into a C long. Previous implementations of this
        # method have thrown an OverflowError when attempting
        # to generate this integer.
        actual = random.integers(np.iinfo('l').max, np.iinfo('l').max,
                                 endpoint=True)

        desired = np.iinfo('l').max
        assert_equal(actual, desired)

    def test_random(self):
        random = Generator(MT19937(self.seed))
        actual = random.random((3, 2))
        desired = np.array([[0.096999199829214, 0.707517457682192],
                            [0.084364834598269, 0.767731206553125],
                            [0.665069021359413, 0.715487190596693]])
        assert_array_almost_equal(actual, desired, decimal=15)

        random = Generator(MT19937(self.seed))
        actual = random.random()
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_random_float(self):
        random = Generator(MT19937(self.seed))
        actual = random.random((3, 2))
        desired = np.array([[0.0969992 , 0.70751746],
                            [0.08436483, 0.76773121],
                            [0.66506902, 0.71548719]])
        assert_array_almost_equal(actual, desired, decimal=7)

    def test_random_float_scalar(self):
        random = Generator(MT19937(self.seed))
        actual = random.random(dtype=np.float32)
        desired = 0.0969992
        assert_array_almost_equal(actual, desired, decimal=7)

    @pytest.mark.parametrize('dtype, uint_view_type',
                             [(np.float32, np.uint32),
                              (np.float64, np.uint64)])
    def test_random_distribution_of_lsb(self, dtype, uint_view_type):
        random = Generator(MT19937(self.seed))
        sample = random.random(100000, dtype=dtype)
        num_ones_in_lsb = np.count_nonzero(sample.view(uint_view_type) & 1)
        # The probability of a 1 in the least significant bit is 0.25.
        # With a sample size of 100000, the probability that num_ones_in_lsb
        # is outside the following range is less than 5e-11.
        assert 24100 < num_ones_in_lsb < 25900

    def test_random_unsupported_type(self):
        assert_raises(TypeError, random.random, dtype='int32')

    def test_choice_uniform_replace(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice(4, 4)
        desired = np.array([0, 0, 2, 2], dtype=np.int64)
        assert_array_equal(actual, desired)

    def test_choice_nonuniform_replace(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice(4, 4, p=[0.4, 0.4, 0.1, 0.1])
        desired = np.array([0, 1, 0, 1], dtype=np.int64)
        assert_array_equal(actual, desired)

    def test_choice_uniform_noreplace(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice(4, 3, replace=False)
        desired = np.array([2, 0, 3], dtype=np.int64)
        assert_array_equal(actual, desired)
        actual = random.choice(4, 4, replace=False, shuffle=False)
        desired = np.arange(4, dtype=np.int64)
        assert_array_equal(actual, desired)

    def test_choice_nonuniform_noreplace(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice(4, 3, replace=False, p=[0.1, 0.3, 0.5, 0.1])
        desired = np.array([0, 2, 3], dtype=np.int64)
        assert_array_equal(actual, desired)

    def test_choice_noninteger(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice(['a', 'b', 'c', 'd'], 4)
        desired = np.array(['a', 'a', 'c', 'c'])
        assert_array_equal(actual, desired)

    def test_choice_multidimensional_default_axis(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice([[0, 1], [2, 3], [4, 5], [6, 7]], 3)
        desired = np.array([[0, 1], [0, 1], [4, 5]])
        assert_array_equal(actual, desired)

    def test_choice_multidimensional_custom_axis(self):
        random = Generator(MT19937(self.seed))
        actual = random.choice([[0, 1], [2, 3], [4, 5], [6, 7]], 1, axis=1)
        desired = np.array([[0], [2], [4], [6]])
        assert_array_equal(actual, desired)

    def test_choice_exceptions(self):
        sample = random.choice
        assert_raises(ValueError, sample, -1, 3)
        assert_raises(ValueError, sample, 3., 3)
        assert_raises(ValueError, sample, [], 3)
        assert_raises(ValueError, sample, [1, 2, 3, 4], 3,
                      p=[[0.25, 0.25], [0.25, 0.25]])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4, 0.2])
        assert_raises(ValueError, sample, [1, 2], 3, p=[1.1, -0.1])
        assert_raises(ValueError, sample, [1, 2], 3, p=[0.4, 0.4])
        assert_raises(ValueError, sample, [1, 2, 3], 4, replace=False)
        # gh-13087
        assert_raises(ValueError, sample, [1, 2, 3], -2, replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], (-1,), replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], (-1, 1), replace=False)
        assert_raises(ValueError, sample, [1, 2, 3], 2,
                      replace=False, p=[1, 0, 0])

    def test_choice_return_shape(self):
        p = [0.1, 0.9]
        # Check scalar
        assert_(np.isscalar(random.choice(2, replace=True)))
        assert_(np.isscalar(random.choice(2, replace=False)))
        assert_(np.isscalar(random.choice(2, replace=True, p=p)))
        assert_(np.isscalar(random.choice(2, replace=False, p=p)))
        assert_(np.isscalar(random.choice([1, 2], replace=True)))
        assert_(random.choice([None], replace=True) is None)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(random.choice(arr, replace=True) is a)

        # Check 0-d array
        s = tuple()
        assert_(not np.isscalar(random.choice(2, s, replace=True)))
        assert_(not np.isscalar(random.choice(2, s, replace=False)))
        assert_(not np.isscalar(random.choice(2, s, replace=True, p=p)))
        assert_(not np.isscalar(random.choice(2, s, replace=False, p=p)))
        assert_(not np.isscalar(random.choice([1, 2], s, replace=True)))
        assert_(random.choice([None], s, replace=True).ndim == 0)
        a = np.array([1, 2])
        arr = np.empty(1, dtype=object)
        arr[0] = a
        assert_(random.choice(arr, s, replace=True).item() is a)

        # Check multi dimensional array
        s = (2, 3)
        p = [0.1, 0.1, 0.1, 0.1, 0.4, 0.2]
        assert_equal(random.choice(6, s, replace=True).shape, s)
        assert_equal(random.choice(6, s, replace=False).shape, s)
        assert_equal(random.choice(6, s, replace=True, p=p).shape, s)
        assert_equal(random.choice(6, s, replace=False, p=p).shape, s)
        assert_equal(random.choice(np.arange(6), s, replace=True).shape, s)

        # Check zero-size
        assert_equal(random.integers(0, 0, size=(3, 0, 4)).shape, (3, 0, 4))
        assert_equal(random.integers(0, -10, size=0).shape, (0,))
        assert_equal(random.integers(10, 10, size=0).shape, (0,))
        assert_equal(random.choice(0, size=0).shape, (0,))
        assert_equal(random.choice([], size=(0,)).shape, (0,))
        assert_equal(random.choice(['a', 'b'], size=(3, 0, 4)).shape,
                     (3, 0, 4))
        assert_raises(ValueError, random.choice, [], 10)

    def test_choice_nan_probabilities(self):
        a = np.array([42, 1, 2])
        p = [None, None, None]
        assert_raises(ValueError, random.choice, a, p=p)

    def test_choice_p_non_contiguous(self):
        p = np.ones(10) / 5
        p[1::2] = 3.0
        random = Generator(MT19937(self.seed))
        non_contig = random.choice(5, 3, p=p[::2])
        random = Generator(MT19937(self.seed))
        contig = random.choice(5, 3, p=np.ascontiguousarray(p[::2]))
        assert_array_equal(non_contig, contig)

    def test_choice_return_type(self):
        # gh 9867
        p = np.ones(4) / 4.
        actual = random.choice(4, 2)
        assert actual.dtype == np.int64
        actual = random.choice(4, 2, replace=False)
        assert actual.dtype == np.int64
        actual = random.choice(4, 2, p=p)
        assert actual.dtype == np.int64
        actual = random.choice(4, 2, p=p, replace=False)
        assert actual.dtype == np.int64

    def test_choice_large_sample(self):
        choice_hash = '4266599d12bfcfb815213303432341c06b4349f5455890446578877bb322e222'
        random = Generator(MT19937(self.seed))
        actual = random.choice(10000, 5000, replace=False)
        if sys.byteorder != 'little':
            actual = actual.byteswap()
        res = hashlib.sha256(actual.view(np.int8)).hexdigest()
        assert_(choice_hash == res)

    def test_bytes(self):
        random = Generator(MT19937(self.seed))
        actual = random.bytes(10)
        desired = b'\x86\xf0\xd4\x18\xe1\x81\t8%\xdd'
        assert_equal(actual, desired)

    def test_shuffle(self):
        # Test lists, arrays (of various dtypes), and multidimensional versions
        # of both, c-contiguous or not:
        for conv in [lambda x: np.array([]),
                     lambda x: x,
                     lambda x: np.asarray(x).astype(np.int8),
                     lambda x: np.asarray(x).astype(np.float32),
                     lambda x: np.asarray(x).astype(np.complex64),
                     lambda x: np.asarray(x).astype(object),
                     lambda x: [(i, i) for i in x],
                     lambda x: np.asarray([[i, i] for i in x]),
                     lambda x: np.vstack([x, x]).T,
                     # gh-11442
                     lambda x: (np.asarray([(i, i) for i in x],
                                           [("a", int), ("b", int)])
                                .view(np.recarray)),
                     # gh-4270
                     lambda x: np.asarray([(i, i) for i in x],
                                          [("a", object, (1,)),
                                           ("b", np.int32, (1,))])]:
            random = Generator(MT19937(self.seed))
            alist = conv([1, 2, 3, 4, 5, 6, 7, 8, 9, 0])
            random.shuffle(alist)
            actual = alist
            desired = conv([4, 1, 9, 8, 0, 5, 3, 6, 2, 7])
            assert_array_equal(actual, desired)

    def test_shuffle_custom_axis(self):
        random = Generator(MT19937(self.seed))
        actual = np.arange(16).reshape((4, 4))
        random.shuffle(actual, axis=1)
        desired = np.array([[ 0,  3,  1,  2],
                            [ 4,  7,  5,  6],
                            [ 8, 11,  9, 10],
                            [12, 15, 13, 14]])
        assert_array_equal(actual, desired)
        random = Generator(MT19937(self.seed))
        actual = np.arange(16).reshape((4, 4))
        random.shuffle(actual, axis=-1)
        assert_array_equal(actual, desired)

    def test_shuffle_custom_axis_empty(self):
        random = Generator(MT19937(self.seed))
        desired = np.array([]).reshape((0, 6))
        for axis in (0, 1):
            actual = np.array([]).reshape((0, 6))
            random.shuffle(actual, axis=axis)
            assert_array_equal(actual, desired)

    def test_shuffle_axis_nonsquare(self):
        y1 = np.arange(20).reshape(2, 10)
        y2 = y1.copy()
        random = Generator(MT19937(self.seed))
        random.shuffle(y1, axis=1)
        random = Generator(MT19937(self.seed))
        random.shuffle(y2.T)
        assert_array_equal(y1, y2)

    def test_shuffle_masked(self):
        # gh-3263
        a = np.ma.masked_values(np.reshape(range(20), (5, 4)) % 3 - 1, -1)
        b = np.ma.masked_values(np.arange(20) % 3 - 1, -1)
        a_orig = a.copy()
        b_orig = b.copy()
        for i in range(50):
            random.shuffle(a)
            assert_equal(
                sorted(a.data[~a.mask]), sorted(a_orig.data[~a_orig.mask]))
            random.shuffle(b)
            assert_equal(
                sorted(b.data[~b.mask]), sorted(b_orig.data[~b_orig.mask]))

    def test_shuffle_exceptions(self):
        random = Generator(MT19937(self.seed))
        arr = np.arange(10)
        assert_raises(np.AxisError, random.shuffle, arr, 1)
        arr = np.arange(9).reshape((3, 3))
        assert_raises(np.AxisError, random.shuffle, arr, 3)
        assert_raises(TypeError, random.shuffle, arr, slice(1, 2, None))
        arr = [[1, 2, 3], [4, 5, 6]]
        assert_raises(NotImplementedError, random.shuffle, arr, 1)

        arr = np.array(3)
        assert_raises(TypeError, random.shuffle, arr)
        arr = np.ones((3, 2))
        assert_raises(np.AxisError, random.shuffle, arr, 2)

    def test_shuffle_not_writeable(self):
        random = Generator(MT19937(self.seed))
        a = np.zeros(5)
        a.flags.writeable = False
        with pytest.raises(ValueError, match='read-only'):
            random.shuffle(a)

    def test_permutation(self):
        random = Generator(MT19937(self.seed))
        alist = [1, 2, 3, 4, 5, 6, 7, 8, 9, 0]
        actual = random.permutation(alist)
        desired = [4, 1, 9, 8, 0, 5, 3, 6, 2, 7]
        assert_array_equal(actual, desired)

        random = Generator(MT19937(self.seed))
        arr_2d = np.atleast_2d([1, 2, 3, 4, 5, 6, 7, 8, 9, 0]).T
        actual = random.permutation(arr_2d)
        assert_array_equal(actual, np.atleast_2d(desired).T)

        bad_x_str = "abcd"
        assert_raises(np.AxisError, random.permutation, bad_x_str)

        bad_x_float = 1.2
        assert_raises(np.AxisError, random.permutation, bad_x_float)

        random = Generator(MT19937(self.seed))
        integer_val = 10
        desired = [3, 0, 8, 7, 9, 4, 2, 5, 1, 6]

        actual = random.permutation(integer_val)
        assert_array_equal(actual, desired)

    def test_permutation_custom_axis(self):
        a = np.arange(16).reshape((4, 4))
        desired = np.array([[ 0,  3,  1,  2],
                            [ 4,  7,  5,  6],
                            [ 8, 11,  9, 10],
                            [12, 15, 13, 14]])
        random = Generator(MT19937(self.seed))
        actual = random.permutation(a, axis=1)
        assert_array_equal(actual, desired)
        random = Generator(MT19937(self.seed))
        actual = random.permutation(a, axis=-1)
        assert_array_equal(actual, desired)

    def test_permutation_exceptions(self):
        random = Generator(MT19937(self.seed))
        arr = np.arange(10)
        assert_raises(np.AxisError, random.permutation, arr, 1)
        arr = np.arange(9).reshape((3, 3))
        assert_raises(np.AxisError, random.permutation, arr, 3)
        assert_raises(TypeError, random.permutation, arr, slice(1, 2, None))

    @pytest.mark.parametrize("dtype", [int, object])
    @pytest.mark.parametrize("axis, expected",
                             [(None, np.array([[3, 7, 0, 9, 10, 11],
                                               [8, 4, 2, 5,  1,  6]])),
                              (0, np.array([[6, 1, 2, 9, 10, 11],
                                            [0, 7, 8, 3,  4,  5]])),
                              (1, np.array([[ 5, 3,  4, 0, 2, 1],
                                            [11, 9, 10, 6, 8, 7]]))])
    def test_permuted(self, dtype, axis, expected):
        random = Generator(MT19937(self.seed))
        x = np.arange(12).reshape(2, 6).astype(dtype)
        random.permuted(x, axis=axis, out=x)
        assert_array_equal(x, expected)

        random = Generator(MT19937(self.seed))
        x = np.arange(12).reshape(2, 6).astype(dtype)
        y = random.permuted(x, axis=axis)
        assert y.dtype == dtype
        assert_array_equal(y, expected)

    def test_permuted_with_strides(self):
        random = Generator(MT19937(self.seed))
        x0 = np.arange(22).reshape(2, 11)
        x1 = x0.copy()
        x = x0[:, ::3]
        y = random.permuted(x, axis=1, out=x)
        expected = np.array([[0, 9, 3, 6],
                             [14, 20, 11, 17]])
        assert_array_equal(y, expected)
        x1[:, ::3] = expected
        # Verify that the original x0 was modified in-place as expected.
        assert_array_equal(x1, x0)

    def test_permuted_empty(self):
        y = random.permuted([])
        assert_array_equal(y, [])

    @pytest.mark.parametrize('outshape', [(2, 3), 5])
    def test_permuted_out_with_wrong_shape(self, outshape):
        a = np.array([1, 2, 3])
        out = np.zeros(outshape, dtype=a.dtype)
        with pytest.raises(ValueError, match='same shape'):
            random.permuted(a, out=out)

    def test_permuted_out_with_wrong_type(self):
        out = np.zeros((3, 5), dtype=np.int32)
        x = np.ones((3, 5))
        with pytest.raises(TypeError, match='Cannot cast'):
            random.permuted(x, axis=1, out=out)

    def test_permuted_not_writeable(self):
        x = np.zeros((2, 5))
        x.flags.writeable = False
        with pytest.raises(ValueError, match='read-only'):
            random.permuted(x, axis=1, out=x)

    def test_beta(self):
        random = Generator(MT19937(self.seed))
        actual = random.beta(.1, .9, size=(3, 2))
        desired = np.array(
            [[1.083029353267698e-10, 2.449965303168024e-11],
             [2.397085162969853e-02, 3.590779671820755e-08],
             [2.830254190078299e-04, 1.744709918330393e-01]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_binomial(self):
        random = Generator(MT19937(self.seed))
        actual = random.binomial(100.123, .456, size=(3, 2))
        desired = np.array([[42, 41],
                            [42, 48],
                            [44, 50]])
        assert_array_equal(actual, desired)

        random = Generator(MT19937(self.seed))
        actual = random.binomial(100.123, .456)
        desired = 42
        assert_array_equal(actual, desired)

    def test_chisquare(self):
        random = Generator(MT19937(self.seed))
        actual = random.chisquare(50, size=(3, 2))
        desired = np.array([[32.9850547060149, 39.0219480493301],
                            [56.2006134779419, 57.3474165711485],
                            [55.4243733880198, 55.4209797925213]])
        assert_array_almost_equal(actual, desired, decimal=13)

    def test_dirichlet(self):
        random = Generator(MT19937(self.seed))
        alpha = np.array([51.72840233779265162, 39.74494232180943953])
        actual = random.dirichlet(alpha, size=(3, 2))
        desired = np.array([[[0.5439892869558927,  0.45601071304410745],
                             [0.5588917345860708,  0.4411082654139292 ]],
                            [[0.5632074165063435,  0.43679258349365657],
                             [0.54862581112627,    0.45137418887373015]],
                            [[0.49961831357047226, 0.5003816864295278 ],
                             [0.52374806183482,    0.47625193816517997]]])
        assert_array_almost_equal(actual, desired, decimal=15)
        bad_alpha = np.array([5.4e-01, -1.0e-16])
        assert_raises(ValueError, random.dirichlet, bad_alpha)

        random = Generator(MT19937(self.seed))
        alpha = np.array([51.72840233779265162, 39.74494232180943953])
        actual = random.dirichlet(alpha)
        assert_array_almost_equal(actual, desired[0, 0], decimal=15)

    def test_dirichlet_size(self):
        # gh-3173
        p = np.array([51.72840233779265162, 39.74494232180943953])
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, np.uint32(1)).shape, (1, 2))
        assert_equal(random.dirichlet(p, [2, 2]).shape, (2, 2, 2))
        assert_equal(random.dirichlet(p, (2, 2)).shape, (2, 2, 2))
        assert_equal(random.dirichlet(p, np.array((2, 2))).shape, (2, 2, 2))

        assert_raises(TypeError, random.dirichlet, p, float(1))

    def test_dirichlet_bad_alpha(self):
        # gh-2089
        alpha = np.array([5.4e-01, -1.0e-16])
        assert_raises(ValueError, random.dirichlet, alpha)

        # gh-15876
        assert_raises(ValueError, random.dirichlet, [[5, 1]])
        assert_raises(ValueError, random.dirichlet, [[5], [1]])
        assert_raises(ValueError, random.dirichlet, [[[5], [1]], [[1], [5]]])
        assert_raises(ValueError, random.dirichlet, np.array([[5, 1], [1, 5]]))

    def test_dirichlet_alpha_non_contiguous(self):
        a = np.array([51.72840233779265162, -1.0, 39.74494232180943953])
        alpha = a[::2]
        random = Generator(MT19937(self.seed))
        non_contig = random.dirichlet(alpha, size=(3, 2))
        random = Generator(MT19937(self.seed))
        contig = random.dirichlet(np.ascontiguousarray(alpha),
                                  size=(3, 2))
        assert_array_almost_equal(non_contig, contig)

    def test_dirichlet_small_alpha(self):
        eps = 1.0e-9  # 1.0e-10 -> runtime x 10; 1e-11 -> runtime x 200, etc.
        alpha = eps * np.array([1., 1.0e-3])
        random = Generator(MT19937(self.seed))
        actual = random.dirichlet(alpha, size=(3, 2))
        expected = np.array([
            [[1., 0.],
             [1., 0.]],
            [[1., 0.],
             [1., 0.]],
            [[1., 0.],
             [1., 0.]]
        ])
        assert_array_almost_equal(actual, expected, decimal=15)

    @pytest.mark.slow
    def test_dirichlet_moderately_small_alpha(self):
        # Use alpha.max() < 0.1 to trigger stick breaking code path
        alpha = np.array([0.02, 0.04, 0.03])
        exact_mean = alpha / alpha.sum()
        random = Generator(MT19937(self.seed))
        sample = random.dirichlet(alpha, size=20000000)
        sample_mean = sample.mean(axis=0)
        assert_allclose(sample_mean, exact_mean, rtol=1e-3)

    def test_exponential(self):
        random = Generator(MT19937(self.seed))
        actual = random.exponential(1.1234, size=(3, 2))
        desired = np.array([[0.098845481066258, 1.560752510746964],
                            [0.075730916041636, 1.769098974710777],
                            [1.488602544592235, 2.49684815275751 ]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_exponential_0(self):
        assert_equal(random.exponential(scale=0), 0)
        assert_raises(ValueError, random.exponential, scale=-0.)

    def test_f(self):
        random = Generator(MT19937(self.seed))
        actual = random.f(12, 77, size=(3, 2))
        desired = np.array([[0.461720027077085, 1.100441958872451],
                            [1.100337455217484, 0.91421736740018 ],
                            [0.500811891303113, 0.826802454552058]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_gamma(self):
        random = Generator(MT19937(self.seed))
        actual = random.gamma(5, 3, size=(3, 2))
        desired = np.array([[ 5.03850858902096,  7.9228656732049 ],
                            [18.73983605132985, 19.57961681699238],
                            [18.17897755150825, 18.17653912505234]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_gamma_0(self):
        assert_equal(random.gamma(shape=0, scale=0), 0)
        assert_raises(ValueError, random.gamma, shape=-0., scale=-0.)

    def test_geometric(self):
        random = Generator(MT19937(self.seed))
        actual = random.geometric(.123456789, size=(3, 2))
        desired = np.array([[1, 11],
                            [1, 12],
                            [11, 17]])
        assert_array_equal(actual, desired)

    def test_geometric_exceptions(self):
        assert_raises(ValueError, random.geometric, 1.1)
        assert_raises(ValueError, random.geometric, [1.1] * 10)
        assert_raises(ValueError, random.geometric, -0.1)
        assert_raises(ValueError, random.geometric, [-0.1] * 10)
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, random.geometric, np.nan)
            assert_raises(ValueError, random.geometric, [np.nan] * 10)

    def test_gumbel(self):
        random = Generator(MT19937(self.seed))
        actual = random.gumbel(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[ 4.688397515056245, -0.289514845417841],
                            [ 4.981176042584683, -0.633224272589149],
                            [-0.055915275687488, -0.333962478257953]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_gumbel_0(self):
        assert_equal(random.gumbel(scale=0), 0)
        assert_raises(ValueError, random.gumbel, scale=-0.)

    def test_hypergeometric(self):
        random = Generator(MT19937(self.seed))
        actual = random.hypergeometric(10.1, 5.5, 14, size=(3, 2))
        desired = np.array([[ 9, 9],
                            [ 9, 9],
                            [10, 9]])
        assert_array_equal(actual, desired)

        # Test nbad = 0
        actual = random.hypergeometric(5, 0, 3, size=4)
        desired = np.array([3, 3, 3, 3])
        assert_array_equal(actual, desired)

        actual = random.hypergeometric(15, 0, 12, size=4)
        desired = np.array([12, 12, 12, 12])
        assert_array_equal(actual, desired)

        # Test ngood = 0
        actual = random.hypergeometric(0, 5, 3, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

        actual = random.hypergeometric(0, 15, 12, size=4)
        desired = np.array([0, 0, 0, 0])
        assert_array_equal(actual, desired)

    def test_laplace(self):
        random = Generator(MT19937(self.seed))
        actual = random.laplace(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[-3.156353949272393,  1.195863024830054],
                            [-3.435458081645966,  1.656882398925444],
                            [ 0.924824032467446,  1.251116432209336]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_laplace_0(self):
        assert_equal(random.laplace(scale=0), 0)
        assert_raises(ValueError, random.laplace, scale=-0.)

    def test_logistic(self):
        random = Generator(MT19937(self.seed))
        actual = random.logistic(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[-4.338584631510999,  1.890171436749954],
                            [-4.64547787337966 ,  2.514545562919217],
                            [ 1.495389489198666,  1.967827627577474]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_lognormal(self):
        random = Generator(MT19937(self.seed))
        actual = random.lognormal(mean=.123456789, sigma=2.0, size=(3, 2))
        desired = np.array([[ 0.0268252166335, 13.9534486483053],
                            [ 0.1204014788936,  2.2422077497792],
                            [ 4.2484199496128, 12.0093343977523]])
        assert_array_almost_equal(actual, desired, decimal=13)

    def test_lognormal_0(self):
        assert_equal(random.lognormal(sigma=0), 1)
        assert_raises(ValueError, random.lognormal, sigma=-0.)

    def test_logseries(self):
        random = Generator(MT19937(self.seed))
        actual = random.logseries(p=.923456789, size=(3, 2))
        desired = np.array([[14, 17],
                            [3, 18],
                            [5, 1]])
        assert_array_equal(actual, desired)

    def test_logseries_zero(self):
        random = Generator(MT19937(self.seed))
        assert random.logseries(0) == 1

    @pytest.mark.parametrize("value", [np.nextafter(0., -1), 1., np.nan, 5.])
    def test_logseries_exceptions(self, value):
        random = Generator(MT19937(self.seed))
        with np.errstate(invalid="ignore"):
            with pytest.raises(ValueError):
                random.logseries(value)
            with pytest.raises(ValueError):
                # contiguous path:
                random.logseries(np.array([value] * 10))
            with pytest.raises(ValueError):
                # non-contiguous path:
                random.logseries(np.array([value] * 10)[::2])

    def test_multinomial(self):
        random = Generator(MT19937(self.seed))
        actual = random.multinomial(20, [1 / 6.] * 6, size=(3, 2))
        desired = np.array([[[1, 5, 1, 6, 4, 3],
                             [4, 2, 6, 2, 4, 2]],
                            [[5, 3, 2, 6, 3, 1],
                             [4, 4, 0, 2, 3, 7]],
                            [[6, 3, 1, 5, 3, 2],
                             [5, 5, 3, 1, 2, 4]]])
        assert_array_equal(actual, desired)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    @pytest.mark.parametrize("method", ["svd", "eigh", "cholesky"])
    def test_multivariate_normal(self, method):
        random = Generator(MT19937(self.seed))
        mean = (.123456789, 10)
        cov = [[1, 0], [0, 1]]
        size = (3, 2)
        actual = random.multivariate_normal(mean, cov, size, method=method)
        desired = np.array([[[-1.747478062846581,  11.25613495182354  ],
                             [-0.9967333370066214, 10.342002097029821 ]],
                            [[ 0.7850019631242964, 11.181113712443013 ],
                             [ 0.8901349653255224,  8.873825399642492 ]],
                            [[ 0.7130260107430003,  9.551628690083056 ],
                             [ 0.7127098726541128, 11.991709234143173 ]]])

        assert_array_almost_equal(actual, desired, decimal=15)

        # Check for default size, was raising deprecation warning
        actual = random.multivariate_normal(mean, cov, method=method)
        desired = np.array([0.233278563284287, 9.424140804347195])
        assert_array_almost_equal(actual, desired, decimal=15)
        # Check that non symmetric covariance input raises exception when
        # check_valid='raises' if using default svd method.
        mean = [0, 0]
        cov = [[1, 2], [1, 2]]
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='raise')

        # Check that non positive-semidefinite covariance warns with
        # RuntimeWarning
        cov = [[1, 2], [2, 1]]
        assert_warns(RuntimeWarning, random.multivariate_normal, mean, cov)
        assert_warns(RuntimeWarning, random.multivariate_normal, mean, cov,
                     method='eigh')
        assert_raises(LinAlgError, random.multivariate_normal, mean, cov,
                      method='cholesky')

        # and that it doesn't warn with RuntimeWarning check_valid='ignore'
        assert_no_warnings(random.multivariate_normal, mean, cov,
                           check_valid='ignore')

        # and that it raises with RuntimeWarning check_valid='raises'
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='raise')
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='raise', method='eigh')

        # check degenerate samples from singular covariance matrix
        cov = [[1, 1], [1, 1]]
        if method in ('svd', 'eigh'):
            samples = random.multivariate_normal(mean, cov, size=(3, 2),
                                                 method=method)
            assert_array_almost_equal(samples[..., 0], samples[..., 1],
                                      decimal=6)
        else:
            assert_raises(LinAlgError, random.multivariate_normal, mean, cov,
                          method='cholesky')

        cov = np.array([[1, 0.1], [0.1, 1]], dtype=np.float32)
        with suppress_warnings() as sup:
            random.multivariate_normal(mean, cov, method=method)
            w = sup.record(RuntimeWarning)
            assert len(w) == 0

        mu = np.zeros(2)
        cov = np.eye(2)
        assert_raises(ValueError, random.multivariate_normal, mean, cov,
                      check_valid='other')
        assert_raises(ValueError, random.multivariate_normal,
                      np.zeros((2, 1, 1)), cov)
        assert_raises(ValueError, random.multivariate_normal,
                      mu, np.empty((3, 2)))
        assert_raises(ValueError, random.multivariate_normal,
                      mu, np.eye(3))
        
    @pytest.mark.parametrize('mean, cov', [([0], [[1+1j]]), ([0j], [[1]])])
    def test_multivariate_normal_disallow_complex(self, mean, cov):
        random = Generator(MT19937(self.seed))
        with pytest.raises(TypeError, match="must not be complex"):
            random.multivariate_normal(mean, cov)

    @pytest.mark.parametrize("method", ["svd", "eigh", "cholesky"])
    def test_multivariate_normal_basic_stats(self, method):
        random = Generator(MT19937(self.seed))
        n_s = 1000
        mean = np.array([1, 2])
        cov = np.array([[2, 1], [1, 2]])
        s = random.multivariate_normal(mean, cov, size=(n_s,), method=method)
        s_center = s - mean
        cov_emp = (s_center.T @ s_center) / (n_s - 1)
        # these are pretty loose and are only designed to detect major errors
        assert np.all(np.abs(s_center.mean(-2)) < 0.1)
        assert np.all(np.abs(cov_emp - cov) < 0.2)

    def test_negative_binomial(self):
        random = Generator(MT19937(self.seed))
        actual = random.negative_binomial(n=100, p=.12345, size=(3, 2))
        desired = np.array([[543, 727],
                            [775, 760],
                            [600, 674]])
        assert_array_equal(actual, desired)

    def test_negative_binomial_exceptions(self):
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, random.negative_binomial, 100, np.nan)
            assert_raises(ValueError, random.negative_binomial, 100,
                          [np.nan] * 10)

    def test_negative_binomial_p0_exception(self):
        # Verify that p=0 raises an exception.
        with assert_raises(ValueError):
            x = random.negative_binomial(1, 0)

    def test_negative_binomial_invalid_p_n_combination(self):
        # Verify that values of p and n that would result in an overflow
        # or infinite loop raise an exception.
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, random.negative_binomial, 2**62, 0.1)
            assert_raises(ValueError, random.negative_binomial, [2**62], [0.1])

    def test_noncentral_chisquare(self):
        random = Generator(MT19937(self.seed))
        actual = random.noncentral_chisquare(df=5, nonc=5, size=(3, 2))
        desired = np.array([[ 1.70561552362133, 15.97378184942111],
                            [13.71483425173724, 20.17859633310629],
                            [11.3615477156643 ,  3.67891108738029]])
        assert_array_almost_equal(actual, desired, decimal=14)

        actual = random.noncentral_chisquare(df=.5, nonc=.2, size=(3, 2))
        desired = np.array([[9.41427665607629e-04, 1.70473157518850e-04],
                            [1.14554372041263e+00, 1.38187755933435e-03],
                            [1.90659181905387e+00, 1.21772577941822e+00]])
        assert_array_almost_equal(actual, desired, decimal=14)

        random = Generator(MT19937(self.seed))
        actual = random.noncentral_chisquare(df=5, nonc=0, size=(3, 2))
        desired = np.array([[0.82947954590419, 1.80139670767078],
                            [6.58720057417794, 7.00491463609814],
                            [6.31101879073157, 6.30982307753005]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_noncentral_f(self):
        random = Generator(MT19937(self.seed))
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=1,
                                     size=(3, 2))
        desired = np.array([[0.060310671139  , 0.23866058175939],
                            [0.86860246709073, 0.2668510459738 ],
                            [0.23375780078364, 1.88922102885943]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_noncentral_f_nan(self):
        random = Generator(MT19937(self.seed))
        actual = random.noncentral_f(dfnum=5, dfden=2, nonc=np.nan)
        assert np.isnan(actual)

    def test_normal(self):
        random = Generator(MT19937(self.seed))
        actual = random.normal(loc=.123456789, scale=2.0, size=(3, 2))
        desired = np.array([[-3.618412914693162,  2.635726692647081],
                            [-2.116923463013243,  0.807460983059643],
                            [ 1.446547137248593,  2.485684213886024]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_normal_0(self):
        assert_equal(random.normal(scale=0), 0)
        assert_raises(ValueError, random.normal, scale=-0.)

    def test_pareto(self):
        random = Generator(MT19937(self.seed))
        actual = random.pareto(a=.123456789, size=(3, 2))
        desired = np.array([[1.0394926776069018e+00, 7.7142534343505773e+04],
                            [7.2640150889064703e-01, 3.4650454783825594e+05],
                            [4.5852344481994740e+04, 6.5851383009539105e+07]])
        # For some reason on 32-bit x86 Ubuntu 12.10 the [1, 0] entry in this
        # matrix differs by 24 nulps. Discussion:
        #   https://mail.python.org/pipermail/numpy-discussion/2012-September/063801.html
        # Consensus is that this is probably some gcc quirk that affects
        # rounding but not in any important way, so we just use a looser
        # tolerance on this test:
        np.testing.assert_array_almost_equal_nulp(actual, desired, nulp=30)

    def test_poisson(self):
        random = Generator(MT19937(self.seed))
        actual = random.poisson(lam=.123456789, size=(3, 2))
        desired = np.array([[0, 0],
                            [0, 0],
                            [0, 0]])
        assert_array_equal(actual, desired)

    def test_poisson_exceptions(self):
        lambig = np.iinfo('int64').max
        lamneg = -1
        assert_raises(ValueError, random.poisson, lamneg)
        assert_raises(ValueError, random.poisson, [lamneg] * 10)
        assert_raises(ValueError, random.poisson, lambig)
        assert_raises(ValueError, random.poisson, [lambig] * 10)
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, random.poisson, np.nan)
            assert_raises(ValueError, random.poisson, [np.nan] * 10)

    def test_power(self):
        random = Generator(MT19937(self.seed))
        actual = random.power(a=.123456789, size=(3, 2))
        desired = np.array([[1.977857368842754e-09, 9.806792196620341e-02],
                            [2.482442984543471e-10, 1.527108843266079e-01],
                            [8.188283434244285e-02, 3.950547209346948e-01]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_rayleigh(self):
        random = Generator(MT19937(self.seed))
        actual = random.rayleigh(scale=10, size=(3, 2))
        desired = np.array([[4.19494429102666, 16.66920198906598],
                            [3.67184544902662, 17.74695521962917],
                            [16.27935397855501, 21.08355560691792]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_rayleigh_0(self):
        assert_equal(random.rayleigh(scale=0), 0)
        assert_raises(ValueError, random.rayleigh, scale=-0.)

    def test_standard_cauchy(self):
        random = Generator(MT19937(self.seed))
        actual = random.standard_cauchy(size=(3, 2))
        desired = np.array([[-1.489437778266206, -3.275389641569784],
                            [ 0.560102864910406, -0.680780916282552],
                            [-1.314912905226277,  0.295852965660225]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_exponential(self):
        random = Generator(MT19937(self.seed))
        actual = random.standard_exponential(size=(3, 2), method='inv')
        desired = np.array([[0.102031839440643, 1.229350298474972],
                            [0.088137284693098, 1.459859985522667],
                            [1.093830802293668, 1.256977002164613]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_expoential_type_error(self):
        assert_raises(TypeError, random.standard_exponential, dtype=np.int32)

    def test_standard_gamma(self):
        random = Generator(MT19937(self.seed))
        actual = random.standard_gamma(shape=3, size=(3, 2))
        desired = np.array([[0.62970724056362, 1.22379851271008],
                            [3.899412530884  , 4.12479964250139],
                            [3.74994102464584, 3.74929307690815]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_standard_gammma_scalar_float(self):
        random = Generator(MT19937(self.seed))
        actual = random.standard_gamma(3, dtype=np.float32)
        desired = 2.9242148399353027
        assert_array_almost_equal(actual, desired, decimal=6)

    def test_standard_gamma_float(self):
        random = Generator(MT19937(self.seed))
        actual = random.standard_gamma(shape=3, size=(3, 2))
        desired = np.array([[0.62971, 1.2238 ],
                            [3.89941, 4.1248 ],
                            [3.74994, 3.74929]])
        assert_array_almost_equal(actual, desired, decimal=5)

    def test_standard_gammma_float_out(self):
        actual = np.zeros((3, 2), dtype=np.float32)
        random = Generator(MT19937(self.seed))
        random.standard_gamma(10.0, out=actual, dtype=np.float32)
        desired = np.array([[10.14987,  7.87012],
                             [ 9.46284, 12.56832],
                             [13.82495,  7.81533]], dtype=np.float32)
        assert_array_almost_equal(actual, desired, decimal=5)

        random = Generator(MT19937(self.seed))
        random.standard_gamma(10.0, out=actual, size=(3, 2), dtype=np.float32)
        assert_array_almost_equal(actual, desired, decimal=5)

    def test_standard_gamma_unknown_type(self):
        assert_raises(TypeError, random.standard_gamma, 1.,
                      dtype='int32')

    def test_out_size_mismatch(self):
        out = np.zeros(10)
        assert_raises(ValueError, random.standard_gamma, 10.0, size=20,
                      out=out)
        assert_raises(ValueError, random.standard_gamma, 10.0, size=(10, 1),
                      out=out)

    def test_standard_gamma_0(self):
        assert_equal(random.standard_gamma(shape=0), 0)
        assert_raises(ValueError, random.standard_gamma, shape=-0.)

    def test_standard_normal(self):
        random = Generator(MT19937(self.seed))
        actual = random.standard_normal(size=(3, 2))
        desired = np.array([[-1.870934851846581,  1.25613495182354 ],
                            [-1.120190126006621,  0.342002097029821],
                            [ 0.661545174124296,  1.181113712443012]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_standard_normal_unsupported_type(self):
        assert_raises(TypeError, random.standard_normal, dtype=np.int32)

    def test_standard_t(self):
        random = Generator(MT19937(self.seed))
        actual = random.standard_t(df=10, size=(3, 2))
        desired = np.array([[-1.484666193042647,  0.30597891831161 ],
                            [ 1.056684299648085, -0.407312602088507],
                            [ 0.130704414281157, -2.038053410490321]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_triangular(self):
        random = Generator(MT19937(self.seed))
        actual = random.triangular(left=5.12, mode=10.23, right=20.34,
                                   size=(3, 2))
        desired = np.array([[ 7.86664070590917, 13.6313848513185 ],
                            [ 7.68152445215983, 14.36169131136546],
                            [13.16105603911429, 13.72341621856971]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_uniform(self):
        random = Generator(MT19937(self.seed))
        actual = random.uniform(low=1.23, high=10.54, size=(3, 2))
        desired = np.array([[2.13306255040998 , 7.816987531021207],
                            [2.015436610109887, 8.377577533009589],
                            [7.421792588856135, 7.891185744455209]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_uniform_range_bounds(self):
        fmin = np.finfo('float').min
        fmax = np.finfo('float').max

        func = random.uniform
        assert_raises(OverflowError, func, -np.inf, 0)
        assert_raises(OverflowError, func, 0, np.inf)
        assert_raises(OverflowError, func, fmin, fmax)
        assert_raises(OverflowError, func, [-np.inf], [0])
        assert_raises(OverflowError, func, [0], [np.inf])

        # (fmax / 1e17) - fmin is within range, so this should not throw
        # account for i386 extended precision DBL_MAX / 1e17 + DBL_MAX >
        # DBL_MAX by increasing fmin a bit
        random.uniform(low=np.nextafter(fmin, 1), high=fmax / 1e17)

    def test_uniform_zero_range(self):
        func = random.uniform
        result = func(1.5, 1.5)
        assert_allclose(result, 1.5)
        result = func([0.0, np.pi], [0.0, np.pi])
        assert_allclose(result, [0.0, np.pi])
        result = func([[2145.12], [2145.12]], [2145.12, 2145.12])
        assert_allclose(result, 2145.12 + np.zeros((2, 2)))

    def test_uniform_neg_range(self):
        func = random.uniform
        assert_raises(ValueError, func, 2, 1)
        assert_raises(ValueError, func,  [1, 2], [1, 1])
        assert_raises(ValueError, func,  [[0, 1],[2, 3]], 2)

    def test_scalar_exception_propagation(self):
        # Tests that exceptions are correctly propagated in distributions
        # when called with objects that throw exceptions when converted to
        # scalars.
        #
        # Regression test for gh: 8865

        class ThrowingFloat(np.ndarray):
            def __float__(self):
                raise TypeError

        throwing_float = np.array(1.0).view(ThrowingFloat)
        assert_raises(TypeError, random.uniform, throwing_float,
                      throwing_float)

        class ThrowingInteger(np.ndarray):
            def __int__(self):
                raise TypeError

        throwing_int = np.array(1).view(ThrowingInteger)
        assert_raises(TypeError, random.hypergeometric, throwing_int, 1, 1)

    def test_vonmises(self):
        random = Generator(MT19937(self.seed))
        actual = random.vonmises(mu=1.23, kappa=1.54, size=(3, 2))
        desired = np.array([[ 1.107972248690106,  2.841536476232361],
                            [ 1.832602376042457,  1.945511926976032],
                            [-0.260147475776542,  2.058047492231698]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_vonmises_small(self):
        # check infinite loop, gh-4720
        random = Generator(MT19937(self.seed))
        r = random.vonmises(mu=0., kappa=1.1e-8, size=10**6)
        assert_(np.isfinite(r).all())

    def test_vonmises_nan(self):
        random = Generator(MT19937(self.seed))
        r = random.vonmises(mu=0., kappa=np.nan)
        assert_(np.isnan(r))

    @pytest.mark.parametrize("kappa", [1e4, 1e15])
    def test_vonmises_large_kappa(self, kappa):
        random = Generator(MT19937(self.seed))
        rs = RandomState(random.bit_generator)
        state = random.bit_generator.state

        random_state_vals = rs.vonmises(0, kappa, size=10)
        random.bit_generator.state = state
        gen_vals = random.vonmises(0, kappa, size=10)
        if kappa < 1e6:
            assert_allclose(random_state_vals, gen_vals)
        else:
            assert np.all(random_state_vals != gen_vals)

    @pytest.mark.parametrize("mu", [-7., -np.pi, -3.1, np.pi, 3.2])
    @pytest.mark.parametrize("kappa", [1e-9, 1e-6, 1, 1e3, 1e15])
    def test_vonmises_large_kappa_range(self, mu, kappa):
        random = Generator(MT19937(self.seed))
        r = random.vonmises(mu, kappa, 50)
        assert_(np.all(r > -np.pi) and np.all(r <= np.pi))

    def test_wald(self):
        random = Generator(MT19937(self.seed))
        actual = random.wald(mean=1.23, scale=1.54, size=(3, 2))
        desired = np.array([[0.26871721804551, 3.2233942732115 ],
                            [2.20328374987066, 2.40958405189353],
                            [2.07093587449261, 0.73073890064369]])
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_weibull(self):
        random = Generator(MT19937(self.seed))
        actual = random.weibull(a=1.23, size=(3, 2))
        desired = np.array([[0.138613914769468, 1.306463419753191],
                            [0.111623365934763, 1.446570494646721],
                            [1.257145775276011, 1.914247725027957]])
        assert_array_almost_equal(actual, desired, decimal=15)

    def test_weibull_0(self):
        random = Generator(MT19937(self.seed))
        assert_equal(random.weibull(a=0, size=12), np.zeros(12))
        assert_raises(ValueError, random.weibull, a=-0.)

    def test_zipf(self):
        random = Generator(MT19937(self.seed))
        actual = random.zipf(a=1.23, size=(3, 2))
        desired = np.array([[  1,   1],
                            [ 10, 867],
                            [354,   2]])
        assert_array_equal(actual, desired)


class TestBroadcast:
    # tests that functions that broadcast behave
    # correctly when presented with non-scalar arguments
    def setup_method(self):
        self.seed = 123456789


    def test_uniform(self):
        random = Generator(MT19937(self.seed))
        low = [0]
        high = [1]
        uniform = random.uniform
        desired = np.array([0.16693771389729, 0.19635129550675, 0.75563050964095])

        random = Generator(MT19937(self.seed))
        actual = random.uniform(low * 3, high)
        assert_array_almost_equal(actual, desired, decimal=14)

        random = Generator(MT19937(self.seed))
        actual = random.uniform(low, high * 3)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_normal(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        random = Generator(MT19937(self.seed))
        desired = np.array([-0.38736406738527,  0.79594375042255,  0.0197076236097])

        random = Generator(MT19937(self.seed))
        actual = random.normal(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.normal, loc * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        normal = random.normal
        actual = normal(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, normal, loc, bad_scale * 3)

    def test_beta(self):
        a = [1]
        b = [2]
        bad_a = [-1]
        bad_b = [-2]
        desired = np.array([0.18719338682602, 0.73234824491364, 0.17928615186455])

        random = Generator(MT19937(self.seed))
        beta = random.beta
        actual = beta(a * 3, b)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, beta, bad_a * 3, b)
        assert_raises(ValueError, beta, a * 3, bad_b)

        random = Generator(MT19937(self.seed))
        actual = random.beta(a, b * 3)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_exponential(self):
        scale = [1]
        bad_scale = [-1]
        desired = np.array([0.67245993212806, 0.21380495318094, 0.7177848928629])

        random = Generator(MT19937(self.seed))
        actual = random.exponential(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.exponential, bad_scale * 3)

    def test_standard_gamma(self):
        shape = [1]
        bad_shape = [-1]
        desired = np.array([0.67245993212806, 0.21380495318094, 0.7177848928629])

        random = Generator(MT19937(self.seed))
        std_gamma = random.standard_gamma
        actual = std_gamma(shape * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, std_gamma, bad_shape * 3)

    def test_gamma(self):
        shape = [1]
        scale = [2]
        bad_shape = [-1]
        bad_scale = [-2]
        desired = np.array([1.34491986425611, 0.42760990636187, 1.4355697857258])

        random = Generator(MT19937(self.seed))
        gamma = random.gamma
        actual = gamma(shape * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape * 3, scale)
        assert_raises(ValueError, gamma, shape * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        gamma = random.gamma
        actual = gamma(shape, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gamma, bad_shape, scale * 3)
        assert_raises(ValueError, gamma, shape, bad_scale * 3)

    def test_f(self):
        dfnum = [1]
        dfden = [2]
        bad_dfnum = [-1]
        bad_dfden = [-2]
        desired = np.array([0.07765056244107, 7.72951397913186, 0.05786093891763])

        random = Generator(MT19937(self.seed))
        f = random.f
        actual = f(dfnum * 3, dfden)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, f, bad_dfnum * 3, dfden)
        assert_raises(ValueError, f, dfnum * 3, bad_dfden)

        random = Generator(MT19937(self.seed))
        f = random.f
        actual = f(dfnum, dfden * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, f, bad_dfnum, dfden * 3)
        assert_raises(ValueError, f, dfnum, bad_dfden * 3)

    def test_noncentral_f(self):
        dfnum = [2]
        dfden = [3]
        nonc = [4]
        bad_dfnum = [0]
        bad_dfden = [-1]
        bad_nonc = [-2]
        desired = np.array([2.02434240411421, 12.91838601070124, 1.24395160354629])

        random = Generator(MT19937(self.seed))
        nonc_f = random.noncentral_f
        actual = nonc_f(dfnum * 3, dfden, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert np.all(np.isnan(nonc_f(dfnum, dfden, [np.nan] * 3)))

        assert_raises(ValueError, nonc_f, bad_dfnum * 3, dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, bad_dfden, nonc)
        assert_raises(ValueError, nonc_f, dfnum * 3, dfden, bad_nonc)

        random = Generator(MT19937(self.seed))
        nonc_f = random.noncentral_f
        actual = nonc_f(dfnum, dfden * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden * 3, nonc)
        assert_raises(ValueError, nonc_f, dfnum, dfden * 3, bad_nonc)

        random = Generator(MT19937(self.seed))
        nonc_f = random.noncentral_f
        actual = nonc_f(dfnum, dfden, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_f, bad_dfnum, dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, bad_dfden, nonc * 3)
        assert_raises(ValueError, nonc_f, dfnum, dfden, bad_nonc * 3)

    def test_noncentral_f_small_df(self):
        random = Generator(MT19937(self.seed))
        desired = np.array([0.04714867120827, 0.1239390327694])
        actual = random.noncentral_f(0.9, 0.9, 2, size=2)
        assert_array_almost_equal(actual, desired, decimal=14)

    def test_chisquare(self):
        df = [1]
        bad_df = [-1]
        desired = np.array([0.05573640064251, 1.47220224353539, 2.9469379318589])

        random = Generator(MT19937(self.seed))
        actual = random.chisquare(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.chisquare, bad_df * 3)

    def test_noncentral_chisquare(self):
        df = [1]
        nonc = [2]
        bad_df = [-1]
        bad_nonc = [-2]
        desired = np.array([0.07710766249436, 5.27829115110304, 0.630732147399])

        random = Generator(MT19937(self.seed))
        nonc_chi = random.noncentral_chisquare
        actual = nonc_chi(df * 3, nonc)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df * 3, nonc)
        assert_raises(ValueError, nonc_chi, df * 3, bad_nonc)

        random = Generator(MT19937(self.seed))
        nonc_chi = random.noncentral_chisquare
        actual = nonc_chi(df, nonc * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, nonc_chi, bad_df, nonc * 3)
        assert_raises(ValueError, nonc_chi, df, bad_nonc * 3)

    def test_standard_t(self):
        df = [1]
        bad_df = [-1]
        desired = np.array([-1.39498829447098, -1.23058658835223, 0.17207021065983])

        random = Generator(MT19937(self.seed))
        actual = random.standard_t(df * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.standard_t, bad_df * 3)

    def test_vonmises(self):
        mu = [2]
        kappa = [1]
        bad_kappa = [-1]
        desired = np.array([2.25935584988528, 2.23326261461399, -2.84152146503326])

        random = Generator(MT19937(self.seed))
        actual = random.vonmises(mu * 3, kappa)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.vonmises, mu * 3, bad_kappa)

        random = Generator(MT19937(self.seed))
        actual = random.vonmises(mu, kappa * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.vonmises, mu, bad_kappa * 3)

    def test_pareto(self):
        a = [1]
        bad_a = [-1]
        desired = np.array([0.95905052946317, 0.2383810889437 , 1.04988745750013])

        random = Generator(MT19937(self.seed))
        actual = random.pareto(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.pareto, bad_a * 3)

    def test_weibull(self):
        a = [1]
        bad_a = [-1]
        desired = np.array([0.67245993212806, 0.21380495318094, 0.7177848928629])

        random = Generator(MT19937(self.seed))
        actual = random.weibull(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.weibull, bad_a * 3)

    def test_power(self):
        a = [1]
        bad_a = [-1]
        desired = np.array([0.48954864361052, 0.19249412888486, 0.51216834058807])

        random = Generator(MT19937(self.seed))
        actual = random.power(a * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.power, bad_a * 3)

    def test_laplace(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        desired = np.array([-1.09698732625119, -0.93470271947368, 0.71592671378202])

        random = Generator(MT19937(self.seed))
        laplace = random.laplace
        actual = laplace(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        laplace = random.laplace
        actual = laplace(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, laplace, loc, bad_scale * 3)

    def test_gumbel(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        desired = np.array([1.70020068231762, 1.52054354273631, -0.34293267607081])

        random = Generator(MT19937(self.seed))
        gumbel = random.gumbel
        actual = gumbel(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        gumbel = random.gumbel
        actual = gumbel(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, gumbel, loc, bad_scale * 3)

    def test_logistic(self):
        loc = [0]
        scale = [1]
        bad_scale = [-1]
        desired = np.array([-1.607487640433, -1.40925686003678, 1.12887112820397])

        random = Generator(MT19937(self.seed))
        actual = random.logistic(loc * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.logistic, loc * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        actual = random.logistic(loc, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.logistic, loc, bad_scale * 3)
        assert_equal(random.logistic(1.0, 0.0), 1.0)

    def test_lognormal(self):
        mean = [0]
        sigma = [1]
        bad_sigma = [-1]
        desired = np.array([0.67884390500697, 2.21653186290321, 1.01990310084276])

        random = Generator(MT19937(self.seed))
        lognormal = random.lognormal
        actual = lognormal(mean * 3, sigma)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, lognormal, mean * 3, bad_sigma)

        random = Generator(MT19937(self.seed))
        actual = random.lognormal(mean, sigma * 3)
        assert_raises(ValueError, random.lognormal, mean, bad_sigma * 3)

    def test_rayleigh(self):
        scale = [1]
        bad_scale = [-1]
        desired = np.array(
            [1.1597068009872629,
             0.6539188836253857,
             1.1981526554349398]
        )

        random = Generator(MT19937(self.seed))
        actual = random.rayleigh(scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.rayleigh, bad_scale * 3)

    def test_wald(self):
        mean = [0.5]
        scale = [1]
        bad_mean = [0]
        bad_scale = [-2]
        desired = np.array([0.38052407392905, 0.50701641508592, 0.484935249864])

        random = Generator(MT19937(self.seed))
        actual = random.wald(mean * 3, scale)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.wald, bad_mean * 3, scale)
        assert_raises(ValueError, random.wald, mean * 3, bad_scale)

        random = Generator(MT19937(self.seed))
        actual = random.wald(mean, scale * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, random.wald, bad_mean, scale * 3)
        assert_raises(ValueError, random.wald, mean, bad_scale * 3)

    def test_triangular(self):
        left = [1]
        right = [3]
        mode = [2]
        bad_left_one = [3]
        bad_mode_one = [4]
        bad_left_two, bad_mode_two = right * 2
        desired = np.array([1.57781954604754, 1.62665986867957, 2.30090130831326])

        random = Generator(MT19937(self.seed))
        triangular = random.triangular
        actual = triangular(left * 3, mode, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one * 3, mode, right)
        assert_raises(ValueError, triangular, left * 3, bad_mode_one, right)
        assert_raises(ValueError, triangular, bad_left_two * 3, bad_mode_two,
                      right)

        random = Generator(MT19937(self.seed))
        triangular = random.triangular
        actual = triangular(left, mode * 3, right)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode * 3, right)
        assert_raises(ValueError, triangular, left, bad_mode_one * 3, right)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two * 3,
                      right)

        random = Generator(MT19937(self.seed))
        triangular = random.triangular
        actual = triangular(left, mode, right * 3)
        assert_array_almost_equal(actual, desired, decimal=14)
        assert_raises(ValueError, triangular, bad_left_one, mode, right * 3)
        assert_raises(ValueError, triangular, left, bad_mode_one, right * 3)
        assert_raises(ValueError, triangular, bad_left_two, bad_mode_two,
                      right * 3)

        assert_raises(ValueError, triangular, 10., 0., 20.)
        assert_raises(ValueError, triangular, 10., 25., 20.)
        assert_raises(ValueError, triangular, 10., 10., 10.)

    def test_binomial(self):
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        desired = np.array([0, 0, 1])

        random = Generator(MT19937(self.seed))
        binom = random.binomial
        actual = binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n * 3, p)
        assert_raises(ValueError, binom, n * 3, bad_p_one)
        assert_raises(ValueError, binom, n * 3, bad_p_two)

        random = Generator(MT19937(self.seed))
        actual = random.binomial(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, binom, bad_n, p * 3)
        assert_raises(ValueError, binom, n, bad_p_one * 3)
        assert_raises(ValueError, binom, n, bad_p_two * 3)

    def test_negative_binomial(self):
        n = [1]
        p = [0.5]
        bad_n = [-1]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        desired = np.array([0, 2, 1], dtype=np.int64)

        random = Generator(MT19937(self.seed))
        neg_binom = random.negative_binomial
        actual = neg_binom(n * 3, p)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n * 3, p)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_one)
        assert_raises(ValueError, neg_binom, n * 3, bad_p_two)

        random = Generator(MT19937(self.seed))
        neg_binom = random.negative_binomial
        actual = neg_binom(n, p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, neg_binom, bad_n, p * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_one * 3)
        assert_raises(ValueError, neg_binom, n, bad_p_two * 3)

    def test_poisson(self):

        lam = [1]
        bad_lam_one = [-1]
        desired = np.array([0, 0, 3])

        random = Generator(MT19937(self.seed))
        max_lam = random._poisson_lam_max
        bad_lam_two = [max_lam * 2]
        poisson = random.poisson
        actual = poisson(lam * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, poisson, bad_lam_one * 3)
        assert_raises(ValueError, poisson, bad_lam_two * 3)

    def test_zipf(self):
        a = [2]
        bad_a = [0]
        desired = np.array([1, 8, 1])

        random = Generator(MT19937(self.seed))
        zipf = random.zipf
        actual = zipf(a * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, zipf, bad_a * 3)
        with np.errstate(invalid='ignore'):
            assert_raises(ValueError, zipf, np.nan)
            assert_raises(ValueError, zipf, [0, 0, np.nan])

    def test_geometric(self):
        p = [0.5]
        bad_p_one = [-1]
        bad_p_two = [1.5]
        desired = np.array([1, 1, 3])

        random = Generator(MT19937(self.seed))
        geometric = random.geometric
        actual = geometric(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, geometric, bad_p_one * 3)
        assert_raises(ValueError, geometric, bad_p_two * 3)

    def test_hypergeometric(self):
        ngood = [1]
        nbad = [2]
        nsample = [2]
        bad_ngood = [-1]
        bad_nbad = [-2]
        bad_nsample_one = [-1]
        bad_nsample_two = [4]
        desired = np.array([0, 0, 1])

        random = Generator(MT19937(self.seed))
        actual = random.hypergeometric(ngood * 3, nbad, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, random.hypergeometric, bad_ngood * 3, nbad, nsample)
        assert_raises(ValueError, random.hypergeometric, ngood * 3, bad_nbad, nsample)
        assert_raises(ValueError, random.hypergeometric, ngood * 3, nbad, bad_nsample_one)
        assert_raises(ValueError, random.hypergeometric, ngood * 3, nbad, bad_nsample_two)

        random = Generator(MT19937(self.seed))
        actual = random.hypergeometric(ngood, nbad * 3, nsample)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, random.hypergeometric, bad_ngood, nbad * 3, nsample)
        assert_raises(ValueError, random.hypergeometric, ngood, bad_nbad * 3, nsample)
        assert_raises(ValueError, random.hypergeometric, ngood, nbad * 3, bad_nsample_one)
        assert_raises(ValueError, random.hypergeometric, ngood, nbad * 3, bad_nsample_two)

        random = Generator(MT19937(self.seed))
        hypergeom = random.hypergeometric
        actual = hypergeom(ngood, nbad, nsample * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, hypergeom, bad_ngood, nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, bad_nbad, nsample * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_one * 3)
        assert_raises(ValueError, hypergeom, ngood, nbad, bad_nsample_two * 3)

        assert_raises(ValueError, hypergeom, -1, 10, 20)
        assert_raises(ValueError, hypergeom, 10, -1, 20)
        assert_raises(ValueError, hypergeom, 10, 10, -1)
        assert_raises(ValueError, hypergeom, 10, 10, 25)

        # ValueError for arguments that are too big.
        assert_raises(ValueError, hypergeom, 2**30, 10, 20)
        assert_raises(ValueError, hypergeom, 999, 2**31, 50)
        assert_raises(ValueError, hypergeom, 999, [2**29, 2**30], 1000)

    def test_logseries(self):
        p = [0.5]
        bad_p_one = [2]
        bad_p_two = [-1]
        desired = np.array([1, 1, 1])

        random = Generator(MT19937(self.seed))
        logseries = random.logseries
        actual = logseries(p * 3)
        assert_array_equal(actual, desired)
        assert_raises(ValueError, logseries, bad_p_one * 3)
        assert_raises(ValueError, logseries, bad_p_two * 3)

    def test_multinomial(self):
        random = Generator(MT19937(self.seed))
        actual = random.multinomial([5, 20], [1 / 6.] * 6, size=(3, 2))
        desired = np.array([[[0, 0, 2, 1, 2, 0],
                             [2, 3, 6, 4, 2, 3]],
                            [[1, 0, 1, 0, 2, 1],
                             [7, 2, 2, 1, 4, 4]],
                            [[0, 2, 0, 1, 2, 0],
                             [3, 2, 3, 3, 4, 5]]], dtype=np.int64)
        assert_array_equal(actual, desired)

        random = Generator(MT19937(self.seed))
        actual = random.multinomial([5, 20], [1 / 6.] * 6)
        desired = np.array([[0, 0, 2, 1, 2, 0],
                            [2, 3, 6, 4, 2, 3]], dtype=np.int64)
        assert_array_equal(actual, desired)

        random = Generator(MT19937(self.seed))
        actual = random.multinomial([5, 20], [[1 / 6.] * 6] * 2)
        desired = np.array([[0, 0, 2, 1, 2, 0],
                            [2, 3, 6, 4, 2, 3]], dtype=np.int64)
        assert_array_equal(actual, desired)

        random = Generator(MT19937(self.seed))
        actual = random.multinomial([[5], [20]], [[1 / 6.] * 6] * 2)
        desired = np.array([[[0, 0, 2, 1, 2, 0],
                             [0, 0, 2, 1, 1, 1]],
                            [[4, 2, 3, 3, 5, 3],
                             [7, 2, 2, 1, 4, 4]]], dtype=np.int64)
        assert_array_equal(actual, desired)

    @pytest.mark.parametrize("n", [10,
                                   np.array([10, 10]),
                                   np.array([[[10]], [[10]]])
                                   ]
                             )
    def test_multinomial_pval_broadcast(self, n):
        random = Generator(MT19937(self.seed))
        pvals = np.array([1 / 4] * 4)
        actual = random.multinomial(n, pvals)
        n_shape = tuple() if isinstance(n, int) else n.shape
        expected_shape = n_shape + (4,)
        assert actual.shape == expected_shape
        pvals = np.vstack([pvals, pvals])
        actual = random.multinomial(n, pvals)
        expected_shape = np.broadcast_shapes(n_shape, pvals.shape[:-1]) + (4,)
        assert actual.shape == expected_shape

        pvals = np.vstack([[pvals], [pvals]])
        actual = random.multinomial(n, pvals)
        expected_shape = np.broadcast_shapes(n_shape, pvals.shape[:-1])
        assert actual.shape == expected_shape + (4,)
        actual = random.multinomial(n, pvals, size=(3, 2) + expected_shape)
        assert actual.shape == (3, 2) + expected_shape + (4,)

        with pytest.raises(ValueError):
            # Ensure that size is not broadcast
            actual = random.multinomial(n, pvals, size=(1,) * 6)

    def test_invalid_pvals_broadcast(self):
        random = Generator(MT19937(self.seed))
        pvals = [[1 / 6] * 6, [1 / 4] * 6]
        assert_raises(ValueError, random.multinomial, 1, pvals)
        assert_raises(ValueError, random.multinomial, 6, 0.5)

    def test_empty_outputs(self):
        random = Generator(MT19937(self.seed))
        actual = random.multinomial(np.empty((10, 0, 6), "i8"), [1 / 6] * 6)
        assert actual.shape == (10, 0, 6, 6)
        actual = random.multinomial(12, np.empty((10, 0, 10)))
        assert actual.shape == (10, 0, 10)
        actual = random.multinomial(np.empty((3, 0, 7), "i8"),
                                    np.empty((3, 0, 7, 4)))
        assert actual.shape == (3, 0, 7, 4)


@pytest.mark.skipif(IS_WASM, reason="can't start thread")
class TestThread:
    # make sure each state produces the same sequence even in threads
    def setup_method(self):
        self.seeds = range(4)

    def check_function(self, function, sz):
        from threading import Thread

        out1 = np.empty((len(self.seeds),) + sz)
        out2 = np.empty((len(self.seeds),) + sz)

        # threaded generation
        t = [Thread(target=function, args=(Generator(MT19937(s)), o))
             for s, o in zip(self.seeds, out1)]
        [x.start() for x in t]
        [x.join() for x in t]

        # the same serial
        for s, o in zip(self.seeds, out2):
            function(Generator(MT19937(s)), o)

        # these platforms change x87 fpu precision mode in threads
        if np.intp().dtype.itemsize == 4 and sys.platform == "win32":
            assert_array_almost_equal(out1, out2)
        else:
            assert_array_equal(out1, out2)

    def test_normal(self):
        def gen_random(state, out):
            out[...] = state.normal(size=10000)

        self.check_function(gen_random, sz=(10000,))

    def test_exp(self):
        def gen_random(state, out):
            out[...] = state.exponential(scale=np.ones((100, 1000)))

        self.check_function(gen_random, sz=(100, 1000))

    def test_multinomial(self):
        def gen_random(state, out):
            out[...] = state.multinomial(10, [1 / 6.] * 6, size=10000)

        self.check_function(gen_random, sz=(10000, 6))


# See Issue #4263
class TestSingleEltArrayInput:
    def setup_method(self):
        self.argOne = np.array([2])
        self.argTwo = np.array([3])
        self.argThree = np.array([4])
        self.tgtShape = (1,)

    def test_one_arg_funcs(self):
        funcs = (random.exponential, random.standard_gamma,
                 random.chisquare, random.standard_t,
                 random.pareto, random.weibull,
                 random.power, random.rayleigh,
                 random.poisson, random.zipf,
                 random.geometric, random.logseries)

        probfuncs = (random.geometric, random.logseries)

        for func in funcs:
            if func in probfuncs:  # p < 1.0
                out = func(np.array([0.5]))

            else:
                out = func(self.argOne)

            assert_equal(out.shape, self.tgtShape)

    def test_two_arg_funcs(self):
        funcs = (random.uniform, random.normal,
                 random.beta, random.gamma,
                 random.f, random.noncentral_chisquare,
                 random.vonmises, random.laplace,
                 random.gumbel, random.logistic,
                 random.lognormal, random.wald,
                 random.binomial, random.negative_binomial)

        probfuncs = (random.binomial, random.negative_binomial)

        for func in funcs:
            if func in probfuncs:  # p <= 1
                argTwo = np.array([0.5])

            else:
                argTwo = self.argTwo

            out = func(self.argOne, argTwo)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne[0], argTwo)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne, argTwo[0])
            assert_equal(out.shape, self.tgtShape)

    def test_integers(self, endpoint):
        itype = [np.bool_, np.int8, np.uint8, np.int16, np.uint16,
                 np.int32, np.uint32, np.int64, np.uint64]
        func = random.integers
        high = np.array([1])
        low = np.array([0])

        for dt in itype:
            out = func(low, high, endpoint=endpoint, dtype=dt)
            assert_equal(out.shape, self.tgtShape)

            out = func(low[0], high, endpoint=endpoint, dtype=dt)
            assert_equal(out.shape, self.tgtShape)

            out = func(low, high[0], endpoint=endpoint, dtype=dt)
            assert_equal(out.shape, self.tgtShape)

    def test_three_arg_funcs(self):
        funcs = [random.noncentral_f, random.triangular,
                 random.hypergeometric]

        for func in funcs:
            out = func(self.argOne, self.argTwo, self.argThree)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne[0], self.argTwo, self.argThree)
            assert_equal(out.shape, self.tgtShape)

            out = func(self.argOne, self.argTwo[0], self.argThree)
            assert_equal(out.shape, self.tgtShape)


@pytest.mark.parametrize("config", JUMP_TEST_DATA)
def test_jumped(config):
    # Each config contains the initial seed, a number of raw steps
    # the sha256 hashes of the initial and the final states' keys and
    # the position of the initial and the final state.
    # These were produced using the original C implementation.
    seed = config["seed"]
    steps = config["steps"]

    mt19937 = MT19937(seed)
    # Burn step
    mt19937.random_raw(steps)
    key = mt19937.state["state"]["key"]
    if sys.byteorder == 'big':
        key = key.byteswap()
    sha256 = hashlib.sha256(key)
    assert mt19937.state["state"]["pos"] == config["initial"]["pos"]
    assert sha256.hexdigest() == config["initial"]["key_sha256"]

    jumped = mt19937.jumped()
    key = jumped.state["state"]["key"]
    if sys.byteorder == 'big':
        key = key.byteswap()
    sha256 = hashlib.sha256(key)
    assert jumped.state["state"]["pos"] == config["jumped"]["pos"]
    assert sha256.hexdigest() == config["jumped"]["key_sha256"]


def test_broadcast_size_error():
    mu = np.ones(3)
    sigma = np.ones((4, 3))
    size = (10, 4, 2)
    assert random.normal(mu, sigma, size=(5, 4, 3)).shape == (5, 4, 3)
    with pytest.raises(ValueError):
        random.normal(mu, sigma, size=size)
    with pytest.raises(ValueError):
        random.normal(mu, sigma, size=(1, 3))
    with pytest.raises(ValueError):
        random.normal(mu, sigma, size=(4, 1, 1))
    # 1 arg
    shape = np.ones((4, 3))
    with pytest.raises(ValueError):
        random.standard_gamma(shape, size=size)
    with pytest.raises(ValueError):
        random.standard_gamma(shape, size=(3,))
    with pytest.raises(ValueError):
        random.standard_gamma(shape, size=3)
    # Check out
    out = np.empty(size)
    with pytest.raises(ValueError):
        random.standard_gamma(shape, out=out)

    # 2 arg
    with pytest.raises(ValueError):
        random.binomial(1, [0.3, 0.7], size=(2, 1))
    with pytest.raises(ValueError):
        random.binomial([1, 2], 0.3, size=(2, 1))
    with pytest.raises(ValueError):
        random.binomial([1, 2], [0.3, 0.7], size=(2, 1))
    with pytest.raises(ValueError):
        random.multinomial([2, 2], [.3, .7], size=(2, 1))

    # 3 arg
    a = random.chisquare(5, size=3)
    b = random.chisquare(5, size=(4, 3))
    c = random.chisquare(5, size=(5, 4, 3))
    assert random.noncentral_f(a, b, c).shape == (5, 4, 3)
    with pytest.raises(ValueError, match=r"Output size \(6, 5, 1, 1\) is"):
        random.noncentral_f(a, b, c, size=(6, 5, 1, 1))


def test_broadcast_size_scalar():
    mu = np.ones(3)
    sigma = np.ones(3)
    random.normal(mu, sigma, size=3)
    with pytest.raises(ValueError):
        random.normal(mu, sigma, size=2)


def test_ragged_shuffle():
    # GH 18142
    seq = [[], [], 1]
    gen = Generator(MT19937(0))
    assert_no_warnings(gen.shuffle, seq)
    assert seq == [1, [], []]


@pytest.mark.parametrize("high", [-2, [-2]])
@pytest.mark.parametrize("endpoint", [True, False])
def test_single_arg_integer_exception(high, endpoint):
    # GH 14333
    gen = Generator(MT19937(0))
    msg = 'high < 0' if endpoint else 'high <= 0'
    with pytest.raises(ValueError, match=msg):
        gen.integers(high, endpoint=endpoint)
    msg = 'low > high' if endpoint else 'low >= high'
    with pytest.raises(ValueError, match=msg):
        gen.integers(-1, high, endpoint=endpoint)
    with pytest.raises(ValueError, match=msg):
        gen.integers([-1], high, endpoint=endpoint)


@pytest.mark.parametrize("dtype", ["f4", "f8"])
def test_c_contig_req_out(dtype):
    # GH 18704
    out = np.empty((2, 3), order="F", dtype=dtype)
    shape = [1, 2, 3]
    with pytest.raises(ValueError, match="Supplied output array"):
        random.standard_gamma(shape, out=out, dtype=dtype)
    with pytest.raises(ValueError, match="Supplied output array"):
        random.standard_gamma(shape, out=out, size=out.shape, dtype=dtype)


@pytest.mark.parametrize("dtype", ["f4", "f8"])
@pytest.mark.parametrize("order", ["F", "C"])
@pytest.mark.parametrize("dist", [random.standard_normal, random.random])
def test_contig_req_out(dist, order, dtype):
    # GH 18704
    out = np.empty((2, 3), dtype=dtype, order=order)
    variates = dist(out=out, dtype=dtype)
    assert variates is out
    variates = dist(out=out, dtype=dtype, size=out.shape)
    assert variates is out


def test_generator_ctor_old_style_pickle():
    rg = np.random.Generator(np.random.PCG64DXSM(0))
    rg.standard_normal(1)
    # Directly call reduce which is used in pickling
    ctor, args, state_a = rg.__reduce__()
    # Simulate unpickling an old pickle that only has the name
    assert args[:1] == ("PCG64DXSM",)
    b = ctor(*args[:1])
    b.bit_generator.state = state_a
    state_b = b.bit_generator.state
    assert state_a == state_b
