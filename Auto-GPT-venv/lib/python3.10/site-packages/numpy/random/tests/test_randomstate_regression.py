import sys

import pytest

from numpy.testing import (
    assert_, assert_array_equal, assert_raises,
    )
import numpy as np

from numpy import random


class TestRegression:

    def test_VonMises_range(self):
        # Make sure generated random variables are in [-pi, pi].
        # Regression test for ticket #986.
        for mu in np.linspace(-7., 7., 5):
            r = random.vonmises(mu, 1, 50)
            assert_(np.all(r > -np.pi) and np.all(r <= np.pi))

    def test_hypergeometric_range(self):
        # Test for ticket #921
        assert_(np.all(random.hypergeometric(3, 18, 11, size=10) < 4))
        assert_(np.all(random.hypergeometric(18, 3, 11, size=10) > 0))

        # Test for ticket #5623
        args = [
            (2**20 - 2, 2**20 - 2, 2**20 - 2),  # Check for 32-bit systems
        ]
        is_64bits = sys.maxsize > 2**32
        if is_64bits and sys.platform != 'win32':
            # Check for 64-bit systems
            args.append((2**40 - 2, 2**40 - 2, 2**40 - 2))
        for arg in args:
            assert_(random.hypergeometric(*arg) > 0)

    def test_logseries_convergence(self):
        # Test for ticket #923
        N = 1000
        random.seed(0)
        rvsn = random.logseries(0.8, size=N)
        # these two frequency counts should be close to theoretical
        # numbers with this large sample
        # theoretical large N result is 0.49706795
        freq = np.sum(rvsn == 1) / N
        msg = f'Frequency was {freq:f}, should be > 0.45'
        assert_(freq > 0.45, msg)
        # theoretical large N result is 0.19882718
        freq = np.sum(rvsn == 2) / N
        msg = f'Frequency was {freq:f}, should be < 0.23'
        assert_(freq < 0.23, msg)

    def test_shuffle_mixed_dimension(self):
        # Test for trac ticket #2074
        for t in [[1, 2, 3, None],
                  [(1, 1), (2, 2), (3, 3), None],
                  [1, (2, 2), (3, 3), None],
                  [(1, 1), 2, 3, None]]:
            random.seed(12345)
            shuffled = list(t)
            random.shuffle(shuffled)
            expected = np.array([t[0], t[3], t[1], t[2]], dtype=object)
            assert_array_equal(np.array(shuffled, dtype=object), expected)

    def test_call_within_randomstate(self):
        # Check that custom RandomState does not call into global state
        m = random.RandomState()
        res = np.array([0, 8, 7, 2, 1, 9, 4, 7, 0, 3])
        for i in range(3):
            random.seed(i)
            m.seed(4321)
            # If m.state is not honored, the result will change
            assert_array_equal(m.choice(10, size=10, p=np.ones(10)/10.), res)

    def test_multivariate_normal_size_types(self):
        # Test for multivariate_normal issue with 'size' argument.
        # Check that the multivariate_normal size argument can be a
        # numpy integer.
        random.multivariate_normal([0], [[0]], size=1)
        random.multivariate_normal([0], [[0]], size=np.int_(1))
        random.multivariate_normal([0], [[0]], size=np.int64(1))

    def test_beta_small_parameters(self):
        # Test that beta with small a and b parameters does not produce
        # NaNs due to roundoff errors causing 0 / 0, gh-5851
        random.seed(1234567890)
        x = random.beta(0.0001, 0.0001, size=100)
        assert_(not np.any(np.isnan(x)), 'Nans in random.beta')

    def test_choice_sum_of_probs_tolerance(self):
        # The sum of probs should be 1.0 with some tolerance.
        # For low precision dtypes the tolerance was too tight.
        # See numpy github issue 6123.
        random.seed(1234)
        a = [1, 2, 3]
        counts = [4, 4, 2]
        for dt in np.float16, np.float32, np.float64:
            probs = np.array(counts, dtype=dt) / sum(counts)
            c = random.choice(a, p=probs)
            assert_(c in a)
            assert_raises(ValueError, random.choice, a, p=probs*0.9)

    def test_shuffle_of_array_of_different_length_strings(self):
        # Test that permuting an array of different length strings
        # will not cause a segfault on garbage collection
        # Tests gh-7710
        random.seed(1234)

        a = np.array(['a', 'a' * 1000])

        for _ in range(100):
            random.shuffle(a)

        # Force Garbage Collection - should not segfault.
        import gc
        gc.collect()

    def test_shuffle_of_array_of_objects(self):
        # Test that permuting an array of objects will not cause
        # a segfault on garbage collection.
        # See gh-7719
        random.seed(1234)
        a = np.array([np.arange(1), np.arange(4)], dtype=object)

        for _ in range(1000):
            random.shuffle(a)

        # Force Garbage Collection - should not segfault.
        import gc
        gc.collect()

    def test_permutation_subclass(self):
        class N(np.ndarray):
            pass

        random.seed(1)
        orig = np.arange(3).view(N)
        perm = random.permutation(orig)
        assert_array_equal(perm, np.array([0, 2, 1]))
        assert_array_equal(orig, np.arange(3).view(N))

        class M:
            a = np.arange(5)

            def __array__(self):
                return self.a

        random.seed(1)
        m = M()
        perm = random.permutation(m)
        assert_array_equal(perm, np.array([2, 1, 4, 0, 3]))
        assert_array_equal(m.__array__(), np.arange(5))

    def test_warns_byteorder(self):
        # GH 13159
        other_byteord_dt = '<i4' if sys.byteorder == 'big' else '>i4'
        with pytest.deprecated_call(match='non-native byteorder is not'):
            random.randint(0, 200, size=10, dtype=other_byteord_dt)

    def test_named_argument_initialization(self):
        # GH 13669
        rs1 = np.random.RandomState(123456789)
        rs2 = np.random.RandomState(seed=123456789)
        assert rs1.randint(0, 100) == rs2.randint(0, 100)

    def test_choice_retun_dtype(self):
        # GH 9867
        c = np.random.choice(10, p=[.1]*10, size=2)
        assert c.dtype == np.dtype(int)
        c = np.random.choice(10, p=[.1]*10, replace=False, size=2)
        assert c.dtype == np.dtype(int)
        c = np.random.choice(10, size=2)
        assert c.dtype == np.dtype(int)
        c = np.random.choice(10, replace=False, size=2)
        assert c.dtype == np.dtype(int)

    @pytest.mark.skipif(np.iinfo('l').max < 2**32,
                        reason='Cannot test with 32-bit C long')
    def test_randint_117(self):
        # GH 14189
        random.seed(0)
        expected = np.array([2357136044, 2546248239, 3071714933, 3626093760,
                             2588848963, 3684848379, 2340255427, 3638918503,
                             1819583497, 2678185683], dtype='int64')
        actual = random.randint(2**32, size=10)
        assert_array_equal(actual, expected)

    def test_p_zero_stream(self):
        # Regression test for gh-14522.  Ensure that future versions
        # generate the same variates as version 1.16.
        np.random.seed(12345)
        assert_array_equal(random.binomial(1, [0, 0.25, 0.5, 0.75, 1]),
                           [0, 0, 0, 1, 1])

    def test_n_zero_stream(self):
        # Regression test for gh-14522.  Ensure that future versions
        # generate the same variates as version 1.16.
        np.random.seed(8675309)
        expected = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [3, 4, 2, 3, 3, 1, 5, 3, 1, 3]])
        assert_array_equal(random.binomial([[0], [10]], 0.25, size=(2, 10)),
                           expected)


def test_multinomial_empty():
    # gh-20483
    # Ensure that empty p-vals are correctly handled
    assert random.multinomial(10, []).shape == (0,)
    assert random.multinomial(3, [], size=(7, 5, 3)).shape == (7, 5, 3, 0)


def test_multinomial_1d_pval():
    # gh-20483
    with pytest.raises(TypeError, match="pvals must be a 1-d"):
        random.multinomial(10, 0.3)
