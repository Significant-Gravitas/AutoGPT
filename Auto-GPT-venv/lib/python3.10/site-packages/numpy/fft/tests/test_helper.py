"""Test functions for fftpack.helper module

Copied from fftpack.helper by Pearu Peterson, October 2005

"""
import numpy as np
from numpy.testing import assert_array_almost_equal
from numpy import fft, pi


class TestFFTShift:

    def test_definition(self):
        x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
        y = [-4, -3, -2, -1, 0, 1, 2, 3, 4]
        assert_array_almost_equal(fft.fftshift(x), y)
        assert_array_almost_equal(fft.ifftshift(y), x)
        x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        y = [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4]
        assert_array_almost_equal(fft.fftshift(x), y)
        assert_array_almost_equal(fft.ifftshift(y), x)

    def test_inverse(self):
        for n in [1, 4, 9, 100, 211]:
            x = np.random.random((n,))
            assert_array_almost_equal(fft.ifftshift(fft.fftshift(x)), x)

    def test_axes_keyword(self):
        freqs = [[0, 1, 2], [3, 4, -4], [-3, -2, -1]]
        shifted = [[-1, -3, -2], [2, 0, 1], [-4, 3, 4]]
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0, 1)), shifted)
        assert_array_almost_equal(fft.fftshift(freqs, axes=0),
                                  fft.fftshift(freqs, axes=(0,)))
        assert_array_almost_equal(fft.ifftshift(shifted, axes=(0, 1)), freqs)
        assert_array_almost_equal(fft.ifftshift(shifted, axes=0),
                                  fft.ifftshift(shifted, axes=(0,)))

        assert_array_almost_equal(fft.fftshift(freqs), shifted)
        assert_array_almost_equal(fft.ifftshift(shifted), freqs)

    def test_uneven_dims(self):
        """ Test 2D input, which has uneven dimension sizes """
        freqs = [
            [0, 1],
            [2, 3],
            [4, 5]
        ]

        # shift in dimension 0
        shift_dim0 = [
            [4, 5],
            [0, 1],
            [2, 3]
        ]
        assert_array_almost_equal(fft.fftshift(freqs, axes=0), shift_dim0)
        assert_array_almost_equal(fft.ifftshift(shift_dim0, axes=0), freqs)
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0,)), shift_dim0)
        assert_array_almost_equal(fft.ifftshift(shift_dim0, axes=[0]), freqs)

        # shift in dimension 1
        shift_dim1 = [
            [1, 0],
            [3, 2],
            [5, 4]
        ]
        assert_array_almost_equal(fft.fftshift(freqs, axes=1), shift_dim1)
        assert_array_almost_equal(fft.ifftshift(shift_dim1, axes=1), freqs)

        # shift in both dimensions
        shift_dim_both = [
            [5, 4],
            [1, 0],
            [3, 2]
        ]
        assert_array_almost_equal(fft.fftshift(freqs, axes=(0, 1)), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=(0, 1)), freqs)
        assert_array_almost_equal(fft.fftshift(freqs, axes=[0, 1]), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=[0, 1]), freqs)

        # axes=None (default) shift in all dimensions
        assert_array_almost_equal(fft.fftshift(freqs, axes=None), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both, axes=None), freqs)
        assert_array_almost_equal(fft.fftshift(freqs), shift_dim_both)
        assert_array_almost_equal(fft.ifftshift(shift_dim_both), freqs)

    def test_equal_to_original(self):
        """ Test that the new (>=v1.15) implementation (see #10073) is equal to the original (<=v1.14) """
        from numpy.core import asarray, concatenate, arange, take

        def original_fftshift(x, axes=None):
            """ How fftshift was implemented in v1.14"""
            tmp = asarray(x)
            ndim = tmp.ndim
            if axes is None:
                axes = list(range(ndim))
            elif isinstance(axes, int):
                axes = (axes,)
            y = tmp
            for k in axes:
                n = tmp.shape[k]
                p2 = (n + 1) // 2
                mylist = concatenate((arange(p2, n), arange(p2)))
                y = take(y, mylist, k)
            return y

        def original_ifftshift(x, axes=None):
            """ How ifftshift was implemented in v1.14 """
            tmp = asarray(x)
            ndim = tmp.ndim
            if axes is None:
                axes = list(range(ndim))
            elif isinstance(axes, int):
                axes = (axes,)
            y = tmp
            for k in axes:
                n = tmp.shape[k]
                p2 = n - (n + 1) // 2
                mylist = concatenate((arange(p2, n), arange(p2)))
                y = take(y, mylist, k)
            return y

        # create possible 2d array combinations and try all possible keywords
        # compare output to original functions
        for i in range(16):
            for j in range(16):
                for axes_keyword in [0, 1, None, (0,), (0, 1)]:
                    inp = np.random.rand(i, j)

                    assert_array_almost_equal(fft.fftshift(inp, axes_keyword),
                                              original_fftshift(inp, axes_keyword))

                    assert_array_almost_equal(fft.ifftshift(inp, axes_keyword),
                                              original_ifftshift(inp, axes_keyword))


class TestFFTFreq:

    def test_definition(self):
        x = [0, 1, 2, 3, 4, -4, -3, -2, -1]
        assert_array_almost_equal(9*fft.fftfreq(9), x)
        assert_array_almost_equal(9*pi*fft.fftfreq(9, pi), x)
        x = [0, 1, 2, 3, 4, -5, -4, -3, -2, -1]
        assert_array_almost_equal(10*fft.fftfreq(10), x)
        assert_array_almost_equal(10*pi*fft.fftfreq(10, pi), x)


class TestRFFTFreq:

    def test_definition(self):
        x = [0, 1, 2, 3, 4]
        assert_array_almost_equal(9*fft.rfftfreq(9), x)
        assert_array_almost_equal(9*pi*fft.rfftfreq(9, pi), x)
        x = [0, 1, 2, 3, 4, 5]
        assert_array_almost_equal(10*fft.rfftfreq(10), x)
        assert_array_almost_equal(10*pi*fft.rfftfreq(10, pi), x)


class TestIRFFTN:

    def test_not_last_axis_success(self):
        ar, ai = np.random.random((2, 16, 8, 32))
        a = ar + 1j*ai

        axes = (-2,)

        # Should not raise error
        fft.irfftn(a, axes=axes)
