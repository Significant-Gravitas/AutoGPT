import numpy as np
import functools
import sys
import pytest

from numpy.lib.shape_base import (
    apply_along_axis, apply_over_axes, array_split, split, hsplit, dsplit,
    vsplit, dstack, column_stack, kron, tile, expand_dims, take_along_axis,
    put_along_axis
    )
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_raises, assert_warns
    )


IS_64BIT = sys.maxsize > 2**32


def _add_keepdims(func):
    """ hack in keepdims behavior into a function taking an axis """
    @functools.wraps(func)
    def wrapped(a, axis, **kwargs):
        res = func(a, axis=axis, **kwargs)
        if axis is None:
            axis = 0  # res is now a scalar, so we can insert this anywhere
        return np.expand_dims(res, axis=axis)
    return wrapped


class TestTakeAlongAxis:
    def test_argequivalent(self):
        """ Test it translates from arg<func> to <func> """
        from numpy.random import rand
        a = rand(3, 4, 5)

        funcs = [
            (np.sort, np.argsort, dict()),
            (_add_keepdims(np.min), _add_keepdims(np.argmin), dict()),
            (_add_keepdims(np.max), _add_keepdims(np.argmax), dict()),
            (np.partition, np.argpartition, dict(kth=2)),
        ]

        for func, argfunc, kwargs in funcs:
            for axis in list(range(a.ndim)) + [None]:
                a_func = func(a, axis=axis, **kwargs)
                ai_func = argfunc(a, axis=axis, **kwargs)
                assert_equal(a_func, take_along_axis(a, ai_func, axis=axis))

    def test_invalid(self):
        """ Test it errors when indices has too few dimensions """
        a = np.ones((10, 10))
        ai = np.ones((10, 2), dtype=np.intp)

        # sanity check
        take_along_axis(a, ai, axis=1)

        # not enough indices
        assert_raises(ValueError, take_along_axis, a, np.array(1), axis=1)
        # bool arrays not allowed
        assert_raises(IndexError, take_along_axis, a, ai.astype(bool), axis=1)
        # float arrays not allowed
        assert_raises(IndexError, take_along_axis, a, ai.astype(float), axis=1)
        # invalid axis
        assert_raises(np.AxisError, take_along_axis, a, ai, axis=10)

    def test_empty(self):
        """ Test everything is ok with empty results, even with inserted dims """
        a  = np.ones((3, 4, 5))
        ai = np.ones((3, 0, 5), dtype=np.intp)

        actual = take_along_axis(a, ai, axis=1)
        assert_equal(actual.shape, ai.shape)

    def test_broadcast(self):
        """ Test that non-indexing dimensions are broadcast in both directions """
        a  = np.ones((3, 4, 1))
        ai = np.ones((1, 2, 5), dtype=np.intp)
        actual = take_along_axis(a, ai, axis=1)
        assert_equal(actual.shape, (3, 2, 5))


class TestPutAlongAxis:
    def test_replace_max(self):
        a_base = np.array([[10, 30, 20], [60, 40, 50]])

        for axis in list(range(a_base.ndim)) + [None]:
            # we mutate this in the loop
            a = a_base.copy()

            # replace the max with a small value
            i_max = _add_keepdims(np.argmax)(a, axis=axis)
            put_along_axis(a, i_max, -99, axis=axis)

            # find the new minimum, which should max
            i_min = _add_keepdims(np.argmin)(a, axis=axis)

            assert_equal(i_min, i_max)

    def test_broadcast(self):
        """ Test that non-indexing dimensions are broadcast in both directions """
        a  = np.ones((3, 4, 1))
        ai = np.arange(10, dtype=np.intp).reshape((1, 2, 5)) % 4
        put_along_axis(a, ai, 20, axis=1)
        assert_equal(take_along_axis(a, ai, axis=1), 20)


class TestApplyAlongAxis:
    def test_simple(self):
        a = np.ones((20, 10), 'd')
        assert_array_equal(
            apply_along_axis(len, 0, a), len(a)*np.ones(a.shape[1]))

    def test_simple101(self):
        a = np.ones((10, 101), 'd')
        assert_array_equal(
            apply_along_axis(len, 0, a), len(a)*np.ones(a.shape[1]))

    def test_3d(self):
        a = np.arange(27).reshape((3, 3, 3))
        assert_array_equal(apply_along_axis(np.sum, 0, a),
                           [[27, 30, 33], [36, 39, 42], [45, 48, 51]])

    def test_preserve_subclass(self):
        def double(row):
            return row * 2

        class MyNDArray(np.ndarray):
            pass

        m = np.array([[0, 1], [2, 3]]).view(MyNDArray)
        expected = np.array([[0, 2], [4, 6]]).view(MyNDArray)

        result = apply_along_axis(double, 0, m)
        assert_(isinstance(result, MyNDArray))
        assert_array_equal(result, expected)

        result = apply_along_axis(double, 1, m)
        assert_(isinstance(result, MyNDArray))
        assert_array_equal(result, expected)

    def test_subclass(self):
        class MinimalSubclass(np.ndarray):
            data = 1

        def minimal_function(array):
            return array.data

        a = np.zeros((6, 3)).view(MinimalSubclass)

        assert_array_equal(
            apply_along_axis(minimal_function, 0, a), np.array([1, 1, 1])
        )

    def test_scalar_array(self, cls=np.ndarray):
        a = np.ones((6, 3)).view(cls)
        res = apply_along_axis(np.sum, 0, a)
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([6, 6, 6]).view(cls))

    def test_0d_array(self, cls=np.ndarray):
        def sum_to_0d(x):
            """ Sum x, returning a 0d array of the same class """
            assert_equal(x.ndim, 1)
            return np.squeeze(np.sum(x, keepdims=True))
        a = np.ones((6, 3)).view(cls)
        res = apply_along_axis(sum_to_0d, 0, a)
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([6, 6, 6]).view(cls))

        res = apply_along_axis(sum_to_0d, 1, a)
        assert_(isinstance(res, cls))
        assert_array_equal(res, np.array([3, 3, 3, 3, 3, 3]).view(cls))

    def test_axis_insertion(self, cls=np.ndarray):
        def f1to2(x):
            """produces an asymmetric non-square matrix from x"""
            assert_equal(x.ndim, 1)
            return (x[::-1] * x[1:,None]).view(cls)

        a2d = np.arange(6*3).reshape((6, 3))

        # 2d insertion along first axis
        actual = apply_along_axis(f1to2, 0, a2d)
        expected = np.stack([
            f1to2(a2d[:,i]) for i in range(a2d.shape[1])
        ], axis=-1).view(cls)
        assert_equal(type(actual), type(expected))
        assert_equal(actual, expected)

        # 2d insertion along last axis
        actual = apply_along_axis(f1to2, 1, a2d)
        expected = np.stack([
            f1to2(a2d[i,:]) for i in range(a2d.shape[0])
        ], axis=0).view(cls)
        assert_equal(type(actual), type(expected))
        assert_equal(actual, expected)

        # 3d insertion along middle axis
        a3d = np.arange(6*5*3).reshape((6, 5, 3))

        actual = apply_along_axis(f1to2, 1, a3d)
        expected = np.stack([
            np.stack([
                f1to2(a3d[i,:,j]) for i in range(a3d.shape[0])
            ], axis=0)
            for j in range(a3d.shape[2])
        ], axis=-1).view(cls)
        assert_equal(type(actual), type(expected))
        assert_equal(actual, expected)

    def test_subclass_preservation(self):
        class MinimalSubclass(np.ndarray):
            pass
        self.test_scalar_array(MinimalSubclass)
        self.test_0d_array(MinimalSubclass)
        self.test_axis_insertion(MinimalSubclass)

    def test_axis_insertion_ma(self):
        def f1to2(x):
            """produces an asymmetric non-square matrix from x"""
            assert_equal(x.ndim, 1)
            res = x[::-1] * x[1:,None]
            return np.ma.masked_where(res%5==0, res)
        a = np.arange(6*3).reshape((6, 3))
        res = apply_along_axis(f1to2, 0, a)
        assert_(isinstance(res, np.ma.masked_array))
        assert_equal(res.ndim, 3)
        assert_array_equal(res[:,:,0].mask, f1to2(a[:,0]).mask)
        assert_array_equal(res[:,:,1].mask, f1to2(a[:,1]).mask)
        assert_array_equal(res[:,:,2].mask, f1to2(a[:,2]).mask)

    def test_tuple_func1d(self):
        def sample_1d(x):
            return x[1], x[0]
        res = np.apply_along_axis(sample_1d, 1, np.array([[1, 2], [3, 4]]))
        assert_array_equal(res, np.array([[2, 1], [4, 3]]))

    def test_empty(self):
        # can't apply_along_axis when there's no chance to call the function
        def never_call(x):
            assert_(False) # should never be reached

        a = np.empty((0, 0))
        assert_raises(ValueError, np.apply_along_axis, never_call, 0, a)
        assert_raises(ValueError, np.apply_along_axis, never_call, 1, a)

        # but it's sometimes ok with some non-zero dimensions
        def empty_to_1(x):
            assert_(len(x) == 0)
            return 1

        a = np.empty((10, 0))
        actual = np.apply_along_axis(empty_to_1, 1, a)
        assert_equal(actual, np.ones(10))
        assert_raises(ValueError, np.apply_along_axis, empty_to_1, 0, a)

    def test_with_iterable_object(self):
        # from issue 5248
        d = np.array([
            [{1, 11}, {2, 22}, {3, 33}],
            [{4, 44}, {5, 55}, {6, 66}]
        ])
        actual = np.apply_along_axis(lambda a: set.union(*a), 0, d)
        expected = np.array([{1, 11, 4, 44}, {2, 22, 5, 55}, {3, 33, 6, 66}])

        assert_equal(actual, expected)

        # issue 8642 - assert_equal doesn't detect this!
        for i in np.ndindex(actual.shape):
            assert_equal(type(actual[i]), type(expected[i]))


class TestApplyOverAxes:
    def test_simple(self):
        a = np.arange(24).reshape(2, 3, 4)
        aoa_a = apply_over_axes(np.sum, a, [0, 2])
        assert_array_equal(aoa_a, np.array([[[60], [92], [124]]]))


class TestExpandDims:
    def test_functionality(self):
        s = (2, 3, 4, 5)
        a = np.empty(s)
        for axis in range(-5, 4):
            b = expand_dims(a, axis)
            assert_(b.shape[axis] == 1)
            assert_(np.squeeze(b).shape == s)

    def test_axis_tuple(self):
        a = np.empty((3, 3, 3))
        assert np.expand_dims(a, axis=(0, 1, 2)).shape == (1, 1, 1, 3, 3, 3)
        assert np.expand_dims(a, axis=(0, -1, -2)).shape == (1, 3, 3, 3, 1, 1)
        assert np.expand_dims(a, axis=(0, 3, 5)).shape == (1, 3, 3, 1, 3, 1)
        assert np.expand_dims(a, axis=(0, -3, -5)).shape == (1, 1, 3, 1, 3, 3)

    def test_axis_out_of_range(self):
        s = (2, 3, 4, 5)
        a = np.empty(s)
        assert_raises(np.AxisError, expand_dims, a, -6)
        assert_raises(np.AxisError, expand_dims, a, 5)

        a = np.empty((3, 3, 3))
        assert_raises(np.AxisError, expand_dims, a, (0, -6))
        assert_raises(np.AxisError, expand_dims, a, (0, 5))

    def test_repeated_axis(self):
        a = np.empty((3, 3, 3))
        assert_raises(ValueError, expand_dims, a, axis=(1, 1))

    def test_subclasses(self):
        a = np.arange(10).reshape((2, 5))
        a = np.ma.array(a, mask=a%3 == 0)

        expanded = np.expand_dims(a, axis=1)
        assert_(isinstance(expanded, np.ma.MaskedArray))
        assert_equal(expanded.shape, (2, 1, 5))
        assert_equal(expanded.mask.shape, (2, 1, 5))


class TestArraySplit:
    def test_integer_0_split(self):
        a = np.arange(10)
        assert_raises(ValueError, array_split, a, 0)

    def test_integer_split(self):
        a = np.arange(10)
        res = array_split(a, 1)
        desired = [np.arange(10)]
        compare_results(res, desired)

        res = array_split(a, 2)
        desired = [np.arange(5), np.arange(5, 10)]
        compare_results(res, desired)

        res = array_split(a, 3)
        desired = [np.arange(4), np.arange(4, 7), np.arange(7, 10)]
        compare_results(res, desired)

        res = array_split(a, 4)
        desired = [np.arange(3), np.arange(3, 6), np.arange(6, 8),
                   np.arange(8, 10)]
        compare_results(res, desired)

        res = array_split(a, 5)
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6),
                   np.arange(6, 8), np.arange(8, 10)]
        compare_results(res, desired)

        res = array_split(a, 6)
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6),
                   np.arange(6, 8), np.arange(8, 9), np.arange(9, 10)]
        compare_results(res, desired)

        res = array_split(a, 7)
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 6),
                   np.arange(6, 7), np.arange(7, 8), np.arange(8, 9),
                   np.arange(9, 10)]
        compare_results(res, desired)

        res = array_split(a, 8)
        desired = [np.arange(2), np.arange(2, 4), np.arange(4, 5),
                   np.arange(5, 6), np.arange(6, 7), np.arange(7, 8),
                   np.arange(8, 9), np.arange(9, 10)]
        compare_results(res, desired)

        res = array_split(a, 9)
        desired = [np.arange(2), np.arange(2, 3), np.arange(3, 4),
                   np.arange(4, 5), np.arange(5, 6), np.arange(6, 7),
                   np.arange(7, 8), np.arange(8, 9), np.arange(9, 10)]
        compare_results(res, desired)

        res = array_split(a, 10)
        desired = [np.arange(1), np.arange(1, 2), np.arange(2, 3),
                   np.arange(3, 4), np.arange(4, 5), np.arange(5, 6),
                   np.arange(6, 7), np.arange(7, 8), np.arange(8, 9),
                   np.arange(9, 10)]
        compare_results(res, desired)

        res = array_split(a, 11)
        desired = [np.arange(1), np.arange(1, 2), np.arange(2, 3),
                   np.arange(3, 4), np.arange(4, 5), np.arange(5, 6),
                   np.arange(6, 7), np.arange(7, 8), np.arange(8, 9),
                   np.arange(9, 10), np.array([])]
        compare_results(res, desired)

    def test_integer_split_2D_rows(self):
        a = np.array([np.arange(10), np.arange(10)])
        res = array_split(a, 3, axis=0)
        tgt = [np.array([np.arange(10)]), np.array([np.arange(10)]),
                   np.zeros((0, 10))]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)

        # Same thing for manual splits:
        res = array_split(a, [0, 1], axis=0)
        tgt = [np.zeros((0, 10)), np.array([np.arange(10)]),
               np.array([np.arange(10)])]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)

    def test_integer_split_2D_cols(self):
        a = np.array([np.arange(10), np.arange(10)])
        res = array_split(a, 3, axis=-1)
        desired = [np.array([np.arange(4), np.arange(4)]),
                   np.array([np.arange(4, 7), np.arange(4, 7)]),
                   np.array([np.arange(7, 10), np.arange(7, 10)])]
        compare_results(res, desired)

    def test_integer_split_2D_default(self):
        """ This will fail if we change default axis
        """
        a = np.array([np.arange(10), np.arange(10)])
        res = array_split(a, 3)
        tgt = [np.array([np.arange(10)]), np.array([np.arange(10)]),
                   np.zeros((0, 10))]
        compare_results(res, tgt)
        assert_(a.dtype.type is res[-1].dtype.type)
        # perhaps should check higher dimensions

    @pytest.mark.skipif(not IS_64BIT, reason="Needs 64bit platform")
    def test_integer_split_2D_rows_greater_max_int32(self):
        a = np.broadcast_to([0], (1 << 32, 2))
        res = array_split(a, 4)
        chunk = np.broadcast_to([0], (1 << 30, 2))
        tgt = [chunk] * 4
        for i in range(len(tgt)):
            assert_equal(res[i].shape, tgt[i].shape)

    def test_index_split_simple(self):
        a = np.arange(10)
        indices = [1, 5, 7]
        res = array_split(a, indices, axis=-1)
        desired = [np.arange(0, 1), np.arange(1, 5), np.arange(5, 7),
                   np.arange(7, 10)]
        compare_results(res, desired)

    def test_index_split_low_bound(self):
        a = np.arange(10)
        indices = [0, 5, 7]
        res = array_split(a, indices, axis=-1)
        desired = [np.array([]), np.arange(0, 5), np.arange(5, 7),
                   np.arange(7, 10)]
        compare_results(res, desired)

    def test_index_split_high_bound(self):
        a = np.arange(10)
        indices = [0, 5, 7, 10, 12]
        res = array_split(a, indices, axis=-1)
        desired = [np.array([]), np.arange(0, 5), np.arange(5, 7),
                   np.arange(7, 10), np.array([]), np.array([])]
        compare_results(res, desired)


class TestSplit:
    # The split function is essentially the same as array_split,
    # except that it test if splitting will result in an
    # equal split.  Only test for this case.

    def test_equal_split(self):
        a = np.arange(10)
        res = split(a, 2)
        desired = [np.arange(5), np.arange(5, 10)]
        compare_results(res, desired)

    def test_unequal_split(self):
        a = np.arange(10)
        assert_raises(ValueError, split, a, 3)


class TestColumnStack:
    def test_non_iterable(self):
        assert_raises(TypeError, column_stack, 1)

    def test_1D_arrays(self):
        # example from docstring
        a = np.array((1, 2, 3))
        b = np.array((2, 3, 4))
        expected = np.array([[1, 2],
                             [2, 3],
                             [3, 4]])
        actual = np.column_stack((a, b))
        assert_equal(actual, expected)

    def test_2D_arrays(self):
        # same as hstack 2D docstring example
        a = np.array([[1], [2], [3]])
        b = np.array([[2], [3], [4]])
        expected = np.array([[1, 2],
                             [2, 3],
                             [3, 4]])
        actual = np.column_stack((a, b))
        assert_equal(actual, expected)

    def test_generator(self):
        with assert_warns(FutureWarning):
            column_stack((np.arange(3) for _ in range(2)))


class TestDstack:
    def test_non_iterable(self):
        assert_raises(TypeError, dstack, 1)

    def test_0D_array(self):
        a = np.array(1)
        b = np.array(2)
        res = dstack([a, b])
        desired = np.array([[[1, 2]]])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = np.array([1])
        b = np.array([2])
        res = dstack([a, b])
        desired = np.array([[[1, 2]]])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = np.array([[1], [2]])
        b = np.array([[1], [2]])
        res = dstack([a, b])
        desired = np.array([[[1, 1]], [[2, 2, ]]])
        assert_array_equal(res, desired)

    def test_2D_array2(self):
        a = np.array([1, 2])
        b = np.array([1, 2])
        res = dstack([a, b])
        desired = np.array([[[1, 1], [2, 2]]])
        assert_array_equal(res, desired)

    def test_generator(self):
        with assert_warns(FutureWarning):
            dstack((np.arange(3) for _ in range(2)))


# array_split has more comprehensive test of splitting.
# only do simple test on hsplit, vsplit, and dsplit
class TestHsplit:
    """Only testing for integer splits.

    """
    def test_non_iterable(self):
        assert_raises(ValueError, hsplit, 1, 1)

    def test_0D_array(self):
        a = np.array(1)
        try:
            hsplit(a, 2)
            assert_(0)
        except ValueError:
            pass

    def test_1D_array(self):
        a = np.array([1, 2, 3, 4])
        res = hsplit(a, 2)
        desired = [np.array([1, 2]), np.array([3, 4])]
        compare_results(res, desired)

    def test_2D_array(self):
        a = np.array([[1, 2, 3, 4],
                  [1, 2, 3, 4]])
        res = hsplit(a, 2)
        desired = [np.array([[1, 2], [1, 2]]), np.array([[3, 4], [3, 4]])]
        compare_results(res, desired)


class TestVsplit:
    """Only testing for integer splits.

    """
    def test_non_iterable(self):
        assert_raises(ValueError, vsplit, 1, 1)

    def test_0D_array(self):
        a = np.array(1)
        assert_raises(ValueError, vsplit, a, 2)

    def test_1D_array(self):
        a = np.array([1, 2, 3, 4])
        try:
            vsplit(a, 2)
            assert_(0)
        except ValueError:
            pass

    def test_2D_array(self):
        a = np.array([[1, 2, 3, 4],
                  [1, 2, 3, 4]])
        res = vsplit(a, 2)
        desired = [np.array([[1, 2, 3, 4]]), np.array([[1, 2, 3, 4]])]
        compare_results(res, desired)


class TestDsplit:
    # Only testing for integer splits.
    def test_non_iterable(self):
        assert_raises(ValueError, dsplit, 1, 1)

    def test_0D_array(self):
        a = np.array(1)
        assert_raises(ValueError, dsplit, a, 2)

    def test_1D_array(self):
        a = np.array([1, 2, 3, 4])
        assert_raises(ValueError, dsplit, a, 2)

    def test_2D_array(self):
        a = np.array([[1, 2, 3, 4],
                  [1, 2, 3, 4]])
        try:
            dsplit(a, 2)
            assert_(0)
        except ValueError:
            pass

    def test_3D_array(self):
        a = np.array([[[1, 2, 3, 4],
                   [1, 2, 3, 4]],
                  [[1, 2, 3, 4],
                   [1, 2, 3, 4]]])
        res = dsplit(a, 2)
        desired = [np.array([[[1, 2], [1, 2]], [[1, 2], [1, 2]]]),
                   np.array([[[3, 4], [3, 4]], [[3, 4], [3, 4]]])]
        compare_results(res, desired)


class TestSqueeze:
    def test_basic(self):
        from numpy.random import rand

        a = rand(20, 10, 10, 1, 1)
        b = rand(20, 1, 10, 1, 20)
        c = rand(1, 1, 20, 10)
        assert_array_equal(np.squeeze(a), np.reshape(a, (20, 10, 10)))
        assert_array_equal(np.squeeze(b), np.reshape(b, (20, 10, 20)))
        assert_array_equal(np.squeeze(c), np.reshape(c, (20, 10)))

        # Squeezing to 0-dim should still give an ndarray
        a = [[[1.5]]]
        res = np.squeeze(a)
        assert_equal(res, 1.5)
        assert_equal(res.ndim, 0)
        assert_equal(type(res), np.ndarray)


class TestKron:
    def test_basic(self):
        # Using 0-dimensional ndarray
        a = np.array(1)
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[1, 2], [3, 4]])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array(1)
        assert_array_equal(np.kron(a, b), k)

        # Using 1-dimensional ndarray
        a = np.array([3])
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[3, 6], [9, 12]])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array([3])
        assert_array_equal(np.kron(a, b), k)

        # Using 3-dimensional ndarray
        a = np.array([[[1]], [[2]]])
        b = np.array([[1, 2], [3, 4]])
        k = np.array([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        assert_array_equal(np.kron(a, b), k)
        a = np.array([[1, 2], [3, 4]])
        b = np.array([[[1]], [[2]]])
        k = np.array([[[1, 2], [3, 4]], [[2, 4], [6, 8]]])
        assert_array_equal(np.kron(a, b), k)

    def test_return_type(self):
        class myarray(np.ndarray):
            __array_priority__ = 1.0

        a = np.ones([2, 2])
        ma = myarray(a.shape, a.dtype, a.data)
        assert_equal(type(kron(a, a)), np.ndarray)
        assert_equal(type(kron(ma, ma)), myarray)
        assert_equal(type(kron(a, ma)), myarray)
        assert_equal(type(kron(ma, a)), myarray)

    @pytest.mark.parametrize(
        "array_class", [np.asarray, np.mat]
    )
    def test_kron_smoke(self, array_class):
        a = array_class(np.ones([3, 3]))
        b = array_class(np.ones([3, 3]))
        k = array_class(np.ones([9, 9]))

        assert_array_equal(np.kron(a, b), k)

    def test_kron_ma(self):
        x = np.ma.array([[1, 2], [3, 4]], mask=[[0, 1], [1, 0]])
        k = np.ma.array(np.diag([1, 4, 4, 16]),
                mask=~np.array(np.identity(4), dtype=bool))

        assert_array_equal(k, np.kron(x, x))

    @pytest.mark.parametrize(
        "shape_a,shape_b", [
            ((1, 1), (1, 1)),
            ((1, 2, 3), (4, 5, 6)),
            ((2, 2), (2, 2, 2)),
            ((1, 0), (1, 1)),
            ((2, 0, 2), (2, 2)),
            ((2, 0, 0, 2), (2, 0, 2)),
        ])
    def test_kron_shape(self, shape_a, shape_b):
        a = np.ones(shape_a)
        b = np.ones(shape_b)
        normalised_shape_a = (1,) * max(0, len(shape_b)-len(shape_a)) + shape_a
        normalised_shape_b = (1,) * max(0, len(shape_a)-len(shape_b)) + shape_b
        expected_shape = np.multiply(normalised_shape_a, normalised_shape_b)

        k = np.kron(a, b)
        assert np.array_equal(
                k.shape, expected_shape), "Unexpected shape from kron"


class TestTile:
    def test_basic(self):
        a = np.array([0, 1, 2])
        b = [[1, 2], [3, 4]]
        assert_equal(tile(a, 2), [0, 1, 2, 0, 1, 2])
        assert_equal(tile(a, (2, 2)), [[0, 1, 2, 0, 1, 2], [0, 1, 2, 0, 1, 2]])
        assert_equal(tile(a, (1, 2)), [[0, 1, 2, 0, 1, 2]])
        assert_equal(tile(b, 2), [[1, 2, 1, 2], [3, 4, 3, 4]])
        assert_equal(tile(b, (2, 1)), [[1, 2], [3, 4], [1, 2], [3, 4]])
        assert_equal(tile(b, (2, 2)), [[1, 2, 1, 2], [3, 4, 3, 4],
                                       [1, 2, 1, 2], [3, 4, 3, 4]])

    def test_tile_one_repetition_on_array_gh4679(self):
        a = np.arange(5)
        b = tile(a, 1)
        b += 2
        assert_equal(a, np.arange(5))

    def test_empty(self):
        a = np.array([[[]]])
        b = np.array([[], []])
        c = tile(b, 2).shape
        d = tile(a, (3, 2, 5)).shape
        assert_equal(c, (2, 0))
        assert_equal(d, (3, 2, 0))

    def test_kroncompare(self):
        from numpy.random import randint

        reps = [(2,), (1, 2), (2, 1), (2, 2), (2, 3, 2), (3, 2)]
        shape = [(3,), (2, 3), (3, 4, 3), (3, 2, 3), (4, 3, 2, 4), (2, 2)]
        for s in shape:
            b = randint(0, 10, size=s)
            for r in reps:
                a = np.ones(r, b.dtype)
                large = tile(b, r)
                klarge = kron(a, b)
                assert_equal(large, klarge)


class TestMayShareMemory:
    def test_basic(self):
        d = np.ones((50, 60))
        d2 = np.ones((30, 60, 6))
        assert_(np.may_share_memory(d, d))
        assert_(np.may_share_memory(d, d[::-1]))
        assert_(np.may_share_memory(d, d[::2]))
        assert_(np.may_share_memory(d, d[1:, ::-1]))

        assert_(not np.may_share_memory(d[::-1], d2))
        assert_(not np.may_share_memory(d[::2], d2))
        assert_(not np.may_share_memory(d[1:, ::-1], d2))
        assert_(np.may_share_memory(d2[1:, ::-1], d2))


# Utility
def compare_results(res, desired):
    """Compare lists of arrays."""
    if len(res) != len(desired):
        raise ValueError("Iterables have different lengths")
    # See also PEP 618 for Python 3.10
    for x, y in zip(res, desired):
        assert_array_equal(x, y)
