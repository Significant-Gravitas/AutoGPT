import pytest
import numpy as np
from numpy.core import (
    array, arange, atleast_1d, atleast_2d, atleast_3d, block, vstack, hstack,
    newaxis, concatenate, stack
    )
from numpy.core.shape_base import (_block_dispatcher, _block_setup,
                                   _block_concatenate, _block_slicing)
from numpy.testing import (
    assert_, assert_raises, assert_array_equal, assert_equal,
    assert_raises_regex, assert_warns, IS_PYPY
    )


class TestAtleast1d:
    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [array([1]), array([2])]
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [array([1, 2]), array([2, 3])]
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_3D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_1d(a), atleast_1d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_r1array(self):
        """ Test to make sure equivalent Travis O's r1array function
        """
        assert_(atleast_1d(3).shape == (1,))
        assert_(atleast_1d(3j).shape == (1,))
        assert_(atleast_1d(3.0).shape == (1,))
        assert_(atleast_1d([[2, 3], [4, 5]]).shape == (2, 2))


class TestAtleast2d:
    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [array([[1]]), array([[2]])]
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [array([[1, 2]]), array([[2, 3]])]
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_3D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_2d(a), atleast_2d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)

    def test_r2array(self):
        """ Test to make sure equivalent Travis O's r2array function
        """
        assert_(atleast_2d(3).shape == (1, 1))
        assert_(atleast_2d([3j, 1]).shape == (1, 2))
        assert_(atleast_2d([[[3, 1], [4, 5]], [[3, 5], [1, 2]]]).shape == (2, 2, 2))


class TestAtleast3d:
    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [array([[[1]]]), array([[[2]]])]
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = array([1, 2])
        b = array([2, 3])
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [array([[[1], [2]]]), array([[[2], [3]]])]
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [a[:,:, newaxis], b[:,:, newaxis]]
        assert_array_equal(res, desired)

    def test_3D_array(self):
        a = array([[1, 2], [1, 2]])
        b = array([[2, 3], [2, 3]])
        a = array([a, a])
        b = array([b, b])
        res = [atleast_3d(a), atleast_3d(b)]
        desired = [a, b]
        assert_array_equal(res, desired)


class TestHstack:
    def test_non_iterable(self):
        assert_raises(TypeError, hstack, 1)

    def test_empty_input(self):
        assert_raises(ValueError, hstack, ())

    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = hstack([a, b])
        desired = array([1, 2])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = array([1])
        b = array([2])
        res = hstack([a, b])
        desired = array([1, 2])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = array([[1], [2]])
        b = array([[1], [2]])
        res = hstack([a, b])
        desired = array([[1, 1], [2, 2]])
        assert_array_equal(res, desired)

    def test_generator(self):
        with assert_warns(FutureWarning):
            hstack((np.arange(3) for _ in range(2)))
        with assert_warns(FutureWarning):
            hstack(map(lambda x: x, np.ones((3, 2))))

    def test_casting_and_dtype(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        res = np.hstack((a, b), casting="unsafe", dtype=np.int64)
        expected_res = np.array([1, 2, 3, 2, 3, 4])
        assert_array_equal(res, expected_res)
    
    def test_casting_and_dtype_type_error(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        with pytest.raises(TypeError):
            hstack((a, b), casting="safe", dtype=np.int64)


class TestVstack:
    def test_non_iterable(self):
        assert_raises(TypeError, vstack, 1)

    def test_empty_input(self):
        assert_raises(ValueError, vstack, ())

    def test_0D_array(self):
        a = array(1)
        b = array(2)
        res = vstack([a, b])
        desired = array([[1], [2]])
        assert_array_equal(res, desired)

    def test_1D_array(self):
        a = array([1])
        b = array([2])
        res = vstack([a, b])
        desired = array([[1], [2]])
        assert_array_equal(res, desired)

    def test_2D_array(self):
        a = array([[1], [2]])
        b = array([[1], [2]])
        res = vstack([a, b])
        desired = array([[1], [2], [1], [2]])
        assert_array_equal(res, desired)

    def test_2D_array2(self):
        a = array([1, 2])
        b = array([1, 2])
        res = vstack([a, b])
        desired = array([[1, 2], [1, 2]])
        assert_array_equal(res, desired)

    def test_generator(self):
        with assert_warns(FutureWarning):
            vstack((np.arange(3) for _ in range(2)))

    def test_casting_and_dtype(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        res = np.vstack((a, b), casting="unsafe", dtype=np.int64)
        expected_res = np.array([[1, 2, 3], [2, 3, 4]])
        assert_array_equal(res, expected_res)
    
    def test_casting_and_dtype_type_error(self):
        a = np.array([1, 2, 3])
        b = np.array([2.5, 3.5, 4.5])
        with pytest.raises(TypeError):
            vstack((a, b), casting="safe", dtype=np.int64)
        


class TestConcatenate:
    def test_returns_copy(self):
        a = np.eye(3)
        b = np.concatenate([a])
        b[0, 0] = 2
        assert b[0, 0] != a[0, 0]

    def test_exceptions(self):
        # test axis must be in bounds
        for ndim in [1, 2, 3]:
            a = np.ones((1,)*ndim)
            np.concatenate((a, a), axis=0)  # OK
            assert_raises(np.AxisError, np.concatenate, (a, a), axis=ndim)
            assert_raises(np.AxisError, np.concatenate, (a, a), axis=-(ndim + 1))

        # Scalars cannot be concatenated
        assert_raises(ValueError, concatenate, (0,))
        assert_raises(ValueError, concatenate, (np.array(0),))

        # dimensionality must match
        assert_raises_regex(
            ValueError,
            r"all the input arrays must have same number of dimensions, but "
            r"the array at index 0 has 1 dimension\(s\) and the array at "
            r"index 1 has 2 dimension\(s\)",
            np.concatenate, (np.zeros(1), np.zeros((1, 1))))

        # test shapes must match except for concatenation axis
        a = np.ones((1, 2, 3))
        b = np.ones((2, 2, 3))
        axis = list(range(3))
        for i in range(3):
            np.concatenate((a, b), axis=axis[0])  # OK
            assert_raises_regex(
                ValueError,
                "all the input array dimensions except for the concatenation axis "
                "must match exactly, but along dimension {}, the array at "
                "index 0 has size 1 and the array at index 1 has size 2"
                .format(i),
                np.concatenate, (a, b), axis=axis[1])
            assert_raises(ValueError, np.concatenate, (a, b), axis=axis[2])
            a = np.moveaxis(a, -1, 0)
            b = np.moveaxis(b, -1, 0)
            axis.append(axis.pop(0))

        # No arrays to concatenate raises ValueError
        assert_raises(ValueError, concatenate, ())

    def test_concatenate_axis_None(self):
        a = np.arange(4, dtype=np.float64).reshape((2, 2))
        b = list(range(3))
        c = ['x']
        r = np.concatenate((a, a), axis=None)
        assert_equal(r.dtype, a.dtype)
        assert_equal(r.ndim, 1)
        r = np.concatenate((a, b), axis=None)
        assert_equal(r.size, a.size + len(b))
        assert_equal(r.dtype, a.dtype)
        r = np.concatenate((a, b, c), axis=None, dtype="U")
        d = array(['0.0', '1.0', '2.0', '3.0',
                   '0', '1', '2', 'x'])
        assert_array_equal(r, d)

        out = np.zeros(a.size + len(b))
        r = np.concatenate((a, b), axis=None)
        rout = np.concatenate((a, b), axis=None, out=out)
        assert_(out is rout)
        assert_equal(r, rout)

    def test_large_concatenate_axis_None(self):
        # When no axis is given, concatenate uses flattened versions.
        # This also had a bug with many arrays (see gh-5979).
        x = np.arange(1, 100)
        r = np.concatenate(x, None)
        assert_array_equal(x, r)

        # This should probably be deprecated:
        r = np.concatenate(x, 100)  # axis is >= MAXDIMS
        assert_array_equal(x, r)

    def test_concatenate(self):
        # Test concatenate function
        # One sequence returns unmodified (but as array)
        r4 = list(range(4))
        assert_array_equal(concatenate((r4,)), r4)
        # Any sequence
        assert_array_equal(concatenate((tuple(r4),)), r4)
        assert_array_equal(concatenate((array(r4),)), r4)
        # 1D default concatenation
        r3 = list(range(3))
        assert_array_equal(concatenate((r4, r3)), r4 + r3)
        # Mixed sequence types
        assert_array_equal(concatenate((tuple(r4), r3)), r4 + r3)
        assert_array_equal(concatenate((array(r4), r3)), r4 + r3)
        # Explicit axis specification
        assert_array_equal(concatenate((r4, r3), 0), r4 + r3)
        # Including negative
        assert_array_equal(concatenate((r4, r3), -1), r4 + r3)
        # 2D
        a23 = array([[10, 11, 12], [13, 14, 15]])
        a13 = array([[0, 1, 2]])
        res = array([[10, 11, 12], [13, 14, 15], [0, 1, 2]])
        assert_array_equal(concatenate((a23, a13)), res)
        assert_array_equal(concatenate((a23, a13), 0), res)
        assert_array_equal(concatenate((a23.T, a13.T), 1), res.T)
        assert_array_equal(concatenate((a23.T, a13.T), -1), res.T)
        # Arrays much match shape
        assert_raises(ValueError, concatenate, (a23.T, a13.T), 0)
        # 3D
        res = arange(2 * 3 * 7).reshape((2, 3, 7))
        a0 = res[..., :4]
        a1 = res[..., 4:6]
        a2 = res[..., 6:]
        assert_array_equal(concatenate((a0, a1, a2), 2), res)
        assert_array_equal(concatenate((a0, a1, a2), -1), res)
        assert_array_equal(concatenate((a0.T, a1.T, a2.T), 0), res.T)

        out = res.copy()
        rout = concatenate((a0, a1, a2), 2, out=out)
        assert_(out is rout)
        assert_equal(res, rout)

    @pytest.mark.skipif(IS_PYPY, reason="PYPY handles sq_concat, nb_add differently than cpython")
    def test_operator_concat(self):
        import operator
        a = array([1, 2])
        b = array([3, 4])
        n = [1,2]
        res = array([1, 2, 3, 4])
        assert_raises(TypeError, operator.concat, a, b)
        assert_raises(TypeError, operator.concat, a, n)
        assert_raises(TypeError, operator.concat, n, a)
        assert_raises(TypeError, operator.concat, a, 1)
        assert_raises(TypeError, operator.concat, 1, a)

    def test_bad_out_shape(self):
        a = array([1, 2])
        b = array([3, 4])

        assert_raises(ValueError, concatenate, (a, b), out=np.empty(5))
        assert_raises(ValueError, concatenate, (a, b), out=np.empty((4,1)))
        assert_raises(ValueError, concatenate, (a, b), out=np.empty((1,4)))
        concatenate((a, b), out=np.empty(4))

    @pytest.mark.parametrize("axis", [None, 0])
    @pytest.mark.parametrize("out_dtype", ["c8", "f4", "f8", ">f8", "i8", "S4"])
    @pytest.mark.parametrize("casting",
            ['no', 'equiv', 'safe', 'same_kind', 'unsafe'])
    def test_out_and_dtype(self, axis, out_dtype, casting):
        # Compare usage of `out=out` with `dtype=out.dtype`
        out = np.empty(4, dtype=out_dtype)
        to_concat = (array([1.1, 2.2]), array([3.3, 4.4]))

        if not np.can_cast(to_concat[0], out_dtype, casting=casting):
            with assert_raises(TypeError):
                concatenate(to_concat, out=out, axis=axis, casting=casting)
            with assert_raises(TypeError):
                concatenate(to_concat, dtype=out.dtype,
                            axis=axis, casting=casting)
        else:
            res_out = concatenate(to_concat, out=out,
                                  axis=axis, casting=casting)
            res_dtype = concatenate(to_concat, dtype=out.dtype,
                                    axis=axis, casting=casting)
            assert res_out is out
            assert_array_equal(out, res_dtype)
            assert res_dtype.dtype == out_dtype

        with assert_raises(TypeError):
            concatenate(to_concat, out=out, dtype=out_dtype, axis=axis)

    @pytest.mark.parametrize("axis", [None, 0])
    @pytest.mark.parametrize("string_dt", ["S", "U", "S0", "U0"])
    @pytest.mark.parametrize("arrs",
            [([0.],), ([0.], [1]), ([0], ["string"], [1.])])
    def test_dtype_with_promotion(self, arrs, string_dt, axis):
        # Note that U0 and S0 should be deprecated eventually and changed to
        # actually give the empty string result (together with `np.array`)
        res = np.concatenate(arrs, axis=axis, dtype=string_dt, casting="unsafe")
        # The actual dtype should be identical to a cast (of a double array):
        assert res.dtype == np.array(1.).astype(string_dt).dtype

    @pytest.mark.parametrize("axis", [None, 0])
    def test_string_dtype_does_not_inspect(self, axis):
        with pytest.raises(TypeError):
            np.concatenate(([None], [1]), dtype="S", axis=axis)
        with pytest.raises(TypeError):
            np.concatenate(([None], [1]), dtype="U", axis=axis)

    @pytest.mark.parametrize("axis", [None, 0])
    def test_subarray_error(self, axis):
        with pytest.raises(TypeError, match=".*subarray dtype"):
            np.concatenate(([1], [1]), dtype="(2,)i", axis=axis)


def test_stack():
    # non-iterable input
    assert_raises(TypeError, stack, 1)

    # 0d input
    for input_ in [(1, 2, 3),
                   [np.int32(1), np.int32(2), np.int32(3)],
                   [np.array(1), np.array(2), np.array(3)]]:
        assert_array_equal(stack(input_), [1, 2, 3])
    # 1d input examples
    a = np.array([1, 2, 3])
    b = np.array([4, 5, 6])
    r1 = array([[1, 2, 3], [4, 5, 6]])
    assert_array_equal(np.stack((a, b)), r1)
    assert_array_equal(np.stack((a, b), axis=1), r1.T)
    # all input types
    assert_array_equal(np.stack(list([a, b])), r1)
    assert_array_equal(np.stack(array([a, b])), r1)
    # all shapes for 1d input
    arrays = [np.random.randn(3) for _ in range(10)]
    axes = [0, 1, -1, -2]
    expected_shapes = [(10, 3), (3, 10), (3, 10), (10, 3)]
    for axis, expected_shape in zip(axes, expected_shapes):
        assert_equal(np.stack(arrays, axis).shape, expected_shape)
    assert_raises_regex(np.AxisError, 'out of bounds', stack, arrays, axis=2)
    assert_raises_regex(np.AxisError, 'out of bounds', stack, arrays, axis=-3)
    # all shapes for 2d input
    arrays = [np.random.randn(3, 4) for _ in range(10)]
    axes = [0, 1, 2, -1, -2, -3]
    expected_shapes = [(10, 3, 4), (3, 10, 4), (3, 4, 10),
                       (3, 4, 10), (3, 10, 4), (10, 3, 4)]
    for axis, expected_shape in zip(axes, expected_shapes):
        assert_equal(np.stack(arrays, axis).shape, expected_shape)
    # empty arrays
    assert_(stack([[], [], []]).shape == (3, 0))
    assert_(stack([[], [], []], axis=1).shape == (0, 3))
    # out
    out = np.zeros_like(r1)
    np.stack((a, b), out=out)
    assert_array_equal(out, r1)
    # edge cases
    assert_raises_regex(ValueError, 'need at least one array', stack, [])
    assert_raises_regex(ValueError, 'must have the same shape',
                        stack, [1, np.arange(3)])
    assert_raises_regex(ValueError, 'must have the same shape',
                        stack, [np.arange(3), 1])
    assert_raises_regex(ValueError, 'must have the same shape',
                        stack, [np.arange(3), 1], axis=1)
    assert_raises_regex(ValueError, 'must have the same shape',
                        stack, [np.zeros((3, 3)), np.zeros(3)], axis=1)
    assert_raises_regex(ValueError, 'must have the same shape',
                        stack, [np.arange(2), np.arange(3)])
    # generator is deprecated
    with assert_warns(FutureWarning):
        result = stack((x for x in range(3)))
    assert_array_equal(result, np.array([0, 1, 2]))
    #casting and dtype test
    a = np.array([1, 2, 3])
    b = np.array([2.5, 3.5, 4.5])
    res = np.stack((a, b), axis=1, casting="unsafe", dtype=np.int64)
    expected_res = np.array([[1, 2], [2, 3], [3, 4]])
    assert_array_equal(res, expected_res)
    #casting and dtype with TypeError
    with assert_raises(TypeError):
        stack((a, b), dtype=np.int64, axis=1, casting="safe")


@pytest.mark.parametrize("axis", [0])
@pytest.mark.parametrize("out_dtype", ["c8", "f4", "f8", ">f8", "i8"])
@pytest.mark.parametrize("casting",
                         ['no', 'equiv', 'safe', 'same_kind', 'unsafe'])
def test_stack_out_and_dtype(axis, out_dtype, casting):
    to_concat = (array([1, 2]), array([3, 4]))
    res = array([[1, 2], [3, 4]])
    out = np.zeros_like(res)

    if not np.can_cast(to_concat[0], out_dtype, casting=casting):
        with assert_raises(TypeError):
            stack(to_concat, dtype=out_dtype,
                  axis=axis, casting=casting)
    else:
        res_out = stack(to_concat, out=out,
                        axis=axis, casting=casting)
        res_dtype = stack(to_concat, dtype=out_dtype,
                          axis=axis, casting=casting)
        assert res_out is out
        assert_array_equal(out, res_dtype)
        assert res_dtype.dtype == out_dtype

    with assert_raises(TypeError):
        stack(to_concat, out=out, dtype=out_dtype, axis=axis)


class TestBlock:
    @pytest.fixture(params=['block', 'force_concatenate', 'force_slicing'])
    def block(self, request):
        # blocking small arrays and large arrays go through different paths.
        # the algorithm is triggered depending on the number of element
        # copies required.
        # We define a test fixture that forces most tests to go through
        # both code paths.
        # Ultimately, this should be removed if a single algorithm is found
        # to be faster for both small and large arrays.
        def _block_force_concatenate(arrays):
            arrays, list_ndim, result_ndim, _ = _block_setup(arrays)
            return _block_concatenate(arrays, list_ndim, result_ndim)

        def _block_force_slicing(arrays):
            arrays, list_ndim, result_ndim, _ = _block_setup(arrays)
            return _block_slicing(arrays, list_ndim, result_ndim)

        if request.param == 'force_concatenate':
            return _block_force_concatenate
        elif request.param == 'force_slicing':
            return _block_force_slicing
        elif request.param == 'block':
            return block
        else:
            raise ValueError('Unknown blocking request. There is a typo in the tests.')

    def test_returns_copy(self, block):
        a = np.eye(3)
        b = block(a)
        b[0, 0] = 2
        assert b[0, 0] != a[0, 0]

    def test_block_total_size_estimate(self, block):
        _, _, _, total_size = _block_setup([1])
        assert total_size == 1

        _, _, _, total_size = _block_setup([[1]])
        assert total_size == 1

        _, _, _, total_size = _block_setup([[1, 1]])
        assert total_size == 2

        _, _, _, total_size = _block_setup([[1], [1]])
        assert total_size == 2

        _, _, _, total_size = _block_setup([[1, 2], [3, 4]])
        assert total_size == 4

    def test_block_simple_row_wise(self, block):
        a_2d = np.ones((2, 2))
        b_2d = 2 * a_2d
        desired = np.array([[1, 1, 2, 2],
                            [1, 1, 2, 2]])
        result = block([a_2d, b_2d])
        assert_equal(desired, result)

    def test_block_simple_column_wise(self, block):
        a_2d = np.ones((2, 2))
        b_2d = 2 * a_2d
        expected = np.array([[1, 1],
                             [1, 1],
                             [2, 2],
                             [2, 2]])
        result = block([[a_2d], [b_2d]])
        assert_equal(expected, result)

    def test_block_with_1d_arrays_row_wise(self, block):
        # # # 1-D vectors are treated as row arrays
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        expected = np.array([1, 2, 3, 2, 3, 4])
        result = block([a, b])
        assert_equal(expected, result)

    def test_block_with_1d_arrays_multiple_rows(self, block):
        a = np.array([1, 2, 3])
        b = np.array([2, 3, 4])
        expected = np.array([[1, 2, 3, 2, 3, 4],
                             [1, 2, 3, 2, 3, 4]])
        result = block([[a, b], [a, b]])
        assert_equal(expected, result)

    def test_block_with_1d_arrays_column_wise(self, block):
        # # # 1-D vectors are treated as row arrays
        a_1d = np.array([1, 2, 3])
        b_1d = np.array([2, 3, 4])
        expected = np.array([[1, 2, 3],
                             [2, 3, 4]])
        result = block([[a_1d], [b_1d]])
        assert_equal(expected, result)

    def test_block_mixed_1d_and_2d(self, block):
        a_2d = np.ones((2, 2))
        b_1d = np.array([2, 2])
        result = block([[a_2d], [b_1d]])
        expected = np.array([[1, 1],
                             [1, 1],
                             [2, 2]])
        assert_equal(expected, result)

    def test_block_complicated(self, block):
        # a bit more complicated
        one_2d = np.array([[1, 1, 1]])
        two_2d = np.array([[2, 2, 2]])
        three_2d = np.array([[3, 3, 3, 3, 3, 3]])
        four_1d = np.array([4, 4, 4, 4, 4, 4])
        five_0d = np.array(5)
        six_1d = np.array([6, 6, 6, 6, 6])
        zero_2d = np.zeros((2, 6))

        expected = np.array([[1, 1, 1, 2, 2, 2],
                             [3, 3, 3, 3, 3, 3],
                             [4, 4, 4, 4, 4, 4],
                             [5, 6, 6, 6, 6, 6],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])

        result = block([[one_2d, two_2d],
                        [three_2d],
                        [four_1d],
                        [five_0d, six_1d],
                        [zero_2d]])
        assert_equal(result, expected)

    def test_nested(self, block):
        one = np.array([1, 1, 1])
        two = np.array([[2, 2, 2], [2, 2, 2], [2, 2, 2]])
        three = np.array([3, 3, 3])
        four = np.array([4, 4, 4])
        five = np.array(5)
        six = np.array([6, 6, 6, 6, 6])
        zero = np.zeros((2, 6))

        result = block([
            [
                block([
                   [one],
                   [three],
                   [four]
                ]),
                two
            ],
            [five, six],
            [zero]
        ])
        expected = np.array([[1, 1, 1, 2, 2, 2],
                             [3, 3, 3, 2, 2, 2],
                             [4, 4, 4, 2, 2, 2],
                             [5, 6, 6, 6, 6, 6],
                             [0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0]])

        assert_equal(result, expected)

    def test_3d(self, block):
        a000 = np.ones((2, 2, 2), int) * 1

        a100 = np.ones((3, 2, 2), int) * 2
        a010 = np.ones((2, 3, 2), int) * 3
        a001 = np.ones((2, 2, 3), int) * 4

        a011 = np.ones((2, 3, 3), int) * 5
        a101 = np.ones((3, 2, 3), int) * 6
        a110 = np.ones((3, 3, 2), int) * 7

        a111 = np.ones((3, 3, 3), int) * 8

        result = block([
            [
                [a000, a001],
                [a010, a011],
            ],
            [
                [a100, a101],
                [a110, a111],
            ]
        ])
        expected = array([[[1, 1, 4, 4, 4],
                           [1, 1, 4, 4, 4],
                           [3, 3, 5, 5, 5],
                           [3, 3, 5, 5, 5],
                           [3, 3, 5, 5, 5]],

                          [[1, 1, 4, 4, 4],
                           [1, 1, 4, 4, 4],
                           [3, 3, 5, 5, 5],
                           [3, 3, 5, 5, 5],
                           [3, 3, 5, 5, 5]],

                          [[2, 2, 6, 6, 6],
                           [2, 2, 6, 6, 6],
                           [7, 7, 8, 8, 8],
                           [7, 7, 8, 8, 8],
                           [7, 7, 8, 8, 8]],

                          [[2, 2, 6, 6, 6],
                           [2, 2, 6, 6, 6],
                           [7, 7, 8, 8, 8],
                           [7, 7, 8, 8, 8],
                           [7, 7, 8, 8, 8]],

                          [[2, 2, 6, 6, 6],
                           [2, 2, 6, 6, 6],
                           [7, 7, 8, 8, 8],
                           [7, 7, 8, 8, 8],
                           [7, 7, 8, 8, 8]]])

        assert_array_equal(result, expected)

    def test_block_with_mismatched_shape(self, block):
        a = np.array([0, 0])
        b = np.eye(2)
        assert_raises(ValueError, block, [a, b])
        assert_raises(ValueError, block, [b, a])

        to_block = [[np.ones((2,3)), np.ones((2,2))],
                    [np.ones((2,2)), np.ones((2,2))]]
        assert_raises(ValueError, block, to_block)
    def test_no_lists(self, block):
        assert_equal(block(1),         np.array(1))
        assert_equal(block(np.eye(3)), np.eye(3))

    def test_invalid_nesting(self, block):
        msg = 'depths are mismatched'
        assert_raises_regex(ValueError, msg, block, [1, [2]])
        assert_raises_regex(ValueError, msg, block, [1, []])
        assert_raises_regex(ValueError, msg, block, [[1], 2])
        assert_raises_regex(ValueError, msg, block, [[], 2])
        assert_raises_regex(ValueError, msg, block, [
            [[1], [2]],
            [[3, 4]],
            [5]  # missing brackets
        ])

    def test_empty_lists(self, block):
        assert_raises_regex(ValueError, 'empty', block, [])
        assert_raises_regex(ValueError, 'empty', block, [[]])
        assert_raises_regex(ValueError, 'empty', block, [[1], []])

    def test_tuple(self, block):
        assert_raises_regex(TypeError, 'tuple', block, ([1, 2], [3, 4]))
        assert_raises_regex(TypeError, 'tuple', block, [(1, 2), (3, 4)])

    def test_different_ndims(self, block):
        a = 1.
        b = 2 * np.ones((1, 2))
        c = 3 * np.ones((1, 1, 3))

        result = block([a, b, c])
        expected = np.array([[[1., 2., 2., 3., 3., 3.]]])

        assert_equal(result, expected)

    def test_different_ndims_depths(self, block):
        a = 1.
        b = 2 * np.ones((1, 2))
        c = 3 * np.ones((1, 2, 3))

        result = block([[a, b], [c]])
        expected = np.array([[[1., 2., 2.],
                              [3., 3., 3.],
                              [3., 3., 3.]]])

        assert_equal(result, expected)

    def test_block_memory_order(self, block):
        # 3D
        arr_c = np.zeros((3,)*3, order='C')
        arr_f = np.zeros((3,)*3, order='F')

        b_c = [[[arr_c, arr_c],
                [arr_c, arr_c]],
               [[arr_c, arr_c],
                [arr_c, arr_c]]]

        b_f = [[[arr_f, arr_f],
                [arr_f, arr_f]],
               [[arr_f, arr_f],
                [arr_f, arr_f]]]

        assert block(b_c).flags['C_CONTIGUOUS']
        assert block(b_f).flags['F_CONTIGUOUS']

        arr_c = np.zeros((3, 3), order='C')
        arr_f = np.zeros((3, 3), order='F')
        # 2D
        b_c = [[arr_c, arr_c],
               [arr_c, arr_c]]

        b_f = [[arr_f, arr_f],
               [arr_f, arr_f]]

        assert block(b_c).flags['C_CONTIGUOUS']
        assert block(b_f).flags['F_CONTIGUOUS']


def test_block_dispatcher():
    class ArrayLike:
        pass
    a = ArrayLike()
    b = ArrayLike()
    c = ArrayLike()
    assert_equal(list(_block_dispatcher(a)), [a])
    assert_equal(list(_block_dispatcher([a])), [a])
    assert_equal(list(_block_dispatcher([a, b])), [a, b])
    assert_equal(list(_block_dispatcher([[a], [b, [c]]])), [a, b, c])
    # don't recurse into non-lists
    assert_equal(list(_block_dispatcher((a, b))), [(a, b)])
