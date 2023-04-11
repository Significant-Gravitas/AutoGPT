import numpy as np
from numpy.core._rational_tests import rational
from numpy.testing import (
    assert_equal, assert_array_equal, assert_raises, assert_,
    assert_raises_regex, assert_warns,
    )
from numpy.lib.stride_tricks import (
    as_strided, broadcast_arrays, _broadcast_shape, broadcast_to,
    broadcast_shapes, sliding_window_view,
    )
import pytest


def assert_shapes_correct(input_shapes, expected_shape):
    # Broadcast a list of arrays with the given input shapes and check the
    # common output shape.

    inarrays = [np.zeros(s) for s in input_shapes]
    outarrays = broadcast_arrays(*inarrays)
    outshapes = [a.shape for a in outarrays]
    expected = [expected_shape] * len(inarrays)
    assert_equal(outshapes, expected)


def assert_incompatible_shapes_raise(input_shapes):
    # Broadcast a list of arrays with the given (incompatible) input shapes
    # and check that they raise a ValueError.

    inarrays = [np.zeros(s) for s in input_shapes]
    assert_raises(ValueError, broadcast_arrays, *inarrays)


def assert_same_as_ufunc(shape0, shape1, transposed=False, flipped=False):
    # Broadcast two shapes against each other and check that the data layout
    # is the same as if a ufunc did the broadcasting.

    x0 = np.zeros(shape0, dtype=int)
    # Note that multiply.reduce's identity element is 1.0, so when shape1==(),
    # this gives the desired n==1.
    n = int(np.multiply.reduce(shape1))
    x1 = np.arange(n).reshape(shape1)
    if transposed:
        x0 = x0.T
        x1 = x1.T
    if flipped:
        x0 = x0[::-1]
        x1 = x1[::-1]
    # Use the add ufunc to do the broadcasting. Since we're adding 0s to x1, the
    # result should be exactly the same as the broadcasted view of x1.
    y = x0 + x1
    b0, b1 = broadcast_arrays(x0, x1)
    assert_array_equal(y, b1)


def test_same():
    x = np.arange(10)
    y = np.arange(10)
    bx, by = broadcast_arrays(x, y)
    assert_array_equal(x, bx)
    assert_array_equal(y, by)

def test_broadcast_kwargs():
    # ensure that a TypeError is appropriately raised when
    # np.broadcast_arrays() is called with any keyword
    # argument other than 'subok'
    x = np.arange(10)
    y = np.arange(10)

    with assert_raises_regex(TypeError, 'got an unexpected keyword'):
        broadcast_arrays(x, y, dtype='float64')


def test_one_off():
    x = np.array([[1, 2, 3]])
    y = np.array([[1], [2], [3]])
    bx, by = broadcast_arrays(x, y)
    bx0 = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    by0 = bx0.T
    assert_array_equal(bx0, bx)
    assert_array_equal(by0, by)


def test_same_input_shapes():
    # Check that the final shape is just the input shape.

    data = [
        (),
        (1,),
        (3,),
        (0, 1),
        (0, 3),
        (1, 0),
        (3, 0),
        (1, 3),
        (3, 1),
        (3, 3),
    ]
    for shape in data:
        input_shapes = [shape]
        # Single input.
        assert_shapes_correct(input_shapes, shape)
        # Double input.
        input_shapes2 = [shape, shape]
        assert_shapes_correct(input_shapes2, shape)
        # Triple input.
        input_shapes3 = [shape, shape, shape]
        assert_shapes_correct(input_shapes3, shape)


def test_two_compatible_by_ones_input_shapes():
    # Check that two different input shapes of the same length, but some have
    # ones, broadcast to the correct shape.

    data = [
        [[(1,), (3,)], (3,)],
        [[(1, 3), (3, 3)], (3, 3)],
        [[(3, 1), (3, 3)], (3, 3)],
        [[(1, 3), (3, 1)], (3, 3)],
        [[(1, 1), (3, 3)], (3, 3)],
        [[(1, 1), (1, 3)], (1, 3)],
        [[(1, 1), (3, 1)], (3, 1)],
        [[(1, 0), (0, 0)], (0, 0)],
        [[(0, 1), (0, 0)], (0, 0)],
        [[(1, 0), (0, 1)], (0, 0)],
        [[(1, 1), (0, 0)], (0, 0)],
        [[(1, 1), (1, 0)], (1, 0)],
        [[(1, 1), (0, 1)], (0, 1)],
    ]
    for input_shapes, expected_shape in data:
        assert_shapes_correct(input_shapes, expected_shape)
        # Reverse the input shapes since broadcasting should be symmetric.
        assert_shapes_correct(input_shapes[::-1], expected_shape)


def test_two_compatible_by_prepending_ones_input_shapes():
    # Check that two different input shapes (of different lengths) broadcast
    # to the correct shape.

    data = [
        [[(), (3,)], (3,)],
        [[(3,), (3, 3)], (3, 3)],
        [[(3,), (3, 1)], (3, 3)],
        [[(1,), (3, 3)], (3, 3)],
        [[(), (3, 3)], (3, 3)],
        [[(1, 1), (3,)], (1, 3)],
        [[(1,), (3, 1)], (3, 1)],
        [[(1,), (1, 3)], (1, 3)],
        [[(), (1, 3)], (1, 3)],
        [[(), (3, 1)], (3, 1)],
        [[(), (0,)], (0,)],
        [[(0,), (0, 0)], (0, 0)],
        [[(0,), (0, 1)], (0, 0)],
        [[(1,), (0, 0)], (0, 0)],
        [[(), (0, 0)], (0, 0)],
        [[(1, 1), (0,)], (1, 0)],
        [[(1,), (0, 1)], (0, 1)],
        [[(1,), (1, 0)], (1, 0)],
        [[(), (1, 0)], (1, 0)],
        [[(), (0, 1)], (0, 1)],
    ]
    for input_shapes, expected_shape in data:
        assert_shapes_correct(input_shapes, expected_shape)
        # Reverse the input shapes since broadcasting should be symmetric.
        assert_shapes_correct(input_shapes[::-1], expected_shape)


def test_incompatible_shapes_raise_valueerror():
    # Check that a ValueError is raised for incompatible shapes.

    data = [
        [(3,), (4,)],
        [(2, 3), (2,)],
        [(3,), (3,), (4,)],
        [(1, 3, 4), (2, 3, 3)],
    ]
    for input_shapes in data:
        assert_incompatible_shapes_raise(input_shapes)
        # Reverse the input shapes since broadcasting should be symmetric.
        assert_incompatible_shapes_raise(input_shapes[::-1])


def test_same_as_ufunc():
    # Check that the data layout is the same as if a ufunc did the operation.

    data = [
        [[(1,), (3,)], (3,)],
        [[(1, 3), (3, 3)], (3, 3)],
        [[(3, 1), (3, 3)], (3, 3)],
        [[(1, 3), (3, 1)], (3, 3)],
        [[(1, 1), (3, 3)], (3, 3)],
        [[(1, 1), (1, 3)], (1, 3)],
        [[(1, 1), (3, 1)], (3, 1)],
        [[(1, 0), (0, 0)], (0, 0)],
        [[(0, 1), (0, 0)], (0, 0)],
        [[(1, 0), (0, 1)], (0, 0)],
        [[(1, 1), (0, 0)], (0, 0)],
        [[(1, 1), (1, 0)], (1, 0)],
        [[(1, 1), (0, 1)], (0, 1)],
        [[(), (3,)], (3,)],
        [[(3,), (3, 3)], (3, 3)],
        [[(3,), (3, 1)], (3, 3)],
        [[(1,), (3, 3)], (3, 3)],
        [[(), (3, 3)], (3, 3)],
        [[(1, 1), (3,)], (1, 3)],
        [[(1,), (3, 1)], (3, 1)],
        [[(1,), (1, 3)], (1, 3)],
        [[(), (1, 3)], (1, 3)],
        [[(), (3, 1)], (3, 1)],
        [[(), (0,)], (0,)],
        [[(0,), (0, 0)], (0, 0)],
        [[(0,), (0, 1)], (0, 0)],
        [[(1,), (0, 0)], (0, 0)],
        [[(), (0, 0)], (0, 0)],
        [[(1, 1), (0,)], (1, 0)],
        [[(1,), (0, 1)], (0, 1)],
        [[(1,), (1, 0)], (1, 0)],
        [[(), (1, 0)], (1, 0)],
        [[(), (0, 1)], (0, 1)],
    ]
    for input_shapes, expected_shape in data:
        assert_same_as_ufunc(input_shapes[0], input_shapes[1],
                             "Shapes: %s %s" % (input_shapes[0], input_shapes[1]))
        # Reverse the input shapes since broadcasting should be symmetric.
        assert_same_as_ufunc(input_shapes[1], input_shapes[0])
        # Try them transposed, too.
        assert_same_as_ufunc(input_shapes[0], input_shapes[1], True)
        # ... and flipped for non-rank-0 inputs in order to test negative
        # strides.
        if () not in input_shapes:
            assert_same_as_ufunc(input_shapes[0], input_shapes[1], False, True)
            assert_same_as_ufunc(input_shapes[0], input_shapes[1], True, True)


def test_broadcast_to_succeeds():
    data = [
        [np.array(0), (0,), np.array(0)],
        [np.array(0), (1,), np.zeros(1)],
        [np.array(0), (3,), np.zeros(3)],
        [np.ones(1), (1,), np.ones(1)],
        [np.ones(1), (2,), np.ones(2)],
        [np.ones(1), (1, 2, 3), np.ones((1, 2, 3))],
        [np.arange(3), (3,), np.arange(3)],
        [np.arange(3), (1, 3), np.arange(3).reshape(1, -1)],
        [np.arange(3), (2, 3), np.array([[0, 1, 2], [0, 1, 2]])],
        # test if shape is not a tuple
        [np.ones(0), 0, np.ones(0)],
        [np.ones(1), 1, np.ones(1)],
        [np.ones(1), 2, np.ones(2)],
        # these cases with size 0 are strange, but they reproduce the behavior
        # of broadcasting with ufuncs (see test_same_as_ufunc above)
        [np.ones(1), (0,), np.ones(0)],
        [np.ones((1, 2)), (0, 2), np.ones((0, 2))],
        [np.ones((2, 1)), (2, 0), np.ones((2, 0))],
    ]
    for input_array, shape, expected in data:
        actual = broadcast_to(input_array, shape)
        assert_array_equal(expected, actual)


def test_broadcast_to_raises():
    data = [
        [(0,), ()],
        [(1,), ()],
        [(3,), ()],
        [(3,), (1,)],
        [(3,), (2,)],
        [(3,), (4,)],
        [(1, 2), (2, 1)],
        [(1, 1), (1,)],
        [(1,), -1],
        [(1,), (-1,)],
        [(1, 2), (-1, 2)],
    ]
    for orig_shape, target_shape in data:
        arr = np.zeros(orig_shape)
        assert_raises(ValueError, lambda: broadcast_to(arr, target_shape))


def test_broadcast_shape():
    # tests internal _broadcast_shape
    # _broadcast_shape is already exercised indirectly by broadcast_arrays
    # _broadcast_shape is also exercised by the public broadcast_shapes function
    assert_equal(_broadcast_shape(), ())
    assert_equal(_broadcast_shape([1, 2]), (2,))
    assert_equal(_broadcast_shape(np.ones((1, 1))), (1, 1))
    assert_equal(_broadcast_shape(np.ones((1, 1)), np.ones((3, 4))), (3, 4))
    assert_equal(_broadcast_shape(*([np.ones((1, 2))] * 32)), (1, 2))
    assert_equal(_broadcast_shape(*([np.ones((1, 2))] * 100)), (1, 2))

    # regression tests for gh-5862
    assert_equal(_broadcast_shape(*([np.ones(2)] * 32 + [1])), (2,))
    bad_args = [np.ones(2)] * 32 + [np.ones(3)] * 32
    assert_raises(ValueError, lambda: _broadcast_shape(*bad_args))


def test_broadcast_shapes_succeeds():
    # tests public broadcast_shapes
    data = [
        [[], ()],
        [[()], ()],
        [[(7,)], (7,)],
        [[(1, 2), (2,)], (1, 2)],
        [[(1, 1)], (1, 1)],
        [[(1, 1), (3, 4)], (3, 4)],
        [[(6, 7), (5, 6, 1), (7,), (5, 1, 7)], (5, 6, 7)],
        [[(5, 6, 1)], (5, 6, 1)],
        [[(1, 3), (3, 1)], (3, 3)],
        [[(1, 0), (0, 0)], (0, 0)],
        [[(0, 1), (0, 0)], (0, 0)],
        [[(1, 0), (0, 1)], (0, 0)],
        [[(1, 1), (0, 0)], (0, 0)],
        [[(1, 1), (1, 0)], (1, 0)],
        [[(1, 1), (0, 1)], (0, 1)],
        [[(), (0,)], (0,)],
        [[(0,), (0, 0)], (0, 0)],
        [[(0,), (0, 1)], (0, 0)],
        [[(1,), (0, 0)], (0, 0)],
        [[(), (0, 0)], (0, 0)],
        [[(1, 1), (0,)], (1, 0)],
        [[(1,), (0, 1)], (0, 1)],
        [[(1,), (1, 0)], (1, 0)],
        [[(), (1, 0)], (1, 0)],
        [[(), (0, 1)], (0, 1)],
        [[(1,), (3,)], (3,)],
        [[2, (3, 2)], (3, 2)],
    ]
    for input_shapes, target_shape in data:
        assert_equal(broadcast_shapes(*input_shapes), target_shape)

    assert_equal(broadcast_shapes(*([(1, 2)] * 32)), (1, 2))
    assert_equal(broadcast_shapes(*([(1, 2)] * 100)), (1, 2))

    # regression tests for gh-5862
    assert_equal(broadcast_shapes(*([(2,)] * 32)), (2,))


def test_broadcast_shapes_raises():
    # tests public broadcast_shapes
    data = [
        [(3,), (4,)],
        [(2, 3), (2,)],
        [(3,), (3,), (4,)],
        [(1, 3, 4), (2, 3, 3)],
        [(1, 2), (3,1), (3,2), (10, 5)],
        [2, (2, 3)],
    ]
    for input_shapes in data:
        assert_raises(ValueError, lambda: broadcast_shapes(*input_shapes))

    bad_args = [(2,)] * 32 + [(3,)] * 32
    assert_raises(ValueError, lambda: broadcast_shapes(*bad_args))


def test_as_strided():
    a = np.array([None])
    a_view = as_strided(a)
    expected = np.array([None])
    assert_array_equal(a_view, np.array([None]))

    a = np.array([1, 2, 3, 4])
    a_view = as_strided(a, shape=(2,), strides=(2 * a.itemsize,))
    expected = np.array([1, 3])
    assert_array_equal(a_view, expected)

    a = np.array([1, 2, 3, 4])
    a_view = as_strided(a, shape=(3, 4), strides=(0, 1 * a.itemsize))
    expected = np.array([[1, 2, 3, 4], [1, 2, 3, 4], [1, 2, 3, 4]])
    assert_array_equal(a_view, expected)

    # Regression test for gh-5081
    dt = np.dtype([('num', 'i4'), ('obj', 'O')])
    a = np.empty((4,), dtype=dt)
    a['num'] = np.arange(1, 5)
    a_view = as_strided(a, shape=(3, 4), strides=(0, a.itemsize))
    expected_num = [[1, 2, 3, 4]] * 3
    expected_obj = [[None]*4]*3
    assert_equal(a_view.dtype, dt)
    assert_array_equal(expected_num, a_view['num'])
    assert_array_equal(expected_obj, a_view['obj'])

    # Make sure that void types without fields are kept unchanged
    a = np.empty((4,), dtype='V4')
    a_view = as_strided(a, shape=(3, 4), strides=(0, a.itemsize))
    assert_equal(a.dtype, a_view.dtype)

    # Make sure that the only type that could fail is properly handled
    dt = np.dtype({'names': [''], 'formats': ['V4']})
    a = np.empty((4,), dtype=dt)
    a_view = as_strided(a, shape=(3, 4), strides=(0, a.itemsize))
    assert_equal(a.dtype, a_view.dtype)

    # Custom dtypes should not be lost (gh-9161)
    r = [rational(i) for i in range(4)]
    a = np.array(r, dtype=rational)
    a_view = as_strided(a, shape=(3, 4), strides=(0, a.itemsize))
    assert_equal(a.dtype, a_view.dtype)
    assert_array_equal([r] * 3, a_view)


class TestSlidingWindowView:
    def test_1d(self):
        arr = np.arange(5)
        arr_view = sliding_window_view(arr, 2)
        expected = np.array([[0, 1],
                             [1, 2],
                             [2, 3],
                             [3, 4]])
        assert_array_equal(arr_view, expected)

    def test_2d(self):
        i, j = np.ogrid[:3, :4]
        arr = 10*i + j
        shape = (2, 2)
        arr_view = sliding_window_view(arr, shape)
        expected = np.array([[[[0, 1], [10, 11]],
                              [[1, 2], [11, 12]],
                              [[2, 3], [12, 13]]],
                             [[[10, 11], [20, 21]],
                              [[11, 12], [21, 22]],
                              [[12, 13], [22, 23]]]])
        assert_array_equal(arr_view, expected)

    def test_2d_with_axis(self):
        i, j = np.ogrid[:3, :4]
        arr = 10*i + j
        arr_view = sliding_window_view(arr, 3, 0)
        expected = np.array([[[0, 10, 20],
                              [1, 11, 21],
                              [2, 12, 22],
                              [3, 13, 23]]])
        assert_array_equal(arr_view, expected)

    def test_2d_repeated_axis(self):
        i, j = np.ogrid[:3, :4]
        arr = 10*i + j
        arr_view = sliding_window_view(arr, (2, 3), (1, 1))
        expected = np.array([[[[0, 1, 2],
                               [1, 2, 3]]],
                             [[[10, 11, 12],
                               [11, 12, 13]]],
                             [[[20, 21, 22],
                               [21, 22, 23]]]])
        assert_array_equal(arr_view, expected)

    def test_2d_without_axis(self):
        i, j = np.ogrid[:4, :4]
        arr = 10*i + j
        shape = (2, 3)
        arr_view = sliding_window_view(arr, shape)
        expected = np.array([[[[0, 1, 2], [10, 11, 12]],
                              [[1, 2, 3], [11, 12, 13]]],
                             [[[10, 11, 12], [20, 21, 22]],
                              [[11, 12, 13], [21, 22, 23]]],
                             [[[20, 21, 22], [30, 31, 32]],
                              [[21, 22, 23], [31, 32, 33]]]])
        assert_array_equal(arr_view, expected)

    def test_errors(self):
        i, j = np.ogrid[:4, :4]
        arr = 10*i + j
        with pytest.raises(ValueError, match='cannot contain negative values'):
            sliding_window_view(arr, (-1, 3))
        with pytest.raises(
                ValueError,
                match='must provide window_shape for all dimensions of `x`'):
            sliding_window_view(arr, (1,))
        with pytest.raises(
                ValueError,
                match='Must provide matching length window_shape and axis'):
            sliding_window_view(arr, (1, 3, 4), axis=(0, 1))
        with pytest.raises(
                ValueError,
                match='window shape cannot be larger than input array'):
            sliding_window_view(arr, (5, 5))

    def test_writeable(self):
        arr = np.arange(5)
        view = sliding_window_view(arr, 2, writeable=False)
        assert_(not view.flags.writeable)
        with pytest.raises(
                ValueError,
                match='assignment destination is read-only'):
            view[0, 0] = 3
        view = sliding_window_view(arr, 2, writeable=True)
        assert_(view.flags.writeable)
        view[0, 1] = 3
        assert_array_equal(arr, np.array([0, 3, 2, 3, 4]))

    def test_subok(self):
        class MyArray(np.ndarray):
            pass

        arr = np.arange(5).view(MyArray)
        assert_(not isinstance(sliding_window_view(arr, 2,
                                                   subok=False),
                               MyArray))
        assert_(isinstance(sliding_window_view(arr, 2, subok=True), MyArray))
        # Default behavior
        assert_(not isinstance(sliding_window_view(arr, 2), MyArray))


def as_strided_writeable():
    arr = np.ones(10)
    view = as_strided(arr, writeable=False)
    assert_(not view.flags.writeable)

    # Check that writeable also is fine:
    view = as_strided(arr, writeable=True)
    assert_(view.flags.writeable)
    view[...] = 3
    assert_array_equal(arr, np.full_like(arr, 3))

    # Test that things do not break down for readonly:
    arr.flags.writeable = False
    view = as_strided(arr, writeable=False)
    view = as_strided(arr, writeable=True)
    assert_(not view.flags.writeable)


class VerySimpleSubClass(np.ndarray):
    def __new__(cls, *args, **kwargs):
        return np.array(*args, subok=True, **kwargs).view(cls)


class SimpleSubClass(VerySimpleSubClass):
    def __new__(cls, *args, **kwargs):
        self = np.array(*args, subok=True, **kwargs).view(cls)
        self.info = 'simple'
        return self

    def __array_finalize__(self, obj):
        self.info = getattr(obj, 'info', '') + ' finalized'


def test_subclasses():
    # test that subclass is preserved only if subok=True
    a = VerySimpleSubClass([1, 2, 3, 4])
    assert_(type(a) is VerySimpleSubClass)
    a_view = as_strided(a, shape=(2,), strides=(2 * a.itemsize,))
    assert_(type(a_view) is np.ndarray)
    a_view = as_strided(a, shape=(2,), strides=(2 * a.itemsize,), subok=True)
    assert_(type(a_view) is VerySimpleSubClass)
    # test that if a subclass has __array_finalize__, it is used
    a = SimpleSubClass([1, 2, 3, 4])
    a_view = as_strided(a, shape=(2,), strides=(2 * a.itemsize,), subok=True)
    assert_(type(a_view) is SimpleSubClass)
    assert_(a_view.info == 'simple finalized')

    # similar tests for broadcast_arrays
    b = np.arange(len(a)).reshape(-1, 1)
    a_view, b_view = broadcast_arrays(a, b)
    assert_(type(a_view) is np.ndarray)
    assert_(type(b_view) is np.ndarray)
    assert_(a_view.shape == b_view.shape)
    a_view, b_view = broadcast_arrays(a, b, subok=True)
    assert_(type(a_view) is SimpleSubClass)
    assert_(a_view.info == 'simple finalized')
    assert_(type(b_view) is np.ndarray)
    assert_(a_view.shape == b_view.shape)

    # and for broadcast_to
    shape = (2, 4)
    a_view = broadcast_to(a, shape)
    assert_(type(a_view) is np.ndarray)
    assert_(a_view.shape == shape)
    a_view = broadcast_to(a, shape, subok=True)
    assert_(type(a_view) is SimpleSubClass)
    assert_(a_view.info == 'simple finalized')
    assert_(a_view.shape == shape)


def test_writeable():
    # broadcast_to should return a readonly array
    original = np.array([1, 2, 3])
    result = broadcast_to(original, (2, 3))
    assert_equal(result.flags.writeable, False)
    assert_raises(ValueError, result.__setitem__, slice(None), 0)

    # but the result of broadcast_arrays needs to be writeable, to
    # preserve backwards compatibility
    for is_broadcast, results in [(False, broadcast_arrays(original,)),
                                  (True, broadcast_arrays(0, original))]:
        for result in results:
            # This will change to False in a future version
            if is_broadcast:
                with assert_warns(FutureWarning):
                    assert_equal(result.flags.writeable, True)
                with assert_warns(DeprecationWarning):
                    result[:] = 0
                # Warning not emitted, writing to the array resets it
                assert_equal(result.flags.writeable, True)
            else:
                # No warning:
                assert_equal(result.flags.writeable, True)

    for results in [broadcast_arrays(original),
                    broadcast_arrays(0, original)]:
        for result in results:
            # resets the warn_on_write DeprecationWarning
            result.flags.writeable = True
            # check: no warning emitted
            assert_equal(result.flags.writeable, True)
            result[:] = 0

    # keep readonly input readonly
    original.flags.writeable = False
    _, result = broadcast_arrays(0, original)
    assert_equal(result.flags.writeable, False)

    # regression test for GH6491
    shape = (2,)
    strides = [0]
    tricky_array = as_strided(np.array(0), shape, strides)
    other = np.zeros((1,))
    first, second = broadcast_arrays(tricky_array, other)
    assert_(first.shape == second.shape)


def test_writeable_memoryview():
    # The result of broadcast_arrays exports as a non-writeable memoryview
    # because otherwise there is no good way to opt in to the new behaviour
    # (i.e. you would need to set writeable to False explicitly).
    # See gh-13929.
    original = np.array([1, 2, 3])

    for is_broadcast, results in [(False, broadcast_arrays(original,)),
                                  (True, broadcast_arrays(0, original))]:
        for result in results:
            # This will change to False in a future version
            if is_broadcast:
                # memoryview(result, writable=True) will give warning but cannot
                # be tested using the python API.
                assert memoryview(result).readonly
            else:
                assert not memoryview(result).readonly


def test_reference_types():
    input_array = np.array('a', dtype=object)
    expected = np.array(['a'] * 3, dtype=object)
    actual = broadcast_to(input_array, (3,))
    assert_array_equal(expected, actual)

    actual, _ = broadcast_arrays(input_array, np.ones(3))
    assert_array_equal(expected, actual)
