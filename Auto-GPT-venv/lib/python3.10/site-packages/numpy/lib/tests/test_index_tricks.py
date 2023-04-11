import pytest

import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_almost_equal,
    assert_array_almost_equal, assert_raises, assert_raises_regex,
    )
from numpy.lib.index_tricks import (
    mgrid, ogrid, ndenumerate, fill_diagonal, diag_indices, diag_indices_from,
    index_exp, ndindex, r_, s_, ix_
    )


class TestRavelUnravelIndex:
    def test_basic(self):
        assert_equal(np.unravel_index(2, (2, 2)), (1, 0))

        # test that new shape argument works properly
        assert_equal(np.unravel_index(indices=2,
                                      shape=(2, 2)),
                                      (1, 0))

        # test that an invalid second keyword argument
        # is properly handled, including the old name `dims`.
        with assert_raises(TypeError):
            np.unravel_index(indices=2, hape=(2, 2))

        with assert_raises(TypeError):
            np.unravel_index(2, hape=(2, 2))

        with assert_raises(TypeError):
            np.unravel_index(254, ims=(17, 94))

        with assert_raises(TypeError):
            np.unravel_index(254, dims=(17, 94))

        assert_equal(np.ravel_multi_index((1, 0), (2, 2)), 2)
        assert_equal(np.unravel_index(254, (17, 94)), (2, 66))
        assert_equal(np.ravel_multi_index((2, 66), (17, 94)), 254)
        assert_raises(ValueError, np.unravel_index, -1, (2, 2))
        assert_raises(TypeError, np.unravel_index, 0.5, (2, 2))
        assert_raises(ValueError, np.unravel_index, 4, (2, 2))
        assert_raises(ValueError, np.ravel_multi_index, (-3, 1), (2, 2))
        assert_raises(ValueError, np.ravel_multi_index, (2, 1), (2, 2))
        assert_raises(ValueError, np.ravel_multi_index, (0, -3), (2, 2))
        assert_raises(ValueError, np.ravel_multi_index, (0, 2), (2, 2))
        assert_raises(TypeError, np.ravel_multi_index, (0.1, 0.), (2, 2))

        assert_equal(np.unravel_index((2*3 + 1)*6 + 4, (4, 3, 6)), [2, 1, 4])
        assert_equal(
            np.ravel_multi_index([2, 1, 4], (4, 3, 6)), (2*3 + 1)*6 + 4)

        arr = np.array([[3, 6, 6], [4, 5, 1]])
        assert_equal(np.ravel_multi_index(arr, (7, 6)), [22, 41, 37])
        assert_equal(
            np.ravel_multi_index(arr, (7, 6), order='F'), [31, 41, 13])
        assert_equal(
            np.ravel_multi_index(arr, (4, 6), mode='clip'), [22, 23, 19])
        assert_equal(np.ravel_multi_index(arr, (4, 4), mode=('clip', 'wrap')),
                     [12, 13, 13])
        assert_equal(np.ravel_multi_index((3, 1, 4, 1), (6, 7, 8, 9)), 1621)

        assert_equal(np.unravel_index(np.array([22, 41, 37]), (7, 6)),
                     [[3, 6, 6], [4, 5, 1]])
        assert_equal(
            np.unravel_index(np.array([31, 41, 13]), (7, 6), order='F'),
            [[3, 6, 6], [4, 5, 1]])
        assert_equal(np.unravel_index(1621, (6, 7, 8, 9)), [3, 1, 4, 1])

    def test_empty_indices(self):
        msg1 = 'indices must be integral: the provided empty sequence was'
        msg2 = 'only int indices permitted'
        assert_raises_regex(TypeError, msg1, np.unravel_index, [], (10, 3, 5))
        assert_raises_regex(TypeError, msg1, np.unravel_index, (), (10, 3, 5))
        assert_raises_regex(TypeError, msg2, np.unravel_index, np.array([]),
                            (10, 3, 5))
        assert_equal(np.unravel_index(np.array([],dtype=int), (10, 3, 5)),
                     [[], [], []])
        assert_raises_regex(TypeError, msg1, np.ravel_multi_index, ([], []),
                            (10, 3))
        assert_raises_regex(TypeError, msg1, np.ravel_multi_index, ([], ['abc']),
                            (10, 3))
        assert_raises_regex(TypeError, msg2, np.ravel_multi_index,
                    (np.array([]), np.array([])), (5, 3))
        assert_equal(np.ravel_multi_index(
                (np.array([], dtype=int), np.array([], dtype=int)), (5, 3)), [])
        assert_equal(np.ravel_multi_index(np.array([[], []], dtype=int),
                     (5, 3)), [])

    def test_big_indices(self):
        # ravel_multi_index for big indices (issue #7546)
        if np.intp == np.int64:
            arr = ([1, 29], [3, 5], [3, 117], [19, 2],
                   [2379, 1284], [2, 2], [0, 1])
            assert_equal(
                np.ravel_multi_index(arr, (41, 7, 120, 36, 2706, 8, 6)),
                [5627771580, 117259570957])

        # test unravel_index for big indices (issue #9538)
        assert_raises(ValueError, np.unravel_index, 1, (2**32-1, 2**31+1))

        # test overflow checking for too big array (issue #7546)
        dummy_arr = ([0],[0])
        half_max = np.iinfo(np.intp).max // 2
        assert_equal(
            np.ravel_multi_index(dummy_arr, (half_max, 2)), [0])
        assert_raises(ValueError,
            np.ravel_multi_index, dummy_arr, (half_max+1, 2))
        assert_equal(
            np.ravel_multi_index(dummy_arr, (half_max, 2), order='F'), [0])
        assert_raises(ValueError,
            np.ravel_multi_index, dummy_arr, (half_max+1, 2), order='F')

    def test_dtypes(self):
        # Test with different data types
        for dtype in [np.int16, np.uint16, np.int32,
                      np.uint32, np.int64, np.uint64]:
            coords = np.array(
                [[1, 0, 1, 2, 3, 4], [1, 6, 1, 3, 2, 0]], dtype=dtype)
            shape = (5, 8)
            uncoords = 8*coords[0]+coords[1]
            assert_equal(np.ravel_multi_index(coords, shape), uncoords)
            assert_equal(coords, np.unravel_index(uncoords, shape))
            uncoords = coords[0]+5*coords[1]
            assert_equal(
                np.ravel_multi_index(coords, shape, order='F'), uncoords)
            assert_equal(coords, np.unravel_index(uncoords, shape, order='F'))

            coords = np.array(
                [[1, 0, 1, 2, 3, 4], [1, 6, 1, 3, 2, 0], [1, 3, 1, 0, 9, 5]],
                dtype=dtype)
            shape = (5, 8, 10)
            uncoords = 10*(8*coords[0]+coords[1])+coords[2]
            assert_equal(np.ravel_multi_index(coords, shape), uncoords)
            assert_equal(coords, np.unravel_index(uncoords, shape))
            uncoords = coords[0]+5*(coords[1]+8*coords[2])
            assert_equal(
                np.ravel_multi_index(coords, shape, order='F'), uncoords)
            assert_equal(coords, np.unravel_index(uncoords, shape, order='F'))

    def test_clipmodes(self):
        # Test clipmodes
        assert_equal(
            np.ravel_multi_index([5, 1, -1, 2], (4, 3, 7, 12), mode='wrap'),
            np.ravel_multi_index([1, 1, 6, 2], (4, 3, 7, 12)))
        assert_equal(np.ravel_multi_index([5, 1, -1, 2], (4, 3, 7, 12),
                                          mode=(
                                              'wrap', 'raise', 'clip', 'raise')),
                     np.ravel_multi_index([1, 1, 0, 2], (4, 3, 7, 12)))
        assert_raises(
            ValueError, np.ravel_multi_index, [5, 1, -1, 2], (4, 3, 7, 12))

    def test_writeability(self):
        # See gh-7269
        x, y = np.unravel_index([1, 2, 3], (4, 5))
        assert_(x.flags.writeable)
        assert_(y.flags.writeable)

    def test_0d(self):
        # gh-580
        x = np.unravel_index(0, ())
        assert_equal(x, ())

        assert_raises_regex(ValueError, "0d array", np.unravel_index, [0], ())
        assert_raises_regex(
            ValueError, "out of bounds", np.unravel_index, [1], ())

    @pytest.mark.parametrize("mode", ["clip", "wrap", "raise"])
    def test_empty_array_ravel(self, mode):
        res = np.ravel_multi_index(
                    np.zeros((3, 0), dtype=np.intp), (2, 1, 0), mode=mode)
        assert(res.shape == (0,))

        with assert_raises(ValueError):
            np.ravel_multi_index(
                    np.zeros((3, 1), dtype=np.intp), (2, 1, 0), mode=mode)

    def test_empty_array_unravel(self):
        res = np.unravel_index(np.zeros(0, dtype=np.intp), (2, 1, 0))
        # res is a tuple of three empty arrays
        assert(len(res) == 3)
        assert(all(a.shape == (0,) for a in res))

        with assert_raises(ValueError):
            np.unravel_index([1], (2, 1, 0))

class TestGrid:
    def test_basic(self):
        a = mgrid[-1:1:10j]
        b = mgrid[-1:1:0.1]
        assert_(a.shape == (10,))
        assert_(b.shape == (20,))
        assert_(a[0] == -1)
        assert_almost_equal(a[-1], 1)
        assert_(b[0] == -1)
        assert_almost_equal(b[1]-b[0], 0.1, 11)
        assert_almost_equal(b[-1], b[0]+19*0.1, 11)
        assert_almost_equal(a[1]-a[0], 2.0/9.0, 11)

    def test_linspace_equivalence(self):
        y, st = np.linspace(2, 10, retstep=True)
        assert_almost_equal(st, 8/49.0)
        assert_array_almost_equal(y, mgrid[2:10:50j], 13)

    def test_nd(self):
        c = mgrid[-1:1:10j, -2:2:10j]
        d = mgrid[-1:1:0.1, -2:2:0.2]
        assert_(c.shape == (2, 10, 10))
        assert_(d.shape == (2, 20, 20))
        assert_array_equal(c[0][0, :], -np.ones(10, 'd'))
        assert_array_equal(c[1][:, 0], -2*np.ones(10, 'd'))
        assert_array_almost_equal(c[0][-1, :], np.ones(10, 'd'), 11)
        assert_array_almost_equal(c[1][:, -1], 2*np.ones(10, 'd'), 11)
        assert_array_almost_equal(d[0, 1, :] - d[0, 0, :],
                                  0.1*np.ones(20, 'd'), 11)
        assert_array_almost_equal(d[1, :, 1] - d[1, :, 0],
                                  0.2*np.ones(20, 'd'), 11)

    def test_sparse(self):
        grid_full   = mgrid[-1:1:10j, -2:2:10j]
        grid_sparse = ogrid[-1:1:10j, -2:2:10j]

        # sparse grids can be made dense by broadcasting
        grid_broadcast = np.broadcast_arrays(*grid_sparse)
        for f, b in zip(grid_full, grid_broadcast):
            assert_equal(f, b)

    @pytest.mark.parametrize("start, stop, step, expected", [
        (None, 10, 10j, (200, 10)),
        (-10, 20, None, (1800, 30)),
        ])
    def test_mgrid_size_none_handling(self, start, stop, step, expected):
        # regression test None value handling for
        # start and step values used by mgrid;
        # internally, this aims to cover previously
        # unexplored code paths in nd_grid()
        grid = mgrid[start:stop:step, start:stop:step]
        # need a smaller grid to explore one of the
        # untested code paths
        grid_small = mgrid[start:stop:step]
        assert_equal(grid.size, expected[0])
        assert_equal(grid_small.size, expected[1])

    def test_accepts_npfloating(self):
        # regression test for #16466
        grid64 = mgrid[0.1:0.33:0.1, ]
        grid32 = mgrid[np.float32(0.1):np.float32(0.33):np.float32(0.1), ]
        assert_(grid32.dtype == np.float64)
        assert_array_almost_equal(grid64, grid32)

        # different code path for single slice
        grid64 = mgrid[0.1:0.33:0.1]
        grid32 = mgrid[np.float32(0.1):np.float32(0.33):np.float32(0.1)]
        assert_(grid32.dtype == np.float64)
        assert_array_almost_equal(grid64, grid32)

    def test_accepts_longdouble(self):
        # regression tests for #16945
        grid64 = mgrid[0.1:0.33:0.1, ]
        grid128 = mgrid[
            np.longdouble(0.1):np.longdouble(0.33):np.longdouble(0.1),
        ]
        assert_(grid128.dtype == np.longdouble)
        assert_array_almost_equal(grid64, grid128)

        grid128c_a = mgrid[0:np.longdouble(1):3.4j]
        grid128c_b = mgrid[0:np.longdouble(1):3.4j, ]
        assert_(grid128c_a.dtype == grid128c_b.dtype == np.longdouble)
        assert_array_equal(grid128c_a, grid128c_b[0])

        # different code path for single slice
        grid64 = mgrid[0.1:0.33:0.1]
        grid128 = mgrid[
            np.longdouble(0.1):np.longdouble(0.33):np.longdouble(0.1)
        ]
        assert_(grid128.dtype == np.longdouble)
        assert_array_almost_equal(grid64, grid128)

    def test_accepts_npcomplexfloating(self):
        # Related to #16466
        assert_array_almost_equal(
            mgrid[0.1:0.3:3j, ], mgrid[0.1:0.3:np.complex64(3j), ]
        )

        # different code path for single slice
        assert_array_almost_equal(
            mgrid[0.1:0.3:3j], mgrid[0.1:0.3:np.complex64(3j)]
        )

        # Related to #16945
        grid64_a = mgrid[0.1:0.3:3.3j]
        grid64_b = mgrid[0.1:0.3:3.3j, ][0]
        assert_(grid64_a.dtype == grid64_b.dtype == np.float64)
        assert_array_equal(grid64_a, grid64_b)

        grid128_a = mgrid[0.1:0.3:np.clongdouble(3.3j)]
        grid128_b = mgrid[0.1:0.3:np.clongdouble(3.3j), ][0]
        assert_(grid128_a.dtype == grid128_b.dtype == np.longdouble)
        assert_array_equal(grid64_a, grid64_b)


class TestConcatenator:
    def test_1d(self):
        assert_array_equal(r_[1, 2, 3, 4, 5, 6], np.array([1, 2, 3, 4, 5, 6]))
        b = np.ones(5)
        c = r_[b, 0, 0, b]
        assert_array_equal(c, [1, 1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1])

    def test_mixed_type(self):
        g = r_[10.1, 1:10]
        assert_(g.dtype == 'f8')

    def test_more_mixed_type(self):
        g = r_[-10.1, np.array([1]), np.array([2, 3, 4]), 10.0]
        assert_(g.dtype == 'f8')

    def test_complex_step(self):
        # Regression test for #12262
        g = r_[0:36:100j]
        assert_(g.shape == (100,))

        # Related to #16466
        g = r_[0:36:np.complex64(100j)]
        assert_(g.shape == (100,))

    def test_2d(self):
        b = np.random.rand(5, 5)
        c = np.random.rand(5, 5)
        d = r_['1', b, c]  # append columns
        assert_(d.shape == (5, 10))
        assert_array_equal(d[:, :5], b)
        assert_array_equal(d[:, 5:], c)
        d = r_[b, c]
        assert_(d.shape == (10, 5))
        assert_array_equal(d[:5, :], b)
        assert_array_equal(d[5:, :], c)

    def test_0d(self):
        assert_equal(r_[0, np.array(1), 2], [0, 1, 2])
        assert_equal(r_[[0, 1, 2], np.array(3)], [0, 1, 2, 3])
        assert_equal(r_[np.array(0), [1, 2, 3]], [0, 1, 2, 3])


class TestNdenumerate:
    def test_basic(self):
        a = np.array([[1, 2], [3, 4]])
        assert_equal(list(ndenumerate(a)),
                     [((0, 0), 1), ((0, 1), 2), ((1, 0), 3), ((1, 1), 4)])


class TestIndexExpression:
    def test_regression_1(self):
        # ticket #1196
        a = np.arange(2)
        assert_equal(a[:-1], a[s_[:-1]])
        assert_equal(a[:-1], a[index_exp[:-1]])

    def test_simple_1(self):
        a = np.random.rand(4, 5, 6)

        assert_equal(a[:, :3, [1, 2]], a[index_exp[:, :3, [1, 2]]])
        assert_equal(a[:, :3, [1, 2]], a[s_[:, :3, [1, 2]]])


class TestIx_:
    def test_regression_1(self):
        # Test empty untyped inputs create outputs of indexing type, gh-5804
        a, = np.ix_(range(0))
        assert_equal(a.dtype, np.intp)

        a, = np.ix_([])
        assert_equal(a.dtype, np.intp)

        # but if the type is specified, don't change it
        a, = np.ix_(np.array([], dtype=np.float32))
        assert_equal(a.dtype, np.float32)

    def test_shape_and_dtype(self):
        sizes = (4, 5, 3, 2)
        # Test both lists and arrays
        for func in (range, np.arange):
            arrays = np.ix_(*[func(sz) for sz in sizes])
            for k, (a, sz) in enumerate(zip(arrays, sizes)):
                assert_equal(a.shape[k], sz)
                assert_(all(sh == 1 for j, sh in enumerate(a.shape) if j != k))
                assert_(np.issubdtype(a.dtype, np.integer))

    def test_bool(self):
        bool_a = [True, False, True, True]
        int_a, = np.nonzero(bool_a)
        assert_equal(np.ix_(bool_a)[0], int_a)

    def test_1d_only(self):
        idx2d = [[1, 2, 3], [4, 5, 6]]
        assert_raises(ValueError, np.ix_, idx2d)

    def test_repeated_input(self):
        length_of_vector = 5
        x = np.arange(length_of_vector)
        out = ix_(x, x)
        assert_equal(out[0].shape, (length_of_vector, 1))
        assert_equal(out[1].shape, (1, length_of_vector))
        # check that input shape is not modified
        assert_equal(x.shape, (length_of_vector,))


def test_c_():
    a = np.c_[np.array([[1, 2, 3]]), 0, 0, np.array([[4, 5, 6]])]
    assert_equal(a, [[1, 2, 3, 0, 0, 4, 5, 6]])


class TestFillDiagonal:
    def test_basic(self):
        a = np.zeros((3, 3), int)
        fill_diagonal(a, 5)
        assert_array_equal(
            a, np.array([[5, 0, 0],
                         [0, 5, 0],
                         [0, 0, 5]])
            )

    def test_tall_matrix(self):
        a = np.zeros((10, 3), int)
        fill_diagonal(a, 5)
        assert_array_equal(
            a, np.array([[5, 0, 0],
                         [0, 5, 0],
                         [0, 0, 5],
                         [0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0],
                         [0, 0, 0]])
            )

    def test_tall_matrix_wrap(self):
        a = np.zeros((10, 3), int)
        fill_diagonal(a, 5, True)
        assert_array_equal(
            a, np.array([[5, 0, 0],
                         [0, 5, 0],
                         [0, 0, 5],
                         [0, 0, 0],
                         [5, 0, 0],
                         [0, 5, 0],
                         [0, 0, 5],
                         [0, 0, 0],
                         [5, 0, 0],
                         [0, 5, 0]])
            )

    def test_wide_matrix(self):
        a = np.zeros((3, 10), int)
        fill_diagonal(a, 5)
        assert_array_equal(
            a, np.array([[5, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 5, 0, 0, 0, 0, 0, 0, 0, 0],
                         [0, 0, 5, 0, 0, 0, 0, 0, 0, 0]])
            )

    def test_operate_4d_array(self):
        a = np.zeros((3, 3, 3, 3), int)
        fill_diagonal(a, 4)
        i = np.array([0, 1, 2])
        assert_equal(np.where(a != 0), (i, i, i, i))

    def test_low_dim_handling(self):
        # raise error with low dimensionality
        a = np.zeros(3, int)
        with assert_raises_regex(ValueError, "at least 2-d"):
            fill_diagonal(a, 5)

    def test_hetero_shape_handling(self):
        # raise error with high dimensionality and
        # shape mismatch
        a = np.zeros((3,3,7,3), int)
        with assert_raises_regex(ValueError, "equal length"):
            fill_diagonal(a, 2)


def test_diag_indices():
    di = diag_indices(4)
    a = np.array([[1, 2, 3, 4],
                  [5, 6, 7, 8],
                  [9, 10, 11, 12],
                  [13, 14, 15, 16]])
    a[di] = 100
    assert_array_equal(
        a, np.array([[100, 2, 3, 4],
                     [5, 100, 7, 8],
                     [9, 10, 100, 12],
                     [13, 14, 15, 100]])
        )

    # Now, we create indices to manipulate a 3-d array:
    d3 = diag_indices(2, 3)

    # And use it to set the diagonal of a zeros array to 1:
    a = np.zeros((2, 2, 2), int)
    a[d3] = 1
    assert_array_equal(
        a, np.array([[[1, 0],
                      [0, 0]],
                     [[0, 0],
                      [0, 1]]])
        )


class TestDiagIndicesFrom:

    def test_diag_indices_from(self):
        x = np.random.random((4, 4))
        r, c = diag_indices_from(x)
        assert_array_equal(r, np.arange(4))
        assert_array_equal(c, np.arange(4))

    def test_error_small_input(self):
        x = np.ones(7)
        with assert_raises_regex(ValueError, "at least 2-d"):
            diag_indices_from(x)

    def test_error_shape_mismatch(self):
        x = np.zeros((3, 3, 2, 3), int)
        with assert_raises_regex(ValueError, "equal length"):
            diag_indices_from(x)


def test_ndindex():
    x = list(ndindex(1, 2, 3))
    expected = [ix for ix, e in ndenumerate(np.zeros((1, 2, 3)))]
    assert_array_equal(x, expected)

    x = list(ndindex((1, 2, 3)))
    assert_array_equal(x, expected)

    # Test use of scalars and tuples
    x = list(ndindex((3,)))
    assert_array_equal(x, list(ndindex(3)))

    # Make sure size argument is optional
    x = list(ndindex())
    assert_equal(x, [()])

    x = list(ndindex(()))
    assert_equal(x, [()])

    # Make sure 0-sized ndindex works correctly
    x = list(ndindex(*[0]))
    assert_equal(x, [])
