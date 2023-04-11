import copy
import sys
import gc
import tempfile
import pytest
from os import path
from io import BytesIO
from itertools import chain

import numpy as np
from numpy.testing import (
        assert_, assert_equal, IS_PYPY, assert_almost_equal,
        assert_array_equal, assert_array_almost_equal, assert_raises,
        assert_raises_regex, assert_warns, suppress_warnings,
        _assert_valid_refcount, HAS_REFCOUNT, IS_PYSTON, IS_WASM
        )
from numpy.testing._private.utils import _no_tracing, requires_memory
from numpy.compat import asbytes, asunicode, pickle


class TestRegression:
    def test_invalid_round(self):
        # Ticket #3
        v = 4.7599999999999998
        assert_array_equal(np.array([v]), np.array(v))

    def test_mem_empty(self):
        # Ticket #7
        np.empty((1,), dtype=[('x', np.int64)])

    def test_pickle_transposed(self):
        # Ticket #16
        a = np.transpose(np.array([[2, 9], [7, 0], [3, 8]]))
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            with BytesIO() as f:
                pickle.dump(a, f, protocol=proto)
                f.seek(0)
                b = pickle.load(f)
            assert_array_equal(a, b)

    def test_dtype_names(self):
        # Ticket #35
        # Should succeed
        np.dtype([(('name', 'label'), np.int32, 3)])

    def test_reduce(self):
        # Ticket #40
        assert_almost_equal(np.add.reduce([1., .5], dtype=None), 1.5)

    def test_zeros_order(self):
        # Ticket #43
        np.zeros([3], int, 'C')
        np.zeros([3], order='C')
        np.zeros([3], int, order='C')

    def test_asarray_with_order(self):
        # Check that nothing is done when order='F' and array C/F-contiguous
        a = np.ones(2)
        assert_(a is np.asarray(a, order='F'))

    def test_ravel_with_order(self):
        # Check that ravel works when order='F' and array C/F-contiguous
        a = np.ones(2)
        assert_(not a.ravel('F').flags.owndata)

    def test_sort_bigendian(self):
        # Ticket #47
        a = np.linspace(0, 10, 11)
        c = a.astype(np.dtype('<f8'))
        c.sort()
        assert_array_almost_equal(c, a)

    def test_negative_nd_indexing(self):
        # Ticket #49
        c = np.arange(125).reshape((5, 5, 5))
        origidx = np.array([-1, 0, 1])
        idx = np.array(origidx)
        c[idx]
        assert_array_equal(idx, origidx)

    def test_char_dump(self):
        # Ticket #50
        ca = np.char.array(np.arange(1000, 1010), itemsize=4)
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            with BytesIO() as f:
                pickle.dump(ca, f, protocol=proto)
                f.seek(0)
                ca = np.load(f, allow_pickle=True)

    def test_noncontiguous_fill(self):
        # Ticket #58.
        a = np.zeros((5, 3))
        b = a[:, :2,]

        def rs():
            b.shape = (10,)

        assert_raises(AttributeError, rs)

    def test_bool(self):
        # Ticket #60
        np.bool_(1)  # Should succeed

    def test_indexing1(self):
        # Ticket #64
        descr = [('x', [('y', [('z', 'c16', (2,)),]),]),]
        buffer = ((([6j, 4j],),),)
        h = np.array(buffer, dtype=descr)
        h['x']['y']['z']

    def test_indexing2(self):
        # Ticket #65
        descr = [('x', 'i4', (2,))]
        buffer = ([3, 2],)
        h = np.array(buffer, dtype=descr)
        h['x']

    def test_round(self):
        # Ticket #67
        x = np.array([1+2j])
        assert_almost_equal(x**(-1), [1/(1+2j)])

    def test_scalar_compare(self):
        # Trac Ticket #72
        # https://github.com/numpy/numpy/issues/565
        a = np.array(['test', 'auto'])
        assert_array_equal(a == 'auto', np.array([False, True]))
        assert_(a[1] == 'auto')
        assert_(a[0] != 'auto')
        b = np.linspace(0, 10, 11)
        # This should return true for now, but will eventually raise an error:
        with suppress_warnings() as sup:
            sup.filter(FutureWarning)
            assert_(b != 'auto')
        assert_(b[0] != 'auto')

    def test_unicode_swapping(self):
        # Ticket #79
        ulen = 1
        ucs_value = '\U0010FFFF'
        ua = np.array([[[ucs_value*ulen]*2]*3]*4, dtype='U%s' % ulen)
        ua.newbyteorder()  # Should succeed.

    def test_object_array_fill(self):
        # Ticket #86
        x = np.zeros(1, 'O')
        x.fill([])

    def test_mem_dtype_align(self):
        # Ticket #93
        assert_raises(TypeError, np.dtype,
                              {'names':['a'], 'formats':['foo']}, align=1)

    def test_endian_bool_indexing(self):
        # Ticket #105
        a = np.arange(10., dtype='>f8')
        b = np.arange(10., dtype='<f8')
        xa = np.where((a > 2) & (a < 6))
        xb = np.where((b > 2) & (b < 6))
        ya = ((a > 2) & (a < 6))
        yb = ((b > 2) & (b < 6))
        assert_array_almost_equal(xa, ya.nonzero())
        assert_array_almost_equal(xb, yb.nonzero())
        assert_(np.all(a[ya] > 0.5))
        assert_(np.all(b[yb] > 0.5))

    def test_endian_where(self):
        # GitHub issue #369
        net = np.zeros(3, dtype='>f4')
        net[1] = 0.00458849
        net[2] = 0.605202
        max_net = net.max()
        test = np.where(net <= 0., max_net, net)
        correct = np.array([ 0.60520202,  0.00458849,  0.60520202])
        assert_array_almost_equal(test, correct)

    def test_endian_recarray(self):
        # Ticket #2185
        dt = np.dtype([
               ('head', '>u4'),
               ('data', '>u4', 2),
            ])
        buf = np.recarray(1, dtype=dt)
        buf[0]['head'] = 1
        buf[0]['data'][:] = [1, 1]

        h = buf[0]['head']
        d = buf[0]['data'][0]
        buf[0]['head'] = h
        buf[0]['data'][0] = d
        assert_(buf[0]['head'] == 1)

    def test_mem_dot(self):
        # Ticket #106
        x = np.random.randn(0, 1)
        y = np.random.randn(10, 1)
        # Dummy array to detect bad memory access:
        _z = np.ones(10)
        _dummy = np.empty((0, 10))
        z = np.lib.stride_tricks.as_strided(_z, _dummy.shape, _dummy.strides)
        np.dot(x, np.transpose(y), out=z)
        assert_equal(_z, np.ones(10))
        # Do the same for the built-in dot:
        np.core.multiarray.dot(x, np.transpose(y), out=z)
        assert_equal(_z, np.ones(10))

    def test_arange_endian(self):
        # Ticket #111
        ref = np.arange(10)
        x = np.arange(10, dtype='<f8')
        assert_array_equal(ref, x)
        x = np.arange(10, dtype='>f8')
        assert_array_equal(ref, x)

    def test_arange_inf_step(self):
        ref = np.arange(0, 1, 10)
        x = np.arange(0, 1, np.inf)
        assert_array_equal(ref, x)

        ref = np.arange(0, 1, -10)
        x = np.arange(0, 1, -np.inf)
        assert_array_equal(ref, x)

        ref = np.arange(0, -1, -10)
        x = np.arange(0, -1, -np.inf)
        assert_array_equal(ref, x)

        ref = np.arange(0, -1, 10)
        x = np.arange(0, -1, np.inf)
        assert_array_equal(ref, x)

    def test_arange_underflow_stop_and_step(self):
        finfo = np.finfo(np.float64)

        ref = np.arange(0, finfo.eps, 2 * finfo.eps)
        x = np.arange(0, finfo.eps, finfo.max)
        assert_array_equal(ref, x)

        ref = np.arange(0, finfo.eps, -2 * finfo.eps)
        x = np.arange(0, finfo.eps, -finfo.max)
        assert_array_equal(ref, x)

        ref = np.arange(0, -finfo.eps, -2 * finfo.eps)
        x = np.arange(0, -finfo.eps, -finfo.max)
        assert_array_equal(ref, x)

        ref = np.arange(0, -finfo.eps, 2 * finfo.eps)
        x = np.arange(0, -finfo.eps, finfo.max)
        assert_array_equal(ref, x)

    def test_argmax(self):
        # Ticket #119
        a = np.random.normal(0, 1, (4, 5, 6, 7, 8))
        for i in range(a.ndim):
            a.argmax(i)  # Should succeed

    def test_mem_divmod(self):
        # Ticket #126
        for i in range(10):
            divmod(np.array([i])[0], 10)

    def test_hstack_invalid_dims(self):
        # Ticket #128
        x = np.arange(9).reshape((3, 3))
        y = np.array([0, 0, 0])
        assert_raises(ValueError, np.hstack, (x, y))

    def test_squeeze_type(self):
        # Ticket #133
        a = np.array([3])
        b = np.array(3)
        assert_(type(a.squeeze()) is np.ndarray)
        assert_(type(b.squeeze()) is np.ndarray)

    def test_add_identity(self):
        # Ticket #143
        assert_equal(0, np.add.identity)

    def test_numpy_float_python_long_addition(self):
        # Check that numpy float and python longs can be added correctly.
        a = np.float_(23.) + 2**135
        assert_equal(a, 23. + 2**135)

    def test_binary_repr_0(self):
        # Ticket #151
        assert_equal('0', np.binary_repr(0))

    def test_rec_iterate(self):
        # Ticket #160
        descr = np.dtype([('i', int), ('f', float), ('s', '|S3')])
        x = np.rec.array([(1, 1.1, '1.0'),
                         (2, 2.2, '2.0')], dtype=descr)
        x[0].tolist()
        [i for i in x[0]]

    def test_unicode_string_comparison(self):
        # Ticket #190
        a = np.array('hello', np.unicode_)
        b = np.array('world')
        a == b

    def test_tobytes_FORTRANORDER_discontiguous(self):
        # Fix in r2836
        # Create non-contiguous Fortran ordered array
        x = np.array(np.random.rand(3, 3), order='F')[:, :2]
        assert_array_almost_equal(x.ravel(), np.frombuffer(x.tobytes()))

    def test_flat_assignment(self):
        # Correct behaviour of ticket #194
        x = np.empty((3, 1))
        x.flat = np.arange(3)
        assert_array_almost_equal(x, [[0], [1], [2]])
        x.flat = np.arange(3, dtype=float)
        assert_array_almost_equal(x, [[0], [1], [2]])

    def test_broadcast_flat_assignment(self):
        # Ticket #194
        x = np.empty((3, 1))

        def bfa():
            x[:] = np.arange(3)

        def bfb():
            x[:] = np.arange(3, dtype=float)

        assert_raises(ValueError, bfa)
        assert_raises(ValueError, bfb)

    @pytest.mark.xfail(IS_WASM, reason="not sure why")
    @pytest.mark.parametrize("index",
            [np.ones(10, dtype=bool), np.arange(10)],
            ids=["boolean-arr-index", "integer-arr-index"])
    def test_nonarray_assignment(self, index):
        # See also Issue gh-2870, test for non-array assignment
        # and equivalent unsafe casted array assignment
        a = np.arange(10)

        with pytest.raises(ValueError):
            a[index] = np.nan

        with np.errstate(invalid="warn"):
            with pytest.warns(RuntimeWarning, match="invalid value"):
                a[index] = np.array(np.nan)  # Only warns

    def test_unpickle_dtype_with_object(self):
        # Implemented in r2840
        dt = np.dtype([('x', int), ('y', np.object_), ('z', 'O')])
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            with BytesIO() as f:
                pickle.dump(dt, f, protocol=proto)
                f.seek(0)
                dt_ = pickle.load(f)
            assert_equal(dt, dt_)

    def test_mem_array_creation_invalid_specification(self):
        # Ticket #196
        dt = np.dtype([('x', int), ('y', np.object_)])
        # Wrong way
        assert_raises(ValueError, np.array, [1, 'object'], dt)
        # Correct way
        np.array([(1, 'object')], dt)

    def test_recarray_single_element(self):
        # Ticket #202
        a = np.array([1, 2, 3], dtype=np.int32)
        b = a.copy()
        r = np.rec.array(a, shape=1, formats=['3i4'], names=['d'])
        assert_array_equal(a, b)
        assert_equal(a, r[0][0])

    def test_zero_sized_array_indexing(self):
        # Ticket #205
        tmp = np.array([])

        def index_tmp():
            tmp[np.array(10)]

        assert_raises(IndexError, index_tmp)

    def test_chararray_rstrip(self):
        # Ticket #222
        x = np.chararray((1,), 5)
        x[0] = b'a   '
        x = x.rstrip()
        assert_equal(x[0], b'a')

    def test_object_array_shape(self):
        # Ticket #239
        assert_equal(np.array([[1, 2], 3, 4], dtype=object).shape, (3,))
        assert_equal(np.array([[1, 2], [3, 4]], dtype=object).shape, (2, 2))
        assert_equal(np.array([(1, 2), (3, 4)], dtype=object).shape, (2, 2))
        assert_equal(np.array([], dtype=object).shape, (0,))
        assert_equal(np.array([[], [], []], dtype=object).shape, (3, 0))
        assert_equal(np.array([[3, 4], [5, 6], None], dtype=object).shape, (3,))

    def test_mem_around(self):
        # Ticket #243
        x = np.zeros((1,))
        y = [0]
        decimal = 6
        np.around(abs(x-y), decimal) <= 10.0**(-decimal)

    def test_character_array_strip(self):
        # Ticket #246
        x = np.char.array(("x", "x ", "x  "))
        for c in x:
            assert_equal(c, "x")

    def test_lexsort(self):
        # Lexsort memory error
        v = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
        assert_equal(np.lexsort(v), 0)

    def test_lexsort_invalid_sequence(self):
        # Issue gh-4123
        class BuggySequence:
            def __len__(self):
                return 4

            def __getitem__(self, key):
                raise KeyError

        assert_raises(KeyError, np.lexsort, BuggySequence())

    def test_lexsort_zerolen_custom_strides(self):
        # Ticket #14228
        xs = np.array([], dtype='i8')
        assert np.lexsort((xs,)).shape[0] == 0 # Works

        xs.strides = (16,)
        assert np.lexsort((xs,)).shape[0] == 0 # Was: MemoryError

    def test_lexsort_zerolen_custom_strides_2d(self):
        xs = np.array([], dtype='i8')

        xs.shape = (0, 2)
        xs.strides = (16, 16)
        assert np.lexsort((xs,), axis=0).shape[0] == 0

        xs.shape = (2, 0)
        xs.strides = (16, 16)
        assert np.lexsort((xs,), axis=0).shape[0] == 2

    def test_lexsort_invalid_axis(self):
        assert_raises(np.AxisError, np.lexsort, (np.arange(1),), axis=2)
        assert_raises(np.AxisError, np.lexsort, (np.array([]),), axis=1)
        assert_raises(np.AxisError, np.lexsort, (np.array(1),), axis=10)

    def test_lexsort_zerolen_element(self):
        dt = np.dtype([])  # a void dtype with no fields
        xs = np.empty(4, dt)

        assert np.lexsort((xs,)).shape[0] == xs.shape[0]

    def test_pickle_py2_bytes_encoding(self):
        # Check that arrays and scalars pickled on Py2 are
        # unpickleable on Py3 using encoding='bytes'

        test_data = [
            # (original, py2_pickle)
            (np.unicode_('\u6f2c'),
             b"cnumpy.core.multiarray\nscalar\np0\n(cnumpy\ndtype\np1\n"
             b"(S'U1'\np2\nI0\nI1\ntp3\nRp4\n(I3\nS'<'\np5\nNNNI4\nI4\n"
             b"I0\ntp6\nbS',o\\x00\\x00'\np7\ntp8\nRp9\n."),

            (np.array([9e123], dtype=np.float64),
             b"cnumpy.core.multiarray\n_reconstruct\np0\n(cnumpy\nndarray\n"
             b"p1\n(I0\ntp2\nS'b'\np3\ntp4\nRp5\n(I1\n(I1\ntp6\ncnumpy\ndtype\n"
             b"p7\n(S'f8'\np8\nI0\nI1\ntp9\nRp10\n(I3\nS'<'\np11\nNNNI-1\nI-1\n"
             b"I0\ntp12\nbI00\nS'O\\x81\\xb7Z\\xaa:\\xabY'\np13\ntp14\nb."),

            (np.array([(9e123,)], dtype=[('name', float)]),
             b"cnumpy.core.multiarray\n_reconstruct\np0\n(cnumpy\nndarray\np1\n"
             b"(I0\ntp2\nS'b'\np3\ntp4\nRp5\n(I1\n(I1\ntp6\ncnumpy\ndtype\np7\n"
             b"(S'V8'\np8\nI0\nI1\ntp9\nRp10\n(I3\nS'|'\np11\nN(S'name'\np12\ntp13\n"
             b"(dp14\ng12\n(g7\n(S'f8'\np15\nI0\nI1\ntp16\nRp17\n(I3\nS'<'\np18\nNNNI-1\n"
             b"I-1\nI0\ntp19\nbI0\ntp20\nsI8\nI1\nI0\ntp21\n"
             b"bI00\nS'O\\x81\\xb7Z\\xaa:\\xabY'\np22\ntp23\nb."),
        ]

        for original, data in test_data:
            result = pickle.loads(data, encoding='bytes')
            assert_equal(result, original)

            if isinstance(result, np.ndarray) and result.dtype.names is not None:
                for name in result.dtype.names:
                    assert_(isinstance(name, str))

    def test_pickle_dtype(self):
        # Ticket #251
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            pickle.dumps(float, protocol=proto)

    def test_swap_real(self):
        # Ticket #265
        assert_equal(np.arange(4, dtype='>c8').imag.max(), 0.0)
        assert_equal(np.arange(4, dtype='<c8').imag.max(), 0.0)
        assert_equal(np.arange(4, dtype='>c8').real.max(), 3.0)
        assert_equal(np.arange(4, dtype='<c8').real.max(), 3.0)

    def test_object_array_from_list(self):
        # Ticket #270 (gh-868)
        assert_(np.array([1, None, 'A']).shape == (3,))

    def test_multiple_assign(self):
        # Ticket #273
        a = np.zeros((3, 1), int)
        a[[1, 2]] = 1

    def test_empty_array_type(self):
        assert_equal(np.array([]).dtype, np.zeros(0).dtype)

    def test_void_copyswap(self):
        dt = np.dtype([('one', '<i4'), ('two', '<i4')])
        x = np.array((1, 2), dtype=dt)
        x = x.byteswap()
        assert_(x['one'] > 1 and x['two'] > 2)

    def test_method_args(self):
        # Make sure methods and functions have same default axis
        # keyword and arguments
        funcs1 = ['argmax', 'argmin', 'sum', ('product', 'prod'),
                 ('sometrue', 'any'),
                 ('alltrue', 'all'), 'cumsum', ('cumproduct', 'cumprod'),
                 'ptp', 'cumprod', 'prod', 'std', 'var', 'mean',
                 'round', 'min', 'max', 'argsort', 'sort']
        funcs2 = ['compress', 'take', 'repeat']

        for func in funcs1:
            arr = np.random.rand(8, 7)
            arr2 = arr.copy()
            if isinstance(func, tuple):
                func_meth = func[1]
                func = func[0]
            else:
                func_meth = func
            res1 = getattr(arr, func_meth)()
            res2 = getattr(np, func)(arr2)
            if res1 is None:
                res1 = arr

            if res1.dtype.kind in 'uib':
                assert_((res1 == res2).all(), func)
            else:
                assert_(abs(res1-res2).max() < 1e-8, func)

        for func in funcs2:
            arr1 = np.random.rand(8, 7)
            arr2 = np.random.rand(8, 7)
            res1 = None
            if func == 'compress':
                arr1 = arr1.ravel()
                res1 = getattr(arr2, func)(arr1)
            else:
                arr2 = (15*arr2).astype(int).ravel()
            if res1 is None:
                res1 = getattr(arr1, func)(arr2)
            res2 = getattr(np, func)(arr1, arr2)
            assert_(abs(res1-res2).max() < 1e-8, func)

    def test_mem_lexsort_strings(self):
        # Ticket #298
        lst = ['abc', 'cde', 'fgh']
        np.lexsort((lst,))

    def test_fancy_index(self):
        # Ticket #302
        x = np.array([1, 2])[np.array([0])]
        assert_equal(x.shape, (1,))

    def test_recarray_copy(self):
        # Ticket #312
        dt = [('x', np.int16), ('y', np.float64)]
        ra = np.array([(1, 2.3)], dtype=dt)
        rb = np.rec.array(ra, dtype=dt)
        rb['x'] = 2.
        assert_(ra['x'] != rb['x'])

    def test_rec_fromarray(self):
        # Ticket #322
        x1 = np.array([[1, 2], [3, 4], [5, 6]])
        x2 = np.array(['a', 'dd', 'xyz'])
        x3 = np.array([1.1, 2, 3])
        np.rec.fromarrays([x1, x2, x3], formats="(2,)i4,a3,f8")

    def test_object_array_assign(self):
        x = np.empty((2, 2), object)
        x.flat[2] = (1, 2, 3)
        assert_equal(x.flat[2], (1, 2, 3))

    def test_ndmin_float64(self):
        # Ticket #324
        x = np.array([1, 2, 3], dtype=np.float64)
        assert_equal(np.array(x, dtype=np.float32, ndmin=2).ndim, 2)
        assert_equal(np.array(x, dtype=np.float64, ndmin=2).ndim, 2)

    def test_ndmin_order(self):
        # Issue #465 and related checks
        assert_(np.array([1, 2], order='C', ndmin=3).flags.c_contiguous)
        assert_(np.array([1, 2], order='F', ndmin=3).flags.f_contiguous)
        assert_(np.array(np.ones((2, 2), order='F'), ndmin=3).flags.f_contiguous)
        assert_(np.array(np.ones((2, 2), order='C'), ndmin=3).flags.c_contiguous)

    def test_mem_axis_minimization(self):
        # Ticket #327
        data = np.arange(5)
        data = np.add.outer(data, data)

    def test_mem_float_imag(self):
        # Ticket #330
        np.float64(1.0).imag

    def test_dtype_tuple(self):
        # Ticket #334
        assert_(np.dtype('i4') == np.dtype(('i4', ())))

    def test_dtype_posttuple(self):
        # Ticket #335
        np.dtype([('col1', '()i4')])

    def test_numeric_carray_compare(self):
        # Ticket #341
        assert_equal(np.array(['X'], 'c'), b'X')

    def test_string_array_size(self):
        # Ticket #342
        assert_raises(ValueError,
                              np.array, [['X'], ['X', 'X', 'X']], '|S1')

    def test_dtype_repr(self):
        # Ticket #344
        dt1 = np.dtype(('uint32', 2))
        dt2 = np.dtype(('uint32', (2,)))
        assert_equal(dt1.__repr__(), dt2.__repr__())

    def test_reshape_order(self):
        # Make sure reshape order works.
        a = np.arange(6).reshape(2, 3, order='F')
        assert_equal(a, [[0, 2, 4], [1, 3, 5]])
        a = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        b = a[:, 1]
        assert_equal(b.reshape(2, 2, order='F'), [[2, 6], [4, 8]])

    def test_reshape_zero_strides(self):
        # Issue #380, test reshaping of zero strided arrays
        a = np.ones(1)
        a = np.lib.stride_tricks.as_strided(a, shape=(5,), strides=(0,))
        assert_(a.reshape(5, 1).strides[0] == 0)

    def test_reshape_zero_size(self):
        # GitHub Issue #2700, setting shape failed for 0-sized arrays
        a = np.ones((0, 2))
        a.shape = (-1, 2)

    # Cannot test if NPY_RELAXED_STRIDES_DEBUG changes the strides.
    # With NPY_RELAXED_STRIDES_DEBUG the test becomes superfluous.
    @pytest.mark.skipif(np.ones(1).strides[0] == np.iinfo(np.intp).max,
                        reason="Using relaxed stride debug")
    def test_reshape_trailing_ones_strides(self):
        # GitHub issue gh-2949, bad strides for trailing ones of new shape
        a = np.zeros(12, dtype=np.int32)[::2]  # not contiguous
        strides_c = (16, 8, 8, 8)
        strides_f = (8, 24, 48, 48)
        assert_equal(a.reshape(3, 2, 1, 1).strides, strides_c)
        assert_equal(a.reshape(3, 2, 1, 1, order='F').strides, strides_f)
        assert_equal(np.array(0, dtype=np.int32).reshape(1, 1).strides, (4, 4))

    def test_repeat_discont(self):
        # Ticket #352
        a = np.arange(12).reshape(4, 3)[:, 2]
        assert_equal(a.repeat(3), [2, 2, 2, 5, 5, 5, 8, 8, 8, 11, 11, 11])

    def test_array_index(self):
        # Make sure optimization is not called in this case.
        a = np.array([1, 2, 3])
        a2 = np.array([[1, 2, 3]])
        assert_equal(a[np.where(a == 3)], a2[np.where(a2 == 3)])

    def test_object_argmax(self):
        a = np.array([1, 2, 3], dtype=object)
        assert_(a.argmax() == 2)

    def test_recarray_fields(self):
        # Ticket #372
        dt0 = np.dtype([('f0', 'i4'), ('f1', 'i4')])
        dt1 = np.dtype([('f0', 'i8'), ('f1', 'i8')])
        for a in [np.array([(1, 2), (3, 4)], "i4,i4"),
                  np.rec.array([(1, 2), (3, 4)], "i4,i4"),
                  np.rec.array([(1, 2), (3, 4)]),
                  np.rec.fromarrays([(1, 2), (3, 4)], "i4,i4"),
                  np.rec.fromarrays([(1, 2), (3, 4)])]:
            assert_(a.dtype in [dt0, dt1])

    def test_random_shuffle(self):
        # Ticket #374
        a = np.arange(5).reshape((5, 1))
        b = a.copy()
        np.random.shuffle(b)
        assert_equal(np.sort(b, axis=0), a)

    def test_refcount_vdot(self):
        # Changeset #3443
        _assert_valid_refcount(np.vdot)

    def test_startswith(self):
        ca = np.char.array(['Hi', 'There'])
        assert_equal(ca.startswith('H'), [True, False])

    def test_noncommutative_reduce_accumulate(self):
        # Ticket #413
        tosubtract = np.arange(5)
        todivide = np.array([2.0, 0.5, 0.25])
        assert_equal(np.subtract.reduce(tosubtract), -10)
        assert_equal(np.divide.reduce(todivide), 16.0)
        assert_array_equal(np.subtract.accumulate(tosubtract),
            np.array([0, -1, -3, -6, -10]))
        assert_array_equal(np.divide.accumulate(todivide),
            np.array([2., 4., 16.]))

    def test_convolve_empty(self):
        # Convolve should raise an error for empty input array.
        assert_raises(ValueError, np.convolve, [], [1])
        assert_raises(ValueError, np.convolve, [1], [])

    def test_multidim_byteswap(self):
        # Ticket #449
        r = np.array([(1, (0, 1, 2))], dtype="i2,3i2")
        assert_array_equal(r.byteswap(),
                           np.array([(256, (0, 256, 512))], r.dtype))

    def test_string_NULL(self):
        # Changeset 3557
        assert_equal(np.array("a\x00\x0b\x0c\x00").item(),
                     'a\x00\x0b\x0c')

    def test_junk_in_string_fields_of_recarray(self):
        # Ticket #483
        r = np.array([[b'abc']], dtype=[('var1', '|S20')])
        assert_(asbytes(r['var1'][0][0]) == b'abc')

    def test_take_output(self):
        # Ensure that 'take' honours output parameter.
        x = np.arange(12).reshape((3, 4))
        a = np.take(x, [0, 2], axis=1)
        b = np.zeros_like(a)
        np.take(x, [0, 2], axis=1, out=b)
        assert_array_equal(a, b)

    def test_take_object_fail(self):
        # Issue gh-3001
        d = 123.
        a = np.array([d, 1], dtype=object)
        if HAS_REFCOUNT:
            ref_d = sys.getrefcount(d)
        try:
            a.take([0, 100])
        except IndexError:
            pass
        if HAS_REFCOUNT:
            assert_(ref_d == sys.getrefcount(d))

    def test_array_str_64bit(self):
        # Ticket #501
        s = np.array([1, np.nan], dtype=np.float64)
        with np.errstate(all='raise'):
            np.array_str(s)  # Should succeed

    def test_frompyfunc_endian(self):
        # Ticket #503
        from math import radians
        uradians = np.frompyfunc(radians, 1, 1)
        big_endian = np.array([83.4, 83.5], dtype='>f8')
        little_endian = np.array([83.4, 83.5], dtype='<f8')
        assert_almost_equal(uradians(big_endian).astype(float),
                            uradians(little_endian).astype(float))

    def test_mem_string_arr(self):
        # Ticket #514
        s = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa"
        t = []
        np.hstack((t, s))

    def test_arr_transpose(self):
        # Ticket #516
        x = np.random.rand(*(2,)*16)
        x.transpose(list(range(16)))  # Should succeed

    def test_string_mergesort(self):
        # Ticket #540
        x = np.array(['a']*32)
        assert_array_equal(x.argsort(kind='m'), np.arange(32))

    def test_argmax_byteorder(self):
        # Ticket #546
        a = np.arange(3, dtype='>f')
        assert_(a[a.argmax()] == a.max())

    def test_rand_seed(self):
        # Ticket #555
        for l in np.arange(4):
            np.random.seed(l)

    def test_mem_deallocation_leak(self):
        # Ticket #562
        a = np.zeros(5, dtype=float)
        b = np.array(a, dtype=float)
        del a, b

    def test_mem_on_invalid_dtype(self):
        "Ticket #583"
        assert_raises(ValueError, np.fromiter, [['12', ''], ['13', '']], str)

    def test_dot_negative_stride(self):
        # Ticket #588
        x = np.array([[1, 5, 25, 125., 625]])
        y = np.array([[20.], [160.], [640.], [1280.], [1024.]])
        z = y[::-1].copy()
        y2 = y[::-1]
        assert_equal(np.dot(x, z), np.dot(x, y2))

    def test_object_casting(self):
        # This used to trigger the object-type version of
        # the bitwise_or operation, because float64 -> object
        # casting succeeds
        def rs():
            x = np.ones([484, 286])
            y = np.zeros([484, 286])
            x |= y

        assert_raises(TypeError, rs)

    def test_unicode_scalar(self):
        # Ticket #600
        x = np.array(["DROND", "DROND1"], dtype="U6")
        el = x[1]
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            new = pickle.loads(pickle.dumps(el, protocol=proto))
            assert_equal(new, el)

    def test_arange_non_native_dtype(self):
        # Ticket #616
        for T in ('>f4', '<f4'):
            dt = np.dtype(T)
            assert_equal(np.arange(0, dtype=dt).dtype, dt)
            assert_equal(np.arange(0.5, dtype=dt).dtype, dt)
            assert_equal(np.arange(5, dtype=dt).dtype, dt)

    def test_bool_flat_indexing_invalid_nr_elements(self):
        s = np.ones(10, dtype=float)
        x = np.array((15,), dtype=float)

        def ia(x, s, v):
            x[(s > 0)] = v

        assert_raises(IndexError, ia, x, s, np.zeros(9, dtype=float))
        assert_raises(IndexError, ia, x, s, np.zeros(11, dtype=float))

        # Old special case (different code path):
        assert_raises(ValueError, ia, x.flat, s, np.zeros(9, dtype=float))
        assert_raises(ValueError, ia, x.flat, s, np.zeros(11, dtype=float))

    def test_mem_scalar_indexing(self):
        # Ticket #603
        x = np.array([0], dtype=float)
        index = np.array(0, dtype=np.int32)
        x[index]

    def test_binary_repr_0_width(self):
        assert_equal(np.binary_repr(0, width=3), '000')

    def test_fromstring(self):
        assert_equal(np.fromstring("12:09:09", dtype=int, sep=":"),
                     [12, 9, 9])

    def test_searchsorted_variable_length(self):
        x = np.array(['a', 'aa', 'b'])
        y = np.array(['d', 'e'])
        assert_equal(x.searchsorted(y), [3, 3])

    def test_string_argsort_with_zeros(self):
        # Check argsort for strings containing zeros.
        x = np.frombuffer(b"\x00\x02\x00\x01", dtype="|S2")
        assert_array_equal(x.argsort(kind='m'), np.array([1, 0]))
        assert_array_equal(x.argsort(kind='q'), np.array([1, 0]))

    def test_string_sort_with_zeros(self):
        # Check sort for strings containing zeros.
        x = np.frombuffer(b"\x00\x02\x00\x01", dtype="|S2")
        y = np.frombuffer(b"\x00\x01\x00\x02", dtype="|S2")
        assert_array_equal(np.sort(x, kind="q"), y)

    def test_copy_detection_zero_dim(self):
        # Ticket #658
        np.indices((0, 3, 4)).T.reshape(-1, 3)

    def test_flat_byteorder(self):
        # Ticket #657
        x = np.arange(10)
        assert_array_equal(x.astype('>i4'), x.astype('<i4').flat[:])
        assert_array_equal(x.astype('>i4').flat[:], x.astype('<i4'))

    def test_sign_bit(self):
        x = np.array([0, -0.0, 0])
        assert_equal(str(np.abs(x)), '[0. 0. 0.]')

    def test_flat_index_byteswap(self):
        for dt in (np.dtype('<i4'), np.dtype('>i4')):
            x = np.array([-1, 0, 1], dtype=dt)
            assert_equal(x.flat[0].dtype, x[0].dtype)

    def test_copy_detection_corner_case(self):
        # Ticket #658
        np.indices((0, 3, 4)).T.reshape(-1, 3)

    # Cannot test if NPY_RELAXED_STRIDES_DEBUG changes the strides.
    # With NPY_RELAXED_STRIDES_DEBUG the test becomes superfluous,
    # 0-sized reshape itself is tested elsewhere.
    @pytest.mark.skipif(np.ones(1).strides[0] == np.iinfo(np.intp).max,
                        reason="Using relaxed stride debug")
    def test_copy_detection_corner_case2(self):
        # Ticket #771: strides are not set correctly when reshaping 0-sized
        # arrays
        b = np.indices((0, 3, 4)).T.reshape(-1, 3)
        assert_equal(b.strides, (3 * b.itemsize, b.itemsize))

    def test_object_array_refcounting(self):
        # Ticket #633
        if not hasattr(sys, 'getrefcount'):
            return

        # NB. this is probably CPython-specific

        cnt = sys.getrefcount

        a = object()
        b = object()
        c = object()

        cnt0_a = cnt(a)
        cnt0_b = cnt(b)
        cnt0_c = cnt(c)

        # -- 0d -> 1-d broadcast slice assignment

        arr = np.zeros(5, dtype=np.object_)

        arr[:] = a
        assert_equal(cnt(a), cnt0_a + 5)

        arr[:] = b
        assert_equal(cnt(a), cnt0_a)
        assert_equal(cnt(b), cnt0_b + 5)

        arr[:2] = c
        assert_equal(cnt(b), cnt0_b + 3)
        assert_equal(cnt(c), cnt0_c + 2)

        del arr

        # -- 1-d -> 2-d broadcast slice assignment

        arr = np.zeros((5, 2), dtype=np.object_)
        arr0 = np.zeros(2, dtype=np.object_)

        arr0[0] = a
        assert_(cnt(a) == cnt0_a + 1)
        arr0[1] = b
        assert_(cnt(b) == cnt0_b + 1)

        arr[:, :] = arr0
        assert_(cnt(a) == cnt0_a + 6)
        assert_(cnt(b) == cnt0_b + 6)

        arr[:, 0] = None
        assert_(cnt(a) == cnt0_a + 1)

        del arr, arr0

        # -- 2-d copying + flattening

        arr = np.zeros((5, 2), dtype=np.object_)

        arr[:, 0] = a
        arr[:, 1] = b
        assert_(cnt(a) == cnt0_a + 5)
        assert_(cnt(b) == cnt0_b + 5)

        arr2 = arr.copy()
        assert_(cnt(a) == cnt0_a + 10)
        assert_(cnt(b) == cnt0_b + 10)

        arr2 = arr[:, 0].copy()
        assert_(cnt(a) == cnt0_a + 10)
        assert_(cnt(b) == cnt0_b + 5)

        arr2 = arr.flatten()
        assert_(cnt(a) == cnt0_a + 10)
        assert_(cnt(b) == cnt0_b + 10)

        del arr, arr2

        # -- concatenate, repeat, take, choose

        arr1 = np.zeros((5, 1), dtype=np.object_)
        arr2 = np.zeros((5, 1), dtype=np.object_)

        arr1[...] = a
        arr2[...] = b
        assert_(cnt(a) == cnt0_a + 5)
        assert_(cnt(b) == cnt0_b + 5)

        tmp = np.concatenate((arr1, arr2))
        assert_(cnt(a) == cnt0_a + 5 + 5)
        assert_(cnt(b) == cnt0_b + 5 + 5)

        tmp = arr1.repeat(3, axis=0)
        assert_(cnt(a) == cnt0_a + 5 + 3*5)

        tmp = arr1.take([1, 2, 3], axis=0)
        assert_(cnt(a) == cnt0_a + 5 + 3)

        x = np.array([[0], [1], [0], [1], [1]], int)
        tmp = x.choose(arr1, arr2)
        assert_(cnt(a) == cnt0_a + 5 + 2)
        assert_(cnt(b) == cnt0_b + 5 + 3)

        del tmp  # Avoid pyflakes unused variable warning

    def test_mem_custom_float_to_array(self):
        # Ticket 702
        class MyFloat:
            def __float__(self):
                return 1.0

        tmp = np.atleast_1d([MyFloat()])
        tmp.astype(float)  # Should succeed

    def test_object_array_refcount_self_assign(self):
        # Ticket #711
        class VictimObject:
            deleted = False

            def __del__(self):
                self.deleted = True

        d = VictimObject()
        arr = np.zeros(5, dtype=np.object_)
        arr[:] = d
        del d
        arr[:] = arr  # refcount of 'd' might hit zero here
        assert_(not arr[0].deleted)
        arr[:] = arr  # trying to induce a segfault by doing it again...
        assert_(not arr[0].deleted)

    def test_mem_fromiter_invalid_dtype_string(self):
        x = [1, 2, 3]
        assert_raises(ValueError,
                              np.fromiter, [xi for xi in x], dtype='S')

    def test_reduce_big_object_array(self):
        # Ticket #713
        oldsize = np.setbufsize(10*16)
        a = np.array([None]*161, object)
        assert_(not np.any(a))
        np.setbufsize(oldsize)

    def test_mem_0d_array_index(self):
        # Ticket #714
        np.zeros(10)[np.array(0)]

    def test_nonnative_endian_fill(self):
        # Non-native endian arrays were incorrectly filled with scalars
        # before r5034.
        if sys.byteorder == 'little':
            dtype = np.dtype('>i4')
        else:
            dtype = np.dtype('<i4')
        x = np.empty([1], dtype=dtype)
        x.fill(1)
        assert_equal(x, np.array([1], dtype=dtype))

    def test_dot_alignment_sse2(self):
        # Test for ticket #551, changeset r5140
        x = np.zeros((30, 40))
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            y = pickle.loads(pickle.dumps(x, protocol=proto))
            # y is now typically not aligned on a 8-byte boundary
            z = np.ones((1, y.shape[0]))
            # This shouldn't cause a segmentation fault:
            np.dot(z, y)

    def test_astype_copy(self):
        # Ticket #788, changeset r5155
        # The test data file was generated by scipy.io.savemat.
        # The dtype is float64, but the isbuiltin attribute is 0.
        data_dir = path.join(path.dirname(__file__), 'data')
        filename = path.join(data_dir, "astype_copy.pkl")
        with open(filename, 'rb') as f:
            xp = pickle.load(f, encoding='latin1')
        xpd = xp.astype(np.float64)
        assert_((xp.__array_interface__['data'][0] !=
                xpd.__array_interface__['data'][0]))

    def test_compress_small_type(self):
        # Ticket #789, changeset 5217.
        # compress with out argument segfaulted if cannot cast safely
        import numpy as np
        a = np.array([[1, 2], [3, 4]])
        b = np.zeros((2, 1), dtype=np.single)
        try:
            a.compress([True, False], axis=1, out=b)
            raise AssertionError("compress with an out which cannot be "
                                 "safely casted should not return "
                                 "successfully")
        except TypeError:
            pass

    def test_attributes(self):
        # Ticket #791
        class TestArray(np.ndarray):
            def __new__(cls, data, info):
                result = np.array(data)
                result = result.view(cls)
                result.info = info
                return result

            def __array_finalize__(self, obj):
                self.info = getattr(obj, 'info', '')

        dat = TestArray([[1, 2, 3, 4], [5, 6, 7, 8]], 'jubba')
        assert_(dat.info == 'jubba')
        dat.resize((4, 2))
        assert_(dat.info == 'jubba')
        dat.sort()
        assert_(dat.info == 'jubba')
        dat.fill(2)
        assert_(dat.info == 'jubba')
        dat.put([2, 3, 4], [6, 3, 4])
        assert_(dat.info == 'jubba')
        dat.setfield(4, np.int32, 0)
        assert_(dat.info == 'jubba')
        dat.setflags()
        assert_(dat.info == 'jubba')
        assert_(dat.all(1).info == 'jubba')
        assert_(dat.any(1).info == 'jubba')
        assert_(dat.argmax(1).info == 'jubba')
        assert_(dat.argmin(1).info == 'jubba')
        assert_(dat.argsort(1).info == 'jubba')
        assert_(dat.astype(TestArray).info == 'jubba')
        assert_(dat.byteswap().info == 'jubba')
        assert_(dat.clip(2, 7).info == 'jubba')
        assert_(dat.compress([0, 1, 1]).info == 'jubba')
        assert_(dat.conj().info == 'jubba')
        assert_(dat.conjugate().info == 'jubba')
        assert_(dat.copy().info == 'jubba')
        dat2 = TestArray([2, 3, 1, 0], 'jubba')
        choices = [[0, 1, 2, 3], [10, 11, 12, 13],
                   [20, 21, 22, 23], [30, 31, 32, 33]]
        assert_(dat2.choose(choices).info == 'jubba')
        assert_(dat.cumprod(1).info == 'jubba')
        assert_(dat.cumsum(1).info == 'jubba')
        assert_(dat.diagonal().info == 'jubba')
        assert_(dat.flatten().info == 'jubba')
        assert_(dat.getfield(np.int32, 0).info == 'jubba')
        assert_(dat.imag.info == 'jubba')
        assert_(dat.max(1).info == 'jubba')
        assert_(dat.mean(1).info == 'jubba')
        assert_(dat.min(1).info == 'jubba')
        assert_(dat.newbyteorder().info == 'jubba')
        assert_(dat.prod(1).info == 'jubba')
        assert_(dat.ptp(1).info == 'jubba')
        assert_(dat.ravel().info == 'jubba')
        assert_(dat.real.info == 'jubba')
        assert_(dat.repeat(2).info == 'jubba')
        assert_(dat.reshape((2, 4)).info == 'jubba')
        assert_(dat.round().info == 'jubba')
        assert_(dat.squeeze().info == 'jubba')
        assert_(dat.std(1).info == 'jubba')
        assert_(dat.sum(1).info == 'jubba')
        assert_(dat.swapaxes(0, 1).info == 'jubba')
        assert_(dat.take([2, 3, 5]).info == 'jubba')
        assert_(dat.transpose().info == 'jubba')
        assert_(dat.T.info == 'jubba')
        assert_(dat.var(1).info == 'jubba')
        assert_(dat.view(TestArray).info == 'jubba')
        # These methods do not preserve subclasses
        assert_(type(dat.nonzero()[0]) is np.ndarray)
        assert_(type(dat.nonzero()[1]) is np.ndarray)

    def test_recarray_tolist(self):
        # Ticket #793, changeset r5215
        # Comparisons fail for NaN, so we can't use random memory
        # for the test.
        buf = np.zeros(40, dtype=np.int8)
        a = np.recarray(2, formats="i4,f8,f8", names="id,x,y", buf=buf)
        b = a.tolist()
        assert_( a[0].tolist() == b[0])
        assert_( a[1].tolist() == b[1])

    def test_nonscalar_item_method(self):
        # Make sure that .item() fails graciously when it should
        a = np.arange(5)
        assert_raises(ValueError, a.item)

    def test_char_array_creation(self):
        a = np.array('123', dtype='c')
        b = np.array([b'1', b'2', b'3'])
        assert_equal(a, b)

    def test_unaligned_unicode_access(self):
        # Ticket #825
        for i in range(1, 9):
            msg = 'unicode offset: %d chars' % i
            t = np.dtype([('a', 'S%d' % i), ('b', 'U2')])
            x = np.array([(b'a', 'b')], dtype=t)
            assert_equal(str(x), "[(b'a', 'b')]", err_msg=msg)

    def test_sign_for_complex_nan(self):
        # Ticket 794.
        with np.errstate(invalid='ignore'):
            C = np.array([-np.inf, -2+1j, 0, 2-1j, np.inf, np.nan])
            have = np.sign(C)
            want = np.array([-1+0j, -1+0j, 0+0j, 1+0j, 1+0j, np.nan])
            assert_equal(have, want)

    def test_for_equal_names(self):
        # Ticket #674
        dt = np.dtype([('foo', float), ('bar', float)])
        a = np.zeros(10, dt)
        b = list(a.dtype.names)
        b[0] = "notfoo"
        a.dtype.names = b
        assert_(a.dtype.names[0] == "notfoo")
        assert_(a.dtype.names[1] == "bar")

    def test_for_object_scalar_creation(self):
        # Ticket #816
        a = np.object_()
        b = np.object_(3)
        b2 = np.object_(3.0)
        c = np.object_([4, 5])
        d = np.object_([None, {}, []])
        assert_(a is None)
        assert_(type(b) is int)
        assert_(type(b2) is float)
        assert_(type(c) is np.ndarray)
        assert_(c.dtype == object)
        assert_(d.dtype == object)

    def test_array_resize_method_system_error(self):
        # Ticket #840 - order should be an invalid keyword.
        x = np.array([[0, 1], [2, 3]])
        assert_raises(TypeError, x.resize, (2, 2), order='C')

    def test_for_zero_length_in_choose(self):
        "Ticket #882"
        a = np.array(1)
        assert_raises(ValueError, lambda x: x.choose([]), a)

    def test_array_ndmin_overflow(self):
        "Ticket #947."
        assert_raises(ValueError, lambda: np.array([1], ndmin=33))

    def test_void_scalar_with_titles(self):
        # No ticket
        data = [('john', 4), ('mary', 5)]
        dtype1 = [(('source:yy', 'name'), 'O'), (('source:xx', 'id'), int)]
        arr = np.array(data, dtype=dtype1)
        assert_(arr[0][0] == 'john')
        assert_(arr[0][1] == 4)

    def test_void_scalar_constructor(self):
        #Issue #1550

        #Create test string data, construct void scalar from data and assert
        #that void scalar contains original data.
        test_string = np.array("test")
        test_string_void_scalar = np.core.multiarray.scalar(
            np.dtype(("V", test_string.dtype.itemsize)), test_string.tobytes())

        assert_(test_string_void_scalar.view(test_string.dtype) == test_string)

        #Create record scalar, construct from data and assert that
        #reconstructed scalar is correct.
        test_record = np.ones((), "i,i")
        test_record_void_scalar = np.core.multiarray.scalar(
            test_record.dtype, test_record.tobytes())

        assert_(test_record_void_scalar == test_record)

        # Test pickle and unpickle of void and record scalars
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            assert_(pickle.loads(
                pickle.dumps(test_string, protocol=proto)) == test_string)
            assert_(pickle.loads(
                pickle.dumps(test_record, protocol=proto)) == test_record)

    @_no_tracing
    def test_blasdot_uninitialized_memory(self):
        # Ticket #950
        for m in [0, 1, 2]:
            for n in [0, 1, 2]:
                for k in range(3):
                    # Try to ensure that x->data contains non-zero floats
                    x = np.array([123456789e199], dtype=np.float64)
                    if IS_PYPY:
                        x.resize((m, 0), refcheck=False)
                    else:
                        x.resize((m, 0))
                    y = np.array([123456789e199], dtype=np.float64)
                    if IS_PYPY:
                        y.resize((0, n), refcheck=False)
                    else:
                        y.resize((0, n))

                    # `dot` should just return zero (m, n) matrix
                    z = np.dot(x, y)
                    assert_(np.all(z == 0))
                    assert_(z.shape == (m, n))

    def test_zeros(self):
        # Regression test for #1061.
        # Set a size which cannot fit into a 64 bits signed integer
        sz = 2 ** 64
        with assert_raises_regex(ValueError,
                                 'Maximum allowed dimension exceeded'):
            np.empty(sz)

    def test_huge_arange(self):
        # Regression test for #1062.
        # Set a size which cannot fit into a 64 bits signed integer
        sz = 2 ** 64
        with assert_raises_regex(ValueError,
                                 'Maximum allowed size exceeded'):
            np.arange(sz)
            assert_(np.size == sz)

    def test_fromiter_bytes(self):
        # Ticket #1058
        a = np.fromiter(list(range(10)), dtype='b')
        b = np.fromiter(list(range(10)), dtype='B')
        assert_(np.alltrue(a == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
        assert_(np.alltrue(b == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))

    def test_array_from_sequence_scalar_array(self):
        # Ticket #1078: segfaults when creating an array with a sequence of
        # 0d arrays.
        a = np.array((np.ones(2), np.array(2)), dtype=object)
        assert_equal(a.shape, (2,))
        assert_equal(a.dtype, np.dtype(object))
        assert_equal(a[0], np.ones(2))
        assert_equal(a[1], np.array(2))

        a = np.array(((1,), np.array(1)), dtype=object)
        assert_equal(a.shape, (2,))
        assert_equal(a.dtype, np.dtype(object))
        assert_equal(a[0], (1,))
        assert_equal(a[1], np.array(1))

    def test_array_from_sequence_scalar_array2(self):
        # Ticket #1081: weird array with strange input...
        t = np.array([np.array([]), np.array(0, object)], dtype=object)
        assert_equal(t.shape, (2,))
        assert_equal(t.dtype, np.dtype(object))

    def test_array_too_big(self):
        # Ticket #1080.
        assert_raises(ValueError, np.zeros, [975]*7, np.int8)
        assert_raises(ValueError, np.zeros, [26244]*5, np.int8)

    def test_dtype_keyerrors_(self):
        # Ticket #1106.
        dt = np.dtype([('f1', np.uint)])
        assert_raises(KeyError, dt.__getitem__, "f2")
        assert_raises(IndexError, dt.__getitem__, 1)
        assert_raises(TypeError, dt.__getitem__, 0.0)

    def test_lexsort_buffer_length(self):
        # Ticket #1217, don't segfault.
        a = np.ones(100, dtype=np.int8)
        b = np.ones(100, dtype=np.int32)
        i = np.lexsort((a[::-1], b))
        assert_equal(i, np.arange(100, dtype=int))

    def test_object_array_to_fixed_string(self):
        # Ticket #1235.
        a = np.array(['abcdefgh', 'ijklmnop'], dtype=np.object_)
        b = np.array(a, dtype=(np.str_, 8))
        assert_equal(a, b)
        c = np.array(a, dtype=(np.str_, 5))
        assert_equal(c, np.array(['abcde', 'ijklm']))
        d = np.array(a, dtype=(np.str_, 12))
        assert_equal(a, d)
        e = np.empty((2, ), dtype=(np.str_, 8))
        e[:] = a[:]
        assert_equal(a, e)

    def test_unicode_to_string_cast(self):
        # Ticket #1240.
        a = np.array([['abc', '\u03a3'],
                      ['asdf', 'erw']],
                     dtype='U')
        assert_raises(UnicodeEncodeError, np.array, a, 'S4')

    def test_unicode_to_string_cast_error(self):
        # gh-15790
        a = np.array(['\x80'] * 129, dtype='U3')
        assert_raises(UnicodeEncodeError, np.array, a, 'S')
        b = a.reshape(3, 43)[:-1, :-1]
        assert_raises(UnicodeEncodeError, np.array, b, 'S')

    def test_mixed_string_byte_array_creation(self):
        a = np.array(['1234', b'123'])
        assert_(a.itemsize == 16)
        a = np.array([b'123', '1234'])
        assert_(a.itemsize == 16)
        a = np.array(['1234', b'123', '12345'])
        assert_(a.itemsize == 20)
        a = np.array([b'123', '1234', b'12345'])
        assert_(a.itemsize == 20)
        a = np.array([b'123', '1234', b'1234'])
        assert_(a.itemsize == 16)

    def test_misaligned_objects_segfault(self):
        # Ticket #1198 and #1267
        a1 = np.zeros((10,), dtype='O,c')
        a2 = np.array(['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'], 'S10')
        a1['f0'] = a2
        repr(a1)
        np.argmax(a1['f0'])
        a1['f0'][1] = "FOO"
        a1['f0'] = "FOO"
        np.array(a1['f0'], dtype='S')
        np.nonzero(a1['f0'])
        a1.sort()
        copy.deepcopy(a1)

    def test_misaligned_scalars_segfault(self):
        # Ticket #1267
        s1 = np.array(('a', 'Foo'), dtype='c,O')
        s2 = np.array(('b', 'Bar'), dtype='c,O')
        s1['f1'] = s2['f1']
        s1['f1'] = 'Baz'

    def test_misaligned_dot_product_objects(self):
        # Ticket #1267
        # This didn't require a fix, but it's worth testing anyway, because
        # it may fail if .dot stops enforcing the arrays to be BEHAVED
        a = np.array([[(1, 'a'), (0, 'a')], [(0, 'a'), (1, 'a')]], dtype='O,c')
        b = np.array([[(4, 'a'), (1, 'a')], [(2, 'a'), (2, 'a')]], dtype='O,c')
        np.dot(a['f0'], b['f0'])

    def test_byteswap_complex_scalar(self):
        # Ticket #1259 and gh-441
        for dtype in [np.dtype('<'+t) for t in np.typecodes['Complex']]:
            z = np.array([2.2-1.1j], dtype)
            x = z[0]  # always native-endian
            y = x.byteswap()
            if x.dtype.byteorder == z.dtype.byteorder:
                # little-endian machine
                assert_equal(x, np.frombuffer(y.tobytes(), dtype=dtype.newbyteorder()))
            else:
                # big-endian machine
                assert_equal(x, np.frombuffer(y.tobytes(), dtype=dtype))
            # double check real and imaginary parts:
            assert_equal(x.real, y.real.byteswap())
            assert_equal(x.imag, y.imag.byteswap())

    def test_structured_arrays_with_objects1(self):
        # Ticket #1299
        stra = 'aaaa'
        strb = 'bbbb'
        x = np.array([[(0, stra), (1, strb)]], 'i8,O')
        x[x.nonzero()] = x.ravel()[:1]
        assert_(x[0, 1] == x[0, 0])

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_structured_arrays_with_objects2(self):
        # Ticket #1299 second test
        stra = 'aaaa'
        strb = 'bbbb'
        numb = sys.getrefcount(strb)
        numa = sys.getrefcount(stra)
        x = np.array([[(0, stra), (1, strb)]], 'i8,O')
        x[x.nonzero()] = x.ravel()[:1]
        assert_(sys.getrefcount(strb) == numb)
        assert_(sys.getrefcount(stra) == numa + 2)

    def test_duplicate_title_and_name(self):
        # Ticket #1254
        dtspec = [(('a', 'a'), 'i'), ('b', 'i')]
        assert_raises(ValueError, np.dtype, dtspec)

    def test_signed_integer_division_overflow(self):
        # Ticket #1317.
        def test_type(t):
            min = np.array([np.iinfo(t).min])
            min //= -1

        with np.errstate(over="ignore"):
            for t in (np.int8, np.int16, np.int32, np.int64, int):
                test_type(t)

    def test_buffer_hashlib(self):
        from hashlib import sha256

        x = np.array([1, 2, 3], dtype=np.dtype('<i4'))
        assert_equal(sha256(x).hexdigest(), '4636993d3e1da4e9d6b8f87b79e8f7c6d018580d52661950eabc3845c5897a4d')

    def test_0d_string_scalar(self):
        # Bug #1436; the following should succeed
        np.asarray('x', '>c')

    def test_log1p_compiler_shenanigans(self):
        # Check if log1p is behaving on 32 bit intel systems.
        assert_(np.isfinite(np.log1p(np.exp2(-53))))

    def test_fromiter_comparison(self):
        a = np.fromiter(list(range(10)), dtype='b')
        b = np.fromiter(list(range(10)), dtype='B')
        assert_(np.alltrue(a == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))
        assert_(np.alltrue(b == np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])))

    def test_fromstring_crash(self):
        # Ticket #1345: the following should not cause a crash
        with assert_warns(DeprecationWarning):
            np.fromstring(b'aa, aa, 1.0', sep=',')

    def test_ticket_1539(self):
        dtypes = [x for x in np.sctypeDict.values()
                  if (issubclass(x, np.number)
                      and not issubclass(x, np.timedelta64))]
        a = np.array([], np.bool_)  # not x[0] because it is unordered
        failures = []

        for x in dtypes:
            b = a.astype(x)
            for y in dtypes:
                c = a.astype(y)
                try:
                    np.dot(b, c)
                except TypeError:
                    failures.append((x, y))
        if failures:
            raise AssertionError("Failures: %r" % failures)

    def test_ticket_1538(self):
        x = np.finfo(np.float32)
        for name in 'eps epsneg max min resolution tiny'.split():
            assert_equal(type(getattr(x, name)), np.float32,
                         err_msg=name)

    def test_ticket_1434(self):
        # Check that the out= argument in var and std has an effect
        data = np.array(((1, 2, 3), (4, 5, 6), (7, 8, 9)))
        out = np.zeros((3,))

        ret = data.var(axis=1, out=out)
        assert_(ret is out)
        assert_array_equal(ret, data.var(axis=1))

        ret = data.std(axis=1, out=out)
        assert_(ret is out)
        assert_array_equal(ret, data.std(axis=1))

    def test_complex_nan_maximum(self):
        cnan = complex(0, np.nan)
        assert_equal(np.maximum(1, cnan), cnan)

    def test_subclass_int_tuple_assignment(self):
        # ticket #1563
        class Subclass(np.ndarray):
            def __new__(cls, i):
                return np.ones((i,)).view(cls)

        x = Subclass(5)
        x[(0,)] = 2  # shouldn't raise an exception
        assert_equal(x[0], 2)

    def test_ufunc_no_unnecessary_views(self):
        # ticket #1548
        class Subclass(np.ndarray):
            pass
        x = np.array([1, 2, 3]).view(Subclass)
        y = np.add(x, x, x)
        assert_equal(id(x), id(y))

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_take_refcount(self):
        # ticket #939
        a = np.arange(16, dtype=float)
        a.shape = (4, 4)
        lut = np.ones((5 + 3, 4), float)
        rgba = np.empty(shape=a.shape + (4,), dtype=lut.dtype)
        c1 = sys.getrefcount(rgba)
        try:
            lut.take(a, axis=0, mode='clip', out=rgba)
        except TypeError:
            pass
        c2 = sys.getrefcount(rgba)
        assert_equal(c1, c2)

    def test_fromfile_tofile_seeks(self):
        # On Python 3, tofile/fromfile used to get (#1610) the Python
        # file handle out of sync
        f0 = tempfile.NamedTemporaryFile()
        f = f0.file
        f.write(np.arange(255, dtype='u1').tobytes())

        f.seek(20)
        ret = np.fromfile(f, count=4, dtype='u1')
        assert_equal(ret, np.array([20, 21, 22, 23], dtype='u1'))
        assert_equal(f.tell(), 24)

        f.seek(40)
        np.array([1, 2, 3], dtype='u1').tofile(f)
        assert_equal(f.tell(), 43)

        f.seek(40)
        data = f.read(3)
        assert_equal(data, b"\x01\x02\x03")

        f.seek(80)
        f.read(4)
        data = np.fromfile(f, dtype='u1', count=4)
        assert_equal(data, np.array([84, 85, 86, 87], dtype='u1'))

        f.close()

    def test_complex_scalar_warning(self):
        for tp in [np.csingle, np.cdouble, np.clongdouble]:
            x = tp(1+2j)
            assert_warns(np.ComplexWarning, float, x)
            with suppress_warnings() as sup:
                sup.filter(np.ComplexWarning)
                assert_equal(float(x), float(x.real))

    def test_complex_scalar_complex_cast(self):
        for tp in [np.csingle, np.cdouble, np.clongdouble]:
            x = tp(1+2j)
            assert_equal(complex(x), 1+2j)

    def test_complex_boolean_cast(self):
        # Ticket #2218
        for tp in [np.csingle, np.cdouble, np.clongdouble]:
            x = np.array([0, 0+0.5j, 0.5+0j], dtype=tp)
            assert_equal(x.astype(bool), np.array([0, 1, 1], dtype=bool))
            assert_(np.any(x))
            assert_(np.all(x[1:]))

    def test_uint_int_conversion(self):
        x = 2**64 - 1
        assert_equal(int(np.uint64(x)), x)

    def test_duplicate_field_names_assign(self):
        ra = np.fromiter(((i*3, i*2) for i in range(10)), dtype='i8,f8')
        ra.dtype.names = ('f1', 'f2')
        repr(ra)  # should not cause a segmentation fault
        assert_raises(ValueError, setattr, ra.dtype, 'names', ('f1', 'f1'))

    def test_eq_string_and_object_array(self):
        # From e-mail thread "__eq__ with str and object" (Keith Goodman)
        a1 = np.array(['a', 'b'], dtype=object)
        a2 = np.array(['a', 'c'])
        assert_array_equal(a1 == a2, [True, False])
        assert_array_equal(a2 == a1, [True, False])

    def test_nonzero_byteswap(self):
        a = np.array([0x80000000, 0x00000080, 0], dtype=np.uint32)
        a.dtype = np.float32
        assert_equal(a.nonzero()[0], [1])
        a = a.byteswap().newbyteorder()
        assert_equal(a.nonzero()[0], [1])  # [0] if nonzero() ignores swap

    def test_find_common_type_boolean(self):
        # Ticket #1695
        assert_(np.find_common_type([], ['?', '?']) == '?')

    def test_empty_mul(self):
        a = np.array([1.])
        a[1:1] *= 2
        assert_equal(a, [1.])

    def test_array_side_effect(self):
        # The second use of itemsize was throwing an exception because in
        # ctors.c, discover_itemsize was calling PyObject_Length without
        # checking the return code.  This failed to get the length of the
        # number 2, and the exception hung around until something checked
        # PyErr_Occurred() and returned an error.
        assert_equal(np.dtype('S10').itemsize, 10)
        np.array([['abc', 2], ['long   ', '0123456789']], dtype=np.string_)
        assert_equal(np.dtype('S10').itemsize, 10)

    def test_any_float(self):
        # all and any for floats
        a = np.array([0.1, 0.9])
        assert_(np.any(a))
        assert_(np.all(a))

    def test_large_float_sum(self):
        a = np.arange(10000, dtype='f')
        assert_equal(a.sum(dtype='d'), a.astype('d').sum())

    def test_ufunc_casting_out(self):
        a = np.array(1.0, dtype=np.float32)
        b = np.array(1.0, dtype=np.float64)
        c = np.array(1.0, dtype=np.float32)
        np.add(a, b, out=c)
        assert_equal(c, 2.0)

    def test_array_scalar_contiguous(self):
        # Array scalars are both C and Fortran contiguous
        assert_(np.array(1.0).flags.c_contiguous)
        assert_(np.array(1.0).flags.f_contiguous)
        assert_(np.array(np.float32(1.0)).flags.c_contiguous)
        assert_(np.array(np.float32(1.0)).flags.f_contiguous)

    def test_squeeze_contiguous(self):
        # Similar to GitHub issue #387
        a = np.zeros((1, 2)).squeeze()
        b = np.zeros((2, 2, 2), order='F')[:, :, ::2].squeeze()
        assert_(a.flags.c_contiguous)
        assert_(a.flags.f_contiguous)
        assert_(b.flags.f_contiguous)

    def test_squeeze_axis_handling(self):
        # Issue #10779
        # Ensure proper handling of objects
        # that don't support axis specification
        # when squeezing

        class OldSqueeze(np.ndarray):

            def __new__(cls,
                        input_array):
                obj = np.asarray(input_array).view(cls)
                return obj

            # it is perfectly reasonable that prior
            # to numpy version 1.7.0 a subclass of ndarray
            # might have been created that did not expect
            # squeeze to have an axis argument
            # NOTE: this example is somewhat artificial;
            # it is designed to simulate an old API
            # expectation to guard against regression
            def squeeze(self):
                return super().squeeze()

        oldsqueeze = OldSqueeze(np.array([[1],[2],[3]]))

        # if no axis argument is specified the old API
        # expectation should give the correct result
        assert_equal(np.squeeze(oldsqueeze),
                     np.array([1,2,3]))

        # likewise, axis=None should work perfectly well
        # with the old API expectation
        assert_equal(np.squeeze(oldsqueeze, axis=None),
                     np.array([1,2,3]))

        # however, specification of any particular axis
        # should raise a TypeError in the context of the
        # old API specification, even when using a valid
        # axis specification like 1 for this array
        with assert_raises(TypeError):
            # this would silently succeed for array
            # subclasses / objects that did not support
            # squeeze axis argument handling before fixing
            # Issue #10779
            np.squeeze(oldsqueeze, axis=1)

        # check for the same behavior when using an invalid
        # axis specification -- in this case axis=0 does not
        # have size 1, but the priority should be to raise
        # a TypeError for the axis argument and NOT a
        # ValueError for squeezing a non-empty dimension
        with assert_raises(TypeError):
            np.squeeze(oldsqueeze, axis=0)

        # the new API knows how to handle the axis
        # argument and will return a ValueError if
        # attempting to squeeze an axis that is not
        # of length 1
        with assert_raises(ValueError):
            np.squeeze(np.array([[1],[2],[3]]), axis=0)

    def test_reduce_contiguous(self):
        # GitHub issue #387
        a = np.add.reduce(np.zeros((2, 1, 2)), (0, 1))
        b = np.add.reduce(np.zeros((2, 1, 2)), 1)
        assert_(a.flags.c_contiguous)
        assert_(a.flags.f_contiguous)
        assert_(b.flags.c_contiguous)

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    def test_object_array_self_reference(self):
        # Object arrays with references to themselves can cause problems
        a = np.array(0, dtype=object)
        a[()] = a
        assert_raises(RecursionError, int, a)
        assert_raises(RecursionError, float, a)
        a[()] = None

    @pytest.mark.skipif(IS_PYSTON, reason="Pyston disables recursion checking")
    def test_object_array_circular_reference(self):
        # Test the same for a circular reference.
        a = np.array(0, dtype=object)
        b = np.array(0, dtype=object)
        a[()] = b
        b[()] = a
        assert_raises(RecursionError, int, a)
        # NumPy has no tp_traverse currently, so circular references
        # cannot be detected. So resolve it:
        a[()] = None

        # This was causing a to become like the above
        a = np.array(0, dtype=object)
        a[...] += 1
        assert_equal(a, 1)

    def test_object_array_nested(self):
        # but is fine with a reference to a different array
        a = np.array(0, dtype=object)
        b = np.array(0, dtype=object)
        a[()] = b
        assert_equal(int(a), int(0))
        assert_equal(float(a), float(0))

    def test_object_array_self_copy(self):
        # An object array being copied into itself DECREF'ed before INCREF'ing
        # causing segmentation faults (gh-3787)
        a = np.array(object(), dtype=object)
        np.copyto(a, a)
        if HAS_REFCOUNT:
            assert_(sys.getrefcount(a[()]) == 2)
        a[()].__class__  # will segfault if object was deleted

    def test_zerosize_accumulate(self):
        "Ticket #1733"
        x = np.array([[42, 0]], dtype=np.uint32)
        assert_equal(np.add.accumulate(x[:-1, 0]), [])

    def test_objectarray_setfield(self):
        # Setfield should not overwrite Object fields with non-Object data
        x = np.array([1, 2, 3], dtype=object)
        assert_raises(TypeError, x.setfield, 4, np.int32, 0)

    def test_setting_rank0_string(self):
        "Ticket #1736"
        s1 = b"hello1"
        s2 = b"hello2"
        a = np.zeros((), dtype="S10")
        a[()] = s1
        assert_equal(a, np.array(s1))
        a[()] = np.array(s2)
        assert_equal(a, np.array(s2))

        a = np.zeros((), dtype='f4')
        a[()] = 3
        assert_equal(a, np.array(3))
        a[()] = np.array(4)
        assert_equal(a, np.array(4))

    def test_string_astype(self):
        "Ticket #1748"
        s1 = b'black'
        s2 = b'white'
        s3 = b'other'
        a = np.array([[s1], [s2], [s3]])
        assert_equal(a.dtype, np.dtype('S5'))
        b = a.astype(np.dtype('S0'))
        assert_equal(b.dtype, np.dtype('S5'))

    def test_ticket_1756(self):
        # Ticket #1756
        s = b'0123456789abcdef'
        a = np.array([s]*5)
        for i in range(1, 17):
            a1 = np.array(a, "|S%d" % i)
            a2 = np.array([s[:i]]*5)
            assert_equal(a1, a2)

    def test_fields_strides(self):
        "gh-2355"
        r = np.frombuffer(b'abcdefghijklmnop'*4*3, dtype='i4,(2,3)u2')
        assert_equal(r[0:3:2]['f1'], r['f1'][0:3:2])
        assert_equal(r[0:3:2]['f1'][0], r[0:3:2][0]['f1'])
        assert_equal(r[0:3:2]['f1'][0][()], r[0:3:2][0]['f1'][()])
        assert_equal(r[0:3:2]['f1'][0].strides, r[0:3:2][0]['f1'].strides)

    def test_alignment_update(self):
        # Check that alignment flag is updated on stride setting
        a = np.arange(10)
        assert_(a.flags.aligned)
        a.strides = 3
        assert_(not a.flags.aligned)

    def test_ticket_1770(self):
        "Should not segfault on python 3k"
        import numpy as np
        try:
            a = np.zeros((1,), dtype=[('f1', 'f')])
            a['f1'] = 1
            a['f2'] = 1
        except ValueError:
            pass
        except Exception:
            raise AssertionError

    def test_ticket_1608(self):
        "x.flat shouldn't modify data"
        x = np.array([[1, 2], [3, 4]]).T
        np.array(x.flat)
        assert_equal(x, [[1, 3], [2, 4]])

    def test_pickle_string_overwrite(self):
        import re

        data = np.array([1], dtype='b')
        blob = pickle.dumps(data, protocol=1)
        data = pickle.loads(blob)

        # Check that loads does not clobber interned strings
        s = re.sub("a(.)", "\x01\\1", "a_")
        assert_equal(s[0], "\x01")
        data[0] = 0x6a
        s = re.sub("a(.)", "\x01\\1", "a_")
        assert_equal(s[0], "\x01")

    def test_pickle_bytes_overwrite(self):
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            data = np.array([1], dtype='b')
            data = pickle.loads(pickle.dumps(data, protocol=proto))
            data[0] = 0x7d
            bytestring = "\x01  ".encode('ascii')
            assert_equal(bytestring[0:1], '\x01'.encode('ascii'))

    def test_pickle_py2_array_latin1_hack(self):
        # Check that unpickling hacks in Py3 that support
        # encoding='latin1' work correctly.

        # Python2 output for pickle.dumps(numpy.array([129], dtype='b'))
        data = (b"cnumpy.core.multiarray\n_reconstruct\np0\n(cnumpy\nndarray\np1\n(I0\n"
                b"tp2\nS'b'\np3\ntp4\nRp5\n(I1\n(I1\ntp6\ncnumpy\ndtype\np7\n(S'i1'\np8\n"
                b"I0\nI1\ntp9\nRp10\n(I3\nS'|'\np11\nNNNI-1\nI-1\nI0\ntp12\nbI00\nS'\\x81'\n"
                b"p13\ntp14\nb.")
        # This should work:
        result = pickle.loads(data, encoding='latin1')
        assert_array_equal(result, np.array([129]).astype('b'))
        # Should not segfault:
        assert_raises(Exception, pickle.loads, data, encoding='koi8-r')

    def test_pickle_py2_scalar_latin1_hack(self):
        # Check that scalar unpickling hack in Py3 that supports
        # encoding='latin1' work correctly.

        # Python2 output for pickle.dumps(...)
        datas = [
            # (original, python2_pickle, koi8r_validity)
            (np.unicode_('\u6bd2'),
             (b"cnumpy.core.multiarray\nscalar\np0\n(cnumpy\ndtype\np1\n"
              b"(S'U1'\np2\nI0\nI1\ntp3\nRp4\n(I3\nS'<'\np5\nNNNI4\nI4\nI0\n"
              b"tp6\nbS'\\xd2k\\x00\\x00'\np7\ntp8\nRp9\n."),
             'invalid'),

            (np.float64(9e123),
             (b"cnumpy.core.multiarray\nscalar\np0\n(cnumpy\ndtype\np1\n(S'f8'\n"
              b"p2\nI0\nI1\ntp3\nRp4\n(I3\nS'<'\np5\nNNNI-1\nI-1\nI0\ntp6\n"
              b"bS'O\\x81\\xb7Z\\xaa:\\xabY'\np7\ntp8\nRp9\n."),
             'invalid'),

            (np.bytes_(b'\x9c'),  # different 8-bit code point in KOI8-R vs latin1
             (b"cnumpy.core.multiarray\nscalar\np0\n(cnumpy\ndtype\np1\n(S'S1'\np2\n"
              b"I0\nI1\ntp3\nRp4\n(I3\nS'|'\np5\nNNNI1\nI1\nI0\ntp6\nbS'\\x9c'\np7\n"
              b"tp8\nRp9\n."),
             'different'),
        ]
        for original, data, koi8r_validity in datas:
            result = pickle.loads(data, encoding='latin1')
            assert_equal(result, original)

            # Decoding under non-latin1 encoding (e.g.) KOI8-R can
            # produce bad results, but should not segfault.
            if koi8r_validity == 'different':
                # Unicode code points happen to lie within latin1,
                # but are different in koi8-r, resulting to silent
                # bogus results
                result = pickle.loads(data, encoding='koi8-r')
                assert_(result != original)
            elif koi8r_validity == 'invalid':
                # Unicode code points outside latin1, so results
                # to an encoding exception
                assert_raises(ValueError, pickle.loads, data, encoding='koi8-r')
            else:
                raise ValueError(koi8r_validity)

    def test_structured_type_to_object(self):
        a_rec = np.array([(0, 1), (3, 2)], dtype='i4,i8')
        a_obj = np.empty((2,), dtype=object)
        a_obj[0] = (0, 1)
        a_obj[1] = (3, 2)
        # astype records -> object
        assert_equal(a_rec.astype(object), a_obj)
        # '=' records -> object
        b = np.empty_like(a_obj)
        b[...] = a_rec
        assert_equal(b, a_obj)
        # '=' object -> records
        b = np.empty_like(a_rec)
        b[...] = a_obj
        assert_equal(b, a_rec)

    def test_assign_obj_listoflists(self):
        # Ticket # 1870
        # The inner list should get assigned to the object elements
        a = np.zeros(4, dtype=object)
        b = a.copy()
        a[0] = [1]
        a[1] = [2]
        a[2] = [3]
        a[3] = [4]
        b[...] = [[1], [2], [3], [4]]
        assert_equal(a, b)
        # The first dimension should get broadcast
        a = np.zeros((2, 2), dtype=object)
        a[...] = [[1, 2]]
        assert_equal(a, [[1, 2], [1, 2]])

    @pytest.mark.slow_pypy
    def test_memoryleak(self):
        # Ticket #1917 - ensure that array data doesn't leak
        for i in range(1000):
            # 100MB times 1000 would give 100GB of memory usage if it leaks
            a = np.empty((100000000,), dtype='i1')
            del a

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_ufunc_reduce_memoryleak(self):
        a = np.arange(6)
        acnt = sys.getrefcount(a)
        np.add.reduce(a)
        assert_equal(sys.getrefcount(a), acnt)

    def test_search_sorted_invalid_arguments(self):
        # Ticket #2021, should not segfault.
        x = np.arange(0, 4, dtype='datetime64[D]')
        assert_raises(TypeError, x.searchsorted, 1)

    def test_string_truncation(self):
        # Ticket #1990 - Data can be truncated in creation of an array from a
        # mixed sequence of numeric values and strings (gh-2583)
        for val in [True, 1234, 123.4, complex(1, 234)]:
            for tostr, dtype in [(asunicode, "U"), (asbytes, "S")]:
                b = np.array([val, tostr('xx')], dtype=dtype)
                assert_equal(tostr(b[0]), tostr(val))
                b = np.array([tostr('xx'), val], dtype=dtype)
                assert_equal(tostr(b[1]), tostr(val))

                # test also with longer strings
                b = np.array([val, tostr('xxxxxxxxxx')], dtype=dtype)
                assert_equal(tostr(b[0]), tostr(val))
                b = np.array([tostr('xxxxxxxxxx'), val], dtype=dtype)
                assert_equal(tostr(b[1]), tostr(val))

    def test_string_truncation_ucs2(self):
        # Ticket #2081. Python compiled with two byte unicode
        # can lead to truncation if itemsize is not properly
        # adjusted for NumPy's four byte unicode.
        a = np.array(['abcd'])
        assert_equal(a.dtype.itemsize, 16)

    def test_unique_stable(self):
        # Ticket #2063 must always choose stable sort for argsort to
        # get consistent results
        v = np.array(([0]*5 + [1]*6 + [2]*6)*4)
        res = np.unique(v, return_index=True)
        tgt = (np.array([0, 1, 2]), np.array([ 0,  5, 11]))
        assert_equal(res, tgt)

    def test_unicode_alloc_dealloc_match(self):
        # Ticket #1578, the mismatch only showed up when running
        # python-debug for python versions >= 2.7, and then as
        # a core dump and error message.
        a = np.array(['abc'], dtype=np.unicode_)[0]
        del a

    def test_refcount_error_in_clip(self):
        # Ticket #1588
        a = np.zeros((2,), dtype='>i2').clip(min=0)
        x = a + a
        # This used to segfault:
        y = str(x)
        # Check the final string:
        assert_(y == "[0 0]")

    def test_searchsorted_wrong_dtype(self):
        # Ticket #2189, it used to segfault, so we check that it raises the
        # proper exception.
        a = np.array([('a', 1)], dtype='S1, int')
        assert_raises(TypeError, np.searchsorted, a, 1.2)
        # Ticket #2066, similar problem:
        dtype = np.format_parser(['i4', 'i4'], [], [])
        a = np.recarray((2,), dtype)
        a[...] = [(1, 2), (3, 4)]
        assert_raises(TypeError, np.searchsorted, a, 1)

    def test_complex64_alignment(self):
        # Issue gh-2668 (trac 2076), segfault on sparc due to misalignment
        dtt = np.complex64
        arr = np.arange(10, dtype=dtt)
        # 2D array
        arr2 = np.reshape(arr, (2, 5))
        # Fortran write followed by (C or F) read caused bus error
        data_str = arr2.tobytes('F')
        data_back = np.ndarray(arr2.shape,
                              arr2.dtype,
                              buffer=data_str,
                              order='F')
        assert_array_equal(arr2, data_back)

    def test_structured_count_nonzero(self):
        arr = np.array([0, 1]).astype('i4, (2)i4')[:1]
        count = np.count_nonzero(arr)
        assert_equal(count, 0)

    def test_copymodule_preserves_f_contiguity(self):
        a = np.empty((2, 2), order='F')
        b = copy.copy(a)
        c = copy.deepcopy(a)
        assert_(b.flags.fortran)
        assert_(b.flags.f_contiguous)
        assert_(c.flags.fortran)
        assert_(c.flags.f_contiguous)

    def test_fortran_order_buffer(self):
        import numpy as np
        a = np.array([['Hello', 'Foob']], dtype='U5', order='F')
        arr = np.ndarray(shape=[1, 2, 5], dtype='U1', buffer=a)
        arr2 = np.array([[['H', 'e', 'l', 'l', 'o'],
                          ['F', 'o', 'o', 'b', '']]])
        assert_array_equal(arr, arr2)

    def test_assign_from_sequence_error(self):
        # Ticket #4024.
        arr = np.array([1, 2, 3])
        assert_raises(ValueError, arr.__setitem__, slice(None), [9, 9])
        arr.__setitem__(slice(None), [9])
        assert_equal(arr, [9, 9, 9])

    def test_format_on_flex_array_element(self):
        # Ticket #4369.
        dt = np.dtype([('date', '<M8[D]'), ('val', '<f8')])
        arr = np.array([('2000-01-01', 1)], dt)
        formatted = '{0}'.format(arr[0])
        assert_equal(formatted, str(arr[0]))

    def test_deepcopy_on_0d_array(self):
        # Ticket #3311.
        arr = np.array(3)
        arr_cp = copy.deepcopy(arr)

        assert_equal(arr, arr_cp)
        assert_equal(arr.shape, arr_cp.shape)
        assert_equal(int(arr), int(arr_cp))
        assert_(arr is not arr_cp)
        assert_(isinstance(arr_cp, type(arr)))

    def test_deepcopy_F_order_object_array(self):
        # Ticket #6456.
        a = {'a': 1}
        b = {'b': 2}
        arr = np.array([[a, b], [a, b]], order='F')
        arr_cp = copy.deepcopy(arr)

        assert_equal(arr, arr_cp)
        assert_(arr is not arr_cp)
        # Ensure that we have actually copied the item.
        assert_(arr[0, 1] is not arr_cp[1, 1])
        # Ensure we are allowed to have references to the same object.
        assert_(arr[0, 1] is arr[1, 1])
        # Check the references hold for the copied objects.
        assert_(arr_cp[0, 1] is arr_cp[1, 1])

    def test_deepcopy_empty_object_array(self):
        # Ticket #8536.
        # Deepcopy should succeed
        a = np.array([], dtype=object)
        b = copy.deepcopy(a)
        assert_(a.shape == b.shape)

    def test_bool_subscript_crash(self):
        # gh-4494
        c = np.rec.array([(1, 2, 3), (4, 5, 6)])
        masked = c[np.array([True, False])]
        base = masked.base
        del masked, c
        base.dtype

    def test_richcompare_crash(self):
        # gh-4613
        import operator as op

        # dummy class where __array__ throws exception
        class Foo:
            __array_priority__ = 1002

            def __array__(self, *args, **kwargs):
                raise Exception()

        rhs = Foo()
        lhs = np.array(1)
        for f in [op.lt, op.le, op.gt, op.ge]:
            assert_raises(TypeError, f, lhs, rhs)
        assert_(not op.eq(lhs, rhs))
        assert_(op.ne(lhs, rhs))

    def test_richcompare_scalar_and_subclass(self):
        # gh-4709
        class Foo(np.ndarray):
            def __eq__(self, other):
                return "OK"

        x = np.array([1, 2, 3]).view(Foo)
        assert_equal(10 == x, "OK")
        assert_equal(np.int32(10) == x, "OK")
        assert_equal(np.array([10]) == x, "OK")

    def test_pickle_empty_string(self):
        # gh-3926
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            test_string = np.string_('')
            assert_equal(pickle.loads(
                pickle.dumps(test_string, protocol=proto)), test_string)

    def test_frompyfunc_many_args(self):
        # gh-5672

        def passer(*args):
            pass

        assert_raises(ValueError, np.frompyfunc, passer, 32, 1)

    def test_repeat_broadcasting(self):
        # gh-5743
        a = np.arange(60).reshape(3, 4, 5)
        for axis in chain(range(-a.ndim, a.ndim), [None]):
            assert_equal(a.repeat(2, axis=axis), a.repeat([2], axis=axis))

    def test_frompyfunc_nout_0(self):
        # gh-2014

        def f(x):
            x[0], x[-1] = x[-1], x[0]

        uf = np.frompyfunc(f, 1, 0)
        a = np.array([[1, 2, 3], [4, 5], [6, 7, 8, 9]], dtype=object)
        assert_equal(uf(a), ())
        expected = np.array([[3, 2, 1], [5, 4], [9, 7, 8, 6]], dtype=object)
        assert_array_equal(a, expected)

    @pytest.mark.skipif(not HAS_REFCOUNT, reason="Python lacks refcounts")
    def test_leak_in_structured_dtype_comparison(self):
        # gh-6250
        recordtype = np.dtype([('a', np.float64),
                               ('b', np.int32),
                               ('d', (str, 5))])

        # Simple case
        a = np.zeros(2, dtype=recordtype)
        for i in range(100):
            a == a
        assert_(sys.getrefcount(a) < 10)

        # The case in the bug report.
        before = sys.getrefcount(a)
        u, v = a[0], a[1]
        u == v
        del u, v
        gc.collect()
        after = sys.getrefcount(a)
        assert_equal(before, after)

    def test_empty_percentile(self):
        # gh-6530 / gh-6553
        assert_array_equal(np.percentile(np.arange(10), []), np.array([]))

    def test_void_compare_segfault(self):
        # gh-6922. The following should not segfault
        a = np.ones(3, dtype=[('object', 'O'), ('int', '<i2')])
        a.sort()

    def test_reshape_size_overflow(self):
        # gh-7455
        a = np.ones(20)[::2]
        if np.dtype(np.intp).itemsize == 8:
            # 64 bit. The following are the prime factors of 2**63 + 5,
            # plus a leading 2, so when multiplied together as int64,
            # the result overflows to a total size of 10.
            new_shape = (2, 13, 419, 691, 823, 2977518503)
        else:
            # 32 bit. The following are the prime factors of 2**31 + 5,
            # plus a leading 2, so when multiplied together as int32,
            # the result overflows to a total size of 10.
            new_shape = (2, 7, 7, 43826197)
        assert_raises(ValueError, a.reshape, new_shape)

    @pytest.mark.skipif(IS_PYPY and sys.implementation.version <= (7, 3, 8),
            reason="PyPy bug in error formatting")
    def test_invalid_structured_dtypes(self):
        # gh-2865
        # mapping python objects to other dtypes
        assert_raises(ValueError, np.dtype, ('O', [('name', 'i8')]))
        assert_raises(ValueError, np.dtype, ('i8', [('name', 'O')]))
        assert_raises(ValueError, np.dtype,
                      ('i8', [('name', [('name', 'O')])]))
        assert_raises(ValueError, np.dtype, ([('a', 'i4'), ('b', 'i4')], 'O'))
        assert_raises(ValueError, np.dtype, ('i8', 'O'))
        # wrong number/type of tuple elements in dict
        assert_raises(ValueError, np.dtype,
                      ('i', {'name': ('i', 0, 'title', 'oops')}))
        assert_raises(ValueError, np.dtype,
                      ('i', {'name': ('i', 'wrongtype', 'title')}))
        # disallowed as of 1.13
        assert_raises(ValueError, np.dtype,
                      ([('a', 'O'), ('b', 'O')], [('c', 'O'), ('d', 'O')]))
        # allowed as a special case due to existing use, see gh-2798
        a = np.ones(1, dtype=('O', [('name', 'O')]))
        assert_equal(a[0], 1)
        # In particular, the above union dtype (and union dtypes in general)
        # should mainly behave like the main (object) dtype:
        assert a[0] is a.item()
        assert type(a[0]) is int

    def test_correct_hash_dict(self):
        # gh-8887 - __hash__ would be None despite tp_hash being set
        all_types = set(np.sctypeDict.values()) - {np.void}
        for t in all_types:
            val = t()

            try:
                hash(val)
            except TypeError as e:
                assert_equal(t.__hash__, None)
            else:
                assert_(t.__hash__ != None)

    def test_scalar_copy(self):
        scalar_types = set(np.sctypeDict.values())
        values = {
            np.void: b"a",
            np.bytes_: b"a",
            np.unicode_: "a",
            np.datetime64: "2017-08-25",
        }
        for sctype in scalar_types:
            item = sctype(values.get(sctype, 1))
            item2 = copy.copy(item)
            assert_equal(item, item2)

    def test_void_item_memview(self):
        va = np.zeros(10, 'V4')
        x = va[:1].item()
        va[0] = b'\xff\xff\xff\xff'
        del va
        assert_equal(x, b'\x00\x00\x00\x00')

    def test_void_getitem(self):
        # Test fix for gh-11668.
        assert_(np.array([b'a'], 'V1').astype('O') == b'a')
        assert_(np.array([b'ab'], 'V2').astype('O') == b'ab')
        assert_(np.array([b'abc'], 'V3').astype('O') == b'abc')
        assert_(np.array([b'abcd'], 'V4').astype('O') == b'abcd')

    def test_structarray_title(self):
        # The following used to segfault on pypy, due to NPY_TITLE_KEY
        # not working properly and resulting to double-decref of the
        # structured array field items:
        # See: https://bitbucket.org/pypy/pypy/issues/2789
        for j in range(5):
            structure = np.array([1], dtype=[(('x', 'X'), np.object_)])
            structure[0]['x'] = np.array([2])
            gc.collect()

    def test_dtype_scalar_squeeze(self):
        # gh-11384
        values = {
            'S': b"a",
            'M': "2018-06-20",
        }
        for ch in np.typecodes['All']:
            if ch in 'O':
                continue
            sctype = np.dtype(ch).type
            scvalue = sctype(values.get(ch, 3))
            for axis in [None, ()]:
                squeezed = scvalue.squeeze(axis=axis)
                assert_equal(squeezed, scvalue)
                assert_equal(type(squeezed), type(scvalue))

    def test_field_access_by_title(self):
        # gh-11507
        s = 'Some long field name'
        if HAS_REFCOUNT:
            base = sys.getrefcount(s)
        t = np.dtype([((s, 'f1'), np.float64)])
        data = np.zeros(10, t)
        for i in range(10):
            str(data[['f1']])
            if HAS_REFCOUNT:
                assert_(base <= sys.getrefcount(s))

    @pytest.mark.parametrize('val', [
        # arrays and scalars
        np.ones((10, 10), dtype='int32'),
        np.uint64(10),
        ])
    @pytest.mark.parametrize('protocol',
        range(2, pickle.HIGHEST_PROTOCOL + 1)
        )
    def test_pickle_module(self, protocol, val):
        # gh-12837
        s = pickle.dumps(val, protocol)
        assert b'_multiarray_umath' not in s
        if protocol == 5 and len(val.shape) > 0:
            # unpickling ndarray goes through _frombuffer for protocol 5
            assert b'numpy.core.numeric' in s
        else:
            assert b'numpy.core.multiarray' in s

    def test_object_casting_errors(self):
        # gh-11993 update to ValueError (see gh-16909), since strings can in
        # principle be converted to complex, but this string cannot.
        arr = np.array(['AAAAA', 18465886.0, 18465886.0], dtype=object)
        assert_raises(ValueError, arr.astype, 'c8')

    def test_eff1d_casting(self):
        # gh-12711
        x = np.array([1, 2, 4, 7, 0], dtype=np.int16)
        res = np.ediff1d(x, to_begin=-99, to_end=np.array([88, 99]))
        assert_equal(res, [-99,   1,   2,   3,  -7,  88,  99])

        # The use of safe casting means, that 1<<20 is cast unsafely, an
        # error may be better, but currently there is no mechanism for it.
        res = np.ediff1d(x, to_begin=(1<<20), to_end=(1<<20))
        assert_equal(res, [0,   1,   2,   3,  -7,  0])

    def test_pickle_datetime64_array(self):
        # gh-12745 (would fail with pickle5 installed)
        d = np.datetime64('2015-07-04 12:59:59.50', 'ns')
        arr = np.array([d])
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            dumped = pickle.dumps(arr, protocol=proto)
            assert_equal(pickle.loads(dumped), arr)

    def test_bad_array_interface(self):
        class T:
            __array_interface__ = {}

        with assert_raises(ValueError):
            np.array([T()])

    def test_2d__array__shape(self):
        class T:
            def __array__(self):
                return np.ndarray(shape=(0,0))

            # Make sure __array__ is used instead of Sequence methods.
            def __iter__(self):
                return iter([])

            def __getitem__(self, idx):
                raise AssertionError("__getitem__ was called")

            def __len__(self):
                return 0


        t = T()
        # gh-13659, would raise in broadcasting [x=t for x in result]
        arr = np.array([t])
        assert arr.shape == (1, 0, 0)

    @pytest.mark.skipif(sys.maxsize < 2 ** 31 + 1, reason='overflows 32-bit python')
    def test_to_ctypes(self):
        #gh-14214
        arr = np.zeros((2 ** 31 + 1,), 'b')
        assert arr.size * arr.itemsize > 2 ** 31
        c_arr = np.ctypeslib.as_ctypes(arr)
        assert_equal(c_arr._length_, arr.size)

    def test_complex_conversion_error(self):
        # gh-17068
        with pytest.raises(TypeError, match=r"Unable to convert dtype.*"):
            complex(np.array("now", np.datetime64))

    def test__array_interface__descr(self):
        # gh-17068
        dt = np.dtype(dict(names=['a', 'b'],
                           offsets=[0, 0],
                           formats=[np.int64, np.int64]))
        descr = np.array((1, 1), dtype=dt).__array_interface__['descr']
        assert descr == [('', '|V8')]  # instead of [(b'', '|V8')]

    @pytest.mark.skipif(sys.maxsize < 2 ** 31 + 1, reason='overflows 32-bit python')
    @requires_memory(free_bytes=9e9)
    def test_dot_big_stride(self):
        # gh-17111
        # blas stride = stride//itemsize > int32 max
        int32_max = np.iinfo(np.int32).max
        n = int32_max + 3
        a = np.empty([n], dtype=np.float32)
        b = a[::n-1]
        b[...] = 1
        assert b.strides[0] > int32_max * b.dtype.itemsize
        assert np.dot(b, b) == 2.0

    def test_frompyfunc_name(self):
        # name conversion was failing for python 3 strings
        # resulting in the default '?' name. Also test utf-8
        # encoding using non-ascii name.
        def cassé(x):
            return x

        f = np.frompyfunc(cassé, 1, 1)
        assert str(f) == "<ufunc 'cassé (vectorized)'>"

    @pytest.mark.parametrize("operation", [
        'add', 'subtract', 'multiply', 'floor_divide',
        'conjugate', 'fmod', 'square', 'reciprocal',
        'power', 'absolute', 'negative', 'positive',
        'greater', 'greater_equal', 'less',
        'less_equal', 'equal', 'not_equal', 'logical_and',
        'logical_not', 'logical_or', 'bitwise_and', 'bitwise_or',
        'bitwise_xor', 'invert', 'left_shift', 'right_shift',
        'gcd', 'lcm'
        ]
    )
    @pytest.mark.parametrize("order", [
        ('b->', 'B->'),
        ('h->', 'H->'),
        ('i->', 'I->'),
        ('l->', 'L->'),
        ('q->', 'Q->'),
        ]
    )
    def test_ufunc_order(self, operation, order):
        # gh-18075
        # Ensure signed types before unsigned
        def get_idx(string, str_lst):
            for i, s in enumerate(str_lst):
                if string in s:
                    return i
            raise ValueError(f"{string} not in list")
        types = getattr(np, operation).types
        assert get_idx(order[0], types) < get_idx(order[1], types), (
                f"Unexpected types order of ufunc in {operation}"
                f"for {order}. Possible fix: Use signed before unsigned"
                "in generate_umath.py")
