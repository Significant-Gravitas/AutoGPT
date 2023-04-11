import sys
import pytest
import weakref
from pathlib import Path

import numpy as np
from numpy.ctypeslib import ndpointer, load_library, as_array
from numpy.distutils.misc_util import get_shared_lib_extension
from numpy.testing import assert_, assert_array_equal, assert_raises, assert_equal

try:
    import ctypes
except ImportError:
    ctypes = None
else:
    cdll = None
    test_cdll = None
    if hasattr(sys, 'gettotalrefcount'):
        try:
            cdll = load_library('_multiarray_umath_d', np.core._multiarray_umath.__file__)
        except OSError:
            pass
        try:
            test_cdll = load_library('_multiarray_tests', np.core._multiarray_tests.__file__)
        except OSError:
            pass
    if cdll is None:
        cdll = load_library('_multiarray_umath', np.core._multiarray_umath.__file__)
    if test_cdll is None:
        test_cdll = load_library('_multiarray_tests', np.core._multiarray_tests.__file__)

    c_forward_pointer = test_cdll.forward_pointer


@pytest.mark.skipif(ctypes is None,
                    reason="ctypes not available in this python")
@pytest.mark.skipif(sys.platform == 'cygwin',
                    reason="Known to fail on cygwin")
class TestLoadLibrary:
    def test_basic(self):
        loader_path = np.core._multiarray_umath.__file__

        out1 = load_library('_multiarray_umath', loader_path)
        out2 = load_library(Path('_multiarray_umath'), loader_path)
        out3 = load_library('_multiarray_umath', Path(loader_path))
        out4 = load_library(b'_multiarray_umath', loader_path)

        assert isinstance(out1, ctypes.CDLL)
        assert out1 is out2 is out3 is out4

    def test_basic2(self):
        # Regression for #801: load_library with a full library name
        # (including extension) does not work.
        try:
            try:
                so = get_shared_lib_extension(is_python_ext=True)
                # Should succeed
                load_library('_multiarray_umath%s' % so, np.core._multiarray_umath.__file__)
            except ImportError:
                print("No distutils available, skipping test.")
        except ImportError as e:
            msg = ("ctypes is not available on this python: skipping the test"
                   " (import error was: %s)" % str(e))
            print(msg)


class TestNdpointer:
    def test_dtype(self):
        dt = np.intc
        p = ndpointer(dtype=dt)
        assert_(p.from_param(np.array([1], dt)))
        dt = '<i4'
        p = ndpointer(dtype=dt)
        assert_(p.from_param(np.array([1], dt)))
        dt = np.dtype('>i4')
        p = ndpointer(dtype=dt)
        p.from_param(np.array([1], dt))
        assert_raises(TypeError, p.from_param,
                          np.array([1], dt.newbyteorder('swap')))
        dtnames = ['x', 'y']
        dtformats = [np.intc, np.float64]
        dtdescr = {'names': dtnames, 'formats': dtformats}
        dt = np.dtype(dtdescr)
        p = ndpointer(dtype=dt)
        assert_(p.from_param(np.zeros((10,), dt)))
        samedt = np.dtype(dtdescr)
        p = ndpointer(dtype=samedt)
        assert_(p.from_param(np.zeros((10,), dt)))
        dt2 = np.dtype(dtdescr, align=True)
        if dt.itemsize != dt2.itemsize:
            assert_raises(TypeError, p.from_param, np.zeros((10,), dt2))
        else:
            assert_(p.from_param(np.zeros((10,), dt2)))

    def test_ndim(self):
        p = ndpointer(ndim=0)
        assert_(p.from_param(np.array(1)))
        assert_raises(TypeError, p.from_param, np.array([1]))
        p = ndpointer(ndim=1)
        assert_raises(TypeError, p.from_param, np.array(1))
        assert_(p.from_param(np.array([1])))
        p = ndpointer(ndim=2)
        assert_(p.from_param(np.array([[1]])))

    def test_shape(self):
        p = ndpointer(shape=(1, 2))
        assert_(p.from_param(np.array([[1, 2]])))
        assert_raises(TypeError, p.from_param, np.array([[1], [2]]))
        p = ndpointer(shape=())
        assert_(p.from_param(np.array(1)))

    def test_flags(self):
        x = np.array([[1, 2], [3, 4]], order='F')
        p = ndpointer(flags='FORTRAN')
        assert_(p.from_param(x))
        p = ndpointer(flags='CONTIGUOUS')
        assert_raises(TypeError, p.from_param, x)
        p = ndpointer(flags=x.flags.num)
        assert_(p.from_param(x))
        assert_raises(TypeError, p.from_param, np.array([[1, 2], [3, 4]]))

    def test_cache(self):
        assert_(ndpointer(dtype=np.float64) is ndpointer(dtype=np.float64))

        # shapes are normalized
        assert_(ndpointer(shape=2) is ndpointer(shape=(2,)))

        # 1.12 <= v < 1.16 had a bug that made these fail
        assert_(ndpointer(shape=2) is not ndpointer(ndim=2))
        assert_(ndpointer(ndim=2) is not ndpointer(shape=2))

@pytest.mark.skipif(ctypes is None,
                    reason="ctypes not available on this python installation")
class TestNdpointerCFunc:
    def test_arguments(self):
        """ Test that arguments are coerced from arrays """
        c_forward_pointer.restype = ctypes.c_void_p
        c_forward_pointer.argtypes = (ndpointer(ndim=2),)

        c_forward_pointer(np.zeros((2, 3)))
        # too many dimensions
        assert_raises(
            ctypes.ArgumentError, c_forward_pointer, np.zeros((2, 3, 4)))

    @pytest.mark.parametrize(
        'dt', [
            float,
            np.dtype(dict(
                formats=['<i4', '<i4'],
                names=['a', 'b'],
                offsets=[0, 2],
                itemsize=6
            ))
        ], ids=[
            'float',
            'overlapping-fields'
        ]
    )
    def test_return(self, dt):
        """ Test that return values are coerced to arrays """
        arr = np.zeros((2, 3), dt)
        ptr_type = ndpointer(shape=arr.shape, dtype=arr.dtype)

        c_forward_pointer.restype = ptr_type
        c_forward_pointer.argtypes = (ptr_type,)

        # check that the arrays are equivalent views on the same data
        arr2 = c_forward_pointer(arr)
        assert_equal(arr2.dtype, arr.dtype)
        assert_equal(arr2.shape, arr.shape)
        assert_equal(
            arr2.__array_interface__['data'],
            arr.__array_interface__['data']
        )

    def test_vague_return_value(self):
        """ Test that vague ndpointer return values do not promote to arrays """
        arr = np.zeros((2, 3))
        ptr_type = ndpointer(dtype=arr.dtype)

        c_forward_pointer.restype = ptr_type
        c_forward_pointer.argtypes = (ptr_type,)

        ret = c_forward_pointer(arr)
        assert_(isinstance(ret, ptr_type))


@pytest.mark.skipif(ctypes is None,
                    reason="ctypes not available on this python installation")
class TestAsArray:
    def test_array(self):
        from ctypes import c_int

        pair_t = c_int * 2
        a = as_array(pair_t(1, 2))
        assert_equal(a.shape, (2,))
        assert_array_equal(a, np.array([1, 2]))
        a = as_array((pair_t * 3)(pair_t(1, 2), pair_t(3, 4), pair_t(5, 6)))
        assert_equal(a.shape, (3, 2))
        assert_array_equal(a, np.array([[1, 2], [3, 4], [5, 6]]))

    def test_pointer(self):
        from ctypes import c_int, cast, POINTER

        p = cast((c_int * 10)(*range(10)), POINTER(c_int))

        a = as_array(p, shape=(10,))
        assert_equal(a.shape, (10,))
        assert_array_equal(a, np.arange(10))

        a = as_array(p, shape=(2, 5))
        assert_equal(a.shape, (2, 5))
        assert_array_equal(a, np.arange(10).reshape((2, 5)))

        # shape argument is required
        assert_raises(TypeError, as_array, p)

    def test_struct_array_pointer(self):
        from ctypes import c_int16, Structure, pointer

        class Struct(Structure):
            _fields_ = [('a', c_int16)]

        Struct3 = 3 * Struct

        c_array = (2 * Struct3)(
            Struct3(Struct(a=1), Struct(a=2), Struct(a=3)),
            Struct3(Struct(a=4), Struct(a=5), Struct(a=6))
        )

        expected = np.array([
            [(1,), (2,), (3,)],
            [(4,), (5,), (6,)],
        ], dtype=[('a', np.int16)])

        def check(x):
            assert_equal(x.dtype, expected.dtype)
            assert_equal(x, expected)

        # all of these should be equivalent
        check(as_array(c_array))
        check(as_array(pointer(c_array), shape=()))
        check(as_array(pointer(c_array[0]), shape=(2,)))
        check(as_array(pointer(c_array[0][0]), shape=(2, 3)))

    def test_reference_cycles(self):
        # related to gh-6511
        import ctypes

        # create array to work with
        # don't use int/long to avoid running into bpo-10746
        N = 100
        a = np.arange(N, dtype=np.short)

        # get pointer to array
        pnt = np.ctypeslib.as_ctypes(a)

        with np.testing.assert_no_gc_cycles():
            # decay the array above to a pointer to its first element
            newpnt = ctypes.cast(pnt, ctypes.POINTER(ctypes.c_short))
            # and construct an array using this data
            b = np.ctypeslib.as_array(newpnt, (N,))
            # now delete both, which should cleanup both objects
            del newpnt, b

    def test_segmentation_fault(self):
        arr = np.zeros((224, 224, 3))
        c_arr = np.ctypeslib.as_ctypes(arr)
        arr_ref = weakref.ref(arr)
        del arr

        # check the reference wasn't cleaned up
        assert_(arr_ref() is not None)

        # check we avoid the segfault
        c_arr[0][0][0]


@pytest.mark.skipif(ctypes is None,
                    reason="ctypes not available on this python installation")
class TestAsCtypesType:
    """ Test conversion from dtypes to ctypes types """
    def test_scalar(self):
        dt = np.dtype('<u2')
        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_equal(ct, ctypes.c_uint16.__ctype_le__)

        dt = np.dtype('>u2')
        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_equal(ct, ctypes.c_uint16.__ctype_be__)

        dt = np.dtype('u2')
        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_equal(ct, ctypes.c_uint16)

    def test_subarray(self):
        dt = np.dtype((np.int32, (2, 3)))
        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_equal(ct, 2 * (3 * ctypes.c_int32))

    def test_structure(self):
        dt = np.dtype([
            ('a', np.uint16),
            ('b', np.uint32),
        ])

        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_(issubclass(ct, ctypes.Structure))
        assert_equal(ctypes.sizeof(ct), dt.itemsize)
        assert_equal(ct._fields_, [
            ('a', ctypes.c_uint16),
            ('b', ctypes.c_uint32),
        ])

    def test_structure_aligned(self):
        dt = np.dtype([
            ('a', np.uint16),
            ('b', np.uint32),
        ], align=True)

        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_(issubclass(ct, ctypes.Structure))
        assert_equal(ctypes.sizeof(ct), dt.itemsize)
        assert_equal(ct._fields_, [
            ('a', ctypes.c_uint16),
            ('', ctypes.c_char * 2),  # padding
            ('b', ctypes.c_uint32),
        ])

    def test_union(self):
        dt = np.dtype(dict(
            names=['a', 'b'],
            offsets=[0, 0],
            formats=[np.uint16, np.uint32]
        ))

        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_(issubclass(ct, ctypes.Union))
        assert_equal(ctypes.sizeof(ct), dt.itemsize)
        assert_equal(ct._fields_, [
            ('a', ctypes.c_uint16),
            ('b', ctypes.c_uint32),
        ])

    def test_padded_union(self):
        dt = np.dtype(dict(
            names=['a', 'b'],
            offsets=[0, 0],
            formats=[np.uint16, np.uint32],
            itemsize=5,
        ))

        ct = np.ctypeslib.as_ctypes_type(dt)
        assert_(issubclass(ct, ctypes.Union))
        assert_equal(ctypes.sizeof(ct), dt.itemsize)
        assert_equal(ct._fields_, [
            ('a', ctypes.c_uint16),
            ('b', ctypes.c_uint32),
            ('', ctypes.c_char * 5),  # padding
        ])

    def test_overlapping(self):
        dt = np.dtype(dict(
            names=['a', 'b'],
            offsets=[0, 2],
            formats=[np.uint32, np.uint32]
        ))
        assert_raises(NotImplementedError, np.ctypeslib.as_ctypes_type, dt)
