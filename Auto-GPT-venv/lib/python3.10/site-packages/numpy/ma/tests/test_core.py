# pylint: disable-msg=W0400,W0511,W0611,W0612,W0614,R0201,E1102
"""Tests suite for MaskedArray & subclassing.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
"""
__author__ = "Pierre GF Gerard-Marchant"

import sys
import warnings
import operator
import itertools
import textwrap
import pytest

from functools import reduce


import numpy as np
import numpy.ma.core
import numpy.core.fromnumeric as fromnumeric
import numpy.core.umath as umath
from numpy.testing import (
    assert_raises, assert_warns, suppress_warnings, IS_WASM
    )
from numpy.testing._private.utils import requires_memory
from numpy import ndarray
from numpy.compat import asbytes
from numpy.ma.testutils import (
    assert_, assert_array_equal, assert_equal, assert_almost_equal,
    assert_equal_records, fail_if_equal, assert_not_equal,
    assert_mask_equal
    )
from numpy.ma.core import (
    MAError, MaskError, MaskType, MaskedArray, abs, absolute, add, all,
    allclose, allequal, alltrue, angle, anom, arange, arccos, arccosh, arctan2,
    arcsin, arctan, argsort, array, asarray, choose, concatenate,
    conjugate, cos, cosh, count, default_fill_value, diag, divide, doc_note,
    empty, empty_like, equal, exp, flatten_mask, filled, fix_invalid,
    flatten_structured_array, fromflex, getmask, getmaskarray, greater,
    greater_equal, identity, inner, isMaskedArray, less, less_equal, log,
    log10, make_mask, make_mask_descr, mask_or, masked, masked_array,
    masked_equal, masked_greater, masked_greater_equal, masked_inside,
    masked_less, masked_less_equal, masked_not_equal, masked_outside,
    masked_print_option, masked_values, masked_where, max, maximum,
    maximum_fill_value, min, minimum, minimum_fill_value, mod, multiply,
    mvoid, nomask, not_equal, ones, ones_like, outer, power, product, put,
    putmask, ravel, repeat, reshape, resize, shape, sin, sinh, sometrue, sort,
    sqrt, subtract, sum, take, tan, tanh, transpose, where, zeros, zeros_like,
    )
from numpy.compat import pickle

pi = np.pi


suppress_copy_mask_on_assignment = suppress_warnings()
suppress_copy_mask_on_assignment.filter(
    numpy.ma.core.MaskedArrayFutureWarning,
    "setting an item on a masked array which has a shared mask will not copy")


# For parametrized numeric testing
num_dts = [np.dtype(dt_) for dt_ in '?bhilqBHILQefdgFD']
num_ids = [dt_.char for dt_ in num_dts]


class TestMaskedArray:
    # Base test class for MaskedArrays.

    def setup_method(self):
        # Base data definition.
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        a10 = 10.
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = masked_array(x, mask=m1)
        ym = masked_array(y, mask=m2)
        z = np.array([-.5, 0., .5, .8])
        zm = masked_array(z, mask=[0, 1, 0, 0])
        xf = np.where(m1, 1e+20, x)
        xm.set_fill_value(1e+20)
        self.d = (x, y, a10, m1, m2, xm, ym, z, zm, xf)

    def test_basicattributes(self):
        # Tests some basic array attributes.
        a = array([1, 3, 2])
        b = array([1, 3, 2], mask=[1, 0, 1])
        assert_equal(a.ndim, 1)
        assert_equal(b.ndim, 1)
        assert_equal(a.size, 3)
        assert_equal(b.size, 3)
        assert_equal(a.shape, (3,))
        assert_equal(b.shape, (3,))

    def test_basic0d(self):
        # Checks masking a scalar
        x = masked_array(0)
        assert_equal(str(x), '0')
        x = masked_array(0, mask=True)
        assert_equal(str(x), str(masked_print_option))
        x = masked_array(0, mask=False)
        assert_equal(str(x), '0')
        x = array(0, mask=1)
        assert_(x.filled().dtype is x._data.dtype)

    def test_basic1d(self):
        # Test of basic array creation and properties in 1 dimension.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        assert_(not isMaskedArray(x))
        assert_(isMaskedArray(xm))
        assert_((xm - ym).filled(0).any())
        fail_if_equal(xm.mask.astype(int), ym.mask.astype(int))
        s = x.shape
        assert_equal(np.shape(xm), s)
        assert_equal(xm.shape, s)
        assert_equal(xm.dtype, x.dtype)
        assert_equal(zm.dtype, z.dtype)
        assert_equal(xm.size, reduce(lambda x, y:x * y, s))
        assert_equal(count(xm), len(m1) - reduce(lambda x, y:x + y, m1))
        assert_array_equal(xm, xf)
        assert_array_equal(filled(xm, 1.e20), xf)
        assert_array_equal(x, xm)

    def test_basic2d(self):
        # Test of basic array creation and properties in 2 dimensions.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        for s in [(4, 3), (6, 2)]:
            x.shape = s
            y.shape = s
            xm.shape = s
            ym.shape = s
            xf.shape = s

            assert_(not isMaskedArray(x))
            assert_(isMaskedArray(xm))
            assert_equal(shape(xm), s)
            assert_equal(xm.shape, s)
            assert_equal(xm.size, reduce(lambda x, y:x * y, s))
            assert_equal(count(xm), len(m1) - reduce(lambda x, y:x + y, m1))
            assert_equal(xm, xf)
            assert_equal(filled(xm, 1.e20), xf)
            assert_equal(x, xm)

    def test_concatenate_basic(self):
        # Tests concatenations.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        # basic concatenation
        assert_equal(np.concatenate((x, y)), concatenate((xm, ym)))
        assert_equal(np.concatenate((x, y)), concatenate((x, y)))
        assert_equal(np.concatenate((x, y)), concatenate((xm, y)))
        assert_equal(np.concatenate((x, y, x)), concatenate((x, ym, x)))

    def test_concatenate_alongaxis(self):
        # Tests concatenations.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        # Concatenation along an axis
        s = (3, 4)
        x.shape = y.shape = xm.shape = ym.shape = s
        assert_equal(xm.mask, np.reshape(m1, s))
        assert_equal(ym.mask, np.reshape(m2, s))
        xmym = concatenate((xm, ym), 1)
        assert_equal(np.concatenate((x, y), 1), xmym)
        assert_equal(np.concatenate((xm.mask, ym.mask), 1), xmym._mask)

        x = zeros(2)
        y = array(ones(2), mask=[False, True])
        z = concatenate((x, y))
        assert_array_equal(z, [0, 0, 1, 1])
        assert_array_equal(z.mask, [False, False, False, True])
        z = concatenate((y, x))
        assert_array_equal(z, [1, 1, 0, 0])
        assert_array_equal(z.mask, [False, True, False, False])

    def test_concatenate_flexible(self):
        # Tests the concatenation on flexible arrays.
        data = masked_array(list(zip(np.random.rand(10),
                                     np.arange(10))),
                            dtype=[('a', float), ('b', int)])

        test = concatenate([data[:5], data[5:]])
        assert_equal_records(test, data)

    def test_creation_ndmin(self):
        # Check the use of ndmin
        x = array([1, 2, 3], mask=[1, 0, 0], ndmin=2)
        assert_equal(x.shape, (1, 3))
        assert_equal(x._data, [[1, 2, 3]])
        assert_equal(x._mask, [[1, 0, 0]])

    def test_creation_ndmin_from_maskedarray(self):
        # Make sure we're not losing the original mask w/ ndmin
        x = array([1, 2, 3])
        x[-1] = masked
        xx = array(x, ndmin=2, dtype=float)
        assert_equal(x.shape, x._mask.shape)
        assert_equal(xx.shape, xx._mask.shape)

    def test_creation_maskcreation(self):
        # Tests how masks are initialized at the creation of Maskedarrays.
        data = arange(24, dtype=float)
        data[[3, 6, 15]] = masked
        dma_1 = MaskedArray(data)
        assert_equal(dma_1.mask, data.mask)
        dma_2 = MaskedArray(dma_1)
        assert_equal(dma_2.mask, dma_1.mask)
        dma_3 = MaskedArray(dma_1, mask=[1, 0, 0, 0] * 6)
        fail_if_equal(dma_3.mask, dma_1.mask)

        x = array([1, 2, 3], mask=True)
        assert_equal(x._mask, [True, True, True])
        x = array([1, 2, 3], mask=False)
        assert_equal(x._mask, [False, False, False])
        y = array([1, 2, 3], mask=x._mask, copy=False)
        assert_(np.may_share_memory(x.mask, y.mask))
        y = array([1, 2, 3], mask=x._mask, copy=True)
        assert_(not np.may_share_memory(x.mask, y.mask))

    def test_masked_singleton_array_creation_warns(self):
        # The first works, but should not (ideally), there may be no way
        # to solve this, however, as long as `np.ma.masked` is an ndarray.
        np.array(np.ma.masked)
        with pytest.warns(UserWarning):
            # Tries to create a float array, using `float(np.ma.masked)`.
            # We may want to define this is invalid behaviour in the future!
            # (requiring np.ma.masked to be a known NumPy scalar probably
            # with a DType.)
            np.array([3., np.ma.masked])

    def test_creation_with_list_of_maskedarrays(self):
        # Tests creating a masked array from a list of masked arrays.
        x = array(np.arange(5), mask=[1, 0, 0, 0, 0])
        data = array((x, x[::-1]))
        assert_equal(data, [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
        assert_equal(data._mask, [[1, 0, 0, 0, 0], [0, 0, 0, 0, 1]])

        x.mask = nomask
        data = array((x, x[::-1]))
        assert_equal(data, [[0, 1, 2, 3, 4], [4, 3, 2, 1, 0]])
        assert_(data.mask is nomask)

    def test_creation_with_list_of_maskedarrays_no_bool_cast(self):
        # Tests the regression in gh-18551
        masked_str = np.ma.masked_array(['a', 'b'], mask=[True, False])
        normal_int = np.arange(2)
        res = np.ma.asarray([masked_str, normal_int], dtype="U21")
        assert_array_equal(res.mask, [[True, False], [False, False]])

        # The above only failed due a long chain of oddity, try also with
        # an object array that cannot be converted to bool always:
        class NotBool():
            def __bool__(self):
                raise ValueError("not a bool!")
        masked_obj = np.ma.masked_array([NotBool(), 'b'], mask=[True, False])
        # Check that the NotBool actually fails like we would expect:
        with pytest.raises(ValueError, match="not a bool!"):
            np.asarray([masked_obj], dtype=bool)

        res = np.ma.asarray([masked_obj, normal_int])
        assert_array_equal(res.mask, [[True, False], [False, False]])

    def test_creation_from_ndarray_with_padding(self):
        x = np.array([('A', 0)], dtype={'names':['f0','f1'],
                                        'formats':['S4','i8'],
                                        'offsets':[0,8]})
        array(x)  # used to fail due to 'V' padding field in x.dtype.descr

    def test_unknown_keyword_parameter(self):
        with pytest.raises(TypeError, match="unexpected keyword argument"):
            MaskedArray([1, 2, 3], maks=[0, 1, 0])  # `mask` is misspelled.

    def test_asarray(self):
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        xm.fill_value = -9999
        xm._hardmask = True
        xmm = asarray(xm)
        assert_equal(xmm._data, xm._data)
        assert_equal(xmm._mask, xm._mask)
        assert_equal(xmm.fill_value, xm.fill_value)
        assert_equal(xmm._hardmask, xm._hardmask)

    def test_asarray_default_order(self):
        # See Issue #6646
        m = np.eye(3).T
        assert_(not m.flags.c_contiguous)

        new_m = asarray(m)
        assert_(new_m.flags.c_contiguous)

    def test_asarray_enforce_order(self):
        # See Issue #6646
        m = np.eye(3).T
        assert_(not m.flags.c_contiguous)

        new_m = asarray(m, order='C')
        assert_(new_m.flags.c_contiguous)

    def test_fix_invalid(self):
        # Checks fix_invalid.
        with np.errstate(invalid='ignore'):
            data = masked_array([np.nan, 0., 1.], mask=[0, 0, 1])
            data_fixed = fix_invalid(data)
            assert_equal(data_fixed._data, [data.fill_value, 0., 1.])
            assert_equal(data_fixed._mask, [1., 0., 1.])

    def test_maskedelement(self):
        # Test of masked element
        x = arange(6)
        x[1] = masked
        assert_(str(masked) == '--')
        assert_(x[1] is masked)
        assert_equal(filled(x[1], 0), 0)

    def test_set_element_as_object(self):
        # Tests setting elements with object
        a = empty(1, dtype=object)
        x = (1, 2, 3, 4, 5)
        a[0] = x
        assert_equal(a[0], x)
        assert_(a[0] is x)

        import datetime
        dt = datetime.datetime.now()
        a[0] = dt
        assert_(a[0] is dt)

    def test_indexing(self):
        # Tests conversions and indexing
        x1 = np.array([1, 2, 4, 3])
        x2 = array(x1, mask=[1, 0, 0, 0])
        x3 = array(x1, mask=[0, 1, 0, 1])
        x4 = array(x1)
        # test conversion to strings
        str(x2)  # raises?
        repr(x2)  # raises?
        assert_equal(np.sort(x1), sort(x2, endwith=False))
        # tests of indexing
        assert_(type(x2[1]) is type(x1[1]))
        assert_(x1[1] == x2[1])
        assert_(x2[0] is masked)
        assert_equal(x1[2], x2[2])
        assert_equal(x1[2:5], x2[2:5])
        assert_equal(x1[:], x2[:])
        assert_equal(x1[1:], x3[1:])
        x1[2] = 9
        x2[2] = 9
        assert_equal(x1, x2)
        x1[1:3] = 99
        x2[1:3] = 99
        assert_equal(x1, x2)
        x2[1] = masked
        assert_equal(x1, x2)
        x2[1:3] = masked
        assert_equal(x1, x2)
        x2[:] = x1
        x2[1] = masked
        assert_(allequal(getmask(x2), array([0, 1, 0, 0])))
        x3[:] = masked_array([1, 2, 3, 4], [0, 1, 1, 0])
        assert_(allequal(getmask(x3), array([0, 1, 1, 0])))
        x4[:] = masked_array([1, 2, 3, 4], [0, 1, 1, 0])
        assert_(allequal(getmask(x4), array([0, 1, 1, 0])))
        assert_(allequal(x4, array([1, 2, 3, 4])))
        x1 = np.arange(5) * 1.0
        x2 = masked_values(x1, 3.0)
        assert_equal(x1, x2)
        assert_(allequal(array([0, 0, 0, 1, 0], MaskType), x2.mask))
        assert_equal(3.0, x2.fill_value)
        x1 = array([1, 'hello', 2, 3], object)
        x2 = np.array([1, 'hello', 2, 3], object)
        s1 = x1[1]
        s2 = x2[1]
        assert_equal(type(s2), str)
        assert_equal(type(s1), str)
        assert_equal(s1, s2)
        assert_(x1[1:1].shape == (0,))

    @suppress_copy_mask_on_assignment
    def test_copy(self):
        # Tests of some subtle points of copying and sizing.
        n = [0, 0, 1, 0, 0]
        m = make_mask(n)
        m2 = make_mask(m)
        assert_(m is m2)
        m3 = make_mask(m, copy=True)
        assert_(m is not m3)

        x1 = np.arange(5)
        y1 = array(x1, mask=m)
        assert_equal(y1._data.__array_interface__, x1.__array_interface__)
        assert_(allequal(x1, y1.data))
        assert_equal(y1._mask.__array_interface__, m.__array_interface__)

        y1a = array(y1)
        # Default for masked array is not to copy; see gh-10318.
        assert_(y1a._data.__array_interface__ ==
                        y1._data.__array_interface__)
        assert_(y1a._mask.__array_interface__ ==
                        y1._mask.__array_interface__)

        y2 = array(x1, mask=m3)
        assert_(y2._data.__array_interface__ == x1.__array_interface__)
        assert_(y2._mask.__array_interface__ == m3.__array_interface__)
        assert_(y2[2] is masked)
        y2[2] = 9
        assert_(y2[2] is not masked)
        assert_(y2._mask.__array_interface__ == m3.__array_interface__)
        assert_(allequal(y2.mask, 0))

        y2a = array(x1, mask=m, copy=1)
        assert_(y2a._data.__array_interface__ != x1.__array_interface__)
        #assert_( y2a._mask is not m)
        assert_(y2a._mask.__array_interface__ != m.__array_interface__)
        assert_(y2a[2] is masked)
        y2a[2] = 9
        assert_(y2a[2] is not masked)
        #assert_( y2a._mask is not m)
        assert_(y2a._mask.__array_interface__ != m.__array_interface__)
        assert_(allequal(y2a.mask, 0))

        y3 = array(x1 * 1.0, mask=m)
        assert_(filled(y3).dtype is (x1 * 1.0).dtype)

        x4 = arange(4)
        x4[2] = masked
        y4 = resize(x4, (8,))
        assert_equal(concatenate([x4, x4]), y4)
        assert_equal(getmask(y4), [0, 0, 1, 0, 0, 0, 1, 0])
        y5 = repeat(x4, (2, 2, 2, 2), axis=0)
        assert_equal(y5, [0, 0, 1, 1, 2, 2, 3, 3])
        y6 = repeat(x4, 2, axis=0)
        assert_equal(y5, y6)
        y7 = x4.repeat((2, 2, 2, 2), axis=0)
        assert_equal(y5, y7)
        y8 = x4.repeat(2, 0)
        assert_equal(y5, y8)

        y9 = x4.copy()
        assert_equal(y9._data, x4._data)
        assert_equal(y9._mask, x4._mask)

        x = masked_array([1, 2, 3], mask=[0, 1, 0])
        # Copy is False by default
        y = masked_array(x)
        assert_equal(y._data.ctypes.data, x._data.ctypes.data)
        assert_equal(y._mask.ctypes.data, x._mask.ctypes.data)
        y = masked_array(x, copy=True)
        assert_not_equal(y._data.ctypes.data, x._data.ctypes.data)
        assert_not_equal(y._mask.ctypes.data, x._mask.ctypes.data)

    def test_copy_0d(self):
        # gh-9430
        x = np.ma.array(43, mask=True)
        xc = x.copy()
        assert_equal(xc.mask, True)

    def test_copy_on_python_builtins(self):
        # Tests copy works on python builtins (issue#8019)
        assert_(isMaskedArray(np.ma.copy([1,2,3])))
        assert_(isMaskedArray(np.ma.copy((1,2,3))))

    def test_copy_immutable(self):
        # Tests that the copy method is immutable, GitHub issue #5247
        a = np.ma.array([1, 2, 3])
        b = np.ma.array([4, 5, 6])
        a_copy_method = a.copy
        b.copy
        assert_equal(a_copy_method(), [1, 2, 3])

    def test_deepcopy(self):
        from copy import deepcopy
        a = array([0, 1, 2], mask=[False, True, False])
        copied = deepcopy(a)
        assert_equal(copied.mask, a.mask)
        assert_not_equal(id(a._mask), id(copied._mask))

        copied[1] = 1
        assert_equal(copied.mask, [0, 0, 0])
        assert_equal(a.mask, [0, 1, 0])

        copied = deepcopy(a)
        assert_equal(copied.mask, a.mask)
        copied.mask[1] = False
        assert_equal(copied.mask, [0, 0, 0])
        assert_equal(a.mask, [0, 1, 0])

    def test_format(self):
        a = array([0, 1, 2], mask=[False, True, False])
        assert_equal(format(a), "[0 -- 2]")
        assert_equal(format(masked), "--")
        assert_equal(format(masked, ""), "--")

        # Postponed from PR #15410, perhaps address in the future.
        # assert_equal(format(masked, " >5"), "   --")
        # assert_equal(format(masked, " <5"), "--   ")

        # Expect a FutureWarning for using format_spec with MaskedElement
        with assert_warns(FutureWarning):
            with_format_string = format(masked, " >5")
        assert_equal(with_format_string, "--")

    def test_str_repr(self):
        a = array([0, 1, 2], mask=[False, True, False])
        assert_equal(str(a), '[0 -- 2]')
        assert_equal(
            repr(a),
            textwrap.dedent('''\
            masked_array(data=[0, --, 2],
                         mask=[False,  True, False],
                   fill_value=999999)''')
        )

        # arrays with a continuation
        a = np.ma.arange(2000)
        a[1:50] = np.ma.masked
        assert_equal(
            repr(a),
            textwrap.dedent('''\
            masked_array(data=[0, --, --, ..., 1997, 1998, 1999],
                         mask=[False,  True,  True, ..., False, False, False],
                   fill_value=999999)''')
        )

        # line-wrapped 1d arrays are correctly aligned
        a = np.ma.arange(20)
        assert_equal(
            repr(a),
            textwrap.dedent('''\
            masked_array(data=[ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13,
                               14, 15, 16, 17, 18, 19],
                         mask=False,
                   fill_value=999999)''')
        )

        # 2d arrays cause wrapping
        a = array([[1, 2, 3], [4, 5, 6]], dtype=np.int8)
        a[1,1] = np.ma.masked
        assert_equal(
            repr(a),
            textwrap.dedent('''\
            masked_array(
              data=[[1, 2, 3],
                    [4, --, 6]],
              mask=[[False, False, False],
                    [False,  True, False]],
              fill_value=999999,
              dtype=int8)''')
        )

        # but not it they're a row vector
        assert_equal(
            repr(a[:1]),
            textwrap.dedent('''\
            masked_array(data=[[1, 2, 3]],
                         mask=[[False, False, False]],
                   fill_value=999999,
                        dtype=int8)''')
        )

        # dtype=int is implied, so not shown
        assert_equal(
            repr(a.astype(int)),
            textwrap.dedent('''\
            masked_array(
              data=[[1, 2, 3],
                    [4, --, 6]],
              mask=[[False, False, False],
                    [False,  True, False]],
              fill_value=999999)''')
        )

    def test_str_repr_legacy(self):
        oldopts = np.get_printoptions()
        np.set_printoptions(legacy='1.13')
        try:
            a = array([0, 1, 2], mask=[False, True, False])
            assert_equal(str(a), '[0 -- 2]')
            assert_equal(repr(a), 'masked_array(data = [0 -- 2],\n'
                                  '             mask = [False  True False],\n'
                                  '       fill_value = 999999)\n')

            a = np.ma.arange(2000)
            a[1:50] = np.ma.masked
            assert_equal(
                repr(a),
                'masked_array(data = [0 -- -- ..., 1997 1998 1999],\n'
                '             mask = [False  True  True ..., False False False],\n'
                '       fill_value = 999999)\n'
            )
        finally:
            np.set_printoptions(**oldopts)

    def test_0d_unicode(self):
        u = 'caf\xe9'
        utype = type(u)

        arr_nomask = np.ma.array(u)
        arr_masked = np.ma.array(u, mask=True)

        assert_equal(utype(arr_nomask), u)
        assert_equal(utype(arr_masked), '--')

    def test_pickling(self):
        # Tests pickling
        for dtype in (int, float, str, object):
            a = arange(10).astype(dtype)
            a.fill_value = 999

            masks = ([0, 0, 0, 1, 0, 1, 0, 1, 0, 1],  # partially masked
                     True,                            # Fully masked
                     False)                           # Fully unmasked

            for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
                for mask in masks:
                    a.mask = mask
                    a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
                    assert_equal(a_pickled._mask, a._mask)
                    assert_equal(a_pickled._data, a._data)
                    if dtype in (object, int):
                        assert_equal(a_pickled.fill_value, 999)
                    else:
                        assert_equal(a_pickled.fill_value, dtype(999))
                    assert_array_equal(a_pickled.mask, mask)

    def test_pickling_subbaseclass(self):
        # Test pickling w/ a subclass of ndarray
        x = np.array([(1.0, 2), (3.0, 4)],
                     dtype=[('x', float), ('y', int)]).view(np.recarray)
        a = masked_array(x, mask=[(True, False), (False, True)])
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
            assert_equal(a_pickled._mask, a._mask)
            assert_equal(a_pickled, a)
            assert_(isinstance(a_pickled._data, np.recarray))

    def test_pickling_maskedconstant(self):
        # Test pickling MaskedConstant
        mc = np.ma.masked
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            mc_pickled = pickle.loads(pickle.dumps(mc, protocol=proto))
            assert_equal(mc_pickled._baseclass, mc._baseclass)
            assert_equal(mc_pickled._mask, mc._mask)
            assert_equal(mc_pickled._data, mc._data)

    def test_pickling_wstructured(self):
        # Tests pickling w/ structured array
        a = array([(1, 1.), (2, 2.)], mask=[(0, 0), (0, 1)],
                  dtype=[('a', int), ('b', float)])
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
            assert_equal(a_pickled._mask, a._mask)
            assert_equal(a_pickled, a)

    def test_pickling_keepalignment(self):
        # Tests pickling w/ F_CONTIGUOUS arrays
        a = arange(10)
        a.shape = (-1, 2)
        b = a.T
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            test = pickle.loads(pickle.dumps(b, protocol=proto))
            assert_equal(test, b)

    def test_single_element_subscript(self):
        # Tests single element subscripts of Maskedarrays.
        a = array([1, 3, 2])
        b = array([1, 3, 2], mask=[1, 0, 1])
        assert_equal(a[0].shape, ())
        assert_equal(b[0].shape, ())
        assert_equal(b[1].shape, ())

    def test_topython(self):
        # Tests some communication issues with Python.
        assert_equal(1, int(array(1)))
        assert_equal(1.0, float(array(1)))
        assert_equal(1, int(array([[[1]]])))
        assert_equal(1.0, float(array([[1]])))
        assert_raises(TypeError, float, array([1, 1]))

        with suppress_warnings() as sup:
            sup.filter(UserWarning, 'Warning: converting a masked element')
            assert_(np.isnan(float(array([1], mask=[1]))))

            a = array([1, 2, 3], mask=[1, 0, 0])
            assert_raises(TypeError, lambda: float(a))
            assert_equal(float(a[-1]), 3.)
            assert_(np.isnan(float(a[0])))
        assert_raises(TypeError, int, a)
        assert_equal(int(a[-1]), 3)
        assert_raises(MAError, lambda:int(a[0]))

    def test_oddfeatures_1(self):
        # Test of other odd features
        x = arange(20)
        x = x.reshape(4, 5)
        x.flat[5] = 12
        assert_(x[1, 0] == 12)
        z = x + 10j * x
        assert_equal(z.real, x)
        assert_equal(z.imag, 10 * x)
        assert_equal((z * conjugate(z)).real, 101 * x * x)
        z.imag[...] = 0.0

        x = arange(10)
        x[3] = masked
        assert_(str(x[3]) == str(masked))
        c = x >= 8
        assert_(count(where(c, masked, masked)) == 0)
        assert_(shape(where(c, masked, masked)) == c.shape)

        z = masked_where(c, x)
        assert_(z.dtype is x.dtype)
        assert_(z[3] is masked)
        assert_(z[4] is not masked)
        assert_(z[7] is not masked)
        assert_(z[8] is masked)
        assert_(z[9] is masked)
        assert_equal(x, z)

    def test_oddfeatures_2(self):
        # Tests some more features.
        x = array([1., 2., 3., 4., 5.])
        c = array([1, 1, 1, 0, 0])
        x[2] = masked
        z = where(c, x, -x)
        assert_equal(z, [1., 2., 0., -4., -5])
        c[0] = masked
        z = where(c, x, -x)
        assert_equal(z, [1., 2., 0., -4., -5])
        assert_(z[0] is masked)
        assert_(z[1] is not masked)
        assert_(z[2] is masked)

    @suppress_copy_mask_on_assignment
    def test_oddfeatures_3(self):
        # Tests some generic features
        atest = array([10], mask=True)
        btest = array([20])
        idx = atest.mask
        atest[idx] = btest[idx]
        assert_equal(atest, [20])

    def test_filled_with_object_dtype(self):
        a = np.ma.masked_all(1, dtype='O')
        assert_equal(a.filled('x')[0], 'x')

    def test_filled_with_flexible_dtype(self):
        # Test filled w/ flexible dtype
        flexi = array([(1, 1, 1)],
                      dtype=[('i', int), ('s', '|S8'), ('f', float)])
        flexi[0] = masked
        assert_equal(flexi.filled(),
                     np.array([(default_fill_value(0),
                                default_fill_value('0'),
                                default_fill_value(0.),)], dtype=flexi.dtype))
        flexi[0] = masked
        assert_equal(flexi.filled(1),
                     np.array([(1, '1', 1.)], dtype=flexi.dtype))

    def test_filled_with_mvoid(self):
        # Test filled w/ mvoid
        ndtype = [('a', int), ('b', float)]
        a = mvoid((1, 2.), mask=[(0, 1)], dtype=ndtype)
        # Filled using default
        test = a.filled()
        assert_equal(tuple(test), (1, default_fill_value(1.)))
        # Explicit fill_value
        test = a.filled((-1, -1))
        assert_equal(tuple(test), (1, -1))
        # Using predefined filling values
        a.fill_value = (-999, -999)
        assert_equal(tuple(a.filled()), (1, -999))

    def test_filled_with_nested_dtype(self):
        # Test filled w/ nested dtype
        ndtype = [('A', int), ('B', [('BA', int), ('BB', int)])]
        a = array([(1, (1, 1)), (2, (2, 2))],
                  mask=[(0, (1, 0)), (0, (0, 1))], dtype=ndtype)
        test = a.filled(0)
        control = np.array([(1, (0, 1)), (2, (2, 0))], dtype=ndtype)
        assert_equal(test, control)

        test = a['B'].filled(0)
        control = np.array([(0, 1), (2, 0)], dtype=a['B'].dtype)
        assert_equal(test, control)

        # test if mask gets set correctly (see #6760)
        Z = numpy.ma.zeros(2, numpy.dtype([("A", "(2,2)i1,(2,2)i1", (2,2))]))
        assert_equal(Z.data.dtype, numpy.dtype([('A', [('f0', 'i1', (2, 2)),
                                          ('f1', 'i1', (2, 2))], (2, 2))]))
        assert_equal(Z.mask.dtype, numpy.dtype([('A', [('f0', '?', (2, 2)),
                                          ('f1', '?', (2, 2))], (2, 2))]))

    def test_filled_with_f_order(self):
        # Test filled w/ F-contiguous array
        a = array(np.array([(0, 1, 2), (4, 5, 6)], order='F'),
                  mask=np.array([(0, 0, 1), (1, 0, 0)], order='F'),
                  order='F')  # this is currently ignored
        assert_(a.flags['F_CONTIGUOUS'])
        assert_(a.filled(0).flags['F_CONTIGUOUS'])

    def test_optinfo_propagation(self):
        # Checks that _optinfo dictionary isn't back-propagated
        x = array([1, 2, 3, ], dtype=float)
        x._optinfo['info'] = '???'
        y = x.copy()
        assert_equal(y._optinfo['info'], '???')
        y._optinfo['info'] = '!!!'
        assert_equal(x._optinfo['info'], '???')

    def test_optinfo_forward_propagation(self):
        a = array([1,2,2,4])
        a._optinfo["key"] = "value"
        assert_equal(a._optinfo["key"], (a == 2)._optinfo["key"])
        assert_equal(a._optinfo["key"], (a != 2)._optinfo["key"])
        assert_equal(a._optinfo["key"], (a > 2)._optinfo["key"])
        assert_equal(a._optinfo["key"], (a >= 2)._optinfo["key"])
        assert_equal(a._optinfo["key"], (a <= 2)._optinfo["key"])
        assert_equal(a._optinfo["key"], (a + 2)._optinfo["key"])
        assert_equal(a._optinfo["key"], (a - 2)._optinfo["key"])
        assert_equal(a._optinfo["key"], (a * 2)._optinfo["key"])
        assert_equal(a._optinfo["key"], (a / 2)._optinfo["key"])
        assert_equal(a._optinfo["key"], a[:2]._optinfo["key"])
        assert_equal(a._optinfo["key"], a[[0,0,2]]._optinfo["key"])
        assert_equal(a._optinfo["key"], np.exp(a)._optinfo["key"])
        assert_equal(a._optinfo["key"], np.abs(a)._optinfo["key"])
        assert_equal(a._optinfo["key"], array(a, copy=True)._optinfo["key"])
        assert_equal(a._optinfo["key"], np.zeros_like(a)._optinfo["key"])

    def test_fancy_printoptions(self):
        # Test printing a masked array w/ fancy dtype.
        fancydtype = np.dtype([('x', int), ('y', [('t', int), ('s', float)])])
        test = array([(1, (2, 3.0)), (4, (5, 6.0))],
                     mask=[(1, (0, 1)), (0, (1, 0))],
                     dtype=fancydtype)
        control = "[(--, (2, --)) (4, (--, 6.0))]"
        assert_equal(str(test), control)

        # Test 0-d array with multi-dimensional dtype
        t_2d0 = masked_array(data = (0, [[0.0, 0.0, 0.0],
                                        [0.0, 0.0, 0.0]],
                                    0.0),
                             mask = (False, [[True, False, True],
                                             [False, False, True]],
                                     False),
                             dtype = "int, (2,3)float, float")
        control = "(0, [[--, 0.0, --], [0.0, 0.0, --]], 0.0)"
        assert_equal(str(t_2d0), control)

    def test_flatten_structured_array(self):
        # Test flatten_structured_array on arrays
        # On ndarray
        ndtype = [('a', int), ('b', float)]
        a = np.array([(1, 1), (2, 2)], dtype=ndtype)
        test = flatten_structured_array(a)
        control = np.array([[1., 1.], [2., 2.]], dtype=float)
        assert_equal(test, control)
        assert_equal(test.dtype, control.dtype)
        # On masked_array
        a = array([(1, 1), (2, 2)], mask=[(0, 1), (1, 0)], dtype=ndtype)
        test = flatten_structured_array(a)
        control = array([[1., 1.], [2., 2.]],
                        mask=[[0, 1], [1, 0]], dtype=float)
        assert_equal(test, control)
        assert_equal(test.dtype, control.dtype)
        assert_equal(test.mask, control.mask)
        # On masked array with nested structure
        ndtype = [('a', int), ('b', [('ba', int), ('bb', float)])]
        a = array([(1, (1, 1.1)), (2, (2, 2.2))],
                  mask=[(0, (1, 0)), (1, (0, 1))], dtype=ndtype)
        test = flatten_structured_array(a)
        control = array([[1., 1., 1.1], [2., 2., 2.2]],
                        mask=[[0, 1, 0], [1, 0, 1]], dtype=float)
        assert_equal(test, control)
        assert_equal(test.dtype, control.dtype)
        assert_equal(test.mask, control.mask)
        # Keeping the initial shape
        ndtype = [('a', int), ('b', float)]
        a = np.array([[(1, 1), ], [(2, 2), ]], dtype=ndtype)
        test = flatten_structured_array(a)
        control = np.array([[[1., 1.], ], [[2., 2.], ]], dtype=float)
        assert_equal(test, control)
        assert_equal(test.dtype, control.dtype)

    def test_void0d(self):
        # Test creating a mvoid object
        ndtype = [('a', int), ('b', int)]
        a = np.array([(1, 2,)], dtype=ndtype)[0]
        f = mvoid(a)
        assert_(isinstance(f, mvoid))

        a = masked_array([(1, 2)], mask=[(1, 0)], dtype=ndtype)[0]
        assert_(isinstance(a, mvoid))

        a = masked_array([(1, 2), (1, 2)], mask=[(1, 0), (0, 0)], dtype=ndtype)
        f = mvoid(a._data[0], a._mask[0])
        assert_(isinstance(f, mvoid))

    def test_mvoid_getitem(self):
        # Test mvoid.__getitem__
        ndtype = [('a', int), ('b', int)]
        a = masked_array([(1, 2,), (3, 4)], mask=[(0, 0), (1, 0)],
                         dtype=ndtype)
        # w/o mask
        f = a[0]
        assert_(isinstance(f, mvoid))
        assert_equal((f[0], f['a']), (1, 1))
        assert_equal(f['b'], 2)
        # w/ mask
        f = a[1]
        assert_(isinstance(f, mvoid))
        assert_(f[0] is masked)
        assert_(f['a'] is masked)
        assert_equal(f[1], 4)

        # exotic dtype
        A = masked_array(data=[([0,1],)],
                         mask=[([True, False],)],
                         dtype=[("A", ">i2", (2,))])
        assert_equal(A[0]["A"], A["A"][0])
        assert_equal(A[0]["A"], masked_array(data=[0, 1],
                         mask=[True, False], dtype=">i2"))

    def test_mvoid_iter(self):
        # Test iteration on __getitem__
        ndtype = [('a', int), ('b', int)]
        a = masked_array([(1, 2,), (3, 4)], mask=[(0, 0), (1, 0)],
                         dtype=ndtype)
        # w/o mask
        assert_equal(list(a[0]), [1, 2])
        # w/ mask
        assert_equal(list(a[1]), [masked, 4])

    def test_mvoid_print(self):
        # Test printing a mvoid
        mx = array([(1, 1), (2, 2)], dtype=[('a', int), ('b', int)])
        assert_equal(str(mx[0]), "(1, 1)")
        mx['b'][0] = masked
        ini_display = masked_print_option._display
        masked_print_option.set_display("-X-")
        try:
            assert_equal(str(mx[0]), "(1, -X-)")
            assert_equal(repr(mx[0]), "(1, -X-)")
        finally:
            masked_print_option.set_display(ini_display)

        # also check if there are object datatypes (see gh-7493)
        mx = array([(1,), (2,)], dtype=[('a', 'O')])
        assert_equal(str(mx[0]), "(1,)")

    def test_mvoid_multidim_print(self):

        # regression test for gh-6019
        t_ma = masked_array(data = [([1, 2, 3],)],
                            mask = [([False, True, False],)],
                            fill_value = ([999999, 999999, 999999],),
                            dtype = [('a', '<i4', (3,))])
        assert_(str(t_ma[0]) == "([1, --, 3],)")
        assert_(repr(t_ma[0]) == "([1, --, 3],)")

        # additional tests with structured arrays

        t_2d = masked_array(data = [([[1, 2], [3,4]],)],
                            mask = [([[False, True], [True, False]],)],
                            dtype = [('a', '<i4', (2,2))])
        assert_(str(t_2d[0]) == "([[1, --], [--, 4]],)")
        assert_(repr(t_2d[0]) == "([[1, --], [--, 4]],)")

        t_0d = masked_array(data = [(1,2)],
                            mask = [(True,False)],
                            dtype = [('a', '<i4'), ('b', '<i4')])
        assert_(str(t_0d[0]) == "(--, 2)")
        assert_(repr(t_0d[0]) == "(--, 2)")

        t_2d = masked_array(data = [([[1, 2], [3,4]], 1)],
                            mask = [([[False, True], [True, False]], False)],
                            dtype = [('a', '<i4', (2,2)), ('b', float)])
        assert_(str(t_2d[0]) == "([[1, --], [--, 4]], 1.0)")
        assert_(repr(t_2d[0]) == "([[1, --], [--, 4]], 1.0)")

        t_ne = masked_array(data=[(1, (1, 1))],
                            mask=[(True, (True, False))],
                            dtype = [('a', '<i4'), ('b', 'i4,i4')])
        assert_(str(t_ne[0]) == "(--, (--, 1))")
        assert_(repr(t_ne[0]) == "(--, (--, 1))")

    def test_object_with_array(self):
        mx1 = masked_array([1.], mask=[True])
        mx2 = masked_array([1., 2.])
        mx = masked_array([mx1, mx2], mask=[False, True], dtype=object)
        assert_(mx[0] is mx1)
        assert_(mx[1] is not mx2)
        assert_(np.all(mx[1].data == mx2.data))
        assert_(np.all(mx[1].mask))
        # check that we return a view.
        mx[1].data[0] = 0.
        assert_(mx2[0] == 0.)


class TestMaskedArrayArithmetic:
    # Base test class for MaskedArrays.

    def setup_method(self):
        # Base data definition.
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        a10 = 10.
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = masked_array(x, mask=m1)
        ym = masked_array(y, mask=m2)
        z = np.array([-.5, 0., .5, .8])
        zm = masked_array(z, mask=[0, 1, 0, 0])
        xf = np.where(m1, 1e+20, x)
        xm.set_fill_value(1e+20)
        self.d = (x, y, a10, m1, m2, xm, ym, z, zm, xf)
        self.err_status = np.geterr()
        np.seterr(divide='ignore', invalid='ignore')

    def teardown_method(self):
        np.seterr(**self.err_status)

    def test_basic_arithmetic(self):
        # Test of basic arithmetic.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        a2d = array([[1, 2], [0, 4]])
        a2dm = masked_array(a2d, [[0, 0], [1, 0]])
        assert_equal(a2d * a2d, a2d * a2dm)
        assert_equal(a2d + a2d, a2d + a2dm)
        assert_equal(a2d - a2d, a2d - a2dm)
        for s in [(12,), (4, 3), (2, 6)]:
            x = x.reshape(s)
            y = y.reshape(s)
            xm = xm.reshape(s)
            ym = ym.reshape(s)
            xf = xf.reshape(s)
            assert_equal(-x, -xm)
            assert_equal(x + y, xm + ym)
            assert_equal(x - y, xm - ym)
            assert_equal(x * y, xm * ym)
            assert_equal(x / y, xm / ym)
            assert_equal(a10 + y, a10 + ym)
            assert_equal(a10 - y, a10 - ym)
            assert_equal(a10 * y, a10 * ym)
            assert_equal(a10 / y, a10 / ym)
            assert_equal(x + a10, xm + a10)
            assert_equal(x - a10, xm - a10)
            assert_equal(x * a10, xm * a10)
            assert_equal(x / a10, xm / a10)
            assert_equal(x ** 2, xm ** 2)
            assert_equal(abs(x) ** 2.5, abs(xm) ** 2.5)
            assert_equal(x ** y, xm ** ym)
            assert_equal(np.add(x, y), add(xm, ym))
            assert_equal(np.subtract(x, y), subtract(xm, ym))
            assert_equal(np.multiply(x, y), multiply(xm, ym))
            assert_equal(np.divide(x, y), divide(xm, ym))

    def test_divide_on_different_shapes(self):
        x = arange(6, dtype=float)
        x.shape = (2, 3)
        y = arange(3, dtype=float)

        z = x / y
        assert_equal(z, [[-1., 1., 1.], [-1., 4., 2.5]])
        assert_equal(z.mask, [[1, 0, 0], [1, 0, 0]])

        z = x / y[None,:]
        assert_equal(z, [[-1., 1., 1.], [-1., 4., 2.5]])
        assert_equal(z.mask, [[1, 0, 0], [1, 0, 0]])

        y = arange(2, dtype=float)
        z = x / y[:, None]
        assert_equal(z, [[-1., -1., -1.], [3., 4., 5.]])
        assert_equal(z.mask, [[1, 1, 1], [0, 0, 0]])

    def test_mixed_arithmetic(self):
        # Tests mixed arithmetic.
        na = np.array([1])
        ma = array([1])
        assert_(isinstance(na + ma, MaskedArray))
        assert_(isinstance(ma + na, MaskedArray))

    def test_limits_arithmetic(self):
        tiny = np.finfo(float).tiny
        a = array([tiny, 1. / tiny, 0.])
        assert_equal(getmaskarray(a / 2), [0, 0, 0])
        assert_equal(getmaskarray(2 / a), [1, 0, 1])

    def test_masked_singleton_arithmetic(self):
        # Tests some scalar arithmetic on MaskedArrays.
        # Masked singleton should remain masked no matter what
        xm = array(0, mask=1)
        assert_((1 / array(0)).mask)
        assert_((1 + xm).mask)
        assert_((-xm).mask)
        assert_(maximum(xm, xm).mask)
        assert_(minimum(xm, xm).mask)

    def test_masked_singleton_equality(self):
        # Tests (in)equality on masked singleton
        a = array([1, 2, 3], mask=[1, 1, 0])
        assert_((a[0] == 0) is masked)
        assert_((a[0] != 0) is masked)
        assert_equal((a[-1] == 0), False)
        assert_equal((a[-1] != 0), True)

    def test_arithmetic_with_masked_singleton(self):
        # Checks that there's no collapsing to masked
        x = masked_array([1, 2])
        y = x * masked
        assert_equal(y.shape, x.shape)
        assert_equal(y._mask, [True, True])
        y = x[0] * masked
        assert_(y is masked)
        y = x + masked
        assert_equal(y.shape, x.shape)
        assert_equal(y._mask, [True, True])

    def test_arithmetic_with_masked_singleton_on_1d_singleton(self):
        # Check that we're not losing the shape of a singleton
        x = masked_array([1, ])
        y = x + masked
        assert_equal(y.shape, x.shape)
        assert_equal(y.mask, [True, ])

    def test_scalar_arithmetic(self):
        x = array(0, mask=0)
        assert_equal(x.filled().ctypes.data, x.ctypes.data)
        # Make sure we don't lose the shape in some circumstances
        xm = array((0, 0)) / 0.
        assert_equal(xm.shape, (2,))
        assert_equal(xm.mask, [1, 1])

    def test_basic_ufuncs(self):
        # Test various functions such as sin, cos.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        assert_equal(np.cos(x), cos(xm))
        assert_equal(np.cosh(x), cosh(xm))
        assert_equal(np.sin(x), sin(xm))
        assert_equal(np.sinh(x), sinh(xm))
        assert_equal(np.tan(x), tan(xm))
        assert_equal(np.tanh(x), tanh(xm))
        assert_equal(np.sqrt(abs(x)), sqrt(xm))
        assert_equal(np.log(abs(x)), log(xm))
        assert_equal(np.log10(abs(x)), log10(xm))
        assert_equal(np.exp(x), exp(xm))
        assert_equal(np.arcsin(z), arcsin(zm))
        assert_equal(np.arccos(z), arccos(zm))
        assert_equal(np.arctan(z), arctan(zm))
        assert_equal(np.arctan2(x, y), arctan2(xm, ym))
        assert_equal(np.absolute(x), absolute(xm))
        assert_equal(np.angle(x + 1j*y), angle(xm + 1j*ym))
        assert_equal(np.angle(x + 1j*y, deg=True), angle(xm + 1j*ym, deg=True))
        assert_equal(np.equal(x, y), equal(xm, ym))
        assert_equal(np.not_equal(x, y), not_equal(xm, ym))
        assert_equal(np.less(x, y), less(xm, ym))
        assert_equal(np.greater(x, y), greater(xm, ym))
        assert_equal(np.less_equal(x, y), less_equal(xm, ym))
        assert_equal(np.greater_equal(x, y), greater_equal(xm, ym))
        assert_equal(np.conjugate(x), conjugate(xm))

    def test_count_func(self):
        # Tests count
        assert_equal(1, count(1))
        assert_equal(0, array(1, mask=[1]))

        ott = array([0., 1., 2., 3.], mask=[1, 0, 0, 0])
        res = count(ott)
        assert_(res.dtype.type is np.intp)
        assert_equal(3, res)

        ott = ott.reshape((2, 2))
        res = count(ott)
        assert_(res.dtype.type is np.intp)
        assert_equal(3, res)
        res = count(ott, 0)
        assert_(isinstance(res, ndarray))
        assert_equal([1, 2], res)
        assert_(getmask(res) is nomask)

        ott = array([0., 1., 2., 3.])
        res = count(ott, 0)
        assert_(isinstance(res, ndarray))
        assert_(res.dtype.type is np.intp)
        assert_raises(np.AxisError, ott.count, axis=1)

    def test_count_on_python_builtins(self):
        # Tests count works on python builtins (issue#8019)
        assert_equal(3, count([1,2,3]))
        assert_equal(2, count((1,2)))

    def test_minmax_func(self):
        # Tests minimum and maximum.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        # max doesn't work if shaped
        xr = np.ravel(x)
        xmr = ravel(xm)
        # following are true because of careful selection of data
        assert_equal(max(xr), maximum.reduce(xmr))
        assert_equal(min(xr), minimum.reduce(xmr))

        assert_equal(minimum([1, 2, 3], [4, 0, 9]), [1, 0, 3])
        assert_equal(maximum([1, 2, 3], [4, 0, 9]), [4, 2, 9])
        x = arange(5)
        y = arange(5) - 2
        x[3] = masked
        y[0] = masked
        assert_equal(minimum(x, y), where(less(x, y), x, y))
        assert_equal(maximum(x, y), where(greater(x, y), x, y))
        assert_(minimum.reduce(x) == 0)
        assert_(maximum.reduce(x) == 4)

        x = arange(4).reshape(2, 2)
        x[-1, -1] = masked
        assert_equal(maximum.reduce(x, axis=None), 2)

    def test_minimummaximum_func(self):
        a = np.ones((2, 2))
        aminimum = minimum(a, a)
        assert_(isinstance(aminimum, MaskedArray))
        assert_equal(aminimum, np.minimum(a, a))

        aminimum = minimum.outer(a, a)
        assert_(isinstance(aminimum, MaskedArray))
        assert_equal(aminimum, np.minimum.outer(a, a))

        amaximum = maximum(a, a)
        assert_(isinstance(amaximum, MaskedArray))
        assert_equal(amaximum, np.maximum(a, a))

        amaximum = maximum.outer(a, a)
        assert_(isinstance(amaximum, MaskedArray))
        assert_equal(amaximum, np.maximum.outer(a, a))

    def test_minmax_reduce(self):
        # Test np.min/maximum.reduce on array w/ full False mask
        a = array([1, 2, 3], mask=[False, False, False])
        b = np.maximum.reduce(a)
        assert_equal(b, 3)

    def test_minmax_funcs_with_output(self):
        # Tests the min/max functions with explicit outputs
        mask = np.random.rand(12).round()
        xm = array(np.random.uniform(0, 10, 12), mask=mask)
        xm.shape = (3, 4)
        for funcname in ('min', 'max'):
            # Initialize
            npfunc = getattr(np, funcname)
            mafunc = getattr(numpy.ma.core, funcname)
            # Use the np version
            nout = np.empty((4,), dtype=int)
            try:
                result = npfunc(xm, axis=0, out=nout)
            except MaskError:
                pass
            nout = np.empty((4,), dtype=float)
            result = npfunc(xm, axis=0, out=nout)
            assert_(result is nout)
            # Use the ma version
            nout.fill(-999)
            result = mafunc(xm, axis=0, out=nout)
            assert_(result is nout)

    def test_minmax_methods(self):
        # Additional tests on max/min
        (_, _, _, _, _, xm, _, _, _, _) = self.d
        xm.shape = (xm.size,)
        assert_equal(xm.max(), 10)
        assert_(xm[0].max() is masked)
        assert_(xm[0].max(0) is masked)
        assert_(xm[0].max(-1) is masked)
        assert_equal(xm.min(), -10.)
        assert_(xm[0].min() is masked)
        assert_(xm[0].min(0) is masked)
        assert_(xm[0].min(-1) is masked)
        assert_equal(xm.ptp(), 20.)
        assert_(xm[0].ptp() is masked)
        assert_(xm[0].ptp(0) is masked)
        assert_(xm[0].ptp(-1) is masked)

        x = array([1, 2, 3], mask=True)
        assert_(x.min() is masked)
        assert_(x.max() is masked)
        assert_(x.ptp() is masked)

    def test_minmax_dtypes(self):
        # Additional tests on max/min for non-standard float and complex dtypes
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        a10 = 10.
        an10 = -10.0
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        xm = masked_array(x, mask=m1)
        xm.set_fill_value(1e+20)
        float_dtypes = [np.half, np.single, np.double,
                        np.longdouble, np.cfloat, np.cdouble, np.clongdouble]
        for float_dtype in float_dtypes:
            assert_equal(masked_array(x, mask=m1, dtype=float_dtype).max(),
                         float_dtype(a10))
            assert_equal(masked_array(x, mask=m1, dtype=float_dtype).min(),
                         float_dtype(an10))

        assert_equal(xm.min(), an10)
        assert_equal(xm.max(), a10)

        # Non-complex type only test
        for float_dtype in float_dtypes[:4]:
            assert_equal(masked_array(x, mask=m1, dtype=float_dtype).max(),
                         float_dtype(a10))
            assert_equal(masked_array(x, mask=m1, dtype=float_dtype).min(),
                         float_dtype(an10))

        # Complex types only test
        for float_dtype in float_dtypes[-3:]:
            ym = masked_array([1e20+1j, 1e20-2j, 1e20-1j], mask=[0, 1, 0],
                          dtype=float_dtype)
            assert_equal(ym.min(), float_dtype(1e20-1j))
            assert_equal(ym.max(), float_dtype(1e20+1j))

            zm = masked_array([np.inf+2j, np.inf+3j, -np.inf-1j], mask=[0, 1, 0],
                              dtype=float_dtype)
            assert_equal(zm.min(), float_dtype(-np.inf-1j))
            assert_equal(zm.max(), float_dtype(np.inf+2j))

            cmax = np.inf - 1j * np.finfo(np.float64).max
            assert masked_array([-cmax, 0], mask=[0, 1]).max() == -cmax
            assert masked_array([cmax, 0], mask=[0, 1]).min() == cmax

    def test_addsumprod(self):
        # Tests add, sum, product.
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        assert_equal(np.add.reduce(x), add.reduce(x))
        assert_equal(np.add.accumulate(x), add.accumulate(x))
        assert_equal(4, sum(array(4), axis=0))
        assert_equal(4, sum(array(4), axis=0))
        assert_equal(np.sum(x, axis=0), sum(x, axis=0))
        assert_equal(np.sum(filled(xm, 0), axis=0), sum(xm, axis=0))
        assert_equal(np.sum(x, 0), sum(x, 0))
        assert_equal(np.product(x, axis=0), product(x, axis=0))
        assert_equal(np.product(x, 0), product(x, 0))
        assert_equal(np.product(filled(xm, 1), axis=0), product(xm, axis=0))
        s = (3, 4)
        x.shape = y.shape = xm.shape = ym.shape = s
        if len(s) > 1:
            assert_equal(np.concatenate((x, y), 1), concatenate((xm, ym), 1))
            assert_equal(np.add.reduce(x, 1), add.reduce(x, 1))
            assert_equal(np.sum(x, 1), sum(x, 1))
            assert_equal(np.product(x, 1), product(x, 1))

    def test_binops_d2D(self):
        # Test binary operations on 2D data
        a = array([[1.], [2.], [3.]], mask=[[False], [True], [True]])
        b = array([[2., 3.], [4., 5.], [6., 7.]])

        test = a * b
        control = array([[2., 3.], [2., 2.], [3., 3.]],
                        mask=[[0, 0], [1, 1], [1, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

        test = b * a
        control = array([[2., 3.], [4., 5.], [6., 7.]],
                        mask=[[0, 0], [1, 1], [1, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

        a = array([[1.], [2.], [3.]])
        b = array([[2., 3.], [4., 5.], [6., 7.]],
                  mask=[[0, 0], [0, 0], [0, 1]])
        test = a * b
        control = array([[2, 3], [8, 10], [18, 3]],
                        mask=[[0, 0], [0, 0], [0, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

        test = b * a
        control = array([[2, 3], [8, 10], [18, 7]],
                        mask=[[0, 0], [0, 0], [0, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

    def test_domained_binops_d2D(self):
        # Test domained binary operations on 2D data
        a = array([[1.], [2.], [3.]], mask=[[False], [True], [True]])
        b = array([[2., 3.], [4., 5.], [6., 7.]])

        test = a / b
        control = array([[1. / 2., 1. / 3.], [2., 2.], [3., 3.]],
                        mask=[[0, 0], [1, 1], [1, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

        test = b / a
        control = array([[2. / 1., 3. / 1.], [4., 5.], [6., 7.]],
                        mask=[[0, 0], [1, 1], [1, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

        a = array([[1.], [2.], [3.]])
        b = array([[2., 3.], [4., 5.], [6., 7.]],
                  mask=[[0, 0], [0, 0], [0, 1]])
        test = a / b
        control = array([[1. / 2, 1. / 3], [2. / 4, 2. / 5], [3. / 6, 3]],
                        mask=[[0, 0], [0, 0], [0, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

        test = b / a
        control = array([[2 / 1., 3 / 1.], [4 / 2., 5 / 2.], [6 / 3., 7]],
                        mask=[[0, 0], [0, 0], [0, 1]])
        assert_equal(test, control)
        assert_equal(test.data, control.data)
        assert_equal(test.mask, control.mask)

    def test_noshrinking(self):
        # Check that we don't shrink a mask when not wanted
        # Binary operations
        a = masked_array([1., 2., 3.], mask=[False, False, False],
                         shrink=False)
        b = a + 1
        assert_equal(b.mask, [0, 0, 0])
        # In place binary operation
        a += 1
        assert_equal(a.mask, [0, 0, 0])
        # Domained binary operation
        b = a / 1.
        assert_equal(b.mask, [0, 0, 0])
        # In place binary operation
        a /= 1.
        assert_equal(a.mask, [0, 0, 0])

    def test_ufunc_nomask(self):
        # check the case ufuncs should set the mask to false
        m = np.ma.array([1])
        # check we don't get array([False], dtype=bool)
        assert_equal(np.true_divide(m, 5).mask.shape, ())

    def test_noshink_on_creation(self):
        # Check that the mask is not shrunk on array creation when not wanted
        a = np.ma.masked_values([1., 2.5, 3.1], 1.5, shrink=False)
        assert_equal(a.mask, [0, 0, 0])

    def test_mod(self):
        # Tests mod
        (x, y, a10, m1, m2, xm, ym, z, zm, xf) = self.d
        assert_equal(mod(x, y), mod(xm, ym))
        test = mod(ym, xm)
        assert_equal(test, np.mod(ym, xm))
        assert_equal(test.mask, mask_or(xm.mask, ym.mask))
        test = mod(xm, ym)
        assert_equal(test, np.mod(xm, ym))
        assert_equal(test.mask, mask_or(mask_or(xm.mask, ym.mask), (ym == 0)))

    def test_TakeTransposeInnerOuter(self):
        # Test of take, transpose, inner, outer products
        x = arange(24)
        y = np.arange(24)
        x[5:6] = masked
        x = x.reshape(2, 3, 4)
        y = y.reshape(2, 3, 4)
        assert_equal(np.transpose(y, (2, 0, 1)), transpose(x, (2, 0, 1)))
        assert_equal(np.take(y, (2, 0, 1), 1), take(x, (2, 0, 1), 1))
        assert_equal(np.inner(filled(x, 0), filled(y, 0)),
                     inner(x, y))
        assert_equal(np.outer(filled(x, 0), filled(y, 0)),
                     outer(x, y))
        y = array(['abc', 1, 'def', 2, 3], object)
        y[2] = masked
        t = take(y, [0, 3, 4])
        assert_(t[0] == 'abc')
        assert_(t[1] == 2)
        assert_(t[2] == 3)

    def test_imag_real(self):
        # Check complex
        xx = array([1 + 10j, 20 + 2j], mask=[1, 0])
        assert_equal(xx.imag, [10, 2])
        assert_equal(xx.imag.filled(), [1e+20, 2])
        assert_equal(xx.imag.dtype, xx._data.imag.dtype)
        assert_equal(xx.real, [1, 20])
        assert_equal(xx.real.filled(), [1e+20, 20])
        assert_equal(xx.real.dtype, xx._data.real.dtype)

    def test_methods_with_output(self):
        xm = array(np.random.uniform(0, 10, 12)).reshape(3, 4)
        xm[:, 0] = xm[0] = xm[-1, -1] = masked

        funclist = ('sum', 'prod', 'var', 'std', 'max', 'min', 'ptp', 'mean',)

        for funcname in funclist:
            npfunc = getattr(np, funcname)
            xmmeth = getattr(xm, funcname)
            # A ndarray as explicit input
            output = np.empty(4, dtype=float)
            output.fill(-9999)
            result = npfunc(xm, axis=0, out=output)
            # ... the result should be the given output
            assert_(result is output)
            assert_equal(result, xmmeth(axis=0, out=output))

            output = empty(4, dtype=int)
            result = xmmeth(axis=0, out=output)
            assert_(result is output)
            assert_(output[0] is masked)

    def test_eq_on_structured(self):
        # Test the equality of structured arrays
        ndtype = [('A', int), ('B', int)]
        a = array([(1, 1), (2, 2)], mask=[(0, 1), (0, 0)], dtype=ndtype)

        test = (a == a)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [False, False])
        assert_(test.fill_value == True)

        test = (a == a[0])
        assert_equal(test.data, [True, False])
        assert_equal(test.mask, [False, False])
        assert_(test.fill_value == True)

        b = array([(1, 1), (2, 2)], mask=[(1, 0), (0, 0)], dtype=ndtype)
        test = (a == b)
        assert_equal(test.data, [False, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

        test = (a[0] == b)
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

        b = array([(1, 1), (2, 2)], mask=[(0, 1), (1, 0)], dtype=ndtype)
        test = (a == b)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [False, False])
        assert_(test.fill_value == True)

        # complicated dtype, 2-dimensional array.
        ndtype = [('A', int), ('B', [('BA', int), ('BB', int)])]
        a = array([[(1, (1, 1)), (2, (2, 2))],
                   [(3, (3, 3)), (4, (4, 4))]],
                  mask=[[(0, (1, 0)), (0, (0, 1))],
                        [(1, (0, 0)), (1, (1, 1))]], dtype=ndtype)
        test = (a[0, 0] == a)
        assert_equal(test.data, [[True, False], [False, False]])
        assert_equal(test.mask, [[False, False], [False, True]])
        assert_(test.fill_value == True)

    def test_ne_on_structured(self):
        # Test the equality of structured arrays
        ndtype = [('A', int), ('B', int)]
        a = array([(1, 1), (2, 2)], mask=[(0, 1), (0, 0)], dtype=ndtype)

        test = (a != a)
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [False, False])
        assert_(test.fill_value == True)

        test = (a != a[0])
        assert_equal(test.data, [False, True])
        assert_equal(test.mask, [False, False])
        assert_(test.fill_value == True)

        b = array([(1, 1), (2, 2)], mask=[(1, 0), (0, 0)], dtype=ndtype)
        test = (a != b)
        assert_equal(test.data, [True, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

        test = (a[0] != b)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

        b = array([(1, 1), (2, 2)], mask=[(0, 1), (1, 0)], dtype=ndtype)
        test = (a != b)
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [False, False])
        assert_(test.fill_value == True)

        # complicated dtype, 2-dimensional array.
        ndtype = [('A', int), ('B', [('BA', int), ('BB', int)])]
        a = array([[(1, (1, 1)), (2, (2, 2))],
                   [(3, (3, 3)), (4, (4, 4))]],
                  mask=[[(0, (1, 0)), (0, (0, 1))],
                        [(1, (0, 0)), (1, (1, 1))]], dtype=ndtype)
        test = (a[0, 0] != a)
        assert_equal(test.data, [[False, True], [True, True]])
        assert_equal(test.mask, [[False, False], [False, True]])
        assert_(test.fill_value == True)

    def test_eq_ne_structured_extra(self):
        # ensure simple examples are symmetric and make sense.
        # from https://github.com/numpy/numpy/pull/8590#discussion_r101126465
        dt = np.dtype('i4,i4')
        for m1 in (mvoid((1, 2), mask=(0, 0), dtype=dt),
                   mvoid((1, 2), mask=(0, 1), dtype=dt),
                   mvoid((1, 2), mask=(1, 0), dtype=dt),
                   mvoid((1, 2), mask=(1, 1), dtype=dt)):
            ma1 = m1.view(MaskedArray)
            r1 = ma1.view('2i4')
            for m2 in (np.array((1, 1), dtype=dt),
                       mvoid((1, 1), dtype=dt),
                       mvoid((1, 0), mask=(0, 1), dtype=dt),
                       mvoid((3, 2), mask=(0, 1), dtype=dt)):
                ma2 = m2.view(MaskedArray)
                r2 = ma2.view('2i4')
                eq_expected = (r1 == r2).all()
                assert_equal(m1 == m2, eq_expected)
                assert_equal(m2 == m1, eq_expected)
                assert_equal(ma1 == m2, eq_expected)
                assert_equal(m1 == ma2, eq_expected)
                assert_equal(ma1 == ma2, eq_expected)
                # Also check it is the same if we do it element by element.
                el_by_el = [m1[name] == m2[name] for name in dt.names]
                assert_equal(array(el_by_el, dtype=bool).all(), eq_expected)
                ne_expected = (r1 != r2).any()
                assert_equal(m1 != m2, ne_expected)
                assert_equal(m2 != m1, ne_expected)
                assert_equal(ma1 != m2, ne_expected)
                assert_equal(m1 != ma2, ne_expected)
                assert_equal(ma1 != ma2, ne_expected)
                el_by_el = [m1[name] != m2[name] for name in dt.names]
                assert_equal(array(el_by_el, dtype=bool).any(), ne_expected)

    @pytest.mark.parametrize('dt', ['S', 'U'])
    @pytest.mark.parametrize('fill', [None, 'A'])
    def test_eq_for_strings(self, dt, fill):
        # Test the equality of structured arrays
        a = array(['a', 'b'], dtype=dt, mask=[0, 1], fill_value=fill)

        test = (a == a)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        test = (a == a[0])
        assert_equal(test.data, [True, False])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        b = array(['a', 'b'], dtype=dt, mask=[1, 0], fill_value=fill)
        test = (a == b)
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, True])
        assert_(test.fill_value == True)

        test = (a[0] == b)
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

        test = (b == a[0])
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

    @pytest.mark.parametrize('dt', ['S', 'U'])
    @pytest.mark.parametrize('fill', [None, 'A'])
    def test_ne_for_strings(self, dt, fill):
        # Test the equality of structured arrays
        a = array(['a', 'b'], dtype=dt, mask=[0, 1], fill_value=fill)

        test = (a != a)
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        test = (a != a[0])
        assert_equal(test.data, [False, True])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        b = array(['a', 'b'], dtype=dt, mask=[1, 0], fill_value=fill)
        test = (a != b)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, True])
        assert_(test.fill_value == True)

        test = (a[0] != b)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

        test = (b != a[0])
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

    @pytest.mark.parametrize('dt1', num_dts, ids=num_ids)
    @pytest.mark.parametrize('dt2', num_dts, ids=num_ids)
    @pytest.mark.parametrize('fill', [None, 1])
    def test_eq_for_numeric(self, dt1, dt2, fill):
        # Test the equality of structured arrays
        a = array([0, 1], dtype=dt1, mask=[0, 1], fill_value=fill)

        test = (a == a)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        test = (a == a[0])
        assert_equal(test.data, [True, False])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        b = array([0, 1], dtype=dt2, mask=[1, 0], fill_value=fill)
        test = (a == b)
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, True])
        assert_(test.fill_value == True)

        test = (a[0] == b)
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

        test = (b == a[0])
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

    @pytest.mark.parametrize('dt1', num_dts, ids=num_ids)
    @pytest.mark.parametrize('dt2', num_dts, ids=num_ids)
    @pytest.mark.parametrize('fill', [None, 1])
    def test_ne_for_numeric(self, dt1, dt2, fill):
        # Test the equality of structured arrays
        a = array([0, 1], dtype=dt1, mask=[0, 1], fill_value=fill)

        test = (a != a)
        assert_equal(test.data, [False, False])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        test = (a != a[0])
        assert_equal(test.data, [False, True])
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        b = array([0, 1], dtype=dt2, mask=[1, 0], fill_value=fill)
        test = (a != b)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, True])
        assert_(test.fill_value == True)

        test = (a[0] != b)
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

        test = (b != a[0])
        assert_equal(test.data, [True, True])
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

    @pytest.mark.parametrize('dt1', num_dts, ids=num_ids)
    @pytest.mark.parametrize('dt2', num_dts, ids=num_ids)
    @pytest.mark.parametrize('fill', [None, 1])
    @pytest.mark.parametrize('op',
            [operator.le, operator.lt, operator.ge, operator.gt])
    def test_comparisons_for_numeric(self, op, dt1, dt2, fill):
        # Test the equality of structured arrays
        a = array([0, 1], dtype=dt1, mask=[0, 1], fill_value=fill)

        test = op(a, a)
        assert_equal(test.data, op(a._data, a._data))
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        test = op(a, a[0])
        assert_equal(test.data, op(a._data, a._data[0]))
        assert_equal(test.mask, [False, True])
        assert_(test.fill_value == True)

        b = array([0, 1], dtype=dt2, mask=[1, 0], fill_value=fill)
        test = op(a, b)
        assert_equal(test.data, op(a._data, b._data))
        assert_equal(test.mask, [True, True])
        assert_(test.fill_value == True)

        test = op(a[0], b)
        assert_equal(test.data, op(a._data[0], b._data))
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

        test = op(b, a[0])
        assert_equal(test.data, op(b._data, a._data[0]))
        assert_equal(test.mask, [True, False])
        assert_(test.fill_value == True)

    @pytest.mark.parametrize('op',
            [operator.le, operator.lt, operator.ge, operator.gt])
    @pytest.mark.parametrize('fill', [None, "N/A"])
    def test_comparisons_strings(self, op, fill):
        # See gh-21770, mask propagation is broken for strings (and some other
        # cases) so we explicitly test strings here.
        # In principle only == and != may need special handling...
        ma1 = masked_array(["a", "b", "cde"], mask=[0, 1, 0], fill_value=fill)
        ma2 = masked_array(["cde", "b", "a"], mask=[0, 1, 0], fill_value=fill)
        assert_equal(op(ma1, ma2)._data, op(ma1._data, ma2._data))

    def test_eq_with_None(self):
        # Really, comparisons with None should not be done, but check them
        # anyway. Note that pep8 will flag these tests.
        # Deprecation is in place for arrays, and when it happens this
        # test will fail (and have to be changed accordingly).

        # With partial mask
        with suppress_warnings() as sup:
            sup.filter(FutureWarning, "Comparison to `None`")
            a = array([None, 1], mask=[0, 1])
            assert_equal(a == None, array([True, False], mask=[0, 1]))
            assert_equal(a.data == None, [True, False])
            assert_equal(a != None, array([False, True], mask=[0, 1]))
            # With nomask
            a = array([None, 1], mask=False)
            assert_equal(a == None, [True, False])
            assert_equal(a != None, [False, True])
            # With complete mask
            a = array([None, 2], mask=True)
            assert_equal(a == None, array([False, True], mask=True))
            assert_equal(a != None, array([True, False], mask=True))
            # Fully masked, even comparison to None should return "masked"
            a = masked
            assert_equal(a == None, masked)

    def test_eq_with_scalar(self):
        a = array(1)
        assert_equal(a == 1, True)
        assert_equal(a == 0, False)
        assert_equal(a != 1, False)
        assert_equal(a != 0, True)
        b = array(1, mask=True)
        assert_equal(b == 0, masked)
        assert_equal(b == 1, masked)
        assert_equal(b != 0, masked)
        assert_equal(b != 1, masked)

    def test_eq_different_dimensions(self):
        m1 = array([1, 1], mask=[0, 1])
        # test comparison with both masked and regular arrays.
        for m2 in (array([[0, 1], [1, 2]]),
                   np.array([[0, 1], [1, 2]])):
            test = (m1 == m2)
            assert_equal(test.data, [[False, False],
                                     [True, False]])
            assert_equal(test.mask, [[False, True],
                                     [False, True]])

    def test_numpyarithmetic(self):
        # Check that the mask is not back-propagated when using numpy functions
        a = masked_array([-1, 0, 1, 2, 3], mask=[0, 0, 0, 0, 1])
        control = masked_array([np.nan, np.nan, 0, np.log(2), -1],
                               mask=[1, 1, 0, 0, 1])

        test = log(a)
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(a.mask, [0, 0, 0, 0, 1])

        test = np.log(a)
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_equal(a.mask, [0, 0, 0, 0, 1])


class TestMaskedArrayAttributes:

    def test_keepmask(self):
        # Tests the keep mask flag
        x = masked_array([1, 2, 3], mask=[1, 0, 0])
        mx = masked_array(x)
        assert_equal(mx.mask, x.mask)
        mx = masked_array(x, mask=[0, 1, 0], keep_mask=False)
        assert_equal(mx.mask, [0, 1, 0])
        mx = masked_array(x, mask=[0, 1, 0], keep_mask=True)
        assert_equal(mx.mask, [1, 1, 0])
        # We default to true
        mx = masked_array(x, mask=[0, 1, 0])
        assert_equal(mx.mask, [1, 1, 0])

    def test_hardmask(self):
        # Test hard_mask
        d = arange(5)
        n = [0, 0, 0, 1, 1]
        m = make_mask(n)
        xh = array(d, mask=m, hard_mask=True)
        # We need to copy, to avoid updating d in xh !
        xs = array(d, mask=m, hard_mask=False, copy=True)
        xh[[1, 4]] = [10, 40]
        xs[[1, 4]] = [10, 40]
        assert_equal(xh._data, [0, 10, 2, 3, 4])
        assert_equal(xs._data, [0, 10, 2, 3, 40])
        assert_equal(xs.mask, [0, 0, 0, 1, 0])
        assert_(xh._hardmask)
        assert_(not xs._hardmask)
        xh[1:4] = [10, 20, 30]
        xs[1:4] = [10, 20, 30]
        assert_equal(xh._data, [0, 10, 20, 3, 4])
        assert_equal(xs._data, [0, 10, 20, 30, 40])
        assert_equal(xs.mask, nomask)
        xh[0] = masked
        xs[0] = masked
        assert_equal(xh.mask, [1, 0, 0, 1, 1])
        assert_equal(xs.mask, [1, 0, 0, 0, 0])
        xh[:] = 1
        xs[:] = 1
        assert_equal(xh._data, [0, 1, 1, 3, 4])
        assert_equal(xs._data, [1, 1, 1, 1, 1])
        assert_equal(xh.mask, [1, 0, 0, 1, 1])
        assert_equal(xs.mask, nomask)
        # Switch to soft mask
        xh.soften_mask()
        xh[:] = arange(5)
        assert_equal(xh._data, [0, 1, 2, 3, 4])
        assert_equal(xh.mask, nomask)
        # Switch back to hard mask
        xh.harden_mask()
        xh[xh < 3] = masked
        assert_equal(xh._data, [0, 1, 2, 3, 4])
        assert_equal(xh._mask, [1, 1, 1, 0, 0])
        xh[filled(xh > 1, False)] = 5
        assert_equal(xh._data, [0, 1, 2, 5, 5])
        assert_equal(xh._mask, [1, 1, 1, 0, 0])

        xh = array([[1, 2], [3, 4]], mask=[[1, 0], [0, 0]], hard_mask=True)
        xh[0] = 0
        assert_equal(xh._data, [[1, 0], [3, 4]])
        assert_equal(xh._mask, [[1, 0], [0, 0]])
        xh[-1, -1] = 5
        assert_equal(xh._data, [[1, 0], [3, 5]])
        assert_equal(xh._mask, [[1, 0], [0, 0]])
        xh[filled(xh < 5, False)] = 2
        assert_equal(xh._data, [[1, 2], [2, 5]])
        assert_equal(xh._mask, [[1, 0], [0, 0]])

    def test_hardmask_again(self):
        # Another test of hardmask
        d = arange(5)
        n = [0, 0, 0, 1, 1]
        m = make_mask(n)
        xh = array(d, mask=m, hard_mask=True)
        xh[4:5] = 999
        xh[0:1] = 999
        assert_equal(xh._data, [999, 1, 2, 3, 4])

    def test_hardmask_oncemore_yay(self):
        # OK, yet another test of hardmask
        # Make sure that harden_mask/soften_mask//unshare_mask returns self
        a = array([1, 2, 3], mask=[1, 0, 0])
        b = a.harden_mask()
        assert_equal(a, b)
        b[0] = 0
        assert_equal(a, b)
        assert_equal(b, array([1, 2, 3], mask=[1, 0, 0]))
        a = b.soften_mask()
        a[0] = 0
        assert_equal(a, b)
        assert_equal(b, array([0, 2, 3], mask=[0, 0, 0]))

    def test_smallmask(self):
        # Checks the behaviour of _smallmask
        a = arange(10)
        a[1] = masked
        a[1] = 1
        assert_equal(a._mask, nomask)
        a = arange(10)
        a._smallmask = False
        a[1] = masked
        a[1] = 1
        assert_equal(a._mask, zeros(10))

    def test_shrink_mask(self):
        # Tests .shrink_mask()
        a = array([1, 2, 3], mask=[0, 0, 0])
        b = a.shrink_mask()
        assert_equal(a, b)
        assert_equal(a.mask, nomask)

        # Mask cannot be shrunk on structured types, so is a no-op
        a = np.ma.array([(1, 2.0)], [('a', int), ('b', float)])
        b = a.copy()
        a.shrink_mask()
        assert_equal(a.mask, b.mask)

    def test_flat(self):
        # Test that flat can return all types of items [#4585, #4615]
        # test 2-D record array
        # ... on structured array w/ masked records
        x = array([[(1, 1.1, 'one'), (2, 2.2, 'two'), (3, 3.3, 'thr')],
                   [(4, 4.4, 'fou'), (5, 5.5, 'fiv'), (6, 6.6, 'six')]],
                  dtype=[('a', int), ('b', float), ('c', '|S8')])
        x['a'][0, 1] = masked
        x['b'][1, 0] = masked
        x['c'][0, 2] = masked
        x[-1, -1] = masked
        xflat = x.flat
        assert_equal(xflat[0], x[0, 0])
        assert_equal(xflat[1], x[0, 1])
        assert_equal(xflat[2], x[0, 2])
        assert_equal(xflat[:3], x[0])
        assert_equal(xflat[3], x[1, 0])
        assert_equal(xflat[4], x[1, 1])
        assert_equal(xflat[5], x[1, 2])
        assert_equal(xflat[3:], x[1])
        assert_equal(xflat[-1], x[-1, -1])
        i = 0
        j = 0
        for xf in xflat:
            assert_equal(xf, x[j, i])
            i += 1
            if i >= x.shape[-1]:
                i = 0
                j += 1

    def test_assign_dtype(self):
        # check that the mask's dtype is updated when dtype is changed
        a = np.zeros(4, dtype='f4,i4')

        m = np.ma.array(a)
        m.dtype = np.dtype('f4')
        repr(m)  # raises?
        assert_equal(m.dtype, np.dtype('f4'))

        # check that dtype changes that change shape of mask too much
        # are not allowed
        def assign():
            m = np.ma.array(a)
            m.dtype = np.dtype('f8')
        assert_raises(ValueError, assign)

        b = a.view(dtype='f4', type=np.ma.MaskedArray)  # raises?
        assert_equal(b.dtype, np.dtype('f4'))

        # check that nomask is preserved
        a = np.zeros(4, dtype='f4')
        m = np.ma.array(a)
        m.dtype = np.dtype('f4,i4')
        assert_equal(m.dtype, np.dtype('f4,i4'))
        assert_equal(m._mask, np.ma.nomask)


class TestFillingValues:

    def test_check_on_scalar(self):
        # Test _check_fill_value set to valid and invalid values
        _check_fill_value = np.ma.core._check_fill_value

        fval = _check_fill_value(0, int)
        assert_equal(fval, 0)
        fval = _check_fill_value(None, int)
        assert_equal(fval, default_fill_value(0))

        fval = _check_fill_value(0, "|S3")
        assert_equal(fval, b"0")
        fval = _check_fill_value(None, "|S3")
        assert_equal(fval, default_fill_value(b"camelot!"))
        assert_raises(TypeError, _check_fill_value, 1e+20, int)
        assert_raises(TypeError, _check_fill_value, 'stuff', int)

    def test_check_on_fields(self):
        # Tests _check_fill_value with records
        _check_fill_value = np.ma.core._check_fill_value
        ndtype = [('a', int), ('b', float), ('c', "|S3")]
        # A check on a list should return a single record
        fval = _check_fill_value([-999, -12345678.9, "???"], ndtype)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), [-999, -12345678.9, b"???"])
        # A check on None should output the defaults
        fval = _check_fill_value(None, ndtype)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), [default_fill_value(0),
                                   default_fill_value(0.),
                                   asbytes(default_fill_value("0"))])
        #.....Using a structured type as fill_value should work
        fill_val = np.array((-999, -12345678.9, "???"), dtype=ndtype)
        fval = _check_fill_value(fill_val, ndtype)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), [-999, -12345678.9, b"???"])

        #.....Using a flexible type w/ a different type shouldn't matter
        # BEHAVIOR in 1.5 and earlier, and 1.13 and later: match structured
        # types by position
        fill_val = np.array((-999, -12345678.9, "???"),
                            dtype=[("A", int), ("B", float), ("C", "|S3")])
        fval = _check_fill_value(fill_val, ndtype)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), [-999, -12345678.9, b"???"])

        #.....Using an object-array shouldn't matter either
        fill_val = np.ndarray(shape=(1,), dtype=object)
        fill_val[0] = (-999, -12345678.9, b"???")
        fval = _check_fill_value(fill_val, object)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), [-999, -12345678.9, b"???"])
        # NOTE: This test was never run properly as "fill_value" rather than
        # "fill_val" was assigned.  Written properly, it fails.
        #fill_val = np.array((-999, -12345678.9, "???"))
        #fval = _check_fill_value(fill_val, ndtype)
        #assert_(isinstance(fval, ndarray))
        #assert_equal(fval.item(), [-999, -12345678.9, b"???"])
        #.....One-field-only flexible type should work as well
        ndtype = [("a", int)]
        fval = _check_fill_value(-999999999, ndtype)
        assert_(isinstance(fval, ndarray))
        assert_equal(fval.item(), (-999999999,))

    def test_fillvalue_conversion(self):
        # Tests the behavior of fill_value during conversion
        # We had a tailored comment to make sure special attributes are
        # properly dealt with
        a = array([b'3', b'4', b'5'])
        a._optinfo.update({'comment':"updated!"})

        b = array(a, dtype=int)
        assert_equal(b._data, [3, 4, 5])
        assert_equal(b.fill_value, default_fill_value(0))

        b = array(a, dtype=float)
        assert_equal(b._data, [3, 4, 5])
        assert_equal(b.fill_value, default_fill_value(0.))

        b = a.astype(int)
        assert_equal(b._data, [3, 4, 5])
        assert_equal(b.fill_value, default_fill_value(0))
        assert_equal(b._optinfo['comment'], "updated!")

        b = a.astype([('a', '|S3')])
        assert_equal(b['a']._data, a._data)
        assert_equal(b['a'].fill_value, a.fill_value)

    def test_default_fill_value(self):
        # check all calling conventions
        f1 = default_fill_value(1.)
        f2 = default_fill_value(np.array(1.))
        f3 = default_fill_value(np.array(1.).dtype)
        assert_equal(f1, f2)
        assert_equal(f1, f3)

    def test_default_fill_value_structured(self):
        fields = array([(1, 1, 1)],
                      dtype=[('i', int), ('s', '|S8'), ('f', float)])

        f1 = default_fill_value(fields)
        f2 = default_fill_value(fields.dtype)
        expected = np.array((default_fill_value(0),
                             default_fill_value('0'),
                             default_fill_value(0.)), dtype=fields.dtype)
        assert_equal(f1, expected)
        assert_equal(f2, expected)

    def test_default_fill_value_void(self):
        dt = np.dtype([('v', 'V7')])
        f = default_fill_value(dt)
        assert_equal(f['v'], np.array(default_fill_value(dt['v']), dt['v']))

    def test_fillvalue(self):
        # Yet more fun with the fill_value
        data = masked_array([1, 2, 3], fill_value=-999)
        series = data[[0, 2, 1]]
        assert_equal(series._fill_value, data._fill_value)

        mtype = [('f', float), ('s', '|S3')]
        x = array([(1, 'a'), (2, 'b'), (pi, 'pi')], dtype=mtype)
        x.fill_value = 999
        assert_equal(x.fill_value.item(), [999., b'999'])
        assert_equal(x['f'].fill_value, 999)
        assert_equal(x['s'].fill_value, b'999')

        x.fill_value = (9, '???')
        assert_equal(x.fill_value.item(), (9, b'???'))
        assert_equal(x['f'].fill_value, 9)
        assert_equal(x['s'].fill_value, b'???')

        x = array([1, 2, 3.1])
        x.fill_value = 999
        assert_equal(np.asarray(x.fill_value).dtype, float)
        assert_equal(x.fill_value, 999.)
        assert_equal(x._fill_value, np.array(999.))

    def test_subarray_fillvalue(self):
        # gh-10483   test multi-field index fill value
        fields = array([(1, 1, 1)],
                      dtype=[('i', int), ('s', '|S8'), ('f', float)])
        with suppress_warnings() as sup:
            sup.filter(FutureWarning, "Numpy has detected")
            subfields = fields[['i', 'f']]
            assert_equal(tuple(subfields.fill_value), (999999, 1.e+20))
            # test comparison does not raise:
            subfields[1:] == subfields[:-1]

    def test_fillvalue_exotic_dtype(self):
        # Tests yet more exotic flexible dtypes
        _check_fill_value = np.ma.core._check_fill_value
        ndtype = [('i', int), ('s', '|S8'), ('f', float)]
        control = np.array((default_fill_value(0),
                            default_fill_value('0'),
                            default_fill_value(0.),),
                           dtype=ndtype)
        assert_equal(_check_fill_value(None, ndtype), control)
        # The shape shouldn't matter
        ndtype = [('f0', float, (2, 2))]
        control = np.array((default_fill_value(0.),),
                           dtype=[('f0', float)]).astype(ndtype)
        assert_equal(_check_fill_value(None, ndtype), control)
        control = np.array((0,), dtype=[('f0', float)]).astype(ndtype)
        assert_equal(_check_fill_value(0, ndtype), control)

        ndtype = np.dtype("int, (2,3)float, float")
        control = np.array((default_fill_value(0),
                            default_fill_value(0.),
                            default_fill_value(0.),),
                           dtype="int, float, float").astype(ndtype)
        test = _check_fill_value(None, ndtype)
        assert_equal(test, control)
        control = np.array((0, 0, 0), dtype="int, float, float").astype(ndtype)
        assert_equal(_check_fill_value(0, ndtype), control)
        # but when indexing, fill value should become scalar not tuple
        # See issue #6723
        M = masked_array(control)
        assert_equal(M["f1"].fill_value.ndim, 0)

    def test_fillvalue_datetime_timedelta(self):
        # Test default fillvalue for datetime64 and timedelta64 types.
        # See issue #4476, this would return '?' which would cause errors
        # elsewhere

        for timecode in ("as", "fs", "ps", "ns", "us", "ms", "s", "m",
                         "h", "D", "W", "M", "Y"):
            control = numpy.datetime64("NaT", timecode)
            test = default_fill_value(numpy.dtype("<M8[" + timecode + "]"))
            np.testing.assert_equal(test, control)

            control = numpy.timedelta64("NaT", timecode)
            test = default_fill_value(numpy.dtype("<m8[" + timecode + "]"))
            np.testing.assert_equal(test, control)

    def test_extremum_fill_value(self):
        # Tests extremum fill values for flexible type.
        a = array([(1, (2, 3)), (4, (5, 6))],
                  dtype=[('A', int), ('B', [('BA', int), ('BB', int)])])
        test = a.fill_value
        assert_equal(test.dtype, a.dtype)
        assert_equal(test['A'], default_fill_value(a['A']))
        assert_equal(test['B']['BA'], default_fill_value(a['B']['BA']))
        assert_equal(test['B']['BB'], default_fill_value(a['B']['BB']))

        test = minimum_fill_value(a)
        assert_equal(test.dtype, a.dtype)
        assert_equal(test[0], minimum_fill_value(a['A']))
        assert_equal(test[1][0], minimum_fill_value(a['B']['BA']))
        assert_equal(test[1][1], minimum_fill_value(a['B']['BB']))
        assert_equal(test[1], minimum_fill_value(a['B']))

        test = maximum_fill_value(a)
        assert_equal(test.dtype, a.dtype)
        assert_equal(test[0], maximum_fill_value(a['A']))
        assert_equal(test[1][0], maximum_fill_value(a['B']['BA']))
        assert_equal(test[1][1], maximum_fill_value(a['B']['BB']))
        assert_equal(test[1], maximum_fill_value(a['B']))

    def test_extremum_fill_value_subdtype(self):
        a = array(([2, 3, 4],), dtype=[('value', np.int8, 3)])

        test = minimum_fill_value(a)
        assert_equal(test.dtype, a.dtype)
        assert_equal(test[0], np.full(3, minimum_fill_value(a['value'])))

        test = maximum_fill_value(a)
        assert_equal(test.dtype, a.dtype)
        assert_equal(test[0], np.full(3, maximum_fill_value(a['value'])))

    def test_fillvalue_individual_fields(self):
        # Test setting fill_value on individual fields
        ndtype = [('a', int), ('b', int)]
        # Explicit fill_value
        a = array(list(zip([1, 2, 3], [4, 5, 6])),
                  fill_value=(-999, -999), dtype=ndtype)
        aa = a['a']
        aa.set_fill_value(10)
        assert_equal(aa._fill_value, np.array(10))
        assert_equal(tuple(a.fill_value), (10, -999))
        a.fill_value['b'] = -10
        assert_equal(tuple(a.fill_value), (10, -10))
        # Implicit fill_value
        t = array(list(zip([1, 2, 3], [4, 5, 6])), dtype=ndtype)
        tt = t['a']
        tt.set_fill_value(10)
        assert_equal(tt._fill_value, np.array(10))
        assert_equal(tuple(t.fill_value), (10, default_fill_value(0)))

    def test_fillvalue_implicit_structured_array(self):
        # Check that fill_value is always defined for structured arrays
        ndtype = ('b', float)
        adtype = ('a', float)
        a = array([(1.,), (2.,)], mask=[(False,), (False,)],
                  fill_value=(np.nan,), dtype=np.dtype([adtype]))
        b = empty(a.shape, dtype=[adtype, ndtype])
        b['a'] = a['a']
        b['a'].set_fill_value(a['a'].fill_value)
        f = b._fill_value[()]
        assert_(np.isnan(f[0]))
        assert_equal(f[-1], default_fill_value(1.))

    def test_fillvalue_as_arguments(self):
        # Test adding a fill_value parameter to empty/ones/zeros
        a = empty(3, fill_value=999.)
        assert_equal(a.fill_value, 999.)

        a = ones(3, fill_value=999., dtype=float)
        assert_equal(a.fill_value, 999.)

        a = zeros(3, fill_value=0., dtype=complex)
        assert_equal(a.fill_value, 0.)

        a = identity(3, fill_value=0., dtype=complex)
        assert_equal(a.fill_value, 0.)

    def test_shape_argument(self):
        # Test that shape can be provides as an argument
        # GH issue 6106
        a = empty(shape=(3, ))
        assert_equal(a.shape, (3, ))

        a = ones(shape=(3, ), dtype=float)
        assert_equal(a.shape, (3, ))

        a = zeros(shape=(3, ), dtype=complex)
        assert_equal(a.shape, (3, ))

    def test_fillvalue_in_view(self):
        # Test the behavior of fill_value in view

        # Create initial masked array
        x = array([1, 2, 3], fill_value=1, dtype=np.int64)

        # Check that fill_value is preserved by default
        y = x.view()
        assert_(y.fill_value == 1)

        # Check that fill_value is preserved if dtype is specified and the
        # dtype is an ndarray sub-class and has a _fill_value attribute
        y = x.view(MaskedArray)
        assert_(y.fill_value == 1)

        # Check that fill_value is preserved if type is specified and the
        # dtype is an ndarray sub-class and has a _fill_value attribute (by
        # default, the first argument is dtype, not type)
        y = x.view(type=MaskedArray)
        assert_(y.fill_value == 1)

        # Check that code does not crash if passed an ndarray sub-class that
        # does not have a _fill_value attribute
        y = x.view(np.ndarray)
        y = x.view(type=np.ndarray)

        # Check that fill_value can be overridden with view
        y = x.view(MaskedArray, fill_value=2)
        assert_(y.fill_value == 2)

        # Check that fill_value can be overridden with view (using type=)
        y = x.view(type=MaskedArray, fill_value=2)
        assert_(y.fill_value == 2)

        # Check that fill_value gets reset if passed a dtype but not a
        # fill_value. This is because even though in some cases one can safely
        # cast the fill_value, e.g. if taking an int64 view of an int32 array,
        # in other cases, this cannot be done (e.g. int32 view of an int64
        # array with a large fill_value).
        y = x.view(dtype=np.int32)
        assert_(y.fill_value == 999999)

    def test_fillvalue_bytes_or_str(self):
        # Test whether fill values work as expected for structured dtypes
        # containing bytes or str.  See issue #7259.
        a = empty(shape=(3, ), dtype="(2)3S,(2)3U")
        assert_equal(a["f0"].fill_value, default_fill_value(b"spam"))
        assert_equal(a["f1"].fill_value, default_fill_value("eggs"))


class TestUfuncs:
    # Test class for the application of ufuncs on MaskedArrays.

    def setup_method(self):
        # Base data definition.
        self.d = (array([1.0, 0, -1, pi / 2] * 2, mask=[0, 1] + [0] * 6),
                  array([1.0, 0, -1, pi / 2] * 2, mask=[1, 0] + [0] * 6),)
        self.err_status = np.geterr()
        np.seterr(divide='ignore', invalid='ignore')

    def teardown_method(self):
        np.seterr(**self.err_status)

    def test_testUfuncRegression(self):
        # Tests new ufuncs on MaskedArrays.
        for f in ['sqrt', 'log', 'log10', 'exp', 'conjugate',
                  'sin', 'cos', 'tan',
                  'arcsin', 'arccos', 'arctan',
                  'sinh', 'cosh', 'tanh',
                  'arcsinh',
                  'arccosh',
                  'arctanh',
                  'absolute', 'fabs', 'negative',
                  'floor', 'ceil',
                  'logical_not',
                  'add', 'subtract', 'multiply',
                  'divide', 'true_divide', 'floor_divide',
                  'remainder', 'fmod', 'hypot', 'arctan2',
                  'equal', 'not_equal', 'less_equal', 'greater_equal',
                  'less', 'greater',
                  'logical_and', 'logical_or', 'logical_xor',
                  ]:
            try:
                uf = getattr(umath, f)
            except AttributeError:
                uf = getattr(fromnumeric, f)
            mf = getattr(numpy.ma.core, f)
            args = self.d[:uf.nin]
            ur = uf(*args)
            mr = mf(*args)
            assert_equal(ur.filled(0), mr.filled(0), f)
            assert_mask_equal(ur.mask, mr.mask, err_msg=f)

    def test_reduce(self):
        # Tests reduce on MaskedArrays.
        a = self.d[0]
        assert_(not alltrue(a, axis=0))
        assert_(sometrue(a, axis=0))
        assert_equal(sum(a[:3], axis=0), 0)
        assert_equal(product(a, axis=0), 0)
        assert_equal(add.reduce(a), pi)

    def test_minmax(self):
        # Tests extrema on MaskedArrays.
        a = arange(1, 13).reshape(3, 4)
        amask = masked_where(a < 5, a)
        assert_equal(amask.max(), a.max())
        assert_equal(amask.min(), 5)
        assert_equal(amask.max(0), a.max(0))
        assert_equal(amask.min(0), [5, 6, 7, 8])
        assert_(amask.max(1)[0].mask)
        assert_(amask.min(1)[0].mask)

    def test_ndarray_mask(self):
        # Check that the mask of the result is a ndarray (not a MaskedArray...)
        a = masked_array([-1, 0, 1, 2, 3], mask=[0, 0, 0, 0, 1])
        test = np.sqrt(a)
        control = masked_array([-1, 0, 1, np.sqrt(2), -1],
                               mask=[1, 0, 0, 0, 1])
        assert_equal(test, control)
        assert_equal(test.mask, control.mask)
        assert_(not isinstance(test.mask, MaskedArray))

    def test_treatment_of_NotImplemented(self):
        # Check that NotImplemented is returned at appropriate places

        a = masked_array([1., 2.], mask=[1, 0])
        assert_raises(TypeError, operator.mul, a, "abc")
        assert_raises(TypeError, operator.truediv, a, "abc")

        class MyClass:
            __array_priority__ = a.__array_priority__ + 1

            def __mul__(self, other):
                return "My mul"

            def __rmul__(self, other):
                return "My rmul"

        me = MyClass()
        assert_(me * a == "My mul")
        assert_(a * me == "My rmul")

        # and that __array_priority__ is respected
        class MyClass2:
            __array_priority__ = 100

            def __mul__(self, other):
                return "Me2mul"

            def __rmul__(self, other):
                return "Me2rmul"

            def __rdiv__(self, other):
                return "Me2rdiv"

            __rtruediv__ = __rdiv__

        me_too = MyClass2()
        assert_(a.__mul__(me_too) is NotImplemented)
        assert_(all(multiply.outer(a, me_too) == "Me2rmul"))
        assert_(a.__truediv__(me_too) is NotImplemented)
        assert_(me_too * a == "Me2mul")
        assert_(a * me_too == "Me2rmul")
        assert_(a / me_too == "Me2rdiv")

    def test_no_masked_nan_warnings(self):
        # check that a nan in masked position does not
        # cause ufunc warnings

        m = np.ma.array([0.5, np.nan], mask=[0,1])

        with warnings.catch_warnings():
            warnings.filterwarnings("error")

            # test unary and binary ufuncs
            exp(m)
            add(m, 1)
            m > 0

            # test different unary domains
            sqrt(m)
            log(m)
            tan(m)
            arcsin(m)
            arccos(m)
            arccosh(m)

            # test binary domains
            divide(m, 2)

            # also check that allclose uses ma ufuncs, to avoid warning
            allclose(m, 0.5)

class TestMaskedArrayInPlaceArithmetic:
    # Test MaskedArray Arithmetic

    def setup_method(self):
        x = arange(10)
        y = arange(10)
        xm = arange(10)
        xm[2] = masked
        self.intdata = (x, y, xm)
        self.floatdata = (x.astype(float), y.astype(float), xm.astype(float))
        self.othertypes = np.typecodes['AllInteger'] + np.typecodes['AllFloat']
        self.othertypes = [np.dtype(_).type for _ in self.othertypes]
        self.uint8data = (
            x.astype(np.uint8),
            y.astype(np.uint8),
            xm.astype(np.uint8)
        )

    def test_inplace_addition_scalar(self):
        # Test of inplace additions
        (x, y, xm) = self.intdata
        xm[2] = masked
        x += 1
        assert_equal(x, y + 1)
        xm += 1
        assert_equal(xm, y + 1)

        (x, _, xm) = self.floatdata
        id1 = x.data.ctypes.data
        x += 1.
        assert_(id1 == x.data.ctypes.data)
        assert_equal(x, y + 1.)

    def test_inplace_addition_array(self):
        # Test of inplace additions
        (x, y, xm) = self.intdata
        m = xm.mask
        a = arange(10, dtype=np.int16)
        a[-1] = masked
        x += a
        xm += a
        assert_equal(x, y + a)
        assert_equal(xm, y + a)
        assert_equal(xm.mask, mask_or(m, a.mask))

    def test_inplace_subtraction_scalar(self):
        # Test of inplace subtractions
        (x, y, xm) = self.intdata
        x -= 1
        assert_equal(x, y - 1)
        xm -= 1
        assert_equal(xm, y - 1)

    def test_inplace_subtraction_array(self):
        # Test of inplace subtractions
        (x, y, xm) = self.floatdata
        m = xm.mask
        a = arange(10, dtype=float)
        a[-1] = masked
        x -= a
        xm -= a
        assert_equal(x, y - a)
        assert_equal(xm, y - a)
        assert_equal(xm.mask, mask_or(m, a.mask))

    def test_inplace_multiplication_scalar(self):
        # Test of inplace multiplication
        (x, y, xm) = self.floatdata
        x *= 2.0
        assert_equal(x, y * 2)
        xm *= 2.0
        assert_equal(xm, y * 2)

    def test_inplace_multiplication_array(self):
        # Test of inplace multiplication
        (x, y, xm) = self.floatdata
        m = xm.mask
        a = arange(10, dtype=float)
        a[-1] = masked
        x *= a
        xm *= a
        assert_equal(x, y * a)
        assert_equal(xm, y * a)
        assert_equal(xm.mask, mask_or(m, a.mask))

    def test_inplace_division_scalar_int(self):
        # Test of inplace division
        (x, y, xm) = self.intdata
        x = arange(10) * 2
        xm = arange(10) * 2
        xm[2] = masked
        x //= 2
        assert_equal(x, y)
        xm //= 2
        assert_equal(xm, y)

    def test_inplace_division_scalar_float(self):
        # Test of inplace division
        (x, y, xm) = self.floatdata
        x /= 2.0
        assert_equal(x, y / 2.0)
        xm /= arange(10)
        assert_equal(xm, ones((10,)))

    def test_inplace_division_array_float(self):
        # Test of inplace division
        (x, y, xm) = self.floatdata
        m = xm.mask
        a = arange(10, dtype=float)
        a[-1] = masked
        x /= a
        xm /= a
        assert_equal(x, y / a)
        assert_equal(xm, y / a)
        assert_equal(xm.mask, mask_or(mask_or(m, a.mask), (a == 0)))

    def test_inplace_division_misc(self):

        x = [1., 1., 1., -2., pi / 2., 4., 5., -10., 10., 1., 2., 3.]
        y = [5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.]
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = masked_array(x, mask=m1)
        ym = masked_array(y, mask=m2)

        z = xm / ym
        assert_equal(z._mask, [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
        assert_equal(z._data,
                     [1., 1., 1., -1., -pi / 2., 4., 5., 1., 1., 1., 2., 3.])

        xm = xm.copy()
        xm /= ym
        assert_equal(xm._mask, [1, 1, 1, 0, 0, 1, 1, 0, 0, 0, 1, 1])
        assert_equal(z._data,
                     [1., 1., 1., -1., -pi / 2., 4., 5., 1., 1., 1., 2., 3.])

    def test_datafriendly_add(self):
        # Test keeping data w/ (inplace) addition
        x = array([1, 2, 3], mask=[0, 0, 1])
        # Test add w/ scalar
        xx = x + 1
        assert_equal(xx.data, [2, 3, 3])
        assert_equal(xx.mask, [0, 0, 1])
        # Test iadd w/ scalar
        x += 1
        assert_equal(x.data, [2, 3, 3])
        assert_equal(x.mask, [0, 0, 1])
        # Test add w/ array
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x + array([1, 2, 3], mask=[1, 0, 0])
        assert_equal(xx.data, [1, 4, 3])
        assert_equal(xx.mask, [1, 0, 1])
        # Test iadd w/ array
        x = array([1, 2, 3], mask=[0, 0, 1])
        x += array([1, 2, 3], mask=[1, 0, 0])
        assert_equal(x.data, [1, 4, 3])
        assert_equal(x.mask, [1, 0, 1])

    def test_datafriendly_sub(self):
        # Test keeping data w/ (inplace) subtraction
        # Test sub w/ scalar
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x - 1
        assert_equal(xx.data, [0, 1, 3])
        assert_equal(xx.mask, [0, 0, 1])
        # Test isub w/ scalar
        x = array([1, 2, 3], mask=[0, 0, 1])
        x -= 1
        assert_equal(x.data, [0, 1, 3])
        assert_equal(x.mask, [0, 0, 1])
        # Test sub w/ array
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x - array([1, 2, 3], mask=[1, 0, 0])
        assert_equal(xx.data, [1, 0, 3])
        assert_equal(xx.mask, [1, 0, 1])
        # Test isub w/ array
        x = array([1, 2, 3], mask=[0, 0, 1])
        x -= array([1, 2, 3], mask=[1, 0, 0])
        assert_equal(x.data, [1, 0, 3])
        assert_equal(x.mask, [1, 0, 1])

    def test_datafriendly_mul(self):
        # Test keeping data w/ (inplace) multiplication
        # Test mul w/ scalar
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x * 2
        assert_equal(xx.data, [2, 4, 3])
        assert_equal(xx.mask, [0, 0, 1])
        # Test imul w/ scalar
        x = array([1, 2, 3], mask=[0, 0, 1])
        x *= 2
        assert_equal(x.data, [2, 4, 3])
        assert_equal(x.mask, [0, 0, 1])
        # Test mul w/ array
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x * array([10, 20, 30], mask=[1, 0, 0])
        assert_equal(xx.data, [1, 40, 3])
        assert_equal(xx.mask, [1, 0, 1])
        # Test imul w/ array
        x = array([1, 2, 3], mask=[0, 0, 1])
        x *= array([10, 20, 30], mask=[1, 0, 0])
        assert_equal(x.data, [1, 40, 3])
        assert_equal(x.mask, [1, 0, 1])

    def test_datafriendly_div(self):
        # Test keeping data w/ (inplace) division
        # Test div on scalar
        x = array([1, 2, 3], mask=[0, 0, 1])
        xx = x / 2.
        assert_equal(xx.data, [1 / 2., 2 / 2., 3])
        assert_equal(xx.mask, [0, 0, 1])
        # Test idiv on scalar
        x = array([1., 2., 3.], mask=[0, 0, 1])
        x /= 2.
        assert_equal(x.data, [1 / 2., 2 / 2., 3])
        assert_equal(x.mask, [0, 0, 1])
        # Test div on array
        x = array([1., 2., 3.], mask=[0, 0, 1])
        xx = x / array([10., 20., 30.], mask=[1, 0, 0])
        assert_equal(xx.data, [1., 2. / 20., 3.])
        assert_equal(xx.mask, [1, 0, 1])
        # Test idiv on array
        x = array([1., 2., 3.], mask=[0, 0, 1])
        x /= array([10., 20., 30.], mask=[1, 0, 0])
        assert_equal(x.data, [1., 2 / 20., 3.])
        assert_equal(x.mask, [1, 0, 1])

    def test_datafriendly_pow(self):
        # Test keeping data w/ (inplace) power
        # Test pow on scalar
        x = array([1., 2., 3.], mask=[0, 0, 1])
        xx = x ** 2.5
        assert_equal(xx.data, [1., 2. ** 2.5, 3.])
        assert_equal(xx.mask, [0, 0, 1])
        # Test ipow on scalar
        x **= 2.5
        assert_equal(x.data, [1., 2. ** 2.5, 3])
        assert_equal(x.mask, [0, 0, 1])

    def test_datafriendly_add_arrays(self):
        a = array([[1, 1], [3, 3]])
        b = array([1, 1], mask=[0, 0])
        a += b
        assert_equal(a, [[2, 2], [4, 4]])
        if a.mask is not nomask:
            assert_equal(a.mask, [[0, 0], [0, 0]])

        a = array([[1, 1], [3, 3]])
        b = array([1, 1], mask=[0, 1])
        a += b
        assert_equal(a, [[2, 2], [4, 4]])
        assert_equal(a.mask, [[0, 1], [0, 1]])

    def test_datafriendly_sub_arrays(self):
        a = array([[1, 1], [3, 3]])
        b = array([1, 1], mask=[0, 0])
        a -= b
        assert_equal(a, [[0, 0], [2, 2]])
        if a.mask is not nomask:
            assert_equal(a.mask, [[0, 0], [0, 0]])

        a = array([[1, 1], [3, 3]])
        b = array([1, 1], mask=[0, 1])
        a -= b
        assert_equal(a, [[0, 0], [2, 2]])
        assert_equal(a.mask, [[0, 1], [0, 1]])

    def test_datafriendly_mul_arrays(self):
        a = array([[1, 1], [3, 3]])
        b = array([1, 1], mask=[0, 0])
        a *= b
        assert_equal(a, [[1, 1], [3, 3]])
        if a.mask is not nomask:
            assert_equal(a.mask, [[0, 0], [0, 0]])

        a = array([[1, 1], [3, 3]])
        b = array([1, 1], mask=[0, 1])
        a *= b
        assert_equal(a, [[1, 1], [3, 3]])
        assert_equal(a.mask, [[0, 1], [0, 1]])

    def test_inplace_addition_scalar_type(self):
        # Test of inplace additions
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                xm[2] = masked
                x += t(1)
                assert_equal(x, y + t(1))
                xm += t(1)
                assert_equal(xm, y + t(1))

    def test_inplace_addition_array_type(self):
        # Test of inplace additions
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = arange(10, dtype=t)
                a[-1] = masked
                x += a
                xm += a
                assert_equal(x, y + a)
                assert_equal(xm, y + a)
                assert_equal(xm.mask, mask_or(m, a.mask))

    def test_inplace_subtraction_scalar_type(self):
        # Test of inplace subtractions
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                x -= t(1)
                assert_equal(x, y - t(1))
                xm -= t(1)
                assert_equal(xm, y - t(1))

    def test_inplace_subtraction_array_type(self):
        # Test of inplace subtractions
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = arange(10, dtype=t)
                a[-1] = masked
                x -= a
                xm -= a
                assert_equal(x, y - a)
                assert_equal(xm, y - a)
                assert_equal(xm.mask, mask_or(m, a.mask))

    def test_inplace_multiplication_scalar_type(self):
        # Test of inplace multiplication
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                x *= t(2)
                assert_equal(x, y * t(2))
                xm *= t(2)
                assert_equal(xm, y * t(2))

    def test_inplace_multiplication_array_type(self):
        # Test of inplace multiplication
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = arange(10, dtype=t)
                a[-1] = masked
                x *= a
                xm *= a
                assert_equal(x, y * a)
                assert_equal(xm, y * a)
                assert_equal(xm.mask, mask_or(m, a.mask))

    def test_inplace_floor_division_scalar_type(self):
        # Test of inplace division
        # Check for TypeError in case of unsupported types
        unsupported = {np.dtype(t).type for t in np.typecodes["Complex"]}
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                x = arange(10, dtype=t) * t(2)
                xm = arange(10, dtype=t) * t(2)
                xm[2] = masked
                try:
                    x //= t(2)
                    xm //= t(2)
                    assert_equal(x, y)
                    assert_equal(xm, y)
                except TypeError:
                    msg = f"Supported type {t} throwing TypeError"
                    assert t in unsupported, msg

    def test_inplace_floor_division_array_type(self):
        # Test of inplace division
        # Check for TypeError in case of unsupported types
        unsupported = {np.dtype(t).type for t in np.typecodes["Complex"]}
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = arange(10, dtype=t)
                a[-1] = masked
                try:
                    x //= a
                    xm //= a
                    assert_equal(x, y // a)
                    assert_equal(xm, y // a)
                    assert_equal(
                        xm.mask,
                        mask_or(mask_or(m, a.mask), (a == t(0)))
                    )
                except TypeError:
                    msg = f"Supported type {t} throwing TypeError"
                    assert t in unsupported, msg

    def test_inplace_division_scalar_type(self):
        # Test of inplace division
        for t in self.othertypes:
            with suppress_warnings() as sup:
                sup.record(UserWarning)

                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                x = arange(10, dtype=t) * t(2)
                xm = arange(10, dtype=t) * t(2)
                xm[2] = masked

                # May get a DeprecationWarning or a TypeError.
                #
                # This is a consequence of the fact that this is true divide
                # and will require casting to float for calculation and
                # casting back to the original type. This will only be raised
                # with integers. Whether it is an error or warning is only
                # dependent on how stringent the casting rules are.
                #
                # Will handle the same way.
                try:
                    x /= t(2)
                    assert_equal(x, y)
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)
                try:
                    xm /= t(2)
                    assert_equal(xm, y)
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)

                if issubclass(t, np.integer):
                    assert_equal(len(sup.log), 2, f'Failed on type={t}.')
                else:
                    assert_equal(len(sup.log), 0, f'Failed on type={t}.')

    def test_inplace_division_array_type(self):
        # Test of inplace division
        for t in self.othertypes:
            with suppress_warnings() as sup:
                sup.record(UserWarning)
                (x, y, xm) = (_.astype(t) for _ in self.uint8data)
                m = xm.mask
                a = arange(10, dtype=t)
                a[-1] = masked

                # May get a DeprecationWarning or a TypeError.
                #
                # This is a consequence of the fact that this is true divide
                # and will require casting to float for calculation and
                # casting back to the original type. This will only be raised
                # with integers. Whether it is an error or warning is only
                # dependent on how stringent the casting rules are.
                #
                # Will handle the same way.
                try:
                    x /= a
                    assert_equal(x, y / a)
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)
                try:
                    xm /= a
                    assert_equal(xm, y / a)
                    assert_equal(
                        xm.mask,
                        mask_or(mask_or(m, a.mask), (a == t(0)))
                    )
                except (DeprecationWarning, TypeError) as e:
                    warnings.warn(str(e), stacklevel=1)

                if issubclass(t, np.integer):
                    assert_equal(len(sup.log), 2, f'Failed on type={t}.')
                else:
                    assert_equal(len(sup.log), 0, f'Failed on type={t}.')

    def test_inplace_pow_type(self):
        # Test keeping data w/ (inplace) power
        for t in self.othertypes:
            with warnings.catch_warnings():
                warnings.filterwarnings("error")
                # Test pow on scalar
                x = array([1, 2, 3], mask=[0, 0, 1], dtype=t)
                xx = x ** t(2)
                xx_r = array([1, 2 ** 2, 3], mask=[0, 0, 1], dtype=t)
                assert_equal(xx.data, xx_r.data)
                assert_equal(xx.mask, xx_r.mask)
                # Test ipow on scalar
                x **= t(2)
                assert_equal(x.data, xx_r.data)
                assert_equal(x.mask, xx_r.mask)


class TestMaskedArrayMethods:
    # Test class for miscellaneous MaskedArrays methods.
    def setup_method(self):
        # Base data definition.
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        X = x.reshape(6, 6)
        XX = x.reshape(3, 2, 2, 3)

        m = np.array([0, 1, 0, 1, 0, 0,
                     1, 0, 1, 1, 0, 1,
                     0, 0, 0, 1, 0, 1,
                     0, 0, 0, 1, 1, 1,
                     1, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 1, 0])
        mx = array(data=x, mask=m)
        mX = array(data=X, mask=m.reshape(X.shape))
        mXX = array(data=XX, mask=m.reshape(XX.shape))

        m2 = np.array([1, 1, 0, 1, 0, 0,
                      1, 1, 1, 1, 0, 1,
                      0, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 1, 0,
                      0, 0, 1, 0, 1, 1])
        m2x = array(data=x, mask=m2)
        m2X = array(data=X, mask=m2.reshape(X.shape))
        m2XX = array(data=XX, mask=m2.reshape(XX.shape))
        self.d = (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX)

    def test_generic_methods(self):
        # Tests some MaskedArray methods.
        a = array([1, 3, 2])
        assert_equal(a.any(), a._data.any())
        assert_equal(a.all(), a._data.all())
        assert_equal(a.argmax(), a._data.argmax())
        assert_equal(a.argmin(), a._data.argmin())
        assert_equal(a.choose(0, 1, 2, 3, 4), a._data.choose(0, 1, 2, 3, 4))
        assert_equal(a.compress([1, 0, 1]), a._data.compress([1, 0, 1]))
        assert_equal(a.conj(), a._data.conj())
        assert_equal(a.conjugate(), a._data.conjugate())

        m = array([[1, 2], [3, 4]])
        assert_equal(m.diagonal(), m._data.diagonal())
        assert_equal(a.sum(), a._data.sum())
        assert_equal(a.take([1, 2]), a._data.take([1, 2]))
        assert_equal(m.transpose(), m._data.transpose())

    def test_allclose(self):
        # Tests allclose on arrays
        a = np.random.rand(10)
        b = a + np.random.rand(10) * 1e-8
        assert_(allclose(a, b))
        # Test allclose w/ infs
        a[0] = np.inf
        assert_(not allclose(a, b))
        b[0] = np.inf
        assert_(allclose(a, b))
        # Test allclose w/ masked
        a = masked_array(a)
        a[-1] = masked
        assert_(allclose(a, b, masked_equal=True))
        assert_(not allclose(a, b, masked_equal=False))
        # Test comparison w/ scalar
        a *= 1e-8
        a[0] = 0
        assert_(allclose(a, 0, masked_equal=True))

        # Test that the function works for MIN_INT integer typed arrays
        a = masked_array([np.iinfo(np.int_).min], dtype=np.int_)
        assert_(allclose(a, a))

    def test_allclose_timedelta(self):
        # Allclose currently works for timedelta64 as long as `atol` is
        # an integer or also a timedelta64
        a = np.array([[1, 2, 3, 4]], dtype="m8[ns]")
        assert allclose(a, a, atol=0)
        assert allclose(a, a, atol=np.timedelta64(1, "ns"))

    def test_allany(self):
        # Checks the any/all methods/functions.
        x = np.array([[0.13, 0.26, 0.90],
                      [0.28, 0.33, 0.63],
                      [0.31, 0.87, 0.70]])
        m = np.array([[True, False, False],
                      [False, False, False],
                      [True, True, False]], dtype=np.bool_)
        mx = masked_array(x, mask=m)
        mxbig = (mx > 0.5)
        mxsmall = (mx < 0.5)

        assert_(not mxbig.all())
        assert_(mxbig.any())
        assert_equal(mxbig.all(0), [False, False, True])
        assert_equal(mxbig.all(1), [False, False, True])
        assert_equal(mxbig.any(0), [False, False, True])
        assert_equal(mxbig.any(1), [True, True, True])

        assert_(not mxsmall.all())
        assert_(mxsmall.any())
        assert_equal(mxsmall.all(0), [True, True, False])
        assert_equal(mxsmall.all(1), [False, False, False])
        assert_equal(mxsmall.any(0), [True, True, False])
        assert_equal(mxsmall.any(1), [True, True, False])

    def test_allany_oddities(self):
        # Some fun with all and any
        store = empty((), dtype=bool)
        full = array([1, 2, 3], mask=True)

        assert_(full.all() is masked)
        full.all(out=store)
        assert_(store)
        assert_(store._mask, True)
        assert_(store is not masked)

        store = empty((), dtype=bool)
        assert_(full.any() is masked)
        full.any(out=store)
        assert_(not store)
        assert_(store._mask, True)
        assert_(store is not masked)

    def test_argmax_argmin(self):
        # Tests argmin & argmax on MaskedArrays.
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d

        assert_equal(mx.argmin(), 35)
        assert_equal(mX.argmin(), 35)
        assert_equal(m2x.argmin(), 4)
        assert_equal(m2X.argmin(), 4)
        assert_equal(mx.argmax(), 28)
        assert_equal(mX.argmax(), 28)
        assert_equal(m2x.argmax(), 31)
        assert_equal(m2X.argmax(), 31)

        assert_equal(mX.argmin(0), [2, 2, 2, 5, 0, 5])
        assert_equal(m2X.argmin(0), [2, 2, 4, 5, 0, 4])
        assert_equal(mX.argmax(0), [0, 5, 0, 5, 4, 0])
        assert_equal(m2X.argmax(0), [5, 5, 0, 5, 1, 0])

        assert_equal(mX.argmin(1), [4, 1, 0, 0, 5, 5, ])
        assert_equal(m2X.argmin(1), [4, 4, 0, 0, 5, 3])
        assert_equal(mX.argmax(1), [2, 4, 1, 1, 4, 1])
        assert_equal(m2X.argmax(1), [2, 4, 1, 1, 1, 1])

    def test_clip(self):
        # Tests clip on MaskedArrays.
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        m = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0])
        mx = array(x, mask=m)
        clipped = mx.clip(2, 8)
        assert_equal(clipped.mask, mx.mask)
        assert_equal(clipped._data, x.clip(2, 8))
        assert_equal(clipped._data, mx._data.clip(2, 8))

    def test_clip_out(self):
        # gh-14140
        a = np.arange(10)
        m = np.ma.MaskedArray(a, mask=[0, 1] * 5)
        m.clip(0, 5, out=m)
        assert_equal(m.mask, [0, 1] * 5)

    def test_compress(self):
        # test compress
        a = masked_array([1., 2., 3., 4., 5.], fill_value=9999)
        condition = (a > 1.5) & (a < 3.5)
        assert_equal(a.compress(condition), [2., 3.])

        a[[2, 3]] = masked
        b = a.compress(condition)
        assert_equal(b._data, [2., 3.])
        assert_equal(b._mask, [0, 1])
        assert_equal(b.fill_value, 9999)
        assert_equal(b, a[condition])

        condition = (a < 4.)
        b = a.compress(condition)
        assert_equal(b._data, [1., 2., 3.])
        assert_equal(b._mask, [0, 0, 1])
        assert_equal(b.fill_value, 9999)
        assert_equal(b, a[condition])

        a = masked_array([[10, 20, 30], [40, 50, 60]],
                         mask=[[0, 0, 1], [1, 0, 0]])
        b = a.compress(a.ravel() >= 22)
        assert_equal(b._data, [30, 40, 50, 60])
        assert_equal(b._mask, [1, 1, 0, 0])

        x = np.array([3, 1, 2])
        b = a.compress(x >= 2, axis=1)
        assert_equal(b._data, [[10, 30], [40, 60]])
        assert_equal(b._mask, [[0, 1], [1, 0]])

    def test_compressed(self):
        # Tests compressed
        a = array([1, 2, 3, 4], mask=[0, 0, 0, 0])
        b = a.compressed()
        assert_equal(b, a)
        a[0] = masked
        b = a.compressed()
        assert_equal(b, [2, 3, 4])

    def test_empty(self):
        # Tests empty/like
        datatype = [('a', int), ('b', float), ('c', '|S8')]
        a = masked_array([(1, 1.1, '1.1'), (2, 2.2, '2.2'), (3, 3.3, '3.3')],
                         dtype=datatype)
        assert_equal(len(a.fill_value.item()), len(datatype))

        b = empty_like(a)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)

        b = empty(len(a), dtype=datatype)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)

        # check empty_like mask handling
        a = masked_array([1, 2, 3], mask=[False, True, False])
        b = empty_like(a)
        assert_(not np.may_share_memory(a.mask, b.mask))
        b = a.view(masked_array)
        assert_(np.may_share_memory(a.mask, b.mask))

    def test_zeros(self):
        # Tests zeros/like
        datatype = [('a', int), ('b', float), ('c', '|S8')]
        a = masked_array([(1, 1.1, '1.1'), (2, 2.2, '2.2'), (3, 3.3, '3.3')],
                         dtype=datatype)
        assert_equal(len(a.fill_value.item()), len(datatype))

        b = zeros(len(a), dtype=datatype)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)

        b = zeros_like(a)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)

        # check zeros_like mask handling
        a = masked_array([1, 2, 3], mask=[False, True, False])
        b = zeros_like(a)
        assert_(not np.may_share_memory(a.mask, b.mask))
        b = a.view()
        assert_(np.may_share_memory(a.mask, b.mask))

    def test_ones(self):
        # Tests ones/like
        datatype = [('a', int), ('b', float), ('c', '|S8')]
        a = masked_array([(1, 1.1, '1.1'), (2, 2.2, '2.2'), (3, 3.3, '3.3')],
                         dtype=datatype)
        assert_equal(len(a.fill_value.item()), len(datatype))

        b = ones(len(a), dtype=datatype)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)

        b = ones_like(a)
        assert_equal(b.shape, a.shape)
        assert_equal(b.fill_value, a.fill_value)

        # check ones_like mask handling
        a = masked_array([1, 2, 3], mask=[False, True, False])
        b = ones_like(a)
        assert_(not np.may_share_memory(a.mask, b.mask))
        b = a.view()
        assert_(np.may_share_memory(a.mask, b.mask))

    @suppress_copy_mask_on_assignment
    def test_put(self):
        # Tests put.
        d = arange(5)
        n = [0, 0, 0, 1, 1]
        m = make_mask(n)
        x = array(d, mask=m)
        assert_(x[3] is masked)
        assert_(x[4] is masked)
        x[[1, 4]] = [10, 40]
        assert_(x[3] is masked)
        assert_(x[4] is not masked)
        assert_equal(x, [0, 10, 2, -1, 40])

        x = masked_array(arange(10), mask=[1, 0, 0, 0, 0] * 2)
        i = [0, 2, 4, 6]
        x.put(i, [6, 4, 2, 0])
        assert_equal(x, asarray([6, 1, 4, 3, 2, 5, 0, 7, 8, 9, ]))
        assert_equal(x.mask, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        x.put(i, masked_array([0, 2, 4, 6], [1, 0, 1, 0]))
        assert_array_equal(x, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ])
        assert_equal(x.mask, [1, 0, 0, 0, 1, 1, 0, 0, 0, 0])

        x = masked_array(arange(10), mask=[1, 0, 0, 0, 0] * 2)
        put(x, i, [6, 4, 2, 0])
        assert_equal(x, asarray([6, 1, 4, 3, 2, 5, 0, 7, 8, 9, ]))
        assert_equal(x.mask, [0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
        put(x, i, masked_array([0, 2, 4, 6], [1, 0, 1, 0]))
        assert_array_equal(x, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, ])
        assert_equal(x.mask, [1, 0, 0, 0, 1, 1, 0, 0, 0, 0])

    def test_put_nomask(self):
        # GitHub issue 6425
        x = zeros(10)
        z = array([3., -1.], mask=[False, True])

        x.put([1, 2], z)
        assert_(x[0] is not masked)
        assert_equal(x[0], 0)
        assert_(x[1] is not masked)
        assert_equal(x[1], 3)
        assert_(x[2] is masked)
        assert_(x[3] is not masked)
        assert_equal(x[3], 0)

    def test_put_hardmask(self):
        # Tests put on hardmask
        d = arange(5)
        n = [0, 0, 0, 1, 1]
        m = make_mask(n)
        xh = array(d + 1, mask=m, hard_mask=True, copy=True)
        xh.put([4, 2, 0, 1, 3], [1, 2, 3, 4, 5])
        assert_equal(xh._data, [3, 4, 2, 4, 5])

    def test_putmask(self):
        x = arange(6) + 1
        mx = array(x, mask=[0, 0, 0, 1, 1, 1])
        mask = [0, 0, 1, 0, 0, 1]
        # w/o mask, w/o masked values
        xx = x.copy()
        putmask(xx, mask, 99)
        assert_equal(xx, [1, 2, 99, 4, 5, 99])
        # w/ mask, w/o masked values
        mxx = mx.copy()
        putmask(mxx, mask, 99)
        assert_equal(mxx._data, [1, 2, 99, 4, 5, 99])
        assert_equal(mxx._mask, [0, 0, 0, 1, 1, 0])
        # w/o mask, w/ masked values
        values = array([10, 20, 30, 40, 50, 60], mask=[1, 1, 1, 0, 0, 0])
        xx = x.copy()
        putmask(xx, mask, values)
        assert_equal(xx._data, [1, 2, 30, 4, 5, 60])
        assert_equal(xx._mask, [0, 0, 1, 0, 0, 0])
        # w/ mask, w/ masked values
        mxx = mx.copy()
        putmask(mxx, mask, values)
        assert_equal(mxx._data, [1, 2, 30, 4, 5, 60])
        assert_equal(mxx._mask, [0, 0, 1, 1, 1, 0])
        # w/ mask, w/ masked values + hardmask
        mxx = mx.copy()
        mxx.harden_mask()
        putmask(mxx, mask, values)
        assert_equal(mxx, [1, 2, 30, 4, 5, 60])

    def test_ravel(self):
        # Tests ravel
        a = array([[1, 2, 3, 4, 5]], mask=[[0, 1, 0, 0, 0]])
        aravel = a.ravel()
        assert_equal(aravel._mask.shape, aravel.shape)
        a = array([0, 0], mask=[1, 1])
        aravel = a.ravel()
        assert_equal(aravel._mask.shape, a.shape)
        # Checks that small_mask is preserved
        a = array([1, 2, 3, 4], mask=[0, 0, 0, 0], shrink=False)
        assert_equal(a.ravel()._mask, [0, 0, 0, 0])
        # Test that the fill_value is preserved
        a.fill_value = -99
        a.shape = (2, 2)
        ar = a.ravel()
        assert_equal(ar._mask, [0, 0, 0, 0])
        assert_equal(ar._data, [1, 2, 3, 4])
        assert_equal(ar.fill_value, -99)
        # Test index ordering
        assert_equal(a.ravel(order='C'), [1, 2, 3, 4])
        assert_equal(a.ravel(order='F'), [1, 3, 2, 4])

    def test_reshape(self):
        # Tests reshape
        x = arange(4)
        x[0] = masked
        y = x.reshape(2, 2)
        assert_equal(y.shape, (2, 2,))
        assert_equal(y._mask.shape, (2, 2,))
        assert_equal(x.shape, (4,))
        assert_equal(x._mask.shape, (4,))

    def test_sort(self):
        # Test sort
        x = array([1, 4, 2, 3], mask=[0, 1, 0, 0], dtype=np.uint8)

        sortedx = sort(x)
        assert_equal(sortedx._data, [1, 2, 3, 4])
        assert_equal(sortedx._mask, [0, 0, 0, 1])

        sortedx = sort(x, endwith=False)
        assert_equal(sortedx._data, [4, 1, 2, 3])
        assert_equal(sortedx._mask, [1, 0, 0, 0])

        x.sort()
        assert_equal(x._data, [1, 2, 3, 4])
        assert_equal(x._mask, [0, 0, 0, 1])

        x = array([1, 4, 2, 3], mask=[0, 1, 0, 0], dtype=np.uint8)
        x.sort(endwith=False)
        assert_equal(x._data, [4, 1, 2, 3])
        assert_equal(x._mask, [1, 0, 0, 0])

        x = [1, 4, 2, 3]
        sortedx = sort(x)
        assert_(not isinstance(sorted, MaskedArray))

        x = array([0, 1, -1, -2, 2], mask=nomask, dtype=np.int8)
        sortedx = sort(x, endwith=False)
        assert_equal(sortedx._data, [-2, -1, 0, 1, 2])
        x = array([0, 1, -1, -2, 2], mask=[0, 1, 0, 0, 1], dtype=np.int8)
        sortedx = sort(x, endwith=False)
        assert_equal(sortedx._data, [1, 2, -2, -1, 0])
        assert_equal(sortedx._mask, [1, 1, 0, 0, 0])

        x = array([0, -1], dtype=np.int8)
        sortedx = sort(x, kind="stable")
        assert_equal(sortedx, array([-1, 0], dtype=np.int8))

    def test_stable_sort(self):
        x = array([1, 2, 3, 1, 2, 3], dtype=np.uint8)
        expected = array([0, 3, 1, 4, 2, 5])
        computed = argsort(x, kind='stable')
        assert_equal(computed, expected)

    def test_argsort_matches_sort(self):
        x = array([1, 4, 2, 3], mask=[0, 1, 0, 0], dtype=np.uint8)

        for kwargs in [dict(),
                       dict(endwith=True),
                       dict(endwith=False),
                       dict(fill_value=2),
                       dict(fill_value=2, endwith=True),
                       dict(fill_value=2, endwith=False)]:
            sortedx = sort(x, **kwargs)
            argsortedx = x[argsort(x, **kwargs)]
            assert_equal(sortedx._data, argsortedx._data)
            assert_equal(sortedx._mask, argsortedx._mask)

    def test_sort_2d(self):
        # Check sort of 2D array.
        # 2D array w/o mask
        a = masked_array([[8, 4, 1], [2, 0, 9]])
        a.sort(0)
        assert_equal(a, [[2, 0, 1], [8, 4, 9]])
        a = masked_array([[8, 4, 1], [2, 0, 9]])
        a.sort(1)
        assert_equal(a, [[1, 4, 8], [0, 2, 9]])
        # 2D array w/mask
        a = masked_array([[8, 4, 1], [2, 0, 9]], mask=[[1, 0, 0], [0, 0, 1]])
        a.sort(0)
        assert_equal(a, [[2, 0, 1], [8, 4, 9]])
        assert_equal(a._mask, [[0, 0, 0], [1, 0, 1]])
        a = masked_array([[8, 4, 1], [2, 0, 9]], mask=[[1, 0, 0], [0, 0, 1]])
        a.sort(1)
        assert_equal(a, [[1, 4, 8], [0, 2, 9]])
        assert_equal(a._mask, [[0, 0, 1], [0, 0, 1]])
        # 3D
        a = masked_array([[[7, 8, 9], [4, 5, 6], [1, 2, 3]],
                          [[1, 2, 3], [7, 8, 9], [4, 5, 6]],
                          [[7, 8, 9], [1, 2, 3], [4, 5, 6]],
                          [[4, 5, 6], [1, 2, 3], [7, 8, 9]]])
        a[a % 4 == 0] = masked
        am = a.copy()
        an = a.filled(99)
        am.sort(0)
        an.sort(0)
        assert_equal(am, an)
        am = a.copy()
        an = a.filled(99)
        am.sort(1)
        an.sort(1)
        assert_equal(am, an)
        am = a.copy()
        an = a.filled(99)
        am.sort(2)
        an.sort(2)
        assert_equal(am, an)

    def test_sort_flexible(self):
        # Test sort on structured dtype.
        a = array(
            data=[(3, 3), (3, 2), (2, 2), (2, 1), (1, 0), (1, 1), (1, 2)],
            mask=[(0, 0), (0, 1), (0, 0), (0, 0), (1, 0), (0, 0), (0, 0)],
            dtype=[('A', int), ('B', int)])
        mask_last = array(
            data=[(1, 1), (1, 2), (2, 1), (2, 2), (3, 3), (3, 2), (1, 0)],
            mask=[(0, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (1, 0)],
            dtype=[('A', int), ('B', int)])
        mask_first = array(
            data=[(1, 0), (1, 1), (1, 2), (2, 1), (2, 2), (3, 2), (3, 3)],
            mask=[(1, 0), (0, 0), (0, 0), (0, 0), (0, 0), (0, 1), (0, 0)],
            dtype=[('A', int), ('B', int)])

        test = sort(a)
        assert_equal(test, mask_last)
        assert_equal(test.mask, mask_last.mask)

        test = sort(a, endwith=False)
        assert_equal(test, mask_first)
        assert_equal(test.mask, mask_first.mask)

        # Test sort on dtype with subarray (gh-8069)
        # Just check that the sort does not error, structured array subarrays
        # are treated as byte strings and that leads to differing behavior
        # depending on endianness and `endwith`.
        dt = np.dtype([('v', int, 2)])
        a = a.view(dt)
        test = sort(a)
        test = sort(a, endwith=False)

    def test_argsort(self):
        # Test argsort
        a = array([1, 5, 2, 4, 3], mask=[1, 0, 0, 1, 0])
        assert_equal(np.argsort(a), argsort(a))

    def test_squeeze(self):
        # Check squeeze
        data = masked_array([[1, 2, 3]])
        assert_equal(data.squeeze(), [1, 2, 3])
        data = masked_array([[1, 2, 3]], mask=[[1, 1, 1]])
        assert_equal(data.squeeze(), [1, 2, 3])
        assert_equal(data.squeeze()._mask, [1, 1, 1])

        # normal ndarrays return a view
        arr = np.array([[1]])
        arr_sq = arr.squeeze()
        assert_equal(arr_sq, 1)
        arr_sq[...] = 2
        assert_equal(arr[0,0], 2)

        # so maskedarrays should too
        m_arr = masked_array([[1]], mask=True)
        m_arr_sq = m_arr.squeeze()
        assert_(m_arr_sq is not np.ma.masked)
        assert_equal(m_arr_sq.mask, True)
        m_arr_sq[...] = 2
        assert_equal(m_arr[0,0], 2)

    def test_swapaxes(self):
        # Tests swapaxes on MaskedArrays.
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        m = np.array([0, 1, 0, 1, 0, 0,
                      1, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 0, 0,
                      0, 0, 1, 0, 1, 0])
        mX = array(x, mask=m).reshape(6, 6)
        mXX = mX.reshape(3, 2, 2, 3)

        mXswapped = mX.swapaxes(0, 1)
        assert_equal(mXswapped[-1], mX[:, -1])

        mXXswapped = mXX.swapaxes(0, 2)
        assert_equal(mXXswapped.shape, (2, 2, 3, 3))

    def test_take(self):
        # Tests take
        x = masked_array([10, 20, 30, 40], [0, 1, 0, 1])
        assert_equal(x.take([0, 0, 3]), masked_array([10, 10, 40], [0, 0, 1]))
        assert_equal(x.take([0, 0, 3]), x[[0, 0, 3]])
        assert_equal(x.take([[0, 1], [0, 1]]),
                     masked_array([[10, 20], [10, 20]], [[0, 1], [0, 1]]))

        # assert_equal crashes when passed np.ma.mask
        assert_(x[1] is np.ma.masked)
        assert_(x.take(1) is np.ma.masked)

        x = array([[10, 20, 30], [40, 50, 60]], mask=[[0, 0, 1], [1, 0, 0, ]])
        assert_equal(x.take([0, 2], axis=1),
                     array([[10, 30], [40, 60]], mask=[[0, 1], [1, 0]]))
        assert_equal(take(x, [0, 2], axis=1),
                     array([[10, 30], [40, 60]], mask=[[0, 1], [1, 0]]))

    def test_take_masked_indices(self):
        # Test take w/ masked indices
        a = np.array((40, 18, 37, 9, 22))
        indices = np.arange(3)[None,:] + np.arange(5)[:, None]
        mindices = array(indices, mask=(indices >= len(a)))
        # No mask
        test = take(a, mindices, mode='clip')
        ctrl = array([[40, 18, 37],
                      [18, 37, 9],
                      [37, 9, 22],
                      [9, 22, 22],
                      [22, 22, 22]])
        assert_equal(test, ctrl)
        # Masked indices
        test = take(a, mindices)
        ctrl = array([[40, 18, 37],
                      [18, 37, 9],
                      [37, 9, 22],
                      [9, 22, 40],
                      [22, 40, 40]])
        ctrl[3, 2] = ctrl[4, 1] = ctrl[4, 2] = masked
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)
        # Masked input + masked indices
        a = array((40, 18, 37, 9, 22), mask=(0, 1, 0, 0, 0))
        test = take(a, mindices)
        ctrl[0, 1] = ctrl[1, 0] = masked
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)

    def test_tolist(self):
        # Tests to list
        # ... on 1D
        x = array(np.arange(12))
        x[[1, -2]] = masked
        xlist = x.tolist()
        assert_(xlist[1] is None)
        assert_(xlist[-2] is None)
        # ... on 2D
        x.shape = (3, 4)
        xlist = x.tolist()
        ctrl = [[0, None, 2, 3], [4, 5, 6, 7], [8, 9, None, 11]]
        assert_equal(xlist[0], [0, None, 2, 3])
        assert_equal(xlist[1], [4, 5, 6, 7])
        assert_equal(xlist[2], [8, 9, None, 11])
        assert_equal(xlist, ctrl)
        # ... on structured array w/ masked records
        x = array(list(zip([1, 2, 3],
                           [1.1, 2.2, 3.3],
                           ['one', 'two', 'thr'])),
                  dtype=[('a', int), ('b', float), ('c', '|S8')])
        x[-1] = masked
        assert_equal(x.tolist(),
                     [(1, 1.1, b'one'),
                      (2, 2.2, b'two'),
                      (None, None, None)])
        # ... on structured array w/ masked fields
        a = array([(1, 2,), (3, 4)], mask=[(0, 1), (0, 0)],
                  dtype=[('a', int), ('b', int)])
        test = a.tolist()
        assert_equal(test, [[1, None], [3, 4]])
        # ... on mvoid
        a = a[0]
        test = a.tolist()
        assert_equal(test, [1, None])

    def test_tolist_specialcase(self):
        # Test mvoid.tolist: make sure we return a standard Python object
        a = array([(0, 1), (2, 3)], dtype=[('a', int), ('b', int)])
        # w/o mask: each entry is a np.void whose elements are standard Python
        for entry in a:
            for item in entry.tolist():
                assert_(not isinstance(item, np.generic))
        # w/ mask: each entry is a ma.void whose elements should be
        # standard Python
        a.mask[0] = (0, 1)
        for entry in a:
            for item in entry.tolist():
                assert_(not isinstance(item, np.generic))

    def test_toflex(self):
        # Test the conversion to records
        data = arange(10)
        record = data.toflex()
        assert_equal(record['_data'], data._data)
        assert_equal(record['_mask'], data._mask)

        data[[0, 1, 2, -1]] = masked
        record = data.toflex()
        assert_equal(record['_data'], data._data)
        assert_equal(record['_mask'], data._mask)

        ndtype = [('i', int), ('s', '|S3'), ('f', float)]
        data = array([(i, s, f) for (i, s, f) in zip(np.arange(10),
                                                     'ABCDEFGHIJKLM',
                                                     np.random.rand(10))],
                     dtype=ndtype)
        data[[0, 1, 2, -1]] = masked
        record = data.toflex()
        assert_equal(record['_data'], data._data)
        assert_equal(record['_mask'], data._mask)

        ndtype = np.dtype("int, (2,3)float, float")
        data = array([(i, f, ff) for (i, f, ff) in zip(np.arange(10),
                                                       np.random.rand(10),
                                                       np.random.rand(10))],
                     dtype=ndtype)
        data[[0, 1, 2, -1]] = masked
        record = data.toflex()
        assert_equal_records(record['_data'], data._data)
        assert_equal_records(record['_mask'], data._mask)

    def test_fromflex(self):
        # Test the reconstruction of a masked_array from a record
        a = array([1, 2, 3])
        test = fromflex(a.toflex())
        assert_equal(test, a)
        assert_equal(test.mask, a.mask)

        a = array([1, 2, 3], mask=[0, 0, 1])
        test = fromflex(a.toflex())
        assert_equal(test, a)
        assert_equal(test.mask, a.mask)

        a = array([(1, 1.), (2, 2.), (3, 3.)], mask=[(1, 0), (0, 0), (0, 1)],
                  dtype=[('A', int), ('B', float)])
        test = fromflex(a.toflex())
        assert_equal(test, a)
        assert_equal(test.data, a.data)

    def test_arraymethod(self):
        # Test a _arraymethod w/ n argument
        marray = masked_array([[1, 2, 3, 4, 5]], mask=[0, 0, 1, 0, 0])
        control = masked_array([[1], [2], [3], [4], [5]],
                               mask=[0, 0, 1, 0, 0])
        assert_equal(marray.T, control)
        assert_equal(marray.transpose(), control)

        assert_equal(MaskedArray.cumsum(marray.T, 0), control.cumsum(0))

    def test_arraymethod_0d(self):
        # gh-9430
        x = np.ma.array(42, mask=True)
        assert_equal(x.T.mask, x.mask)
        assert_equal(x.T.data, x.data)

    def test_transpose_view(self):
        x = np.ma.array([[1, 2, 3], [4, 5, 6]])
        x[0,1] = np.ma.masked
        xt = x.T

        xt[1,0] = 10
        xt[0,1] = np.ma.masked

        assert_equal(x.data, xt.T.data)
        assert_equal(x.mask, xt.T.mask)

    def test_diagonal_view(self):
        x = np.ma.zeros((3,3))
        x[0,0] = 10
        x[1,1] = np.ma.masked
        x[2,2] = 20
        xd = x.diagonal()
        x[1,1] = 15
        assert_equal(xd.mask, x.diagonal().mask)
        assert_equal(xd.data, x.diagonal().data)


class TestMaskedArrayMathMethods:

    def setup_method(self):
        # Base data definition.
        x = np.array([8.375, 7.545, 8.828, 8.5, 1.757, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479,
                      7.189, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993])
        X = x.reshape(6, 6)
        XX = x.reshape(3, 2, 2, 3)

        m = np.array([0, 1, 0, 1, 0, 0,
                     1, 0, 1, 1, 0, 1,
                     0, 0, 0, 1, 0, 1,
                     0, 0, 0, 1, 1, 1,
                     1, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 1, 0])
        mx = array(data=x, mask=m)
        mX = array(data=X, mask=m.reshape(X.shape))
        mXX = array(data=XX, mask=m.reshape(XX.shape))

        m2 = np.array([1, 1, 0, 1, 0, 0,
                      1, 1, 1, 1, 0, 1,
                      0, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 1, 0,
                      0, 0, 1, 0, 1, 1])
        m2x = array(data=x, mask=m2)
        m2X = array(data=X, mask=m2.reshape(X.shape))
        m2XX = array(data=XX, mask=m2.reshape(XX.shape))
        self.d = (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX)

    def test_cumsumprod(self):
        # Tests cumsum & cumprod on MaskedArrays.
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        mXcp = mX.cumsum(0)
        assert_equal(mXcp._data, mX.filled(0).cumsum(0))
        mXcp = mX.cumsum(1)
        assert_equal(mXcp._data, mX.filled(0).cumsum(1))

        mXcp = mX.cumprod(0)
        assert_equal(mXcp._data, mX.filled(1).cumprod(0))
        mXcp = mX.cumprod(1)
        assert_equal(mXcp._data, mX.filled(1).cumprod(1))

    def test_cumsumprod_with_output(self):
        # Tests cumsum/cumprod w/ output
        xm = array(np.random.uniform(0, 10, 12)).reshape(3, 4)
        xm[:, 0] = xm[0] = xm[-1, -1] = masked

        for funcname in ('cumsum', 'cumprod'):
            npfunc = getattr(np, funcname)
            xmmeth = getattr(xm, funcname)

            # A ndarray as explicit input
            output = np.empty((3, 4), dtype=float)
            output.fill(-9999)
            result = npfunc(xm, axis=0, out=output)
            # ... the result should be the given output
            assert_(result is output)
            assert_equal(result, xmmeth(axis=0, out=output))

            output = empty((3, 4), dtype=int)
            result = xmmeth(axis=0, out=output)
            assert_(result is output)

    def test_ptp(self):
        # Tests ptp on MaskedArrays.
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        (n, m) = X.shape
        assert_equal(mx.ptp(), mx.compressed().ptp())
        rows = np.zeros(n, float)
        cols = np.zeros(m, float)
        for k in range(m):
            cols[k] = mX[:, k].compressed().ptp()
        for k in range(n):
            rows[k] = mX[k].compressed().ptp()
        assert_equal(mX.ptp(0), cols)
        assert_equal(mX.ptp(1), rows)

    def test_add_object(self):
        x = masked_array(['a', 'b'], mask=[1, 0], dtype=object)
        y = x + 'x'
        assert_equal(y[1], 'bx')
        assert_(y.mask[0])

    def test_sum_object(self):
        # Test sum on object dtype
        a = masked_array([1, 2, 3], mask=[1, 0, 0], dtype=object)
        assert_equal(a.sum(), 5)
        a = masked_array([[1, 2, 3], [4, 5, 6]], dtype=object)
        assert_equal(a.sum(axis=0), [5, 7, 9])

    def test_prod_object(self):
        # Test prod on object dtype
        a = masked_array([1, 2, 3], mask=[1, 0, 0], dtype=object)
        assert_equal(a.prod(), 2 * 3)
        a = masked_array([[1, 2, 3], [4, 5, 6]], dtype=object)
        assert_equal(a.prod(axis=0), [4, 10, 18])

    def test_meananom_object(self):
        # Test mean/anom on object dtype
        a = masked_array([1, 2, 3], dtype=object)
        assert_equal(a.mean(), 2)
        assert_equal(a.anom(), [-1, 0, 1])

    def test_anom_shape(self):
        a = masked_array([1, 2, 3])
        assert_equal(a.anom().shape, a.shape)
        a.mask = True
        assert_equal(a.anom().shape, a.shape)
        assert_(np.ma.is_masked(a.anom()))

    def test_anom(self):
        a = masked_array(np.arange(1, 7).reshape(2, 3))
        assert_almost_equal(a.anom(),
                            [[-2.5, -1.5, -0.5], [0.5, 1.5, 2.5]])
        assert_almost_equal(a.anom(axis=0),
                            [[-1.5, -1.5, -1.5], [1.5, 1.5, 1.5]])
        assert_almost_equal(a.anom(axis=1),
                            [[-1., 0., 1.], [-1., 0., 1.]])
        a.mask = [[0, 0, 1], [0, 1, 0]]
        mval = -99
        assert_almost_equal(a.anom().filled(mval),
                            [[-2.25, -1.25, mval], [0.75, mval, 2.75]])
        assert_almost_equal(a.anom(axis=0).filled(mval),
                            [[-1.5, 0.0, mval], [1.5, mval, 0.0]])
        assert_almost_equal(a.anom(axis=1).filled(mval),
                            [[-0.5, 0.5, mval], [-1.0, mval, 1.0]])

    def test_trace(self):
        # Tests trace on MaskedArrays.
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        mXdiag = mX.diagonal()
        assert_equal(mX.trace(), mX.diagonal().compressed().sum())
        assert_almost_equal(mX.trace(),
                            X.trace() - sum(mXdiag.mask * X.diagonal(),
                                            axis=0))
        assert_equal(np.trace(mX), mX.trace())

        # gh-5560
        arr = np.arange(2*4*4).reshape(2,4,4)
        m_arr = np.ma.masked_array(arr, False)
        assert_equal(arr.trace(axis1=1, axis2=2), m_arr.trace(axis1=1, axis2=2))

    def test_dot(self):
        # Tests dot on MaskedArrays.
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        fx = mx.filled(0)
        r = mx.dot(mx)
        assert_almost_equal(r.filled(0), fx.dot(fx))
        assert_(r.mask is nomask)

        fX = mX.filled(0)
        r = mX.dot(mX)
        assert_almost_equal(r.filled(0), fX.dot(fX))
        assert_(r.mask[1,3])
        r1 = empty_like(r)
        mX.dot(mX, out=r1)
        assert_almost_equal(r, r1)

        mYY = mXX.swapaxes(-1, -2)
        fXX, fYY = mXX.filled(0), mYY.filled(0)
        r = mXX.dot(mYY)
        assert_almost_equal(r.filled(0), fXX.dot(fYY))
        r1 = empty_like(r)
        mXX.dot(mYY, out=r1)
        assert_almost_equal(r, r1)

    def test_dot_shape_mismatch(self):
        # regression test
        x = masked_array([[1,2],[3,4]], mask=[[0,1],[0,0]])
        y = masked_array([[1,2],[3,4]], mask=[[0,1],[0,0]])
        z = masked_array([[0,1],[3,3]])
        x.dot(y, out=z)
        assert_almost_equal(z.filled(0), [[1, 0], [15, 16]])
        assert_almost_equal(z.mask, [[0, 1], [0, 0]])

    def test_varmean_nomask(self):
        # gh-5769
        foo = array([1,2,3,4], dtype='f8')
        bar = array([1,2,3,4], dtype='f8')
        assert_equal(type(foo.mean()), np.float64)
        assert_equal(type(foo.var()), np.float64)
        assert((foo.mean() == bar.mean()) is np.bool_(True))

        # check array type is preserved and out works
        foo = array(np.arange(16).reshape((4,4)), dtype='f8')
        bar = empty(4, dtype='f4')
        assert_equal(type(foo.mean(axis=1)), MaskedArray)
        assert_equal(type(foo.var(axis=1)), MaskedArray)
        assert_(foo.mean(axis=1, out=bar) is bar)
        assert_(foo.var(axis=1, out=bar) is bar)

    def test_varstd(self):
        # Tests var & std on MaskedArrays.
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        assert_almost_equal(mX.var(axis=None), mX.compressed().var())
        assert_almost_equal(mX.std(axis=None), mX.compressed().std())
        assert_almost_equal(mX.std(axis=None, ddof=1),
                            mX.compressed().std(ddof=1))
        assert_almost_equal(mX.var(axis=None, ddof=1),
                            mX.compressed().var(ddof=1))
        assert_equal(mXX.var(axis=3).shape, XX.var(axis=3).shape)
        assert_equal(mX.var().shape, X.var().shape)
        (mXvar0, mXvar1) = (mX.var(axis=0), mX.var(axis=1))
        assert_almost_equal(mX.var(axis=None, ddof=2),
                            mX.compressed().var(ddof=2))
        assert_almost_equal(mX.std(axis=None, ddof=2),
                            mX.compressed().std(ddof=2))
        for k in range(6):
            assert_almost_equal(mXvar1[k], mX[k].compressed().var())
            assert_almost_equal(mXvar0[k], mX[:, k].compressed().var())
            assert_almost_equal(np.sqrt(mXvar0[k]),
                                mX[:, k].compressed().std())

    @suppress_copy_mask_on_assignment
    def test_varstd_specialcases(self):
        # Test a special case for var
        nout = np.array(-1, dtype=float)
        mout = array(-1, dtype=float)

        x = array(arange(10), mask=True)
        for methodname in ('var', 'std'):
            method = getattr(x, methodname)
            assert_(method() is masked)
            assert_(method(0) is masked)
            assert_(method(-1) is masked)
            # Using a masked array as explicit output
            method(out=mout)
            assert_(mout is not masked)
            assert_equal(mout.mask, True)
            # Using a ndarray as explicit output
            method(out=nout)
            assert_(np.isnan(nout))

        x = array(arange(10), mask=True)
        x[-1] = 9
        for methodname in ('var', 'std'):
            method = getattr(x, methodname)
            assert_(method(ddof=1) is masked)
            assert_(method(0, ddof=1) is masked)
            assert_(method(-1, ddof=1) is masked)
            # Using a masked array as explicit output
            method(out=mout, ddof=1)
            assert_(mout is not masked)
            assert_equal(mout.mask, True)
            # Using a ndarray as explicit output
            method(out=nout, ddof=1)
            assert_(np.isnan(nout))

    def test_varstd_ddof(self):
        a = array([[1, 1, 0], [1, 1, 0]], mask=[[0, 0, 1], [0, 0, 1]])
        test = a.std(axis=0, ddof=0)
        assert_equal(test.filled(0), [0, 0, 0])
        assert_equal(test.mask, [0, 0, 1])
        test = a.std(axis=0, ddof=1)
        assert_equal(test.filled(0), [0, 0, 0])
        assert_equal(test.mask, [0, 0, 1])
        test = a.std(axis=0, ddof=2)
        assert_equal(test.filled(0), [0, 0, 0])
        assert_equal(test.mask, [1, 1, 1])

    def test_diag(self):
        # Test diag
        x = arange(9).reshape((3, 3))
        x[1, 1] = masked
        out = np.diag(x)
        assert_equal(out, [0, 4, 8])
        out = diag(x)
        assert_equal(out, [0, 4, 8])
        assert_equal(out.mask, [0, 1, 0])
        out = diag(out)
        control = array([[0, 0, 0], [0, 4, 0], [0, 0, 8]],
                        mask=[[0, 0, 0], [0, 1, 0], [0, 0, 0]])
        assert_equal(out, control)

    def test_axis_methods_nomask(self):
        # Test the combination nomask & methods w/ axis
        a = array([[1, 2, 3], [4, 5, 6]])

        assert_equal(a.sum(0), [5, 7, 9])
        assert_equal(a.sum(-1), [6, 15])
        assert_equal(a.sum(1), [6, 15])

        assert_equal(a.prod(0), [4, 10, 18])
        assert_equal(a.prod(-1), [6, 120])
        assert_equal(a.prod(1), [6, 120])

        assert_equal(a.min(0), [1, 2, 3])
        assert_equal(a.min(-1), [1, 4])
        assert_equal(a.min(1), [1, 4])

        assert_equal(a.max(0), [4, 5, 6])
        assert_equal(a.max(-1), [3, 6])
        assert_equal(a.max(1), [3, 6])

    @requires_memory(free_bytes=2 * 10000 * 1000 * 2)
    def test_mean_overflow(self):
        # Test overflow in masked arrays
        # gh-20272
        a = masked_array(np.full((10000, 10000), 65535, dtype=np.uint16),
                         mask=np.zeros((10000, 10000)))
        assert_equal(a.mean(), 65535.0)

class TestMaskedArrayMathMethodsComplex:
    # Test class for miscellaneous MaskedArrays methods.
    def setup_method(self):
        # Base data definition.
        x = np.array([8.375j, 7.545j, 8.828j, 8.5j, 1.757j, 5.928,
                      8.43, 7.78, 9.865, 5.878, 8.979, 4.732,
                      3.012, 6.022, 5.095, 3.116, 5.238, 3.957,
                      6.04, 9.63, 7.712, 3.382, 4.489, 6.479j,
                      7.189j, 9.645, 5.395, 4.961, 9.894, 2.893,
                      7.357, 9.828, 6.272, 3.758, 6.693, 0.993j])
        X = x.reshape(6, 6)
        XX = x.reshape(3, 2, 2, 3)

        m = np.array([0, 1, 0, 1, 0, 0,
                     1, 0, 1, 1, 0, 1,
                     0, 0, 0, 1, 0, 1,
                     0, 0, 0, 1, 1, 1,
                     1, 0, 0, 1, 0, 0,
                     0, 0, 1, 0, 1, 0])
        mx = array(data=x, mask=m)
        mX = array(data=X, mask=m.reshape(X.shape))
        mXX = array(data=XX, mask=m.reshape(XX.shape))

        m2 = np.array([1, 1, 0, 1, 0, 0,
                      1, 1, 1, 1, 0, 1,
                      0, 0, 1, 1, 0, 1,
                      0, 0, 0, 1, 1, 1,
                      1, 0, 0, 1, 1, 0,
                      0, 0, 1, 0, 1, 1])
        m2x = array(data=x, mask=m2)
        m2X = array(data=X, mask=m2.reshape(X.shape))
        m2XX = array(data=XX, mask=m2.reshape(XX.shape))
        self.d = (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX)

    def test_varstd(self):
        # Tests var & std on MaskedArrays.
        (x, X, XX, m, mx, mX, mXX, m2x, m2X, m2XX) = self.d
        assert_almost_equal(mX.var(axis=None), mX.compressed().var())
        assert_almost_equal(mX.std(axis=None), mX.compressed().std())
        assert_equal(mXX.var(axis=3).shape, XX.var(axis=3).shape)
        assert_equal(mX.var().shape, X.var().shape)
        (mXvar0, mXvar1) = (mX.var(axis=0), mX.var(axis=1))
        assert_almost_equal(mX.var(axis=None, ddof=2),
                            mX.compressed().var(ddof=2))
        assert_almost_equal(mX.std(axis=None, ddof=2),
                            mX.compressed().std(ddof=2))
        for k in range(6):
            assert_almost_equal(mXvar1[k], mX[k].compressed().var())
            assert_almost_equal(mXvar0[k], mX[:, k].compressed().var())
            assert_almost_equal(np.sqrt(mXvar0[k]),
                                mX[:, k].compressed().std())


class TestMaskedArrayFunctions:
    # Test class for miscellaneous functions.

    def setup_method(self):
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = masked_array(x, mask=m1)
        ym = masked_array(y, mask=m2)
        xm.set_fill_value(1e+20)
        self.info = (xm, ym)

    def test_masked_where_bool(self):
        x = [1, 2]
        y = masked_where(False, x)
        assert_equal(y, [1, 2])
        assert_equal(y[1], 2)

    def test_masked_equal_wlist(self):
        x = [1, 2, 3]
        mx = masked_equal(x, 3)
        assert_equal(mx, x)
        assert_equal(mx._mask, [0, 0, 1])
        mx = masked_not_equal(x, 3)
        assert_equal(mx, x)
        assert_equal(mx._mask, [1, 1, 0])

    def test_masked_equal_fill_value(self):
        x = [1, 2, 3]
        mx = masked_equal(x, 3)
        assert_equal(mx._mask, [0, 0, 1])
        assert_equal(mx.fill_value, 3)

    def test_masked_where_condition(self):
        # Tests masking functions.
        x = array([1., 2., 3., 4., 5.])
        x[2] = masked
        assert_equal(masked_where(greater(x, 2), x), masked_greater(x, 2))
        assert_equal(masked_where(greater_equal(x, 2), x),
                     masked_greater_equal(x, 2))
        assert_equal(masked_where(less(x, 2), x), masked_less(x, 2))
        assert_equal(masked_where(less_equal(x, 2), x),
                     masked_less_equal(x, 2))
        assert_equal(masked_where(not_equal(x, 2), x), masked_not_equal(x, 2))
        assert_equal(masked_where(equal(x, 2), x), masked_equal(x, 2))
        assert_equal(masked_where(not_equal(x, 2), x), masked_not_equal(x, 2))
        assert_equal(masked_where([1, 1, 0, 0, 0], [1, 2, 3, 4, 5]),
                     [99, 99, 3, 4, 5])

    def test_masked_where_oddities(self):
        # Tests some generic features.
        atest = ones((10, 10, 10), dtype=float)
        btest = zeros(atest.shape, MaskType)
        ctest = masked_where(btest, atest)
        assert_equal(atest, ctest)

    def test_masked_where_shape_constraint(self):
        a = arange(10)
        with assert_raises(IndexError):
            masked_equal(1, a)
        test = masked_equal(a, 1)
        assert_equal(test.mask, [0, 1, 0, 0, 0, 0, 0, 0, 0, 0])

    def test_masked_where_structured(self):
        # test that masked_where on a structured array sets a structured
        # mask (see issue #2972)
        a = np.zeros(10, dtype=[("A", "<f2"), ("B", "<f4")])
        with np.errstate(over="ignore"):
            # NOTE: The float16 "uses" 1e20 as mask, which overflows to inf
            #       and warns.  Unrelated to this test, but probably undesired.
            #       But NumPy previously did not warn for this overflow.
            am = np.ma.masked_where(a["A"] < 5, a)
        assert_equal(am.mask.dtype.names, am.dtype.names)
        assert_equal(am["A"],
                    np.ma.masked_array(np.zeros(10), np.ones(10)))

    def test_masked_where_mismatch(self):
        # gh-4520
        x = np.arange(10)
        y = np.arange(5)
        assert_raises(IndexError, np.ma.masked_where, y > 6, x)

    def test_masked_otherfunctions(self):
        assert_equal(masked_inside(list(range(5)), 1, 3),
                     [0, 199, 199, 199, 4])
        assert_equal(masked_outside(list(range(5)), 1, 3), [199, 1, 2, 3, 199])
        assert_equal(masked_inside(array(list(range(5)),
                                         mask=[1, 0, 0, 0, 0]), 1, 3).mask,
                     [1, 1, 1, 1, 0])
        assert_equal(masked_outside(array(list(range(5)),
                                          mask=[0, 1, 0, 0, 0]), 1, 3).mask,
                     [1, 1, 0, 0, 1])
        assert_equal(masked_equal(array(list(range(5)),
                                        mask=[1, 0, 0, 0, 0]), 2).mask,
                     [1, 0, 1, 0, 0])
        assert_equal(masked_not_equal(array([2, 2, 1, 2, 1],
                                            mask=[1, 0, 0, 0, 0]), 2).mask,
                     [1, 0, 1, 0, 1])

    def test_round(self):
        a = array([1.23456, 2.34567, 3.45678, 4.56789, 5.67890],
                  mask=[0, 1, 0, 0, 0])
        assert_equal(a.round(), [1., 2., 3., 5., 6.])
        assert_equal(a.round(1), [1.2, 2.3, 3.5, 4.6, 5.7])
        assert_equal(a.round(3), [1.235, 2.346, 3.457, 4.568, 5.679])
        b = empty_like(a)
        a.round(out=b)
        assert_equal(b, [1., 2., 3., 5., 6.])

        x = array([1., 2., 3., 4., 5.])
        c = array([1, 1, 1, 0, 0])
        x[2] = masked
        z = where(c, x, -x)
        assert_equal(z, [1., 2., 0., -4., -5])
        c[0] = masked
        z = where(c, x, -x)
        assert_equal(z, [1., 2., 0., -4., -5])
        assert_(z[0] is masked)
        assert_(z[1] is not masked)
        assert_(z[2] is masked)

    def test_round_with_output(self):
        # Testing round with an explicit output

        xm = array(np.random.uniform(0, 10, 12)).reshape(3, 4)
        xm[:, 0] = xm[0] = xm[-1, -1] = masked

        # A ndarray as explicit input
        output = np.empty((3, 4), dtype=float)
        output.fill(-9999)
        result = np.round(xm, decimals=2, out=output)
        # ... the result should be the given output
        assert_(result is output)
        assert_equal(result, xm.round(decimals=2, out=output))

        output = empty((3, 4), dtype=float)
        result = xm.round(decimals=2, out=output)
        assert_(result is output)

    def test_round_with_scalar(self):
        # Testing round with scalar/zero dimension input
        # GH issue 2244
        a = array(1.1, mask=[False])
        assert_equal(a.round(), 1)

        a = array(1.1, mask=[True])
        assert_(a.round() is masked)

        a = array(1.1, mask=[False])
        output = np.empty(1, dtype=float)
        output.fill(-9999)
        a.round(out=output)
        assert_equal(output, 1)

        a = array(1.1, mask=[False])
        output = array(-9999., mask=[True])
        a.round(out=output)
        assert_equal(output[()], 1)

        a = array(1.1, mask=[True])
        output = array(-9999., mask=[False])
        a.round(out=output)
        assert_(output[()] is masked)

    def test_identity(self):
        a = identity(5)
        assert_(isinstance(a, MaskedArray))
        assert_equal(a, np.identity(5))

    def test_power(self):
        x = -1.1
        assert_almost_equal(power(x, 2.), 1.21)
        assert_(power(x, masked) is masked)
        x = array([-1.1, -1.1, 1.1, 1.1, 0.])
        b = array([0.5, 2., 0.5, 2., -1.], mask=[0, 0, 0, 0, 1])
        y = power(x, b)
        assert_almost_equal(y, [0, 1.21, 1.04880884817, 1.21, 0.])
        assert_equal(y._mask, [1, 0, 0, 0, 1])
        b.mask = nomask
        y = power(x, b)
        assert_equal(y._mask, [1, 0, 0, 0, 1])
        z = x ** b
        assert_equal(z._mask, y._mask)
        assert_almost_equal(z, y)
        assert_almost_equal(z._data, y._data)
        x **= b
        assert_equal(x._mask, y._mask)
        assert_almost_equal(x, y)
        assert_almost_equal(x._data, y._data)

    def test_power_with_broadcasting(self):
        # Test power w/ broadcasting
        a2 = np.array([[1., 2., 3.], [4., 5., 6.]])
        a2m = array(a2, mask=[[1, 0, 0], [0, 0, 1]])
        b1 = np.array([2, 4, 3])
        b2 = np.array([b1, b1])
        b2m = array(b2, mask=[[0, 1, 0], [0, 1, 0]])

        ctrl = array([[1 ** 2, 2 ** 4, 3 ** 3], [4 ** 2, 5 ** 4, 6 ** 3]],
                     mask=[[1, 1, 0], [0, 1, 1]])
        # No broadcasting, base & exp w/ mask
        test = a2m ** b2m
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)
        # No broadcasting, base w/ mask, exp w/o mask
        test = a2m ** b2
        assert_equal(test, ctrl)
        assert_equal(test.mask, a2m.mask)
        # No broadcasting, base w/o mask, exp w/ mask
        test = a2 ** b2m
        assert_equal(test, ctrl)
        assert_equal(test.mask, b2m.mask)

        ctrl = array([[2 ** 2, 4 ** 4, 3 ** 3], [2 ** 2, 4 ** 4, 3 ** 3]],
                     mask=[[0, 1, 0], [0, 1, 0]])
        test = b1 ** b2m
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)
        test = b2m ** b1
        assert_equal(test, ctrl)
        assert_equal(test.mask, ctrl.mask)

    @pytest.mark.skipif(IS_WASM, reason="fp errors don't work in wasm")
    def test_where(self):
        # Test the where function
        x = np.array([1., 1., 1., -2., pi/2.0, 4., 5., -10., 10., 1., 2., 3.])
        y = np.array([5., 0., 3., 2., -1., -4., 0., -10., 10., 1., 0., 3.])
        m1 = [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]
        m2 = [0, 0, 1, 0, 0, 1, 1, 0, 0, 0, 0, 1]
        xm = masked_array(x, mask=m1)
        ym = masked_array(y, mask=m2)
        xm.set_fill_value(1e+20)

        d = where(xm > 2, xm, -9)
        assert_equal(d, [-9., -9., -9., -9., -9., 4.,
                         -9., -9., 10., -9., -9., 3.])
        assert_equal(d._mask, xm._mask)
        d = where(xm > 2, -9, ym)
        assert_equal(d, [5., 0., 3., 2., -1., -9.,
                         -9., -10., -9., 1., 0., -9.])
        assert_equal(d._mask, [1, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0])
        d = where(xm > 2, xm, masked)
        assert_equal(d, [-9., -9., -9., -9., -9., 4.,
                         -9., -9., 10., -9., -9., 3.])
        tmp = xm._mask.copy()
        tmp[(xm <= 2).filled(True)] = True
        assert_equal(d._mask, tmp)

        with np.errstate(invalid="warn"):
            # The fill value is 1e20, it cannot be converted to `int`:
            with pytest.warns(RuntimeWarning, match="invalid value"):
                ixm = xm.astype(int)
        d = where(ixm > 2, ixm, masked)
        assert_equal(d, [-9, -9, -9, -9, -9, 4, -9, -9, 10, -9, -9, 3])
        assert_equal(d.dtype, ixm.dtype)

    def test_where_object(self):
        a = np.array(None)
        b = masked_array(None)
        r = b.copy()
        assert_equal(np.ma.where(True, a, a), r)
        assert_equal(np.ma.where(True, b, b), r)

    def test_where_with_masked_choice(self):
        x = arange(10)
        x[3] = masked
        c = x >= 8
        # Set False to masked
        z = where(c, x, masked)
        assert_(z.dtype is x.dtype)
        assert_(z[3] is masked)
        assert_(z[4] is masked)
        assert_(z[7] is masked)
        assert_(z[8] is not masked)
        assert_(z[9] is not masked)
        assert_equal(x, z)
        # Set True to masked
        z = where(c, masked, x)
        assert_(z.dtype is x.dtype)
        assert_(z[3] is masked)
        assert_(z[4] is not masked)
        assert_(z[7] is not masked)
        assert_(z[8] is masked)
        assert_(z[9] is masked)

    def test_where_with_masked_condition(self):
        x = array([1., 2., 3., 4., 5.])
        c = array([1, 1, 1, 0, 0])
        x[2] = masked
        z = where(c, x, -x)
        assert_equal(z, [1., 2., 0., -4., -5])
        c[0] = masked
        z = where(c, x, -x)
        assert_equal(z, [1., 2., 0., -4., -5])
        assert_(z[0] is masked)
        assert_(z[1] is not masked)
        assert_(z[2] is masked)

        x = arange(1, 6)
        x[-1] = masked
        y = arange(1, 6) * 10
        y[2] = masked
        c = array([1, 1, 1, 0, 0], mask=[1, 0, 0, 0, 0])
        cm = c.filled(1)
        z = where(c, x, y)
        zm = where(cm, x, y)
        assert_equal(z, zm)
        assert_(getmask(zm) is nomask)
        assert_equal(zm, [1, 2, 3, 40, 50])
        z = where(c, masked, 1)
        assert_equal(z, [99, 99, 99, 1, 1])
        z = where(c, 1, masked)
        assert_equal(z, [99, 1, 1, 99, 99])

    def test_where_type(self):
        # Test the type conservation with where
        x = np.arange(4, dtype=np.int32)
        y = np.arange(4, dtype=np.float32) * 2.2
        test = where(x > 1.5, y, x).dtype
        control = np.find_common_type([np.int32, np.float32], [])
        assert_equal(test, control)

    def test_where_broadcast(self):
        # Issue 8599
        x = np.arange(9).reshape(3, 3)
        y = np.zeros(3)
        core = np.where([1, 0, 1], x, y)
        ma = where([1, 0, 1], x, y)

        assert_equal(core, ma)
        assert_equal(core.dtype, ma.dtype)

    def test_where_structured(self):
        # Issue 8600
        dt = np.dtype([('a', int), ('b', int)])
        x = np.array([(1, 2), (3, 4), (5, 6)], dtype=dt)
        y = np.array((10, 20), dtype=dt)
        core = np.where([0, 1, 1], x, y)
        ma = np.where([0, 1, 1], x, y)

        assert_equal(core, ma)
        assert_equal(core.dtype, ma.dtype)

    def test_where_structured_masked(self):
        dt = np.dtype([('a', int), ('b', int)])
        x = np.array([(1, 2), (3, 4), (5, 6)], dtype=dt)

        ma = where([0, 1, 1], x, masked)
        expected = masked_where([1, 0, 0], x)

        assert_equal(ma.dtype, expected.dtype)
        assert_equal(ma, expected)
        assert_equal(ma.mask, expected.mask)

    def test_masked_invalid_error(self):
        a = np.arange(5, dtype=object)
        a[3] = np.PINF
        a[2] = np.NaN
        with pytest.raises(TypeError,
                           match="not supported for the input types"):
            np.ma.masked_invalid(a)

    def test_masked_invalid_pandas(self):
        # getdata() used to be bad for pandas series due to its _data
        # attribute.  This test is a regression test mainly and may be
        # removed if getdata() is adjusted.
        class Series():
            _data = "nonsense"

            def __array__(self):
                return np.array([5, np.nan, np.inf])

        arr = np.ma.masked_invalid(Series())
        assert_array_equal(arr._data, np.array(Series()))
        assert_array_equal(arr._mask, [False, True, True])

    @pytest.mark.parametrize("copy", [True, False])
    def test_masked_invalid_full_mask(self, copy):
        # Matplotlib relied on masked_invalid always returning a full mask
        # (Also astropy projects, but were ok with it gh-22720 and gh-22842)
        a = np.ma.array([1, 2, 3, 4])
        assert a._mask is nomask
        res = np.ma.masked_invalid(a, copy=copy)
        assert res.mask is not nomask
        # mask of a should not be mutated
        assert a.mask is nomask
        assert np.may_share_memory(a._data, res._data) != copy

    def test_choose(self):
        # Test choose
        choices = [[0, 1, 2, 3], [10, 11, 12, 13],
                   [20, 21, 22, 23], [30, 31, 32, 33]]
        chosen = choose([2, 3, 1, 0], choices)
        assert_equal(chosen, array([20, 31, 12, 3]))
        chosen = choose([2, 4, 1, 0], choices, mode='clip')
        assert_equal(chosen, array([20, 31, 12, 3]))
        chosen = choose([2, 4, 1, 0], choices, mode='wrap')
        assert_equal(chosen, array([20, 1, 12, 3]))
        # Check with some masked indices
        indices_ = array([2, 4, 1, 0], mask=[1, 0, 0, 1])
        chosen = choose(indices_, choices, mode='wrap')
        assert_equal(chosen, array([99, 1, 12, 99]))
        assert_equal(chosen.mask, [1, 0, 0, 1])
        # Check with some masked choices
        choices = array(choices, mask=[[0, 0, 0, 1], [1, 1, 0, 1],
                                       [1, 0, 0, 0], [0, 0, 0, 0]])
        indices_ = [2, 3, 1, 0]
        chosen = choose(indices_, choices, mode='wrap')
        assert_equal(chosen, array([20, 31, 12, 3]))
        assert_equal(chosen.mask, [1, 0, 0, 1])

    def test_choose_with_out(self):
        # Test choose with an explicit out keyword
        choices = [[0, 1, 2, 3], [10, 11, 12, 13],
                   [20, 21, 22, 23], [30, 31, 32, 33]]
        store = empty(4, dtype=int)
        chosen = choose([2, 3, 1, 0], choices, out=store)
        assert_equal(store, array([20, 31, 12, 3]))
        assert_(store is chosen)
        # Check with some masked indices + out
        store = empty(4, dtype=int)
        indices_ = array([2, 3, 1, 0], mask=[1, 0, 0, 1])
        chosen = choose(indices_, choices, mode='wrap', out=store)
        assert_equal(store, array([99, 31, 12, 99]))
        assert_equal(store.mask, [1, 0, 0, 1])
        # Check with some masked choices + out ina ndarray !
        choices = array(choices, mask=[[0, 0, 0, 1], [1, 1, 0, 1],
                                       [1, 0, 0, 0], [0, 0, 0, 0]])
        indices_ = [2, 3, 1, 0]
        store = empty(4, dtype=int).view(ndarray)
        chosen = choose(indices_, choices, mode='wrap', out=store)
        assert_equal(store, array([999999, 31, 12, 999999]))

    def test_reshape(self):
        a = arange(10)
        a[0] = masked
        # Try the default
        b = a.reshape((5, 2))
        assert_equal(b.shape, (5, 2))
        assert_(b.flags['C'])
        # Try w/ arguments as list instead of tuple
        b = a.reshape(5, 2)
        assert_equal(b.shape, (5, 2))
        assert_(b.flags['C'])
        # Try w/ order
        b = a.reshape((5, 2), order='F')
        assert_equal(b.shape, (5, 2))
        assert_(b.flags['F'])
        # Try w/ order
        b = a.reshape(5, 2, order='F')
        assert_equal(b.shape, (5, 2))
        assert_(b.flags['F'])

        c = np.reshape(a, (2, 5))
        assert_(isinstance(c, MaskedArray))
        assert_equal(c.shape, (2, 5))
        assert_(c[0, 0] is masked)
        assert_(c.flags['C'])

    def test_make_mask_descr(self):
        # Flexible
        ntype = [('a', float), ('b', float)]
        test = make_mask_descr(ntype)
        assert_equal(test, [('a', bool), ('b', bool)])
        assert_(test is make_mask_descr(test))

        # Standard w/ shape
        ntype = (float, 2)
        test = make_mask_descr(ntype)
        assert_equal(test, (bool, 2))
        assert_(test is make_mask_descr(test))

        # Standard standard
        ntype = float
        test = make_mask_descr(ntype)
        assert_equal(test, np.dtype(bool))
        assert_(test is make_mask_descr(test))

        # Nested
        ntype = [('a', float), ('b', [('ba', float), ('bb', float)])]
        test = make_mask_descr(ntype)
        control = np.dtype([('a', 'b1'), ('b', [('ba', 'b1'), ('bb', 'b1')])])
        assert_equal(test, control)
        assert_(test is make_mask_descr(test))

        # Named+ shape
        ntype = [('a', (float, 2))]
        test = make_mask_descr(ntype)
        assert_equal(test, np.dtype([('a', (bool, 2))]))
        assert_(test is make_mask_descr(test))

        # 2 names
        ntype = [(('A', 'a'), float)]
        test = make_mask_descr(ntype)
        assert_equal(test, np.dtype([(('A', 'a'), bool)]))
        assert_(test is make_mask_descr(test))

        # nested boolean types should preserve identity
        base_type = np.dtype([('a', int, 3)])
        base_mtype = make_mask_descr(base_type)
        sub_type = np.dtype([('a', int), ('b', base_mtype)])
        test = make_mask_descr(sub_type)
        assert_equal(test, np.dtype([('a', bool), ('b', [('a', bool, 3)])]))
        assert_(test.fields['b'][0] is base_mtype)

    def test_make_mask(self):
        # Test make_mask
        # w/ a list as an input
        mask = [0, 1]
        test = make_mask(mask)
        assert_equal(test.dtype, MaskType)
        assert_equal(test, [0, 1])
        # w/ a ndarray as an input
        mask = np.array([0, 1], dtype=bool)
        test = make_mask(mask)
        assert_equal(test.dtype, MaskType)
        assert_equal(test, [0, 1])
        # w/ a flexible-type ndarray as an input - use default
        mdtype = [('a', bool), ('b', bool)]
        mask = np.array([(0, 0), (0, 1)], dtype=mdtype)
        test = make_mask(mask)
        assert_equal(test.dtype, MaskType)
        assert_equal(test, [1, 1])
        # w/ a flexible-type ndarray as an input - use input dtype
        mdtype = [('a', bool), ('b', bool)]
        mask = np.array([(0, 0), (0, 1)], dtype=mdtype)
        test = make_mask(mask, dtype=mask.dtype)
        assert_equal(test.dtype, mdtype)
        assert_equal(test, mask)
        # w/ a flexible-type ndarray as an input - use input dtype
        mdtype = [('a', float), ('b', float)]
        bdtype = [('a', bool), ('b', bool)]
        mask = np.array([(0, 0), (0, 1)], dtype=mdtype)
        test = make_mask(mask, dtype=mask.dtype)
        assert_equal(test.dtype, bdtype)
        assert_equal(test, np.array([(0, 0), (0, 1)], dtype=bdtype))
        # Ensure this also works for void
        mask = np.array((False, True), dtype='?,?')[()]
        assert_(isinstance(mask, np.void))
        test = make_mask(mask, dtype=mask.dtype)
        assert_equal(test, mask)
        assert_(test is not mask)
        mask = np.array((0, 1), dtype='i4,i4')[()]
        test2 = make_mask(mask, dtype=mask.dtype)
        assert_equal(test2, test)
        # test that nomask is returned when m is nomask.
        bools = [True, False]
        dtypes = [MaskType, float]
        msgformat = 'copy=%s, shrink=%s, dtype=%s'
        for cpy, shr, dt in itertools.product(bools, bools, dtypes):
            res = make_mask(nomask, copy=cpy, shrink=shr, dtype=dt)
            assert_(res is nomask, msgformat % (cpy, shr, dt))

    def test_mask_or(self):
        # Initialize
        mtype = [('a', bool), ('b', bool)]
        mask = np.array([(0, 0), (0, 1), (1, 0), (0, 0)], dtype=mtype)
        # Test using nomask as input
        test = mask_or(mask, nomask)
        assert_equal(test, mask)
        test = mask_or(nomask, mask)
        assert_equal(test, mask)
        # Using False as input
        test = mask_or(mask, False)
        assert_equal(test, mask)
        # Using another array w / the same dtype
        other = np.array([(0, 1), (0, 1), (0, 1), (0, 1)], dtype=mtype)
        test = mask_or(mask, other)
        control = np.array([(0, 1), (0, 1), (1, 1), (0, 1)], dtype=mtype)
        assert_equal(test, control)
        # Using another array w / a different dtype
        othertype = [('A', bool), ('B', bool)]
        other = np.array([(0, 1), (0, 1), (0, 1), (0, 1)], dtype=othertype)
        try:
            test = mask_or(mask, other)
        except ValueError:
            pass
        # Using nested arrays
        dtype = [('a', bool), ('b', [('ba', bool), ('bb', bool)])]
        amask = np.array([(0, (1, 0)), (0, (1, 0))], dtype=dtype)
        bmask = np.array([(1, (0, 1)), (0, (0, 0))], dtype=dtype)
        cntrl = np.array([(1, (1, 1)), (0, (1, 0))], dtype=dtype)
        assert_equal(mask_or(amask, bmask), cntrl)

    def test_flatten_mask(self):
        # Tests flatten mask
        # Standard dtype
        mask = np.array([0, 0, 1], dtype=bool)
        assert_equal(flatten_mask(mask), mask)
        # Flexible dtype
        mask = np.array([(0, 0), (0, 1)], dtype=[('a', bool), ('b', bool)])
        test = flatten_mask(mask)
        control = np.array([0, 0, 0, 1], dtype=bool)
        assert_equal(test, control)

        mdtype = [('a', bool), ('b', [('ba', bool), ('bb', bool)])]
        data = [(0, (0, 0)), (0, (0, 1))]
        mask = np.array(data, dtype=mdtype)
        test = flatten_mask(mask)
        control = np.array([0, 0, 0, 0, 0, 1], dtype=bool)
        assert_equal(test, control)

    def test_on_ndarray(self):
        # Test functions on ndarrays
        a = np.array([1, 2, 3, 4])
        m = array(a, mask=False)
        test = anom(a)
        assert_equal(test, m.anom())
        test = reshape(a, (2, 2))
        assert_equal(test, m.reshape(2, 2))

    def test_compress(self):
        # Test compress function on ndarray and masked array
        # Address Github #2495.
        arr = np.arange(8)
        arr.shape = 4, 2
        cond = np.array([True, False, True, True])
        control = arr[[0, 2, 3]]
        test = np.ma.compress(cond, arr, axis=0)
        assert_equal(test, control)
        marr = np.ma.array(arr)
        test = np.ma.compress(cond, marr, axis=0)
        assert_equal(test, control)

    def test_compressed(self):
        # Test ma.compressed function.
        # Address gh-4026
        a = np.ma.array([1, 2])
        test = np.ma.compressed(a)
        assert_(type(test) is np.ndarray)

        # Test case when input data is ndarray subclass
        class A(np.ndarray):
            pass

        a = np.ma.array(A(shape=0))
        test = np.ma.compressed(a)
        assert_(type(test) is A)

        # Test that compress flattens
        test = np.ma.compressed([[1],[2]])
        assert_equal(test.ndim, 1)
        test = np.ma.compressed([[[[[1]]]]])
        assert_equal(test.ndim, 1)

        # Test case when input is MaskedArray subclass
        class M(MaskedArray):
            pass

        test = np.ma.compressed(M([[[]], [[]]]))
        assert_equal(test.ndim, 1)

        # with .compressed() overridden
        class M(MaskedArray):
            def compressed(self):
                return 42

        test = np.ma.compressed(M([[[]], [[]]]))
        assert_equal(test, 42)

    def test_convolve(self):
        a = masked_equal(np.arange(5), 2)
        b = np.array([1, 1])
        test = np.ma.convolve(a, b)
        assert_equal(test, masked_equal([0, 1, -1, -1, 7, 4], -1))

        test = np.ma.convolve(a, b, propagate_mask=False)
        assert_equal(test, masked_equal([0, 1, 1, 3, 7, 4], -1))

        test = np.ma.convolve([1, 1], [1, 1, 1])
        assert_equal(test, masked_equal([1, 2, 2, 1], -1))

        a = [1, 1]
        b = masked_equal([1, -1, -1, 1], -1)
        test = np.ma.convolve(a, b, propagate_mask=False)
        assert_equal(test, masked_equal([1, 1, -1, 1, 1], -1))
        test = np.ma.convolve(a, b, propagate_mask=True)
        assert_equal(test, masked_equal([-1, -1, -1, -1, -1], -1))


class TestMaskedFields:

    def setup_method(self):
        ilist = [1, 2, 3, 4, 5]
        flist = [1.1, 2.2, 3.3, 4.4, 5.5]
        slist = ['one', 'two', 'three', 'four', 'five']
        ddtype = [('a', int), ('b', float), ('c', '|S8')]
        mdtype = [('a', bool), ('b', bool), ('c', bool)]
        mask = [0, 1, 0, 0, 1]
        base = array(list(zip(ilist, flist, slist)), mask=mask, dtype=ddtype)
        self.data = dict(base=base, mask=mask, ddtype=ddtype, mdtype=mdtype)

    def test_set_records_masks(self):
        base = self.data['base']
        mdtype = self.data['mdtype']
        # Set w/ nomask or masked
        base.mask = nomask
        assert_equal_records(base._mask, np.zeros(base.shape, dtype=mdtype))
        base.mask = masked
        assert_equal_records(base._mask, np.ones(base.shape, dtype=mdtype))
        # Set w/ simple boolean
        base.mask = False
        assert_equal_records(base._mask, np.zeros(base.shape, dtype=mdtype))
        base.mask = True
        assert_equal_records(base._mask, np.ones(base.shape, dtype=mdtype))
        # Set w/ list
        base.mask = [0, 0, 0, 1, 1]
        assert_equal_records(base._mask,
                             np.array([(x, x, x) for x in [0, 0, 0, 1, 1]],
                                      dtype=mdtype))

    def test_set_record_element(self):
        # Check setting an element of a record)
        base = self.data['base']
        (base_a, base_b, base_c) = (base['a'], base['b'], base['c'])
        base[0] = (pi, pi, 'pi')

        assert_equal(base_a.dtype, int)
        assert_equal(base_a._data, [3, 2, 3, 4, 5])

        assert_equal(base_b.dtype, float)
        assert_equal(base_b._data, [pi, 2.2, 3.3, 4.4, 5.5])

        assert_equal(base_c.dtype, '|S8')
        assert_equal(base_c._data,
                     [b'pi', b'two', b'three', b'four', b'five'])

    def test_set_record_slice(self):
        base = self.data['base']
        (base_a, base_b, base_c) = (base['a'], base['b'], base['c'])
        base[:3] = (pi, pi, 'pi')

        assert_equal(base_a.dtype, int)
        assert_equal(base_a._data, [3, 3, 3, 4, 5])

        assert_equal(base_b.dtype, float)
        assert_equal(base_b._data, [pi, pi, pi, 4.4, 5.5])

        assert_equal(base_c.dtype, '|S8')
        assert_equal(base_c._data,
                     [b'pi', b'pi', b'pi', b'four', b'five'])

    def test_mask_element(self):
        "Check record access"
        base = self.data['base']
        base[0] = masked

        for n in ('a', 'b', 'c'):
            assert_equal(base[n].mask, [1, 1, 0, 0, 1])
            assert_equal(base[n]._data, base._data[n])

    def test_getmaskarray(self):
        # Test getmaskarray on flexible dtype
        ndtype = [('a', int), ('b', float)]
        test = empty(3, dtype=ndtype)
        assert_equal(getmaskarray(test),
                     np.array([(0, 0), (0, 0), (0, 0)],
                              dtype=[('a', '|b1'), ('b', '|b1')]))
        test[:] = masked
        assert_equal(getmaskarray(test),
                     np.array([(1, 1), (1, 1), (1, 1)],
                              dtype=[('a', '|b1'), ('b', '|b1')]))

    def test_view(self):
        # Test view w/ flexible dtype
        iterator = list(zip(np.arange(10), np.random.rand(10)))
        data = np.array(iterator)
        a = array(iterator, dtype=[('a', float), ('b', float)])
        a.mask[0] = (1, 0)
        controlmask = np.array([1] + 19 * [0], dtype=bool)
        # Transform globally to simple dtype
        test = a.view(float)
        assert_equal(test, data.ravel())
        assert_equal(test.mask, controlmask)
        # Transform globally to dty
        test = a.view((float, 2))
        assert_equal(test, data)
        assert_equal(test.mask, controlmask.reshape(-1, 2))

    def test_getitem(self):
        ndtype = [('a', float), ('b', float)]
        a = array(list(zip(np.random.rand(10), np.arange(10))), dtype=ndtype)
        a.mask = np.array(list(zip([0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
                                   [1, 0, 0, 0, 0, 0, 0, 0, 1, 0])),
                          dtype=[('a', bool), ('b', bool)])

        def _test_index(i):
            assert_equal(type(a[i]), mvoid)
            assert_equal_records(a[i]._data, a._data[i])
            assert_equal_records(a[i]._mask, a._mask[i])

            assert_equal(type(a[i, ...]), MaskedArray)
            assert_equal_records(a[i,...]._data, a._data[i,...])
            assert_equal_records(a[i,...]._mask, a._mask[i,...])

        _test_index(1)   # No mask
        _test_index(0)   # One element masked
        _test_index(-2)  # All element masked

    def test_setitem(self):
        # Issue 4866: check that one can set individual items in [record][col]
        # and [col][record] order
        ndtype = np.dtype([('a', float), ('b', int)])
        ma = np.ma.MaskedArray([(1.0, 1), (2.0, 2)], dtype=ndtype)
        ma['a'][1] = 3.0
        assert_equal(ma['a'], np.array([1.0, 3.0]))
        ma[1]['a'] = 4.0
        assert_equal(ma['a'], np.array([1.0, 4.0]))
        # Issue 2403
        mdtype = np.dtype([('a', bool), ('b', bool)])
        # soft mask
        control = np.array([(False, True), (True, True)], dtype=mdtype)
        a = np.ma.masked_all((2,), dtype=ndtype)
        a['a'][0] = 2
        assert_equal(a.mask, control)
        a = np.ma.masked_all((2,), dtype=ndtype)
        a[0]['a'] = 2
        assert_equal(a.mask, control)
        # hard mask
        control = np.array([(True, True), (True, True)], dtype=mdtype)
        a = np.ma.masked_all((2,), dtype=ndtype)
        a.harden_mask()
        a['a'][0] = 2
        assert_equal(a.mask, control)
        a = np.ma.masked_all((2,), dtype=ndtype)
        a.harden_mask()
        a[0]['a'] = 2
        assert_equal(a.mask, control)

    def test_setitem_scalar(self):
        # 8510
        mask_0d = np.ma.masked_array(1, mask=True)
        arr = np.ma.arange(3)
        arr[0] = mask_0d
        assert_array_equal(arr.mask, [True, False, False])

    def test_element_len(self):
        # check that len() works for mvoid (Github issue #576)
        for rec in self.data['base']:
            assert_equal(len(rec), len(self.data['ddtype']))


class TestMaskedObjectArray:

    def test_getitem(self):
        arr = np.ma.array([None, None])
        for dt in [float, object]:
            a0 = np.eye(2).astype(dt)
            a1 = np.eye(3).astype(dt)
            arr[0] = a0
            arr[1] = a1

            assert_(arr[0] is a0)
            assert_(arr[1] is a1)
            assert_(isinstance(arr[0,...], MaskedArray))
            assert_(isinstance(arr[1,...], MaskedArray))
            assert_(arr[0,...][()] is a0)
            assert_(arr[1,...][()] is a1)

            arr[0] = np.ma.masked

            assert_(arr[1] is a1)
            assert_(isinstance(arr[0,...], MaskedArray))
            assert_(isinstance(arr[1,...], MaskedArray))
            assert_equal(arr[0,...].mask, True)
            assert_(arr[1,...][()] is a1)

            # gh-5962 - object arrays of arrays do something special
            assert_equal(arr[0].data, a0)
            assert_equal(arr[0].mask, True)
            assert_equal(arr[0,...][()].data, a0)
            assert_equal(arr[0,...][()].mask, True)

    def test_nested_ma(self):

        arr = np.ma.array([None, None])
        # set the first object to be an unmasked masked constant. A little fiddly
        arr[0,...] = np.array([np.ma.masked], object)[0,...]

        # check the above line did what we were aiming for
        assert_(arr.data[0] is np.ma.masked)

        # test that getitem returned the value by identity
        assert_(arr[0] is np.ma.masked)

        # now mask the masked value!
        arr[0] = np.ma.masked
        assert_(arr[0] is np.ma.masked)


class TestMaskedView:

    def setup_method(self):
        iterator = list(zip(np.arange(10), np.random.rand(10)))
        data = np.array(iterator)
        a = array(iterator, dtype=[('a', float), ('b', float)])
        a.mask[0] = (1, 0)
        controlmask = np.array([1] + 19 * [0], dtype=bool)
        self.data = (data, a, controlmask)

    def test_view_to_nothing(self):
        (data, a, controlmask) = self.data
        test = a.view()
        assert_(isinstance(test, MaskedArray))
        assert_equal(test._data, a._data)
        assert_equal(test._mask, a._mask)

    def test_view_to_type(self):
        (data, a, controlmask) = self.data
        test = a.view(np.ndarray)
        assert_(not isinstance(test, MaskedArray))
        assert_equal(test, a._data)
        assert_equal_records(test, data.view(a.dtype).squeeze())

    def test_view_to_simple_dtype(self):
        (data, a, controlmask) = self.data
        # View globally
        test = a.view(float)
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, data.ravel())
        assert_equal(test.mask, controlmask)

    def test_view_to_flexible_dtype(self):
        (data, a, controlmask) = self.data

        test = a.view([('A', float), ('B', float)])
        assert_equal(test.mask.dtype.names, ('A', 'B'))
        assert_equal(test['A'], a['a'])
        assert_equal(test['B'], a['b'])

        test = a[0].view([('A', float), ('B', float)])
        assert_(isinstance(test, MaskedArray))
        assert_equal(test.mask.dtype.names, ('A', 'B'))
        assert_equal(test['A'], a['a'][0])
        assert_equal(test['B'], a['b'][0])

        test = a[-1].view([('A', float), ('B', float)])
        assert_(isinstance(test, MaskedArray))
        assert_equal(test.dtype.names, ('A', 'B'))
        assert_equal(test['A'], a['a'][-1])
        assert_equal(test['B'], a['b'][-1])

    def test_view_to_subdtype(self):
        (data, a, controlmask) = self.data
        # View globally
        test = a.view((float, 2))
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, data)
        assert_equal(test.mask, controlmask.reshape(-1, 2))
        # View on 1 masked element
        test = a[0].view((float, 2))
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, data[0])
        assert_equal(test.mask, (1, 0))
        # View on 1 unmasked element
        test = a[-1].view((float, 2))
        assert_(isinstance(test, MaskedArray))
        assert_equal(test, data[-1])

    def test_view_to_dtype_and_type(self):
        (data, a, controlmask) = self.data

        test = a.view((float, 2), np.recarray)
        assert_equal(test, data)
        assert_(isinstance(test, np.recarray))
        assert_(not isinstance(test, MaskedArray))


class TestOptionalArgs:
    def test_ndarrayfuncs(self):
        # test axis arg behaves the same as ndarray (including multiple axes)

        d = np.arange(24.0).reshape((2,3,4))
        m = np.zeros(24, dtype=bool).reshape((2,3,4))
        # mask out last element of last dimension
        m[:,:,-1] = True
        a = np.ma.array(d, mask=m)

        def testaxis(f, a, d):
            numpy_f = numpy.__getattribute__(f)
            ma_f = np.ma.__getattribute__(f)

            # test axis arg
            assert_equal(ma_f(a, axis=1)[...,:-1], numpy_f(d[...,:-1], axis=1))
            assert_equal(ma_f(a, axis=(0,1))[...,:-1],
                         numpy_f(d[...,:-1], axis=(0,1)))

        def testkeepdims(f, a, d):
            numpy_f = numpy.__getattribute__(f)
            ma_f = np.ma.__getattribute__(f)

            # test keepdims arg
            assert_equal(ma_f(a, keepdims=True).shape,
                         numpy_f(d, keepdims=True).shape)
            assert_equal(ma_f(a, keepdims=False).shape,
                         numpy_f(d, keepdims=False).shape)

            # test both at once
            assert_equal(ma_f(a, axis=1, keepdims=True)[...,:-1],
                         numpy_f(d[...,:-1], axis=1, keepdims=True))
            assert_equal(ma_f(a, axis=(0,1), keepdims=True)[...,:-1],
                         numpy_f(d[...,:-1], axis=(0,1), keepdims=True))

        for f in ['sum', 'prod', 'mean', 'var', 'std']:
            testaxis(f, a, d)
            testkeepdims(f, a, d)

        for f in ['min', 'max']:
            testaxis(f, a, d)

        d = (np.arange(24).reshape((2,3,4))%2 == 0)
        a = np.ma.array(d, mask=m)
        for f in ['all', 'any']:
            testaxis(f, a, d)
            testkeepdims(f, a, d)

    def test_count(self):
        # test np.ma.count specially

        d = np.arange(24.0).reshape((2,3,4))
        m = np.zeros(24, dtype=bool).reshape((2,3,4))
        m[:,0,:] = True
        a = np.ma.array(d, mask=m)

        assert_equal(count(a), 16)
        assert_equal(count(a, axis=1), 2*ones((2,4)))
        assert_equal(count(a, axis=(0,1)), 4*ones((4,)))
        assert_equal(count(a, keepdims=True), 16*ones((1,1,1)))
        assert_equal(count(a, axis=1, keepdims=True), 2*ones((2,1,4)))
        assert_equal(count(a, axis=(0,1), keepdims=True), 4*ones((1,1,4)))
        assert_equal(count(a, axis=-2), 2*ones((2,4)))
        assert_raises(ValueError, count, a, axis=(1,1))
        assert_raises(np.AxisError, count, a, axis=3)

        # check the 'nomask' path
        a = np.ma.array(d, mask=nomask)

        assert_equal(count(a), 24)
        assert_equal(count(a, axis=1), 3*ones((2,4)))
        assert_equal(count(a, axis=(0,1)), 6*ones((4,)))
        assert_equal(count(a, keepdims=True), 24*ones((1,1,1)))
        assert_equal(np.ndim(count(a, keepdims=True)), 3)
        assert_equal(count(a, axis=1, keepdims=True), 3*ones((2,1,4)))
        assert_equal(count(a, axis=(0,1), keepdims=True), 6*ones((1,1,4)))
        assert_equal(count(a, axis=-2), 3*ones((2,4)))
        assert_raises(ValueError, count, a, axis=(1,1))
        assert_raises(np.AxisError, count, a, axis=3)

        # check the 'masked' singleton
        assert_equal(count(np.ma.masked), 0)

        # check 0-d arrays do not allow axis > 0
        assert_raises(np.AxisError, count, np.ma.array(1), axis=1)


class TestMaskedConstant:
    def _do_add_test(self, add):
        # sanity check
        assert_(add(np.ma.masked, 1) is np.ma.masked)

        # now try with a vector
        vector = np.array([1, 2, 3])
        result = add(np.ma.masked, vector)

        # lots of things could go wrong here
        assert_(result is not np.ma.masked)
        assert_(not isinstance(result, np.ma.core.MaskedConstant))
        assert_equal(result.shape, vector.shape)
        assert_equal(np.ma.getmask(result), np.ones(vector.shape, dtype=bool))

    def test_ufunc(self):
        self._do_add_test(np.add)

    def test_operator(self):
        self._do_add_test(lambda a, b: a + b)

    def test_ctor(self):
        m = np.ma.array(np.ma.masked)

        # most importantly, we do not want to create a new MaskedConstant
        # instance
        assert_(not isinstance(m, np.ma.core.MaskedConstant))
        assert_(m is not np.ma.masked)

    def test_repr(self):
        # copies should not exist, but if they do, it should be obvious that
        # something is wrong
        assert_equal(repr(np.ma.masked), 'masked')

        # create a new instance in a weird way
        masked2 = np.ma.MaskedArray.__new__(np.ma.core.MaskedConstant)
        assert_not_equal(repr(masked2), 'masked')

    def test_pickle(self):
        from io import BytesIO

        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            with BytesIO() as f:
                pickle.dump(np.ma.masked, f, protocol=proto)
                f.seek(0)
                res = pickle.load(f)
            assert_(res is np.ma.masked)

    def test_copy(self):
        # gh-9328
        # copy is a no-op, like it is with np.True_
        assert_equal(
            np.ma.masked.copy() is np.ma.masked,
            np.True_.copy() is np.True_)

    def test__copy(self):
        import copy
        assert_(
            copy.copy(np.ma.masked) is np.ma.masked)

    def test_deepcopy(self):
        import copy
        assert_(
            copy.deepcopy(np.ma.masked) is np.ma.masked)

    def test_immutable(self):
        orig = np.ma.masked
        assert_raises(np.ma.core.MaskError, operator.setitem, orig, (), 1)
        assert_raises(ValueError,operator.setitem, orig.data, (), 1)
        assert_raises(ValueError, operator.setitem, orig.mask, (), False)

        view = np.ma.masked.view(np.ma.MaskedArray)
        assert_raises(ValueError, operator.setitem, view, (), 1)
        assert_raises(ValueError, operator.setitem, view.data, (), 1)
        assert_raises(ValueError, operator.setitem, view.mask, (), False)

    def test_coercion_int(self):
        a_i = np.zeros((), int)
        assert_raises(MaskError, operator.setitem, a_i, (), np.ma.masked)
        assert_raises(MaskError, int, np.ma.masked)

    def test_coercion_float(self):
        a_f = np.zeros((), float)
        assert_warns(UserWarning, operator.setitem, a_f, (), np.ma.masked)
        assert_(np.isnan(a_f[()]))

    @pytest.mark.xfail(reason="See gh-9750")
    def test_coercion_unicode(self):
        a_u = np.zeros((), 'U10')
        a_u[()] = np.ma.masked
        assert_equal(a_u[()], '--')

    @pytest.mark.xfail(reason="See gh-9750")
    def test_coercion_bytes(self):
        a_b = np.zeros((), 'S10')
        a_b[()] = np.ma.masked
        assert_equal(a_b[()], b'--')

    def test_subclass(self):
        # https://github.com/astropy/astropy/issues/6645
        class Sub(type(np.ma.masked)): pass

        a = Sub()
        assert_(a is Sub())
        assert_(a is not np.ma.masked)
        assert_not_equal(repr(a), 'masked')

    def test_attributes_readonly(self):
        assert_raises(AttributeError, setattr, np.ma.masked, 'shape', (1,))
        assert_raises(AttributeError, setattr, np.ma.masked, 'dtype', np.int64)


class TestMaskedWhereAliases:

    # TODO: Test masked_object, masked_equal, ...

    def test_masked_values(self):
        res = masked_values(np.array([-32768.0]), np.int16(-32768))
        assert_equal(res.mask, [True])

        res = masked_values(np.inf, np.inf)
        assert_equal(res.mask, True)

        res = np.ma.masked_values(np.inf, -np.inf)
        assert_equal(res.mask, False)

        res = np.ma.masked_values([1, 2, 3, 4], 5, shrink=True)
        assert_(res.mask is np.ma.nomask)

        res = np.ma.masked_values([1, 2, 3, 4], 5, shrink=False)
        assert_equal(res.mask, [False] * 4)


def test_masked_array():
    a = np.ma.array([0, 1, 2, 3], mask=[0, 0, 1, 0])
    assert_equal(np.argwhere(a), [[1], [3]])

def test_masked_array_no_copy():
    # check nomask array is updated in place
    a = np.ma.array([1, 2, 3, 4])
    _ = np.ma.masked_where(a == 3, a, copy=False)
    assert_array_equal(a.mask, [False, False, True, False])
    # check masked array is updated in place
    a = np.ma.array([1, 2, 3, 4], mask=[1, 0, 0, 0])
    _ = np.ma.masked_where(a == 3, a, copy=False)
    assert_array_equal(a.mask, [True, False, True, False])
    # check masked array with masked_invalid is updated in place
    a = np.ma.array([np.inf, 1, 2, 3, 4])
    _ = np.ma.masked_invalid(a, copy=False)
    assert_array_equal(a.mask, [True, False, False, False, False])

def test_append_masked_array():
    a = np.ma.masked_equal([1,2,3], value=2)
    b = np.ma.masked_equal([4,3,2], value=2)

    result = np.ma.append(a, b)
    expected_data = [1, 2, 3, 4, 3, 2]
    expected_mask = [False, True, False, False, False, True]
    assert_array_equal(result.data, expected_data)
    assert_array_equal(result.mask, expected_mask)

    a = np.ma.masked_all((2,2))
    b = np.ma.ones((3,1))

    result = np.ma.append(a, b)
    expected_data = [1] * 3
    expected_mask = [True] * 4 + [False] * 3
    assert_array_equal(result.data[-3], expected_data)
    assert_array_equal(result.mask, expected_mask)

    result = np.ma.append(a, b, axis=None)
    assert_array_equal(result.data[-3], expected_data)
    assert_array_equal(result.mask, expected_mask)


def test_append_masked_array_along_axis():
    a = np.ma.masked_equal([1,2,3], value=2)
    b = np.ma.masked_values([[4, 5, 6], [7, 8, 9]], 7)

    # When `axis` is specified, `values` must have the correct shape.
    assert_raises(ValueError, np.ma.append, a, b, axis=0)

    result = np.ma.append(a[np.newaxis,:], b, axis=0)
    expected = np.ma.arange(1, 10)
    expected[[1, 6]] = np.ma.masked
    expected = expected.reshape((3,3))
    assert_array_equal(result.data, expected.data)
    assert_array_equal(result.mask, expected.mask)

def test_default_fill_value_complex():
    # regression test for Python 3, where 'unicode' was not defined
    assert_(default_fill_value(1 + 1j) == 1.e20 + 0.0j)


def test_ufunc_with_output():
    # check that giving an output argument always returns that output.
    # Regression test for gh-8416.
    x = array([1., 2., 3.], mask=[0, 0, 1])
    y = np.add(x, 1., out=x)
    assert_(y is x)


def test_ufunc_with_out_varied():
    """ Test that masked arrays are immune to gh-10459 """
    # the mask of the output should not affect the result, however it is passed
    a        = array([ 1,  2,  3], mask=[1, 0, 0])
    b        = array([10, 20, 30], mask=[1, 0, 0])
    out      = array([ 0,  0,  0], mask=[0, 0, 1])
    expected = array([11, 22, 33], mask=[1, 0, 0])

    out_pos = out.copy()
    res_pos = np.add(a, b, out_pos)

    out_kw = out.copy()
    res_kw = np.add(a, b, out=out_kw)

    out_tup = out.copy()
    res_tup = np.add(a, b, out=(out_tup,))

    assert_equal(res_kw.mask,  expected.mask)
    assert_equal(res_kw.data,  expected.data)
    assert_equal(res_tup.mask, expected.mask)
    assert_equal(res_tup.data, expected.data)
    assert_equal(res_pos.mask, expected.mask)
    assert_equal(res_pos.data, expected.data)


def test_astype_mask_ordering():
    descr = np.dtype([('v', int, 3), ('x', [('y', float)])])
    x = array([
        [([1, 2, 3], (1.0,)),  ([1, 2, 3], (2.0,))],
        [([1, 2, 3], (3.0,)),  ([1, 2, 3], (4.0,))]], dtype=descr)
    x[0]['v'][0] = np.ma.masked

    x_a = x.astype(descr)
    assert x_a.dtype.names == np.dtype(descr).names
    assert x_a.mask.dtype.names == np.dtype(descr).names
    assert_equal(x, x_a)

    assert_(x is x.astype(x.dtype, copy=False))
    assert_equal(type(x.astype(x.dtype, subok=False)), np.ndarray)

    x_f = x.astype(x.dtype, order='F')
    assert_(x_f.flags.f_contiguous)
    assert_(x_f.mask.flags.f_contiguous)

    # Also test the same indirectly, via np.array
    x_a2 = np.array(x, dtype=descr, subok=True)
    assert x_a2.dtype.names == np.dtype(descr).names
    assert x_a2.mask.dtype.names == np.dtype(descr).names
    assert_equal(x, x_a2)

    assert_(x is np.array(x, dtype=descr, copy=False, subok=True))

    x_f2 = np.array(x, dtype=x.dtype, order='F', subok=True)
    assert_(x_f2.flags.f_contiguous)
    assert_(x_f2.mask.flags.f_contiguous)


@pytest.mark.parametrize('dt1', num_dts, ids=num_ids)
@pytest.mark.parametrize('dt2', num_dts, ids=num_ids)
@pytest.mark.filterwarnings('ignore::numpy.ComplexWarning')
def test_astype_basic(dt1, dt2):
    # See gh-12070
    src = np.ma.array(ones(3, dt1), fill_value=1)
    dst = src.astype(dt2)

    assert_(src.fill_value == 1)
    assert_(src.dtype == dt1)
    assert_(src.fill_value.dtype == dt1)

    assert_(dst.fill_value == 1)
    assert_(dst.dtype == dt2)
    assert_(dst.fill_value.dtype == dt2)

    assert_equal(src, dst)


def test_fieldless_void():
    dt = np.dtype([])  # a void dtype with no fields
    x = np.empty(4, dt)

    # these arrays contain no values, so there's little to test - but this
    # shouldn't crash
    mx = np.ma.array(x)
    assert_equal(mx.dtype, x.dtype)
    assert_equal(mx.shape, x.shape)

    mx = np.ma.array(x, mask=x)
    assert_equal(mx.dtype, x.dtype)
    assert_equal(mx.shape, x.shape)


def test_mask_shape_assignment_does_not_break_masked():
    a = np.ma.masked
    b = np.ma.array(1, mask=a.mask)
    b.shape = (1,)
    assert_equal(a.mask.shape, ())

@pytest.mark.skipif(sys.flags.optimize > 1,
                    reason="no docstrings present to inspect when PYTHONOPTIMIZE/Py_OptimizeFlag > 1")
def test_doc_note():
    def method(self):
        """This docstring

        Has multiple lines

        And notes

        Notes
        -----
        original note
        """
        pass

    expected_doc = """This docstring

Has multiple lines

And notes

Notes
-----
note

original note"""

    assert_equal(np.ma.core.doc_note(method.__doc__, "note"), expected_doc)
