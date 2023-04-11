# pylint: disable-msg=W0611, W0612, W0511,R0201
"""Tests suite for MaskedArray & subclassing.

:author: Pierre Gerard-Marchant
:contact: pierregm_at_uga_dot_edu
:version: $Id: test_subclassing.py 3473 2007-10-29 15:18:13Z jarrod.millman $

"""
import numpy as np
from numpy.lib.mixins import NDArrayOperatorsMixin
from numpy.testing import assert_, assert_raises
from numpy.ma.testutils import assert_equal
from numpy.ma.core import (
    array, arange, masked, MaskedArray, masked_array, log, add, hypot,
    divide, asarray, asanyarray, nomask
    )
# from numpy.ma.core import (

def assert_startswith(a, b):
    # produces a better error message than assert_(a.startswith(b))
    assert_equal(a[:len(b)], b)

class SubArray(np.ndarray):
    # Defines a generic np.ndarray subclass, that stores some metadata
    # in the  dictionary `info`.
    def __new__(cls,arr,info={}):
        x = np.asanyarray(arr).view(cls)
        x.info = info.copy()
        return x

    def __array_finalize__(self, obj):
        super().__array_finalize__(obj)
        self.info = getattr(obj, 'info', {}).copy()
        return

    def __add__(self, other):
        result = super().__add__(other)
        result.info['added'] = result.info.get('added', 0) + 1
        return result

    def __iadd__(self, other):
        result = super().__iadd__(other)
        result.info['iadded'] = result.info.get('iadded', 0) + 1
        return result


subarray = SubArray


class SubMaskedArray(MaskedArray):
    """Pure subclass of MaskedArray, keeping some info on subclass."""
    def __new__(cls, info=None, **kwargs):
        obj = super().__new__(cls, **kwargs)
        obj._optinfo['info'] = info
        return obj


class MSubArray(SubArray, MaskedArray):

    def __new__(cls, data, info={}, mask=nomask):
        subarr = SubArray(data, info)
        _data = MaskedArray.__new__(cls, data=subarr, mask=mask)
        _data.info = subarr.info
        return _data

    @property
    def _series(self):
        _view = self.view(MaskedArray)
        _view._sharedmask = False
        return _view

msubarray = MSubArray


# Also a subclass that overrides __str__, __repr__ and __setitem__, disallowing
# setting to non-class values (and thus np.ma.core.masked_print_option)
# and overrides __array_wrap__, updating the info dict, to check that this
# doesn't get destroyed by MaskedArray._update_from.  But this one also needs
# its own iterator...
class CSAIterator:
    """
    Flat iterator object that uses its own setter/getter
    (works around ndarray.flat not propagating subclass setters/getters
    see https://github.com/numpy/numpy/issues/4564)
    roughly following MaskedIterator
    """
    def __init__(self, a):
        self._original = a
        self._dataiter = a.view(np.ndarray).flat

    def __iter__(self):
        return self

    def __getitem__(self, indx):
        out = self._dataiter.__getitem__(indx)
        if not isinstance(out, np.ndarray):
            out = out.__array__()
        out = out.view(type(self._original))
        return out

    def __setitem__(self, index, value):
        self._dataiter[index] = self._original._validate_input(value)

    def __next__(self):
        return next(self._dataiter).__array__().view(type(self._original))


class ComplicatedSubArray(SubArray):

    def __str__(self):
        return f'myprefix {self.view(SubArray)} mypostfix'

    def __repr__(self):
        # Return a repr that does not start with 'name('
        return f'<{self.__class__.__name__} {self}>'

    def _validate_input(self, value):
        if not isinstance(value, ComplicatedSubArray):
            raise ValueError("Can only set to MySubArray values")
        return value

    def __setitem__(self, item, value):
        # validation ensures direct assignment with ndarray or
        # masked_print_option will fail
        super().__setitem__(item, self._validate_input(value))

    def __getitem__(self, item):
        # ensure getter returns our own class also for scalars
        value = super().__getitem__(item)
        if not isinstance(value, np.ndarray):  # scalar
            value = value.__array__().view(ComplicatedSubArray)
        return value

    @property
    def flat(self):
        return CSAIterator(self)

    @flat.setter
    def flat(self, value):
        y = self.ravel()
        y[:] = value

    def __array_wrap__(self, obj, context=None):
        obj = super().__array_wrap__(obj, context)
        if context is not None and context[0] is np.multiply:
            obj.info['multiplied'] = obj.info.get('multiplied', 0) + 1

        return obj


class WrappedArray(NDArrayOperatorsMixin):
    """
    Wrapping a MaskedArray rather than subclassing to test that
    ufunc deferrals are commutative.
    See: https://github.com/numpy/numpy/issues/15200)
    """
    __array_priority__ = 20

    def __init__(self, array, **attrs):
        self._array = array
        self.attrs = attrs

    def __repr__(self):
        return f"{self.__class__.__name__}(\n{self._array}\n{self.attrs}\n)"

    def __array__(self):
        return np.asarray(self._array)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        if method == '__call__':
            inputs = [arg._array if isinstance(arg, self.__class__) else arg
                      for arg in inputs]
            return self.__class__(ufunc(*inputs, **kwargs), **self.attrs)
        else:
            return NotImplemented


class TestSubclassing:
    # Test suite for masked subclasses of ndarray.

    def setup_method(self):
        x = np.arange(5, dtype='float')
        mx = msubarray(x, mask=[0, 1, 0, 0, 0])
        self.data = (x, mx)

    def test_data_subclassing(self):
        # Tests whether the subclass is kept.
        x = np.arange(5)
        m = [0, 0, 1, 0, 0]
        xsub = SubArray(x)
        xmsub = masked_array(xsub, mask=m)
        assert_(isinstance(xmsub, MaskedArray))
        assert_equal(xmsub._data, xsub)
        assert_(isinstance(xmsub._data, SubArray))

    def test_maskedarray_subclassing(self):
        # Tests subclassing MaskedArray
        (x, mx) = self.data
        assert_(isinstance(mx._data, subarray))

    def test_masked_unary_operations(self):
        # Tests masked_unary_operation
        (x, mx) = self.data
        with np.errstate(divide='ignore'):
            assert_(isinstance(log(mx), msubarray))
            assert_equal(log(x), np.log(x))

    def test_masked_binary_operations(self):
        # Tests masked_binary_operation
        (x, mx) = self.data
        # Result should be a msubarray
        assert_(isinstance(add(mx, mx), msubarray))
        assert_(isinstance(add(mx, x), msubarray))
        # Result should work
        assert_equal(add(mx, x), mx+x)
        assert_(isinstance(add(mx, mx)._data, subarray))
        assert_(isinstance(add.outer(mx, mx), msubarray))
        assert_(isinstance(hypot(mx, mx), msubarray))
        assert_(isinstance(hypot(mx, x), msubarray))

    def test_masked_binary_operations2(self):
        # Tests domained_masked_binary_operation
        (x, mx) = self.data
        xmx = masked_array(mx.data.__array__(), mask=mx.mask)
        assert_(isinstance(divide(mx, mx), msubarray))
        assert_(isinstance(divide(mx, x), msubarray))
        assert_equal(divide(mx, mx), divide(xmx, xmx))

    def test_attributepropagation(self):
        x = array(arange(5), mask=[0]+[1]*4)
        my = masked_array(subarray(x))
        ym = msubarray(x)
        #
        z = (my+1)
        assert_(isinstance(z, MaskedArray))
        assert_(not isinstance(z, MSubArray))
        assert_(isinstance(z._data, SubArray))
        assert_equal(z._data.info, {})
        #
        z = (ym+1)
        assert_(isinstance(z, MaskedArray))
        assert_(isinstance(z, MSubArray))
        assert_(isinstance(z._data, SubArray))
        assert_(z._data.info['added'] > 0)
        # Test that inplace methods from data get used (gh-4617)
        ym += 1
        assert_(isinstance(ym, MaskedArray))
        assert_(isinstance(ym, MSubArray))
        assert_(isinstance(ym._data, SubArray))
        assert_(ym._data.info['iadded'] > 0)
        #
        ym._set_mask([1, 0, 0, 0, 1])
        assert_equal(ym._mask, [1, 0, 0, 0, 1])
        ym._series._set_mask([0, 0, 0, 0, 1])
        assert_equal(ym._mask, [0, 0, 0, 0, 1])
        #
        xsub = subarray(x, info={'name':'x'})
        mxsub = masked_array(xsub)
        assert_(hasattr(mxsub, 'info'))
        assert_equal(mxsub.info, xsub.info)

    def test_subclasspreservation(self):
        # Checks that masked_array(...,subok=True) preserves the class.
        x = np.arange(5)
        m = [0, 0, 1, 0, 0]
        xinfo = [(i, j) for (i, j) in zip(x, m)]
        xsub = MSubArray(x, mask=m, info={'xsub':xinfo})
        #
        mxsub = masked_array(xsub, subok=False)
        assert_(not isinstance(mxsub, MSubArray))
        assert_(isinstance(mxsub, MaskedArray))
        assert_equal(mxsub._mask, m)
        #
        mxsub = asarray(xsub)
        assert_(not isinstance(mxsub, MSubArray))
        assert_(isinstance(mxsub, MaskedArray))
        assert_equal(mxsub._mask, m)
        #
        mxsub = masked_array(xsub, subok=True)
        assert_(isinstance(mxsub, MSubArray))
        assert_equal(mxsub.info, xsub.info)
        assert_equal(mxsub._mask, xsub._mask)
        #
        mxsub = asanyarray(xsub)
        assert_(isinstance(mxsub, MSubArray))
        assert_equal(mxsub.info, xsub.info)
        assert_equal(mxsub._mask, m)

    def test_subclass_items(self):
        """test that getter and setter go via baseclass"""
        x = np.arange(5)
        xcsub = ComplicatedSubArray(x)
        mxcsub = masked_array(xcsub, mask=[True, False, True, False, False])
        # getter should  return a ComplicatedSubArray, even for single item
        # first check we wrote ComplicatedSubArray correctly
        assert_(isinstance(xcsub[1], ComplicatedSubArray))
        assert_(isinstance(xcsub[1,...], ComplicatedSubArray))
        assert_(isinstance(xcsub[1:4], ComplicatedSubArray))

        # now that it propagates inside the MaskedArray
        assert_(isinstance(mxcsub[1], ComplicatedSubArray))
        assert_(isinstance(mxcsub[1,...].data, ComplicatedSubArray))
        assert_(mxcsub[0] is masked)
        assert_(isinstance(mxcsub[0,...].data, ComplicatedSubArray))
        assert_(isinstance(mxcsub[1:4].data, ComplicatedSubArray))

        # also for flattened version (which goes via MaskedIterator)
        assert_(isinstance(mxcsub.flat[1].data, ComplicatedSubArray))
        assert_(mxcsub.flat[0] is masked)
        assert_(isinstance(mxcsub.flat[1:4].base, ComplicatedSubArray))

        # setter should only work with ComplicatedSubArray input
        # first check we wrote ComplicatedSubArray correctly
        assert_raises(ValueError, xcsub.__setitem__, 1, x[4])
        # now that it propagates inside the MaskedArray
        assert_raises(ValueError, mxcsub.__setitem__, 1, x[4])
        assert_raises(ValueError, mxcsub.__setitem__, slice(1, 4), x[1:4])
        mxcsub[1] = xcsub[4]
        mxcsub[1:4] = xcsub[1:4]
        # also for flattened version (which goes via MaskedIterator)
        assert_raises(ValueError, mxcsub.flat.__setitem__, 1, x[4])
        assert_raises(ValueError, mxcsub.flat.__setitem__, slice(1, 4), x[1:4])
        mxcsub.flat[1] = xcsub[4]
        mxcsub.flat[1:4] = xcsub[1:4]

    def test_subclass_nomask_items(self):
        x = np.arange(5)
        xcsub = ComplicatedSubArray(x)
        mxcsub_nomask = masked_array(xcsub)

        assert_(isinstance(mxcsub_nomask[1,...].data, ComplicatedSubArray))
        assert_(isinstance(mxcsub_nomask[0,...].data, ComplicatedSubArray))

        assert_(isinstance(mxcsub_nomask[1], ComplicatedSubArray))
        assert_(isinstance(mxcsub_nomask[0], ComplicatedSubArray))

    def test_subclass_repr(self):
        """test that repr uses the name of the subclass
        and 'array' for np.ndarray"""
        x = np.arange(5)
        mx = masked_array(x, mask=[True, False, True, False, False])
        assert_startswith(repr(mx), 'masked_array')
        xsub = SubArray(x)
        mxsub = masked_array(xsub, mask=[True, False, True, False, False])
        assert_startswith(repr(mxsub),
            f'masked_{SubArray.__name__}(data=[--, 1, --, 3, 4]')

    def test_subclass_str(self):
        """test str with subclass that has overridden str, setitem"""
        # first without override
        x = np.arange(5)
        xsub = SubArray(x)
        mxsub = masked_array(xsub, mask=[True, False, True, False, False])
        assert_equal(str(mxsub), '[-- 1 -- 3 4]')

        xcsub = ComplicatedSubArray(x)
        assert_raises(ValueError, xcsub.__setitem__, 0,
                      np.ma.core.masked_print_option)
        mxcsub = masked_array(xcsub, mask=[True, False, True, False, False])
        assert_equal(str(mxcsub), 'myprefix [-- 1 -- 3 4] mypostfix')

    def test_pure_subclass_info_preservation(self):
        # Test that ufuncs and methods conserve extra information consistently;
        # see gh-7122.
        arr1 = SubMaskedArray('test', data=[1,2,3,4,5,6])
        arr2 = SubMaskedArray(data=[0,1,2,3,4,5])
        diff1 = np.subtract(arr1, arr2)
        assert_('info' in diff1._optinfo)
        assert_(diff1._optinfo['info'] == 'test')
        diff2 = arr1 - arr2
        assert_('info' in diff2._optinfo)
        assert_(diff2._optinfo['info'] == 'test')


class ArrayNoInheritance:
    """Quantity-like class that does not inherit from ndarray"""
    def __init__(self, data, units):
        self.magnitude = data
        self.units = units

    def __getattr__(self, attr):
        return getattr(self.magnitude, attr)


def test_array_no_inheritance():
    data_masked = np.ma.array([1, 2, 3], mask=[True, False, True])
    data_masked_units = ArrayNoInheritance(data_masked, 'meters')

    # Get the masked representation of the Quantity-like class
    new_array = np.ma.array(data_masked_units)
    assert_equal(data_masked.data, new_array.data)
    assert_equal(data_masked.mask, new_array.mask)
    # Test sharing the mask
    data_masked.mask = [True, False, False]
    assert_equal(data_masked.mask, new_array.mask)
    assert_(new_array.sharedmask)

    # Get the masked representation of the Quantity-like class
    new_array = np.ma.array(data_masked_units, copy=True)
    assert_equal(data_masked.data, new_array.data)
    assert_equal(data_masked.mask, new_array.mask)
    # Test that the mask is not shared when copy=True
    data_masked.mask = [True, False, True]
    assert_equal([True, False, False], new_array.mask)
    assert_(not new_array.sharedmask)

    # Get the masked representation of the Quantity-like class
    new_array = np.ma.array(data_masked_units, keep_mask=False)
    assert_equal(data_masked.data, new_array.data)
    # The change did not affect the original mask
    assert_equal(data_masked.mask, [True, False, True])
    # Test that the mask is False and not shared when keep_mask=False
    assert_(not new_array.mask)
    assert_(not new_array.sharedmask)


class TestClassWrapping:
    # Test suite for classes that wrap MaskedArrays

    def setup_method(self):
        m = np.ma.masked_array([1, 3, 5], mask=[False, True, False])
        wm = WrappedArray(m)
        self.data = (m, wm)

    def test_masked_unary_operations(self):
        # Tests masked_unary_operation
        (m, wm) = self.data
        with np.errstate(divide='ignore'):
            assert_(isinstance(np.log(wm), WrappedArray))

    def test_masked_binary_operations(self):
        # Tests masked_binary_operation
        (m, wm) = self.data
        # Result should be a WrappedArray
        assert_(isinstance(np.add(wm, wm), WrappedArray))
        assert_(isinstance(np.add(m, wm), WrappedArray))
        assert_(isinstance(np.add(wm, m), WrappedArray))
        # add and '+' should call the same ufunc
        assert_equal(np.add(m, wm), m + wm)
        assert_(isinstance(np.hypot(m, wm), WrappedArray))
        assert_(isinstance(np.hypot(wm, m), WrappedArray))
        # Test domained binary operations
        assert_(isinstance(np.divide(wm, m), WrappedArray))
        assert_(isinstance(np.divide(m, wm), WrappedArray))
        assert_equal(np.divide(wm, m) * m, np.divide(m, m) * wm)
        # Test broadcasting
        m2 = np.stack([m, m])
        assert_(isinstance(np.divide(wm, m2), WrappedArray))
        assert_(isinstance(np.divide(m2, wm), WrappedArray))
        assert_equal(np.divide(m2, wm), np.divide(wm, m2))
