import numpy as np
from numpy.testing import assert_warns
from numpy.ma.testutils import (assert_, assert_equal, assert_raises,
                                assert_array_equal)
from numpy.ma.core import (masked_array, masked_values, masked, allequal,
                           MaskType, getmask, MaskedArray, nomask,
                           log, add, hypot, divide)
from numpy.ma.extras import mr_
from numpy.compat import pickle


class MMatrix(MaskedArray, np.matrix,):

    def __new__(cls, data, mask=nomask):
        mat = np.matrix(data)
        _data = MaskedArray.__new__(cls, data=mat, mask=mask)
        return _data

    def __array_finalize__(self, obj):
        np.matrix.__array_finalize__(self, obj)
        MaskedArray.__array_finalize__(self, obj)
        return

    @property
    def _series(self):
        _view = self.view(MaskedArray)
        _view._sharedmask = False
        return _view


class TestMaskedMatrix:
    def test_matrix_indexing(self):
        # Tests conversions and indexing
        x1 = np.matrix([[1, 2, 3], [4, 3, 2]])
        x2 = masked_array(x1, mask=[[1, 0, 0], [0, 1, 0]])
        x3 = masked_array(x1, mask=[[0, 1, 0], [1, 0, 0]])
        x4 = masked_array(x1)
        # test conversion to strings
        str(x2)  # raises?
        repr(x2)  # raises?
        # tests of indexing
        assert_(type(x2[1, 0]) is type(x1[1, 0]))
        assert_(x1[1, 0] == x2[1, 0])
        assert_(x2[1, 1] is masked)
        assert_equal(x1[0, 2], x2[0, 2])
        assert_equal(x1[0, 1:], x2[0, 1:])
        assert_equal(x1[:, 2], x2[:, 2])
        assert_equal(x1[:], x2[:])
        assert_equal(x1[1:], x3[1:])
        x1[0, 2] = 9
        x2[0, 2] = 9
        assert_equal(x1, x2)
        x1[0, 1:] = 99
        x2[0, 1:] = 99
        assert_equal(x1, x2)
        x2[0, 1] = masked
        assert_equal(x1, x2)
        x2[0, 1:] = masked
        assert_equal(x1, x2)
        x2[0, :] = x1[0, :]
        x2[0, 1] = masked
        assert_(allequal(getmask(x2), np.array([[0, 1, 0], [0, 1, 0]])))
        x3[1, :] = masked_array([1, 2, 3], [1, 1, 0])
        assert_(allequal(getmask(x3)[1], masked_array([1, 1, 0])))
        assert_(allequal(getmask(x3[1]), masked_array([1, 1, 0])))
        x4[1, :] = masked_array([1, 2, 3], [1, 1, 0])
        assert_(allequal(getmask(x4[1]), masked_array([1, 1, 0])))
        assert_(allequal(x4[1], masked_array([1, 2, 3])))
        x1 = np.matrix(np.arange(5) * 1.0)
        x2 = masked_values(x1, 3.0)
        assert_equal(x1, x2)
        assert_(allequal(masked_array([0, 0, 0, 1, 0], dtype=MaskType),
                         x2.mask))
        assert_equal(3.0, x2.fill_value)

    def test_pickling_subbaseclass(self):
        # Test pickling w/ a subclass of ndarray
        a = masked_array(np.matrix(list(range(10))), mask=[1, 0, 1, 0, 0] * 2)
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            a_pickled = pickle.loads(pickle.dumps(a, protocol=proto))
            assert_equal(a_pickled._mask, a._mask)
            assert_equal(a_pickled, a)
            assert_(isinstance(a_pickled._data, np.matrix))

    def test_count_mean_with_matrix(self):
        m = masked_array(np.matrix([[1, 2], [3, 4]]), mask=np.zeros((2, 2)))

        assert_equal(m.count(axis=0).shape, (1, 2))
        assert_equal(m.count(axis=1).shape, (2, 1))

        # Make sure broadcasting inside mean and var work
        assert_equal(m.mean(axis=0), [[2., 3.]])
        assert_equal(m.mean(axis=1), [[1.5], [3.5]])

    def test_flat(self):
        # Test that flat can return items even for matrices [#4585, #4615]
        # test simple access
        test = masked_array(np.matrix([[1, 2, 3]]), mask=[0, 0, 1])
        assert_equal(test.flat[1], 2)
        assert_equal(test.flat[2], masked)
        assert_(np.all(test.flat[0:2] == test[0, 0:2]))
        # Test flat on masked_matrices
        test = masked_array(np.matrix([[1, 2, 3]]), mask=[0, 0, 1])
        test.flat = masked_array([3, 2, 1], mask=[1, 0, 0])
        control = masked_array(np.matrix([[3, 2, 1]]), mask=[1, 0, 0])
        assert_equal(test, control)
        # Test setting
        test = masked_array(np.matrix([[1, 2, 3]]), mask=[0, 0, 1])
        testflat = test.flat
        testflat[:] = testflat[[2, 1, 0]]
        assert_equal(test, control)
        testflat[0] = 9
        # test that matrices keep the correct shape (#4615)
        a = masked_array(np.matrix(np.eye(2)), mask=0)
        b = a.flat
        b01 = b[:2]
        assert_equal(b01.data, np.array([[1., 0.]]))
        assert_equal(b01.mask, np.array([[False, False]]))

    def test_allany_onmatrices(self):
        x = np.array([[0.13, 0.26, 0.90],
                      [0.28, 0.33, 0.63],
                      [0.31, 0.87, 0.70]])
        X = np.matrix(x)
        m = np.array([[True, False, False],
                      [False, False, False],
                      [True, True, False]], dtype=np.bool_)
        mX = masked_array(X, mask=m)
        mXbig = (mX > 0.5)
        mXsmall = (mX < 0.5)

        assert_(not mXbig.all())
        assert_(mXbig.any())
        assert_equal(mXbig.all(0), np.matrix([False, False, True]))
        assert_equal(mXbig.all(1), np.matrix([False, False, True]).T)
        assert_equal(mXbig.any(0), np.matrix([False, False, True]))
        assert_equal(mXbig.any(1), np.matrix([True, True, True]).T)

        assert_(not mXsmall.all())
        assert_(mXsmall.any())
        assert_equal(mXsmall.all(0), np.matrix([True, True, False]))
        assert_equal(mXsmall.all(1), np.matrix([False, False, False]).T)
        assert_equal(mXsmall.any(0), np.matrix([True, True, False]))
        assert_equal(mXsmall.any(1), np.matrix([True, True, False]).T)

    def test_compressed(self):
        a = masked_array(np.matrix([1, 2, 3, 4]), mask=[0, 0, 0, 0])
        b = a.compressed()
        assert_equal(b, a)
        assert_(isinstance(b, np.matrix))
        a[0, 0] = masked
        b = a.compressed()
        assert_equal(b, [[2, 3, 4]])

    def test_ravel(self):
        a = masked_array(np.matrix([1, 2, 3, 4, 5]), mask=[[0, 1, 0, 0, 0]])
        aravel = a.ravel()
        assert_equal(aravel.shape, (1, 5))
        assert_equal(aravel._mask.shape, a.shape)

    def test_view(self):
        # Test view w/ flexible dtype
        iterator = list(zip(np.arange(10), np.random.rand(10)))
        data = np.array(iterator)
        a = masked_array(iterator, dtype=[('a', float), ('b', float)])
        a.mask[0] = (1, 0)
        test = a.view((float, 2), np.matrix)
        assert_equal(test, data)
        assert_(isinstance(test, np.matrix))
        assert_(not isinstance(test, MaskedArray))


class TestSubclassing:
    # Test suite for masked subclasses of ndarray.

    def setup_method(self):
        x = np.arange(5, dtype='float')
        mx = MMatrix(x, mask=[0, 1, 0, 0, 0])
        self.data = (x, mx)

    def test_maskedarray_subclassing(self):
        # Tests subclassing MaskedArray
        (x, mx) = self.data
        assert_(isinstance(mx._data, np.matrix))

    def test_masked_unary_operations(self):
        # Tests masked_unary_operation
        (x, mx) = self.data
        with np.errstate(divide='ignore'):
            assert_(isinstance(log(mx), MMatrix))
            assert_equal(log(x), np.log(x))

    def test_masked_binary_operations(self):
        # Tests masked_binary_operation
        (x, mx) = self.data
        # Result should be a MMatrix
        assert_(isinstance(add(mx, mx), MMatrix))
        assert_(isinstance(add(mx, x), MMatrix))
        # Result should work
        assert_equal(add(mx, x), mx+x)
        assert_(isinstance(add(mx, mx)._data, np.matrix))
        with assert_warns(DeprecationWarning):
            assert_(isinstance(add.outer(mx, mx), MMatrix))
        assert_(isinstance(hypot(mx, mx), MMatrix))
        assert_(isinstance(hypot(mx, x), MMatrix))

    def test_masked_binary_operations2(self):
        # Tests domained_masked_binary_operation
        (x, mx) = self.data
        xmx = masked_array(mx.data.__array__(), mask=mx.mask)
        assert_(isinstance(divide(mx, mx), MMatrix))
        assert_(isinstance(divide(mx, x), MMatrix))
        assert_equal(divide(mx, mx), divide(xmx, xmx))

class TestConcatenator:
    # Tests for mr_, the equivalent of r_ for masked arrays.

    def test_matrix_builder(self):
        assert_raises(np.ma.MAError, lambda: mr_['1, 2; 3, 4'])

    def test_matrix(self):
        # Test consistency with unmasked version.  If we ever deprecate
        # matrix, this test should either still pass, or both actual and
        # expected should fail to be build.
        actual = mr_['r', 1, 2, 3]
        expected = np.ma.array(np.r_['r', 1, 2, 3])
        assert_array_equal(actual, expected)

        # outer type is masked array, inner type is matrix
        assert_equal(type(actual), type(expected))
        assert_equal(type(actual.data), type(expected.data))
