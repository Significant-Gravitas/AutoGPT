import collections.abc

import numpy as np
from numpy import matrix, asmatrix, bmat
from numpy.testing import (
    assert_, assert_equal, assert_almost_equal, assert_array_equal,
    assert_array_almost_equal, assert_raises
    )
from numpy.linalg import matrix_power
from numpy.matrixlib import mat

class TestCtor:
    def test_basic(self):
        A = np.array([[1, 2], [3, 4]])
        mA = matrix(A)
        assert_(np.all(mA.A == A))

        B = bmat("A,A;A,A")
        C = bmat([[A, A], [A, A]])
        D = np.array([[1, 2, 1, 2],
                      [3, 4, 3, 4],
                      [1, 2, 1, 2],
                      [3, 4, 3, 4]])
        assert_(np.all(B.A == D))
        assert_(np.all(C.A == D))

        E = np.array([[5, 6], [7, 8]])
        AEresult = matrix([[1, 2, 5, 6], [3, 4, 7, 8]])
        assert_(np.all(bmat([A, E]) == AEresult))

        vec = np.arange(5)
        mvec = matrix(vec)
        assert_(mvec.shape == (1, 5))

    def test_exceptions(self):
        # Check for ValueError when called with invalid string data.
        assert_raises(ValueError, matrix, "invalid")

    def test_bmat_nondefault_str(self):
        A = np.array([[1, 2], [3, 4]])
        B = np.array([[5, 6], [7, 8]])
        Aresult = np.array([[1, 2, 1, 2],
                            [3, 4, 3, 4],
                            [1, 2, 1, 2],
                            [3, 4, 3, 4]])
        mixresult = np.array([[1, 2, 5, 6],
                              [3, 4, 7, 8],
                              [5, 6, 1, 2],
                              [7, 8, 3, 4]])
        assert_(np.all(bmat("A,A;A,A") == Aresult))
        assert_(np.all(bmat("A,A;A,A", ldict={'A':B}) == Aresult))
        assert_raises(TypeError, bmat, "A,A;A,A", gdict={'A':B})
        assert_(
            np.all(bmat("A,A;A,A", ldict={'A':A}, gdict={'A':B}) == Aresult))
        b2 = bmat("A,B;C,D", ldict={'A':A,'B':B}, gdict={'C':B,'D':A})
        assert_(np.all(b2 == mixresult))


class TestProperties:
    def test_sum(self):
        """Test whether matrix.sum(axis=1) preserves orientation.
        Fails in NumPy <= 0.9.6.2127.
        """
        M = matrix([[1, 2, 0, 0],
                   [3, 4, 0, 0],
                   [1, 2, 1, 2],
                   [3, 4, 3, 4]])
        sum0 = matrix([8, 12, 4, 6])
        sum1 = matrix([3, 7, 6, 14]).T
        sumall = 30
        assert_array_equal(sum0, M.sum(axis=0))
        assert_array_equal(sum1, M.sum(axis=1))
        assert_equal(sumall, M.sum())

        assert_array_equal(sum0, np.sum(M, axis=0))
        assert_array_equal(sum1, np.sum(M, axis=1))
        assert_equal(sumall, np.sum(M))

    def test_prod(self):
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(x.prod(), 720)
        assert_equal(x.prod(0), matrix([[4, 10, 18]]))
        assert_equal(x.prod(1), matrix([[6], [120]]))

        assert_equal(np.prod(x), 720)
        assert_equal(np.prod(x, axis=0), matrix([[4, 10, 18]]))
        assert_equal(np.prod(x, axis=1), matrix([[6], [120]]))

        y = matrix([0, 1, 3])
        assert_(y.prod() == 0)

    def test_max(self):
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(x.max(), 6)
        assert_equal(x.max(0), matrix([[4, 5, 6]]))
        assert_equal(x.max(1), matrix([[3], [6]]))

        assert_equal(np.max(x), 6)
        assert_equal(np.max(x, axis=0), matrix([[4, 5, 6]]))
        assert_equal(np.max(x, axis=1), matrix([[3], [6]]))

    def test_min(self):
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(x.min(), 1)
        assert_equal(x.min(0), matrix([[1, 2, 3]]))
        assert_equal(x.min(1), matrix([[1], [4]]))

        assert_equal(np.min(x), 1)
        assert_equal(np.min(x, axis=0), matrix([[1, 2, 3]]))
        assert_equal(np.min(x, axis=1), matrix([[1], [4]]))

    def test_ptp(self):
        x = np.arange(4).reshape((2, 2))
        assert_(x.ptp() == 3)
        assert_(np.all(x.ptp(0) == np.array([2, 2])))
        assert_(np.all(x.ptp(1) == np.array([1, 1])))

    def test_var(self):
        x = np.arange(9).reshape((3, 3))
        mx = x.view(np.matrix)
        assert_equal(x.var(ddof=0), mx.var(ddof=0))
        assert_equal(x.var(ddof=1), mx.var(ddof=1))

    def test_basic(self):
        import numpy.linalg as linalg

        A = np.array([[1., 2.],
                      [3., 4.]])
        mA = matrix(A)
        assert_(np.allclose(linalg.inv(A), mA.I))
        assert_(np.all(np.array(np.transpose(A) == mA.T)))
        assert_(np.all(np.array(np.transpose(A) == mA.H)))
        assert_(np.all(A == mA.A))

        B = A + 2j*A
        mB = matrix(B)
        assert_(np.allclose(linalg.inv(B), mB.I))
        assert_(np.all(np.array(np.transpose(B) == mB.T)))
        assert_(np.all(np.array(np.transpose(B).conj() == mB.H)))

    def test_pinv(self):
        x = matrix(np.arange(6).reshape(2, 3))
        xpinv = matrix([[-0.77777778,  0.27777778],
                        [-0.11111111,  0.11111111],
                        [ 0.55555556, -0.05555556]])
        assert_almost_equal(x.I, xpinv)

    def test_comparisons(self):
        A = np.arange(100).reshape(10, 10)
        mA = matrix(A)
        mB = matrix(A) + 0.1
        assert_(np.all(mB == A+0.1))
        assert_(np.all(mB == matrix(A+0.1)))
        assert_(not np.any(mB == matrix(A-0.1)))
        assert_(np.all(mA < mB))
        assert_(np.all(mA <= mB))
        assert_(np.all(mA <= mA))
        assert_(not np.any(mA < mA))

        assert_(not np.any(mB < mA))
        assert_(np.all(mB >= mA))
        assert_(np.all(mB >= mB))
        assert_(not np.any(mB > mB))

        assert_(np.all(mA == mA))
        assert_(not np.any(mA == mB))
        assert_(np.all(mB != mA))

        assert_(not np.all(abs(mA) > 0))
        assert_(np.all(abs(mB > 0)))

    def test_asmatrix(self):
        A = np.arange(100).reshape(10, 10)
        mA = asmatrix(A)
        A[0, 0] = -10
        assert_(A[0, 0] == mA[0, 0])

    def test_noaxis(self):
        A = matrix([[1, 0], [0, 1]])
        assert_(A.sum() == matrix(2))
        assert_(A.mean() == matrix(0.5))

    def test_repr(self):
        A = matrix([[1, 0], [0, 1]])
        assert_(repr(A) == "matrix([[1, 0],\n        [0, 1]])")

    def test_make_bool_matrix_from_str(self):
        A = matrix('True; True; False')
        B = matrix([[True], [True], [False]])
        assert_array_equal(A, B)

class TestCasting:
    def test_basic(self):
        A = np.arange(100).reshape(10, 10)
        mA = matrix(A)

        mB = mA.copy()
        O = np.ones((10, 10), np.float64) * 0.1
        mB = mB + O
        assert_(mB.dtype.type == np.float64)
        assert_(np.all(mA != mB))
        assert_(np.all(mB == mA+0.1))

        mC = mA.copy()
        O = np.ones((10, 10), np.complex128)
        mC = mC * O
        assert_(mC.dtype.type == np.complex128)
        assert_(np.all(mA != mB))


class TestAlgebra:
    def test_basic(self):
        import numpy.linalg as linalg

        A = np.array([[1., 2.], [3., 4.]])
        mA = matrix(A)

        B = np.identity(2)
        for i in range(6):
            assert_(np.allclose((mA ** i).A, B))
            B = np.dot(B, A)

        Ainv = linalg.inv(A)
        B = np.identity(2)
        for i in range(6):
            assert_(np.allclose((mA ** -i).A, B))
            B = np.dot(B, Ainv)

        assert_(np.allclose((mA * mA).A, np.dot(A, A)))
        assert_(np.allclose((mA + mA).A, (A + A)))
        assert_(np.allclose((3*mA).A, (3*A)))

        mA2 = matrix(A)
        mA2 *= 3
        assert_(np.allclose(mA2.A, 3*A))

    def test_pow(self):
        """Test raising a matrix to an integer power works as expected."""
        m = matrix("1. 2.; 3. 4.")
        m2 = m.copy()
        m2 **= 2
        mi = m.copy()
        mi **= -1
        m4 = m2.copy()
        m4 **= 2
        assert_array_almost_equal(m2, m**2)
        assert_array_almost_equal(m4, np.dot(m2, m2))
        assert_array_almost_equal(np.dot(mi, m), np.eye(2))

    def test_scalar_type_pow(self):
        m = matrix([[1, 2], [3, 4]])
        for scalar_t in [np.int8, np.uint8]:
            two = scalar_t(2)
            assert_array_almost_equal(m ** 2, m ** two)

    def test_notimplemented(self):
        '''Check that 'not implemented' operations produce a failure.'''
        A = matrix([[1., 2.],
                    [3., 4.]])

        # __rpow__
        with assert_raises(TypeError):
            1.0**A

        # __mul__ with something not a list, ndarray, tuple, or scalar
        with assert_raises(TypeError):
            A*object()


class TestMatrixReturn:
    def test_instance_methods(self):
        a = matrix([1.0], dtype='f8')
        methodargs = {
            'astype': ('intc',),
            'clip': (0.0, 1.0),
            'compress': ([1],),
            'repeat': (1,),
            'reshape': (1,),
            'swapaxes': (0, 0),
            'dot': np.array([1.0]),
            }
        excluded_methods = [
            'argmin', 'choose', 'dump', 'dumps', 'fill', 'getfield',
            'getA', 'getA1', 'item', 'nonzero', 'put', 'putmask', 'resize',
            'searchsorted', 'setflags', 'setfield', 'sort',
            'partition', 'argpartition',
            'take', 'tofile', 'tolist', 'tostring', 'tobytes', 'all', 'any',
            'sum', 'argmax', 'argmin', 'min', 'max', 'mean', 'var', 'ptp',
            'prod', 'std', 'ctypes', 'itemset',
            ]
        for attrib in dir(a):
            if attrib.startswith('_') or attrib in excluded_methods:
                continue
            f = getattr(a, attrib)
            if isinstance(f, collections.abc.Callable):
                # reset contents of a
                a.astype('f8')
                a.fill(1.0)
                if attrib in methodargs:
                    args = methodargs[attrib]
                else:
                    args = ()
                b = f(*args)
                assert_(type(b) is matrix, "%s" % attrib)
        assert_(type(a.real) is matrix)
        assert_(type(a.imag) is matrix)
        c, d = matrix([0.0]).nonzero()
        assert_(type(c) is np.ndarray)
        assert_(type(d) is np.ndarray)


class TestIndexing:
    def test_basic(self):
        x = asmatrix(np.zeros((3, 2), float))
        y = np.zeros((3, 1), float)
        y[:, 0] = [0.8, 0.2, 0.3]
        x[:, 1] = y > 0.5
        assert_equal(x, [[0, 1], [0, 0], [0, 0]])


class TestNewScalarIndexing:
    a = matrix([[1, 2], [3, 4]])

    def test_dimesions(self):
        a = self.a
        x = a[0]
        assert_equal(x.ndim, 2)

    def test_array_from_matrix_list(self):
        a = self.a
        x = np.array([a, a])
        assert_equal(x.shape, [2, 2, 2])

    def test_array_to_list(self):
        a = self.a
        assert_equal(a.tolist(), [[1, 2], [3, 4]])

    def test_fancy_indexing(self):
        a = self.a
        x = a[1, [0, 1, 0]]
        assert_(isinstance(x, matrix))
        assert_equal(x, matrix([[3,  4,  3]]))
        x = a[[1, 0]]
        assert_(isinstance(x, matrix))
        assert_equal(x, matrix([[3,  4], [1, 2]]))
        x = a[[[1], [0]], [[1, 0], [0, 1]]]
        assert_(isinstance(x, matrix))
        assert_equal(x, matrix([[4,  3], [1,  2]]))

    def test_matrix_element(self):
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(x[0][0], matrix([[1, 2, 3]]))
        assert_equal(x[0][0].shape, (1, 3))
        assert_equal(x[0].shape, (1, 3))
        assert_equal(x[:, 0].shape, (2, 1))

        x = matrix(0)
        assert_equal(x[0, 0], 0)
        assert_equal(x[0], 0)
        assert_equal(x[:, 0].shape, x.shape)

    def test_scalar_indexing(self):
        x = asmatrix(np.zeros((3, 2), float))
        assert_equal(x[0, 0], x[0][0])

    def test_row_column_indexing(self):
        x = asmatrix(np.eye(2))
        assert_array_equal(x[0,:], [[1, 0]])
        assert_array_equal(x[1,:], [[0, 1]])
        assert_array_equal(x[:, 0], [[1], [0]])
        assert_array_equal(x[:, 1], [[0], [1]])

    def test_boolean_indexing(self):
        A = np.arange(6)
        A.shape = (3, 2)
        x = asmatrix(A)
        assert_array_equal(x[:, np.array([True, False])], x[:, 0])
        assert_array_equal(x[np.array([True, False, False]),:], x[0,:])

    def test_list_indexing(self):
        A = np.arange(6)
        A.shape = (3, 2)
        x = asmatrix(A)
        assert_array_equal(x[:, [1, 0]], x[:, ::-1])
        assert_array_equal(x[[2, 1, 0],:], x[::-1,:])


class TestPower:
    def test_returntype(self):
        a = np.array([[0, 1], [0, 0]])
        assert_(type(matrix_power(a, 2)) is np.ndarray)
        a = mat(a)
        assert_(type(matrix_power(a, 2)) is matrix)

    def test_list(self):
        assert_array_equal(matrix_power([[0, 1], [0, 0]], 2), [[0, 0], [0, 0]])


class TestShape:

    a = np.array([[1], [2]])
    m = matrix([[1], [2]])

    def test_shape(self):
        assert_equal(self.a.shape, (2, 1))
        assert_equal(self.m.shape, (2, 1))

    def test_numpy_ravel(self):
        assert_equal(np.ravel(self.a).shape, (2,))
        assert_equal(np.ravel(self.m).shape, (2,))

    def test_member_ravel(self):
        assert_equal(self.a.ravel().shape, (2,))
        assert_equal(self.m.ravel().shape, (1, 2))

    def test_member_flatten(self):
        assert_equal(self.a.flatten().shape, (2,))
        assert_equal(self.m.flatten().shape, (1, 2))

    def test_numpy_ravel_order(self):
        x = np.array([[1, 2, 3], [4, 5, 6]])
        assert_equal(np.ravel(x), [1, 2, 3, 4, 5, 6])
        assert_equal(np.ravel(x, order='F'), [1, 4, 2, 5, 3, 6])
        assert_equal(np.ravel(x.T), [1, 4, 2, 5, 3, 6])
        assert_equal(np.ravel(x.T, order='A'), [1, 2, 3, 4, 5, 6])
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(np.ravel(x), [1, 2, 3, 4, 5, 6])
        assert_equal(np.ravel(x, order='F'), [1, 4, 2, 5, 3, 6])
        assert_equal(np.ravel(x.T), [1, 4, 2, 5, 3, 6])
        assert_equal(np.ravel(x.T, order='A'), [1, 2, 3, 4, 5, 6])

    def test_matrix_ravel_order(self):
        x = matrix([[1, 2, 3], [4, 5, 6]])
        assert_equal(x.ravel(), [[1, 2, 3, 4, 5, 6]])
        assert_equal(x.ravel(order='F'), [[1, 4, 2, 5, 3, 6]])
        assert_equal(x.T.ravel(), [[1, 4, 2, 5, 3, 6]])
        assert_equal(x.T.ravel(order='A'), [[1, 2, 3, 4, 5, 6]])

    def test_array_memory_sharing(self):
        assert_(np.may_share_memory(self.a, self.a.ravel()))
        assert_(not np.may_share_memory(self.a, self.a.flatten()))

    def test_matrix_memory_sharing(self):
        assert_(np.may_share_memory(self.m, self.m.ravel()))
        assert_(not np.may_share_memory(self.m, self.m.flatten()))

    def test_expand_dims_matrix(self):
        # matrices are always 2d - so expand_dims only makes sense when the
        # type is changed away from matrix.
        a = np.arange(10).reshape((2, 5)).view(np.matrix)
        expanded = np.expand_dims(a, axis=1)
        assert_equal(expanded.ndim, 3)
        assert_(not isinstance(expanded, np.matrix))
