import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises


class TestRegression:
    def test_kron_matrix(self):
        # Ticket #71
        x = np.matrix('[1 0; 1 0]')
        assert_equal(type(np.kron(x, x)), type(x))

    def test_matrix_properties(self):
        # Ticket #125
        a = np.matrix([1.0], dtype=float)
        assert_(type(a.real) is np.matrix)
        assert_(type(a.imag) is np.matrix)
        c, d = np.matrix([0.0]).nonzero()
        assert_(type(c) is np.ndarray)
        assert_(type(d) is np.ndarray)

    def test_matrix_multiply_by_1d_vector(self):
        # Ticket #473
        def mul():
            np.mat(np.eye(2))*np.ones(2)

        assert_raises(ValueError, mul)

    def test_matrix_std_argmax(self):
        # Ticket #83
        x = np.asmatrix(np.random.uniform(0, 1, (3, 3)))
        assert_equal(x.std().shape, ())
        assert_equal(x.argmax().shape, ())
