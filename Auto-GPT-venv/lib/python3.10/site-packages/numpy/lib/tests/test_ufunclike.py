import numpy as np
import numpy.core as nx
import numpy.lib.ufunclike as ufl
from numpy.testing import (
    assert_, assert_equal, assert_array_equal, assert_warns, assert_raises
)


class TestUfunclike:

    def test_isposinf(self):
        a = nx.array([nx.inf, -nx.inf, nx.nan, 0.0, 3.0, -3.0])
        out = nx.zeros(a.shape, bool)
        tgt = nx.array([True, False, False, False, False, False])

        res = ufl.isposinf(a)
        assert_equal(res, tgt)
        res = ufl.isposinf(a, out)
        assert_equal(res, tgt)
        assert_equal(out, tgt)

        a = a.astype(np.complex_)
        with assert_raises(TypeError):
            ufl.isposinf(a)

    def test_isneginf(self):
        a = nx.array([nx.inf, -nx.inf, nx.nan, 0.0, 3.0, -3.0])
        out = nx.zeros(a.shape, bool)
        tgt = nx.array([False, True, False, False, False, False])

        res = ufl.isneginf(a)
        assert_equal(res, tgt)
        res = ufl.isneginf(a, out)
        assert_equal(res, tgt)
        assert_equal(out, tgt)

        a = a.astype(np.complex_)
        with assert_raises(TypeError):
            ufl.isneginf(a)

    def test_fix(self):
        a = nx.array([[1.0, 1.1, 1.5, 1.8], [-1.0, -1.1, -1.5, -1.8]])
        out = nx.zeros(a.shape, float)
        tgt = nx.array([[1., 1., 1., 1.], [-1., -1., -1., -1.]])

        res = ufl.fix(a)
        assert_equal(res, tgt)
        res = ufl.fix(a, out)
        assert_equal(res, tgt)
        assert_equal(out, tgt)
        assert_equal(ufl.fix(3.14), 3)

    def test_fix_with_subclass(self):
        class MyArray(nx.ndarray):
            def __new__(cls, data, metadata=None):
                res = nx.array(data, copy=True).view(cls)
                res.metadata = metadata
                return res

            def __array_wrap__(self, obj, context=None):
                if isinstance(obj, MyArray):
                    obj.metadata = self.metadata
                return obj

            def __array_finalize__(self, obj):
                self.metadata = getattr(obj, 'metadata', None)
                return self

        a = nx.array([1.1, -1.1])
        m = MyArray(a, metadata='foo')
        f = ufl.fix(m)
        assert_array_equal(f, nx.array([1, -1]))
        assert_(isinstance(f, MyArray))
        assert_equal(f.metadata, 'foo')

        # check 0d arrays don't decay to scalars
        m0d = m[0,...]
        m0d.metadata = 'bar'
        f0d = ufl.fix(m0d)
        assert_(isinstance(f0d, MyArray))
        assert_equal(f0d.metadata, 'bar')

    def test_deprecated(self):
        # NumPy 1.13.0, 2017-04-26
        assert_warns(DeprecationWarning, ufl.fix, [1, 2], y=nx.empty(2))
        assert_warns(DeprecationWarning, ufl.isposinf, [1, 2], y=nx.empty(2))
        assert_warns(DeprecationWarning, ufl.isneginf, [1, 2], y=nx.empty(2))

    def test_scalar(self):
        x = np.inf
        actual = np.isposinf(x)
        expected = np.True_
        assert_equal(actual, expected)
        assert_equal(type(actual), type(expected))

        x = -3.4
        actual = np.fix(x)
        expected = np.float64(-3.0)
        assert_equal(actual, expected)
        assert_equal(type(actual), type(expected))

        out = np.array(0.0)
        actual = np.fix(x, out=out)
        assert_(actual is out)
