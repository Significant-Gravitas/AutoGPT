""" Test printing of scalar types.

"""
import pytest

import numpy as np
from numpy.testing import assert_, assert_raises


class A:
    pass
class B(A, np.float64):
    pass

class C(B):
    pass
class D(C, B):
    pass

class B0(np.float64, A):
    pass
class C0(B0):
    pass

class HasNew:
    def __new__(cls, *args, **kwargs):
        return cls, args, kwargs

class B1(np.float64, HasNew):
    pass


class TestInherit:
    def test_init(self):
        x = B(1.0)
        assert_(str(x) == '1.0')
        y = C(2.0)
        assert_(str(y) == '2.0')
        z = D(3.0)
        assert_(str(z) == '3.0')

    def test_init2(self):
        x = B0(1.0)
        assert_(str(x) == '1.0')
        y = C0(2.0)
        assert_(str(y) == '2.0')

    def test_gh_15395(self):
        # HasNew is the second base, so `np.float64` should have priority
        x = B1(1.0)
        assert_(str(x) == '1.0')

        # previously caused RecursionError!?
        with pytest.raises(TypeError):
            B1(1.0, 2.0)


class TestCharacter:
    def test_char_radd(self):
        # GH issue 9620, reached gentype_add and raise TypeError
        np_s = np.string_('abc')
        np_u = np.unicode_('abc')
        s = b'def'
        u = 'def'
        assert_(np_s.__radd__(np_s) is NotImplemented)
        assert_(np_s.__radd__(np_u) is NotImplemented)
        assert_(np_s.__radd__(s) is NotImplemented)
        assert_(np_s.__radd__(u) is NotImplemented)
        assert_(np_u.__radd__(np_s) is NotImplemented)
        assert_(np_u.__radd__(np_u) is NotImplemented)
        assert_(np_u.__radd__(s) is NotImplemented)
        assert_(np_u.__radd__(u) is NotImplemented)
        assert_(s + np_s == b'defabc')
        assert_(u + np_u == 'defabc')

        class MyStr(str, np.generic):
            # would segfault
            pass

        with assert_raises(TypeError):
            # Previously worked, but gave completely wrong result
            ret = s + MyStr('abc')

        class MyBytes(bytes, np.generic):
            # would segfault
            pass

        ret = s + MyBytes(b'abc')
        assert(type(ret) is type(s))
        assert ret == b"defabc"

    def test_char_repeat(self):
        np_s = np.string_('abc')
        np_u = np.unicode_('abc')
        res_s = b'abc' * 5
        res_u = 'abc' * 5
        assert_(np_s * 5 == res_s)
        assert_(np_u * 5 == res_u)
