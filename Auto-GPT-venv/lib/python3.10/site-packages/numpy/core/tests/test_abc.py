from numpy.testing import assert_

import numbers

import numpy as np
from numpy.core.numerictypes import sctypes

class TestABC:
    def test_abstract(self):
        assert_(issubclass(np.number, numbers.Number))

        assert_(issubclass(np.inexact, numbers.Complex))
        assert_(issubclass(np.complexfloating, numbers.Complex))
        assert_(issubclass(np.floating, numbers.Real))

        assert_(issubclass(np.integer, numbers.Integral))
        assert_(issubclass(np.signedinteger, numbers.Integral))
        assert_(issubclass(np.unsignedinteger, numbers.Integral))

    def test_floats(self):
        for t in sctypes['float']:
            assert_(isinstance(t(), numbers.Real),
                    f"{t.__name__} is not instance of Real")
            assert_(issubclass(t, numbers.Real),
                    f"{t.__name__} is not subclass of Real")
            assert_(not isinstance(t(), numbers.Rational),
                    f"{t.__name__} is instance of Rational")
            assert_(not issubclass(t, numbers.Rational),
                    f"{t.__name__} is subclass of Rational")

    def test_complex(self):
        for t in sctypes['complex']:
            assert_(isinstance(t(), numbers.Complex),
                    f"{t.__name__} is not instance of Complex")
            assert_(issubclass(t, numbers.Complex),
                    f"{t.__name__} is not subclass of Complex")
            assert_(not isinstance(t(), numbers.Real),
                    f"{t.__name__} is instance of Real")
            assert_(not issubclass(t, numbers.Real),
                    f"{t.__name__} is subclass of Real")

    def test_int(self):
        for t in sctypes['int']:
            assert_(isinstance(t(), numbers.Integral),
                    f"{t.__name__} is not instance of Integral")
            assert_(issubclass(t, numbers.Integral),
                    f"{t.__name__} is not subclass of Integral")

    def test_uint(self):
        for t in sctypes['uint']:
            assert_(isinstance(t(), numbers.Integral),
                    f"{t.__name__} is not instance of Integral")
            assert_(issubclass(t, numbers.Integral),
                    f"{t.__name__} is not subclass of Integral")
