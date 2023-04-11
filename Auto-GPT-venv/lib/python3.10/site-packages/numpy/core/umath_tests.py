"""
Shim for _umath_tests to allow a deprecation period for the new name.

"""
import warnings

# 2018-04-04, numpy 1.15.0
warnings.warn(("numpy.core.umath_tests is an internal NumPy "
               "module and should not be imported. It will "
               "be removed in a future NumPy release."),
              category=DeprecationWarning, stacklevel=2)

from ._umath_tests import *
