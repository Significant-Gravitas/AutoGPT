"""
Create the numpy.core.umath namespace for backward compatibility. In v1.16
the multiarray and umath c-extension modules were merged into a single
_multiarray_umath extension module. So we replicate the old namespace
by importing from the extension module.

"""

from . import _multiarray_umath
from ._multiarray_umath import *  # noqa: F403
# These imports are needed for backward compatibility,
# do not change them. issue gh-11862
# _ones_like is semi-public, on purpose not added to __all__
from ._multiarray_umath import _UFUNC_API, _add_newdoc_ufunc, _ones_like

__all__ = [
    '_UFUNC_API', 'ERR_CALL', 'ERR_DEFAULT', 'ERR_IGNORE', 'ERR_LOG',
    'ERR_PRINT', 'ERR_RAISE', 'ERR_WARN', 'FLOATING_POINT_SUPPORT',
    'FPE_DIVIDEBYZERO', 'FPE_INVALID', 'FPE_OVERFLOW', 'FPE_UNDERFLOW', 'NAN',
    'NINF', 'NZERO', 'PINF', 'PZERO', 'SHIFT_DIVIDEBYZERO', 'SHIFT_INVALID',
    'SHIFT_OVERFLOW', 'SHIFT_UNDERFLOW', 'UFUNC_BUFSIZE_DEFAULT',
    'UFUNC_PYVALS_NAME', '_add_newdoc_ufunc', 'absolute', 'add',
    'arccos', 'arccosh', 'arcsin', 'arcsinh', 'arctan', 'arctan2', 'arctanh',
    'bitwise_and', 'bitwise_or', 'bitwise_xor', 'cbrt', 'ceil', 'conj',
    'conjugate', 'copysign', 'cos', 'cosh', 'deg2rad', 'degrees', 'divide',
    'divmod', 'e', 'equal', 'euler_gamma', 'exp', 'exp2', 'expm1', 'fabs',
    'floor', 'floor_divide', 'float_power', 'fmax', 'fmin', 'fmod', 'frexp',
    'frompyfunc', 'gcd', 'geterrobj', 'greater', 'greater_equal', 'heaviside',
    'hypot', 'invert', 'isfinite', 'isinf', 'isnan', 'isnat', 'lcm', 'ldexp',
    'left_shift', 'less', 'less_equal', 'log', 'log10', 'log1p', 'log2',
    'logaddexp', 'logaddexp2', 'logical_and', 'logical_not', 'logical_or',
    'logical_xor', 'maximum', 'minimum', 'mod', 'modf', 'multiply', 'negative',
    'nextafter', 'not_equal', 'pi', 'positive', 'power', 'rad2deg', 'radians',
    'reciprocal', 'remainder', 'right_shift', 'rint', 'seterrobj', 'sign',
    'signbit', 'sin', 'sinh', 'spacing', 'sqrt', 'square', 'subtract', 'tan',
    'tanh', 'true_divide', 'trunc']
