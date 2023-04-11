from inspect import getfullargspec

from numpy.testing import assert_raises

from .. import asarray, _elementwise_functions
from .._elementwise_functions import bitwise_left_shift, bitwise_right_shift
from .._dtypes import (
    _dtype_categories,
    _boolean_dtypes,
    _floating_dtypes,
    _integer_dtypes,
)


def nargs(func):
    return len(getfullargspec(func).args)


def test_function_types():
    # Test that every function accepts only the required input types. We only
    # test the negative cases here (error). The positive cases are tested in
    # the array API test suite.

    elementwise_function_input_types = {
        "abs": "numeric",
        "acos": "floating-point",
        "acosh": "floating-point",
        "add": "numeric",
        "asin": "floating-point",
        "asinh": "floating-point",
        "atan": "floating-point",
        "atan2": "floating-point",
        "atanh": "floating-point",
        "bitwise_and": "integer or boolean",
        "bitwise_invert": "integer or boolean",
        "bitwise_left_shift": "integer",
        "bitwise_or": "integer or boolean",
        "bitwise_right_shift": "integer",
        "bitwise_xor": "integer or boolean",
        "ceil": "numeric",
        "cos": "floating-point",
        "cosh": "floating-point",
        "divide": "floating-point",
        "equal": "all",
        "exp": "floating-point",
        "expm1": "floating-point",
        "floor": "numeric",
        "floor_divide": "numeric",
        "greater": "numeric",
        "greater_equal": "numeric",
        "isfinite": "numeric",
        "isinf": "numeric",
        "isnan": "numeric",
        "less": "numeric",
        "less_equal": "numeric",
        "log": "floating-point",
        "logaddexp": "floating-point",
        "log10": "floating-point",
        "log1p": "floating-point",
        "log2": "floating-point",
        "logical_and": "boolean",
        "logical_not": "boolean",
        "logical_or": "boolean",
        "logical_xor": "boolean",
        "multiply": "numeric",
        "negative": "numeric",
        "not_equal": "all",
        "positive": "numeric",
        "pow": "numeric",
        "remainder": "numeric",
        "round": "numeric",
        "sign": "numeric",
        "sin": "floating-point",
        "sinh": "floating-point",
        "sqrt": "floating-point",
        "square": "numeric",
        "subtract": "numeric",
        "tan": "floating-point",
        "tanh": "floating-point",
        "trunc": "numeric",
    }

    def _array_vals():
        for d in _integer_dtypes:
            yield asarray(1, dtype=d)
        for d in _boolean_dtypes:
            yield asarray(False, dtype=d)
        for d in _floating_dtypes:
            yield asarray(1.0, dtype=d)

    for x in _array_vals():
        for func_name, types in elementwise_function_input_types.items():
            dtypes = _dtype_categories[types]
            func = getattr(_elementwise_functions, func_name)
            if nargs(func) == 2:
                for y in _array_vals():
                    if x.dtype not in dtypes or y.dtype not in dtypes:
                        assert_raises(TypeError, lambda: func(x, y))
            else:
                if x.dtype not in dtypes:
                    assert_raises(TypeError, lambda: func(x))


def test_bitwise_shift_error():
    # bitwise shift functions should raise when the second argument is negative
    assert_raises(
        ValueError, lambda: bitwise_left_shift(asarray([1, 1]), asarray([1, -1]))
    )
    assert_raises(
        ValueError, lambda: bitwise_right_shift(asarray([1, 1]), asarray([1, -1]))
    )
