from __future__ import annotations

from ._dtypes import (
    _boolean_dtypes,
    _floating_dtypes,
    _integer_dtypes,
    _integer_or_boolean_dtypes,
    _numeric_dtypes,
    _result_type,
)
from ._array_object import Array

import numpy as np


def abs(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.abs <numpy.abs>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in abs")
    return Array._new(np.abs(x._array))


# Note: the function name is different here
def acos(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.arccos <numpy.arccos>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in acos")
    return Array._new(np.arccos(x._array))


# Note: the function name is different here
def acosh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.arccosh <numpy.arccosh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in acosh")
    return Array._new(np.arccosh(x._array))


def add(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.add <numpy.add>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in add")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.add(x1._array, x2._array))


# Note: the function name is different here
def asin(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.arcsin <numpy.arcsin>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in asin")
    return Array._new(np.arcsin(x._array))


# Note: the function name is different here
def asinh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.arcsinh <numpy.arcsinh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in asinh")
    return Array._new(np.arcsinh(x._array))


# Note: the function name is different here
def atan(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.arctan <numpy.arctan>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atan")
    return Array._new(np.arctan(x._array))


# Note: the function name is different here
def atan2(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.arctan2 <numpy.arctan2>`.

    See its docstring for more information.
    """
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atan2")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.arctan2(x1._array, x2._array))


# Note: the function name is different here
def atanh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.arctanh <numpy.arctanh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in atanh")
    return Array._new(np.arctanh(x._array))


def bitwise_and(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.bitwise_and <numpy.bitwise_and>`.

    See its docstring for more information.
    """
    if (
        x1.dtype not in _integer_or_boolean_dtypes
        or x2.dtype not in _integer_or_boolean_dtypes
    ):
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_and")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.bitwise_and(x1._array, x2._array))


# Note: the function name is different here
def bitwise_left_shift(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.left_shift <numpy.left_shift>`.

    See its docstring for more information.
    """
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError("Only integer dtypes are allowed in bitwise_left_shift")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    # Note: bitwise_left_shift is only defined for x2 nonnegative.
    if np.any(x2._array < 0):
        raise ValueError("bitwise_left_shift(x1, x2) is only defined for x2 >= 0")
    return Array._new(np.left_shift(x1._array, x2._array))


# Note: the function name is different here
def bitwise_invert(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.invert <numpy.invert>`.

    See its docstring for more information.
    """
    if x.dtype not in _integer_or_boolean_dtypes:
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_invert")
    return Array._new(np.invert(x._array))


def bitwise_or(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.bitwise_or <numpy.bitwise_or>`.

    See its docstring for more information.
    """
    if (
        x1.dtype not in _integer_or_boolean_dtypes
        or x2.dtype not in _integer_or_boolean_dtypes
    ):
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_or")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.bitwise_or(x1._array, x2._array))


# Note: the function name is different here
def bitwise_right_shift(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.right_shift <numpy.right_shift>`.

    See its docstring for more information.
    """
    if x1.dtype not in _integer_dtypes or x2.dtype not in _integer_dtypes:
        raise TypeError("Only integer dtypes are allowed in bitwise_right_shift")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    # Note: bitwise_right_shift is only defined for x2 nonnegative.
    if np.any(x2._array < 0):
        raise ValueError("bitwise_right_shift(x1, x2) is only defined for x2 >= 0")
    return Array._new(np.right_shift(x1._array, x2._array))


def bitwise_xor(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.bitwise_xor <numpy.bitwise_xor>`.

    See its docstring for more information.
    """
    if (
        x1.dtype not in _integer_or_boolean_dtypes
        or x2.dtype not in _integer_or_boolean_dtypes
    ):
        raise TypeError("Only integer or boolean dtypes are allowed in bitwise_xor")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.bitwise_xor(x1._array, x2._array))


def ceil(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.ceil <numpy.ceil>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in ceil")
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of ceil is the same as the input
        return x
    return Array._new(np.ceil(x._array))


def cos(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.cos <numpy.cos>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in cos")
    return Array._new(np.cos(x._array))


def cosh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.cosh <numpy.cosh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in cosh")
    return Array._new(np.cosh(x._array))


def divide(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.divide <numpy.divide>`.

    See its docstring for more information.
    """
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in divide")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.divide(x1._array, x2._array))


def equal(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.equal <numpy.equal>`.

    See its docstring for more information.
    """
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.equal(x1._array, x2._array))


def exp(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.exp <numpy.exp>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in exp")
    return Array._new(np.exp(x._array))


def expm1(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.expm1 <numpy.expm1>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in expm1")
    return Array._new(np.expm1(x._array))


def floor(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.floor <numpy.floor>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in floor")
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of floor is the same as the input
        return x
    return Array._new(np.floor(x._array))


def floor_divide(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.floor_divide <numpy.floor_divide>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in floor_divide")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.floor_divide(x1._array, x2._array))


def greater(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.greater <numpy.greater>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in greater")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.greater(x1._array, x2._array))


def greater_equal(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.greater_equal <numpy.greater_equal>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in greater_equal")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.greater_equal(x1._array, x2._array))


def isfinite(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.isfinite <numpy.isfinite>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isfinite")
    return Array._new(np.isfinite(x._array))


def isinf(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.isinf <numpy.isinf>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isinf")
    return Array._new(np.isinf(x._array))


def isnan(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.isnan <numpy.isnan>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in isnan")
    return Array._new(np.isnan(x._array))


def less(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.less <numpy.less>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in less")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.less(x1._array, x2._array))


def less_equal(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.less_equal <numpy.less_equal>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in less_equal")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.less_equal(x1._array, x2._array))


def log(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.log <numpy.log>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log")
    return Array._new(np.log(x._array))


def log1p(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.log1p <numpy.log1p>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log1p")
    return Array._new(np.log1p(x._array))


def log2(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.log2 <numpy.log2>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log2")
    return Array._new(np.log2(x._array))


def log10(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.log10 <numpy.log10>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in log10")
    return Array._new(np.log10(x._array))


def logaddexp(x1: Array, x2: Array) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.logaddexp <numpy.logaddexp>`.

    See its docstring for more information.
    """
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in logaddexp")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.logaddexp(x1._array, x2._array))


def logical_and(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.logical_and <numpy.logical_and>`.

    See its docstring for more information.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_and")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.logical_and(x1._array, x2._array))


def logical_not(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.logical_not <numpy.logical_not>`.

    See its docstring for more information.
    """
    if x.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_not")
    return Array._new(np.logical_not(x._array))


def logical_or(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.logical_or <numpy.logical_or>`.

    See its docstring for more information.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_or")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.logical_or(x1._array, x2._array))


def logical_xor(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.logical_xor <numpy.logical_xor>`.

    See its docstring for more information.
    """
    if x1.dtype not in _boolean_dtypes or x2.dtype not in _boolean_dtypes:
        raise TypeError("Only boolean dtypes are allowed in logical_xor")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.logical_xor(x1._array, x2._array))


def multiply(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.multiply <numpy.multiply>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in multiply")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.multiply(x1._array, x2._array))


def negative(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.negative <numpy.negative>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in negative")
    return Array._new(np.negative(x._array))


def not_equal(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.not_equal <numpy.not_equal>`.

    See its docstring for more information.
    """
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.not_equal(x1._array, x2._array))


def positive(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.positive <numpy.positive>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in positive")
    return Array._new(np.positive(x._array))


# Note: the function name is different here
def pow(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.power <numpy.power>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in pow")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.power(x1._array, x2._array))


def remainder(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.remainder <numpy.remainder>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in remainder")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.remainder(x1._array, x2._array))


def round(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.round <numpy.round>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in round")
    return Array._new(np.round(x._array))


def sign(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.sign <numpy.sign>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in sign")
    return Array._new(np.sign(x._array))


def sin(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.sin <numpy.sin>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sin")
    return Array._new(np.sin(x._array))


def sinh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.sinh <numpy.sinh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sinh")
    return Array._new(np.sinh(x._array))


def square(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.square <numpy.square>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in square")
    return Array._new(np.square(x._array))


def sqrt(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.sqrt <numpy.sqrt>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in sqrt")
    return Array._new(np.sqrt(x._array))


def subtract(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.subtract <numpy.subtract>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in subtract")
    # Call result type here just to raise on disallowed type combinations
    _result_type(x1.dtype, x2.dtype)
    x1, x2 = Array._normalize_two_args(x1, x2)
    return Array._new(np.subtract(x1._array, x2._array))


def tan(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.tan <numpy.tan>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in tan")
    return Array._new(np.tan(x._array))


def tanh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.tanh <numpy.tanh>`.

    See its docstring for more information.
    """
    if x.dtype not in _floating_dtypes:
        raise TypeError("Only floating-point dtypes are allowed in tanh")
    return Array._new(np.tanh(x._array))


def trunc(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.trunc <numpy.trunc>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError("Only numeric dtypes are allowed in trunc")
    if x.dtype in _integer_dtypes:
        # Note: The return dtype of trunc is the same as the input
        return x
    return Array._new(np.trunc(x._array))
