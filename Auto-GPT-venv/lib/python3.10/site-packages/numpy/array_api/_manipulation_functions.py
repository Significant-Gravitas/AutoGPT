from __future__ import annotations

from ._array_object import Array
from ._data_type_functions import result_type

from typing import List, Optional, Tuple, Union

import numpy as np

# Note: the function name is different here
def concat(
    arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: Optional[int] = 0
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.concatenate <numpy.concatenate>`.

    See its docstring for more information.
    """
    # Note: Casting rules here are different from the np.concatenate default
    # (no for scalars with axis=None, no cross-kind casting)
    dtype = result_type(*arrays)
    arrays = tuple(a._array for a in arrays)
    return Array._new(np.concatenate(arrays, axis=axis, dtype=dtype))


def expand_dims(x: Array, /, *, axis: int) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.expand_dims <numpy.expand_dims>`.

    See its docstring for more information.
    """
    return Array._new(np.expand_dims(x._array, axis))


def flip(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.flip <numpy.flip>`.

    See its docstring for more information.
    """
    return Array._new(np.flip(x._array, axis=axis))


# Note: The function name is different here (see also matrix_transpose).
# Unlike transpose(), the axes argument is required.
def permute_dims(x: Array, /, axes: Tuple[int, ...]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.transpose <numpy.transpose>`.

    See its docstring for more information.
    """
    return Array._new(np.transpose(x._array, axes))


# Note: the optional argument is called 'shape', not 'newshape'
def reshape(x: Array, /, shape: Tuple[int, ...]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.reshape <numpy.reshape>`.

    See its docstring for more information.
    """
    return Array._new(np.reshape(x._array, shape))


def roll(
    x: Array,
    /,
    shift: Union[int, Tuple[int, ...]],
    *,
    axis: Optional[Union[int, Tuple[int, ...]]] = None,
) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.roll <numpy.roll>`.

    See its docstring for more information.
    """
    return Array._new(np.roll(x._array, shift, axis=axis))


def squeeze(x: Array, /, axis: Union[int, Tuple[int, ...]]) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.squeeze <numpy.squeeze>`.

    See its docstring for more information.
    """
    return Array._new(np.squeeze(x._array, axis=axis))


def stack(arrays: Union[Tuple[Array, ...], List[Array]], /, *, axis: int = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.stack <numpy.stack>`.

    See its docstring for more information.
    """
    # Call result type here just to raise on disallowed type combinations
    result_type(*arrays)
    arrays = tuple(a._array for a in arrays)
    return Array._new(np.stack(arrays, axis=axis))
