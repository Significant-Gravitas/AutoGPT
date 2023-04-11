from __future__ import annotations

from ._array_object import Array

from typing import NamedTuple

import numpy as np

# Note: np.unique() is split into four functions in the array API:
# unique_all, unique_counts, unique_inverse, and unique_values (this is done
# to remove polymorphic return types).

# Note: The various unique() functions are supposed to return multiple NaNs.
# This does not match the NumPy behavior, however, this is currently left as a
# TODO in this implementation as this behavior may be reverted in np.unique().
# See https://github.com/numpy/numpy/issues/20326.

# Note: The functions here return a namedtuple (np.unique() returns a normal
# tuple).

class UniqueAllResult(NamedTuple):
    values: Array
    indices: Array
    inverse_indices: Array
    counts: Array


class UniqueCountsResult(NamedTuple):
    values: Array
    counts: Array


class UniqueInverseResult(NamedTuple):
    values: Array
    inverse_indices: Array


def unique_all(x: Array, /) -> UniqueAllResult:
    """
    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.

    See its docstring for more information.
    """
    values, indices, inverse_indices, counts = np.unique(
        x._array,
        return_counts=True,
        return_index=True,
        return_inverse=True,
        equal_nan=False,
    )
    # np.unique() flattens inverse indices, but they need to share x's shape
    # See https://github.com/numpy/numpy/issues/20638
    inverse_indices = inverse_indices.reshape(x.shape)
    return UniqueAllResult(
        Array._new(values),
        Array._new(indices),
        Array._new(inverse_indices),
        Array._new(counts),
    )


def unique_counts(x: Array, /) -> UniqueCountsResult:
    res = np.unique(
        x._array,
        return_counts=True,
        return_index=False,
        return_inverse=False,
        equal_nan=False,
    )

    return UniqueCountsResult(*[Array._new(i) for i in res])


def unique_inverse(x: Array, /) -> UniqueInverseResult:
    """
    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.

    See its docstring for more information.
    """
    values, inverse_indices = np.unique(
        x._array,
        return_counts=False,
        return_index=False,
        return_inverse=True,
        equal_nan=False,
    )
    # np.unique() flattens inverse indices, but they need to share x's shape
    # See https://github.com/numpy/numpy/issues/20638
    inverse_indices = inverse_indices.reshape(x.shape)
    return UniqueInverseResult(Array._new(values), Array._new(inverse_indices))


def unique_values(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.unique <numpy.unique>`.

    See its docstring for more information.
    """
    res = np.unique(
        x._array,
        return_counts=False,
        return_index=False,
        return_inverse=False,
        equal_nan=False,
    )
    return Array._new(res)
