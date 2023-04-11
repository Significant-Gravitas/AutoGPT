"""
Tests for :mod:`numpy.core.numeric`.

Does not include tests which fall under ``array_constructors``.

"""

from __future__ import annotations

import numpy as np

class SubClass(np.ndarray):
    ...

i8 = np.int64(1)

A = np.arange(27).reshape(3, 3, 3)
B: list[list[list[int]]] = A.tolist()
C = np.empty((27, 27)).view(SubClass)

np.count_nonzero(i8)
np.count_nonzero(A)
np.count_nonzero(B)
np.count_nonzero(A, keepdims=True)
np.count_nonzero(A, axis=0)

np.isfortran(i8)
np.isfortran(A)

np.argwhere(i8)
np.argwhere(A)

np.flatnonzero(i8)
np.flatnonzero(A)

np.correlate(B[0][0], A.ravel(), mode="valid")
np.correlate(A.ravel(), A.ravel(), mode="same")

np.convolve(B[0][0], A.ravel(), mode="valid")
np.convolve(A.ravel(), A.ravel(), mode="same")

np.outer(i8, A)
np.outer(B, A)
np.outer(A, A)
np.outer(A, A, out=C)

np.tensordot(B, A)
np.tensordot(A, A)
np.tensordot(A, A, axes=0)
np.tensordot(A, A, axes=(0, 1))

np.isscalar(i8)
np.isscalar(A)
np.isscalar(B)

np.roll(A, 1)
np.roll(A, (1, 2))
np.roll(B, 1)

np.rollaxis(A, 0, 1)

np.moveaxis(A, 0, 1)
np.moveaxis(A, (0, 1), (1, 2))

np.cross(B, A)
np.cross(A, A)

np.indices([0, 1, 2])
np.indices([0, 1, 2], sparse=False)
np.indices([0, 1, 2], sparse=True)

np.binary_repr(1)

np.base_repr(1)

np.allclose(i8, A)
np.allclose(B, A)
np.allclose(A, A)

np.isclose(i8, A)
np.isclose(B, A)
np.isclose(A, A)

np.array_equal(i8, A)
np.array_equal(B, A)
np.array_equal(A, A)

np.array_equiv(i8, A)
np.array_equiv(B, A)
np.array_equiv(A, A)
