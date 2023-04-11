import sys
from typing import Any
import numpy as np


class Index:
    def __index__(self) -> int:
        return 0


class SubClass(np.ndarray):
    pass


def func(i: int, j: int, **kwargs: Any) -> SubClass:
    return B


i8 = np.int64(1)

A = np.array([1])
B = A.view(SubClass).copy()
B_stack = np.array([[1], [1]]).view(SubClass)
C = [1]

np.ndarray(Index())
np.ndarray([Index()])

np.array(1, dtype=float)
np.array(1, copy=False)
np.array(1, order='F')
np.array(1, order=None)
np.array(1, subok=True)
np.array(1, ndmin=3)
np.array(1, str, copy=True, order='C', subok=False, ndmin=2)

np.asarray(A)
np.asarray(B)
np.asarray(C)

np.asanyarray(A)
np.asanyarray(B)
np.asanyarray(B, dtype=int)
np.asanyarray(C)

np.ascontiguousarray(A)
np.ascontiguousarray(B)
np.ascontiguousarray(C)

np.asfortranarray(A)
np.asfortranarray(B)
np.asfortranarray(C)

np.require(A)
np.require(B)
np.require(B, dtype=int)
np.require(B, requirements=None)
np.require(B, requirements="E")
np.require(B, requirements=["ENSUREARRAY"])
np.require(B, requirements={"F", "E"})
np.require(B, requirements=["C", "OWNDATA"])
np.require(B, requirements="W")
np.require(B, requirements="A")
np.require(C)

np.linspace(0, 2)
np.linspace(0.5, [0, 1, 2])
np.linspace([0, 1, 2], 3)
np.linspace(0j, 2)
np.linspace(0, 2, num=10)
np.linspace(0, 2, endpoint=True)
np.linspace(0, 2, retstep=True)
np.linspace(0j, 2j, retstep=True)
np.linspace(0, 2, dtype=bool)
np.linspace([0, 1], [2, 3], axis=Index())

np.logspace(0, 2, base=2)
np.logspace(0, 2, base=2)
np.logspace(0, 2, base=[1j, 2j], num=2)

np.geomspace(1, 2)

np.zeros_like(A)
np.zeros_like(C)
np.zeros_like(B)
np.zeros_like(B, dtype=np.int64)

np.ones_like(A)
np.ones_like(C)
np.ones_like(B)
np.ones_like(B, dtype=np.int64)

np.empty_like(A)
np.empty_like(C)
np.empty_like(B)
np.empty_like(B, dtype=np.int64)

np.full_like(A, i8)
np.full_like(C, i8)
np.full_like(B, i8)
np.full_like(B, i8, dtype=np.int64)

np.ones(1)
np.ones([1, 1, 1])

np.full(1, i8)
np.full([1, 1, 1], i8)

np.indices([1, 2, 3])
np.indices([1, 2, 3], sparse=True)

np.fromfunction(func, (3, 5))

np.identity(10)

np.atleast_1d(C)
np.atleast_1d(A)
np.atleast_1d(C, C)
np.atleast_1d(C, A)
np.atleast_1d(A, A)

np.atleast_2d(C)

np.atleast_3d(C)

np.vstack([C, C])
np.vstack([C, A])
np.vstack([A, A])

np.hstack([C, C])

np.stack([C, C])
np.stack([C, C], axis=0)
np.stack([C, C], out=B_stack)

np.block([[C, C], [C, C]])
np.block(A)
