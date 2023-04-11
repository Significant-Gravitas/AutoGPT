from typing import Any, TypeVar
from pathlib import Path

import numpy as np
import numpy.typing as npt

_SCT = TypeVar("_SCT", bound=np.generic, covariant=True)

class SubClass(np.ndarray[Any, np.dtype[_SCT]]): ...

i8: np.int64

A: npt.NDArray[np.float64]
B: SubClass[np.float64]
C: list[int]

def func(i: int, j: int, **kwargs: Any) -> SubClass[np.float64]: ...

reveal_type(np.empty_like(A))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.empty_like(B))  # E: SubClass[{float64}]
reveal_type(np.empty_like([1, 1.0]))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.empty_like(A, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.empty_like(A, dtype='c16'))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.array(A))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.array(B))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.array(B, subok=True))  # E: SubClass[{float64}]
reveal_type(np.array([1, 1.0]))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.array(A, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.array(A, dtype='c16'))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.array(A, like=A))  # E: ndarray[Any, dtype[{float64}]]

reveal_type(np.zeros([1, 5, 6]))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.zeros([1, 5, 6], dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.zeros([1, 5, 6], dtype='c16'))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.empty([1, 5, 6]))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.empty([1, 5, 6], dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.empty([1, 5, 6], dtype='c16'))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.concatenate(A))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.concatenate([A, A]))  # E: Any
reveal_type(np.concatenate([[1], A]))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.concatenate([[1], [1]]))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.concatenate((A, A)))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.concatenate(([1], [1])))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.concatenate([1, 1.0]))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.concatenate(A, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.concatenate(A, dtype='c16'))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.concatenate([1, 1.0], out=A))  # E: ndarray[Any, dtype[{float64}]]

reveal_type(np.asarray(A))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.asarray(B))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.asarray([1, 1.0]))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.asarray(A, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.asarray(A, dtype='c16'))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.asanyarray(A))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.asanyarray(B))  # E: SubClass[{float64}]
reveal_type(np.asanyarray([1, 1.0]))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.asanyarray(A, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.asanyarray(A, dtype='c16'))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.ascontiguousarray(A))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.ascontiguousarray(B))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.ascontiguousarray([1, 1.0]))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.ascontiguousarray(A, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.ascontiguousarray(A, dtype='c16'))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.asfortranarray(A))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.asfortranarray(B))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.asfortranarray([1, 1.0]))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.asfortranarray(A, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.asfortranarray(A, dtype='c16'))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.fromstring("1 1 1", sep=" "))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.fromstring(b"1 1 1", sep=" "))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.fromstring("1 1 1", dtype=np.int64, sep=" "))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.fromstring(b"1 1 1", dtype=np.int64, sep=" "))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.fromstring("1 1 1", dtype="c16", sep=" "))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.fromstring(b"1 1 1", dtype="c16", sep=" "))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.fromfile("test.txt", sep=" "))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.fromfile("test.txt", dtype=np.int64, sep=" "))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.fromfile("test.txt", dtype="c16", sep=" "))  # E: ndarray[Any, dtype[Any]]
with open("test.txt") as f:
    reveal_type(np.fromfile(f, sep=" "))  # E: ndarray[Any, dtype[{float64}]]
    reveal_type(np.fromfile(b"test.txt", sep=" "))  # E: ndarray[Any, dtype[{float64}]]
    reveal_type(np.fromfile(Path("test.txt"), sep=" "))  # E: ndarray[Any, dtype[{float64}]]

reveal_type(np.fromiter("12345", np.float64))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.fromiter("12345", float))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.frombuffer(A))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.frombuffer(A, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.frombuffer(A, dtype="c16"))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.arange(False, True))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.arange(10))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.arange(0, 10, step=2))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.arange(10.0))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.arange(start=0, stop=10.0))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.arange(np.timedelta64(0)))  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(np.arange(0, np.timedelta64(10)))  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(np.arange(np.datetime64("0"), np.datetime64("10")))  # E: ndarray[Any, dtype[datetime64]]
reveal_type(np.arange(10, dtype=np.float64))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.arange(0, 10, step=2, dtype=np.int16))  # E: ndarray[Any, dtype[{int16}]]
reveal_type(np.arange(10, dtype=int))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.arange(0, 10, dtype="f8"))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.require(A))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.require(B))  # E: SubClass[{float64}]
reveal_type(np.require(B, requirements=None))  # E: SubClass[{float64}]
reveal_type(np.require(B, dtype=int))  # E: ndarray[Any, Any]
reveal_type(np.require(B, requirements="E"))  # E: ndarray[Any, Any]
reveal_type(np.require(B, requirements=["ENSUREARRAY"]))  # E: ndarray[Any, Any]
reveal_type(np.require(B, requirements={"F", "E"}))  # E: ndarray[Any, Any]
reveal_type(np.require(B, requirements=["C", "OWNDATA"]))  # E: SubClass[{float64}]
reveal_type(np.require(B, requirements="W"))  # E: SubClass[{float64}]
reveal_type(np.require(B, requirements="A"))  # E: SubClass[{float64}]
reveal_type(np.require(C))  # E: ndarray[Any, Any]

reveal_type(np.linspace(0, 10))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.linspace(0, 10j))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.linspace(0, 10, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.linspace(0, 10, dtype=int))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.linspace(0, 10, retstep=True))  # E: Tuple[ndarray[Any, dtype[floating[Any]]], floating[Any]]
reveal_type(np.linspace(0j, 10, retstep=True))  # E: Tuple[ndarray[Any, dtype[complexfloating[Any, Any]]], complexfloating[Any, Any]]
reveal_type(np.linspace(0, 10, retstep=True, dtype=np.int64))  # E: Tuple[ndarray[Any, dtype[{int64}]], {int64}]
reveal_type(np.linspace(0j, 10, retstep=True, dtype=int))  # E: Tuple[ndarray[Any, dtype[Any]], Any]

reveal_type(np.logspace(0, 10))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.logspace(0, 10j))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.logspace(0, 10, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.logspace(0, 10, dtype=int))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.geomspace(0, 10))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.geomspace(0, 10j))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.geomspace(0, 10, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.geomspace(0, 10, dtype=int))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.zeros_like(A))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.zeros_like(C))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.zeros_like(A, dtype=float))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.zeros_like(B))  # E: SubClass[{float64}]
reveal_type(np.zeros_like(B, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]

reveal_type(np.ones_like(A))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.ones_like(C))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.ones_like(A, dtype=float))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.ones_like(B))  # E: SubClass[{float64}]
reveal_type(np.ones_like(B, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]

reveal_type(np.full_like(A, i8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.full_like(C, i8))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.full_like(A, i8, dtype=int))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.full_like(B, i8))  # E: SubClass[{float64}]
reveal_type(np.full_like(B, i8, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]

reveal_type(np.ones(1))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.ones([1, 1, 1]))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.ones(5, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.ones(5, dtype=int))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.full(1, i8))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.full([1, 1, 1], i8))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.full(1, i8, dtype=np.float64))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.full(1, i8, dtype=float))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.indices([1, 2, 3]))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(np.indices([1, 2, 3], sparse=True))  # E: tuple[ndarray[Any, dtype[{int_}]], ...]

reveal_type(np.fromfunction(func, (3, 5)))  # E: SubClass[{float64}]

reveal_type(np.identity(10))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.identity(10, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.identity(10, dtype=int))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.atleast_1d(A))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.atleast_1d(C))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.atleast_1d(A, A))  # E: list[ndarray[Any, dtype[Any]]]
reveal_type(np.atleast_1d(A, C))  # E: list[ndarray[Any, dtype[Any]]]
reveal_type(np.atleast_1d(C, C))  # E: list[ndarray[Any, dtype[Any]]]

reveal_type(np.atleast_2d(A))  # E: ndarray[Any, dtype[{float64}]]

reveal_type(np.atleast_3d(A))  # E: ndarray[Any, dtype[{float64}]]

reveal_type(np.vstack([A, A]))  # E: ndarray[Any, Any]
reveal_type(np.vstack([A, A], dtype=np.float64))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.vstack([A, C]))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.vstack([C, C]))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.hstack([A, A]))  # E: ndarray[Any, Any]
reveal_type(np.hstack([A, A], dtype=np.float64))  # E: ndarray[Any, dtype[{float64}]]

reveal_type(np.stack([A, A]))  # E: Any
reveal_type(np.stack([A, A], dtype=np.float64))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.stack([A, C]))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.stack([C, C]))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.stack([A, A], axis=0))  # E: Any
reveal_type(np.stack([A, A], out=B))  # E: SubClass[{float64}]

reveal_type(np.block([[A, A], [A, A]]))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.block(C))  # E: ndarray[Any, dtype[Any]]
