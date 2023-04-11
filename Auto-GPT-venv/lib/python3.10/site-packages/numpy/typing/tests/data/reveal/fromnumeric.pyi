"""Tests for :mod:`core.fromnumeric`."""

import numpy as np
import numpy.typing as npt

class NDArraySubclass(npt.NDArray[np.complex128]):
    ...

AR_b: npt.NDArray[np.bool_]
AR_f4: npt.NDArray[np.float32]
AR_c16: npt.NDArray[np.complex128]
AR_u8: npt.NDArray[np.uint64]
AR_i8: npt.NDArray[np.int64]
AR_O: npt.NDArray[np.object_]
AR_subclass: NDArraySubclass

b: np.bool_
f4: np.float32
i8: np.int64
f: float

reveal_type(np.take(b, 0))  # E: bool_
reveal_type(np.take(f4, 0))  # E: {float32}
reveal_type(np.take(f, 0))  # E: Any
reveal_type(np.take(AR_b, 0))  # E: bool_
reveal_type(np.take(AR_f4, 0))  # E: {float32}
reveal_type(np.take(AR_b, [0]))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.take(AR_f4, [0]))  # E: ndarray[Any, dtype[{float32}]]
reveal_type(np.take([1], [0]))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.take(AR_f4, [0], out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.reshape(b, 1))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.reshape(f4, 1))  # E: ndarray[Any, dtype[{float32}]]
reveal_type(np.reshape(f, 1))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.reshape(AR_b, 1))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.reshape(AR_f4, 1))  # E: ndarray[Any, dtype[{float32}]]

reveal_type(np.choose(1, [True, True]))  # E: Any
reveal_type(np.choose([1], [True, True]))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.choose([1], AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.choose([1], AR_b, out=AR_f4))  # E: ndarray[Any, dtype[{float32}]]

reveal_type(np.repeat(b, 1))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.repeat(f4, 1))  # E: ndarray[Any, dtype[{float32}]]
reveal_type(np.repeat(f, 1))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.repeat(AR_b, 1))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.repeat(AR_f4, 1))  # E: ndarray[Any, dtype[{float32}]]

# TODO: array_bdd tests for np.put()

reveal_type(np.swapaxes([[0, 1]], 0, 0))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.swapaxes(AR_b, 0, 0))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.swapaxes(AR_f4, 0, 0))  # E: ndarray[Any, dtype[{float32}]]

reveal_type(np.transpose(b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.transpose(f4))  # E: ndarray[Any, dtype[{float32}]]
reveal_type(np.transpose(f))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.transpose(AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.transpose(AR_f4))  # E: ndarray[Any, dtype[{float32}]]

reveal_type(np.partition(b, 0, axis=None))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.partition(f4, 0, axis=None))  # E: ndarray[Any, dtype[{float32}]]
reveal_type(np.partition(f, 0, axis=None))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.partition(AR_b, 0))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.partition(AR_f4, 0))  # E: ndarray[Any, dtype[{float32}]]

reveal_type(np.argpartition(b, 0))  # E: ndarray[Any, dtype[{intp}]]
reveal_type(np.argpartition(f4, 0))  # E: ndarray[Any, dtype[{intp}]]
reveal_type(np.argpartition(f, 0))  # E: ndarray[Any, dtype[{intp}]]
reveal_type(np.argpartition(AR_b, 0))  # E: ndarray[Any, dtype[{intp}]]
reveal_type(np.argpartition(AR_f4, 0))  # E: ndarray[Any, dtype[{intp}]]

reveal_type(np.sort([2, 1], 0))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.sort(AR_b, 0))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.sort(AR_f4, 0))  # E: ndarray[Any, dtype[{float32}]]

reveal_type(np.argsort(AR_b, 0))  # E: ndarray[Any, dtype[{intp}]]
reveal_type(np.argsort(AR_f4, 0))  # E: ndarray[Any, dtype[{intp}]]

reveal_type(np.argmax(AR_b))  # E: {intp}
reveal_type(np.argmax(AR_f4))  # E: {intp}
reveal_type(np.argmax(AR_b, axis=0))  # E: Any
reveal_type(np.argmax(AR_f4, axis=0))  # E: Any
reveal_type(np.argmax(AR_f4, out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.argmin(AR_b))  # E: {intp}
reveal_type(np.argmin(AR_f4))  # E: {intp}
reveal_type(np.argmin(AR_b, axis=0))  # E: Any
reveal_type(np.argmin(AR_f4, axis=0))  # E: Any
reveal_type(np.argmin(AR_f4, out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.searchsorted(AR_b[0], 0))  # E: {intp}
reveal_type(np.searchsorted(AR_f4[0], 0))  # E: {intp}
reveal_type(np.searchsorted(AR_b[0], [0]))  # E: ndarray[Any, dtype[{intp}]]
reveal_type(np.searchsorted(AR_f4[0], [0]))  # E: ndarray[Any, dtype[{intp}]]

reveal_type(np.resize(b, (5, 5)))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.resize(f4, (5, 5)))  # E: ndarray[Any, dtype[{float32}]]
reveal_type(np.resize(f, (5, 5)))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.resize(AR_b, (5, 5)))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.resize(AR_f4, (5, 5)))  # E: ndarray[Any, dtype[{float32}]]

reveal_type(np.squeeze(b))  # E: bool_
reveal_type(np.squeeze(f4))  # E: {float32}
reveal_type(np.squeeze(f))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.squeeze(AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.squeeze(AR_f4))  # E: ndarray[Any, dtype[{float32}]]

reveal_type(np.diagonal(AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.diagonal(AR_f4))  # E: ndarray[Any, dtype[{float32}]]

reveal_type(np.trace(AR_b))  # E: Any
reveal_type(np.trace(AR_f4))  # E: Any
reveal_type(np.trace(AR_f4, out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.ravel(b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.ravel(f4))  # E: ndarray[Any, dtype[{float32}]]
reveal_type(np.ravel(f))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.ravel(AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.ravel(AR_f4))  # E: ndarray[Any, dtype[{float32}]]

reveal_type(np.nonzero(b))  # E: tuple[ndarray[Any, dtype[{intp}]], ...]
reveal_type(np.nonzero(f4))  # E: tuple[ndarray[Any, dtype[{intp}]], ...]
reveal_type(np.nonzero(f))  # E: tuple[ndarray[Any, dtype[{intp}]], ...]
reveal_type(np.nonzero(AR_b))  # E: tuple[ndarray[Any, dtype[{intp}]], ...]
reveal_type(np.nonzero(AR_f4))  # E: tuple[ndarray[Any, dtype[{intp}]], ...]

reveal_type(np.shape(b))  # E: tuple[builtins.int, ...]
reveal_type(np.shape(f4))  # E: tuple[builtins.int, ...]
reveal_type(np.shape(f))  # E: tuple[builtins.int, ...]
reveal_type(np.shape(AR_b))  # E: tuple[builtins.int, ...]
reveal_type(np.shape(AR_f4))  # E: tuple[builtins.int, ...]

reveal_type(np.compress([True], b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.compress([True], f4))  # E: ndarray[Any, dtype[{float32}]]
reveal_type(np.compress([True], f))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.compress([True], AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.compress([True], AR_f4))  # E: ndarray[Any, dtype[{float32}]]

reveal_type(np.clip(b, 0, 1.0))  # E: bool_
reveal_type(np.clip(f4, -1, 1))  # E: {float32}
reveal_type(np.clip(f, 0, 1))  # E: Any
reveal_type(np.clip(AR_b, 0, 1))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.clip(AR_f4, 0, 1))  # E: ndarray[Any, dtype[{float32}]]
reveal_type(np.clip([0], 0, 1))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.clip(AR_b, 0, 1, out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.sum(b))  # E: bool_
reveal_type(np.sum(f4))  # E: {float32}
reveal_type(np.sum(f))  # E: Any
reveal_type(np.sum(AR_b))  # E: bool_
reveal_type(np.sum(AR_f4))  # E: {float32}
reveal_type(np.sum(AR_b, axis=0))  # E: Any
reveal_type(np.sum(AR_f4, axis=0))  # E: Any
reveal_type(np.sum(AR_f4, out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.all(b))  # E: bool_
reveal_type(np.all(f4))  # E: bool_
reveal_type(np.all(f))  # E: bool_
reveal_type(np.all(AR_b))  # E: bool_
reveal_type(np.all(AR_f4))  # E: bool_
reveal_type(np.all(AR_b, axis=0))  # E: Any
reveal_type(np.all(AR_f4, axis=0))  # E: Any
reveal_type(np.all(AR_b, keepdims=True))  # E: Any
reveal_type(np.all(AR_f4, keepdims=True))  # E: Any
reveal_type(np.all(AR_f4, out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.any(b))  # E: bool_
reveal_type(np.any(f4))  # E: bool_
reveal_type(np.any(f))  # E: bool_
reveal_type(np.any(AR_b))  # E: bool_
reveal_type(np.any(AR_f4))  # E: bool_
reveal_type(np.any(AR_b, axis=0))  # E: Any
reveal_type(np.any(AR_f4, axis=0))  # E: Any
reveal_type(np.any(AR_b, keepdims=True))  # E: Any
reveal_type(np.any(AR_f4, keepdims=True))  # E: Any
reveal_type(np.any(AR_f4, out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.cumsum(b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.cumsum(f4))  # E: ndarray[Any, dtype[{float32}]]
reveal_type(np.cumsum(f))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.cumsum(AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.cumsum(AR_f4))  # E: ndarray[Any, dtype[{float32}]]
reveal_type(np.cumsum(f, dtype=float))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.cumsum(f, dtype=np.float64))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.cumsum(AR_f4, out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.ptp(b))  # E: bool_
reveal_type(np.ptp(f4))  # E: {float32}
reveal_type(np.ptp(f))  # E: Any
reveal_type(np.ptp(AR_b))  # E: bool_
reveal_type(np.ptp(AR_f4))  # E: {float32}
reveal_type(np.ptp(AR_b, axis=0))  # E: Any
reveal_type(np.ptp(AR_f4, axis=0))  # E: Any
reveal_type(np.ptp(AR_b, keepdims=True))  # E: Any
reveal_type(np.ptp(AR_f4, keepdims=True))  # E: Any
reveal_type(np.ptp(AR_f4, out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.amax(b))  # E: bool_
reveal_type(np.amax(f4))  # E: {float32}
reveal_type(np.amax(f))  # E: Any
reveal_type(np.amax(AR_b))  # E: bool_
reveal_type(np.amax(AR_f4))  # E: {float32}
reveal_type(np.amax(AR_b, axis=0))  # E: Any
reveal_type(np.amax(AR_f4, axis=0))  # E: Any
reveal_type(np.amax(AR_b, keepdims=True))  # E: Any
reveal_type(np.amax(AR_f4, keepdims=True))  # E: Any
reveal_type(np.amax(AR_f4, out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.amin(b))  # E: bool_
reveal_type(np.amin(f4))  # E: {float32}
reveal_type(np.amin(f))  # E: Any
reveal_type(np.amin(AR_b))  # E: bool_
reveal_type(np.amin(AR_f4))  # E: {float32}
reveal_type(np.amin(AR_b, axis=0))  # E: Any
reveal_type(np.amin(AR_f4, axis=0))  # E: Any
reveal_type(np.amin(AR_b, keepdims=True))  # E: Any
reveal_type(np.amin(AR_f4, keepdims=True))  # E: Any
reveal_type(np.amin(AR_f4, out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.prod(AR_b))  # E: {int_}
reveal_type(np.prod(AR_u8))  # E: {uint64}
reveal_type(np.prod(AR_i8))  # E: {int64}
reveal_type(np.prod(AR_f4))  # E: floating[Any]
reveal_type(np.prod(AR_c16))  # E: complexfloating[Any, Any]
reveal_type(np.prod(AR_O))  # E: Any
reveal_type(np.prod(AR_f4, axis=0))  # E: Any
reveal_type(np.prod(AR_f4, keepdims=True))  # E: Any
reveal_type(np.prod(AR_f4, dtype=np.float64))  # E: {float64}
reveal_type(np.prod(AR_f4, dtype=float))  # E: Any
reveal_type(np.prod(AR_f4, out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.cumprod(AR_b))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(np.cumprod(AR_u8))  # E: ndarray[Any, dtype[{uint64}]]
reveal_type(np.cumprod(AR_i8))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.cumprod(AR_f4))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.cumprod(AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.cumprod(AR_O))  # E: ndarray[Any, dtype[object_]]
reveal_type(np.cumprod(AR_f4, axis=0))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.cumprod(AR_f4, dtype=np.float64))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.cumprod(AR_f4, dtype=float))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.cumprod(AR_f4, out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.ndim(b))  # E: int
reveal_type(np.ndim(f4))  # E: int
reveal_type(np.ndim(f))  # E: int
reveal_type(np.ndim(AR_b))  # E: int
reveal_type(np.ndim(AR_f4))  # E: int

reveal_type(np.size(b))  # E: int
reveal_type(np.size(f4))  # E: int
reveal_type(np.size(f))  # E: int
reveal_type(np.size(AR_b))  # E: int
reveal_type(np.size(AR_f4))  # E: int

reveal_type(np.around(b))  # E: {float16}
reveal_type(np.around(f))  # E: Any
reveal_type(np.around(i8))  # E: {int64}
reveal_type(np.around(f4))  # E: {float32}
reveal_type(np.around(AR_b))  # E: ndarray[Any, dtype[{float16}]]
reveal_type(np.around(AR_i8))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.around(AR_f4))  # E: ndarray[Any, dtype[{float32}]]
reveal_type(np.around([1.5]))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.around(AR_f4, out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.mean(AR_b))  # E: floating[Any]
reveal_type(np.mean(AR_i8))  # E: floating[Any]
reveal_type(np.mean(AR_f4))  # E: floating[Any]
reveal_type(np.mean(AR_c16))  # E: complexfloating[Any, Any]
reveal_type(np.mean(AR_O))  # E: Any
reveal_type(np.mean(AR_f4, axis=0))  # E: Any
reveal_type(np.mean(AR_f4, keepdims=True))  # E: Any
reveal_type(np.mean(AR_f4, dtype=float))  # E: Any
reveal_type(np.mean(AR_f4, dtype=np.float64))  # E: {float64}
reveal_type(np.mean(AR_f4, out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.std(AR_b))  # E: floating[Any]
reveal_type(np.std(AR_i8))  # E: floating[Any]
reveal_type(np.std(AR_f4))  # E: floating[Any]
reveal_type(np.std(AR_c16))  # E: floating[Any]
reveal_type(np.std(AR_O))  # E: Any
reveal_type(np.std(AR_f4, axis=0))  # E: Any
reveal_type(np.std(AR_f4, keepdims=True))  # E: Any
reveal_type(np.std(AR_f4, dtype=float))  # E: Any
reveal_type(np.std(AR_f4, dtype=np.float64))  # E: {float64}
reveal_type(np.std(AR_f4, out=AR_subclass))  # E: NDArraySubclass

reveal_type(np.var(AR_b))  # E: floating[Any]
reveal_type(np.var(AR_i8))  # E: floating[Any]
reveal_type(np.var(AR_f4))  # E: floating[Any]
reveal_type(np.var(AR_c16))  # E: floating[Any]
reveal_type(np.var(AR_O))  # E: Any
reveal_type(np.var(AR_f4, axis=0))  # E: Any
reveal_type(np.var(AR_f4, keepdims=True))  # E: Any
reveal_type(np.var(AR_f4, dtype=float))  # E: Any
reveal_type(np.var(AR_f4, dtype=np.float64))  # E: {float64}
reveal_type(np.var(AR_f4, out=AR_subclass))  # E: NDArraySubclass
