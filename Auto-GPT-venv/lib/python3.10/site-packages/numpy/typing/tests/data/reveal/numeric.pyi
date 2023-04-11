"""
Tests for :mod:`core.numeric`.

Does not include tests which fall under ``array_constructors``.

"""

import numpy as np
import numpy.typing as npt

class SubClass(npt.NDArray[np.int64]):
    ...

i8: np.int64

AR_b: npt.NDArray[np.bool_]
AR_u8: npt.NDArray[np.uint64]
AR_i8: npt.NDArray[np.int64]
AR_f8: npt.NDArray[np.float64]
AR_c16: npt.NDArray[np.complex128]
AR_m: npt.NDArray[np.timedelta64]
AR_O: npt.NDArray[np.object_]

B: list[int]
C: SubClass

reveal_type(np.count_nonzero(i8))  # E: int
reveal_type(np.count_nonzero(AR_i8))  # E: int
reveal_type(np.count_nonzero(B))  # E: int
reveal_type(np.count_nonzero(AR_i8, keepdims=True))  # E: Any
reveal_type(np.count_nonzero(AR_i8, axis=0))  # E: Any

reveal_type(np.isfortran(i8))  # E: bool
reveal_type(np.isfortran(AR_i8))  # E: bool

reveal_type(np.argwhere(i8))  # E: ndarray[Any, dtype[{intp}]]
reveal_type(np.argwhere(AR_i8))  # E: ndarray[Any, dtype[{intp}]]

reveal_type(np.flatnonzero(i8))  # E: ndarray[Any, dtype[{intp}]]
reveal_type(np.flatnonzero(AR_i8))  # E: ndarray[Any, dtype[{intp}]]

reveal_type(np.correlate(B, AR_i8, mode="valid"))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.correlate(AR_i8, AR_i8, mode="same"))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.correlate(AR_b, AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.correlate(AR_b, AR_u8))  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(np.correlate(AR_i8, AR_b))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.correlate(AR_i8, AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.correlate(AR_i8, AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.correlate(AR_i8, AR_m))  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(np.correlate(AR_O, AR_O))  # E: ndarray[Any, dtype[object_]]

reveal_type(np.convolve(B, AR_i8, mode="valid"))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.convolve(AR_i8, AR_i8, mode="same"))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.convolve(AR_b, AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.convolve(AR_b, AR_u8))  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(np.convolve(AR_i8, AR_b))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.convolve(AR_i8, AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.convolve(AR_i8, AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.convolve(AR_i8, AR_m))  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(np.convolve(AR_O, AR_O))  # E: ndarray[Any, dtype[object_]]

reveal_type(np.outer(i8, AR_i8))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.outer(B, AR_i8))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.outer(AR_i8, AR_i8))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.outer(AR_i8, AR_i8, out=C))  # E: SubClass
reveal_type(np.outer(AR_b, AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.outer(AR_b, AR_u8))  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(np.outer(AR_i8, AR_b))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.convolve(AR_i8, AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.outer(AR_i8, AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.outer(AR_i8, AR_m))  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(np.outer(AR_O, AR_O))  # E: ndarray[Any, dtype[object_]]

reveal_type(np.tensordot(B, AR_i8))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.tensordot(AR_i8, AR_i8))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.tensordot(AR_i8, AR_i8, axes=0))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.tensordot(AR_i8, AR_i8, axes=(0, 1)))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.tensordot(AR_b, AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.tensordot(AR_b, AR_u8))  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(np.tensordot(AR_i8, AR_b))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.tensordot(AR_i8, AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.tensordot(AR_i8, AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.tensordot(AR_i8, AR_m))  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(np.tensordot(AR_O, AR_O))  # E: ndarray[Any, dtype[object_]]

reveal_type(np.isscalar(i8))  # E: bool
reveal_type(np.isscalar(AR_i8))  # E: bool
reveal_type(np.isscalar(B))  # E: bool

reveal_type(np.roll(AR_i8, 1))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.roll(AR_i8, (1, 2)))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.roll(B, 1))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.rollaxis(AR_i8, 0, 1))  # E: ndarray[Any, dtype[{int64}]]

reveal_type(np.moveaxis(AR_i8, 0, 1))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.moveaxis(AR_i8, (0, 1), (1, 2)))  # E: ndarray[Any, dtype[{int64}]]

reveal_type(np.cross(B, AR_i8))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.cross(AR_i8, AR_i8))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.cross(AR_b, AR_u8))  # E: ndarray[Any, dtype[unsignedinteger[Any]]]
reveal_type(np.cross(AR_i8, AR_b))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.cross(AR_i8, AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.cross(AR_i8, AR_c16))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.cross(AR_O, AR_O))  # E: ndarray[Any, dtype[object_]]

reveal_type(np.indices([0, 1, 2]))  # E: ndarray[Any, dtype[{int_}]]
reveal_type(np.indices([0, 1, 2], sparse=True))  # E: tuple[ndarray[Any, dtype[{int_}]], ...]
reveal_type(np.indices([0, 1, 2], dtype=np.float64))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.indices([0, 1, 2], sparse=True, dtype=np.float64))  # E: tuple[ndarray[Any, dtype[{float64}]], ...]
reveal_type(np.indices([0, 1, 2], dtype=float))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.indices([0, 1, 2], sparse=True, dtype=float))  # E: tuple[ndarray[Any, dtype[Any]], ...]

reveal_type(np.binary_repr(1))  # E: str

reveal_type(np.base_repr(1))  # E: str

reveal_type(np.allclose(i8, AR_i8))  # E: bool
reveal_type(np.allclose(B, AR_i8))  # E: bool
reveal_type(np.allclose(AR_i8, AR_i8))  # E: bool

reveal_type(np.isclose(i8, i8))  # E: bool_
reveal_type(np.isclose(i8, AR_i8))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.isclose(B, AR_i8))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.isclose(AR_i8, AR_i8))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.array_equal(i8, AR_i8))  # E: bool
reveal_type(np.array_equal(B, AR_i8))  # E: bool
reveal_type(np.array_equal(AR_i8, AR_i8))  # E: bool

reveal_type(np.array_equiv(i8, AR_i8))  # E: bool
reveal_type(np.array_equiv(B, AR_i8))  # E: bool
reveal_type(np.array_equiv(AR_i8, AR_i8))  # E: bool
