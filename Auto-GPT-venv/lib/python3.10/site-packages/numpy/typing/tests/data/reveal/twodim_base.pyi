from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt

_SCT = TypeVar("_SCT", bound=np.generic)


def func1(ar: npt.NDArray[_SCT], a: int) -> npt.NDArray[_SCT]:
    pass


def func2(ar: npt.NDArray[np.number[Any]], a: str) -> npt.NDArray[np.float64]:
    pass


AR_b: npt.NDArray[np.bool_]
AR_u: npt.NDArray[np.uint64]
AR_i: npt.NDArray[np.int64]
AR_f: npt.NDArray[np.float64]
AR_c: npt.NDArray[np.complex128]
AR_O: npt.NDArray[np.object_]

AR_LIKE_b: list[bool]

reveal_type(np.fliplr(AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.fliplr(AR_LIKE_b))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.flipud(AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.flipud(AR_LIKE_b))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.eye(10))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.eye(10, M=20, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.eye(10, k=2, dtype=int))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.diag(AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.diag(AR_LIKE_b, k=0))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.diagflat(AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.diagflat(AR_LIKE_b, k=0))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.tri(10))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.tri(10, M=20, dtype=np.int64))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.tri(10, k=2, dtype=int))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.tril(AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.tril(AR_LIKE_b, k=0))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.triu(AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.triu(AR_LIKE_b, k=0))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.vander(AR_b))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.vander(AR_u))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.vander(AR_i, N=2))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.vander(AR_f, increasing=True))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.vander(AR_c))  # E: ndarray[Any, dtype[complexfloating[Any, Any]]]
reveal_type(np.vander(AR_O))  # E: ndarray[Any, dtype[object_]]

reveal_type(np.histogram2d(AR_i, AR_b))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[floating[Any]]]]
reveal_type(np.histogram2d(AR_f, AR_f))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[floating[Any]]], ndarray[Any, dtype[floating[Any]]]]
reveal_type(np.histogram2d(AR_f, AR_c, weights=AR_LIKE_b))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[complexfloating[Any, Any]]], ndarray[Any, dtype[complexfloating[Any, Any]]]]

reveal_type(np.mask_indices(10, func1))  # E: Tuple[ndarray[Any, dtype[{intp}]], ndarray[Any, dtype[{intp}]]]
reveal_type(np.mask_indices(8, func2, "0"))  # E: Tuple[ndarray[Any, dtype[{intp}]], ndarray[Any, dtype[{intp}]]]

reveal_type(np.tril_indices(10))  # E: Tuple[ndarray[Any, dtype[{int_}]], ndarray[Any, dtype[{int_}]]]

reveal_type(np.tril_indices_from(AR_b))  # E: Tuple[ndarray[Any, dtype[{int_}]], ndarray[Any, dtype[{int_}]]]

reveal_type(np.triu_indices(10))  # E: Tuple[ndarray[Any, dtype[{int_}]], ndarray[Any, dtype[{int_}]]]

reveal_type(np.triu_indices_from(AR_b))  # E: Tuple[ndarray[Any, dtype[{int_}]], ndarray[Any, dtype[{int_}]]]
