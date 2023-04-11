import numpy as np
from numpy._typing import NDArray
from typing import Any

i8: np.int64
f8: np.float64

AR_b: NDArray[np.bool_]
AR_i8: NDArray[np.int64]
AR_f8: NDArray[np.float64]

AR_LIKE_f8: list[float]

reveal_type(np.take_along_axis(AR_f8, AR_i8, axis=1))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.take_along_axis(f8, AR_i8, axis=None))  # E: ndarray[Any, dtype[{float64}]]

reveal_type(np.put_along_axis(AR_f8, AR_i8, "1.0", axis=1))  # E: None

reveal_type(np.expand_dims(AR_i8, 2))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.expand_dims(AR_LIKE_f8, 2))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.column_stack([AR_i8]))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.column_stack([AR_LIKE_f8]))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.dstack([AR_i8]))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.dstack([AR_LIKE_f8]))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.row_stack([AR_i8]))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.row_stack([AR_LIKE_f8]))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.array_split(AR_i8, [3, 5, 6, 10]))  # E: list[ndarray[Any, dtype[{int64}]]]
reveal_type(np.array_split(AR_LIKE_f8, [3, 5, 6, 10]))  # E: list[ndarray[Any, dtype[Any]]]

reveal_type(np.split(AR_i8, [3, 5, 6, 10]))  # E: list[ndarray[Any, dtype[{int64}]]]
reveal_type(np.split(AR_LIKE_f8, [3, 5, 6, 10]))  # E: list[ndarray[Any, dtype[Any]]]

reveal_type(np.hsplit(AR_i8, [3, 5, 6, 10]))  # E: list[ndarray[Any, dtype[{int64}]]]
reveal_type(np.hsplit(AR_LIKE_f8, [3, 5, 6, 10]))  # E: list[ndarray[Any, dtype[Any]]]

reveal_type(np.vsplit(AR_i8, [3, 5, 6, 10]))  # E: list[ndarray[Any, dtype[{int64}]]]
reveal_type(np.vsplit(AR_LIKE_f8, [3, 5, 6, 10]))  # E: list[ndarray[Any, dtype[Any]]]

reveal_type(np.dsplit(AR_i8, [3, 5, 6, 10]))  # E: list[ndarray[Any, dtype[{int64}]]]
reveal_type(np.dsplit(AR_LIKE_f8, [3, 5, 6, 10]))  # E: list[ndarray[Any, dtype[Any]]]

reveal_type(np.lib.shape_base.get_array_prepare(AR_i8))  # E: lib.shape_base._ArrayPrepare
reveal_type(np.lib.shape_base.get_array_prepare(AR_i8, 1))  # E: Union[None, lib.shape_base._ArrayPrepare]

reveal_type(np.get_array_wrap(AR_i8))  # E: lib.shape_base._ArrayWrap
reveal_type(np.get_array_wrap(AR_i8, 1))  # E: Union[None, lib.shape_base._ArrayWrap]

reveal_type(np.kron(AR_b, AR_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.kron(AR_b, AR_i8))  # E: ndarray[Any, dtype[signedinteger[Any]]]
reveal_type(np.kron(AR_f8, AR_f8))  # E: ndarray[Any, dtype[floating[Any]]]

reveal_type(np.tile(AR_i8, 5))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.tile(AR_LIKE_f8, [2, 2]))  # E: ndarray[Any, dtype[Any]]
