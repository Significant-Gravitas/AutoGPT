from typing import Any
import numpy as np

AR_LIKE_b: list[bool]
AR_LIKE_u: list[np.uint32]
AR_LIKE_i: list[int]
AR_LIKE_f: list[float]
AR_LIKE_O: list[np.object_]

AR_U: np.ndarray[Any, np.dtype[np.str_]]

reveal_type(np.fix(AR_LIKE_b))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.fix(AR_LIKE_u))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.fix(AR_LIKE_i))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.fix(AR_LIKE_f))  # E: ndarray[Any, dtype[floating[Any]]]
reveal_type(np.fix(AR_LIKE_O))  # E: Any
reveal_type(np.fix(AR_LIKE_f, out=AR_U))  # E: ndarray[Any, dtype[str_]]

reveal_type(np.isposinf(AR_LIKE_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.isposinf(AR_LIKE_u))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.isposinf(AR_LIKE_i))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.isposinf(AR_LIKE_f))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.isposinf(AR_LIKE_f, out=AR_U))  # E: ndarray[Any, dtype[str_]]

reveal_type(np.isneginf(AR_LIKE_b))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.isneginf(AR_LIKE_u))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.isneginf(AR_LIKE_i))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.isneginf(AR_LIKE_f))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.isneginf(AR_LIKE_f, out=AR_U))  # E: ndarray[Any, dtype[str_]]
