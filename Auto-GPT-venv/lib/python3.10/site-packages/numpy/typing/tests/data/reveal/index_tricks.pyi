from typing import Any
import numpy as np

AR_LIKE_b: list[bool]
AR_LIKE_i: list[int]
AR_LIKE_f: list[float]
AR_LIKE_U: list[str]

AR_i8: np.ndarray[Any, np.dtype[np.int64]]

reveal_type(np.ndenumerate(AR_i8))  # E: ndenumerate[{int64}]
reveal_type(np.ndenumerate(AR_LIKE_f))  # E: ndenumerate[{double}]
reveal_type(np.ndenumerate(AR_LIKE_U))  # E: ndenumerate[str_]

reveal_type(np.ndenumerate(AR_i8).iter)  # E: flatiter[ndarray[Any, dtype[{int64}]]]
reveal_type(np.ndenumerate(AR_LIKE_f).iter)  # E: flatiter[ndarray[Any, dtype[{double}]]]
reveal_type(np.ndenumerate(AR_LIKE_U).iter)  # E: flatiter[ndarray[Any, dtype[str_]]]

reveal_type(next(np.ndenumerate(AR_i8)))  # E: Tuple[builtins.tuple[builtins.int, ...], {int64}]
reveal_type(next(np.ndenumerate(AR_LIKE_f)))  # E: Tuple[builtins.tuple[builtins.int, ...], {double}]
reveal_type(next(np.ndenumerate(AR_LIKE_U)))  # E: Tuple[builtins.tuple[builtins.int, ...], str_]

reveal_type(iter(np.ndenumerate(AR_i8)))  # E: ndenumerate[{int64}]
reveal_type(iter(np.ndenumerate(AR_LIKE_f)))  # E: ndenumerate[{double}]
reveal_type(iter(np.ndenumerate(AR_LIKE_U)))  # E: ndenumerate[str_]

reveal_type(np.ndindex(1, 2, 3))  # E: numpy.ndindex
reveal_type(np.ndindex((1, 2, 3)))  # E: numpy.ndindex
reveal_type(iter(np.ndindex(1, 2, 3)))  # E: ndindex
reveal_type(next(np.ndindex(1, 2, 3)))  # E: builtins.tuple[builtins.int, ...]

reveal_type(np.unravel_index([22, 41, 37], (7, 6)))  # E: tuple[ndarray[Any, dtype[{intp}]], ...]
reveal_type(np.unravel_index([31, 41, 13], (7, 6), order="F"))  # E: tuple[ndarray[Any, dtype[{intp}]], ...]
reveal_type(np.unravel_index(1621, (6, 7, 8, 9)))  # E: tuple[{intp}, ...]

reveal_type(np.ravel_multi_index([[1]], (7, 6)))  # E: ndarray[Any, dtype[{intp}]]
reveal_type(np.ravel_multi_index(AR_LIKE_i, (7, 6)))  # E: {intp}
reveal_type(np.ravel_multi_index(AR_LIKE_i, (7, 6), order="F"))  # E: {intp}
reveal_type(np.ravel_multi_index(AR_LIKE_i, (4, 6), mode="clip"))  # E: {intp}
reveal_type(np.ravel_multi_index(AR_LIKE_i, (4, 4), mode=("clip", "wrap")))  # E: {intp}
reveal_type(np.ravel_multi_index((3, 1, 4, 1), (6, 7, 8, 9)))  # E: {intp}

reveal_type(np.mgrid[1:1:2])  # E: ndarray[Any, dtype[Any]]
reveal_type(np.mgrid[1:1:2, None:10])  # E: ndarray[Any, dtype[Any]]

reveal_type(np.ogrid[1:1:2])  # E: list[ndarray[Any, dtype[Any]]]
reveal_type(np.ogrid[1:1:2, None:10])  # E: list[ndarray[Any, dtype[Any]]]

reveal_type(np.index_exp[0:1])  # E: Tuple[builtins.slice]
reveal_type(np.index_exp[0:1, None:3])  # E: Tuple[builtins.slice, builtins.slice]
reveal_type(np.index_exp[0, 0:1, ..., [0, 1, 3]])  # E: Tuple[Literal[0]?, builtins.slice, builtins.ellipsis, builtins.list[builtins.int]]

reveal_type(np.s_[0:1])  # E: builtins.slice
reveal_type(np.s_[0:1, None:3])  # E: Tuple[builtins.slice, builtins.slice]
reveal_type(np.s_[0, 0:1, ..., [0, 1, 3]])  # E: Tuple[Literal[0]?, builtins.slice, builtins.ellipsis, builtins.list[builtins.int]]

reveal_type(np.ix_(AR_LIKE_b))  # E: tuple[ndarray[Any, dtype[bool_]], ...]
reveal_type(np.ix_(AR_LIKE_i, AR_LIKE_f))  # E: tuple[ndarray[Any, dtype[{double}]], ...]
reveal_type(np.ix_(AR_i8))  # E: tuple[ndarray[Any, dtype[{int64}]], ...]

reveal_type(np.fill_diagonal(AR_i8, 5))  # E: None

reveal_type(np.diag_indices(4))  # E: tuple[ndarray[Any, dtype[{int_}]], ...]
reveal_type(np.diag_indices(2, 3))  # E: tuple[ndarray[Any, dtype[{int_}]], ...]

reveal_type(np.diag_indices_from(AR_i8))  # E: tuple[ndarray[Any, dtype[{int_}]], ...]
