import numpy as np
import numpy.typing as npt

AR_b: npt.NDArray[np.bool_]
AR_i8: npt.NDArray[np.int64]
AR_f8: npt.NDArray[np.float64]
AR_M: npt.NDArray[np.datetime64]
AR_O: npt.NDArray[np.object_]

AR_LIKE_f8: list[float]

reveal_type(np.ediff1d(AR_b))  # E: ndarray[Any, dtype[{int8}]]
reveal_type(np.ediff1d(AR_i8, to_end=[1, 2, 3]))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.ediff1d(AR_M))  # E: ndarray[Any, dtype[timedelta64]]
reveal_type(np.ediff1d(AR_O))  # E: ndarray[Any, dtype[object_]]
reveal_type(np.ediff1d(AR_LIKE_f8, to_begin=[1, 1.5]))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.intersect1d(AR_i8, AR_i8))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.intersect1d(AR_M, AR_M, assume_unique=True))  # E: ndarray[Any, dtype[datetime64]]
reveal_type(np.intersect1d(AR_f8, AR_i8))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.intersect1d(AR_f8, AR_f8, return_indices=True))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{intp}]], ndarray[Any, dtype[{intp}]]]

reveal_type(np.setxor1d(AR_i8, AR_i8))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.setxor1d(AR_M, AR_M, assume_unique=True))  # E: ndarray[Any, dtype[datetime64]]
reveal_type(np.setxor1d(AR_f8, AR_i8))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.in1d(AR_i8, AR_i8))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.in1d(AR_M, AR_M, assume_unique=True))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.in1d(AR_f8, AR_i8))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.in1d(AR_f8, AR_LIKE_f8, invert=True))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.isin(AR_i8, AR_i8))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.isin(AR_M, AR_M, assume_unique=True))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.isin(AR_f8, AR_i8))  # E: ndarray[Any, dtype[bool_]]
reveal_type(np.isin(AR_f8, AR_LIKE_f8, invert=True))  # E: ndarray[Any, dtype[bool_]]

reveal_type(np.union1d(AR_i8, AR_i8))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.union1d(AR_M, AR_M))  # E: ndarray[Any, dtype[datetime64]]
reveal_type(np.union1d(AR_f8, AR_i8))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.setdiff1d(AR_i8, AR_i8))  # E: ndarray[Any, dtype[{int64}]]
reveal_type(np.setdiff1d(AR_M, AR_M, assume_unique=True))  # E: ndarray[Any, dtype[datetime64]]
reveal_type(np.setdiff1d(AR_f8, AR_i8))  # E: ndarray[Any, dtype[Any]]

reveal_type(np.unique(AR_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.unique(AR_LIKE_f8, axis=0))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.unique(AR_f8, return_index=True))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{intp}]]]
reveal_type(np.unique(AR_LIKE_f8, return_index=True))  # E: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[{intp}]]]
reveal_type(np.unique(AR_f8, return_inverse=True))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{intp}]]]
reveal_type(np.unique(AR_LIKE_f8, return_inverse=True))  # E: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[{intp}]]]
reveal_type(np.unique(AR_f8, return_counts=True))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{intp}]]]
reveal_type(np.unique(AR_LIKE_f8, return_counts=True))  # E: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[{intp}]]]
reveal_type(np.unique(AR_f8, return_index=True, return_inverse=True))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{intp}]], ndarray[Any, dtype[{intp}]]]
reveal_type(np.unique(AR_LIKE_f8, return_index=True, return_inverse=True))  # E: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[{intp}]], ndarray[Any, dtype[{intp}]]]
reveal_type(np.unique(AR_f8, return_index=True, return_counts=True))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{intp}]], ndarray[Any, dtype[{intp}]]]
reveal_type(np.unique(AR_LIKE_f8, return_index=True, return_counts=True))  # E: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[{intp}]], ndarray[Any, dtype[{intp}]]]
reveal_type(np.unique(AR_f8, return_inverse=True, return_counts=True))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{intp}]], ndarray[Any, dtype[{intp}]]]
reveal_type(np.unique(AR_LIKE_f8, return_inverse=True, return_counts=True))  # E: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[{intp}]], ndarray[Any, dtype[{intp}]]]
reveal_type(np.unique(AR_f8, return_index=True, return_inverse=True, return_counts=True))  # E: Tuple[ndarray[Any, dtype[{float64}]], ndarray[Any, dtype[{intp}]], ndarray[Any, dtype[{intp}]], ndarray[Any, dtype[{intp}]]]
reveal_type(np.unique(AR_LIKE_f8, return_index=True, return_inverse=True, return_counts=True))  # E: Tuple[ndarray[Any, dtype[Any]], ndarray[Any, dtype[{intp}]], ndarray[Any, dtype[{intp}]], ndarray[Any, dtype[{intp}]]]
