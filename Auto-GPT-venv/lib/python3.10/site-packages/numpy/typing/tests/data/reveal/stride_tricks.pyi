from typing import Any
import numpy as np
import numpy.typing as npt

AR_f8: npt.NDArray[np.float64]
AR_LIKE_f: list[float]
interface_dict: dict[str, Any]

reveal_type(np.lib.stride_tricks.DummyArray(interface_dict))  # E: lib.stride_tricks.DummyArray

reveal_type(np.lib.stride_tricks.as_strided(AR_f8))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.lib.stride_tricks.as_strided(AR_LIKE_f))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.lib.stride_tricks.as_strided(AR_f8, strides=(1, 5)))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.lib.stride_tricks.as_strided(AR_f8, shape=[9, 20]))  # E: ndarray[Any, dtype[{float64}]]

reveal_type(np.lib.stride_tricks.sliding_window_view(AR_f8, 5))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.lib.stride_tricks.sliding_window_view(AR_LIKE_f, (1, 5)))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.lib.stride_tricks.sliding_window_view(AR_f8, [9], axis=1))  # E: ndarray[Any, dtype[{float64}]]

reveal_type(np.broadcast_to(AR_f8, 5))  # E: ndarray[Any, dtype[{float64}]]
reveal_type(np.broadcast_to(AR_LIKE_f, (1, 5)))  # E: ndarray[Any, dtype[Any]]
reveal_type(np.broadcast_to(AR_f8, [4, 6], subok=True))  # E: ndarray[Any, dtype[{float64}]]

reveal_type(np.broadcast_shapes((1, 2), [3, 1], (3, 2)))  # E: tuple[builtins.int, ...]
reveal_type(np.broadcast_shapes((6, 7), (5, 6, 1), 7, (5, 1, 7)))  # E: tuple[builtins.int, ...]

reveal_type(np.broadcast_arrays(AR_f8, AR_f8))  # E: list[ndarray[Any, dtype[Any]]]
reveal_type(np.broadcast_arrays(AR_f8, AR_LIKE_f))  # E: list[ndarray[Any, dtype[Any]]]
