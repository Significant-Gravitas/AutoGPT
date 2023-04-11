from typing import Any
import numpy as np

AR_i8: np.ndarray[Any, np.dtype[np.int64]]
ar_iter = np.lib.Arrayterator(AR_i8)

np.lib.Arrayterator(np.int64())  # E: incompatible type
ar_iter.shape = (10, 5)  # E: is read-only
ar_iter[None]  # E: Invalid index type
ar_iter[None, 1]  # E: Invalid index type
ar_iter[np.intp()]  # E: Invalid index type
ar_iter[np.intp(), ...]  # E: Invalid index type
ar_iter[AR_i8]  # E: Invalid index type
ar_iter[AR_i8, :]  # E: Invalid index type
