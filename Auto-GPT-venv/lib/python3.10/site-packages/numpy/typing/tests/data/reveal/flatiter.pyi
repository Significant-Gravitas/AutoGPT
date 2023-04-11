from typing import Any
import numpy as np

a: np.flatiter[np.ndarray[Any, np.dtype[np.str_]]]

reveal_type(a.base)  # E: ndarray[Any, dtype[str_]]
reveal_type(a.copy())  # E: ndarray[Any, dtype[str_]]
reveal_type(a.coords)  # E: tuple[builtins.int, ...]
reveal_type(a.index)  # E: int
reveal_type(iter(a))  # E: Any
reveal_type(next(a))  # E: str_
reveal_type(a[0])  # E: str_
reveal_type(a[[0, 1, 2]])  # E: ndarray[Any, dtype[str_]]
reveal_type(a[...])  # E: ndarray[Any, dtype[str_]]
reveal_type(a[:])  # E: ndarray[Any, dtype[str_]]
reveal_type(a[(...,)])  # E: ndarray[Any, dtype[str_]]
reveal_type(a[(0,)])  # E: str_
reveal_type(a.__array__())  # E: ndarray[Any, dtype[str_]]
reveal_type(a.__array__(np.dtype(np.float64)))  # E: ndarray[Any, dtype[{float64}]]
a[0] = "a"
a[:5] = "a"
a[...] = "a"
a[(...,)] = "a"
