from typing import Any, TypeVar

import numpy as np
import numpy.typing as npt


def func1(ar: npt.NDArray[Any], a: int) -> npt.NDArray[np.str_]:
    pass


def func2(ar: npt.NDArray[Any], a: float) -> float:
    pass


AR_b: npt.NDArray[np.bool_]
AR_m: npt.NDArray[np.timedelta64]

AR_LIKE_b: list[bool]

np.eye(10, M=20.0)  # E: No overload variant
np.eye(10, k=2.5, dtype=int)  # E: No overload variant

np.diag(AR_b, k=0.5)  # E: No overload variant
np.diagflat(AR_b, k=0.5)  # E: No overload variant

np.tri(10, M=20.0)  # E: No overload variant
np.tri(10, k=2.5, dtype=int)  # E: No overload variant

np.tril(AR_b, k=0.5)  # E: No overload variant
np.triu(AR_b, k=0.5)  # E: No overload variant

np.vander(AR_m)  # E: incompatible type

np.histogram2d(AR_m)  # E: No overload variant

np.mask_indices(10, func1)  # E: incompatible type
np.mask_indices(10, func2, 10.5)  # E: incompatible type
