from __future__ import annotations

from typing import TypeVar
import numpy as np
import numpy.typing as npt

T1 = TypeVar("T1", bound=npt.NBitBase)
T2 = TypeVar("T2", bound=npt.NBitBase)

def add(a: np.floating[T1], b: np.integer[T2]) -> np.floating[T1 | T2]:
    return a + b

i8: np.int64
i4: np.int32
f8: np.float64
f4: np.float32

reveal_type(add(f8, i8))  # E: {float64}
reveal_type(add(f4, i8))  # E: {float64}
reveal_type(add(f8, i4))  # E: {float64}
reveal_type(add(f4, i4))  # E: {float32}
