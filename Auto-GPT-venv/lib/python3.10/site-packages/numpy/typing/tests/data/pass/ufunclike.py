from __future__ import annotations
from typing import Any
import numpy as np


class Object:
    def __ceil__(self) -> Object:
        return self

    def __floor__(self) -> Object:
        return self

    def __ge__(self, value: object) -> bool:
        return True

    def __array__(self) -> np.ndarray[Any, np.dtype[np.object_]]:
        ret = np.empty((), dtype=object)
        ret[()] = self
        return ret


AR_LIKE_b = [True, True, False]
AR_LIKE_u = [np.uint32(1), np.uint32(2), np.uint32(3)]
AR_LIKE_i = [1, 2, 3]
AR_LIKE_f = [1.0, 2.0, 3.0]
AR_LIKE_O = [Object(), Object(), Object()]
AR_U: np.ndarray[Any, np.dtype[np.str_]] = np.zeros(3, dtype="U5")

np.fix(AR_LIKE_b)
np.fix(AR_LIKE_u)
np.fix(AR_LIKE_i)
np.fix(AR_LIKE_f)
np.fix(AR_LIKE_O)
np.fix(AR_LIKE_f, out=AR_U)

np.isposinf(AR_LIKE_b)
np.isposinf(AR_LIKE_u)
np.isposinf(AR_LIKE_i)
np.isposinf(AR_LIKE_f)
np.isposinf(AR_LIKE_f, out=AR_U)

np.isneginf(AR_LIKE_b)
np.isneginf(AR_LIKE_u)
np.isneginf(AR_LIKE_i)
np.isneginf(AR_LIKE_f)
np.isneginf(AR_LIKE_f, out=AR_U)
