from typing import Any
import numpy as np

AR_c: np.ndarray[Any, np.dtype[np.complex128]]
AR_m: np.ndarray[Any, np.dtype[np.timedelta64]]
AR_M: np.ndarray[Any, np.dtype[np.datetime64]]
AR_O: np.ndarray[Any, np.dtype[np.object_]]

np.fix(AR_c)  # E: incompatible type
np.fix(AR_m)  # E: incompatible type
np.fix(AR_M)  # E: incompatible type

np.isposinf(AR_c)  # E: incompatible type
np.isposinf(AR_m)  # E: incompatible type
np.isposinf(AR_M)  # E: incompatible type
np.isposinf(AR_O)  # E: incompatible type

np.isneginf(AR_c)  # E: incompatible type
np.isneginf(AR_m)  # E: incompatible type
np.isneginf(AR_M)  # E: incompatible type
np.isneginf(AR_O)  # E: incompatible type
