from typing import Any
import numpy as np

AR_i: np.ndarray[Any, np.dtype[np.int64]]
AR_f: np.ndarray[Any, np.dtype[np.float64]]
AR_m: np.ndarray[Any, np.dtype[np.timedelta64]]
AR_O: np.ndarray[Any, np.dtype[np.object_]]
AR_U: np.ndarray[Any, np.dtype[np.str_]]

np.einsum("i,i->i", AR_i, AR_m)  # E: incompatible type
np.einsum("i,i->i", AR_O, AR_O)  # E: incompatible type
np.einsum("i,i->i", AR_f, AR_f, dtype=np.int32)  # E: incompatible type
np.einsum("i,i->i", AR_i, AR_i, dtype=np.timedelta64, casting="unsafe")  # E: No overload variant
np.einsum("i,i->i", AR_i, AR_i, out=AR_U)  # E: Value of type variable "_ArrayType" of "einsum" cannot be
np.einsum("i,i->i", AR_i, AR_i, out=AR_U, casting="unsafe")  # E: No overload variant
