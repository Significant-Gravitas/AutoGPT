import numpy as np
import numpy.typing as npt

DTYPE_i8: np.dtype[np.int64]

np.mintypecode(DTYPE_i8)  # E: incompatible type
np.iscomplexobj(DTYPE_i8)  # E: incompatible type
np.isrealobj(DTYPE_i8)  # E: incompatible type

np.typename(DTYPE_i8)  # E: No overload variant
np.typename("invalid")  # E: No overload variant

np.common_type(np.timedelta64())  # E: incompatible type
