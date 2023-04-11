import numpy as np
import numpy.typing as npt

AR_i8: npt.NDArray[np.int64]

np.pad(AR_i8, 2, mode="bob")  # E: No overload variant
