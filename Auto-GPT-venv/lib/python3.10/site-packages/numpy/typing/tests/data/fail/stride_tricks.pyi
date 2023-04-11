import numpy as np
import numpy.typing as npt

AR_f8: npt.NDArray[np.float64]

np.lib.stride_tricks.as_strided(AR_f8, shape=8)  # E: No overload variant
np.lib.stride_tricks.as_strided(AR_f8, strides=8)  # E: No overload variant

np.lib.stride_tricks.sliding_window_view(AR_f8, axis=(1,))  # E: No overload variant
