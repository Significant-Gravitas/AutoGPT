import numpy as np
import numpy.typing as npt

AR_i8: npt.NDArray[np.int64]
AR_f8: npt.NDArray[np.float64]

np.histogram_bin_edges(AR_i8, range=(0, 1, 2))  # E: incompatible type

np.histogram(AR_i8, range=(0, 1, 2))  # E: incompatible type

np.histogramdd(AR_i8, range=(0, 1))  # E: incompatible type
np.histogramdd(AR_i8, range=[(0, 1, 2)])  # E: incompatible type
