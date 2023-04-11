import numpy as np

class DTypeLike:
    dtype: np.dtype[np.int_]

dtype_like: DTypeLike

np.expand_dims(dtype_like, (5, 10))  # E: No overload variant
