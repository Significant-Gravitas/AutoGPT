import numpy as np


class Test1:
    not_dtype = np.dtype(float)


class Test2:
    dtype = float


np.dtype(Test1())  # E: No overload variant of "dtype" matches
np.dtype(Test2())  # E: incompatible type

np.dtype(  # E: No overload variant of "dtype" matches
    {
        "field1": (float, 1),
        "field2": (int, 3),
    }
)
