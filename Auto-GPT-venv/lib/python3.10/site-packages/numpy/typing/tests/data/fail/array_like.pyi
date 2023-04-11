import numpy as np
from numpy._typing import ArrayLike


class A:
    pass


x1: ArrayLike = (i for i in range(10))  # E: Incompatible types in assignment
x2: ArrayLike = A()  # E: Incompatible types in assignment
x3: ArrayLike = {1: "foo", 2: "bar"}  # E: Incompatible types in assignment

scalar = np.int64(1)
scalar.__array__(dtype=np.float64)  # E: No overload variant
array = np.array([1])
array.__array__(dtype=np.float64)  # E: No overload variant
