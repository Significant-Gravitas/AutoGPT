import numpy as np

AR_LIKE_i: list[int]
AR_LIKE_f: list[float]

np.ndindex([1, 2, 3])  # E: No overload variant
np.unravel_index(AR_LIKE_f, (1, 2, 3))  # E: incompatible type
np.ravel_multi_index(AR_LIKE_i, (1, 2, 3), mode="bob")  # E: No overload variant
np.mgrid[1]  # E: Invalid index type
np.mgrid[...]  # E: Invalid index type
np.ogrid[1]  # E: Invalid index type
np.ogrid[...]  # E: Invalid index type
np.fill_diagonal(AR_LIKE_f, 2)  # E: incompatible type
np.diag_indices(1.0)  # E: incompatible type
