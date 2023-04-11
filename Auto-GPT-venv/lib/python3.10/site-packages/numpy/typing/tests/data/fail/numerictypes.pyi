import numpy as np

# Technically this works, but probably shouldn't. See
#
# https://github.com/numpy/numpy/issues/16366
#
np.maximum_sctype(1)  # E: No overload variant

np.issubsctype(1, np.int64)  # E: incompatible type

np.issubdtype(1, np.int64)  # E: incompatible type

np.find_common_type(np.int64, np.int64)  # E: incompatible type
