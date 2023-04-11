import numpy as np

np.AxisError(1.0)  # E: No overload variant
np.AxisError(1, ndim=2.0)  # E: No overload variant
np.AxisError(2, msg_prefix=404)  # E: No overload variant
