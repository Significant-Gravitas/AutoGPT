import numpy as np

np.AxisError("test")
np.AxisError(1, ndim=2)
np.AxisError(1, ndim=2, msg_prefix="error")
np.AxisError(1, ndim=2, msg_prefix=None)
