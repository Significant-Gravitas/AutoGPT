import numpy as np
import numpy.typing as npt

AR_f8: npt.NDArray[np.float64]

# NOTE: Mypy bug presumably due to the special-casing of heterogeneous tuples;
# xref numpy/numpy#20901
#
# The expected output should be no different than, e.g., when using a
# list instead of a tuple
np.concatenate(([1], AR_f8))  # E: Argument 1 to "concatenate" has incompatible type
