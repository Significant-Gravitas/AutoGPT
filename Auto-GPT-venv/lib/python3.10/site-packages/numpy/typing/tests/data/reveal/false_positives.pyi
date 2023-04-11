from typing import Any
import numpy.typing as npt

AR_Any: npt.NDArray[Any]

# Mypy bug where overload ambiguity is ignored for `Any`-parametrized types;
# xref numpy/numpy#20099 and python/mypy#11347
#
# The expected output would be something akin to `ndarray[Any, dtype[Any]]`
reveal_type(AR_Any + 2)  # E: ndarray[Any, dtype[signedinteger[Any]]]
