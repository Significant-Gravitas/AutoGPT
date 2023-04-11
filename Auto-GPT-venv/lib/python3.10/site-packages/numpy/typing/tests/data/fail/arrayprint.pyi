from collections.abc import Callable
from typing import Any
import numpy as np

AR: np.ndarray
func1: Callable[[Any], str]
func2: Callable[[np.integer[Any]], str]

np.array2string(AR, style=None)  # E: Unexpected keyword argument
np.array2string(AR, legacy="1.14")  # E: incompatible type
np.array2string(AR, sign="*")  # E: incompatible type
np.array2string(AR, floatmode="default")  # E: incompatible type
np.array2string(AR, formatter={"A": func1})  # E: incompatible type
np.array2string(AR, formatter={"float": func2})  # E: Incompatible types
