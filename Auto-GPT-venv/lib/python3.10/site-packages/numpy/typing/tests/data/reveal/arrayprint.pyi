from collections.abc import Callable
from typing import Any
import numpy as np

AR: np.ndarray[Any, Any]
func_float: Callable[[np.floating[Any]], str]
func_int: Callable[[np.integer[Any]], str]

reveal_type(np.get_printoptions())  # E: TypedDict
reveal_type(np.array2string(  # E: str
    AR, formatter={'float_kind': func_float, 'int_kind': func_int}
))
reveal_type(np.format_float_scientific(1.0))  # E: str
reveal_type(np.format_float_positional(1))  # E: str
reveal_type(np.array_repr(AR))  # E: str
reveal_type(np.array_str(AR))  # E: str

reveal_type(np.printoptions())  # E: contextlib._GeneratorContextManager
with np.printoptions() as dct:
    reveal_type(dct)  # E: TypedDict
