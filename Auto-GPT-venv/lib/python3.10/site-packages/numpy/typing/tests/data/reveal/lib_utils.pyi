from io import StringIO
from typing import Any

import numpy as np

AR: np.ndarray[Any, np.dtype[np.float64]]
AR_DICT: dict[str, np.ndarray[Any, np.dtype[np.float64]]]
FILE: StringIO

def func(a: int) -> bool: ...

reveal_type(np.deprecate(func))  # E: def (a: builtins.int) -> builtins.bool
reveal_type(np.deprecate())  # E: _Deprecate

reveal_type(np.deprecate_with_doc("test"))  # E: _Deprecate
reveal_type(np.deprecate_with_doc(None))  # E: _Deprecate

reveal_type(np.byte_bounds(AR))  # E: Tuple[builtins.int, builtins.int]
reveal_type(np.byte_bounds(np.float64()))  # E: Tuple[builtins.int, builtins.int]

reveal_type(np.who(None))  # E: None
reveal_type(np.who(AR_DICT))  # E: None

reveal_type(np.info(1, output=FILE))  # E: None

reveal_type(np.source(np.interp, output=FILE))  # E: None

reveal_type(np.lookfor("binary representation", output=FILE))  # E: None

reveal_type(np.safe_eval("1 + 1"))  # E: Any
