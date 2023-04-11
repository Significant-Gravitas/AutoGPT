from __future__ import annotations

from io import StringIO

import numpy as np

FILE = StringIO()
AR = np.arange(10, dtype=np.float64)

def func(a: int) -> bool: ...

np.deprecate(func)
np.deprecate()

np.deprecate_with_doc("test")
np.deprecate_with_doc(None)

np.byte_bounds(AR)
np.byte_bounds(np.float64())

np.info(1, output=FILE)

np.source(np.interp, output=FILE)

np.lookfor("binary representation", output=FILE)
