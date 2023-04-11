
from __future__ import annotations

from typing import Any
import numpy as np

AR_i8: np.ndarray[Any, np.dtype[np.int_]] = np.arange(10)
ar_iter = np.lib.Arrayterator(AR_i8)

ar_iter.var
ar_iter.buf_size
ar_iter.start
ar_iter.stop
ar_iter.step
ar_iter.shape
ar_iter.flat

ar_iter.__array__()

for i in ar_iter:
    pass

ar_iter[0]
ar_iter[...]
ar_iter[:]
ar_iter[0, 0, 0]
ar_iter[..., 0, :]
