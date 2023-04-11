from typing import Any
import numpy as np

AR_i8: np.ndarray[Any, np.dtype[np.int64]]
ar_iter = np.lib.Arrayterator(AR_i8)

reveal_type(ar_iter.var)  # E: ndarray[Any, dtype[{int64}]]
reveal_type(ar_iter.buf_size)  # E: Union[None, builtins.int]
reveal_type(ar_iter.start)  # E: builtins.list[builtins.int]
reveal_type(ar_iter.stop)  # E: builtins.list[builtins.int]
reveal_type(ar_iter.step)  # E: builtins.list[builtins.int]
reveal_type(ar_iter.shape)  # E: builtins.tuple[builtins.int, ...]
reveal_type(ar_iter.flat)  # E: typing.Generator[{int64}, None, None]

reveal_type(ar_iter.__array__())  # E: ndarray[Any, dtype[{int64}]]

for i in ar_iter:
    reveal_type(i)  # E: ndarray[Any, dtype[{int64}]]

reveal_type(ar_iter[0])  # E: lib.arrayterator.Arrayterator[Any, dtype[{int64}]]
reveal_type(ar_iter[...])  # E: lib.arrayterator.Arrayterator[Any, dtype[{int64}]]
reveal_type(ar_iter[:])  # E: lib.arrayterator.Arrayterator[Any, dtype[{int64}]]
reveal_type(ar_iter[0, 0, 0])  # E: lib.arrayterator.Arrayterator[Any, dtype[{int64}]]
reveal_type(ar_iter[..., 0, :])  # E: lib.arrayterator.Arrayterator[Any, dtype[{int64}]]
