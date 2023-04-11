import numpy as np
from typing import Any

memmap_obj: np.memmap[Any, np.dtype[np.str_]]

reveal_type(np.memmap.__array_priority__)  # E: float
reveal_type(memmap_obj.__array_priority__)  # E: float
reveal_type(memmap_obj.filename)  # E: Union[builtins.str, None]
reveal_type(memmap_obj.offset)  # E: int
reveal_type(memmap_obj.mode)  # E: str
reveal_type(memmap_obj.flush())  # E: None

reveal_type(np.memmap("file.txt", offset=5))  # E: memmap[Any, dtype[{uint8}]]
reveal_type(np.memmap(b"file.txt", dtype=np.float64, shape=(10, 3)))  # E: memmap[Any, dtype[{float64}]]
with open("file.txt", "rb") as f:
    reveal_type(np.memmap(f, dtype=float, order="K"))  # E: memmap[Any, dtype[Any]]

reveal_type(memmap_obj.__array_finalize__(object()))  # E: None
