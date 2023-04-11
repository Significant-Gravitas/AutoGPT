"""Typing tests for `numpy.core._ufunc_config`."""

import numpy as np

def func1(a: str, b: int, c: float) -> None: ...
def func2(a: str, *, b: int) -> None: ...

class Write1:
    def write1(self, a: str) -> None: ...

class Write2:
    def write(self, a: str, b: str) -> None: ...

class Write3:
    def write(self, *, a: str) -> None: ...

np.seterrcall(func1)  # E: Argument 1 to "seterrcall" has incompatible type
np.seterrcall(func2)  # E: Argument 1 to "seterrcall" has incompatible type
np.seterrcall(Write1())  # E: Argument 1 to "seterrcall" has incompatible type
np.seterrcall(Write2())  # E: Argument 1 to "seterrcall" has incompatible type
np.seterrcall(Write3())  # E: Argument 1 to "seterrcall" has incompatible type
