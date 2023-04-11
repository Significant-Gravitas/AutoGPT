from typing import Any

import numpy as np
from numpy._typing import _SupportsArray


class Index:
    def __index__(self) -> int:
        ...


a: "np.flatiter[np.ndarray]"
supports_array: _SupportsArray

a.base = Any  # E: Property "base" defined in "flatiter" is read-only
a.coords = Any  # E: Property "coords" defined in "flatiter" is read-only
a.index = Any  # E: Property "index" defined in "flatiter" is read-only
a.copy(order='C')  # E: Unexpected keyword argument

# NOTE: Contrary to `ndarray.__getitem__` its counterpart in `flatiter`
# does not accept objects with the `__array__` or `__index__` protocols;
# boolean indexing is just plain broken (gh-17175)
a[np.bool_()]  # E: No overload variant of "__getitem__"
a[Index()]  # E: No overload variant of "__getitem__"
a[supports_array]  # E: No overload variant of "__getitem__"
