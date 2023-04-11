from collections.abc import Sequence, Mapping
from typing import Any
from numpy import matrix as matrix
from numpy._typing import ArrayLike, DTypeLike, NDArray

__all__: list[str]

def bmat(
    obj: str | Sequence[ArrayLike] | NDArray[Any],
    ldict: None | Mapping[str, Any] = ...,
    gdict: None | Mapping[str, Any] = ...,
) -> matrix[Any, Any]: ...

def asmatrix(data: ArrayLike, dtype: DTypeLike = ...) -> matrix[Any, Any]: ...

mat = asmatrix
