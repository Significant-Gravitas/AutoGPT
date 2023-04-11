from collections.abc import Sequence
from typing import (
    Literal as L,
    Any,
    SupportsIndex,
)

from numpy._typing import (
    NDArray,
    ArrayLike,
)

_BinKind = L[
    "stone",
    "auto",
    "doane",
    "fd",
    "rice",
    "scott",
    "sqrt",
    "sturges",
]

__all__: list[str]

def histogram_bin_edges(
    a: ArrayLike,
    bins: _BinKind | SupportsIndex | ArrayLike = ...,
    range: None | tuple[float, float] = ...,
    weights: None | ArrayLike = ...,
) -> NDArray[Any]: ...

def histogram(
    a: ArrayLike,
    bins: _BinKind | SupportsIndex | ArrayLike = ...,
    range: None | tuple[float, float] = ...,
    density: bool = ...,
    weights: None | ArrayLike = ...,
) -> tuple[NDArray[Any], NDArray[Any]]: ...

def histogramdd(
    sample: ArrayLike,
    bins: SupportsIndex | ArrayLike = ...,
    range: Sequence[tuple[float, float]] = ...,
    density: None | bool = ...,
    weights: None | ArrayLike = ...,
) -> tuple[NDArray[Any], list[NDArray[Any]]]: ...
