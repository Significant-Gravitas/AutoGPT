from collections.abc import Generator
from typing import (
    Any,
    TypeVar,
    Union,
    overload,
)

from numpy import ndarray, dtype, generic
from numpy._typing import DTypeLike

# TODO: Set a shape bound once we've got proper shape support
_Shape = TypeVar("_Shape", bound=Any)
_DType = TypeVar("_DType", bound=dtype[Any])
_ScalarType = TypeVar("_ScalarType", bound=generic)

_Index = Union[
    Union[ellipsis, int, slice],
    tuple[Union[ellipsis, int, slice], ...],
]

__all__: list[str]

# NOTE: In reality `Arrayterator` does not actually inherit from `ndarray`,
# but its ``__getattr__` method does wrap around the former and thus has
# access to all its methods

class Arrayterator(ndarray[_Shape, _DType]):
    var: ndarray[_Shape, _DType]  # type: ignore[assignment]
    buf_size: None | int
    start: list[int]
    stop: list[int]
    step: list[int]

    @property  # type: ignore[misc]
    def shape(self) -> tuple[int, ...]: ...
    @property
    def flat(  # type: ignore[override]
        self: ndarray[Any, dtype[_ScalarType]]
    ) -> Generator[_ScalarType, None, None]: ...
    def __init__(
        self, var: ndarray[_Shape, _DType], buf_size: None | int = ...
    ) -> None: ...
    @overload
    def __array__(self, dtype: None = ...) -> ndarray[Any, _DType]: ...
    @overload
    def __array__(self, dtype: DTypeLike) -> ndarray[Any, dtype[Any]]: ...
    def __getitem__(self, index: _Index) -> Arrayterator[Any, _DType]: ...
    def __iter__(self) -> Generator[ndarray[Any, _DType], None, None]: ...
