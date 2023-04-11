from collections.abc import Callable, Sequence
from typing import TypeVar, Any, overload, SupportsIndex, Protocol

from numpy import (
    generic,
    integer,
    ufunc,
    bool_,
    unsignedinteger,
    signedinteger,
    floating,
    complexfloating,
    object_,
)

from numpy._typing import (
    ArrayLike,
    NDArray,
    _ShapeLike,
    _ArrayLike,
    _ArrayLikeBool_co,
    _ArrayLikeUInt_co,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeObject_co,
)

from numpy.core.shape_base import vstack

_SCT = TypeVar("_SCT", bound=generic)

# The signatures of `__array_wrap__` and `__array_prepare__` are the same;
# give them unique names for the sake of clarity
class _ArrayWrap(Protocol):
    def __call__(
        self,
        array: NDArray[Any],
        context: None | tuple[ufunc, tuple[Any, ...], int] = ...,
        /,
    ) -> Any: ...

class _ArrayPrepare(Protocol):
    def __call__(
        self,
        array: NDArray[Any],
        context: None | tuple[ufunc, tuple[Any, ...], int] = ...,
        /,
    ) -> Any: ...

class _SupportsArrayWrap(Protocol):
    @property
    def __array_wrap__(self) -> _ArrayWrap: ...

class _SupportsArrayPrepare(Protocol):
    @property
    def __array_prepare__(self) -> _ArrayPrepare: ...

__all__: list[str]

row_stack = vstack

def take_along_axis(
    arr: _SCT | NDArray[_SCT],
    indices: NDArray[integer[Any]],
    axis: None | int,
) -> NDArray[_SCT]: ...

def put_along_axis(
    arr: NDArray[_SCT],
    indices: NDArray[integer[Any]],
    values: ArrayLike,
    axis: None | int,
) -> None: ...

# TODO: Use PEP 612 `ParamSpec` once mypy supports `Concatenate`
# xref python/mypy#8645
@overload
def apply_along_axis(
    func1d: Callable[..., _ArrayLike[_SCT]],
    axis: SupportsIndex,
    arr: ArrayLike,
    *args: Any,
    **kwargs: Any,
) -> NDArray[_SCT]: ...
@overload
def apply_along_axis(
    func1d: Callable[..., ArrayLike],
    axis: SupportsIndex,
    arr: ArrayLike,
    *args: Any,
    **kwargs: Any,
) -> NDArray[Any]: ...

def apply_over_axes(
    func: Callable[[NDArray[Any], int], NDArray[_SCT]],
    a: ArrayLike,
    axes: int | Sequence[int],
) -> NDArray[_SCT]: ...

@overload
def expand_dims(
    a: _ArrayLike[_SCT],
    axis: _ShapeLike,
) -> NDArray[_SCT]: ...
@overload
def expand_dims(
    a: ArrayLike,
    axis: _ShapeLike,
) -> NDArray[Any]: ...

@overload
def column_stack(tup: Sequence[_ArrayLike[_SCT]]) -> NDArray[_SCT]: ...
@overload
def column_stack(tup: Sequence[ArrayLike]) -> NDArray[Any]: ...

@overload
def dstack(tup: Sequence[_ArrayLike[_SCT]]) -> NDArray[_SCT]: ...
@overload
def dstack(tup: Sequence[ArrayLike]) -> NDArray[Any]: ...

@overload
def array_split(
    ary: _ArrayLike[_SCT],
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = ...,
) -> list[NDArray[_SCT]]: ...
@overload
def array_split(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = ...,
) -> list[NDArray[Any]]: ...

@overload
def split(
    ary: _ArrayLike[_SCT],
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = ...,
) -> list[NDArray[_SCT]]: ...
@overload
def split(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
    axis: SupportsIndex = ...,
) -> list[NDArray[Any]]: ...

@overload
def hsplit(
    ary: _ArrayLike[_SCT],
    indices_or_sections: _ShapeLike,
) -> list[NDArray[_SCT]]: ...
@overload
def hsplit(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
) -> list[NDArray[Any]]: ...

@overload
def vsplit(
    ary: _ArrayLike[_SCT],
    indices_or_sections: _ShapeLike,
) -> list[NDArray[_SCT]]: ...
@overload
def vsplit(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
) -> list[NDArray[Any]]: ...

@overload
def dsplit(
    ary: _ArrayLike[_SCT],
    indices_or_sections: _ShapeLike,
) -> list[NDArray[_SCT]]: ...
@overload
def dsplit(
    ary: ArrayLike,
    indices_or_sections: _ShapeLike,
) -> list[NDArray[Any]]: ...

@overload
def get_array_prepare(*args: _SupportsArrayPrepare) -> _ArrayPrepare: ...
@overload
def get_array_prepare(*args: object) -> None | _ArrayPrepare: ...

@overload
def get_array_wrap(*args: _SupportsArrayWrap) -> _ArrayWrap: ...
@overload
def get_array_wrap(*args: object) -> None | _ArrayWrap: ...

@overload
def kron(a: _ArrayLikeBool_co, b: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
@overload
def kron(a: _ArrayLikeUInt_co, b: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
@overload
def kron(a: _ArrayLikeInt_co, b: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
@overload
def kron(a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
@overload
def kron(a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def kron(a: _ArrayLikeObject_co, b: Any) -> NDArray[object_]: ...
@overload
def kron(a: Any, b: _ArrayLikeObject_co) -> NDArray[object_]: ...

@overload
def tile(
    A: _ArrayLike[_SCT],
    reps: int | Sequence[int],
) -> NDArray[_SCT]: ...
@overload
def tile(
    A: ArrayLike,
    reps: int | Sequence[int],
) -> NDArray[Any]: ...
