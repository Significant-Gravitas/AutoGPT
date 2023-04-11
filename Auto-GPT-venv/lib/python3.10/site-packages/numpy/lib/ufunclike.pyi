from typing import Any, overload, TypeVar

from numpy import floating, bool_, object_, ndarray
from numpy._typing import (
    NDArray,
    _FloatLike_co,
    _ArrayLikeFloat_co,
    _ArrayLikeObject_co,
)

_ArrayType = TypeVar("_ArrayType", bound=ndarray[Any, Any])

__all__: list[str]

@overload
def fix(  # type: ignore[misc]
    x: _FloatLike_co,
    out: None = ...,
) -> floating[Any]: ...
@overload
def fix(
    x: _ArrayLikeFloat_co,
    out: None = ...,
) -> NDArray[floating[Any]]: ...
@overload
def fix(
    x: _ArrayLikeObject_co,
    out: None = ...,
) -> NDArray[object_]: ...
@overload
def fix(
    x: _ArrayLikeFloat_co | _ArrayLikeObject_co,
    out: _ArrayType,
) -> _ArrayType: ...

@overload
def isposinf(  # type: ignore[misc]
    x: _FloatLike_co,
    out: None = ...,
) -> bool_: ...
@overload
def isposinf(
    x: _ArrayLikeFloat_co,
    out: None = ...,
) -> NDArray[bool_]: ...
@overload
def isposinf(
    x: _ArrayLikeFloat_co,
    out: _ArrayType,
) -> _ArrayType: ...

@overload
def isneginf(  # type: ignore[misc]
    x: _FloatLike_co,
    out: None = ...,
) -> bool_: ...
@overload
def isneginf(
    x: _ArrayLikeFloat_co,
    out: None = ...,
) -> NDArray[bool_]: ...
@overload
def isneginf(
    x: _ArrayLikeFloat_co,
    out: _ArrayType,
) -> _ArrayType: ...
