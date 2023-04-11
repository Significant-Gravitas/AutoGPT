from typing import (
    Literal as L,
    Any,
    overload,
    TypeVar,
    Protocol,
)

from numpy import generic

from numpy._typing import (
    ArrayLike,
    NDArray,
    _ArrayLikeInt,
    _ArrayLike,
)

_SCT = TypeVar("_SCT", bound=generic)

class _ModeFunc(Protocol):
    def __call__(
        self,
        vector: NDArray[Any],
        iaxis_pad_width: tuple[int, int],
        iaxis: int,
        kwargs: dict[str, Any],
        /,
    ) -> None: ...

_ModeKind = L[
    "constant",
    "edge",
    "linear_ramp",
    "maximum",
    "mean",
    "median",
    "minimum",
    "reflect",
    "symmetric",
    "wrap",
    "empty",
]

__all__: list[str]

# TODO: In practice each keyword argument is exclusive to one or more
# specific modes. Consider adding more overloads to express this in the future.

# Expand `**kwargs` into explicit keyword-only arguments
@overload
def pad(
    array: _ArrayLike[_SCT],
    pad_width: _ArrayLikeInt,
    mode: _ModeKind = ...,
    *,
    stat_length: None | _ArrayLikeInt = ...,
    constant_values: ArrayLike = ...,
    end_values: ArrayLike = ...,
    reflect_type: L["odd", "even"] = ...,
) -> NDArray[_SCT]: ...
@overload
def pad(
    array: ArrayLike,
    pad_width: _ArrayLikeInt,
    mode: _ModeKind = ...,
    *,
    stat_length: None | _ArrayLikeInt = ...,
    constant_values: ArrayLike = ...,
    end_values: ArrayLike = ...,
    reflect_type: L["odd", "even"] = ...,
) -> NDArray[Any]: ...
@overload
def pad(
    array: _ArrayLike[_SCT],
    pad_width: _ArrayLikeInt,
    mode: _ModeFunc,
    **kwargs: Any,
) -> NDArray[_SCT]: ...
@overload
def pad(
    array: ArrayLike,
    pad_width: _ArrayLikeInt,
    mode: _ModeFunc,
    **kwargs: Any,
) -> NDArray[Any]: ...
