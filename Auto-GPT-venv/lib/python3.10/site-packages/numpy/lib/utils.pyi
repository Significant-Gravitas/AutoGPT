from ast import AST
from collections.abc import Callable, Mapping, Sequence
from typing import (
    Any,
    overload,
    TypeVar,
    Protocol,
)

from numpy import ndarray, generic

from numpy.core.numerictypes import (
    issubclass_ as issubclass_,
    issubdtype as issubdtype,
    issubsctype as issubsctype,
)

_T_contra = TypeVar("_T_contra", contravariant=True)
_FuncType = TypeVar("_FuncType", bound=Callable[..., Any])

# A file-like object opened in `w` mode
class _SupportsWrite(Protocol[_T_contra]):
    def write(self, s: _T_contra, /) -> Any: ...

__all__: list[str]

class _Deprecate:
    old_name: None | str
    new_name: None | str
    message: None | str
    def __init__(
        self,
        old_name: None | str = ...,
        new_name: None | str = ...,
        message: None | str = ...,
    ) -> None: ...
    # NOTE: `__call__` can in principle take arbitrary `*args` and `**kwargs`,
    # even though they aren't used for anything
    def __call__(self, func: _FuncType) -> _FuncType: ...

def get_include() -> str: ...

@overload
def deprecate(
    *,
    old_name: None | str = ...,
    new_name: None | str = ...,
    message: None | str = ...,
) -> _Deprecate: ...
@overload
def deprecate(
    func: _FuncType,
    /,
    old_name: None | str = ...,
    new_name: None | str = ...,
    message: None | str = ...,
) -> _FuncType: ...

def deprecate_with_doc(msg: None | str) -> _Deprecate: ...

# NOTE: In practice `byte_bounds` can (potentially) take any object
# implementing the `__array_interface__` protocol. The caveat is
# that certain keys, marked as optional in the spec, must be present for
#  `byte_bounds`. This concerns `"strides"` and `"data"`.
def byte_bounds(a: generic | ndarray[Any, Any]) -> tuple[int, int]: ...

def who(vardict: None | Mapping[str, ndarray[Any, Any]] = ...) -> None: ...

def info(
    object: object = ...,
    maxwidth: int = ...,
    output: None | _SupportsWrite[str] = ...,
    toplevel: str = ...,
) -> None: ...

def source(
    object: object,
    output: None | _SupportsWrite[str] = ...,
) -> None: ...

def lookfor(
    what: str,
    module: None | str | Sequence[str] = ...,
    import_modules: bool = ...,
    regenerate: bool = ...,
    output: None | _SupportsWrite[str] =...,
) -> None: ...

def safe_eval(source: str | AST) -> Any: ...

def show_runtime() -> None: ...
