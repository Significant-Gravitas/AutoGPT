from typing import (
    Any,
    AnyStr,
    Callable,
    Container,
    ContextManager,
    Iterable,
    List,
    Mapping,
    Match,
    Optional,
    Pattern,
    Tuple,
    Type,
    TypeVar,
    Union,
    overload,
)

from . import _ValidatorType
from . import _ValidatorArgType

_T = TypeVar("_T")
_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_T3 = TypeVar("_T3")
_I = TypeVar("_I", bound=Iterable)
_K = TypeVar("_K")
_V = TypeVar("_V")
_M = TypeVar("_M", bound=Mapping)

def set_disabled(run: bool) -> None: ...
def get_disabled() -> bool: ...
def disabled() -> ContextManager[None]: ...

# To be more precise on instance_of use some overloads.
# If there are more than 3 items in the tuple then we fall back to Any
@overload
def instance_of(type: Type[_T]) -> _ValidatorType[_T]: ...
@overload
def instance_of(type: Tuple[Type[_T]]) -> _ValidatorType[_T]: ...
@overload
def instance_of(
    type: Tuple[Type[_T1], Type[_T2]]
) -> _ValidatorType[Union[_T1, _T2]]: ...
@overload
def instance_of(
    type: Tuple[Type[_T1], Type[_T2], Type[_T3]]
) -> _ValidatorType[Union[_T1, _T2, _T3]]: ...
@overload
def instance_of(type: Tuple[type, ...]) -> _ValidatorType[Any]: ...
def provides(interface: Any) -> _ValidatorType[Any]: ...
def optional(
    validator: Union[_ValidatorType[_T], List[_ValidatorType[_T]]]
) -> _ValidatorType[Optional[_T]]: ...
def in_(options: Container[_T]) -> _ValidatorType[_T]: ...
def and_(*validators: _ValidatorType[_T]) -> _ValidatorType[_T]: ...
def matches_re(
    regex: Union[Pattern[AnyStr], AnyStr],
    flags: int = ...,
    func: Optional[
        Callable[[AnyStr, AnyStr, int], Optional[Match[AnyStr]]]
    ] = ...,
) -> _ValidatorType[AnyStr]: ...
def deep_iterable(
    member_validator: _ValidatorArgType[_T],
    iterable_validator: Optional[_ValidatorType[_I]] = ...,
) -> _ValidatorType[_I]: ...
def deep_mapping(
    key_validator: _ValidatorType[_K],
    value_validator: _ValidatorType[_V],
    mapping_validator: Optional[_ValidatorType[_M]] = ...,
) -> _ValidatorType[_M]: ...
def is_callable() -> _ValidatorType[_T]: ...
def lt(val: _T) -> _ValidatorType[_T]: ...
def le(val: _T) -> _ValidatorType[_T]: ...
def ge(val: _T) -> _ValidatorType[_T]: ...
def gt(val: _T) -> _ValidatorType[_T]: ...
def max_len(length: int) -> _ValidatorType[_T]: ...
def min_len(length: int) -> _ValidatorType[_T]: ...
def not_(
    validator: _ValidatorType[_T],
    *,
    msg: Optional[str] = None,
    exc_types: Union[Type[Exception], Iterable[Type[Exception]]] = ...
) -> _ValidatorType[_T]: ...
