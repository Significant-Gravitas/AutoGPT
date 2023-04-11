from collections.abc import Container, Iterable
from typing import (
    Literal as L,
    Any,
    overload,
    TypeVar,
    Protocol,
)

from numpy import (
    dtype,
    generic,
    bool_,
    floating,
    float64,
    complexfloating,
    integer,
)

from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NBitBase,
    NDArray,
    _64Bit,
    _SupportsDType,
    _ScalarLike_co,
    _ArrayLike,
    _DTypeLikeComplex,
)

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_SCT = TypeVar("_SCT", bound=generic)
_NBit1 = TypeVar("_NBit1", bound=NBitBase)
_NBit2 = TypeVar("_NBit2", bound=NBitBase)

class _SupportsReal(Protocol[_T_co]):
    @property
    def real(self) -> _T_co: ...

class _SupportsImag(Protocol[_T_co]):
    @property
    def imag(self) -> _T_co: ...

__all__: list[str]

def mintypecode(
    typechars: Iterable[str | ArrayLike],
    typeset: Container[str] = ...,
    default: str = ...,
) -> str: ...

# `asfarray` ignores dtypes if they're not inexact

@overload
def asfarray(
    a: object,
    dtype: None | type[float] = ...,
) -> NDArray[float64]: ...
@overload
def asfarray(  # type: ignore[misc]
    a: Any,
    dtype: _DTypeLikeComplex,
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def asfarray(
    a: Any,
    dtype: DTypeLike,
) -> NDArray[floating[Any]]: ...

@overload
def real(val: _SupportsReal[_T]) -> _T: ...
@overload
def real(val: ArrayLike) -> NDArray[Any]: ...

@overload
def imag(val: _SupportsImag[_T]) -> _T: ...
@overload
def imag(val: ArrayLike) -> NDArray[Any]: ...

@overload
def iscomplex(x: _ScalarLike_co) -> bool_: ...  # type: ignore[misc]
@overload
def iscomplex(x: ArrayLike) -> NDArray[bool_]: ...

@overload
def isreal(x: _ScalarLike_co) -> bool_: ...  # type: ignore[misc]
@overload
def isreal(x: ArrayLike) -> NDArray[bool_]: ...

def iscomplexobj(x: _SupportsDType[dtype[Any]] | ArrayLike) -> bool: ...

def isrealobj(x: _SupportsDType[dtype[Any]] | ArrayLike) -> bool: ...

@overload
def nan_to_num(  # type: ignore[misc]
    x: _SCT,
    copy: bool = ...,
    nan: float = ...,
    posinf: None | float = ...,
    neginf: None | float = ...,
) -> _SCT: ...
@overload
def nan_to_num(
    x: _ScalarLike_co,
    copy: bool = ...,
    nan: float = ...,
    posinf: None | float = ...,
    neginf: None | float = ...,
) -> Any: ...
@overload
def nan_to_num(
    x: _ArrayLike[_SCT],
    copy: bool = ...,
    nan: float = ...,
    posinf: None | float = ...,
    neginf: None | float = ...,
) -> NDArray[_SCT]: ...
@overload
def nan_to_num(
    x: ArrayLike,
    copy: bool = ...,
    nan: float = ...,
    posinf: None | float = ...,
    neginf: None | float = ...,
) -> NDArray[Any]: ...

# If one passes a complex array to `real_if_close`, then one is reasonably
# expected to verify the output dtype (so we can return an unsafe union here)

@overload
def real_if_close(  # type: ignore[misc]
    a: _ArrayLike[complexfloating[_NBit1, _NBit1]],
    tol: float = ...,
) -> NDArray[floating[_NBit1]] | NDArray[complexfloating[_NBit1, _NBit1]]: ...
@overload
def real_if_close(
    a: _ArrayLike[_SCT],
    tol: float = ...,
) -> NDArray[_SCT]: ...
@overload
def real_if_close(
    a: ArrayLike,
    tol: float = ...,
) -> NDArray[Any]: ...

@overload
def typename(char: L['S1']) -> L['character']: ...
@overload
def typename(char: L['?']) -> L['bool']: ...
@overload
def typename(char: L['b']) -> L['signed char']: ...
@overload
def typename(char: L['B']) -> L['unsigned char']: ...
@overload
def typename(char: L['h']) -> L['short']: ...
@overload
def typename(char: L['H']) -> L['unsigned short']: ...
@overload
def typename(char: L['i']) -> L['integer']: ...
@overload
def typename(char: L['I']) -> L['unsigned integer']: ...
@overload
def typename(char: L['l']) -> L['long integer']: ...
@overload
def typename(char: L['L']) -> L['unsigned long integer']: ...
@overload
def typename(char: L['q']) -> L['long long integer']: ...
@overload
def typename(char: L['Q']) -> L['unsigned long long integer']: ...
@overload
def typename(char: L['f']) -> L['single precision']: ...
@overload
def typename(char: L['d']) -> L['double precision']: ...
@overload
def typename(char: L['g']) -> L['long precision']: ...
@overload
def typename(char: L['F']) -> L['complex single precision']: ...
@overload
def typename(char: L['D']) -> L['complex double precision']: ...
@overload
def typename(char: L['G']) -> L['complex long double precision']: ...
@overload
def typename(char: L['S']) -> L['string']: ...
@overload
def typename(char: L['U']) -> L['unicode']: ...
@overload
def typename(char: L['V']) -> L['void']: ...
@overload
def typename(char: L['O']) -> L['object']: ...

@overload
def common_type(  # type: ignore[misc]
    *arrays: _SupportsDType[dtype[
        integer[Any]
    ]]
) -> type[floating[_64Bit]]: ...
@overload
def common_type(  # type: ignore[misc]
    *arrays: _SupportsDType[dtype[
        floating[_NBit1]
    ]]
) -> type[floating[_NBit1]]: ...
@overload
def common_type(  # type: ignore[misc]
    *arrays: _SupportsDType[dtype[
        integer[Any] | floating[_NBit1]
    ]]
) -> type[floating[_NBit1 | _64Bit]]: ...
@overload
def common_type(  # type: ignore[misc]
    *arrays: _SupportsDType[dtype[
        floating[_NBit1] | complexfloating[_NBit2, _NBit2]
    ]]
) -> type[complexfloating[_NBit1 | _NBit2, _NBit1 | _NBit2]]: ...
@overload
def common_type(
    *arrays: _SupportsDType[dtype[
        integer[Any] | floating[_NBit1] | complexfloating[_NBit2, _NBit2]
    ]]
) -> type[complexfloating[_64Bit | _NBit1 | _NBit2, _64Bit | _NBit1 | _NBit2]]: ...
