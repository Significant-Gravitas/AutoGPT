import sys
import types
from collections.abc import Iterable
from typing import (
    Literal as L,
    Union,
    overload,
    Any,
    TypeVar,
    Protocol,
    TypedDict,
)

from numpy import (
    ndarray,
    dtype,
    generic,
    bool_,
    ubyte,
    ushort,
    uintc,
    uint,
    ulonglong,
    byte,
    short,
    intc,
    int_,
    longlong,
    half,
    single,
    double,
    longdouble,
    csingle,
    cdouble,
    clongdouble,
    datetime64,
    timedelta64,
    object_,
    str_,
    bytes_,
    void,
)

from numpy.core._type_aliases import (
    sctypeDict as sctypeDict,
    sctypes as sctypes,
)

from numpy._typing import DTypeLike, ArrayLike, _DTypeLike

_T = TypeVar("_T")
_SCT = TypeVar("_SCT", bound=generic)

class _CastFunc(Protocol):
    def __call__(
        self, x: ArrayLike, k: DTypeLike = ...
    ) -> ndarray[Any, dtype[Any]]: ...

class _TypeCodes(TypedDict):
    Character: L['c']
    Integer: L['bhilqp']
    UnsignedInteger: L['BHILQP']
    Float: L['efdg']
    Complex: L['FDG']
    AllInteger: L['bBhHiIlLqQpP']
    AllFloat: L['efdgFDG']
    Datetime: L['Mm']
    All: L['?bhilqpBHILQPefdgFDGSUVOMm']

class _typedict(dict[type[generic], _T]):
    def __getitem__(self, key: DTypeLike) -> _T: ...

if sys.version_info >= (3, 10):
    _TypeTuple = Union[
        type[Any],
        types.UnionType,
        tuple[Union[type[Any], types.UnionType, tuple[Any, ...]], ...],
    ]
else:
    _TypeTuple = Union[
        type[Any],
        tuple[Union[type[Any], tuple[Any, ...]], ...],
    ]

__all__: list[str]

@overload
def maximum_sctype(t: _DTypeLike[_SCT]) -> type[_SCT]: ...
@overload
def maximum_sctype(t: DTypeLike) -> type[Any]: ...

@overload
def issctype(rep: dtype[Any] | type[Any]) -> bool: ...
@overload
def issctype(rep: object) -> L[False]: ...

@overload
def obj2sctype(rep: _DTypeLike[_SCT], default: None = ...) -> None | type[_SCT]: ...
@overload
def obj2sctype(rep: _DTypeLike[_SCT], default: _T) -> _T | type[_SCT]: ...
@overload
def obj2sctype(rep: DTypeLike, default: None = ...) -> None | type[Any]: ...
@overload
def obj2sctype(rep: DTypeLike, default: _T) -> _T | type[Any]: ...
@overload
def obj2sctype(rep: object, default: None = ...) -> None: ...
@overload
def obj2sctype(rep: object, default: _T) -> _T: ...

@overload
def issubclass_(arg1: type[Any], arg2: _TypeTuple) -> bool: ...
@overload
def issubclass_(arg1: object, arg2: object) -> L[False]: ...

def issubsctype(arg1: DTypeLike, arg2: DTypeLike) -> bool: ...

def issubdtype(arg1: DTypeLike, arg2: DTypeLike) -> bool: ...

def sctype2char(sctype: DTypeLike) -> str: ...

def find_common_type(
    array_types: Iterable[DTypeLike],
    scalar_types: Iterable[DTypeLike],
) -> dtype[Any]: ...

cast: _typedict[_CastFunc]
nbytes: _typedict[int]
typecodes: _TypeCodes
ScalarType: tuple[
    type[int],
    type[float],
    type[complex],
    type[bool],
    type[bytes],
    type[str],
    type[memoryview],
    type[bool_],
    type[csingle],
    type[cdouble],
    type[clongdouble],
    type[half],
    type[single],
    type[double],
    type[longdouble],
    type[byte],
    type[short],
    type[intc],
    type[int_],
    type[longlong],
    type[timedelta64],
    type[datetime64],
    type[object_],
    type[bytes_],
    type[str_],
    type[ubyte],
    type[ushort],
    type[uintc],
    type[uint],
    type[ulonglong],
    type[void],
]
