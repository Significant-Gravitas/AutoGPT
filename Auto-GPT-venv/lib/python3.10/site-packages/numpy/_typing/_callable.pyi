"""
A module with various ``typing.Protocol`` subclasses that implement
the ``__call__`` magic method.

See the `Mypy documentation`_ on protocols for more details.

.. _`Mypy documentation`: https://mypy.readthedocs.io/en/stable/protocols.html#callback-protocols

"""

from __future__ import annotations

from typing import (
    TypeVar,
    overload,
    Any,
    NoReturn,
    Protocol,
)

from numpy import (
    ndarray,
    dtype,
    generic,
    bool_,
    timedelta64,
    number,
    integer,
    unsignedinteger,
    signedinteger,
    int8,
    int_,
    floating,
    float64,
    complexfloating,
    complex128,
)
from ._nbit import _NBitInt, _NBitDouble
from ._scalars import (
    _BoolLike_co,
    _IntLike_co,
    _FloatLike_co,
    _NumberLike_co,
)
from . import NBitBase
from ._generic_alias import NDArray
from ._nested_sequence import _NestedSequence

_T1 = TypeVar("_T1")
_T2 = TypeVar("_T2")
_T1_contra = TypeVar("_T1_contra", contravariant=True)
_T2_contra = TypeVar("_T2_contra", contravariant=True)
_2Tuple = tuple[_T1, _T1]

_NBit1 = TypeVar("_NBit1", bound=NBitBase)
_NBit2 = TypeVar("_NBit2", bound=NBitBase)

_IntType = TypeVar("_IntType", bound=integer)
_FloatType = TypeVar("_FloatType", bound=floating)
_NumberType = TypeVar("_NumberType", bound=number)
_NumberType_co = TypeVar("_NumberType_co", covariant=True, bound=number)
_GenericType_co = TypeVar("_GenericType_co", covariant=True, bound=generic)

class _BoolOp(Protocol[_GenericType_co]):
    @overload
    def __call__(self, other: _BoolLike_co, /) -> _GenericType_co: ...
    @overload  # platform dependent
    def __call__(self, other: int, /) -> int_: ...
    @overload
    def __call__(self, other: float, /) -> float64: ...
    @overload
    def __call__(self, other: complex, /) -> complex128: ...
    @overload
    def __call__(self, other: _NumberType, /) -> _NumberType: ...

class _BoolBitOp(Protocol[_GenericType_co]):
    @overload
    def __call__(self, other: _BoolLike_co, /) -> _GenericType_co: ...
    @overload  # platform dependent
    def __call__(self, other: int, /) -> int_: ...
    @overload
    def __call__(self, other: _IntType, /) -> _IntType: ...

class _BoolSub(Protocol):
    # Note that `other: bool_` is absent here
    @overload
    def __call__(self, other: bool, /) -> NoReturn: ...
    @overload  # platform dependent
    def __call__(self, other: int, /) -> int_: ...
    @overload
    def __call__(self, other: float, /) -> float64: ...
    @overload
    def __call__(self, other: complex, /) -> complex128: ...
    @overload
    def __call__(self, other: _NumberType, /) -> _NumberType: ...

class _BoolTrueDiv(Protocol):
    @overload
    def __call__(self, other: float | _IntLike_co, /) -> float64: ...
    @overload
    def __call__(self, other: complex, /) -> complex128: ...
    @overload
    def __call__(self, other: _NumberType, /) -> _NumberType: ...

class _BoolMod(Protocol):
    @overload
    def __call__(self, other: _BoolLike_co, /) -> int8: ...
    @overload  # platform dependent
    def __call__(self, other: int, /) -> int_: ...
    @overload
    def __call__(self, other: float, /) -> float64: ...
    @overload
    def __call__(self, other: _IntType, /) -> _IntType: ...
    @overload
    def __call__(self, other: _FloatType, /) -> _FloatType: ...

class _BoolDivMod(Protocol):
    @overload
    def __call__(self, other: _BoolLike_co, /) -> _2Tuple[int8]: ...
    @overload  # platform dependent
    def __call__(self, other: int, /) -> _2Tuple[int_]: ...
    @overload
    def __call__(self, other: float, /) -> _2Tuple[floating[_NBit1 | _NBitDouble]]: ...
    @overload
    def __call__(self, other: _IntType, /) -> _2Tuple[_IntType]: ...
    @overload
    def __call__(self, other: _FloatType, /) -> _2Tuple[_FloatType]: ...

class _TD64Div(Protocol[_NumberType_co]):
    @overload
    def __call__(self, other: timedelta64, /) -> _NumberType_co: ...
    @overload
    def __call__(self, other: _BoolLike_co, /) -> NoReturn: ...
    @overload
    def __call__(self, other: _FloatLike_co, /) -> timedelta64: ...

class _IntTrueDiv(Protocol[_NBit1]):
    @overload
    def __call__(self, other: bool, /) -> floating[_NBit1]: ...
    @overload
    def __call__(self, other: int, /) -> floating[_NBit1 | _NBitInt]: ...
    @overload
    def __call__(self, other: float, /) -> floating[_NBit1 | _NBitDouble]: ...
    @overload
    def __call__(
        self, other: complex, /,
    ) -> complexfloating[_NBit1 | _NBitDouble, _NBit1 | _NBitDouble]: ...
    @overload
    def __call__(self, other: integer[_NBit2], /) -> floating[_NBit1 | _NBit2]: ...

class _UnsignedIntOp(Protocol[_NBit1]):
    # NOTE: `uint64 + signedinteger -> float64`
    @overload
    def __call__(self, other: bool, /) -> unsignedinteger[_NBit1]: ...
    @overload
    def __call__(
        self, other: int | signedinteger[Any], /
    ) -> Any: ...
    @overload
    def __call__(self, other: float, /) -> floating[_NBit1 | _NBitDouble]: ...
    @overload
    def __call__(
        self, other: complex, /,
    ) -> complexfloating[_NBit1 | _NBitDouble, _NBit1 | _NBitDouble]: ...
    @overload
    def __call__(
        self, other: unsignedinteger[_NBit2], /
    ) -> unsignedinteger[_NBit1 | _NBit2]: ...

class _UnsignedIntBitOp(Protocol[_NBit1]):
    @overload
    def __call__(self, other: bool, /) -> unsignedinteger[_NBit1]: ...
    @overload
    def __call__(self, other: int, /) -> signedinteger[Any]: ...
    @overload
    def __call__(self, other: signedinteger[Any], /) -> signedinteger[Any]: ...
    @overload
    def __call__(
        self, other: unsignedinteger[_NBit2], /
    ) -> unsignedinteger[_NBit1 | _NBit2]: ...

class _UnsignedIntMod(Protocol[_NBit1]):
    @overload
    def __call__(self, other: bool, /) -> unsignedinteger[_NBit1]: ...
    @overload
    def __call__(
        self, other: int | signedinteger[Any], /
    ) -> Any: ...
    @overload
    def __call__(self, other: float, /) -> floating[_NBit1 | _NBitDouble]: ...
    @overload
    def __call__(
        self, other: unsignedinteger[_NBit2], /
    ) -> unsignedinteger[_NBit1 | _NBit2]: ...

class _UnsignedIntDivMod(Protocol[_NBit1]):
    @overload
    def __call__(self, other: bool, /) -> _2Tuple[signedinteger[_NBit1]]: ...
    @overload
    def __call__(
        self, other: int | signedinteger[Any], /
    ) -> _2Tuple[Any]: ...
    @overload
    def __call__(self, other: float, /) -> _2Tuple[floating[_NBit1 | _NBitDouble]]: ...
    @overload
    def __call__(
        self, other: unsignedinteger[_NBit2], /
    ) -> _2Tuple[unsignedinteger[_NBit1 | _NBit2]]: ...

class _SignedIntOp(Protocol[_NBit1]):
    @overload
    def __call__(self, other: bool, /) -> signedinteger[_NBit1]: ...
    @overload
    def __call__(self, other: int, /) -> signedinteger[_NBit1 | _NBitInt]: ...
    @overload
    def __call__(self, other: float, /) -> floating[_NBit1 | _NBitDouble]: ...
    @overload
    def __call__(
        self, other: complex, /,
    ) -> complexfloating[_NBit1 | _NBitDouble, _NBit1 | _NBitDouble]: ...
    @overload
    def __call__(
        self, other: signedinteger[_NBit2], /,
    ) -> signedinteger[_NBit1 | _NBit2]: ...

class _SignedIntBitOp(Protocol[_NBit1]):
    @overload
    def __call__(self, other: bool, /) -> signedinteger[_NBit1]: ...
    @overload
    def __call__(self, other: int, /) -> signedinteger[_NBit1 | _NBitInt]: ...
    @overload
    def __call__(
        self, other: signedinteger[_NBit2], /,
    ) -> signedinteger[_NBit1 | _NBit2]: ...

class _SignedIntMod(Protocol[_NBit1]):
    @overload
    def __call__(self, other: bool, /) -> signedinteger[_NBit1]: ...
    @overload
    def __call__(self, other: int, /) -> signedinteger[_NBit1 | _NBitInt]: ...
    @overload
    def __call__(self, other: float, /) -> floating[_NBit1 | _NBitDouble]: ...
    @overload
    def __call__(
        self, other: signedinteger[_NBit2], /,
    ) -> signedinteger[_NBit1 | _NBit2]: ...

class _SignedIntDivMod(Protocol[_NBit1]):
    @overload
    def __call__(self, other: bool, /) -> _2Tuple[signedinteger[_NBit1]]: ...
    @overload
    def __call__(self, other: int, /) -> _2Tuple[signedinteger[_NBit1 | _NBitInt]]: ...
    @overload
    def __call__(self, other: float, /) -> _2Tuple[floating[_NBit1 | _NBitDouble]]: ...
    @overload
    def __call__(
        self, other: signedinteger[_NBit2], /,
    ) -> _2Tuple[signedinteger[_NBit1 | _NBit2]]: ...

class _FloatOp(Protocol[_NBit1]):
    @overload
    def __call__(self, other: bool, /) -> floating[_NBit1]: ...
    @overload
    def __call__(self, other: int, /) -> floating[_NBit1 | _NBitInt]: ...
    @overload
    def __call__(self, other: float, /) -> floating[_NBit1 | _NBitDouble]: ...
    @overload
    def __call__(
        self, other: complex, /,
    ) -> complexfloating[_NBit1 | _NBitDouble, _NBit1 | _NBitDouble]: ...
    @overload
    def __call__(
        self, other: integer[_NBit2] | floating[_NBit2], /
    ) -> floating[_NBit1 | _NBit2]: ...

class _FloatMod(Protocol[_NBit1]):
    @overload
    def __call__(self, other: bool, /) -> floating[_NBit1]: ...
    @overload
    def __call__(self, other: int, /) -> floating[_NBit1 | _NBitInt]: ...
    @overload
    def __call__(self, other: float, /) -> floating[_NBit1 | _NBitDouble]: ...
    @overload
    def __call__(
        self, other: integer[_NBit2] | floating[_NBit2], /
    ) -> floating[_NBit1 | _NBit2]: ...

class _FloatDivMod(Protocol[_NBit1]):
    @overload
    def __call__(self, other: bool, /) -> _2Tuple[floating[_NBit1]]: ...
    @overload
    def __call__(self, other: int, /) -> _2Tuple[floating[_NBit1 | _NBitInt]]: ...
    @overload
    def __call__(self, other: float, /) -> _2Tuple[floating[_NBit1 | _NBitDouble]]: ...
    @overload
    def __call__(
        self, other: integer[_NBit2] | floating[_NBit2], /
    ) -> _2Tuple[floating[_NBit1 | _NBit2]]: ...

class _ComplexOp(Protocol[_NBit1]):
    @overload
    def __call__(self, other: bool, /) -> complexfloating[_NBit1, _NBit1]: ...
    @overload
    def __call__(self, other: int, /) -> complexfloating[_NBit1 | _NBitInt, _NBit1 | _NBitInt]: ...
    @overload
    def __call__(
        self, other: complex, /,
    ) -> complexfloating[_NBit1 | _NBitDouble, _NBit1 | _NBitDouble]: ...
    @overload
    def __call__(
        self,
        other: (
            integer[_NBit2]
            | floating[_NBit2]
            | complexfloating[_NBit2, _NBit2]
        ), /,
    ) -> complexfloating[_NBit1 | _NBit2, _NBit1 | _NBit2]: ...

class _NumberOp(Protocol):
    def __call__(self, other: _NumberLike_co, /) -> Any: ...

class _SupportsLT(Protocol):
    def __lt__(self, other: Any, /) -> object: ...

class _SupportsGT(Protocol):
    def __gt__(self, other: Any, /) -> object: ...

class _ComparisonOp(Protocol[_T1_contra, _T2_contra]):
    @overload
    def __call__(self, other: _T1_contra, /) -> bool_: ...
    @overload
    def __call__(self, other: _T2_contra, /) -> NDArray[bool_]: ...
    @overload
    def __call__(
        self,
        other: _SupportsLT | _SupportsGT | _NestedSequence[_SupportsLT | _SupportsGT],
        /,
    ) -> Any: ...
