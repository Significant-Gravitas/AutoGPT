"""Private counterpart of ``numpy.typing``."""

from __future__ import annotations

from numpy import ufunc
from numpy.core.overrides import set_module
from typing import TYPE_CHECKING, final


@final  # Disallow the creation of arbitrary `NBitBase` subclasses
@set_module("numpy.typing")
class NBitBase:
    """
    A type representing `numpy.number` precision during static type checking.

    Used exclusively for the purpose static type checking, `NBitBase`
    represents the base of a hierarchical set of subclasses.
    Each subsequent subclass is herein used for representing a lower level
    of precision, *e.g.* ``64Bit > 32Bit > 16Bit``.

    .. versionadded:: 1.20

    Examples
    --------
    Below is a typical usage example: `NBitBase` is herein used for annotating
    a function that takes a float and integer of arbitrary precision
    as arguments and returns a new float of whichever precision is largest
    (*e.g.* ``np.float16 + np.int64 -> np.float64``).

    .. code-block:: python

        >>> from __future__ import annotations
        >>> from typing import TypeVar, TYPE_CHECKING
        >>> import numpy as np
        >>> import numpy.typing as npt

        >>> T1 = TypeVar("T1", bound=npt.NBitBase)
        >>> T2 = TypeVar("T2", bound=npt.NBitBase)

        >>> def add(a: np.floating[T1], b: np.integer[T2]) -> np.floating[T1 | T2]:
        ...     return a + b

        >>> a = np.float16()
        >>> b = np.int64()
        >>> out = add(a, b)

        >>> if TYPE_CHECKING:
        ...     reveal_locals()
        ...     # note: Revealed local types are:
        ...     # note:     a: numpy.floating[numpy.typing._16Bit*]
        ...     # note:     b: numpy.signedinteger[numpy.typing._64Bit*]
        ...     # note:     out: numpy.floating[numpy.typing._64Bit*]

    """

    def __init_subclass__(cls) -> None:
        allowed_names = {
            "NBitBase", "_256Bit", "_128Bit", "_96Bit", "_80Bit",
            "_64Bit", "_32Bit", "_16Bit", "_8Bit",
        }
        if cls.__name__ not in allowed_names:
            raise TypeError('cannot inherit from final class "NBitBase"')
        super().__init_subclass__()


# Silence errors about subclassing a `@final`-decorated class
class _256Bit(NBitBase):  # type: ignore[misc]
    pass

class _128Bit(_256Bit):  # type: ignore[misc]
    pass

class _96Bit(_128Bit):  # type: ignore[misc]
    pass

class _80Bit(_96Bit):  # type: ignore[misc]
    pass

class _64Bit(_80Bit):  # type: ignore[misc]
    pass

class _32Bit(_64Bit):  # type: ignore[misc]
    pass

class _16Bit(_32Bit):  # type: ignore[misc]
    pass

class _8Bit(_16Bit):  # type: ignore[misc]
    pass


from ._nested_sequence import (
    _NestedSequence as _NestedSequence,
)
from ._nbit import (
    _NBitByte as _NBitByte,
    _NBitShort as _NBitShort,
    _NBitIntC as _NBitIntC,
    _NBitIntP as _NBitIntP,
    _NBitInt as _NBitInt,
    _NBitLongLong as _NBitLongLong,
    _NBitHalf as _NBitHalf,
    _NBitSingle as _NBitSingle,
    _NBitDouble as _NBitDouble,
    _NBitLongDouble as _NBitLongDouble,
)
from ._char_codes import (
    _BoolCodes as _BoolCodes,
    _UInt8Codes as _UInt8Codes,
    _UInt16Codes as _UInt16Codes,
    _UInt32Codes as _UInt32Codes,
    _UInt64Codes as _UInt64Codes,
    _Int8Codes as _Int8Codes,
    _Int16Codes as _Int16Codes,
    _Int32Codes as _Int32Codes,
    _Int64Codes as _Int64Codes,
    _Float16Codes as _Float16Codes,
    _Float32Codes as _Float32Codes,
    _Float64Codes as _Float64Codes,
    _Complex64Codes as _Complex64Codes,
    _Complex128Codes as _Complex128Codes,
    _ByteCodes as _ByteCodes,
    _ShortCodes as _ShortCodes,
    _IntCCodes as _IntCCodes,
    _IntPCodes as _IntPCodes,
    _IntCodes as _IntCodes,
    _LongLongCodes as _LongLongCodes,
    _UByteCodes as _UByteCodes,
    _UShortCodes as _UShortCodes,
    _UIntCCodes as _UIntCCodes,
    _UIntPCodes as _UIntPCodes,
    _UIntCodes as _UIntCodes,
    _ULongLongCodes as _ULongLongCodes,
    _HalfCodes as _HalfCodes,
    _SingleCodes as _SingleCodes,
    _DoubleCodes as _DoubleCodes,
    _LongDoubleCodes as _LongDoubleCodes,
    _CSingleCodes as _CSingleCodes,
    _CDoubleCodes as _CDoubleCodes,
    _CLongDoubleCodes as _CLongDoubleCodes,
    _DT64Codes as _DT64Codes,
    _TD64Codes as _TD64Codes,
    _StrCodes as _StrCodes,
    _BytesCodes as _BytesCodes,
    _VoidCodes as _VoidCodes,
    _ObjectCodes as _ObjectCodes,
)
from ._scalars import (
    _CharLike_co as _CharLike_co,
    _BoolLike_co as _BoolLike_co,
    _UIntLike_co as _UIntLike_co,
    _IntLike_co as _IntLike_co,
    _FloatLike_co as _FloatLike_co,
    _ComplexLike_co as _ComplexLike_co,
    _TD64Like_co as _TD64Like_co,
    _NumberLike_co as _NumberLike_co,
    _ScalarLike_co as _ScalarLike_co,
    _VoidLike_co as _VoidLike_co,
)
from ._shape import (
    _Shape as _Shape,
    _ShapeLike as _ShapeLike,
)
from ._dtype_like import (
    DTypeLike as DTypeLike,
    _DTypeLike as _DTypeLike,
    _SupportsDType as _SupportsDType,
    _VoidDTypeLike as _VoidDTypeLike,
    _DTypeLikeBool as _DTypeLikeBool,
    _DTypeLikeUInt as _DTypeLikeUInt,
    _DTypeLikeInt as _DTypeLikeInt,
    _DTypeLikeFloat as _DTypeLikeFloat,
    _DTypeLikeComplex as _DTypeLikeComplex,
    _DTypeLikeTD64 as _DTypeLikeTD64,
    _DTypeLikeDT64 as _DTypeLikeDT64,
    _DTypeLikeObject as _DTypeLikeObject,
    _DTypeLikeVoid as _DTypeLikeVoid,
    _DTypeLikeStr as _DTypeLikeStr,
    _DTypeLikeBytes as _DTypeLikeBytes,
    _DTypeLikeComplex_co as _DTypeLikeComplex_co,
)
from ._array_like import (
    ArrayLike as ArrayLike,
    _ArrayLike as _ArrayLike,
    _FiniteNestedSequence as _FiniteNestedSequence,
    _SupportsArray as _SupportsArray,
    _SupportsArrayFunc as _SupportsArrayFunc,
    _ArrayLikeInt as _ArrayLikeInt,
    _ArrayLikeBool_co as _ArrayLikeBool_co,
    _ArrayLikeUInt_co as _ArrayLikeUInt_co,
    _ArrayLikeInt_co as _ArrayLikeInt_co,
    _ArrayLikeFloat_co as _ArrayLikeFloat_co,
    _ArrayLikeComplex_co as _ArrayLikeComplex_co,
    _ArrayLikeNumber_co as _ArrayLikeNumber_co,
    _ArrayLikeTD64_co as _ArrayLikeTD64_co,
    _ArrayLikeDT64_co as _ArrayLikeDT64_co,
    _ArrayLikeObject_co as _ArrayLikeObject_co,
    _ArrayLikeVoid_co as _ArrayLikeVoid_co,
    _ArrayLikeStr_co as _ArrayLikeStr_co,
    _ArrayLikeBytes_co as _ArrayLikeBytes_co,
    _ArrayLikeUnknown as _ArrayLikeUnknown,
    _UnknownType as _UnknownType,
)
from ._generic_alias import (
    NDArray as NDArray,
    _DType as _DType,
    _GenericAlias as _GenericAlias,
)

if TYPE_CHECKING:
    from ._ufunc import (
        _UFunc_Nin1_Nout1 as _UFunc_Nin1_Nout1,
        _UFunc_Nin2_Nout1 as _UFunc_Nin2_Nout1,
        _UFunc_Nin1_Nout2 as _UFunc_Nin1_Nout2,
        _UFunc_Nin2_Nout2 as _UFunc_Nin2_Nout2,
        _GUFunc_Nin2_Nout1 as _GUFunc_Nin2_Nout1,
    )
else:
    # Declare the (type-check-only) ufunc subclasses as ufunc aliases during
    # runtime; this helps autocompletion tools such as Jedi (numpy/numpy#19834)
    _UFunc_Nin1_Nout1 = ufunc
    _UFunc_Nin2_Nout1 = ufunc
    _UFunc_Nin1_Nout2 = ufunc
    _UFunc_Nin2_Nout2 = ufunc
    _GUFunc_Nin2_Nout1 = ufunc
