import builtins
import os
import sys
import mmap
import ctypes as ct
import array as _array
import datetime as dt
import enum
from abc import abstractmethod
from types import TracebackType, MappingProxyType
from contextlib import ContextDecorator
from contextlib import contextmanager

if sys.version_info >= (3, 9):
    from types import GenericAlias

from numpy._pytesttester import PytestTester
from numpy.core._internal import _ctypes

from numpy._typing import (
    # Arrays
    ArrayLike,
    NDArray,
    _SupportsArray,
    _NestedSequence,
    _FiniteNestedSequence,
    _SupportsArray,
    _ArrayLikeBool_co,
    _ArrayLikeUInt_co,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeNumber_co,
    _ArrayLikeTD64_co,
    _ArrayLikeDT64_co,
    _ArrayLikeObject_co,
    _ArrayLikeStr_co,
    _ArrayLikeBytes_co,
    _ArrayLikeUnknown,
    _UnknownType,

    # DTypes
    DTypeLike,
    _DTypeLike,
    _DTypeLikeVoid,
    _SupportsDType,
    _VoidDTypeLike,

    # Shapes
    _Shape,
    _ShapeLike,

    # Scalars
    _CharLike_co,
    _BoolLike_co,
    _IntLike_co,
    _FloatLike_co,
    _ComplexLike_co,
    _TD64Like_co,
    _NumberLike_co,
    _ScalarLike_co,

    # `number` precision
    NBitBase,
    _256Bit,
    _128Bit,
    _96Bit,
    _80Bit,
    _64Bit,
    _32Bit,
    _16Bit,
    _8Bit,
    _NBitByte,
    _NBitShort,
    _NBitIntC,
    _NBitIntP,
    _NBitInt,
    _NBitLongLong,
    _NBitHalf,
    _NBitSingle,
    _NBitDouble,
    _NBitLongDouble,

    # Character codes
    _BoolCodes,
    _UInt8Codes,
    _UInt16Codes,
    _UInt32Codes,
    _UInt64Codes,
    _Int8Codes,
    _Int16Codes,
    _Int32Codes,
    _Int64Codes,
    _Float16Codes,
    _Float32Codes,
    _Float64Codes,
    _Complex64Codes,
    _Complex128Codes,
    _ByteCodes,
    _ShortCodes,
    _IntCCodes,
    _IntPCodes,
    _IntCodes,
    _LongLongCodes,
    _UByteCodes,
    _UShortCodes,
    _UIntCCodes,
    _UIntPCodes,
    _UIntCodes,
    _ULongLongCodes,
    _HalfCodes,
    _SingleCodes,
    _DoubleCodes,
    _LongDoubleCodes,
    _CSingleCodes,
    _CDoubleCodes,
    _CLongDoubleCodes,
    _DT64Codes,
    _TD64Codes,
    _StrCodes,
    _BytesCodes,
    _VoidCodes,
    _ObjectCodes,

    # Ufuncs
    _UFunc_Nin1_Nout1,
    _UFunc_Nin2_Nout1,
    _UFunc_Nin1_Nout2,
    _UFunc_Nin2_Nout2,
    _GUFunc_Nin2_Nout1,
)

from numpy._typing._callable import (
    _BoolOp,
    _BoolBitOp,
    _BoolSub,
    _BoolTrueDiv,
    _BoolMod,
    _BoolDivMod,
    _TD64Div,
    _IntTrueDiv,
    _UnsignedIntOp,
    _UnsignedIntBitOp,
    _UnsignedIntMod,
    _UnsignedIntDivMod,
    _SignedIntOp,
    _SignedIntBitOp,
    _SignedIntMod,
    _SignedIntDivMod,
    _FloatOp,
    _FloatMod,
    _FloatDivMod,
    _ComplexOp,
    _NumberOp,
    _ComparisonOp,
)

# NOTE: Numpy's mypy plugin is used for removing the types unavailable
# to the specific platform
from numpy._typing._extended_precision import (
    uint128 as uint128,
    uint256 as uint256,
    int128 as int128,
    int256 as int256,
    float80 as float80,
    float96 as float96,
    float128 as float128,
    float256 as float256,
    complex160 as complex160,
    complex192 as complex192,
    complex256 as complex256,
    complex512 as complex512,
)

from collections.abc import (
    Callable,
    Container,
    Iterable,
    Iterator,
    Mapping,
    Sequence,
    Sized,
)
from typing import (
    Literal as L,
    Any,
    Generator,
    Generic,
    IO,
    NoReturn,
    overload,
    SupportsComplex,
    SupportsFloat,
    SupportsInt,
    TypeVar,
    Union,
    Protocol,
    SupportsIndex,
    Final,
    final,
    ClassVar,
)

# Ensures that the stubs are picked up
from numpy import (
    ctypeslib as ctypeslib,
    fft as fft,
    lib as lib,
    linalg as linalg,
    ma as ma,
    polynomial as polynomial,
    random as random,
    testing as testing,
    version as version,
)

from numpy.core import defchararray, records
char = defchararray
rec = records

from numpy.core.function_base import (
    linspace as linspace,
    logspace as logspace,
    geomspace as geomspace,
)

from numpy.core.fromnumeric import (
    take as take,
    reshape as reshape,
    choose as choose,
    repeat as repeat,
    put as put,
    swapaxes as swapaxes,
    transpose as transpose,
    partition as partition,
    argpartition as argpartition,
    sort as sort,
    argsort as argsort,
    argmax as argmax,
    argmin as argmin,
    searchsorted as searchsorted,
    resize as resize,
    squeeze as squeeze,
    diagonal as diagonal,
    trace as trace,
    ravel as ravel,
    nonzero as nonzero,
    shape as shape,
    compress as compress,
    clip as clip,
    sum as sum,
    all as all,
    any as any,
    cumsum as cumsum,
    ptp as ptp,
    amax as amax,
    amin as amin,
    prod as prod,
    cumprod as cumprod,
    ndim as ndim,
    size as size,
    around as around,
    mean as mean,
    std as std,
    var as var,
)

from numpy.core._asarray import (
    require as require,
)

from numpy.core._type_aliases import (
    sctypes as sctypes,
    sctypeDict as sctypeDict,
)

from numpy.core._ufunc_config import (
    seterr as seterr,
    geterr as geterr,
    setbufsize as setbufsize,
    getbufsize as getbufsize,
    seterrcall as seterrcall,
    geterrcall as geterrcall,
    _ErrKind,
    _ErrFunc,
    _ErrDictOptional,
)

from numpy.core.arrayprint import (
    set_printoptions as set_printoptions,
    get_printoptions as get_printoptions,
    array2string as array2string,
    format_float_scientific as format_float_scientific,
    format_float_positional as format_float_positional,
    array_repr as array_repr,
    array_str as array_str,
    set_string_function as set_string_function,
    printoptions as printoptions,
)

from numpy.core.einsumfunc import (
    einsum as einsum,
    einsum_path as einsum_path,
)

from numpy.core.multiarray import (
    ALLOW_THREADS as ALLOW_THREADS,
    BUFSIZE as BUFSIZE,
    CLIP as CLIP,
    MAXDIMS as MAXDIMS,
    MAY_SHARE_BOUNDS as MAY_SHARE_BOUNDS,
    MAY_SHARE_EXACT as MAY_SHARE_EXACT,
    RAISE as RAISE,
    WRAP as WRAP,
    tracemalloc_domain as tracemalloc_domain,
    array as array,
    empty_like as empty_like,
    empty as empty,
    zeros as zeros,
    concatenate as concatenate,
    inner as inner,
    where as where,
    lexsort as lexsort,
    can_cast as can_cast,
    min_scalar_type as min_scalar_type,
    result_type as result_type,
    dot as dot,
    vdot as vdot,
    bincount as bincount,
    copyto as copyto,
    putmask as putmask,
    packbits as packbits,
    unpackbits as unpackbits,
    shares_memory as shares_memory,
    may_share_memory as may_share_memory,
    asarray as asarray,
    asanyarray as asanyarray,
    ascontiguousarray as ascontiguousarray,
    asfortranarray as asfortranarray,
    arange as arange,
    busday_count as busday_count,
    busday_offset as busday_offset,
    compare_chararrays as compare_chararrays,
    datetime_as_string as datetime_as_string,
    datetime_data as datetime_data,
    frombuffer as frombuffer,
    fromfile as fromfile,
    fromiter as fromiter,
    is_busday as is_busday,
    promote_types as promote_types,
    seterrobj as seterrobj,
    geterrobj as geterrobj,
    fromstring as fromstring,
    frompyfunc as frompyfunc,
    nested_iters as nested_iters,
    flagsobj,
)

from numpy.core.numeric import (
    zeros_like as zeros_like,
    ones as ones,
    ones_like as ones_like,
    full as full,
    full_like as full_like,
    count_nonzero as count_nonzero,
    isfortran as isfortran,
    argwhere as argwhere,
    flatnonzero as flatnonzero,
    correlate as correlate,
    convolve as convolve,
    outer as outer,
    tensordot as tensordot,
    roll as roll,
    rollaxis as rollaxis,
    moveaxis as moveaxis,
    cross as cross,
    indices as indices,
    fromfunction as fromfunction,
    isscalar as isscalar,
    binary_repr as binary_repr,
    base_repr as base_repr,
    identity as identity,
    allclose as allclose,
    isclose as isclose,
    array_equal as array_equal,
    array_equiv as array_equiv,
)

from numpy.core.numerictypes import (
    maximum_sctype as maximum_sctype,
    issctype as issctype,
    obj2sctype as obj2sctype,
    issubclass_ as issubclass_,
    issubsctype as issubsctype,
    issubdtype as issubdtype,
    sctype2char as sctype2char,
    find_common_type as find_common_type,
    nbytes as nbytes,
    cast as cast,
    ScalarType as ScalarType,
    typecodes as typecodes,
)

from numpy.core.shape_base import (
    atleast_1d as atleast_1d,
    atleast_2d as atleast_2d,
    atleast_3d as atleast_3d,
    block as block,
    hstack as hstack,
    stack as stack,
    vstack as vstack,
)

from numpy.lib import (
    emath as emath,
)

from numpy.lib.arraypad import (
    pad as pad,
)

from numpy.lib.arraysetops import (
    ediff1d as ediff1d,
    intersect1d as intersect1d,
    setxor1d as setxor1d,
    union1d as union1d,
    setdiff1d as setdiff1d,
    unique as unique,
    in1d as in1d,
    isin as isin,
)

from numpy.lib.arrayterator import (
    Arrayterator as Arrayterator,
)

from numpy.lib.function_base import (
    select as select,
    piecewise as piecewise,
    trim_zeros as trim_zeros,
    copy as copy,
    iterable as iterable,
    percentile as percentile,
    diff as diff,
    gradient as gradient,
    angle as angle,
    unwrap as unwrap,
    sort_complex as sort_complex,
    disp as disp,
    flip as flip,
    rot90 as rot90,
    extract as extract,
    place as place,
    asarray_chkfinite as asarray_chkfinite,
    average as average,
    bincount as bincount,
    digitize as digitize,
    cov as cov,
    corrcoef as corrcoef,
    msort as msort,
    median as median,
    sinc as sinc,
    hamming as hamming,
    hanning as hanning,
    bartlett as bartlett,
    blackman as blackman,
    kaiser as kaiser,
    trapz as trapz,
    i0 as i0,
    add_newdoc as add_newdoc,
    add_docstring as add_docstring,
    meshgrid as meshgrid,
    delete as delete,
    insert as insert,
    append as append,
    interp as interp,
    add_newdoc_ufunc as add_newdoc_ufunc,
    quantile as quantile,
)

from numpy.lib.histograms import (
    histogram_bin_edges as histogram_bin_edges,
    histogram as histogram,
    histogramdd as histogramdd,
)

from numpy.lib.index_tricks import (
    ravel_multi_index as ravel_multi_index,
    unravel_index as unravel_index,
    mgrid as mgrid,
    ogrid as ogrid,
    r_ as r_,
    c_ as c_,
    s_ as s_,
    index_exp as index_exp,
    ix_ as ix_,
    fill_diagonal as fill_diagonal,
    diag_indices as diag_indices,
    diag_indices_from as diag_indices_from,
)

from numpy.lib.nanfunctions import (
    nansum as nansum,
    nanmax as nanmax,
    nanmin as nanmin,
    nanargmax as nanargmax,
    nanargmin as nanargmin,
    nanmean as nanmean,
    nanmedian as nanmedian,
    nanpercentile as nanpercentile,
    nanvar as nanvar,
    nanstd as nanstd,
    nanprod as nanprod,
    nancumsum as nancumsum,
    nancumprod as nancumprod,
    nanquantile as nanquantile,
)

from numpy.lib.npyio import (
    savetxt as savetxt,
    loadtxt as loadtxt,
    genfromtxt as genfromtxt,
    recfromtxt as recfromtxt,
    recfromcsv as recfromcsv,
    load as load,
    save as save,
    savez as savez,
    savez_compressed as savez_compressed,
    packbits as packbits,
    unpackbits as unpackbits,
    fromregex as fromregex,
)

from numpy.lib.polynomial import (
    poly as poly,
    roots as roots,
    polyint as polyint,
    polyder as polyder,
    polyadd as polyadd,
    polysub as polysub,
    polymul as polymul,
    polydiv as polydiv,
    polyval as polyval,
    polyfit as polyfit,
)

from numpy.lib.shape_base import (
    column_stack as column_stack,
    row_stack as row_stack,
    dstack as dstack,
    array_split as array_split,
    split as split,
    hsplit as hsplit,
    vsplit as vsplit,
    dsplit as dsplit,
    apply_over_axes as apply_over_axes,
    expand_dims as expand_dims,
    apply_along_axis as apply_along_axis,
    kron as kron,
    tile as tile,
    get_array_wrap as get_array_wrap,
    take_along_axis as take_along_axis,
    put_along_axis as put_along_axis,
)

from numpy.lib.stride_tricks import (
    broadcast_to as broadcast_to,
    broadcast_arrays as broadcast_arrays,
    broadcast_shapes as broadcast_shapes,
)

from numpy.lib.twodim_base import (
    diag as diag,
    diagflat as diagflat,
    eye as eye,
    fliplr as fliplr,
    flipud as flipud,
    tri as tri,
    triu as triu,
    tril as tril,
    vander as vander,
    histogram2d as histogram2d,
    mask_indices as mask_indices,
    tril_indices as tril_indices,
    tril_indices_from as tril_indices_from,
    triu_indices as triu_indices,
    triu_indices_from as triu_indices_from,
)

from numpy.lib.type_check import (
    mintypecode as mintypecode,
    asfarray as asfarray,
    real as real,
    imag as imag,
    iscomplex as iscomplex,
    isreal as isreal,
    iscomplexobj as iscomplexobj,
    isrealobj as isrealobj,
    nan_to_num as nan_to_num,
    real_if_close as real_if_close,
    typename as typename,
    common_type as common_type,
)

from numpy.lib.ufunclike import (
    fix as fix,
    isposinf as isposinf,
    isneginf as isneginf,
)

from numpy.lib.utils import (
    issubclass_ as issubclass_,
    issubsctype as issubsctype,
    issubdtype as issubdtype,
    deprecate as deprecate,
    deprecate_with_doc as deprecate_with_doc,
    get_include as get_include,
    info as info,
    source as source,
    who as who,
    lookfor as lookfor,
    byte_bounds as byte_bounds,
    safe_eval as safe_eval,
    show_runtime as show_runtime,
)

from numpy.matrixlib import (
    asmatrix as asmatrix,
    mat as mat,
    bmat as bmat,
)

_AnyStr_contra = TypeVar("_AnyStr_contra", str, bytes, contravariant=True)

# Protocol for representing file-like-objects accepted
# by `ndarray.tofile` and `fromfile`
class _IOProtocol(Protocol):
    def flush(self) -> object: ...
    def fileno(self) -> int: ...
    def tell(self) -> SupportsIndex: ...
    def seek(self, offset: int, whence: int, /) -> object: ...

# NOTE: `seek`, `write` and `flush` are technically only required
# for `readwrite`/`write` modes
class _MemMapIOProtocol(Protocol):
    def flush(self) -> object: ...
    def fileno(self) -> SupportsIndex: ...
    def tell(self) -> int: ...
    def seek(self, offset: int, whence: int, /) -> object: ...
    def write(self, s: bytes, /) -> object: ...
    @property
    def read(self) -> object: ...

class _SupportsWrite(Protocol[_AnyStr_contra]):
    def write(self, s: _AnyStr_contra, /) -> object: ...

__all__: list[str]
__path__: list[str]
__version__: str
__git_version__: str
test: PytestTester

# TODO: Move placeholders to their respective module once
# their annotations are properly implemented
#
# Placeholders for classes

# Some of these are aliases; others are wrappers with an identical signature
round = around
round_ = around
max = amax
min = amin
product = prod
cumproduct = cumprod
sometrue = any
alltrue = all

def show_config() -> None: ...

_NdArraySubClass = TypeVar("_NdArraySubClass", bound=ndarray)
_DTypeScalar_co = TypeVar("_DTypeScalar_co", covariant=True, bound=generic)
_ByteOrder = L["S", "<", ">", "=", "|", "L", "B", "N", "I"]

@final
class dtype(Generic[_DTypeScalar_co]):
    names: None | tuple[builtins.str, ...]
    # Overload for subclass of generic
    @overload
    def __new__(
        cls,
        dtype: type[_DTypeScalar_co],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[_DTypeScalar_co]: ...
    # Overloads for string aliases, Python types, and some assorted
    # other special cases. Order is sometimes important because of the
    # subtype relationships
    #
    # bool < int < float < complex < object
    #
    # so we have to make sure the overloads for the narrowest type is
    # first.
    # Builtin types
    @overload
    def __new__(cls, dtype: type[bool], align: bool = ..., copy: bool = ...) -> dtype[bool_]: ...
    @overload
    def __new__(cls, dtype: type[int], align: bool = ..., copy: bool = ...) -> dtype[int_]: ...
    @overload
    def __new__(cls, dtype: None | type[float], align: bool = ..., copy: bool = ...) -> dtype[float_]: ...
    @overload
    def __new__(cls, dtype: type[complex], align: bool = ..., copy: bool = ...) -> dtype[complex_]: ...
    @overload
    def __new__(cls, dtype: type[builtins.str], align: bool = ..., copy: bool = ...) -> dtype[str_]: ...
    @overload
    def __new__(cls, dtype: type[bytes], align: bool = ..., copy: bool = ...) -> dtype[bytes_]: ...

    # `unsignedinteger` string-based representations and ctypes
    @overload
    def __new__(cls, dtype: _UInt8Codes | type[ct.c_uint8], align: bool = ..., copy: bool = ...) -> dtype[uint8]: ...
    @overload
    def __new__(cls, dtype: _UInt16Codes | type[ct.c_uint16], align: bool = ..., copy: bool = ...) -> dtype[uint16]: ...
    @overload
    def __new__(cls, dtype: _UInt32Codes | type[ct.c_uint32], align: bool = ..., copy: bool = ...) -> dtype[uint32]: ...
    @overload
    def __new__(cls, dtype: _UInt64Codes | type[ct.c_uint64], align: bool = ..., copy: bool = ...) -> dtype[uint64]: ...
    @overload
    def __new__(cls, dtype: _UByteCodes | type[ct.c_ubyte], align: bool = ..., copy: bool = ...) -> dtype[ubyte]: ...
    @overload
    def __new__(cls, dtype: _UShortCodes | type[ct.c_ushort], align: bool = ..., copy: bool = ...) -> dtype[ushort]: ...
    @overload
    def __new__(cls, dtype: _UIntCCodes | type[ct.c_uint], align: bool = ..., copy: bool = ...) -> dtype[uintc]: ...

    # NOTE: We're assuming here that `uint_ptr_t == size_t`,
    # an assumption that does not hold in rare cases (same for `ssize_t`)
    @overload
    def __new__(cls, dtype: _UIntPCodes | type[ct.c_void_p] | type[ct.c_size_t], align: bool = ..., copy: bool = ...) -> dtype[uintp]: ...
    @overload
    def __new__(cls, dtype: _UIntCodes | type[ct.c_ulong], align: bool = ..., copy: bool = ...) -> dtype[uint]: ...
    @overload
    def __new__(cls, dtype: _ULongLongCodes | type[ct.c_ulonglong], align: bool = ..., copy: bool = ...) -> dtype[ulonglong]: ...

    # `signedinteger` string-based representations and ctypes
    @overload
    def __new__(cls, dtype: _Int8Codes | type[ct.c_int8], align: bool = ..., copy: bool = ...) -> dtype[int8]: ...
    @overload
    def __new__(cls, dtype: _Int16Codes | type[ct.c_int16], align: bool = ..., copy: bool = ...) -> dtype[int16]: ...
    @overload
    def __new__(cls, dtype: _Int32Codes | type[ct.c_int32], align: bool = ..., copy: bool = ...) -> dtype[int32]: ...
    @overload
    def __new__(cls, dtype: _Int64Codes | type[ct.c_int64], align: bool = ..., copy: bool = ...) -> dtype[int64]: ...
    @overload
    def __new__(cls, dtype: _ByteCodes | type[ct.c_byte], align: bool = ..., copy: bool = ...) -> dtype[byte]: ...
    @overload
    def __new__(cls, dtype: _ShortCodes | type[ct.c_short], align: bool = ..., copy: bool = ...) -> dtype[short]: ...
    @overload
    def __new__(cls, dtype: _IntCCodes | type[ct.c_int], align: bool = ..., copy: bool = ...) -> dtype[intc]: ...
    @overload
    def __new__(cls, dtype: _IntPCodes | type[ct.c_ssize_t], align: bool = ..., copy: bool = ...) -> dtype[intp]: ...
    @overload
    def __new__(cls, dtype: _IntCodes | type[ct.c_long], align: bool = ..., copy: bool = ...) -> dtype[int_]: ...
    @overload
    def __new__(cls, dtype: _LongLongCodes | type[ct.c_longlong], align: bool = ..., copy: bool = ...) -> dtype[longlong]: ...

    # `floating` string-based representations and ctypes
    @overload
    def __new__(cls, dtype: _Float16Codes, align: bool = ..., copy: bool = ...) -> dtype[float16]: ...
    @overload
    def __new__(cls, dtype: _Float32Codes, align: bool = ..., copy: bool = ...) -> dtype[float32]: ...
    @overload
    def __new__(cls, dtype: _Float64Codes, align: bool = ..., copy: bool = ...) -> dtype[float64]: ...
    @overload
    def __new__(cls, dtype: _HalfCodes, align: bool = ..., copy: bool = ...) -> dtype[half]: ...
    @overload
    def __new__(cls, dtype: _SingleCodes | type[ct.c_float], align: bool = ..., copy: bool = ...) -> dtype[single]: ...
    @overload
    def __new__(cls, dtype: _DoubleCodes | type[ct.c_double], align: bool = ..., copy: bool = ...) -> dtype[double]: ...
    @overload
    def __new__(cls, dtype: _LongDoubleCodes | type[ct.c_longdouble], align: bool = ..., copy: bool = ...) -> dtype[longdouble]: ...

    # `complexfloating` string-based representations
    @overload
    def __new__(cls, dtype: _Complex64Codes, align: bool = ..., copy: bool = ...) -> dtype[complex64]: ...
    @overload
    def __new__(cls, dtype: _Complex128Codes, align: bool = ..., copy: bool = ...) -> dtype[complex128]: ...
    @overload
    def __new__(cls, dtype: _CSingleCodes, align: bool = ..., copy: bool = ...) -> dtype[csingle]: ...
    @overload
    def __new__(cls, dtype: _CDoubleCodes, align: bool = ..., copy: bool = ...) -> dtype[cdouble]: ...
    @overload
    def __new__(cls, dtype: _CLongDoubleCodes, align: bool = ..., copy: bool = ...) -> dtype[clongdouble]: ...

    # Miscellaneous string-based representations and ctypes
    @overload
    def __new__(cls, dtype: _BoolCodes | type[ct.c_bool], align: bool = ..., copy: bool = ...) -> dtype[bool_]: ...
    @overload
    def __new__(cls, dtype: _TD64Codes, align: bool = ..., copy: bool = ...) -> dtype[timedelta64]: ...
    @overload
    def __new__(cls, dtype: _DT64Codes, align: bool = ..., copy: bool = ...) -> dtype[datetime64]: ...
    @overload
    def __new__(cls, dtype: _StrCodes, align: bool = ..., copy: bool = ...) -> dtype[str_]: ...
    @overload
    def __new__(cls, dtype: _BytesCodes | type[ct.c_char], align: bool = ..., copy: bool = ...) -> dtype[bytes_]: ...
    @overload
    def __new__(cls, dtype: _VoidCodes, align: bool = ..., copy: bool = ...) -> dtype[void]: ...
    @overload
    def __new__(cls, dtype: _ObjectCodes | type[ct.py_object], align: bool = ..., copy: bool = ...) -> dtype[object_]: ...

    # dtype of a dtype is the same dtype
    @overload
    def __new__(
        cls,
        dtype: dtype[_DTypeScalar_co],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[_DTypeScalar_co]: ...
    @overload
    def __new__(
        cls,
        dtype: _SupportsDType[dtype[_DTypeScalar_co]],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[_DTypeScalar_co]: ...
    # Handle strings that can't be expressed as literals; i.e. s1, s2, ...
    @overload
    def __new__(
        cls,
        dtype: builtins.str,
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[Any]: ...
    # Catchall overload for void-likes
    @overload
    def __new__(
        cls,
        dtype: _VoidDTypeLike,
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[void]: ...
    # Catchall overload for object-likes
    @overload
    def __new__(
        cls,
        dtype: type[object],
        align: bool = ...,
        copy: bool = ...,
    ) -> dtype[object_]: ...

    if sys.version_info >= (3, 9):
        def __class_getitem__(self, item: Any) -> GenericAlias: ...

    @overload
    def __getitem__(self: dtype[void], key: list[builtins.str]) -> dtype[void]: ...
    @overload
    def __getitem__(self: dtype[void], key: builtins.str | SupportsIndex) -> dtype[Any]: ...

    # NOTE: In the future 1-based multiplications will also yield `flexible` dtypes
    @overload
    def __mul__(self: _DType, value: L[1]) -> _DType: ...
    @overload
    def __mul__(self: _FlexDType, value: SupportsIndex) -> _FlexDType: ...
    @overload
    def __mul__(self, value: SupportsIndex) -> dtype[void]: ...

    # NOTE: `__rmul__` seems to be broken when used in combination with
    # literals as of mypy 0.902. Set the return-type to `dtype[Any]` for
    # now for non-flexible dtypes.
    @overload
    def __rmul__(self: _FlexDType, value: SupportsIndex) -> _FlexDType: ...
    @overload
    def __rmul__(self, value: SupportsIndex) -> dtype[Any]: ...

    def __gt__(self, other: DTypeLike) -> bool: ...
    def __ge__(self, other: DTypeLike) -> bool: ...
    def __lt__(self, other: DTypeLike) -> bool: ...
    def __le__(self, other: DTypeLike) -> bool: ...

    # Explicitly defined `__eq__` and `__ne__` to get around mypy's
    # `strict_equality` option; even though their signatures are
    # identical to their `object`-based counterpart
    def __eq__(self, other: Any) -> bool: ...
    def __ne__(self, other: Any) -> bool: ...

    @property
    def alignment(self) -> int: ...
    @property
    def base(self) -> dtype[Any]: ...
    @property
    def byteorder(self) -> builtins.str: ...
    @property
    def char(self) -> builtins.str: ...
    @property
    def descr(self) -> list[tuple[builtins.str, builtins.str] | tuple[builtins.str, builtins.str, _Shape]]: ...
    @property
    def fields(
        self,
    ) -> None | MappingProxyType[builtins.str, tuple[dtype[Any], int] | tuple[dtype[Any], int, Any]]: ...
    @property
    def flags(self) -> int: ...
    @property
    def hasobject(self) -> bool: ...
    @property
    def isbuiltin(self) -> int: ...
    @property
    def isnative(self) -> bool: ...
    @property
    def isalignedstruct(self) -> bool: ...
    @property
    def itemsize(self) -> int: ...
    @property
    def kind(self) -> builtins.str: ...
    @property
    def metadata(self) -> None | MappingProxyType[builtins.str, Any]: ...
    @property
    def name(self) -> builtins.str: ...
    @property
    def num(self) -> int: ...
    @property
    def shape(self) -> _Shape: ...
    @property
    def ndim(self) -> int: ...
    @property
    def subdtype(self) -> None | tuple[dtype[Any], _Shape]: ...
    def newbyteorder(self: _DType, __new_order: _ByteOrder = ...) -> _DType: ...
    @property
    def str(self) -> builtins.str: ...
    @property
    def type(self) -> type[_DTypeScalar_co]: ...

_ArrayLikeInt = Union[
    int,
    integer,
    Sequence[Union[int, integer]],
    Sequence[Sequence[Any]],  # TODO: wait for support for recursive types
    ndarray
]

_FlatIterSelf = TypeVar("_FlatIterSelf", bound=flatiter)

@final
class flatiter(Generic[_NdArraySubClass]):
    __hash__: ClassVar[None]
    @property
    def base(self) -> _NdArraySubClass: ...
    @property
    def coords(self) -> _Shape: ...
    @property
    def index(self) -> int: ...
    def copy(self) -> _NdArraySubClass: ...
    def __iter__(self: _FlatIterSelf) -> _FlatIterSelf: ...
    def __next__(self: flatiter[ndarray[Any, dtype[_ScalarType]]]) -> _ScalarType: ...
    def __len__(self) -> int: ...
    @overload
    def __getitem__(
        self: flatiter[ndarray[Any, dtype[_ScalarType]]],
        key: int | integer | tuple[int | integer],
    ) -> _ScalarType: ...
    @overload
    def __getitem__(
        self,
        key: _ArrayLikeInt | slice | ellipsis | tuple[_ArrayLikeInt | slice | ellipsis],
    ) -> _NdArraySubClass: ...
    # TODO: `__setitem__` operates via `unsafe` casting rules, and can
    # thus accept any type accepted by the relevant underlying `np.generic`
    # constructor.
    # This means that `value` must in reality be a supertype of `npt.ArrayLike`.
    def __setitem__(
        self,
        key: _ArrayLikeInt | slice | ellipsis | tuple[_ArrayLikeInt | slice | ellipsis],
        value: Any,
    ) -> None: ...
    @overload
    def __array__(self: flatiter[ndarray[Any, _DType]], dtype: None = ..., /) -> ndarray[Any, _DType]: ...
    @overload
    def __array__(self, dtype: _DType, /) -> ndarray[Any, _DType]: ...

_OrderKACF = L[None, "K", "A", "C", "F"]
_OrderACF = L[None, "A", "C", "F"]
_OrderCF = L[None, "C", "F"]

_ModeKind = L["raise", "wrap", "clip"]
_PartitionKind = L["introselect"]
_SortKind = L["quicksort", "mergesort", "heapsort", "stable"]
_SortSide = L["left", "right"]

_ArraySelf = TypeVar("_ArraySelf", bound=_ArrayOrScalarCommon)

class _ArrayOrScalarCommon:
    @property
    def T(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def data(self) -> memoryview: ...
    @property
    def flags(self) -> flagsobj: ...
    @property
    def itemsize(self) -> int: ...
    @property
    def nbytes(self) -> int: ...
    def __bool__(self) -> bool: ...
    def __bytes__(self) -> bytes: ...
    def __str__(self) -> str: ...
    def __repr__(self) -> str: ...
    def __copy__(self: _ArraySelf) -> _ArraySelf: ...
    def __deepcopy__(self: _ArraySelf, memo: None | dict[int, Any], /) -> _ArraySelf: ...

    # TODO: How to deal with the non-commutative nature of `==` and `!=`?
    # xref numpy/numpy#17368
    def __eq__(self, other: Any) -> Any: ...
    def __ne__(self, other: Any) -> Any: ...
    def copy(self: _ArraySelf, order: _OrderKACF = ...) -> _ArraySelf: ...
    def dump(self, file: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _SupportsWrite[bytes]) -> None: ...
    def dumps(self) -> bytes: ...
    def tobytes(self, order: _OrderKACF = ...) -> bytes: ...
    # NOTE: `tostring()` is deprecated and therefore excluded
    # def tostring(self, order=...): ...
    def tofile(
        self,
        fid: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _IOProtocol,
        sep: str = ...,
        format: str = ...,
    ) -> None: ...
    # generics and 0d arrays return builtin scalars
    def tolist(self) -> Any: ...

    @property
    def __array_interface__(self) -> dict[str, Any]: ...
    @property
    def __array_priority__(self) -> float: ...
    @property
    def __array_struct__(self) -> Any: ...  # builtins.PyCapsule
    def __setstate__(self, state: tuple[
        SupportsIndex,  # version
        _ShapeLike,  # Shape
        _DType_co,  # DType
        bool,  # F-continuous
        bytes | list[Any],  # Data
    ], /) -> None: ...
    # a `bool_` is returned when `keepdims=True` and `self` is a 0d array

    @overload
    def all(
        self,
        axis: None = ...,
        out: None = ...,
        keepdims: L[False] = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> bool_: ...
    @overload
    def all(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def all(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def any(
        self,
        axis: None = ...,
        out: None = ...,
        keepdims: L[False] = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> bool_: ...
    @overload
    def any(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def any(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def argmax(
        self,
        axis: None = ...,
        out: None = ...,
        *,
        keepdims: L[False] = ...,
    ) -> intp: ...
    @overload
    def argmax(
        self,
        axis: SupportsIndex = ...,
        out: None = ...,
        *,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def argmax(
        self,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
        *,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def argmin(
        self,
        axis: None = ...,
        out: None = ...,
        *,
        keepdims: L[False] = ...,
    ) -> intp: ...
    @overload
    def argmin(
        self,
        axis: SupportsIndex = ...,
        out: None = ...,
        *,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def argmin(
        self,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
        *,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...

    def argsort(
        self,
        axis: None | SupportsIndex = ...,
        kind: None | _SortKind = ...,
        order: None | str | Sequence[str] = ...,
    ) -> ndarray: ...

    @overload
    def choose(
        self,
        choices: ArrayLike,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> ndarray: ...
    @overload
    def choose(
        self,
        choices: ArrayLike,
        out: _NdArraySubClass = ...,
        mode: _ModeKind = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def clip(
        self,
        min: ArrayLike = ...,
        max: None | ArrayLike = ...,
        out: None = ...,
        **kwargs: Any,
    ) -> ndarray: ...
    @overload
    def clip(
        self,
        min: None = ...,
        max: ArrayLike = ...,
        out: None = ...,
        **kwargs: Any,
    ) -> ndarray: ...
    @overload
    def clip(
        self,
        min: ArrayLike = ...,
        max: None | ArrayLike = ...,
        out: _NdArraySubClass = ...,
        **kwargs: Any,
    ) -> _NdArraySubClass: ...
    @overload
    def clip(
        self,
        min: None = ...,
        max: ArrayLike = ...,
        out: _NdArraySubClass = ...,
        **kwargs: Any,
    ) -> _NdArraySubClass: ...

    @overload
    def compress(
        self,
        a: ArrayLike,
        axis: None | SupportsIndex = ...,
        out: None = ...,
    ) -> ndarray: ...
    @overload
    def compress(
        self,
        a: ArrayLike,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    def conj(self: _ArraySelf) -> _ArraySelf: ...

    def conjugate(self: _ArraySelf) -> _ArraySelf: ...

    @overload
    def cumprod(
        self,
        axis: None | SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> ndarray: ...
    @overload
    def cumprod(
        self,
        axis: None | SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def cumsum(
        self,
        axis: None | SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> ndarray: ...
    @overload
    def cumsum(
        self,
        axis: None | SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def max(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def max(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def mean(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def mean(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def min(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def min(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    def newbyteorder(
        self: _ArraySelf,
        __new_order: _ByteOrder = ...,
    ) -> _ArraySelf: ...

    @overload
    def prod(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def prod(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def ptp(
        self,
        axis: None | _ShapeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
    ) -> Any: ...
    @overload
    def ptp(
        self,
        axis: None | _ShapeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def round(
        self: _ArraySelf,
        decimals: SupportsIndex = ...,
        out: None = ...,
    ) -> _ArraySelf: ...
    @overload
    def round(
        self,
        decimals: SupportsIndex = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def std(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        ddof: float = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def std(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        ddof: float = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def sum(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def sum(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        keepdims: bool = ...,
        initial: _NumberLike_co = ...,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def var(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
        ddof: float = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> Any: ...
    @overload
    def var(
        self,
        axis: None | _ShapeLike = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
        ddof: float = ...,
        keepdims: bool = ...,
        *,
        where: _ArrayLikeBool_co = ...,
    ) -> _NdArraySubClass: ...

_DType = TypeVar("_DType", bound=dtype[Any])
_DType_co = TypeVar("_DType_co", covariant=True, bound=dtype[Any])
_FlexDType = TypeVar("_FlexDType", bound=dtype[flexible])

# TODO: Set the `bound` to something more suitable once we
# have proper shape support
_ShapeType = TypeVar("_ShapeType", bound=Any)
_ShapeType2 = TypeVar("_ShapeType2", bound=Any)
_NumberType = TypeVar("_NumberType", bound=number[Any])

# There is currently no exhaustive way to type the buffer protocol,
# as it is implemented exclusively in the C API (python/typing#593)
_SupportsBuffer = Union[
    bytes,
    bytearray,
    memoryview,
    _array.array[Any],
    mmap.mmap,
    NDArray[Any],
    generic,
]

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_T_contra = TypeVar("_T_contra", contravariant=True)
_2Tuple = tuple[_T, _T]
_CastingKind = L["no", "equiv", "safe", "same_kind", "unsafe"]

_ArrayUInt_co = NDArray[Union[bool_, unsignedinteger[Any]]]
_ArrayInt_co = NDArray[Union[bool_, integer[Any]]]
_ArrayFloat_co = NDArray[Union[bool_, integer[Any], floating[Any]]]
_ArrayComplex_co = NDArray[Union[bool_, integer[Any], floating[Any], complexfloating[Any, Any]]]
_ArrayNumber_co = NDArray[Union[bool_, number[Any]]]
_ArrayTD64_co = NDArray[Union[bool_, integer[Any], timedelta64]]

# Introduce an alias for `dtype` to avoid naming conflicts.
_dtype = dtype

# `builtins.PyCapsule` unfortunately lacks annotations as of the moment;
# use `Any` as a stopgap measure
_PyCapsule = Any

class _SupportsItem(Protocol[_T_co]):
    def item(self, args: Any, /) -> _T_co: ...

class _SupportsReal(Protocol[_T_co]):
    @property
    def real(self) -> _T_co: ...

class _SupportsImag(Protocol[_T_co]):
    @property
    def imag(self) -> _T_co: ...

class ndarray(_ArrayOrScalarCommon, Generic[_ShapeType, _DType_co]):
    __hash__: ClassVar[None]
    @property
    def base(self) -> None | ndarray: ...
    @property
    def ndim(self) -> int: ...
    @property
    def size(self) -> int: ...
    @property
    def real(
        self: ndarray[_ShapeType, dtype[_SupportsReal[_ScalarType]]],  # type: ignore[type-var]
    ) -> ndarray[_ShapeType, _dtype[_ScalarType]]: ...
    @real.setter
    def real(self, value: ArrayLike) -> None: ...
    @property
    def imag(
        self: ndarray[_ShapeType, dtype[_SupportsImag[_ScalarType]]],  # type: ignore[type-var]
    ) -> ndarray[_ShapeType, _dtype[_ScalarType]]: ...
    @imag.setter
    def imag(self, value: ArrayLike) -> None: ...
    def __new__(
        cls: type[_ArraySelf],
        shape: _ShapeLike,
        dtype: DTypeLike = ...,
        buffer: None | _SupportsBuffer = ...,
        offset: SupportsIndex = ...,
        strides: None | _ShapeLike = ...,
        order: _OrderKACF = ...,
    ) -> _ArraySelf: ...

    if sys.version_info >= (3, 9):
        def __class_getitem__(self, item: Any) -> GenericAlias: ...

    @overload
    def __array__(self, dtype: None = ..., /) -> ndarray[Any, _DType_co]: ...
    @overload
    def __array__(self, dtype: _DType, /) -> ndarray[Any, _DType]: ...

    def __array_ufunc__(
        self,
        ufunc: ufunc,
        method: L["__call__", "reduce", "reduceat", "accumulate", "outer", "inner"],
        *inputs: Any,
        **kwargs: Any,
    ) -> Any: ...

    def __array_function__(
        self,
        func: Callable[..., Any],
        types: Iterable[type],
        args: Iterable[Any],
        kwargs: Mapping[str, Any],
    ) -> Any: ...

    # NOTE: In practice any object is accepted by `obj`, but as `__array_finalize__`
    # is a pseudo-abstract method the type has been narrowed down in order to
    # grant subclasses a bit more flexiblity
    def __array_finalize__(self, obj: None | NDArray[Any], /) -> None: ...

    def __array_wrap__(
        self,
        array: ndarray[_ShapeType2, _DType],
        context: None | tuple[ufunc, tuple[Any, ...], int] = ...,
        /,
    ) -> ndarray[_ShapeType2, _DType]: ...

    def __array_prepare__(
        self,
        array: ndarray[_ShapeType2, _DType],
        context: None | tuple[ufunc, tuple[Any, ...], int] = ...,
        /,
    ) -> ndarray[_ShapeType2, _DType]: ...

    @overload
    def __getitem__(self, key: (
        NDArray[integer[Any]]
        | NDArray[bool_]
        | tuple[NDArray[integer[Any]] | NDArray[bool_], ...]
    )) -> ndarray[Any, _DType_co]: ...
    @overload
    def __getitem__(self, key: SupportsIndex | tuple[SupportsIndex, ...]) -> Any: ...
    @overload
    def __getitem__(self, key: (
        None
        | slice
        | ellipsis
        | SupportsIndex
        | _ArrayLikeInt_co
        | tuple[None | slice | ellipsis | _ArrayLikeInt_co | SupportsIndex, ...]
    )) -> ndarray[Any, _DType_co]: ...
    @overload
    def __getitem__(self: NDArray[void], key: str) -> NDArray[Any]: ...
    @overload
    def __getitem__(self: NDArray[void], key: list[str]) -> ndarray[_ShapeType, _dtype[void]]: ...

    @property
    def ctypes(self) -> _ctypes[int]: ...
    @property
    def shape(self) -> _Shape: ...
    @shape.setter
    def shape(self, value: _ShapeLike) -> None: ...
    @property
    def strides(self) -> _Shape: ...
    @strides.setter
    def strides(self, value: _ShapeLike) -> None: ...
    def byteswap(self: _ArraySelf, inplace: bool = ...) -> _ArraySelf: ...
    def fill(self, value: Any) -> None: ...
    @property
    def flat(self: _NdArraySubClass) -> flatiter[_NdArraySubClass]: ...

    # Use the same output type as that of the underlying `generic`
    @overload
    def item(
        self: ndarray[Any, _dtype[_SupportsItem[_T]]],  # type: ignore[type-var]
        *args: SupportsIndex,
    ) -> _T: ...
    @overload
    def item(
        self: ndarray[Any, _dtype[_SupportsItem[_T]]],  # type: ignore[type-var]
        args: tuple[SupportsIndex, ...],
        /,
    ) -> _T: ...

    @overload
    def itemset(self, value: Any, /) -> None: ...
    @overload
    def itemset(self, item: _ShapeLike, value: Any, /) -> None: ...

    @overload
    def resize(self, new_shape: _ShapeLike, /, *, refcheck: bool = ...) -> None: ...
    @overload
    def resize(self, *new_shape: SupportsIndex, refcheck: bool = ...) -> None: ...

    def setflags(
        self, write: bool = ..., align: bool = ..., uic: bool = ...
    ) -> None: ...

    def squeeze(
        self,
        axis: None | SupportsIndex | tuple[SupportsIndex, ...] = ...,
    ) -> ndarray[Any, _DType_co]: ...

    def swapaxes(
        self,
        axis1: SupportsIndex,
        axis2: SupportsIndex,
    ) -> ndarray[Any, _DType_co]: ...

    @overload
    def transpose(self: _ArraySelf, axes: None | _ShapeLike, /) -> _ArraySelf: ...
    @overload
    def transpose(self: _ArraySelf, *axes: SupportsIndex) -> _ArraySelf: ...

    def argpartition(
        self,
        kth: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
        kind: _PartitionKind = ...,
        order: None | str | Sequence[str] = ...,
    ) -> ndarray[Any, _dtype[intp]]: ...

    def diagonal(
        self,
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
    ) -> ndarray[Any, _DType_co]: ...

    # 1D + 1D returns a scalar;
    # all other with at least 1 non-0D array return an ndarray.
    @overload
    def dot(self, b: _ScalarLike_co, out: None = ...) -> ndarray: ...
    @overload
    def dot(self, b: ArrayLike, out: None = ...) -> Any: ...  # type: ignore[misc]
    @overload
    def dot(self, b: ArrayLike, out: _NdArraySubClass) -> _NdArraySubClass: ...

    # `nonzero()` is deprecated for 0d arrays/generics
    def nonzero(self) -> tuple[ndarray[Any, _dtype[intp]], ...]: ...

    def partition(
        self,
        kth: _ArrayLikeInt_co,
        axis: SupportsIndex = ...,
        kind: _PartitionKind = ...,
        order: None | str | Sequence[str] = ...,
    ) -> None: ...

    # `put` is technically available to `generic`,
    # but is pointless as `generic`s are immutable
    def put(
        self,
        ind: _ArrayLikeInt_co,
        v: ArrayLike,
        mode: _ModeKind = ...,
    ) -> None: ...

    @overload
    def searchsorted(  # type: ignore[misc]
        self,  # >= 1D array
        v: _ScalarLike_co,  # 0D array-like
        side: _SortSide = ...,
        sorter: None | _ArrayLikeInt_co = ...,
    ) -> intp: ...
    @overload
    def searchsorted(
        self,  # >= 1D array
        v: ArrayLike,
        side: _SortSide = ...,
        sorter: None | _ArrayLikeInt_co = ...,
    ) -> ndarray[Any, _dtype[intp]]: ...

    def setfield(
        self,
        val: ArrayLike,
        dtype: DTypeLike,
        offset: SupportsIndex = ...,
    ) -> None: ...

    def sort(
        self,
        axis: SupportsIndex = ...,
        kind: None | _SortKind = ...,
        order: None | str | Sequence[str] = ...,
    ) -> None: ...

    @overload
    def trace(
        self,  # >= 2D array
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: None = ...,
    ) -> Any: ...
    @overload
    def trace(
        self,  # >= 2D array
        offset: SupportsIndex = ...,
        axis1: SupportsIndex = ...,
        axis2: SupportsIndex = ...,
        dtype: DTypeLike = ...,
        out: _NdArraySubClass = ...,
    ) -> _NdArraySubClass: ...

    @overload
    def take(  # type: ignore[misc]
        self: ndarray[Any, _dtype[_ScalarType]],
        indices: _IntLike_co,
        axis: None | SupportsIndex = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> _ScalarType: ...
    @overload
    def take(  # type: ignore[misc]
        self,
        indices: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> ndarray[Any, _DType_co]: ...
    @overload
    def take(
        self,
        indices: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
        mode: _ModeKind = ...,
    ) -> _NdArraySubClass: ...

    def repeat(
        self,
        repeats: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
    ) -> ndarray[Any, _DType_co]: ...

    def flatten(
        self,
        order: _OrderKACF = ...,
    ) -> ndarray[Any, _DType_co]: ...

    def ravel(
        self,
        order: _OrderKACF = ...,
    ) -> ndarray[Any, _DType_co]: ...

    @overload
    def reshape(
        self, shape: _ShapeLike, /, *, order: _OrderACF = ...
    ) -> ndarray[Any, _DType_co]: ...
    @overload
    def reshape(
        self, *shape: SupportsIndex, order: _OrderACF = ...
    ) -> ndarray[Any, _DType_co]: ...

    @overload
    def astype(
        self,
        dtype: _DTypeLike[_ScalarType],
        order: _OrderKACF = ...,
        casting: _CastingKind = ...,
        subok: bool = ...,
        copy: bool | _CopyMode = ...,
    ) -> NDArray[_ScalarType]: ...
    @overload
    def astype(
        self,
        dtype: DTypeLike,
        order: _OrderKACF = ...,
        casting: _CastingKind = ...,
        subok: bool = ...,
        copy: bool | _CopyMode = ...,
    ) -> NDArray[Any]: ...

    @overload
    def view(self: _ArraySelf) -> _ArraySelf: ...
    @overload
    def view(self, type: type[_NdArraySubClass]) -> _NdArraySubClass: ...
    @overload
    def view(self, dtype: _DTypeLike[_ScalarType]) -> NDArray[_ScalarType]: ...
    @overload
    def view(self, dtype: DTypeLike) -> NDArray[Any]: ...
    @overload
    def view(
        self,
        dtype: DTypeLike,
        type: type[_NdArraySubClass],
    ) -> _NdArraySubClass: ...

    @overload
    def getfield(
        self,
        dtype: _DTypeLike[_ScalarType],
        offset: SupportsIndex = ...
    ) -> NDArray[_ScalarType]: ...
    @overload
    def getfield(
        self,
        dtype: DTypeLike,
        offset: SupportsIndex = ...
    ) -> NDArray[Any]: ...

    # Dispatch to the underlying `generic` via protocols
    def __int__(
        self: ndarray[Any, _dtype[SupportsInt]],  # type: ignore[type-var]
    ) -> int: ...

    def __float__(
        self: ndarray[Any, _dtype[SupportsFloat]],  # type: ignore[type-var]
    ) -> float: ...

    def __complex__(
        self: ndarray[Any, _dtype[SupportsComplex]],  # type: ignore[type-var]
    ) -> complex: ...

    def __index__(
        self: ndarray[Any, _dtype[SupportsIndex]],  # type: ignore[type-var]
    ) -> int: ...

    def __len__(self) -> int: ...
    def __setitem__(self, key, value): ...
    def __iter__(self) -> Any: ...
    def __contains__(self, key) -> bool: ...

    # The last overload is for catching recursive objects whose
    # nesting is too deep.
    # The first overload is for catching `bytes` (as they are a subtype of
    # `Sequence[int]`) and `str`. As `str` is a recursive sequence of
    # strings, it will pass through the final overload otherwise

    @overload
    def __lt__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> NDArray[bool_]: ...
    @overload
    def __lt__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[bool_]: ...
    @overload
    def __lt__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[bool_]: ...
    @overload
    def __lt__(self: NDArray[object_], other: Any) -> NDArray[bool_]: ...
    @overload
    def __lt__(self: NDArray[Any], other: _ArrayLikeObject_co) -> NDArray[bool_]: ...

    @overload
    def __le__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> NDArray[bool_]: ...
    @overload
    def __le__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[bool_]: ...
    @overload
    def __le__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[bool_]: ...
    @overload
    def __le__(self: NDArray[object_], other: Any) -> NDArray[bool_]: ...
    @overload
    def __le__(self: NDArray[Any], other: _ArrayLikeObject_co) -> NDArray[bool_]: ...

    @overload
    def __gt__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> NDArray[bool_]: ...
    @overload
    def __gt__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[bool_]: ...
    @overload
    def __gt__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[bool_]: ...
    @overload
    def __gt__(self: NDArray[object_], other: Any) -> NDArray[bool_]: ...
    @overload
    def __gt__(self: NDArray[Any], other: _ArrayLikeObject_co) -> NDArray[bool_]: ...

    @overload
    def __ge__(self: _ArrayNumber_co, other: _ArrayLikeNumber_co) -> NDArray[bool_]: ...
    @overload
    def __ge__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[bool_]: ...
    @overload
    def __ge__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[bool_]: ...
    @overload
    def __ge__(self: NDArray[object_], other: Any) -> NDArray[bool_]: ...
    @overload
    def __ge__(self: NDArray[Any], other: _ArrayLikeObject_co) -> NDArray[bool_]: ...

    # Unary ops
    @overload
    def __abs__(self: NDArray[bool_]) -> NDArray[bool_]: ...
    @overload
    def __abs__(self: NDArray[complexfloating[_NBit1, _NBit1]]) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __abs__(self: NDArray[_NumberType]) -> NDArray[_NumberType]: ...
    @overload
    def __abs__(self: NDArray[timedelta64]) -> NDArray[timedelta64]: ...
    @overload
    def __abs__(self: NDArray[object_]) -> Any: ...

    @overload
    def __invert__(self: NDArray[bool_]) -> NDArray[bool_]: ...
    @overload
    def __invert__(self: NDArray[_IntType]) -> NDArray[_IntType]: ...
    @overload
    def __invert__(self: NDArray[object_]) -> Any: ...

    @overload
    def __pos__(self: NDArray[_NumberType]) -> NDArray[_NumberType]: ...
    @overload
    def __pos__(self: NDArray[timedelta64]) -> NDArray[timedelta64]: ...
    @overload
    def __pos__(self: NDArray[object_]) -> Any: ...

    @overload
    def __neg__(self: NDArray[_NumberType]) -> NDArray[_NumberType]: ...
    @overload
    def __neg__(self: NDArray[timedelta64]) -> NDArray[timedelta64]: ...
    @overload
    def __neg__(self: NDArray[object_]) -> Any: ...

    # Binary ops
    # NOTE: `ndarray` does not implement `__imatmul__`
    @overload
    def __matmul__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __matmul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __matmul__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __matmul__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __matmul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rmatmul__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmatmul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __rmatmul__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __rmatmul__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rmatmul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __mod__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __mod__(self: _ArrayTD64_co, other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[timedelta64]: ...
    @overload
    def __mod__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __mod__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rmod__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmod__(self: _ArrayTD64_co, other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[timedelta64]: ...
    @overload
    def __rmod__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rmod__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __divmod__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> _2Tuple[NDArray[int8]]: ...  # type: ignore[misc]
    @overload
    def __divmod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _2Tuple[NDArray[unsignedinteger[Any]]]: ...  # type: ignore[misc]
    @overload
    def __divmod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _2Tuple[NDArray[signedinteger[Any]]]: ...  # type: ignore[misc]
    @overload
    def __divmod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _2Tuple[NDArray[floating[Any]]]: ...  # type: ignore[misc]
    @overload
    def __divmod__(self: _ArrayTD64_co, other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> tuple[NDArray[int64], NDArray[timedelta64]]: ...

    @overload
    def __rdivmod__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> _2Tuple[NDArray[int8]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> _2Tuple[NDArray[unsignedinteger[Any]]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> _2Tuple[NDArray[signedinteger[Any]]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> _2Tuple[NDArray[floating[Any]]]: ...  # type: ignore[misc]
    @overload
    def __rdivmod__(self: _ArrayTD64_co, other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> tuple[NDArray[int64], NDArray[timedelta64]]: ...

    @overload
    def __add__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __add__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __add__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...  # type: ignore[misc]
    @overload
    def __add__(self: _ArrayTD64_co, other: _ArrayLikeDT64_co) -> NDArray[datetime64]: ...
    @overload
    def __add__(self: NDArray[datetime64], other: _ArrayLikeTD64_co) -> NDArray[datetime64]: ...
    @overload
    def __add__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __add__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __radd__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __radd__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...  # type: ignore[misc]
    @overload
    def __radd__(self: _ArrayTD64_co, other: _ArrayLikeDT64_co) -> NDArray[datetime64]: ...
    @overload
    def __radd__(self: NDArray[datetime64], other: _ArrayLikeTD64_co) -> NDArray[datetime64]: ...
    @overload
    def __radd__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __radd__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __sub__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    @overload
    def __sub__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __sub__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __sub__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...  # type: ignore[misc]
    @overload
    def __sub__(self: NDArray[datetime64], other: _ArrayLikeTD64_co) -> NDArray[datetime64]: ...
    @overload
    def __sub__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[timedelta64]: ...
    @overload
    def __sub__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __sub__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rsub__(self: NDArray[_UnknownType], other: _ArrayLikeUnknown) -> NDArray[Any]: ...
    @overload
    def __rsub__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __rsub__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __rsub__(self: _ArrayTD64_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: _ArrayTD64_co, other: _ArrayLikeDT64_co) -> NDArray[datetime64]: ...  # type: ignore[misc]
    @overload
    def __rsub__(self: NDArray[datetime64], other: _ArrayLikeDT64_co) -> NDArray[timedelta64]: ...
    @overload
    def __rsub__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rsub__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __mul__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __mul__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __mul__(self: _ArrayTD64_co, other: _ArrayLikeFloat_co) -> NDArray[timedelta64]: ...
    @overload
    def __mul__(self: _ArrayFloat_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    @overload
    def __mul__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __mul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rmul__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __rmul__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __rmul__(self: _ArrayTD64_co, other: _ArrayLikeFloat_co) -> NDArray[timedelta64]: ...
    @overload
    def __rmul__(self: _ArrayFloat_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    @overload
    def __rmul__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rmul__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __floordiv__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __floordiv__(self: NDArray[timedelta64], other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[int64]: ...
    @overload
    def __floordiv__(self: NDArray[timedelta64], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __floordiv__(self: NDArray[timedelta64], other: _ArrayLikeFloat_co) -> NDArray[timedelta64]: ...
    @overload
    def __floordiv__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __floordiv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rfloordiv__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rfloordiv__(self: NDArray[timedelta64], other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[int64]: ...
    @overload
    def __rfloordiv__(self: NDArray[bool_], other: _ArrayLikeTD64_co) -> NoReturn: ...
    @overload
    def __rfloordiv__(self: _ArrayFloat_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    @overload
    def __rfloordiv__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rfloordiv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __pow__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __pow__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __pow__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __pow__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __pow__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __pow__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __pow__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __pow__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rpow__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __rpow__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rpow__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rpow__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rpow__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...
    @overload
    def __rpow__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __rpow__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rpow__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __truediv__(self: _ArrayInt_co, other: _ArrayInt_co) -> NDArray[float64]: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __truediv__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __truediv__(self: NDArray[timedelta64], other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[float64]: ...
    @overload
    def __truediv__(self: NDArray[timedelta64], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __truediv__(self: NDArray[timedelta64], other: _ArrayLikeFloat_co) -> NDArray[timedelta64]: ...
    @overload
    def __truediv__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __truediv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rtruediv__(self: _ArrayInt_co, other: _ArrayInt_co) -> NDArray[float64]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: _ArrayFloat_co, other: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: _ArrayComplex_co, other: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...  # type: ignore[misc]
    @overload
    def __rtruediv__(self: NDArray[number[Any]], other: _ArrayLikeNumber_co) -> NDArray[number[Any]]: ...
    @overload
    def __rtruediv__(self: NDArray[timedelta64], other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[float64]: ...
    @overload
    def __rtruediv__(self: NDArray[bool_], other: _ArrayLikeTD64_co) -> NoReturn: ...
    @overload
    def __rtruediv__(self: _ArrayFloat_co, other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    @overload
    def __rtruediv__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rtruediv__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __lshift__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __lshift__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __lshift__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __lshift__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __lshift__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rlshift__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __rlshift__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rlshift__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rlshift__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rlshift__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rshift__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __rshift__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rshift__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rshift__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rshift__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rrshift__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[int8]: ...  # type: ignore[misc]
    @overload
    def __rrshift__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rrshift__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rrshift__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rrshift__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __and__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __and__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __and__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __and__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __and__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rand__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __rand__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rand__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rand__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rand__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __xor__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __xor__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __xor__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __xor__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __xor__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __rxor__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __rxor__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __rxor__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __rxor__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __rxor__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __or__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __or__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __or__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __or__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __or__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    @overload
    def __ror__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...  # type: ignore[misc]
    @overload
    def __ror__(self: _ArrayUInt_co, other: _ArrayLikeUInt_co) -> NDArray[unsignedinteger[Any]]: ...  # type: ignore[misc]
    @overload
    def __ror__(self: _ArrayInt_co, other: _ArrayLikeInt_co) -> NDArray[signedinteger[Any]]: ...
    @overload
    def __ror__(self: NDArray[object_], other: Any) -> Any: ...
    @overload
    def __ror__(self: NDArray[Any], other: _ArrayLikeObject_co) -> Any: ...

    # `np.generic` does not support inplace operations

    # NOTE: Inplace ops generally use "same_kind" casting w.r.t. to the left
    # operand. An exception to this rule are unsigned integers though, which
    # also accepts a signed integer for the right operand as long it is a 0D
    # object and its value is >= 0
    @overload
    def __iadd__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __iadd__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __iadd__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __iadd__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __iadd__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __iadd__(self: NDArray[timedelta64], other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    @overload
    def __iadd__(self: NDArray[datetime64], other: _ArrayLikeTD64_co) -> NDArray[datetime64]: ...
    @overload
    def __iadd__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __isub__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __isub__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __isub__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __isub__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __isub__(self: NDArray[timedelta64], other: _ArrayLikeTD64_co) -> NDArray[timedelta64]: ...
    @overload
    def __isub__(self: NDArray[datetime64], other: _ArrayLikeTD64_co) -> NDArray[datetime64]: ...
    @overload
    def __isub__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __imul__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __imul__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __imul__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __imul__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __imul__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __imul__(self: NDArray[timedelta64], other: _ArrayLikeFloat_co) -> NDArray[timedelta64]: ...
    @overload
    def __imul__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __itruediv__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __itruediv__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __itruediv__(self: NDArray[timedelta64], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __itruediv__(self: NDArray[timedelta64], other: _ArrayLikeInt_co) -> NDArray[timedelta64]: ...
    @overload
    def __itruediv__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __ifloordiv__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __ifloordiv__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __ifloordiv__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __ifloordiv__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __ifloordiv__(self: NDArray[timedelta64], other: _ArrayLikeBool_co) -> NoReturn: ...
    @overload
    def __ifloordiv__(self: NDArray[timedelta64], other: _ArrayLikeInt_co) -> NDArray[timedelta64]: ...
    @overload
    def __ifloordiv__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __ipow__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __ipow__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __ipow__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __ipow__(self: NDArray[complexfloating[_NBit1, _NBit1]], other: _ArrayLikeComplex_co) -> NDArray[complexfloating[_NBit1, _NBit1]]: ...
    @overload
    def __ipow__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __imod__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __imod__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __imod__(self: NDArray[floating[_NBit1]], other: _ArrayLikeFloat_co) -> NDArray[floating[_NBit1]]: ...
    @overload
    def __imod__(self: NDArray[timedelta64], other: _SupportsArray[_dtype[timedelta64]] | _NestedSequence[_SupportsArray[_dtype[timedelta64]]]) -> NDArray[timedelta64]: ...
    @overload
    def __imod__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __ilshift__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __ilshift__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __ilshift__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __irshift__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __irshift__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __irshift__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __iand__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __iand__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __iand__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __iand__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __ixor__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __ixor__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __ixor__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __ixor__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    @overload
    def __ior__(self: NDArray[bool_], other: _ArrayLikeBool_co) -> NDArray[bool_]: ...
    @overload
    def __ior__(self: NDArray[unsignedinteger[_NBit1]], other: _ArrayLikeUInt_co | _IntLike_co) -> NDArray[unsignedinteger[_NBit1]]: ...
    @overload
    def __ior__(self: NDArray[signedinteger[_NBit1]], other: _ArrayLikeInt_co) -> NDArray[signedinteger[_NBit1]]: ...
    @overload
    def __ior__(self: NDArray[object_], other: Any) -> NDArray[object_]: ...

    def __dlpack__(self: NDArray[number[Any]], *, stream: None = ...) -> _PyCapsule: ...
    def __dlpack_device__(self) -> tuple[int, L[0]]: ...

    # Keep `dtype` at the bottom to avoid name conflicts with `np.dtype`
    @property
    def dtype(self) -> _DType_co: ...

# NOTE: while `np.generic` is not technically an instance of `ABCMeta`,
# the `@abstractmethod` decorator is herein used to (forcefully) deny
# the creation of `np.generic` instances.
# The `# type: ignore` comments are necessary to silence mypy errors regarding
# the missing `ABCMeta` metaclass.

# See https://github.com/numpy/numpy-stubs/pull/80 for more details.

_ScalarType = TypeVar("_ScalarType", bound=generic)
_NBit1 = TypeVar("_NBit1", bound=NBitBase)
_NBit2 = TypeVar("_NBit2", bound=NBitBase)

class generic(_ArrayOrScalarCommon):
    @abstractmethod
    def __init__(self, *args: Any, **kwargs: Any) -> None: ...
    @overload
    def __array__(self: _ScalarType, dtype: None = ..., /) -> ndarray[Any, _dtype[_ScalarType]]: ...
    @overload
    def __array__(self, dtype: _DType, /) -> ndarray[Any, _DType]: ...
    @property
    def base(self) -> None: ...
    @property
    def ndim(self) -> L[0]: ...
    @property
    def size(self) -> L[1]: ...
    @property
    def shape(self) -> tuple[()]: ...
    @property
    def strides(self) -> tuple[()]: ...
    def byteswap(self: _ScalarType, inplace: L[False] = ...) -> _ScalarType: ...
    @property
    def flat(self: _ScalarType) -> flatiter[ndarray[Any, _dtype[_ScalarType]]]: ...

    @overload
    def astype(
        self,
        dtype: _DTypeLike[_ScalarType],
        order: _OrderKACF = ...,
        casting: _CastingKind = ...,
        subok: bool = ...,
        copy: bool | _CopyMode = ...,
    ) -> _ScalarType: ...
    @overload
    def astype(
        self,
        dtype: DTypeLike,
        order: _OrderKACF = ...,
        casting: _CastingKind = ...,
        subok: bool = ...,
        copy: bool | _CopyMode = ...,
    ) -> Any: ...

    # NOTE: `view` will perform a 0D->scalar cast,
    # thus the array `type` is irrelevant to the output type
    @overload
    def view(
        self: _ScalarType,
        type: type[ndarray[Any, Any]] = ...,
    ) -> _ScalarType: ...
    @overload
    def view(
        self,
        dtype: _DTypeLike[_ScalarType],
        type: type[ndarray[Any, Any]] = ...,
    ) -> _ScalarType: ...
    @overload
    def view(
        self,
        dtype: DTypeLike,
        type: type[ndarray[Any, Any]] = ...,
    ) -> Any: ...

    @overload
    def getfield(
        self,
        dtype: _DTypeLike[_ScalarType],
        offset: SupportsIndex = ...
    ) -> _ScalarType: ...
    @overload
    def getfield(
        self,
        dtype: DTypeLike,
        offset: SupportsIndex = ...
    ) -> Any: ...

    def item(
        self, args: L[0] | tuple[()] | tuple[L[0]] = ..., /,
    ) -> Any: ...

    @overload
    def take(  # type: ignore[misc]
        self: _ScalarType,
        indices: _IntLike_co,
        axis: None | SupportsIndex = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> _ScalarType: ...
    @overload
    def take(  # type: ignore[misc]
        self: _ScalarType,
        indices: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
        out: None = ...,
        mode: _ModeKind = ...,
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...
    @overload
    def take(
        self,
        indices: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
        out: _NdArraySubClass = ...,
        mode: _ModeKind = ...,
    ) -> _NdArraySubClass: ...

    def repeat(
        self: _ScalarType,
        repeats: _ArrayLikeInt_co,
        axis: None | SupportsIndex = ...,
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...

    def flatten(
        self: _ScalarType,
        order: _OrderKACF = ...,
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...

    def ravel(
        self: _ScalarType,
        order: _OrderKACF = ...,
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...

    @overload
    def reshape(
        self: _ScalarType, shape: _ShapeLike, /, *, order: _OrderACF = ...
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...
    @overload
    def reshape(
        self: _ScalarType, *shape: SupportsIndex, order: _OrderACF = ...
    ) -> ndarray[Any, _dtype[_ScalarType]]: ...

    def squeeze(
        self: _ScalarType, axis: None | L[0] | tuple[()] = ...
    ) -> _ScalarType: ...
    def transpose(self: _ScalarType, axes: None | tuple[()] = ..., /) -> _ScalarType: ...
    # Keep `dtype` at the bottom to avoid name conflicts with `np.dtype`
    @property
    def dtype(self: _ScalarType) -> _dtype[_ScalarType]: ...

class number(generic, Generic[_NBit1]):  # type: ignore
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    if sys.version_info >= (3, 9):
        def __class_getitem__(self, item: Any) -> GenericAlias: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __neg__(self: _ArraySelf) -> _ArraySelf: ...
    def __pos__(self: _ArraySelf) -> _ArraySelf: ...
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...
    # Ensure that objects annotated as `number` support arithmetic operations
    __add__: _NumberOp
    __radd__: _NumberOp
    __sub__: _NumberOp
    __rsub__: _NumberOp
    __mul__: _NumberOp
    __rmul__: _NumberOp
    __floordiv__: _NumberOp
    __rfloordiv__: _NumberOp
    __pow__: _NumberOp
    __rpow__: _NumberOp
    __truediv__: _NumberOp
    __rtruediv__: _NumberOp
    __lt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __le__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __gt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __ge__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]

class bool_(generic):
    def __init__(self, value: object = ..., /) -> None: ...
    def item(
        self, args: L[0] | tuple[()] | tuple[L[0]] = ..., /,
    ) -> bool: ...
    def tolist(self) -> bool: ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...
    __add__: _BoolOp[bool_]
    __radd__: _BoolOp[bool_]
    __sub__: _BoolSub
    __rsub__: _BoolSub
    __mul__: _BoolOp[bool_]
    __rmul__: _BoolOp[bool_]
    __floordiv__: _BoolOp[int8]
    __rfloordiv__: _BoolOp[int8]
    __pow__: _BoolOp[int8]
    __rpow__: _BoolOp[int8]
    __truediv__: _BoolTrueDiv
    __rtruediv__: _BoolTrueDiv
    def __invert__(self) -> bool_: ...
    __lshift__: _BoolBitOp[int8]
    __rlshift__: _BoolBitOp[int8]
    __rshift__: _BoolBitOp[int8]
    __rrshift__: _BoolBitOp[int8]
    __and__: _BoolBitOp[bool_]
    __rand__: _BoolBitOp[bool_]
    __xor__: _BoolBitOp[bool_]
    __rxor__: _BoolBitOp[bool_]
    __or__: _BoolBitOp[bool_]
    __ror__: _BoolBitOp[bool_]
    __mod__: _BoolMod
    __rmod__: _BoolMod
    __divmod__: _BoolDivMod
    __rdivmod__: _BoolDivMod
    __lt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __le__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __gt__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]
    __ge__: _ComparisonOp[_NumberLike_co, _ArrayLikeNumber_co]

class object_(generic):
    def __init__(self, value: object = ..., /) -> None: ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    # The 3 protocols below may or may not raise,
    # depending on the underlying object
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...

# The `datetime64` constructors requires an object with the three attributes below,
# and thus supports datetime duck typing
class _DatetimeScalar(Protocol):
    @property
    def day(self) -> int: ...
    @property
    def month(self) -> int: ...
    @property
    def year(self) -> int: ...

# TODO: `item`/`tolist` returns either `dt.date`, `dt.datetime` or `int`
# depending on the unit
class datetime64(generic):
    @overload
    def __init__(
        self,
        value: None | datetime64 | _CharLike_co | _DatetimeScalar = ...,
        format: _CharLike_co | tuple[_CharLike_co, _IntLike_co] = ...,
        /,
    ) -> None: ...
    @overload
    def __init__(
        self,
        value: int,
        format: _CharLike_co | tuple[_CharLike_co, _IntLike_co],
        /,
    ) -> None: ...
    def __add__(self, other: _TD64Like_co) -> datetime64: ...
    def __radd__(self, other: _TD64Like_co) -> datetime64: ...
    @overload
    def __sub__(self, other: datetime64) -> timedelta64: ...
    @overload
    def __sub__(self, other: _TD64Like_co) -> datetime64: ...
    def __rsub__(self, other: datetime64) -> timedelta64: ...
    __lt__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]
    __le__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]
    __gt__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]
    __ge__: _ComparisonOp[datetime64, _ArrayLikeDT64_co]

_IntValue = Union[SupportsInt, _CharLike_co, SupportsIndex]
_FloatValue = Union[None, _CharLike_co, SupportsFloat, SupportsIndex]
_ComplexValue = Union[
    None,
    _CharLike_co,
    SupportsFloat,
    SupportsComplex,
    SupportsIndex,
    complex,  # `complex` is not a subtype of `SupportsComplex`
]

class integer(number[_NBit1]):  # type: ignore
    @property
    def numerator(self: _ScalarType) -> _ScalarType: ...
    @property
    def denominator(self) -> L[1]: ...
    @overload
    def __round__(self, ndigits: None = ...) -> int: ...
    @overload
    def __round__(self: _ScalarType, ndigits: SupportsIndex) -> _ScalarType: ...

    # NOTE: `__index__` is technically defined in the bottom-most
    # sub-classes (`int64`, `uint32`, etc)
    def item(
        self, args: L[0] | tuple[()] | tuple[L[0]] = ..., /,
    ) -> int: ...
    def tolist(self) -> int: ...
    def is_integer(self) -> L[True]: ...
    def bit_count(self: _ScalarType) -> int: ...
    def __index__(self) -> int: ...
    __truediv__: _IntTrueDiv[_NBit1]
    __rtruediv__: _IntTrueDiv[_NBit1]
    def __mod__(self, value: _IntLike_co) -> integer: ...
    def __rmod__(self, value: _IntLike_co) -> integer: ...
    def __invert__(self: _IntType) -> _IntType: ...
    # Ensure that objects annotated as `integer` support bit-wise operations
    def __lshift__(self, other: _IntLike_co) -> integer: ...
    def __rlshift__(self, other: _IntLike_co) -> integer: ...
    def __rshift__(self, other: _IntLike_co) -> integer: ...
    def __rrshift__(self, other: _IntLike_co) -> integer: ...
    def __and__(self, other: _IntLike_co) -> integer: ...
    def __rand__(self, other: _IntLike_co) -> integer: ...
    def __or__(self, other: _IntLike_co) -> integer: ...
    def __ror__(self, other: _IntLike_co) -> integer: ...
    def __xor__(self, other: _IntLike_co) -> integer: ...
    def __rxor__(self, other: _IntLike_co) -> integer: ...

class signedinteger(integer[_NBit1]):
    def __init__(self, value: _IntValue = ..., /) -> None: ...
    __add__: _SignedIntOp[_NBit1]
    __radd__: _SignedIntOp[_NBit1]
    __sub__: _SignedIntOp[_NBit1]
    __rsub__: _SignedIntOp[_NBit1]
    __mul__: _SignedIntOp[_NBit1]
    __rmul__: _SignedIntOp[_NBit1]
    __floordiv__: _SignedIntOp[_NBit1]
    __rfloordiv__: _SignedIntOp[_NBit1]
    __pow__: _SignedIntOp[_NBit1]
    __rpow__: _SignedIntOp[_NBit1]
    __lshift__: _SignedIntBitOp[_NBit1]
    __rlshift__: _SignedIntBitOp[_NBit1]
    __rshift__: _SignedIntBitOp[_NBit1]
    __rrshift__: _SignedIntBitOp[_NBit1]
    __and__: _SignedIntBitOp[_NBit1]
    __rand__: _SignedIntBitOp[_NBit1]
    __xor__: _SignedIntBitOp[_NBit1]
    __rxor__: _SignedIntBitOp[_NBit1]
    __or__: _SignedIntBitOp[_NBit1]
    __ror__: _SignedIntBitOp[_NBit1]
    __mod__: _SignedIntMod[_NBit1]
    __rmod__: _SignedIntMod[_NBit1]
    __divmod__: _SignedIntDivMod[_NBit1]
    __rdivmod__: _SignedIntDivMod[_NBit1]

int8 = signedinteger[_8Bit]
int16 = signedinteger[_16Bit]
int32 = signedinteger[_32Bit]
int64 = signedinteger[_64Bit]

byte = signedinteger[_NBitByte]
short = signedinteger[_NBitShort]
intc = signedinteger[_NBitIntC]
intp = signedinteger[_NBitIntP]
int_ = signedinteger[_NBitInt]
longlong = signedinteger[_NBitLongLong]

# TODO: `item`/`tolist` returns either `dt.timedelta` or `int`
# depending on the unit
class timedelta64(generic):
    def __init__(
        self,
        value: None | int | _CharLike_co | dt.timedelta | timedelta64 = ...,
        format: _CharLike_co | tuple[_CharLike_co, _IntLike_co] = ...,
        /,
    ) -> None: ...
    @property
    def numerator(self: _ScalarType) -> _ScalarType: ...
    @property
    def denominator(self) -> L[1]: ...

    # NOTE: Only a limited number of units support conversion
    # to builtin scalar types: `Y`, `M`, `ns`, `ps`, `fs`, `as`
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...
    def __complex__(self) -> complex: ...
    def __neg__(self: _ArraySelf) -> _ArraySelf: ...
    def __pos__(self: _ArraySelf) -> _ArraySelf: ...
    def __abs__(self: _ArraySelf) -> _ArraySelf: ...
    def __add__(self, other: _TD64Like_co) -> timedelta64: ...
    def __radd__(self, other: _TD64Like_co) -> timedelta64: ...
    def __sub__(self, other: _TD64Like_co) -> timedelta64: ...
    def __rsub__(self, other: _TD64Like_co) -> timedelta64: ...
    def __mul__(self, other: _FloatLike_co) -> timedelta64: ...
    def __rmul__(self, other: _FloatLike_co) -> timedelta64: ...
    __truediv__: _TD64Div[float64]
    __floordiv__: _TD64Div[int64]
    def __rtruediv__(self, other: timedelta64) -> float64: ...
    def __rfloordiv__(self, other: timedelta64) -> int64: ...
    def __mod__(self, other: timedelta64) -> timedelta64: ...
    def __rmod__(self, other: timedelta64) -> timedelta64: ...
    def __divmod__(self, other: timedelta64) -> tuple[int64, timedelta64]: ...
    def __rdivmod__(self, other: timedelta64) -> tuple[int64, timedelta64]: ...
    __lt__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]
    __le__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]
    __gt__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]
    __ge__: _ComparisonOp[_TD64Like_co, _ArrayLikeTD64_co]

class unsignedinteger(integer[_NBit1]):
    # NOTE: `uint64 + signedinteger -> float64`
    def __init__(self, value: _IntValue = ..., /) -> None: ...
    __add__: _UnsignedIntOp[_NBit1]
    __radd__: _UnsignedIntOp[_NBit1]
    __sub__: _UnsignedIntOp[_NBit1]
    __rsub__: _UnsignedIntOp[_NBit1]
    __mul__: _UnsignedIntOp[_NBit1]
    __rmul__: _UnsignedIntOp[_NBit1]
    __floordiv__: _UnsignedIntOp[_NBit1]
    __rfloordiv__: _UnsignedIntOp[_NBit1]
    __pow__: _UnsignedIntOp[_NBit1]
    __rpow__: _UnsignedIntOp[_NBit1]
    __lshift__: _UnsignedIntBitOp[_NBit1]
    __rlshift__: _UnsignedIntBitOp[_NBit1]
    __rshift__: _UnsignedIntBitOp[_NBit1]
    __rrshift__: _UnsignedIntBitOp[_NBit1]
    __and__: _UnsignedIntBitOp[_NBit1]
    __rand__: _UnsignedIntBitOp[_NBit1]
    __xor__: _UnsignedIntBitOp[_NBit1]
    __rxor__: _UnsignedIntBitOp[_NBit1]
    __or__: _UnsignedIntBitOp[_NBit1]
    __ror__: _UnsignedIntBitOp[_NBit1]
    __mod__: _UnsignedIntMod[_NBit1]
    __rmod__: _UnsignedIntMod[_NBit1]
    __divmod__: _UnsignedIntDivMod[_NBit1]
    __rdivmod__: _UnsignedIntDivMod[_NBit1]

uint8 = unsignedinteger[_8Bit]
uint16 = unsignedinteger[_16Bit]
uint32 = unsignedinteger[_32Bit]
uint64 = unsignedinteger[_64Bit]

ubyte = unsignedinteger[_NBitByte]
ushort = unsignedinteger[_NBitShort]
uintc = unsignedinteger[_NBitIntC]
uintp = unsignedinteger[_NBitIntP]
uint = unsignedinteger[_NBitInt]
ulonglong = unsignedinteger[_NBitLongLong]

class inexact(number[_NBit1]):  # type: ignore
    def __getnewargs__(self: inexact[_64Bit]) -> tuple[float, ...]: ...

_IntType = TypeVar("_IntType", bound=integer)
_FloatType = TypeVar('_FloatType', bound=floating)

class floating(inexact[_NBit1]):
    def __init__(self, value: _FloatValue = ..., /) -> None: ...
    def item(
        self, args: L[0] | tuple[()] | tuple[L[0]] = ...,
        /,
    ) -> float: ...
    def tolist(self) -> float: ...
    def is_integer(self) -> bool: ...
    def hex(self: float64) -> str: ...
    @classmethod
    def fromhex(cls: type[float64], string: str, /) -> float64: ...
    def as_integer_ratio(self) -> tuple[int, int]: ...
    if sys.version_info >= (3, 9):
        def __ceil__(self: float64) -> int: ...
        def __floor__(self: float64) -> int: ...
    def __trunc__(self: float64) -> int: ...
    def __getnewargs__(self: float64) -> tuple[float]: ...
    def __getformat__(self: float64, typestr: L["double", "float"], /) -> str: ...
    @overload
    def __round__(self, ndigits: None = ...) -> int: ...
    @overload
    def __round__(self: _ScalarType, ndigits: SupportsIndex) -> _ScalarType: ...
    __add__: _FloatOp[_NBit1]
    __radd__: _FloatOp[_NBit1]
    __sub__: _FloatOp[_NBit1]
    __rsub__: _FloatOp[_NBit1]
    __mul__: _FloatOp[_NBit1]
    __rmul__: _FloatOp[_NBit1]
    __truediv__: _FloatOp[_NBit1]
    __rtruediv__: _FloatOp[_NBit1]
    __floordiv__: _FloatOp[_NBit1]
    __rfloordiv__: _FloatOp[_NBit1]
    __pow__: _FloatOp[_NBit1]
    __rpow__: _FloatOp[_NBit1]
    __mod__: _FloatMod[_NBit1]
    __rmod__: _FloatMod[_NBit1]
    __divmod__: _FloatDivMod[_NBit1]
    __rdivmod__: _FloatDivMod[_NBit1]

float16 = floating[_16Bit]
float32 = floating[_32Bit]
float64 = floating[_64Bit]

half = floating[_NBitHalf]
single = floating[_NBitSingle]
double = floating[_NBitDouble]
float_ = floating[_NBitDouble]
longdouble = floating[_NBitLongDouble]
longfloat = floating[_NBitLongDouble]

# The main reason for `complexfloating` having two typevars is cosmetic.
# It is used to clarify why `complex128`s precision is `_64Bit`, the latter
# describing the two 64 bit floats representing its real and imaginary component

class complexfloating(inexact[_NBit1], Generic[_NBit1, _NBit2]):
    def __init__(self, value: _ComplexValue = ..., /) -> None: ...
    def item(
        self, args: L[0] | tuple[()] | tuple[L[0]] = ..., /,
    ) -> complex: ...
    def tolist(self) -> complex: ...
    @property
    def real(self) -> floating[_NBit1]: ...  # type: ignore[override]
    @property
    def imag(self) -> floating[_NBit2]: ...  # type: ignore[override]
    def __abs__(self) -> floating[_NBit1]: ...  # type: ignore[override]
    def __getnewargs__(self: complex128) -> tuple[float, float]: ...
    # NOTE: Deprecated
    # def __round__(self, ndigits=...): ...
    __add__: _ComplexOp[_NBit1]
    __radd__: _ComplexOp[_NBit1]
    __sub__: _ComplexOp[_NBit1]
    __rsub__: _ComplexOp[_NBit1]
    __mul__: _ComplexOp[_NBit1]
    __rmul__: _ComplexOp[_NBit1]
    __truediv__: _ComplexOp[_NBit1]
    __rtruediv__: _ComplexOp[_NBit1]
    __pow__: _ComplexOp[_NBit1]
    __rpow__: _ComplexOp[_NBit1]

complex64 = complexfloating[_32Bit, _32Bit]
complex128 = complexfloating[_64Bit, _64Bit]

csingle = complexfloating[_NBitSingle, _NBitSingle]
singlecomplex = complexfloating[_NBitSingle, _NBitSingle]
cdouble = complexfloating[_NBitDouble, _NBitDouble]
complex_ = complexfloating[_NBitDouble, _NBitDouble]
cfloat = complexfloating[_NBitDouble, _NBitDouble]
clongdouble = complexfloating[_NBitLongDouble, _NBitLongDouble]
clongfloat = complexfloating[_NBitLongDouble, _NBitLongDouble]
longcomplex = complexfloating[_NBitLongDouble, _NBitLongDouble]

class flexible(generic): ...  # type: ignore

# TODO: `item`/`tolist` returns either `bytes` or `tuple`
# depending on whether or not it's used as an opaque bytes sequence
# or a structure
class void(flexible):
    @overload
    def __init__(self, value: _IntLike_co | bytes, /, dtype : None = ...) -> None: ...
    @overload
    def __init__(self, value: Any, /, dtype: _DTypeLikeVoid) -> None: ...
    @property
    def real(self: _ArraySelf) -> _ArraySelf: ...
    @property
    def imag(self: _ArraySelf) -> _ArraySelf: ...
    def setfield(
        self, val: ArrayLike, dtype: DTypeLike, offset: int = ...
    ) -> None: ...
    @overload
    def __getitem__(self, key: str | SupportsIndex) -> Any: ...
    @overload
    def __getitem__(self, key: list[str]) -> void: ...
    def __setitem__(
        self,
        key: str | list[str] | SupportsIndex,
        value: ArrayLike,
    ) -> None: ...

class character(flexible):  # type: ignore
    def __int__(self) -> int: ...
    def __float__(self) -> float: ...

# NOTE: Most `np.bytes_` / `np.str_` methods return their
# builtin `bytes` / `str` counterpart

class bytes_(character, bytes):
    @overload
    def __init__(self, value: object = ..., /) -> None: ...
    @overload
    def __init__(
        self, value: str, /, encoding: str = ..., errors: str = ...
    ) -> None: ...
    def item(
        self, args: L[0] | tuple[()] | tuple[L[0]] = ..., /,
    ) -> bytes: ...
    def tolist(self) -> bytes: ...

string_ = bytes_

class str_(character, str):
    @overload
    def __init__(self, value: object = ..., /) -> None: ...
    @overload
    def __init__(
        self, value: bytes, /, encoding: str = ..., errors: str = ...
    ) -> None: ...
    def item(
        self, args: L[0] | tuple[()] | tuple[L[0]] = ..., /,
    ) -> str: ...
    def tolist(self) -> str: ...

unicode_ = str_

#
# Constants
#

Inf: Final[float]
Infinity: Final[float]
NAN: Final[float]
NINF: Final[float]
NZERO: Final[float]
NaN: Final[float]
PINF: Final[float]
PZERO: Final[float]
e: Final[float]
euler_gamma: Final[float]
inf: Final[float]
infty: Final[float]
nan: Final[float]
pi: Final[float]

CLIP: L[0]
WRAP: L[1]
RAISE: L[2]

ERR_IGNORE: L[0]
ERR_WARN: L[1]
ERR_RAISE: L[2]
ERR_CALL: L[3]
ERR_PRINT: L[4]
ERR_LOG: L[5]
ERR_DEFAULT: L[521]

SHIFT_DIVIDEBYZERO: L[0]
SHIFT_OVERFLOW: L[3]
SHIFT_UNDERFLOW: L[6]
SHIFT_INVALID: L[9]

FPE_DIVIDEBYZERO: L[1]
FPE_OVERFLOW: L[2]
FPE_UNDERFLOW: L[4]
FPE_INVALID: L[8]

FLOATING_POINT_SUPPORT: L[1]
UFUNC_BUFSIZE_DEFAULT = BUFSIZE

little_endian: Final[bool]
True_: Final[bool_]
False_: Final[bool_]

UFUNC_PYVALS_NAME: L["UFUNC_PYVALS"]

newaxis: None

# See `numpy._typing._ufunc` for more concrete nin-/nout-specific stubs
@final
class ufunc:
    @property
    def __name__(self) -> str: ...
    @property
    def __doc__(self) -> str: ...
    __call__: Callable[..., Any]
    @property
    def nin(self) -> int: ...
    @property
    def nout(self) -> int: ...
    @property
    def nargs(self) -> int: ...
    @property
    def ntypes(self) -> int: ...
    @property
    def types(self) -> list[str]: ...
    # Broad return type because it has to encompass things like
    #
    # >>> np.logical_and.identity is True
    # True
    # >>> np.add.identity is 0
    # True
    # >>> np.sin.identity is None
    # True
    #
    # and any user-defined ufuncs.
    @property
    def identity(self) -> Any: ...
    # This is None for ufuncs and a string for gufuncs.
    @property
    def signature(self) -> None | str: ...
    # The next four methods will always exist, but they will just
    # raise a ValueError ufuncs with that don't accept two input
    # arguments and return one output argument. Because of that we
    # can't type them very precisely.
    reduce: Any
    accumulate: Any
    reduce: Any
    outer: Any
    # Similarly at won't be defined for ufuncs that return multiple
    # outputs, so we can't type it very precisely.
    at: Any

# Parameters: `__name__`, `ntypes` and `identity`
absolute: _UFunc_Nin1_Nout1[L['absolute'], L[20], None]
add: _UFunc_Nin2_Nout1[L['add'], L[22], L[0]]
arccos: _UFunc_Nin1_Nout1[L['arccos'], L[8], None]
arccosh: _UFunc_Nin1_Nout1[L['arccosh'], L[8], None]
arcsin: _UFunc_Nin1_Nout1[L['arcsin'], L[8], None]
arcsinh: _UFunc_Nin1_Nout1[L['arcsinh'], L[8], None]
arctan2: _UFunc_Nin2_Nout1[L['arctan2'], L[5], None]
arctan: _UFunc_Nin1_Nout1[L['arctan'], L[8], None]
arctanh: _UFunc_Nin1_Nout1[L['arctanh'], L[8], None]
bitwise_and: _UFunc_Nin2_Nout1[L['bitwise_and'], L[12], L[-1]]
bitwise_not: _UFunc_Nin1_Nout1[L['invert'], L[12], None]
bitwise_or: _UFunc_Nin2_Nout1[L['bitwise_or'], L[12], L[0]]
bitwise_xor: _UFunc_Nin2_Nout1[L['bitwise_xor'], L[12], L[0]]
cbrt: _UFunc_Nin1_Nout1[L['cbrt'], L[5], None]
ceil: _UFunc_Nin1_Nout1[L['ceil'], L[7], None]
conj: _UFunc_Nin1_Nout1[L['conjugate'], L[18], None]
conjugate: _UFunc_Nin1_Nout1[L['conjugate'], L[18], None]
copysign: _UFunc_Nin2_Nout1[L['copysign'], L[4], None]
cos: _UFunc_Nin1_Nout1[L['cos'], L[9], None]
cosh: _UFunc_Nin1_Nout1[L['cosh'], L[8], None]
deg2rad: _UFunc_Nin1_Nout1[L['deg2rad'], L[5], None]
degrees: _UFunc_Nin1_Nout1[L['degrees'], L[5], None]
divide: _UFunc_Nin2_Nout1[L['true_divide'], L[11], None]
divmod: _UFunc_Nin2_Nout2[L['divmod'], L[15], None]
equal: _UFunc_Nin2_Nout1[L['equal'], L[23], None]
exp2: _UFunc_Nin1_Nout1[L['exp2'], L[8], None]
exp: _UFunc_Nin1_Nout1[L['exp'], L[10], None]
expm1: _UFunc_Nin1_Nout1[L['expm1'], L[8], None]
fabs: _UFunc_Nin1_Nout1[L['fabs'], L[5], None]
float_power: _UFunc_Nin2_Nout1[L['float_power'], L[4], None]
floor: _UFunc_Nin1_Nout1[L['floor'], L[7], None]
floor_divide: _UFunc_Nin2_Nout1[L['floor_divide'], L[21], None]
fmax: _UFunc_Nin2_Nout1[L['fmax'], L[21], None]
fmin: _UFunc_Nin2_Nout1[L['fmin'], L[21], None]
fmod: _UFunc_Nin2_Nout1[L['fmod'], L[15], None]
frexp: _UFunc_Nin1_Nout2[L['frexp'], L[4], None]
gcd: _UFunc_Nin2_Nout1[L['gcd'], L[11], L[0]]
greater: _UFunc_Nin2_Nout1[L['greater'], L[23], None]
greater_equal: _UFunc_Nin2_Nout1[L['greater_equal'], L[23], None]
heaviside: _UFunc_Nin2_Nout1[L['heaviside'], L[4], None]
hypot: _UFunc_Nin2_Nout1[L['hypot'], L[5], L[0]]
invert: _UFunc_Nin1_Nout1[L['invert'], L[12], None]
isfinite: _UFunc_Nin1_Nout1[L['isfinite'], L[20], None]
isinf: _UFunc_Nin1_Nout1[L['isinf'], L[20], None]
isnan: _UFunc_Nin1_Nout1[L['isnan'], L[20], None]
isnat: _UFunc_Nin1_Nout1[L['isnat'], L[2], None]
lcm: _UFunc_Nin2_Nout1[L['lcm'], L[11], None]
ldexp: _UFunc_Nin2_Nout1[L['ldexp'], L[8], None]
left_shift: _UFunc_Nin2_Nout1[L['left_shift'], L[11], None]
less: _UFunc_Nin2_Nout1[L['less'], L[23], None]
less_equal: _UFunc_Nin2_Nout1[L['less_equal'], L[23], None]
log10: _UFunc_Nin1_Nout1[L['log10'], L[8], None]
log1p: _UFunc_Nin1_Nout1[L['log1p'], L[8], None]
log2: _UFunc_Nin1_Nout1[L['log2'], L[8], None]
log: _UFunc_Nin1_Nout1[L['log'], L[10], None]
logaddexp2: _UFunc_Nin2_Nout1[L['logaddexp2'], L[4], float]
logaddexp: _UFunc_Nin2_Nout1[L['logaddexp'], L[4], float]
logical_and: _UFunc_Nin2_Nout1[L['logical_and'], L[20], L[True]]
logical_not: _UFunc_Nin1_Nout1[L['logical_not'], L[20], None]
logical_or: _UFunc_Nin2_Nout1[L['logical_or'], L[20], L[False]]
logical_xor: _UFunc_Nin2_Nout1[L['logical_xor'], L[19], L[False]]
matmul: _GUFunc_Nin2_Nout1[L['matmul'], L[19], None]
maximum: _UFunc_Nin2_Nout1[L['maximum'], L[21], None]
minimum: _UFunc_Nin2_Nout1[L['minimum'], L[21], None]
mod: _UFunc_Nin2_Nout1[L['remainder'], L[16], None]
modf: _UFunc_Nin1_Nout2[L['modf'], L[4], None]
multiply: _UFunc_Nin2_Nout1[L['multiply'], L[23], L[1]]
negative: _UFunc_Nin1_Nout1[L['negative'], L[19], None]
nextafter: _UFunc_Nin2_Nout1[L['nextafter'], L[4], None]
not_equal: _UFunc_Nin2_Nout1[L['not_equal'], L[23], None]
positive: _UFunc_Nin1_Nout1[L['positive'], L[19], None]
power: _UFunc_Nin2_Nout1[L['power'], L[18], None]
rad2deg: _UFunc_Nin1_Nout1[L['rad2deg'], L[5], None]
radians: _UFunc_Nin1_Nout1[L['radians'], L[5], None]
reciprocal: _UFunc_Nin1_Nout1[L['reciprocal'], L[18], None]
remainder: _UFunc_Nin2_Nout1[L['remainder'], L[16], None]
right_shift: _UFunc_Nin2_Nout1[L['right_shift'], L[11], None]
rint: _UFunc_Nin1_Nout1[L['rint'], L[10], None]
sign: _UFunc_Nin1_Nout1[L['sign'], L[19], None]
signbit: _UFunc_Nin1_Nout1[L['signbit'], L[4], None]
sin: _UFunc_Nin1_Nout1[L['sin'], L[9], None]
sinh: _UFunc_Nin1_Nout1[L['sinh'], L[8], None]
spacing: _UFunc_Nin1_Nout1[L['spacing'], L[4], None]
sqrt: _UFunc_Nin1_Nout1[L['sqrt'], L[10], None]
square: _UFunc_Nin1_Nout1[L['square'], L[18], None]
subtract: _UFunc_Nin2_Nout1[L['subtract'], L[21], None]
tan: _UFunc_Nin1_Nout1[L['tan'], L[8], None]
tanh: _UFunc_Nin1_Nout1[L['tanh'], L[8], None]
true_divide: _UFunc_Nin2_Nout1[L['true_divide'], L[11], None]
trunc: _UFunc_Nin1_Nout1[L['trunc'], L[7], None]

abs = absolute

class _CopyMode(enum.Enum):
    ALWAYS: L[True]
    IF_NEEDED: L[False]
    NEVER: L[2]

# Warnings
class ModuleDeprecationWarning(DeprecationWarning): ...
class VisibleDeprecationWarning(UserWarning): ...
class ComplexWarning(RuntimeWarning): ...
class RankWarning(UserWarning): ...

# Errors
class TooHardError(RuntimeError): ...

class AxisError(ValueError, IndexError):
    axis: None | int
    ndim: None | int
    @overload
    def __init__(self, axis: str, ndim: None = ..., msg_prefix: None = ...) -> None: ...
    @overload
    def __init__(self, axis: int, ndim: int, msg_prefix: None | str = ...) -> None: ...

_CallType = TypeVar("_CallType", bound=_ErrFunc | _SupportsWrite[str])

class errstate(Generic[_CallType], ContextDecorator):
    call: _CallType
    kwargs: _ErrDictOptional

    # Expand `**kwargs` into explicit keyword-only arguments
    def __init__(
        self,
        *,
        call: _CallType = ...,
        all: None | _ErrKind = ...,
        divide: None | _ErrKind = ...,
        over: None | _ErrKind = ...,
        under: None | _ErrKind = ...,
        invalid: None | _ErrKind = ...,
    ) -> None: ...
    def __enter__(self) -> None: ...
    def __exit__(
        self,
        exc_type: None | type[BaseException],
        exc_value: None | BaseException,
        traceback: None | TracebackType,
        /,
    ) -> None: ...

@contextmanager
def _no_nep50_warning() -> Generator[None, None, None]: ...
def _get_promotion_state() -> str: ...
def _set_promotion_state(state: str, /) -> None: ...

class ndenumerate(Generic[_ScalarType]):
    iter: flatiter[NDArray[_ScalarType]]
    @overload
    def __new__(
        cls, arr: _FiniteNestedSequence[_SupportsArray[dtype[_ScalarType]]],
    ) -> ndenumerate[_ScalarType]: ...
    @overload
    def __new__(cls, arr: str | _NestedSequence[str]) -> ndenumerate[str_]: ...
    @overload
    def __new__(cls, arr: bytes | _NestedSequence[bytes]) -> ndenumerate[bytes_]: ...
    @overload
    def __new__(cls, arr: bool | _NestedSequence[bool]) -> ndenumerate[bool_]: ...
    @overload
    def __new__(cls, arr: int | _NestedSequence[int]) -> ndenumerate[int_]: ...
    @overload
    def __new__(cls, arr: float | _NestedSequence[float]) -> ndenumerate[float_]: ...
    @overload
    def __new__(cls, arr: complex | _NestedSequence[complex]) -> ndenumerate[complex_]: ...
    def __next__(self: ndenumerate[_ScalarType]) -> tuple[_Shape, _ScalarType]: ...
    def __iter__(self: _T) -> _T: ...

class ndindex:
    @overload
    def __init__(self, shape: tuple[SupportsIndex, ...], /) -> None: ...
    @overload
    def __init__(self, *shape: SupportsIndex) -> None: ...
    def __iter__(self: _T) -> _T: ...
    def __next__(self) -> _Shape: ...

class DataSource:
    def __init__(
        self,
        destpath: None | str | os.PathLike[str] = ...,
    ) -> None: ...
    def __del__(self) -> None: ...
    def abspath(self, path: str) -> str: ...
    def exists(self, path: str) -> bool: ...

    # Whether the file-object is opened in string or bytes mode (by default)
    # depends on the file-extension of `path`
    def open(
        self,
        path: str,
        mode: str = ...,
        encoding: None | str = ...,
        newline: None | str = ...,
    ) -> IO[Any]: ...

# TODO: The type of each `__next__` and `iters` return-type depends
# on the length and dtype of `args`; we can't describe this behavior yet
# as we lack variadics (PEP 646).
@final
class broadcast:
    def __new__(cls, *args: ArrayLike) -> broadcast: ...
    @property
    def index(self) -> int: ...
    @property
    def iters(self) -> tuple[flatiter[Any], ...]: ...
    @property
    def nd(self) -> int: ...
    @property
    def ndim(self) -> int: ...
    @property
    def numiter(self) -> int: ...
    @property
    def shape(self) -> _Shape: ...
    @property
    def size(self) -> int: ...
    def __next__(self) -> tuple[Any, ...]: ...
    def __iter__(self: _T) -> _T: ...
    def reset(self) -> None: ...

@final
class busdaycalendar:
    def __new__(
        cls,
        weekmask: ArrayLike = ...,
        holidays: ArrayLike | dt.date | _NestedSequence[dt.date] = ...,
    ) -> busdaycalendar: ...
    @property
    def weekmask(self) -> NDArray[bool_]: ...
    @property
    def holidays(self) -> NDArray[datetime64]: ...

class finfo(Generic[_FloatType]):
    dtype: dtype[_FloatType]
    bits: int
    eps: _FloatType
    epsneg: _FloatType
    iexp: int
    machep: int
    max: _FloatType
    maxexp: int
    min: _FloatType
    minexp: int
    negep: int
    nexp: int
    nmant: int
    precision: int
    resolution: _FloatType
    smallest_subnormal: _FloatType
    @property
    def smallest_normal(self) -> _FloatType: ...
    @property
    def tiny(self) -> _FloatType: ...
    @overload
    def __new__(
        cls, dtype: inexact[_NBit1] | _DTypeLike[inexact[_NBit1]]
    ) -> finfo[floating[_NBit1]]: ...
    @overload
    def __new__(
        cls, dtype: complex | float | type[complex] | type[float]
    ) -> finfo[float_]: ...
    @overload
    def __new__(
        cls, dtype: str
    ) -> finfo[floating[Any]]: ...

class iinfo(Generic[_IntType]):
    dtype: dtype[_IntType]
    kind: str
    bits: int
    key: str
    @property
    def min(self) -> int: ...
    @property
    def max(self) -> int: ...

    @overload
    def __new__(cls, dtype: _IntType | _DTypeLike[_IntType]) -> iinfo[_IntType]: ...
    @overload
    def __new__(cls, dtype: int | type[int]) -> iinfo[int_]: ...
    @overload
    def __new__(cls, dtype: str) -> iinfo[Any]: ...

class format_parser:
    dtype: dtype[void]
    def __init__(
        self,
        formats: DTypeLike,
        names: None | str | Sequence[str],
        titles: None | str | Sequence[str],
        aligned: bool = ...,
        byteorder: None | _ByteOrder = ...,
    ) -> None: ...

class recarray(ndarray[_ShapeType, _DType_co]):
    # NOTE: While not strictly mandatory, we're demanding here that arguments
    # for the `format_parser`- and `dtype`-based dtype constructors are
    # mutually exclusive
    @overload
    def __new__(
        subtype,
        shape: _ShapeLike,
        dtype: None = ...,
        buf: None | _SupportsBuffer = ...,
        offset: SupportsIndex = ...,
        strides: None | _ShapeLike = ...,
        *,
        formats: DTypeLike,
        names: None | str | Sequence[str] = ...,
        titles: None | str | Sequence[str] = ...,
        byteorder: None | _ByteOrder = ...,
        aligned: bool = ...,
        order: _OrderKACF = ...,
    ) -> recarray[Any, dtype[record]]: ...
    @overload
    def __new__(
        subtype,
        shape: _ShapeLike,
        dtype: DTypeLike,
        buf: None | _SupportsBuffer = ...,
        offset: SupportsIndex = ...,
        strides: None | _ShapeLike = ...,
        formats: None = ...,
        names: None = ...,
        titles: None = ...,
        byteorder: None = ...,
        aligned: L[False] = ...,
        order: _OrderKACF = ...,
    ) -> recarray[Any, dtype[Any]]: ...
    def __array_finalize__(self, obj: object) -> None: ...
    def __getattribute__(self, attr: str) -> Any: ...
    def __setattr__(self, attr: str, val: ArrayLike) -> None: ...
    @overload
    def __getitem__(self, indx: (
        SupportsIndex
        | _ArrayLikeInt_co
        | tuple[SupportsIndex | _ArrayLikeInt_co, ...]
    )) -> Any: ...
    @overload
    def __getitem__(self: recarray[Any, dtype[void]], indx: (
        None
        | slice
        | ellipsis
        | SupportsIndex
        | _ArrayLikeInt_co
        | tuple[None | slice | ellipsis | _ArrayLikeInt_co | SupportsIndex, ...]
    )) -> recarray[Any, _DType_co]: ...
    @overload
    def __getitem__(self, indx: (
        None
        | slice
        | ellipsis
        | SupportsIndex
        | _ArrayLikeInt_co
        | tuple[None | slice | ellipsis | _ArrayLikeInt_co | SupportsIndex, ...]
    )) -> ndarray[Any, _DType_co]: ...
    @overload
    def __getitem__(self, indx: str) -> NDArray[Any]: ...
    @overload
    def __getitem__(self, indx: list[str]) -> recarray[_ShapeType, dtype[record]]: ...
    @overload
    def field(self, attr: int | str, val: None = ...) -> Any: ...
    @overload
    def field(self, attr: int | str, val: ArrayLike) -> None: ...

class record(void):
    def __getattribute__(self, attr: str) -> Any: ...
    def __setattr__(self, attr: str, val: ArrayLike) -> None: ...
    def pprint(self) -> str: ...
    @overload
    def __getitem__(self, key: str | SupportsIndex) -> Any: ...
    @overload
    def __getitem__(self, key: list[str]) -> record: ...

_NDIterFlagsKind = L[
    "buffered",
    "c_index",
    "copy_if_overlap",
    "common_dtype",
    "delay_bufalloc",
    "external_loop",
    "f_index",
    "grow_inner", "growinner",
    "multi_index",
    "ranged",
    "refs_ok",
    "reduce_ok",
    "zerosize_ok",
]

_NDIterOpFlagsKind = L[
    "aligned",
    "allocate",
    "arraymask",
    "copy",
    "config",
    "nbo",
    "no_subtype",
    "no_broadcast",
    "overlap_assume_elementwise",
    "readonly",
    "readwrite",
    "updateifcopy",
    "virtual",
    "writeonly",
    "writemasked"
]

@final
class nditer:
    def __new__(
        cls,
        op: ArrayLike | Sequence[ArrayLike],
        flags: None | Sequence[_NDIterFlagsKind] = ...,
        op_flags: None | Sequence[Sequence[_NDIterOpFlagsKind]] = ...,
        op_dtypes: DTypeLike | Sequence[DTypeLike] = ...,
        order: _OrderKACF = ...,
        casting: _CastingKind = ...,
        op_axes: None | Sequence[Sequence[SupportsIndex]] = ...,
        itershape: None | _ShapeLike = ...,
        buffersize: SupportsIndex = ...,
    ) -> nditer: ...
    def __enter__(self) -> nditer: ...
    def __exit__(
        self,
        exc_type: None | type[BaseException],
        exc_value: None | BaseException,
        traceback: None | TracebackType,
    ) -> None: ...
    def __iter__(self) -> nditer: ...
    def __next__(self) -> tuple[NDArray[Any], ...]: ...
    def __len__(self) -> int: ...
    def __copy__(self) -> nditer: ...
    @overload
    def __getitem__(self, index: SupportsIndex) -> NDArray[Any]: ...
    @overload
    def __getitem__(self, index: slice) -> tuple[NDArray[Any], ...]: ...
    def __setitem__(self, index: slice | SupportsIndex, value: ArrayLike) -> None: ...
    def close(self) -> None: ...
    def copy(self) -> nditer: ...
    def debug_print(self) -> None: ...
    def enable_external_loop(self) -> None: ...
    def iternext(self) -> bool: ...
    def remove_axis(self, i: SupportsIndex, /) -> None: ...
    def remove_multi_index(self) -> None: ...
    def reset(self) -> None: ...
    @property
    def dtypes(self) -> tuple[dtype[Any], ...]: ...
    @property
    def finished(self) -> bool: ...
    @property
    def has_delayed_bufalloc(self) -> bool: ...
    @property
    def has_index(self) -> bool: ...
    @property
    def has_multi_index(self) -> bool: ...
    @property
    def index(self) -> int: ...
    @property
    def iterationneedsapi(self) -> bool: ...
    @property
    def iterindex(self) -> int: ...
    @property
    def iterrange(self) -> tuple[int, ...]: ...
    @property
    def itersize(self) -> int: ...
    @property
    def itviews(self) -> tuple[NDArray[Any], ...]: ...
    @property
    def multi_index(self) -> tuple[int, ...]: ...
    @property
    def ndim(self) -> int: ...
    @property
    def nop(self) -> int: ...
    @property
    def operands(self) -> tuple[NDArray[Any], ...]: ...
    @property
    def shape(self) -> tuple[int, ...]: ...
    @property
    def value(self) -> tuple[NDArray[Any], ...]: ...

_MemMapModeKind = L[
    "readonly", "r",
    "copyonwrite", "c",
    "readwrite", "r+",
    "write", "w+",
]

class memmap(ndarray[_ShapeType, _DType_co]):
    __array_priority__: ClassVar[float]
    filename: str | None
    offset: int
    mode: str
    @overload
    def __new__(
        subtype,
        filename: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _MemMapIOProtocol,
        dtype: type[uint8] = ...,
        mode: _MemMapModeKind = ...,
        offset: int = ...,
        shape: None | int | tuple[int, ...] = ...,
        order: _OrderKACF = ...,
    ) -> memmap[Any, dtype[uint8]]: ...
    @overload
    def __new__(
        subtype,
        filename: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _MemMapIOProtocol,
        dtype: _DTypeLike[_ScalarType],
        mode: _MemMapModeKind = ...,
        offset: int = ...,
        shape: None | int | tuple[int, ...] = ...,
        order: _OrderKACF = ...,
    ) -> memmap[Any, dtype[_ScalarType]]: ...
    @overload
    def __new__(
        subtype,
        filename: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _MemMapIOProtocol,
        dtype: DTypeLike,
        mode: _MemMapModeKind = ...,
        offset: int = ...,
        shape: None | int | tuple[int, ...] = ...,
        order: _OrderKACF = ...,
    ) -> memmap[Any, dtype[Any]]: ...
    def __array_finalize__(self, obj: object) -> None: ...
    def __array_wrap__(
        self,
        array: memmap[_ShapeType, _DType_co],
        context: None | tuple[ufunc, tuple[Any, ...], int] = ...,
    ) -> Any: ...
    def flush(self) -> None: ...

# TODO: Add a mypy plugin for managing functions whose output type is dependent
# on the literal value of some sort of signature (e.g. `einsum` and `vectorize`)
class vectorize:
    pyfunc: Callable[..., Any]
    cache: bool
    signature: None | str
    otypes: None | str
    excluded: set[int | str]
    __doc__: None | str
    def __init__(
        self,
        pyfunc: Callable[..., Any],
        otypes: None | str | Iterable[DTypeLike] = ...,
        doc: None | str = ...,
        excluded: None | Iterable[int | str] = ...,
        cache: bool = ...,
        signature: None | str = ...,
    ) -> None: ...
    def __call__(self, *args: Any, **kwargs: Any) -> Any: ...

class poly1d:
    @property
    def variable(self) -> str: ...
    @property
    def order(self) -> int: ...
    @property
    def o(self) -> int: ...
    @property
    def roots(self) -> NDArray[Any]: ...
    @property
    def r(self) -> NDArray[Any]: ...

    @property
    def coeffs(self) -> NDArray[Any]: ...
    @coeffs.setter
    def coeffs(self, value: NDArray[Any]) -> None: ...

    @property
    def c(self) -> NDArray[Any]: ...
    @c.setter
    def c(self, value: NDArray[Any]) -> None: ...

    @property
    def coef(self) -> NDArray[Any]: ...
    @coef.setter
    def coef(self, value: NDArray[Any]) -> None: ...

    @property
    def coefficients(self) -> NDArray[Any]: ...
    @coefficients.setter
    def coefficients(self, value: NDArray[Any]) -> None: ...

    __hash__: ClassVar[None]  # type: ignore

    @overload
    def __array__(self, t: None = ...) -> NDArray[Any]: ...
    @overload
    def __array__(self, t: _DType) -> ndarray[Any, _DType]: ...

    @overload
    def __call__(self, val: _ScalarLike_co) -> Any: ...
    @overload
    def __call__(self, val: poly1d) -> poly1d: ...
    @overload
    def __call__(self, val: ArrayLike) -> NDArray[Any]: ...

    def __init__(
        self,
        c_or_r: ArrayLike,
        r: bool = ...,
        variable: None | str = ...,
    ) -> None: ...
    def __len__(self) -> int: ...
    def __neg__(self) -> poly1d: ...
    def __pos__(self) -> poly1d: ...
    def __mul__(self, other: ArrayLike) -> poly1d: ...
    def __rmul__(self, other: ArrayLike) -> poly1d: ...
    def __add__(self, other: ArrayLike) -> poly1d: ...
    def __radd__(self, other: ArrayLike) -> poly1d: ...
    def __pow__(self, val: _FloatLike_co) -> poly1d: ...  # Integral floats are accepted
    def __sub__(self, other: ArrayLike) -> poly1d: ...
    def __rsub__(self, other: ArrayLike) -> poly1d: ...
    def __div__(self, other: ArrayLike) -> poly1d: ...
    def __truediv__(self, other: ArrayLike) -> poly1d: ...
    def __rdiv__(self, other: ArrayLike) -> poly1d: ...
    def __rtruediv__(self, other: ArrayLike) -> poly1d: ...
    def __getitem__(self, val: int) -> Any: ...
    def __setitem__(self, key: int, val: Any) -> None: ...
    def __iter__(self) -> Iterator[Any]: ...
    def deriv(self, m: SupportsInt | SupportsIndex = ...) -> poly1d: ...
    def integ(
        self,
        m: SupportsInt | SupportsIndex = ...,
        k: None | _ArrayLikeComplex_co | _ArrayLikeObject_co = ...,
    ) -> poly1d: ...

class matrix(ndarray[_ShapeType, _DType_co]):
    __array_priority__: ClassVar[float]
    def __new__(
        subtype,
        data: ArrayLike,
        dtype: DTypeLike = ...,
        copy: bool = ...,
    ) -> matrix[Any, Any]: ...
    def __array_finalize__(self, obj: object) -> None: ...

    @overload
    def __getitem__(self, key: (
        SupportsIndex
        | _ArrayLikeInt_co
        | tuple[SupportsIndex | _ArrayLikeInt_co, ...]
    )) -> Any: ...
    @overload
    def __getitem__(self, key: (
        None
        | slice
        | ellipsis
        | SupportsIndex
        | _ArrayLikeInt_co
        | tuple[None | slice | ellipsis | _ArrayLikeInt_co | SupportsIndex, ...]
    )) -> matrix[Any, _DType_co]: ...
    @overload
    def __getitem__(self: NDArray[void], key: str) -> matrix[Any, dtype[Any]]: ...
    @overload
    def __getitem__(self: NDArray[void], key: list[str]) -> matrix[_ShapeType, dtype[void]]: ...

    def __mul__(self, other: ArrayLike) -> matrix[Any, Any]: ...
    def __rmul__(self, other: ArrayLike) -> matrix[Any, Any]: ...
    def __imul__(self, other: ArrayLike) -> matrix[_ShapeType, _DType_co]: ...
    def __pow__(self, other: ArrayLike) -> matrix[Any, Any]: ...
    def __ipow__(self, other: ArrayLike) -> matrix[_ShapeType, _DType_co]: ...

    @overload
    def sum(self, axis: None = ..., dtype: DTypeLike = ..., out: None = ...) -> Any: ...
    @overload
    def sum(self, axis: _ShapeLike, dtype: DTypeLike = ..., out: None = ...) -> matrix[Any, Any]: ...
    @overload
    def sum(self, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def mean(self, axis: None = ..., dtype: DTypeLike = ..., out: None = ...) -> Any: ...
    @overload
    def mean(self, axis: _ShapeLike, dtype: DTypeLike = ..., out: None = ...) -> matrix[Any, Any]: ...
    @overload
    def mean(self, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def std(self, axis: None = ..., dtype: DTypeLike = ..., out: None = ..., ddof: float = ...) -> Any: ...
    @overload
    def std(self, axis: _ShapeLike, dtype: DTypeLike = ..., out: None = ..., ddof: float = ...) -> matrix[Any, Any]: ...
    @overload
    def std(self, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _NdArraySubClass = ..., ddof: float = ...) -> _NdArraySubClass: ...

    @overload
    def var(self, axis: None = ..., dtype: DTypeLike = ..., out: None = ..., ddof: float = ...) -> Any: ...
    @overload
    def var(self, axis: _ShapeLike, dtype: DTypeLike = ..., out: None = ..., ddof: float = ...) -> matrix[Any, Any]: ...
    @overload
    def var(self, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _NdArraySubClass = ..., ddof: float = ...) -> _NdArraySubClass: ...

    @overload
    def prod(self, axis: None = ..., dtype: DTypeLike = ..., out: None = ...) -> Any: ...
    @overload
    def prod(self, axis: _ShapeLike, dtype: DTypeLike = ..., out: None = ...) -> matrix[Any, Any]: ...
    @overload
    def prod(self, axis: None | _ShapeLike = ..., dtype: DTypeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def any(self, axis: None = ..., out: None = ...) -> bool_: ...
    @overload
    def any(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, dtype[bool_]]: ...
    @overload
    def any(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def all(self, axis: None = ..., out: None = ...) -> bool_: ...
    @overload
    def all(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, dtype[bool_]]: ...
    @overload
    def all(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def max(self: NDArray[_ScalarType], axis: None = ..., out: None = ...) -> _ScalarType: ...
    @overload
    def max(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, _DType_co]: ...
    @overload
    def max(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def min(self: NDArray[_ScalarType], axis: None = ..., out: None = ...) -> _ScalarType: ...
    @overload
    def min(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, _DType_co]: ...
    @overload
    def min(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def argmax(self: NDArray[_ScalarType], axis: None = ..., out: None = ...) -> intp: ...
    @overload
    def argmax(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, dtype[intp]]: ...
    @overload
    def argmax(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def argmin(self: NDArray[_ScalarType], axis: None = ..., out: None = ...) -> intp: ...
    @overload
    def argmin(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, dtype[intp]]: ...
    @overload
    def argmin(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    @overload
    def ptp(self: NDArray[_ScalarType], axis: None = ..., out: None = ...) -> _ScalarType: ...
    @overload
    def ptp(self, axis: _ShapeLike, out: None = ...) -> matrix[Any, _DType_co]: ...
    @overload
    def ptp(self, axis: None | _ShapeLike = ..., out: _NdArraySubClass = ...) -> _NdArraySubClass: ...

    def squeeze(self, axis: None | _ShapeLike = ...) -> matrix[Any, _DType_co]: ...
    def tolist(self: matrix[Any, dtype[_SupportsItem[_T]]]) -> list[list[_T]]: ...  # type: ignore[typevar]
    def ravel(self, order: _OrderKACF = ...) -> matrix[Any, _DType_co]: ...
    def flatten(self, order: _OrderKACF = ...) -> matrix[Any, _DType_co]: ...

    @property
    def T(self) -> matrix[Any, _DType_co]: ...
    @property
    def I(self) -> matrix[Any, Any]: ...
    @property
    def A(self) -> ndarray[_ShapeType, _DType_co]: ...
    @property
    def A1(self) -> ndarray[Any, _DType_co]: ...
    @property
    def H(self) -> matrix[Any, _DType_co]: ...
    def getT(self) -> matrix[Any, _DType_co]: ...
    def getI(self) -> matrix[Any, Any]: ...
    def getA(self) -> ndarray[_ShapeType, _DType_co]: ...
    def getA1(self) -> ndarray[Any, _DType_co]: ...
    def getH(self) -> matrix[Any, _DType_co]: ...

_CharType = TypeVar("_CharType", str_, bytes_)
_CharDType = TypeVar("_CharDType", dtype[str_], dtype[bytes_])
_CharArray = chararray[Any, dtype[_CharType]]

class chararray(ndarray[_ShapeType, _CharDType]):
    @overload
    def __new__(
        subtype,
        shape: _ShapeLike,
        itemsize: SupportsIndex | SupportsInt = ...,
        unicode: L[False] = ...,
        buffer: _SupportsBuffer = ...,
        offset: SupportsIndex = ...,
        strides: _ShapeLike = ...,
        order: _OrderKACF = ...,
    ) -> chararray[Any, dtype[bytes_]]: ...
    @overload
    def __new__(
        subtype,
        shape: _ShapeLike,
        itemsize: SupportsIndex | SupportsInt = ...,
        unicode: L[True] = ...,
        buffer: _SupportsBuffer = ...,
        offset: SupportsIndex = ...,
        strides: _ShapeLike = ...,
        order: _OrderKACF = ...,
    ) -> chararray[Any, dtype[str_]]: ...

    def __array_finalize__(self, obj: object) -> None: ...
    def __mul__(self, other: _ArrayLikeInt_co) -> chararray[Any, _CharDType]: ...
    def __rmul__(self, other: _ArrayLikeInt_co) -> chararray[Any, _CharDType]: ...
    def __mod__(self, i: Any) -> chararray[Any, _CharDType]: ...

    @overload
    def __eq__(
        self: _CharArray[str_],
        other: _ArrayLikeStr_co,
    ) -> NDArray[bool_]: ...
    @overload
    def __eq__(
        self: _CharArray[bytes_],
        other: _ArrayLikeBytes_co,
    ) -> NDArray[bool_]: ...

    @overload
    def __ne__(
        self: _CharArray[str_],
        other: _ArrayLikeStr_co,
    ) -> NDArray[bool_]: ...
    @overload
    def __ne__(
        self: _CharArray[bytes_],
        other: _ArrayLikeBytes_co,
    ) -> NDArray[bool_]: ...

    @overload
    def __ge__(
        self: _CharArray[str_],
        other: _ArrayLikeStr_co,
    ) -> NDArray[bool_]: ...
    @overload
    def __ge__(
        self: _CharArray[bytes_],
        other: _ArrayLikeBytes_co,
    ) -> NDArray[bool_]: ...

    @overload
    def __le__(
        self: _CharArray[str_],
        other: _ArrayLikeStr_co,
    ) -> NDArray[bool_]: ...
    @overload
    def __le__(
        self: _CharArray[bytes_],
        other: _ArrayLikeBytes_co,
    ) -> NDArray[bool_]: ...

    @overload
    def __gt__(
        self: _CharArray[str_],
        other: _ArrayLikeStr_co,
    ) -> NDArray[bool_]: ...
    @overload
    def __gt__(
        self: _CharArray[bytes_],
        other: _ArrayLikeBytes_co,
    ) -> NDArray[bool_]: ...

    @overload
    def __lt__(
        self: _CharArray[str_],
        other: _ArrayLikeStr_co,
    ) -> NDArray[bool_]: ...
    @overload
    def __lt__(
        self: _CharArray[bytes_],
        other: _ArrayLikeBytes_co,
    ) -> NDArray[bool_]: ...

    @overload
    def __add__(
        self: _CharArray[str_],
        other: _ArrayLikeStr_co,
    ) -> _CharArray[str_]: ...
    @overload
    def __add__(
        self: _CharArray[bytes_],
        other: _ArrayLikeBytes_co,
    ) -> _CharArray[bytes_]: ...

    @overload
    def __radd__(
        self: _CharArray[str_],
        other: _ArrayLikeStr_co,
    ) -> _CharArray[str_]: ...
    @overload
    def __radd__(
        self: _CharArray[bytes_],
        other: _ArrayLikeBytes_co,
    ) -> _CharArray[bytes_]: ...

    @overload
    def center(
        self: _CharArray[str_],
        width: _ArrayLikeInt_co,
        fillchar: _ArrayLikeStr_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def center(
        self: _CharArray[bytes_],
        width: _ArrayLikeInt_co,
        fillchar: _ArrayLikeBytes_co = ...,
    ) -> _CharArray[bytes_]: ...

    @overload
    def count(
        self: _CharArray[str_],
        sub: _ArrayLikeStr_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...
    @overload
    def count(
        self: _CharArray[bytes_],
        sub: _ArrayLikeBytes_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...

    def decode(
        self: _CharArray[bytes_],
        encoding: None | str = ...,
        errors: None | str = ...,
    ) -> _CharArray[str_]: ...

    def encode(
        self: _CharArray[str_],
        encoding: None | str = ...,
        errors: None | str = ...,
    ) -> _CharArray[bytes_]: ...

    @overload
    def endswith(
        self: _CharArray[str_],
        suffix: _ArrayLikeStr_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[bool_]: ...
    @overload
    def endswith(
        self: _CharArray[bytes_],
        suffix: _ArrayLikeBytes_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[bool_]: ...

    def expandtabs(
        self,
        tabsize: _ArrayLikeInt_co = ...,
    ) -> chararray[Any, _CharDType]: ...

    @overload
    def find(
        self: _CharArray[str_],
        sub: _ArrayLikeStr_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...
    @overload
    def find(
        self: _CharArray[bytes_],
        sub: _ArrayLikeBytes_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...

    @overload
    def index(
        self: _CharArray[str_],
        sub: _ArrayLikeStr_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...
    @overload
    def index(
        self: _CharArray[bytes_],
        sub: _ArrayLikeBytes_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...

    @overload
    def join(
        self: _CharArray[str_],
        seq: _ArrayLikeStr_co,
    ) -> _CharArray[str_]: ...
    @overload
    def join(
        self: _CharArray[bytes_],
        seq: _ArrayLikeBytes_co,
    ) -> _CharArray[bytes_]: ...

    @overload
    def ljust(
        self: _CharArray[str_],
        width: _ArrayLikeInt_co,
        fillchar: _ArrayLikeStr_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def ljust(
        self: _CharArray[bytes_],
        width: _ArrayLikeInt_co,
        fillchar: _ArrayLikeBytes_co = ...,
    ) -> _CharArray[bytes_]: ...

    @overload
    def lstrip(
        self: _CharArray[str_],
        chars: None | _ArrayLikeStr_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def lstrip(
        self: _CharArray[bytes_],
        chars: None | _ArrayLikeBytes_co = ...,
    ) -> _CharArray[bytes_]: ...

    @overload
    def partition(
        self: _CharArray[str_],
        sep: _ArrayLikeStr_co,
    ) -> _CharArray[str_]: ...
    @overload
    def partition(
        self: _CharArray[bytes_],
        sep: _ArrayLikeBytes_co,
    ) -> _CharArray[bytes_]: ...

    @overload
    def replace(
        self: _CharArray[str_],
        old: _ArrayLikeStr_co,
        new: _ArrayLikeStr_co,
        count: None | _ArrayLikeInt_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def replace(
        self: _CharArray[bytes_],
        old: _ArrayLikeBytes_co,
        new: _ArrayLikeBytes_co,
        count: None | _ArrayLikeInt_co = ...,
    ) -> _CharArray[bytes_]: ...

    @overload
    def rfind(
        self: _CharArray[str_],
        sub: _ArrayLikeStr_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...
    @overload
    def rfind(
        self: _CharArray[bytes_],
        sub: _ArrayLikeBytes_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...

    @overload
    def rindex(
        self: _CharArray[str_],
        sub: _ArrayLikeStr_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...
    @overload
    def rindex(
        self: _CharArray[bytes_],
        sub: _ArrayLikeBytes_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[int_]: ...

    @overload
    def rjust(
        self: _CharArray[str_],
        width: _ArrayLikeInt_co,
        fillchar: _ArrayLikeStr_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def rjust(
        self: _CharArray[bytes_],
        width: _ArrayLikeInt_co,
        fillchar: _ArrayLikeBytes_co = ...,
    ) -> _CharArray[bytes_]: ...

    @overload
    def rpartition(
        self: _CharArray[str_],
        sep: _ArrayLikeStr_co,
    ) -> _CharArray[str_]: ...
    @overload
    def rpartition(
        self: _CharArray[bytes_],
        sep: _ArrayLikeBytes_co,
    ) -> _CharArray[bytes_]: ...

    @overload
    def rsplit(
        self: _CharArray[str_],
        sep: None | _ArrayLikeStr_co = ...,
        maxsplit: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[object_]: ...
    @overload
    def rsplit(
        self: _CharArray[bytes_],
        sep: None | _ArrayLikeBytes_co = ...,
        maxsplit: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[object_]: ...

    @overload
    def rstrip(
        self: _CharArray[str_],
        chars: None | _ArrayLikeStr_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def rstrip(
        self: _CharArray[bytes_],
        chars: None | _ArrayLikeBytes_co = ...,
    ) -> _CharArray[bytes_]: ...

    @overload
    def split(
        self: _CharArray[str_],
        sep: None | _ArrayLikeStr_co = ...,
        maxsplit: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[object_]: ...
    @overload
    def split(
        self: _CharArray[bytes_],
        sep: None | _ArrayLikeBytes_co = ...,
        maxsplit: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[object_]: ...

    def splitlines(self, keepends: None | _ArrayLikeBool_co = ...) -> NDArray[object_]: ...

    @overload
    def startswith(
        self: _CharArray[str_],
        prefix: _ArrayLikeStr_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[bool_]: ...
    @overload
    def startswith(
        self: _CharArray[bytes_],
        prefix: _ArrayLikeBytes_co,
        start: _ArrayLikeInt_co = ...,
        end: None | _ArrayLikeInt_co = ...,
    ) -> NDArray[bool_]: ...

    @overload
    def strip(
        self: _CharArray[str_],
        chars: None | _ArrayLikeStr_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def strip(
        self: _CharArray[bytes_],
        chars: None | _ArrayLikeBytes_co = ...,
    ) -> _CharArray[bytes_]: ...

    @overload
    def translate(
        self: _CharArray[str_],
        table: _ArrayLikeStr_co,
        deletechars: None | _ArrayLikeStr_co = ...,
    ) -> _CharArray[str_]: ...
    @overload
    def translate(
        self: _CharArray[bytes_],
        table: _ArrayLikeBytes_co,
        deletechars: None | _ArrayLikeBytes_co = ...,
    ) -> _CharArray[bytes_]: ...

    def zfill(self, width: _ArrayLikeInt_co) -> chararray[Any, _CharDType]: ...
    def capitalize(self) -> chararray[_ShapeType, _CharDType]: ...
    def title(self) -> chararray[_ShapeType, _CharDType]: ...
    def swapcase(self) -> chararray[_ShapeType, _CharDType]: ...
    def lower(self) -> chararray[_ShapeType, _CharDType]: ...
    def upper(self) -> chararray[_ShapeType, _CharDType]: ...
    def isalnum(self) -> ndarray[_ShapeType, dtype[bool_]]: ...
    def isalpha(self) -> ndarray[_ShapeType, dtype[bool_]]: ...
    def isdigit(self) -> ndarray[_ShapeType, dtype[bool_]]: ...
    def islower(self) -> ndarray[_ShapeType, dtype[bool_]]: ...
    def isspace(self) -> ndarray[_ShapeType, dtype[bool_]]: ...
    def istitle(self) -> ndarray[_ShapeType, dtype[bool_]]: ...
    def isupper(self) -> ndarray[_ShapeType, dtype[bool_]]: ...
    def isnumeric(self) -> ndarray[_ShapeType, dtype[bool_]]: ...
    def isdecimal(self) -> ndarray[_ShapeType, dtype[bool_]]: ...

# NOTE: Deprecated
# class MachAr: ...

class _SupportsDLPack(Protocol[_T_contra]):
    def __dlpack__(self, *, stream: None | _T_contra = ...) -> _PyCapsule: ...

def from_dlpack(obj: _SupportsDLPack[None], /) -> NDArray[Any]: ...
