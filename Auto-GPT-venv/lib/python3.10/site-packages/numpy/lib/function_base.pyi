import sys
from collections.abc import Sequence, Iterator, Callable, Iterable
from typing import (
    Literal as L,
    Any,
    TypeVar,
    overload,
    Protocol,
    SupportsIndex,
    SupportsInt,
)

if sys.version_info >= (3, 10):
    from typing import TypeGuard
else:
    from typing_extensions import TypeGuard

from numpy import (
    vectorize as vectorize,
    ufunc,
    generic,
    floating,
    complexfloating,
    intp,
    float64,
    complex128,
    timedelta64,
    datetime64,
    object_,
    _OrderKACF,
)

from numpy._typing import (
    NDArray,
    ArrayLike,
    DTypeLike,
    _ShapeLike,
    _ScalarLike_co,
    _DTypeLike,
    _ArrayLike,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeTD64_co,
    _ArrayLikeDT64_co,
    _ArrayLikeObject_co,
    _FloatLike_co,
    _ComplexLike_co,
)

from numpy.core.function_base import (
    add_newdoc as add_newdoc,
)

from numpy.core.multiarray import (
    add_docstring as add_docstring,
    bincount as bincount,
)

from numpy.core.umath import _add_newdoc_ufunc

_T = TypeVar("_T")
_T_co = TypeVar("_T_co", covariant=True)
_SCT = TypeVar("_SCT", bound=generic)
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])

_2Tuple = tuple[_T, _T]

class _TrimZerosSequence(Protocol[_T_co]):
    def __len__(self) -> int: ...
    def __getitem__(self, key: slice, /) -> _T_co: ...
    def __iter__(self) -> Iterator[Any]: ...

class _SupportsWriteFlush(Protocol):
    def write(self, s: str, /) -> object: ...
    def flush(self) -> object: ...

__all__: list[str]

# NOTE: This is in reality a re-export of `np.core.umath._add_newdoc_ufunc`
def add_newdoc_ufunc(ufunc: ufunc, new_docstring: str, /) -> None: ...

@overload
def rot90(
    m: _ArrayLike[_SCT],
    k: int = ...,
    axes: tuple[int, int] = ...,
) -> NDArray[_SCT]: ...
@overload
def rot90(
    m: ArrayLike,
    k: int = ...,
    axes: tuple[int, int] = ...,
) -> NDArray[Any]: ...

@overload
def flip(m: _SCT, axis: None = ...) -> _SCT: ...
@overload
def flip(m: _ScalarLike_co, axis: None = ...) -> Any: ...
@overload
def flip(m: _ArrayLike[_SCT], axis: None | _ShapeLike = ...) -> NDArray[_SCT]: ...
@overload
def flip(m: ArrayLike, axis: None | _ShapeLike = ...) -> NDArray[Any]: ...

def iterable(y: object) -> TypeGuard[Iterable[Any]]: ...

@overload
def average(
    a: _ArrayLikeFloat_co,
    axis: None = ...,
    weights: None | _ArrayLikeFloat_co= ...,
    returned: L[False] = ...,
    keepdims: L[False] = ...,
) -> floating[Any]: ...
@overload
def average(
    a: _ArrayLikeComplex_co,
    axis: None = ...,
    weights: None | _ArrayLikeComplex_co = ...,
    returned: L[False] = ...,
    keepdims: L[False] = ...,
) -> complexfloating[Any, Any]: ...
@overload
def average(
    a: _ArrayLikeObject_co,
    axis: None = ...,
    weights: None | Any = ...,
    returned: L[False] = ...,
    keepdims: L[False] = ...,
) -> Any: ...
@overload
def average(
    a: _ArrayLikeFloat_co,
    axis: None = ...,
    weights: None | _ArrayLikeFloat_co= ...,
    returned: L[True] = ...,
    keepdims: L[False] = ...,
) -> _2Tuple[floating[Any]]: ...
@overload
def average(
    a: _ArrayLikeComplex_co,
    axis: None = ...,
    weights: None | _ArrayLikeComplex_co = ...,
    returned: L[True] = ...,
    keepdims: L[False] = ...,
) -> _2Tuple[complexfloating[Any, Any]]: ...
@overload
def average(
    a: _ArrayLikeObject_co,
    axis: None = ...,
    weights: None | Any = ...,
    returned: L[True] = ...,
    keepdims: L[False] = ...,
) -> _2Tuple[Any]: ...
@overload
def average(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None | _ShapeLike = ...,
    weights: None | Any = ...,
    returned: L[False] = ...,
    keepdims: bool = ...,
) -> Any: ...
@overload
def average(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    axis: None | _ShapeLike = ...,
    weights: None | Any = ...,
    returned: L[True] = ...,
    keepdims: bool = ...,
) -> _2Tuple[Any]: ...

@overload
def asarray_chkfinite(
    a: _ArrayLike[_SCT],
    dtype: None = ...,
    order: _OrderKACF = ...,
) -> NDArray[_SCT]: ...
@overload
def asarray_chkfinite(
    a: object,
    dtype: None = ...,
    order: _OrderKACF = ...,
) -> NDArray[Any]: ...
@overload
def asarray_chkfinite(
    a: Any,
    dtype: _DTypeLike[_SCT],
    order: _OrderKACF = ...,
) -> NDArray[_SCT]: ...
@overload
def asarray_chkfinite(
    a: Any,
    dtype: DTypeLike,
    order: _OrderKACF = ...,
) -> NDArray[Any]: ...

# TODO: Use PEP 612 `ParamSpec` once mypy supports `Concatenate`
# xref python/mypy#8645
@overload
def piecewise(
    x: _ArrayLike[_SCT],
    condlist: ArrayLike,
    funclist: Sequence[Any | Callable[..., Any]],
    *args: Any,
    **kw: Any,
) -> NDArray[_SCT]: ...
@overload
def piecewise(
    x: ArrayLike,
    condlist: ArrayLike,
    funclist: Sequence[Any | Callable[..., Any]],
    *args: Any,
    **kw: Any,
) -> NDArray[Any]: ...

def select(
    condlist: Sequence[ArrayLike],
    choicelist: Sequence[ArrayLike],
    default: ArrayLike = ...,
) -> NDArray[Any]: ...

@overload
def copy(
    a: _ArrayType,
    order: _OrderKACF,
    subok: L[True],
) -> _ArrayType: ...
@overload
def copy(
    a: _ArrayType,
    order: _OrderKACF = ...,
    *,
    subok: L[True],
) -> _ArrayType: ...
@overload
def copy(
    a: _ArrayLike[_SCT],
    order: _OrderKACF = ...,
    subok: L[False] = ...,
) -> NDArray[_SCT]: ...
@overload
def copy(
    a: ArrayLike,
    order: _OrderKACF = ...,
    subok: L[False] = ...,
) -> NDArray[Any]: ...

def gradient(
    f: ArrayLike,
    *varargs: ArrayLike,
    axis: None | _ShapeLike = ...,
    edge_order: L[1, 2] = ...,
) -> Any: ...

@overload
def diff(
    a: _T,
    n: L[0],
    axis: SupportsIndex = ...,
    prepend: ArrayLike = ...,
    append: ArrayLike = ...,
) -> _T: ...
@overload
def diff(
    a: ArrayLike,
    n: int = ...,
    axis: SupportsIndex = ...,
    prepend: ArrayLike = ...,
    append: ArrayLike = ...,
) -> NDArray[Any]: ...

@overload
def interp(
    x: _ArrayLikeFloat_co,
    xp: _ArrayLikeFloat_co,
    fp: _ArrayLikeFloat_co,
    left: None | _FloatLike_co = ...,
    right: None | _FloatLike_co = ...,
    period: None | _FloatLike_co = ...,
) -> NDArray[float64]: ...
@overload
def interp(
    x: _ArrayLikeFloat_co,
    xp: _ArrayLikeFloat_co,
    fp: _ArrayLikeComplex_co,
    left: None | _ComplexLike_co = ...,
    right: None | _ComplexLike_co = ...,
    period: None | _FloatLike_co = ...,
) -> NDArray[complex128]: ...

@overload
def angle(z: _ComplexLike_co, deg: bool = ...) -> floating[Any]: ...
@overload
def angle(z: object_, deg: bool = ...) -> Any: ...
@overload
def angle(z: _ArrayLikeComplex_co, deg: bool = ...) -> NDArray[floating[Any]]: ...
@overload
def angle(z: _ArrayLikeObject_co, deg: bool = ...) -> NDArray[object_]: ...

@overload
def unwrap(
    p: _ArrayLikeFloat_co,
    discont: None | float = ...,
    axis: int = ...,
    *,
    period: float = ...,
) -> NDArray[floating[Any]]: ...
@overload
def unwrap(
    p: _ArrayLikeObject_co,
    discont: None | float = ...,
    axis: int = ...,
    *,
    period: float = ...,
) -> NDArray[object_]: ...

def sort_complex(a: ArrayLike) -> NDArray[complexfloating[Any, Any]]: ...

def trim_zeros(
    filt: _TrimZerosSequence[_T],
    trim: L["f", "b", "fb", "bf"] = ...,
) -> _T: ...

@overload
def extract(condition: ArrayLike, arr: _ArrayLike[_SCT]) -> NDArray[_SCT]: ...
@overload
def extract(condition: ArrayLike, arr: ArrayLike) -> NDArray[Any]: ...

def place(arr: NDArray[Any], mask: ArrayLike, vals: Any) -> None: ...

def disp(
    mesg: object,
    device: None | _SupportsWriteFlush = ...,
    linefeed: bool = ...,
) -> None: ...

@overload
def cov(
    m: _ArrayLikeFloat_co,
    y: None | _ArrayLikeFloat_co = ...,
    rowvar: bool = ...,
    bias: bool = ...,
    ddof: None | SupportsIndex | SupportsInt = ...,
    fweights: None | ArrayLike = ...,
    aweights: None | ArrayLike = ...,
    *,
    dtype: None = ...,
) -> NDArray[floating[Any]]: ...
@overload
def cov(
    m: _ArrayLikeComplex_co,
    y: None | _ArrayLikeComplex_co = ...,
    rowvar: bool = ...,
    bias: bool = ...,
    ddof: None | SupportsIndex | SupportsInt = ...,
    fweights: None | ArrayLike = ...,
    aweights: None | ArrayLike = ...,
    *,
    dtype: None = ...,
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def cov(
    m: _ArrayLikeComplex_co,
    y: None | _ArrayLikeComplex_co = ...,
    rowvar: bool = ...,
    bias: bool = ...,
    ddof: None | SupportsIndex | SupportsInt = ...,
    fweights: None | ArrayLike = ...,
    aweights: None | ArrayLike = ...,
    *,
    dtype: _DTypeLike[_SCT],
) -> NDArray[_SCT]: ...
@overload
def cov(
    m: _ArrayLikeComplex_co,
    y: None | _ArrayLikeComplex_co = ...,
    rowvar: bool = ...,
    bias: bool = ...,
    ddof: None | SupportsIndex | SupportsInt = ...,
    fweights: None | ArrayLike = ...,
    aweights: None | ArrayLike = ...,
    *,
    dtype: DTypeLike,
) -> NDArray[Any]: ...

# NOTE `bias` and `ddof` have been deprecated
@overload
def corrcoef(
    m: _ArrayLikeFloat_co,
    y: None | _ArrayLikeFloat_co = ...,
    rowvar: bool = ...,
    *,
    dtype: None = ...,
) -> NDArray[floating[Any]]: ...
@overload
def corrcoef(
    m: _ArrayLikeComplex_co,
    y: None | _ArrayLikeComplex_co = ...,
    rowvar: bool = ...,
    *,
    dtype: None = ...,
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def corrcoef(
    m: _ArrayLikeComplex_co,
    y: None | _ArrayLikeComplex_co = ...,
    rowvar: bool = ...,
    *,
    dtype: _DTypeLike[_SCT],
) -> NDArray[_SCT]: ...
@overload
def corrcoef(
    m: _ArrayLikeComplex_co,
    y: None | _ArrayLikeComplex_co = ...,
    rowvar: bool = ...,
    *,
    dtype: DTypeLike,
) -> NDArray[Any]: ...

def blackman(M: _FloatLike_co) -> NDArray[floating[Any]]: ...

def bartlett(M: _FloatLike_co) -> NDArray[floating[Any]]: ...

def hanning(M: _FloatLike_co) -> NDArray[floating[Any]]: ...

def hamming(M: _FloatLike_co) -> NDArray[floating[Any]]: ...

def i0(x: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...

def kaiser(
    M: _FloatLike_co,
    beta: _FloatLike_co,
) -> NDArray[floating[Any]]: ...

@overload
def sinc(x: _FloatLike_co) -> floating[Any]: ...
@overload
def sinc(x: _ComplexLike_co) -> complexfloating[Any, Any]: ...
@overload
def sinc(x: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...
@overload
def sinc(x: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

# NOTE: Deprecated
# def msort(a: ArrayLike) -> NDArray[Any]: ...

@overload
def median(
    a: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: L[False] = ...,
) -> floating[Any]: ...
@overload
def median(
    a: _ArrayLikeComplex_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: L[False] = ...,
) -> complexfloating[Any, Any]: ...
@overload
def median(
    a: _ArrayLikeTD64_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: L[False] = ...,
) -> timedelta64: ...
@overload
def median(
    a: _ArrayLikeObject_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: L[False] = ...,
) -> Any: ...
@overload
def median(
    a: _ArrayLikeFloat_co | _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    axis: None | _ShapeLike = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    keepdims: bool = ...,
) -> Any: ...
@overload
def median(
    a: _ArrayLikeFloat_co | _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    axis: None | _ShapeLike = ...,
    out: _ArrayType = ...,
    overwrite_input: bool = ...,
    keepdims: bool = ...,
) -> _ArrayType: ...

_MethodKind = L[
    "inverted_cdf",
    "averaged_inverted_cdf",
    "closest_observation",
    "interpolated_inverted_cdf",
    "hazen",
    "weibull",
    "linear",
    "median_unbiased",
    "normal_unbiased",
    "lower",
    "higher",
    "midpoint",
    "nearest",
]

@overload
def percentile(
    a: _ArrayLikeFloat_co,
    q: _FloatLike_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
) -> floating[Any]: ...
@overload
def percentile(
    a: _ArrayLikeComplex_co,
    q: _FloatLike_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
) -> complexfloating[Any, Any]: ...
@overload
def percentile(
    a: _ArrayLikeTD64_co,
    q: _FloatLike_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
) -> timedelta64: ...
@overload
def percentile(
    a: _ArrayLikeDT64_co,
    q: _FloatLike_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
) -> datetime64: ...
@overload
def percentile(
    a: _ArrayLikeObject_co,
    q: _FloatLike_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
) -> Any: ...
@overload
def percentile(
    a: _ArrayLikeFloat_co,
    q: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
) -> NDArray[floating[Any]]: ...
@overload
def percentile(
    a: _ArrayLikeComplex_co,
    q: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
) -> NDArray[complexfloating[Any, Any]]: ...
@overload
def percentile(
    a: _ArrayLikeTD64_co,
    q: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
) -> NDArray[timedelta64]: ...
@overload
def percentile(
    a: _ArrayLikeDT64_co,
    q: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
) -> NDArray[datetime64]: ...
@overload
def percentile(
    a: _ArrayLikeObject_co,
    q: _ArrayLikeFloat_co,
    axis: None = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: L[False] = ...,
) -> NDArray[object_]: ...
@overload
def percentile(
    a: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    q: _ArrayLikeFloat_co,
    axis: None | _ShapeLike = ...,
    out: None = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: bool = ...,
) -> Any: ...
@overload
def percentile(
    a: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    q: _ArrayLikeFloat_co,
    axis: None | _ShapeLike = ...,
    out: _ArrayType = ...,
    overwrite_input: bool = ...,
    method: _MethodKind = ...,
    keepdims: bool = ...,
) -> _ArrayType: ...

# NOTE: Not an alias, but they do have identical signatures
# (that we can reuse)
quantile = percentile

# TODO: Returns a scalar for <= 1D array-likes; returns an ndarray otherwise
def trapz(
    y: _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co,
    x: None | _ArrayLikeComplex_co | _ArrayLikeTD64_co | _ArrayLikeObject_co = ...,
    dx: float = ...,
    axis: SupportsIndex = ...,
) -> Any: ...

def meshgrid(
    *xi: ArrayLike,
    copy: bool = ...,
    sparse: bool = ...,
    indexing: L["xy", "ij"] = ...,
) -> list[NDArray[Any]]: ...

@overload
def delete(
    arr: _ArrayLike[_SCT],
    obj: slice | _ArrayLikeInt_co,
    axis: None | SupportsIndex = ...,
) -> NDArray[_SCT]: ...
@overload
def delete(
    arr: ArrayLike,
    obj: slice | _ArrayLikeInt_co,
    axis: None | SupportsIndex = ...,
) -> NDArray[Any]: ...

@overload
def insert(
    arr: _ArrayLike[_SCT],
    obj: slice | _ArrayLikeInt_co,
    values: ArrayLike,
    axis: None | SupportsIndex = ...,
) -> NDArray[_SCT]: ...
@overload
def insert(
    arr: ArrayLike,
    obj: slice | _ArrayLikeInt_co,
    values: ArrayLike,
    axis: None | SupportsIndex = ...,
) -> NDArray[Any]: ...

def append(
    arr: ArrayLike,
    values: ArrayLike,
    axis: None | SupportsIndex = ...,
) -> NDArray[Any]: ...

@overload
def digitize(
    x: _FloatLike_co,
    bins: _ArrayLikeFloat_co,
    right: bool = ...,
) -> intp: ...
@overload
def digitize(
    x: _ArrayLikeFloat_co,
    bins: _ArrayLikeFloat_co,
    right: bool = ...,
) -> NDArray[intp]: ...
