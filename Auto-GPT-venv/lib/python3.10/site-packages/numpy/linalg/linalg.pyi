from collections.abc import Iterable
from typing import (
    Literal as L,
    overload,
    TypeVar,
    Any,
    SupportsIndex,
    SupportsInt,
)

from numpy import (
    generic,
    floating,
    complexfloating,
    int32,
    float64,
    complex128,
)

from numpy.linalg import LinAlgError as LinAlgError

from numpy._typing import (
    NDArray,
    ArrayLike,
    _ArrayLikeInt_co,
    _ArrayLikeFloat_co,
    _ArrayLikeComplex_co,
    _ArrayLikeTD64_co,
    _ArrayLikeObject_co,
)

_T = TypeVar("_T")
_ArrayType = TypeVar("_ArrayType", bound=NDArray[Any])

_2Tuple = tuple[_T, _T]
_ModeKind = L["reduced", "complete", "r", "raw"]

__all__: list[str]

@overload
def tensorsolve(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
    axes: None | Iterable[int] =...,
) -> NDArray[float64]: ...
@overload
def tensorsolve(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
    axes: None | Iterable[int] =...,
) -> NDArray[floating[Any]]: ...
@overload
def tensorsolve(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
    axes: None | Iterable[int] =...,
) -> NDArray[complexfloating[Any, Any]]: ...

@overload
def solve(
    a: _ArrayLikeInt_co,
    b: _ArrayLikeInt_co,
) -> NDArray[float64]: ...
@overload
def solve(
    a: _ArrayLikeFloat_co,
    b: _ArrayLikeFloat_co,
) -> NDArray[floating[Any]]: ...
@overload
def solve(
    a: _ArrayLikeComplex_co,
    b: _ArrayLikeComplex_co,
) -> NDArray[complexfloating[Any, Any]]: ...

@overload
def tensorinv(
    a: _ArrayLikeInt_co,
    ind: int = ...,
) -> NDArray[float64]: ...
@overload
def tensorinv(
    a: _ArrayLikeFloat_co,
    ind: int = ...,
) -> NDArray[floating[Any]]: ...
@overload
def tensorinv(
    a: _ArrayLikeComplex_co,
    ind: int = ...,
) -> NDArray[complexfloating[Any, Any]]: ...

@overload
def inv(a: _ArrayLikeInt_co) -> NDArray[float64]: ...
@overload
def inv(a: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...
@overload
def inv(a: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

# TODO: The supported input and output dtypes are dependent on the value of `n`.
# For example: `n < 0` always casts integer types to float64
def matrix_power(
    a: _ArrayLikeComplex_co | _ArrayLikeObject_co,
    n: SupportsIndex,
) -> NDArray[Any]: ...

@overload
def cholesky(a: _ArrayLikeInt_co) -> NDArray[float64]: ...
@overload
def cholesky(a: _ArrayLikeFloat_co) -> NDArray[floating[Any]]: ...
@overload
def cholesky(a: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

@overload
def qr(a: _ArrayLikeInt_co, mode: _ModeKind = ...) -> _2Tuple[NDArray[float64]]: ...
@overload
def qr(a: _ArrayLikeFloat_co, mode: _ModeKind = ...) -> _2Tuple[NDArray[floating[Any]]]: ...
@overload
def qr(a: _ArrayLikeComplex_co, mode: _ModeKind = ...) -> _2Tuple[NDArray[complexfloating[Any, Any]]]: ...

@overload
def eigvals(a: _ArrayLikeInt_co) -> NDArray[float64] | NDArray[complex128]: ...
@overload
def eigvals(a: _ArrayLikeFloat_co) -> NDArray[floating[Any]] | NDArray[complexfloating[Any, Any]]: ...
@overload
def eigvals(a: _ArrayLikeComplex_co) -> NDArray[complexfloating[Any, Any]]: ...

@overload
def eigvalsh(a: _ArrayLikeInt_co, UPLO: L["L", "U", "l", "u"] = ...) -> NDArray[float64]: ...
@overload
def eigvalsh(a: _ArrayLikeComplex_co, UPLO: L["L", "U", "l", "u"] = ...) -> NDArray[floating[Any]]: ...

@overload
def eig(a: _ArrayLikeInt_co) -> _2Tuple[NDArray[float64]] | _2Tuple[NDArray[complex128]]: ...
@overload
def eig(a: _ArrayLikeFloat_co) -> _2Tuple[NDArray[floating[Any]]] | _2Tuple[NDArray[complexfloating[Any, Any]]]: ...
@overload
def eig(a: _ArrayLikeComplex_co) -> _2Tuple[NDArray[complexfloating[Any, Any]]]: ...

@overload
def eigh(
    a: _ArrayLikeInt_co,
    UPLO: L["L", "U", "l", "u"] = ...,
) -> tuple[NDArray[float64], NDArray[float64]]: ...
@overload
def eigh(
    a: _ArrayLikeFloat_co,
    UPLO: L["L", "U", "l", "u"] = ...,
) -> tuple[NDArray[floating[Any]], NDArray[floating[Any]]]: ...
@overload
def eigh(
    a: _ArrayLikeComplex_co,
    UPLO: L["L", "U", "l", "u"] = ...,
) -> tuple[NDArray[floating[Any]], NDArray[complexfloating[Any, Any]]]: ...

@overload
def svd(
    a: _ArrayLikeInt_co,
    full_matrices: bool = ...,
    compute_uv: L[True] = ...,
    hermitian: bool = ...,
) -> tuple[
    NDArray[float64],
    NDArray[float64],
    NDArray[float64],
]: ...
@overload
def svd(
    a: _ArrayLikeFloat_co,
    full_matrices: bool = ...,
    compute_uv: L[True] = ...,
    hermitian: bool = ...,
) -> tuple[
    NDArray[floating[Any]],
    NDArray[floating[Any]],
    NDArray[floating[Any]],
]: ...
@overload
def svd(
    a: _ArrayLikeComplex_co,
    full_matrices: bool = ...,
    compute_uv: L[True] = ...,
    hermitian: bool = ...,
) -> tuple[
    NDArray[complexfloating[Any, Any]],
    NDArray[floating[Any]],
    NDArray[complexfloating[Any, Any]],
]: ...
@overload
def svd(
    a: _ArrayLikeInt_co,
    full_matrices: bool = ...,
    compute_uv: L[False] = ...,
    hermitian: bool = ...,
) -> NDArray[float64]: ...
@overload
def svd(
    a: _ArrayLikeComplex_co,
    full_matrices: bool = ...,
    compute_uv: L[False] = ...,
    hermitian: bool = ...,
) -> NDArray[floating[Any]]: ...

# TODO: Returns a scalar for 2D arrays and
# a `(x.ndim - 2)`` dimensionl array otherwise
def cond(x: _ArrayLikeComplex_co, p: None | float | L["fro", "nuc"] = ...) -> Any: ...

# TODO: Returns `int` for <2D arrays and `intp` otherwise
def matrix_rank(
    A: _ArrayLikeComplex_co,
    tol: None | _ArrayLikeFloat_co = ...,
    hermitian: bool = ...,
) -> Any: ...

@overload
def pinv(
    a: _ArrayLikeInt_co,
    rcond: _ArrayLikeFloat_co = ...,
    hermitian: bool = ...,
) -> NDArray[float64]: ...
@overload
def pinv(
    a: _ArrayLikeFloat_co,
    rcond: _ArrayLikeFloat_co = ...,
    hermitian: bool = ...,
) -> NDArray[floating[Any]]: ...
@overload
def pinv(
    a: _ArrayLikeComplex_co,
    rcond: _ArrayLikeFloat_co = ...,
    hermitian: bool = ...,
) -> NDArray[complexfloating[Any, Any]]: ...

# TODO: Returns a 2-tuple of scalars for 2D arrays and
# a 2-tuple of `(a.ndim - 2)`` dimensionl arrays otherwise
def slogdet(a: _ArrayLikeComplex_co) -> _2Tuple[Any]: ...

# TODO: Returns a 2-tuple of scalars for 2D arrays and
# a 2-tuple of `(a.ndim - 2)`` dimensionl arrays otherwise
def det(a: _ArrayLikeComplex_co) -> Any: ...

@overload
def lstsq(a: _ArrayLikeInt_co, b: _ArrayLikeInt_co, rcond: None | float = ...) -> tuple[
    NDArray[float64],
    NDArray[float64],
    int32,
    NDArray[float64],
]: ...
@overload
def lstsq(a: _ArrayLikeFloat_co, b: _ArrayLikeFloat_co, rcond: None | float = ...) -> tuple[
    NDArray[floating[Any]],
    NDArray[floating[Any]],
    int32,
    NDArray[floating[Any]],
]: ...
@overload
def lstsq(a: _ArrayLikeComplex_co, b: _ArrayLikeComplex_co, rcond: None | float = ...) -> tuple[
    NDArray[complexfloating[Any, Any]],
    NDArray[floating[Any]],
    int32,
    NDArray[floating[Any]],
]: ...

@overload
def norm(
    x: ArrayLike,
    ord: None | float | L["fro", "nuc"] = ...,
    axis: None = ...,
    keepdims: bool = ...,
) -> floating[Any]: ...
@overload
def norm(
    x: ArrayLike,
    ord: None | float | L["fro", "nuc"] = ...,
    axis: SupportsInt | SupportsIndex | tuple[int, ...] = ...,
    keepdims: bool = ...,
) -> Any: ...

# TODO: Returns a scalar or array
def multi_dot(
    arrays: Iterable[_ArrayLikeComplex_co | _ArrayLikeObject_co | _ArrayLikeTD64_co],
    *,
    out: None | NDArray[Any] = ...,
) -> Any: ...
