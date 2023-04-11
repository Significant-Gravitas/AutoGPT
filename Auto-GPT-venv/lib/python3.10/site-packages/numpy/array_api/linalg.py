from __future__ import annotations

from ._dtypes import _floating_dtypes, _numeric_dtypes
from ._manipulation_functions import reshape
from ._array_object import Array

from ..core.numeric import normalize_axis_tuple

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from ._typing import Literal, Optional, Sequence, Tuple, Union

from typing import NamedTuple

import numpy.linalg
import numpy as np

class EighResult(NamedTuple):
    eigenvalues: Array
    eigenvectors: Array

class QRResult(NamedTuple):
    Q: Array
    R: Array

class SlogdetResult(NamedTuple):
    sign: Array
    logabsdet: Array

class SVDResult(NamedTuple):
    U: Array
    S: Array
    Vh: Array

# Note: the inclusion of the upper keyword is different from
# np.linalg.cholesky, which does not have it.
def cholesky(x: Array, /, *, upper: bool = False) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.cholesky <numpy.linalg.cholesky>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.cholesky.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in cholesky')
    L = np.linalg.cholesky(x._array)
    if upper:
        return Array._new(L).mT
    return Array._new(L)

# Note: cross is the numpy top-level namespace, not np.linalg
def cross(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.cross <numpy.cross>`.

    See its docstring for more information.
    """
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in cross')
    # Note: this is different from np.cross(), which broadcasts
    if x1.shape != x2.shape:
        raise ValueError('x1 and x2 must have the same shape')
    if x1.ndim == 0:
        raise ValueError('cross() requires arrays of dimension at least 1')
    # Note: this is different from np.cross(), which allows dimension 2
    if x1.shape[axis] != 3:
        raise ValueError('cross() dimension must equal 3')
    return Array._new(np.cross(x1._array, x2._array, axis=axis))

def det(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.det <numpy.linalg.det>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.det.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in det')
    return Array._new(np.linalg.det(x._array))

# Note: diagonal is the numpy top-level namespace, not np.linalg
def diagonal(x: Array, /, *, offset: int = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.diagonal <numpy.diagonal>`.

    See its docstring for more information.
    """
    # Note: diagonal always operates on the last two axes, whereas np.diagonal
    # operates on the first two axes by default
    return Array._new(np.diagonal(x._array, offset=offset, axis1=-2, axis2=-1))


def eigh(x: Array, /) -> EighResult:
    """
    Array API compatible wrapper for :py:func:`np.linalg.eigh <numpy.linalg.eigh>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.eigh.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in eigh')

    # Note: the return type here is a namedtuple, which is different from
    # np.eigh, which only returns a tuple.
    return EighResult(*map(Array._new, np.linalg.eigh(x._array)))


def eigvalsh(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.eigvalsh <numpy.linalg.eigvalsh>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.eigvalsh.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in eigvalsh')

    return Array._new(np.linalg.eigvalsh(x._array))

def inv(x: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.inv <numpy.linalg.inv>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.inv.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in inv')

    return Array._new(np.linalg.inv(x._array))


# Note: matmul is the numpy top-level namespace but not in np.linalg
def matmul(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.matmul <numpy.matmul>`.

    See its docstring for more information.
    """
    # Note: the restriction to numeric dtypes only is different from
    # np.matmul.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in matmul')

    return Array._new(np.matmul(x1._array, x2._array))


# Note: the name here is different from norm(). The array API norm is split
# into matrix_norm and vector_norm().

# The type for ord should be Optional[Union[int, float, Literal[np.inf,
# -np.inf, 'fro', 'nuc']]], but Literal does not support floating-point
# literals.
def matrix_norm(x: Array, /, *, keepdims: bool = False, ord: Optional[Union[int, float, Literal['fro', 'nuc']]] = 'fro') -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.norm <numpy.linalg.norm>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.norm.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in matrix_norm')

    return Array._new(np.linalg.norm(x._array, axis=(-2, -1), keepdims=keepdims, ord=ord))


def matrix_power(x: Array, n: int, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.matrix_power <numpy.matrix_power>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.matrix_power.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed for the first argument of matrix_power')

    # np.matrix_power already checks if n is an integer
    return Array._new(np.linalg.matrix_power(x._array, n))

# Note: the keyword argument name rtol is different from np.linalg.matrix_rank
def matrix_rank(x: Array, /, *, rtol: Optional[Union[float, Array]] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.matrix_rank <numpy.matrix_rank>`.

    See its docstring for more information.
    """
    # Note: this is different from np.linalg.matrix_rank, which supports 1
    # dimensional arrays.
    if x.ndim < 2:
        raise np.linalg.LinAlgError("1-dimensional array given. Array must be at least two-dimensional")
    S = np.linalg.svd(x._array, compute_uv=False)
    if rtol is None:
        tol = S.max(axis=-1, keepdims=True) * max(x.shape[-2:]) * np.finfo(S.dtype).eps
    else:
        if isinstance(rtol, Array):
            rtol = rtol._array
        # Note: this is different from np.linalg.matrix_rank, which does not multiply
        # the tolerance by the largest singular value.
        tol = S.max(axis=-1, keepdims=True)*np.asarray(rtol)[..., np.newaxis]
    return Array._new(np.count_nonzero(S > tol, axis=-1))


# Note: this function is new in the array API spec. Unlike transpose, it only
# transposes the last two axes.
def matrix_transpose(x: Array, /) -> Array:
    if x.ndim < 2:
        raise ValueError("x must be at least 2-dimensional for matrix_transpose")
    return Array._new(np.swapaxes(x._array, -1, -2))

# Note: outer is the numpy top-level namespace, not np.linalg
def outer(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.outer <numpy.outer>`.

    See its docstring for more information.
    """
    # Note: the restriction to numeric dtypes only is different from
    # np.outer.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in outer')

    # Note: the restriction to only 1-dim arrays is different from np.outer
    if x1.ndim != 1 or x2.ndim != 1:
        raise ValueError('The input arrays to outer must be 1-dimensional')

    return Array._new(np.outer(x1._array, x2._array))

# Note: the keyword argument name rtol is different from np.linalg.pinv
def pinv(x: Array, /, *, rtol: Optional[Union[float, Array]] = None) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.pinv <numpy.linalg.pinv>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.pinv.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in pinv')

    # Note: this is different from np.linalg.pinv, which does not multiply the
    # default tolerance by max(M, N).
    if rtol is None:
        rtol = max(x.shape[-2:]) * np.finfo(x.dtype).eps
    return Array._new(np.linalg.pinv(x._array, rcond=rtol))

def qr(x: Array, /, *, mode: Literal['reduced', 'complete'] = 'reduced') -> QRResult:
    """
    Array API compatible wrapper for :py:func:`np.linalg.qr <numpy.linalg.qr>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.qr.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in qr')

    # Note: the return type here is a namedtuple, which is different from
    # np.linalg.qr, which only returns a tuple.
    return QRResult(*map(Array._new, np.linalg.qr(x._array, mode=mode)))

def slogdet(x: Array, /) -> SlogdetResult:
    """
    Array API compatible wrapper for :py:func:`np.linalg.slogdet <numpy.linalg.slogdet>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.slogdet.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in slogdet')

    # Note: the return type here is a namedtuple, which is different from
    # np.linalg.slogdet, which only returns a tuple.
    return SlogdetResult(*map(Array._new, np.linalg.slogdet(x._array)))

# Note: unlike np.linalg.solve, the array API solve() only accepts x2 as a
# vector when it is exactly 1-dimensional. All other cases treat x2 as a stack
# of matrices. The np.linalg.solve behavior of allowing stacks of both
# matrices and vectors is ambiguous c.f.
# https://github.com/numpy/numpy/issues/15349 and
# https://github.com/data-apis/array-api/issues/285.

# To workaround this, the below is the code from np.linalg.solve except
# only calling solve1 in the exactly 1D case.
def _solve(a, b):
    from ..linalg.linalg import (_makearray, _assert_stacked_2d,
                                 _assert_stacked_square, _commonType,
                                 isComplexType, get_linalg_error_extobj,
                                 _raise_linalgerror_singular)
    from ..linalg import _umath_linalg

    a, _ = _makearray(a)
    _assert_stacked_2d(a)
    _assert_stacked_square(a)
    b, wrap = _makearray(b)
    t, result_t = _commonType(a, b)

    # This part is different from np.linalg.solve
    if b.ndim == 1:
        gufunc = _umath_linalg.solve1
    else:
        gufunc = _umath_linalg.solve

    # This does nothing currently but is left in because it will be relevant
    # when complex dtype support is added to the spec in 2022.
    signature = 'DD->D' if isComplexType(t) else 'dd->d'
    extobj = get_linalg_error_extobj(_raise_linalgerror_singular)
    r = gufunc(a, b, signature=signature, extobj=extobj)

    return wrap(r.astype(result_t, copy=False))

def solve(x1: Array, x2: Array, /) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.solve <numpy.linalg.solve>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.solve.
    if x1.dtype not in _floating_dtypes or x2.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in solve')

    return Array._new(_solve(x1._array, x2._array))

def svd(x: Array, /, *, full_matrices: bool = True) -> SVDResult:
    """
    Array API compatible wrapper for :py:func:`np.linalg.svd <numpy.linalg.svd>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.svd.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in svd')

    # Note: the return type here is a namedtuple, which is different from
    # np.svd, which only returns a tuple.
    return SVDResult(*map(Array._new, np.linalg.svd(x._array, full_matrices=full_matrices)))

# Note: svdvals is not in NumPy (but it is in SciPy). It is equivalent to
# np.linalg.svd(compute_uv=False).
def svdvals(x: Array, /) -> Union[Array, Tuple[Array, ...]]:
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in svdvals')
    return Array._new(np.linalg.svd(x._array, compute_uv=False))

# Note: tensordot is the numpy top-level namespace but not in np.linalg

# Note: axes must be a tuple, unlike np.tensordot where it can be an array or array-like.
def tensordot(x1: Array, x2: Array, /, *, axes: Union[int, Tuple[Sequence[int], Sequence[int]]] = 2) -> Array:
    # Note: the restriction to numeric dtypes only is different from
    # np.tensordot.
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in tensordot')

    return Array._new(np.tensordot(x1._array, x2._array, axes=axes))

# Note: trace is the numpy top-level namespace, not np.linalg
def trace(x: Array, /, *, offset: int = 0) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.trace <numpy.trace>`.

    See its docstring for more information.
    """
    if x.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in trace')
    # Note: trace always operates on the last two axes, whereas np.trace
    # operates on the first two axes by default
    return Array._new(np.asarray(np.trace(x._array, offset=offset, axis1=-2, axis2=-1)))

# Note: vecdot is not in NumPy
def vecdot(x1: Array, x2: Array, /, *, axis: int = -1) -> Array:
    if x1.dtype not in _numeric_dtypes or x2.dtype not in _numeric_dtypes:
        raise TypeError('Only numeric dtypes are allowed in vecdot')
    ndim = max(x1.ndim, x2.ndim)
    x1_shape = (1,)*(ndim - x1.ndim) + tuple(x1.shape)
    x2_shape = (1,)*(ndim - x2.ndim) + tuple(x2.shape)
    if x1_shape[axis] != x2_shape[axis]:
        raise ValueError("x1 and x2 must have the same size along the given axis")

    x1_, x2_ = np.broadcast_arrays(x1._array, x2._array)
    x1_ = np.moveaxis(x1_, axis, -1)
    x2_ = np.moveaxis(x2_, axis, -1)

    res = x1_[..., None, :] @ x2_[..., None]
    return Array._new(res[..., 0, 0])


# Note: the name here is different from norm(). The array API norm is split
# into matrix_norm and vector_norm().

# The type for ord should be Optional[Union[int, float, Literal[np.inf,
# -np.inf]]] but Literal does not support floating-point literals.
def vector_norm(x: Array, /, *, axis: Optional[Union[int, Tuple[int, ...]]] = None, keepdims: bool = False, ord: Optional[Union[int, float]] = 2) -> Array:
    """
    Array API compatible wrapper for :py:func:`np.linalg.norm <numpy.linalg.norm>`.

    See its docstring for more information.
    """
    # Note: the restriction to floating-point dtypes only is different from
    # np.linalg.norm.
    if x.dtype not in _floating_dtypes:
        raise TypeError('Only floating-point dtypes are allowed in norm')

    # np.linalg.norm tries to do a matrix norm whenever axis is a 2-tuple or
    # when axis=None and the input is 2-D, so to force a vector norm, we make
    # it so the input is 1-D (for axis=None), or reshape so that norm is done
    # on a single dimension.
    a = x._array
    if axis is None:
        # Note: np.linalg.norm() doesn't handle 0-D arrays
        a = a.ravel()
        _axis = 0
    elif isinstance(axis, tuple):
        # Note: The axis argument supports any number of axes, whereas
        # np.linalg.norm() only supports a single axis for vector norm.
        normalized_axis = normalize_axis_tuple(axis, x.ndim)
        rest = tuple(i for i in range(a.ndim) if i not in normalized_axis)
        newshape = axis + rest
        a = np.transpose(a, newshape).reshape(
            (np.prod([a.shape[i] for i in axis], dtype=int), *[a.shape[i] for i in rest]))
        _axis = 0
    else:
        _axis = axis

    res = Array._new(np.linalg.norm(a, axis=_axis, ord=ord))

    if keepdims:
        # We can't reuse np.linalg.norm(keepdims) because of the reshape hacks
        # above to avoid matrix norm logic.
        shape = list(x.shape)
        _axis = normalize_axis_tuple(range(x.ndim) if axis is None else axis, x.ndim)
        for i in _axis:
            shape[i] = 1
        res = reshape(res, tuple(shape))

    return res

__all__ = ['cholesky', 'cross', 'det', 'diagonal', 'eigh', 'eigvalsh', 'inv', 'matmul', 'matrix_norm', 'matrix_power', 'matrix_rank', 'matrix_transpose', 'outer', 'pinv', 'qr', 'slogdet', 'solve', 'svd', 'svdvals', 'tensordot', 'trace', 'vecdot', 'vector_norm']
