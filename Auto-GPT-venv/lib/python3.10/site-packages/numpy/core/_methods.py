"""
Array methods which are called by both the C-code for the method
and the Python code for the NumPy-namespace function

"""
import warnings
from contextlib import nullcontext

from numpy.core import multiarray as mu
from numpy.core import umath as um
from numpy.core.multiarray import asanyarray
from numpy.core import numerictypes as nt
from numpy.core import _exceptions
from numpy.core._ufunc_config import _no_nep50_warning
from numpy._globals import _NoValue
from numpy.compat import pickle, os_fspath

# save those O(100) nanoseconds!
umr_maximum = um.maximum.reduce
umr_minimum = um.minimum.reduce
umr_sum = um.add.reduce
umr_prod = um.multiply.reduce
umr_any = um.logical_or.reduce
umr_all = um.logical_and.reduce

# Complex types to -> (2,)float view for fast-path computation in _var()
_complex_to_float = {
    nt.dtype(nt.csingle) : nt.dtype(nt.single),
    nt.dtype(nt.cdouble) : nt.dtype(nt.double),
}
# Special case for windows: ensure double takes precedence
if nt.dtype(nt.longdouble) != nt.dtype(nt.double):
    _complex_to_float.update({
        nt.dtype(nt.clongdouble) : nt.dtype(nt.longdouble),
    })

# avoid keyword arguments to speed up parsing, saves about 15%-20% for very
# small reductions
def _amax(a, axis=None, out=None, keepdims=False,
          initial=_NoValue, where=True):
    return umr_maximum(a, axis, None, out, keepdims, initial, where)

def _amin(a, axis=None, out=None, keepdims=False,
          initial=_NoValue, where=True):
    return umr_minimum(a, axis, None, out, keepdims, initial, where)

def _sum(a, axis=None, dtype=None, out=None, keepdims=False,
         initial=_NoValue, where=True):
    return umr_sum(a, axis, dtype, out, keepdims, initial, where)

def _prod(a, axis=None, dtype=None, out=None, keepdims=False,
          initial=_NoValue, where=True):
    return umr_prod(a, axis, dtype, out, keepdims, initial, where)

def _any(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    # Parsing keyword arguments is currently fairly slow, so avoid it for now
    if where is True:
        return umr_any(a, axis, dtype, out, keepdims)
    return umr_any(a, axis, dtype, out, keepdims, where=where)

def _all(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    # Parsing keyword arguments is currently fairly slow, so avoid it for now
    if where is True:
        return umr_all(a, axis, dtype, out, keepdims)
    return umr_all(a, axis, dtype, out, keepdims, where=where)

def _count_reduce_items(arr, axis, keepdims=False, where=True):
    # fast-path for the default case
    if where is True:
        # no boolean mask given, calculate items according to axis
        if axis is None:
            axis = tuple(range(arr.ndim))
        elif not isinstance(axis, tuple):
            axis = (axis,)
        items = 1
        for ax in axis:
            items *= arr.shape[mu.normalize_axis_index(ax, arr.ndim)]
        items = nt.intp(items)
    else:
        # TODO: Optimize case when `where` is broadcast along a non-reduction
        # axis and full sum is more excessive than needed.

        # guarded to protect circular imports
        from numpy.lib.stride_tricks import broadcast_to
        # count True values in (potentially broadcasted) boolean mask
        items = umr_sum(broadcast_to(where, arr.shape), axis, nt.intp, None,
                        keepdims)
    return items

# Numpy 1.17.0, 2019-02-24
# Various clip behavior deprecations, marked with _clip_dep as a prefix.

def _clip_dep_is_scalar_nan(a):
    # guarded to protect circular imports
    from numpy.core.fromnumeric import ndim
    if ndim(a) != 0:
        return False
    try:
        return um.isnan(a)
    except TypeError:
        return False

def _clip_dep_is_byte_swapped(a):
    if isinstance(a, mu.ndarray):
        return not a.dtype.isnative
    return False

def _clip_dep_invoke_with_casting(ufunc, *args, out=None, casting=None, **kwargs):
    # normal path
    if casting is not None:
        return ufunc(*args, out=out, casting=casting, **kwargs)

    # try to deal with broken casting rules
    try:
        return ufunc(*args, out=out, **kwargs)
    except _exceptions._UFuncOutputCastingError as e:
        # Numpy 1.17.0, 2019-02-24
        warnings.warn(
            "Converting the output of clip from {!r} to {!r} is deprecated. "
            "Pass `casting=\"unsafe\"` explicitly to silence this warning, or "
            "correct the type of the variables.".format(e.from_, e.to),
            DeprecationWarning,
            stacklevel=2
        )
        return ufunc(*args, out=out, casting="unsafe", **kwargs)

def _clip(a, min=None, max=None, out=None, *, casting=None, **kwargs):
    if min is None and max is None:
        raise ValueError("One of max or min must be given")

    # Numpy 1.17.0, 2019-02-24
    # This deprecation probably incurs a substantial slowdown for small arrays,
    # it will be good to get rid of it.
    if not _clip_dep_is_byte_swapped(a) and not _clip_dep_is_byte_swapped(out):
        using_deprecated_nan = False
        if _clip_dep_is_scalar_nan(min):
            min = -float('inf')
            using_deprecated_nan = True
        if _clip_dep_is_scalar_nan(max):
            max = float('inf')
            using_deprecated_nan = True
        if using_deprecated_nan:
            warnings.warn(
                "Passing `np.nan` to mean no clipping in np.clip has always "
                "been unreliable, and is now deprecated. "
                "In future, this will always return nan, like it already does "
                "when min or max are arrays that contain nan. "
                "To skip a bound, pass either None or an np.inf of an "
                "appropriate sign.",
                DeprecationWarning,
                stacklevel=2
            )

    if min is None:
        return _clip_dep_invoke_with_casting(
            um.minimum, a, max, out=out, casting=casting, **kwargs)
    elif max is None:
        return _clip_dep_invoke_with_casting(
            um.maximum, a, min, out=out, casting=casting, **kwargs)
    else:
        return _clip_dep_invoke_with_casting(
            um.clip, a, min, max, out=out, casting=casting, **kwargs)

def _mean(a, axis=None, dtype=None, out=None, keepdims=False, *, where=True):
    arr = asanyarray(a)

    is_float16_result = False

    rcount = _count_reduce_items(arr, axis, keepdims=keepdims, where=where)
    if rcount == 0 if where is True else umr_any(rcount == 0, axis=None):
        warnings.warn("Mean of empty slice.", RuntimeWarning, stacklevel=2)

    # Cast bool, unsigned int, and int to float64 by default
    if dtype is None:
        if issubclass(arr.dtype.type, (nt.integer, nt.bool_)):
            dtype = mu.dtype('f8')
        elif issubclass(arr.dtype.type, nt.float16):
            dtype = mu.dtype('f4')
            is_float16_result = True

    ret = umr_sum(arr, axis, dtype, out, keepdims, where=where)
    if isinstance(ret, mu.ndarray):
        with _no_nep50_warning():
            ret = um.true_divide(
                    ret, rcount, out=ret, casting='unsafe', subok=False)
        if is_float16_result and out is None:
            ret = arr.dtype.type(ret)
    elif hasattr(ret, 'dtype'):
        if is_float16_result:
            ret = arr.dtype.type(ret / rcount)
        else:
            ret = ret.dtype.type(ret / rcount)
    else:
        ret = ret / rcount

    return ret

def _var(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *,
         where=True):
    arr = asanyarray(a)

    rcount = _count_reduce_items(arr, axis, keepdims=keepdims, where=where)
    # Make this warning show up on top.
    if ddof >= rcount if where is True else umr_any(ddof >= rcount, axis=None):
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning,
                      stacklevel=2)

    # Cast bool, unsigned int, and int to float64 by default
    if dtype is None and issubclass(arr.dtype.type, (nt.integer, nt.bool_)):
        dtype = mu.dtype('f8')

    # Compute the mean.
    # Note that if dtype is not of inexact type then arraymean will
    # not be either.
    arrmean = umr_sum(arr, axis, dtype, keepdims=True, where=where)
    # The shape of rcount has to match arrmean to not change the shape of out
    # in broadcasting. Otherwise, it cannot be stored back to arrmean.
    if rcount.ndim == 0:
        # fast-path for default case when where is True
        div = rcount
    else:
        # matching rcount to arrmean when where is specified as array
        div = rcount.reshape(arrmean.shape)
    if isinstance(arrmean, mu.ndarray):
        with _no_nep50_warning():
            arrmean = um.true_divide(arrmean, div, out=arrmean,
                                     casting='unsafe', subok=False)
    elif hasattr(arrmean, "dtype"):
        arrmean = arrmean.dtype.type(arrmean / rcount)
    else:
        arrmean = arrmean / rcount

    # Compute sum of squared deviations from mean
    # Note that x may not be inexact and that we need it to be an array,
    # not a scalar.
    x = asanyarray(arr - arrmean)

    if issubclass(arr.dtype.type, (nt.floating, nt.integer)):
        x = um.multiply(x, x, out=x)
    # Fast-paths for built-in complex types
    elif x.dtype in _complex_to_float:
        xv = x.view(dtype=(_complex_to_float[x.dtype], (2,)))
        um.multiply(xv, xv, out=xv)
        x = um.add(xv[..., 0], xv[..., 1], out=x.real).real
    # Most general case; includes handling object arrays containing imaginary
    # numbers and complex types with non-native byteorder
    else:
        x = um.multiply(x, um.conjugate(x), out=x).real

    ret = umr_sum(x, axis, dtype, out, keepdims=keepdims, where=where)

    # Compute degrees of freedom and make sure it is not negative.
    rcount = um.maximum(rcount - ddof, 0)

    # divide by degrees of freedom
    if isinstance(ret, mu.ndarray):
        with _no_nep50_warning():
            ret = um.true_divide(
                    ret, rcount, out=ret, casting='unsafe', subok=False)
    elif hasattr(ret, 'dtype'):
        ret = ret.dtype.type(ret / rcount)
    else:
        ret = ret / rcount

    return ret

def _std(a, axis=None, dtype=None, out=None, ddof=0, keepdims=False, *,
         where=True):
    ret = _var(a, axis=axis, dtype=dtype, out=out, ddof=ddof,
               keepdims=keepdims, where=where)

    if isinstance(ret, mu.ndarray):
        ret = um.sqrt(ret, out=ret)
    elif hasattr(ret, 'dtype'):
        ret = ret.dtype.type(um.sqrt(ret))
    else:
        ret = um.sqrt(ret)

    return ret

def _ptp(a, axis=None, out=None, keepdims=False):
    return um.subtract(
        umr_maximum(a, axis, None, out, keepdims),
        umr_minimum(a, axis, None, None, keepdims),
        out
    )

def _dump(self, file, protocol=2):
    if hasattr(file, 'write'):
        ctx = nullcontext(file)
    else:
        ctx = open(os_fspath(file), "wb")
    with ctx as f:
        pickle.dump(self, f, protocol=protocol)

def _dumps(self, protocol=2):
    return pickle.dumps(self, protocol=protocol)
