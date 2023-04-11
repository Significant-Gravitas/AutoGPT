"""
Module of functions that are like ufuncs in acting on arrays and optionally
storing results in an output array.

"""
__all__ = ['fix', 'isneginf', 'isposinf']

import numpy.core.numeric as nx
from numpy.core.overrides import (
    array_function_dispatch, ARRAY_FUNCTION_ENABLED,
)
import warnings
import functools


def _deprecate_out_named_y(f):
    """
    Allow the out argument to be passed as the name `y` (deprecated)

    In future, this decorator should be removed.
    """
    @functools.wraps(f)
    def func(x, out=None, **kwargs):
        if 'y' in kwargs:
            if 'out' in kwargs:
                raise TypeError(
                    "{} got multiple values for argument 'out'/'y'"
                    .format(f.__name__)
                )
            out = kwargs.pop('y')
            # NumPy 1.13.0, 2017-04-26
            warnings.warn(
                "The name of the out argument to {} has changed from `y` to "
                "`out`, to match other ufuncs.".format(f.__name__),
                DeprecationWarning, stacklevel=3)
        return f(x, out=out, **kwargs)

    return func


def _fix_out_named_y(f):
    """
    Allow the out argument to be passed as the name `y` (deprecated)

    This decorator should only be used if _deprecate_out_named_y is used on
    a corresponding dispatcher function.
    """
    @functools.wraps(f)
    def func(x, out=None, **kwargs):
        if 'y' in kwargs:
            # we already did error checking in _deprecate_out_named_y
            out = kwargs.pop('y')
        return f(x, out=out, **kwargs)

    return func


def _fix_and_maybe_deprecate_out_named_y(f):
    """
    Use the appropriate decorator, depending upon if dispatching is being used.
    """
    if ARRAY_FUNCTION_ENABLED:
        return _fix_out_named_y(f)
    else:
        return _deprecate_out_named_y(f)


@_deprecate_out_named_y
def _dispatcher(x, out=None):
    return (x, out)


@array_function_dispatch(_dispatcher, verify=False, module='numpy')
@_fix_and_maybe_deprecate_out_named_y
def fix(x, out=None):
    """
    Round to nearest integer towards zero.

    Round an array of floats element-wise to nearest integer towards zero.
    The rounded values are returned as floats.

    Parameters
    ----------
    x : array_like
        An array of floats to be rounded
    out : ndarray, optional
        A location into which the result is stored. If provided, it must have
        a shape that the input broadcasts to. If not provided or None, a
        freshly-allocated array is returned.

    Returns
    -------
    out : ndarray of floats
        A float array with the same dimensions as the input.
        If second argument is not supplied then a float array is returned
        with the rounded values.

        If a second argument is supplied the result is stored there.
        The return value `out` is then a reference to that array.

    See Also
    --------
    rint, trunc, floor, ceil
    around : Round to given number of decimals

    Examples
    --------
    >>> np.fix(3.14)
    3.0
    >>> np.fix(3)
    3.0
    >>> np.fix([2.1, 2.9, -2.1, -2.9])
    array([ 2.,  2., -2., -2.])

    """
    # promote back to an array if flattened
    res = nx.asanyarray(nx.ceil(x, out=out))
    res = nx.floor(x, out=res, where=nx.greater_equal(x, 0))

    # when no out argument is passed and no subclasses are involved, flatten
    # scalars
    if out is None and type(res) is nx.ndarray:
        res = res[()]
    return res


@array_function_dispatch(_dispatcher, verify=False, module='numpy')
@_fix_and_maybe_deprecate_out_named_y
def isposinf(x, out=None):
    """
    Test element-wise for positive infinity, return result as bool array.

    Parameters
    ----------
    x : array_like
        The input array.
    out : array_like, optional
        A location into which the result is stored. If provided, it must have a
        shape that the input broadcasts to. If not provided or None, a
        freshly-allocated boolean array is returned.

    Returns
    -------
    out : ndarray
        A boolean array with the same dimensions as the input.
        If second argument is not supplied then a boolean array is returned
        with values True where the corresponding element of the input is
        positive infinity and values False where the element of the input is
        not positive infinity.

        If a second argument is supplied the result is stored there. If the
        type of that array is a numeric type the result is represented as zeros
        and ones, if the type is boolean then as False and True.
        The return value `out` is then a reference to that array.

    See Also
    --------
    isinf, isneginf, isfinite, isnan

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754).

    Errors result if the second argument is also supplied when x is a scalar
    input, if first and second arguments have different shapes, or if the
    first argument has complex values

    Examples
    --------
    >>> np.isposinf(np.PINF)
    True
    >>> np.isposinf(np.inf)
    True
    >>> np.isposinf(np.NINF)
    False
    >>> np.isposinf([-np.inf, 0., np.inf])
    array([False, False,  True])

    >>> x = np.array([-np.inf, 0., np.inf])
    >>> y = np.array([2, 2, 2])
    >>> np.isposinf(x, y)
    array([0, 0, 1])
    >>> y
    array([0, 0, 1])

    """
    is_inf = nx.isinf(x)
    try:
        signbit = ~nx.signbit(x)
    except TypeError as e:
        dtype = nx.asanyarray(x).dtype
        raise TypeError(f'This operation is not supported for {dtype} values '
                        'because it would be ambiguous.') from e
    else:
        return nx.logical_and(is_inf, signbit, out)


@array_function_dispatch(_dispatcher, verify=False, module='numpy')
@_fix_and_maybe_deprecate_out_named_y
def isneginf(x, out=None):
    """
    Test element-wise for negative infinity, return result as bool array.

    Parameters
    ----------
    x : array_like
        The input array.
    out : array_like, optional
        A location into which the result is stored. If provided, it must have a
        shape that the input broadcasts to. If not provided or None, a
        freshly-allocated boolean array is returned.

    Returns
    -------
    out : ndarray
        A boolean array with the same dimensions as the input.
        If second argument is not supplied then a numpy boolean array is
        returned with values True where the corresponding element of the
        input is negative infinity and values False where the element of
        the input is not negative infinity.

        If a second argument is supplied the result is stored there. If the
        type of that array is a numeric type the result is represented as
        zeros and ones, if the type is boolean then as False and True. The
        return value `out` is then a reference to that array.

    See Also
    --------
    isinf, isposinf, isnan, isfinite

    Notes
    -----
    NumPy uses the IEEE Standard for Binary Floating-Point for Arithmetic
    (IEEE 754).

    Errors result if the second argument is also supplied when x is a scalar
    input, if first and second arguments have different shapes, or if the
    first argument has complex values.

    Examples
    --------
    >>> np.isneginf(np.NINF)
    True
    >>> np.isneginf(np.inf)
    False
    >>> np.isneginf(np.PINF)
    False
    >>> np.isneginf([-np.inf, 0., np.inf])
    array([ True, False, False])

    >>> x = np.array([-np.inf, 0., np.inf])
    >>> y = np.array([2, 2, 2])
    >>> np.isneginf(x, y)
    array([1, 0, 0])
    >>> y
    array([1, 0, 0])

    """
    is_inf = nx.isinf(x)
    try:
        signbit = nx.signbit(x)
    except TypeError as e:
        dtype = nx.asanyarray(x).dtype
        raise TypeError(f'This operation is not supported for {dtype} values '
                        'because it would be ambiguous.') from e
    else:
        return nx.logical_and(is_inf, signbit, out)
