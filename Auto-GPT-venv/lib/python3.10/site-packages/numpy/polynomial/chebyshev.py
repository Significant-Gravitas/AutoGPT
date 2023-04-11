"""
====================================================
Chebyshev Series (:mod:`numpy.polynomial.chebyshev`)
====================================================

This module provides a number of objects (mostly functions) useful for
dealing with Chebyshev series, including a `Chebyshev` class that
encapsulates the usual arithmetic operations.  (General information
on how this module represents and works with such polynomials is in the
docstring for its "parent" sub-package, `numpy.polynomial`).

Classes
-------

.. autosummary::
   :toctree: generated/

   Chebyshev


Constants
---------

.. autosummary::
   :toctree: generated/

   chebdomain
   chebzero
   chebone
   chebx

Arithmetic
----------

.. autosummary::
   :toctree: generated/

   chebadd
   chebsub
   chebmulx
   chebmul
   chebdiv
   chebpow
   chebval
   chebval2d
   chebval3d
   chebgrid2d
   chebgrid3d

Calculus
--------

.. autosummary::
   :toctree: generated/

   chebder
   chebint

Misc Functions
--------------

.. autosummary::
   :toctree: generated/

   chebfromroots
   chebroots
   chebvander
   chebvander2d
   chebvander3d
   chebgauss
   chebweight
   chebcompanion
   chebfit
   chebpts1
   chebpts2
   chebtrim
   chebline
   cheb2poly
   poly2cheb
   chebinterpolate

See also
--------
`numpy.polynomial`

Notes
-----
The implementations of multiplication, division, integration, and
differentiation use the algebraic identities [1]_:

.. math::
    T_n(x) = \\frac{z^n + z^{-n}}{2} \\\\
    z\\frac{dx}{dz} = \\frac{z - z^{-1}}{2}.

where

.. math:: x = \\frac{z + z^{-1}}{2}.

These identities allow a Chebyshev series to be expressed as a finite,
symmetric Laurent series.  In this module, this sort of Laurent series
is referred to as a "z-series."

References
----------
.. [1] A. T. Benjamin, et al., "Combinatorial Trigonometry with Chebyshev
  Polynomials," *Journal of Statistical Planning and Inference 14*, 2008
  (https://web.archive.org/web/20080221202153/https://www.math.hmc.edu/~benjamin/papers/CombTrig.pdf, pg. 4)

"""
import numpy as np
import numpy.linalg as la
from numpy.core.multiarray import normalize_axis_index

from . import polyutils as pu
from ._polybase import ABCPolyBase

__all__ = [
    'chebzero', 'chebone', 'chebx', 'chebdomain', 'chebline', 'chebadd',
    'chebsub', 'chebmulx', 'chebmul', 'chebdiv', 'chebpow', 'chebval',
    'chebder', 'chebint', 'cheb2poly', 'poly2cheb', 'chebfromroots',
    'chebvander', 'chebfit', 'chebtrim', 'chebroots', 'chebpts1',
    'chebpts2', 'Chebyshev', 'chebval2d', 'chebval3d', 'chebgrid2d',
    'chebgrid3d', 'chebvander2d', 'chebvander3d', 'chebcompanion',
    'chebgauss', 'chebweight', 'chebinterpolate']

chebtrim = pu.trimcoef

#
# A collection of functions for manipulating z-series. These are private
# functions and do minimal error checking.
#

def _cseries_to_zseries(c):
    """Convert Chebyshev series to z-series.

    Convert a Chebyshev series to the equivalent z-series. The result is
    never an empty array. The dtype of the return is the same as that of
    the input. No checks are run on the arguments as this routine is for
    internal use.

    Parameters
    ----------
    c : 1-D ndarray
        Chebyshev coefficients, ordered from low to high

    Returns
    -------
    zs : 1-D ndarray
        Odd length symmetric z-series, ordered from  low to high.

    """
    n = c.size
    zs = np.zeros(2*n-1, dtype=c.dtype)
    zs[n-1:] = c/2
    return zs + zs[::-1]


def _zseries_to_cseries(zs):
    """Convert z-series to a Chebyshev series.

    Convert a z series to the equivalent Chebyshev series. The result is
    never an empty array. The dtype of the return is the same as that of
    the input. No checks are run on the arguments as this routine is for
    internal use.

    Parameters
    ----------
    zs : 1-D ndarray
        Odd length symmetric z-series, ordered from  low to high.

    Returns
    -------
    c : 1-D ndarray
        Chebyshev coefficients, ordered from  low to high.

    """
    n = (zs.size + 1)//2
    c = zs[n-1:].copy()
    c[1:n] *= 2
    return c


def _zseries_mul(z1, z2):
    """Multiply two z-series.

    Multiply two z-series to produce a z-series.

    Parameters
    ----------
    z1, z2 : 1-D ndarray
        The arrays must be 1-D but this is not checked.

    Returns
    -------
    product : 1-D ndarray
        The product z-series.

    Notes
    -----
    This is simply convolution. If symmetric/anti-symmetric z-series are
    denoted by S/A then the following rules apply:

    S*S, A*A -> S
    S*A, A*S -> A

    """
    return np.convolve(z1, z2)


def _zseries_div(z1, z2):
    """Divide the first z-series by the second.

    Divide `z1` by `z2` and return the quotient and remainder as z-series.
    Warning: this implementation only applies when both z1 and z2 have the
    same symmetry, which is sufficient for present purposes.

    Parameters
    ----------
    z1, z2 : 1-D ndarray
        The arrays must be 1-D and have the same symmetry, but this is not
        checked.

    Returns
    -------

    (quotient, remainder) : 1-D ndarrays
        Quotient and remainder as z-series.

    Notes
    -----
    This is not the same as polynomial division on account of the desired form
    of the remainder. If symmetric/anti-symmetric z-series are denoted by S/A
    then the following rules apply:

    S/S -> S,S
    A/A -> S,A

    The restriction to types of the same symmetry could be fixed but seems like
    unneeded generality. There is no natural form for the remainder in the case
    where there is no symmetry.

    """
    z1 = z1.copy()
    z2 = z2.copy()
    lc1 = len(z1)
    lc2 = len(z2)
    if lc2 == 1:
        z1 /= z2
        return z1, z1[:1]*0
    elif lc1 < lc2:
        return z1[:1]*0, z1
    else:
        dlen = lc1 - lc2
        scl = z2[0]
        z2 /= scl
        quo = np.empty(dlen + 1, dtype=z1.dtype)
        i = 0
        j = dlen
        while i < j:
            r = z1[i]
            quo[i] = z1[i]
            quo[dlen - i] = r
            tmp = r*z2
            z1[i:i+lc2] -= tmp
            z1[j:j+lc2] -= tmp
            i += 1
            j -= 1
        r = z1[i]
        quo[i] = r
        tmp = r*z2
        z1[i:i+lc2] -= tmp
        quo /= scl
        rem = z1[i+1:i-1+lc2].copy()
        return quo, rem


def _zseries_der(zs):
    """Differentiate a z-series.

    The derivative is with respect to x, not z. This is achieved using the
    chain rule and the value of dx/dz given in the module notes.

    Parameters
    ----------
    zs : z-series
        The z-series to differentiate.

    Returns
    -------
    derivative : z-series
        The derivative

    Notes
    -----
    The zseries for x (ns) has been multiplied by two in order to avoid
    using floats that are incompatible with Decimal and likely other
    specialized scalar types. This scaling has been compensated by
    multiplying the value of zs by two also so that the two cancels in the
    division.

    """
    n = len(zs)//2
    ns = np.array([-1, 0, 1], dtype=zs.dtype)
    zs *= np.arange(-n, n+1)*2
    d, r = _zseries_div(zs, ns)
    return d


def _zseries_int(zs):
    """Integrate a z-series.

    The integral is with respect to x, not z. This is achieved by a change
    of variable using dx/dz given in the module notes.

    Parameters
    ----------
    zs : z-series
        The z-series to integrate

    Returns
    -------
    integral : z-series
        The indefinite integral

    Notes
    -----
    The zseries for x (ns) has been multiplied by two in order to avoid
    using floats that are incompatible with Decimal and likely other
    specialized scalar types. This scaling has been compensated by
    dividing the resulting zs by two.

    """
    n = 1 + len(zs)//2
    ns = np.array([-1, 0, 1], dtype=zs.dtype)
    zs = _zseries_mul(zs, ns)
    div = np.arange(-n, n+1)*2
    zs[:n] /= div[:n]
    zs[n+1:] /= div[n+1:]
    zs[n] = 0
    return zs

#
# Chebyshev series functions
#


def poly2cheb(pol):
    """
    Convert a polynomial to a Chebyshev series.

    Convert an array representing the coefficients of a polynomial (relative
    to the "standard" basis) ordered from lowest degree to highest, to an
    array of the coefficients of the equivalent Chebyshev series, ordered
    from lowest to highest degree.

    Parameters
    ----------
    pol : array_like
        1-D array containing the polynomial coefficients

    Returns
    -------
    c : ndarray
        1-D array containing the coefficients of the equivalent Chebyshev
        series.

    See Also
    --------
    cheb2poly

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy import polynomial as P
    >>> p = P.Polynomial(range(4))
    >>> p
    Polynomial([0., 1., 2., 3.], domain=[-1,  1], window=[-1,  1])
    >>> c = p.convert(kind=P.Chebyshev)
    >>> c
    Chebyshev([1.  , 3.25, 1.  , 0.75], domain=[-1.,  1.], window=[-1.,  1.])
    >>> P.chebyshev.poly2cheb(range(4))
    array([1.  , 3.25, 1.  , 0.75])

    """
    [pol] = pu.as_series([pol])
    deg = len(pol) - 1
    res = 0
    for i in range(deg, -1, -1):
        res = chebadd(chebmulx(res), pol[i])
    return res


def cheb2poly(c):
    """
    Convert a Chebyshev series to a polynomial.

    Convert an array representing the coefficients of a Chebyshev series,
    ordered from lowest degree to highest, to an array of the coefficients
    of the equivalent polynomial (relative to the "standard" basis) ordered
    from lowest to highest degree.

    Parameters
    ----------
    c : array_like
        1-D array containing the Chebyshev series coefficients, ordered
        from lowest order term to highest.

    Returns
    -------
    pol : ndarray
        1-D array containing the coefficients of the equivalent polynomial
        (relative to the "standard" basis) ordered from lowest order term
        to highest.

    See Also
    --------
    poly2cheb

    Notes
    -----
    The easy way to do conversions between polynomial basis sets
    is to use the convert method of a class instance.

    Examples
    --------
    >>> from numpy import polynomial as P
    >>> c = P.Chebyshev(range(4))
    >>> c
    Chebyshev([0., 1., 2., 3.], domain=[-1,  1], window=[-1,  1])
    >>> p = c.convert(kind=P.Polynomial)
    >>> p
    Polynomial([-2., -8.,  4., 12.], domain=[-1.,  1.], window=[-1.,  1.])
    >>> P.chebyshev.cheb2poly(range(4))
    array([-2.,  -8.,   4.,  12.])

    """
    from .polynomial import polyadd, polysub, polymulx

    [c] = pu.as_series([c])
    n = len(c)
    if n < 3:
        return c
    else:
        c0 = c[-2]
        c1 = c[-1]
        # i is the current degree of c1
        for i in range(n - 1, 1, -1):
            tmp = c0
            c0 = polysub(c[i - 2], c1)
            c1 = polyadd(tmp, polymulx(c1)*2)
        return polyadd(c0, polymulx(c1))


#
# These are constant arrays are of integer type so as to be compatible
# with the widest range of other types, such as Decimal.
#

# Chebyshev default domain.
chebdomain = np.array([-1, 1])

# Chebyshev coefficients representing zero.
chebzero = np.array([0])

# Chebyshev coefficients representing one.
chebone = np.array([1])

# Chebyshev coefficients representing the identity x.
chebx = np.array([0, 1])


def chebline(off, scl):
    """
    Chebyshev series whose graph is a straight line.

    Parameters
    ----------
    off, scl : scalars
        The specified line is given by ``off + scl*x``.

    Returns
    -------
    y : ndarray
        This module's representation of the Chebyshev series for
        ``off + scl*x``.

    See Also
    --------
    numpy.polynomial.polynomial.polyline
    numpy.polynomial.legendre.legline
    numpy.polynomial.laguerre.lagline
    numpy.polynomial.hermite.hermline
    numpy.polynomial.hermite_e.hermeline

    Examples
    --------
    >>> import numpy.polynomial.chebyshev as C
    >>> C.chebline(3,2)
    array([3, 2])
    >>> C.chebval(-3, C.chebline(3,2)) # should be -3
    -3.0

    """
    if scl != 0:
        return np.array([off, scl])
    else:
        return np.array([off])


def chebfromroots(roots):
    """
    Generate a Chebyshev series with given roots.

    The function returns the coefficients of the polynomial

    .. math:: p(x) = (x - r_0) * (x - r_1) * ... * (x - r_n),

    in Chebyshev form, where the `r_n` are the roots specified in `roots`.
    If a zero has multiplicity n, then it must appear in `roots` n times.
    For instance, if 2 is a root of multiplicity three and 3 is a root of
    multiplicity 2, then `roots` looks something like [2, 2, 2, 3, 3]. The
    roots can appear in any order.

    If the returned coefficients are `c`, then

    .. math:: p(x) = c_0 + c_1 * T_1(x) + ... +  c_n * T_n(x)

    The coefficient of the last term is not generally 1 for monic
    polynomials in Chebyshev form.

    Parameters
    ----------
    roots : array_like
        Sequence containing the roots.

    Returns
    -------
    out : ndarray
        1-D array of coefficients.  If all roots are real then `out` is a
        real array, if some of the roots are complex, then `out` is complex
        even if all the coefficients in the result are real (see Examples
        below).

    See Also
    --------
    numpy.polynomial.polynomial.polyfromroots
    numpy.polynomial.legendre.legfromroots
    numpy.polynomial.laguerre.lagfromroots
    numpy.polynomial.hermite.hermfromroots
    numpy.polynomial.hermite_e.hermefromroots

    Examples
    --------
    >>> import numpy.polynomial.chebyshev as C
    >>> C.chebfromroots((-1,0,1)) # x^3 - x relative to the standard basis
    array([ 0.  , -0.25,  0.  ,  0.25])
    >>> j = complex(0,1)
    >>> C.chebfromroots((-j,j)) # x^2 + 1 relative to the standard basis
    array([1.5+0.j, 0. +0.j, 0.5+0.j])

    """
    return pu._fromroots(chebline, chebmul, roots)


def chebadd(c1, c2):
    """
    Add one Chebyshev series to another.

    Returns the sum of two Chebyshev series `c1` + `c2`.  The arguments
    are sequences of coefficients ordered from lowest order term to
    highest, i.e., [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the Chebyshev series of their sum.

    See Also
    --------
    chebsub, chebmulx, chebmul, chebdiv, chebpow

    Notes
    -----
    Unlike multiplication, division, etc., the sum of two Chebyshev series
    is a Chebyshev series (without having to "reproject" the result onto
    the basis set) so addition, just like that of "standard" polynomials,
    is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebadd(c1,c2)
    array([4., 4., 4.])

    """
    return pu._add(c1, c2)


def chebsub(c1, c2):
    """
    Subtract one Chebyshev series from another.

    Returns the difference of two Chebyshev series `c1` - `c2`.  The
    sequences of coefficients are from lowest order term to highest, i.e.,
    [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Chebyshev series coefficients representing their difference.

    See Also
    --------
    chebadd, chebmulx, chebmul, chebdiv, chebpow

    Notes
    -----
    Unlike multiplication, division, etc., the difference of two Chebyshev
    series is a Chebyshev series (without having to "reproject" the result
    onto the basis set) so subtraction, just like that of "standard"
    polynomials, is simply "component-wise."

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebsub(c1,c2)
    array([-2.,  0.,  2.])
    >>> C.chebsub(c2,c1) # -C.chebsub(c1,c2)
    array([ 2.,  0., -2.])

    """
    return pu._sub(c1, c2)


def chebmulx(c):
    """Multiply a Chebyshev series by x.

    Multiply the polynomial `c` by x, where x is the independent
    variable.


    Parameters
    ----------
    c : array_like
        1-D array of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Array representing the result of the multiplication.

    Notes
    -----

    .. versionadded:: 1.5.0

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> C.chebmulx([1,2,3])
    array([1. , 2.5, 1. , 1.5])

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    # The zero series needs special treatment
    if len(c) == 1 and c[0] == 0:
        return c

    prd = np.empty(len(c) + 1, dtype=c.dtype)
    prd[0] = c[0]*0
    prd[1] = c[0]
    if len(c) > 1:
        tmp = c[1:]/2
        prd[2:] = tmp
        prd[0:-2] += tmp
    return prd


def chebmul(c1, c2):
    """
    Multiply one Chebyshev series by another.

    Returns the product of two Chebyshev series `c1` * `c2`.  The arguments
    are sequences of coefficients, from lowest order "term" to highest,
    e.g., [1,2,3] represents the series ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    out : ndarray
        Of Chebyshev series coefficients representing their product.

    See Also
    --------
    chebadd, chebsub, chebmulx, chebdiv, chebpow

    Notes
    -----
    In general, the (polynomial) product of two C-series results in terms
    that are not in the Chebyshev polynomial basis set.  Thus, to express
    the product as a C-series, it is typically necessary to "reproject"
    the product onto said basis set, which typically produces
    "unintuitive live" (but correct) results; see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebmul(c1,c2) # multiplication requires "reprojection"
    array([  6.5,  12. ,  12. ,   4. ,   1.5])

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    z1 = _cseries_to_zseries(c1)
    z2 = _cseries_to_zseries(c2)
    prd = _zseries_mul(z1, z2)
    ret = _zseries_to_cseries(prd)
    return pu.trimseq(ret)


def chebdiv(c1, c2):
    """
    Divide one Chebyshev series by another.

    Returns the quotient-with-remainder of two Chebyshev series
    `c1` / `c2`.  The arguments are sequences of coefficients from lowest
    order "term" to highest, e.g., [1,2,3] represents the series
    ``T_0 + 2*T_1 + 3*T_2``.

    Parameters
    ----------
    c1, c2 : array_like
        1-D arrays of Chebyshev series coefficients ordered from low to
        high.

    Returns
    -------
    [quo, rem] : ndarrays
        Of Chebyshev series coefficients representing the quotient and
        remainder.

    See Also
    --------
    chebadd, chebsub, chebmulx, chebmul, chebpow

    Notes
    -----
    In general, the (polynomial) division of one C-series by another
    results in quotient and remainder terms that are not in the Chebyshev
    polynomial basis set.  Thus, to express these results as C-series, it
    is typically necessary to "reproject" the results onto said basis
    set, which typically produces "unintuitive" (but correct) results;
    see Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c1 = (1,2,3)
    >>> c2 = (3,2,1)
    >>> C.chebdiv(c1,c2) # quotient "intuitive," remainder not
    (array([3.]), array([-8., -4.]))
    >>> c2 = (0,1,2,3)
    >>> C.chebdiv(c2,c1) # neither "intuitive"
    (array([0., 2.]), array([-2., -4.]))

    """
    # c1, c2 are trimmed copies
    [c1, c2] = pu.as_series([c1, c2])
    if c2[-1] == 0:
        raise ZeroDivisionError()

    # note: this is more efficient than `pu._div(chebmul, c1, c2)`
    lc1 = len(c1)
    lc2 = len(c2)
    if lc1 < lc2:
        return c1[:1]*0, c1
    elif lc2 == 1:
        return c1/c2[-1], c1[:1]*0
    else:
        z1 = _cseries_to_zseries(c1)
        z2 = _cseries_to_zseries(c2)
        quo, rem = _zseries_div(z1, z2)
        quo = pu.trimseq(_zseries_to_cseries(quo))
        rem = pu.trimseq(_zseries_to_cseries(rem))
        return quo, rem


def chebpow(c, pow, maxpower=16):
    """Raise a Chebyshev series to a power.

    Returns the Chebyshev series `c` raised to the power `pow`. The
    argument `c` is a sequence of coefficients ordered from low to high.
    i.e., [1,2,3] is the series  ``T_0 + 2*T_1 + 3*T_2.``

    Parameters
    ----------
    c : array_like
        1-D array of Chebyshev series coefficients ordered from low to
        high.
    pow : integer
        Power to which the series will be raised
    maxpower : integer, optional
        Maximum power allowed. This is mainly to limit growth of the series
        to unmanageable size. Default is 16

    Returns
    -------
    coef : ndarray
        Chebyshev series of power.

    See Also
    --------
    chebadd, chebsub, chebmulx, chebmul, chebdiv

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> C.chebpow([1, 2, 3, 4], 2)
    array([15.5, 22. , 16. , ..., 12.5, 12. ,  8. ])

    """
    # note: this is more efficient than `pu._pow(chebmul, c1, c2)`, as it
    # avoids converting between z and c series repeatedly

    # c is a trimmed copy
    [c] = pu.as_series([c])
    power = int(pow)
    if power != pow or power < 0:
        raise ValueError("Power must be a non-negative integer.")
    elif maxpower is not None and power > maxpower:
        raise ValueError("Power is too large")
    elif power == 0:
        return np.array([1], dtype=c.dtype)
    elif power == 1:
        return c
    else:
        # This can be made more efficient by using powers of two
        # in the usual way.
        zs = _cseries_to_zseries(c)
        prd = zs
        for i in range(2, power + 1):
            prd = np.convolve(prd, zs)
        return _zseries_to_cseries(prd)


def chebder(c, m=1, scl=1, axis=0):
    """
    Differentiate a Chebyshev series.

    Returns the Chebyshev series coefficients `c` differentiated `m` times
    along `axis`.  At each iteration the result is multiplied by `scl` (the
    scaling factor is for use in a linear change of variable). The argument
    `c` is an array of coefficients from low to high degree along each
    axis, e.g., [1,2,3] represents the series ``1*T_0 + 2*T_1 + 3*T_2``
    while [[1,2],[1,2]] represents ``1*T_0(x)*T_0(y) + 1*T_1(x)*T_0(y) +
    2*T_0(x)*T_1(y) + 2*T_1(x)*T_1(y)`` if axis=0 is ``x`` and axis=1 is
    ``y``.

    Parameters
    ----------
    c : array_like
        Array of Chebyshev series coefficients. If c is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
    m : int, optional
        Number of derivatives taken, must be non-negative. (Default: 1)
    scl : scalar, optional
        Each differentiation is multiplied by `scl`.  The end result is
        multiplication by ``scl**m``.  This is for use in a linear change of
        variable. (Default: 1)
    axis : int, optional
        Axis over which the derivative is taken. (Default: 0).

        .. versionadded:: 1.7.0

    Returns
    -------
    der : ndarray
        Chebyshev series of the derivative.

    See Also
    --------
    chebint

    Notes
    -----
    In general, the result of differentiating a C-series needs to be
    "reprojected" onto the C-series basis set. Thus, typically, the
    result of this function is "unintuitive," albeit correct; see Examples
    section below.

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c = (1,2,3,4)
    >>> C.chebder(c)
    array([14., 12., 24.])
    >>> C.chebder(c,3)
    array([96.])
    >>> C.chebder(c,scl=-1)
    array([-14., -12., -24.])
    >>> C.chebder(c,2,-1)
    array([12.,  96.])

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    cnt = pu._deprecate_as_int(m, "the order of derivation")
    iaxis = pu._deprecate_as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of derivation must be non-negative")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = np.moveaxis(c, iaxis, 0)
    n = len(c)
    if cnt >= n:
        c = c[:1]*0
    else:
        for i in range(cnt):
            n = n - 1
            c *= scl
            der = np.empty((n,) + c.shape[1:], dtype=c.dtype)
            for j in range(n, 2, -1):
                der[j - 1] = (2*j)*c[j]
                c[j - 2] += (j*c[j])/(j - 2)
            if n > 1:
                der[1] = 4*c[2]
            der[0] = c[1]
            c = der
    c = np.moveaxis(c, 0, iaxis)
    return c


def chebint(c, m=1, k=[], lbnd=0, scl=1, axis=0):
    """
    Integrate a Chebyshev series.

    Returns the Chebyshev series coefficients `c` integrated `m` times from
    `lbnd` along `axis`. At each iteration the resulting series is
    **multiplied** by `scl` and an integration constant, `k`, is added.
    The scaling factor is for use in a linear change of variable.  ("Buyer
    beware": note that, depending on what one is doing, one may want `scl`
    to be the reciprocal of what one might expect; for more information,
    see the Notes section below.)  The argument `c` is an array of
    coefficients from low to high degree along each axis, e.g., [1,2,3]
    represents the series ``T_0 + 2*T_1 + 3*T_2`` while [[1,2],[1,2]]
    represents ``1*T_0(x)*T_0(y) + 1*T_1(x)*T_0(y) + 2*T_0(x)*T_1(y) +
    2*T_1(x)*T_1(y)`` if axis=0 is ``x`` and axis=1 is ``y``.

    Parameters
    ----------
    c : array_like
        Array of Chebyshev series coefficients. If c is multidimensional
        the different axis correspond to different variables with the
        degree in each axis given by the corresponding index.
    m : int, optional
        Order of integration, must be positive. (Default: 1)
    k : {[], list, scalar}, optional
        Integration constant(s).  The value of the first integral at zero
        is the first value in the list, the value of the second integral
        at zero is the second value, etc.  If ``k == []`` (the default),
        all constants are set to zero.  If ``m == 1``, a single scalar can
        be given instead of a list.
    lbnd : scalar, optional
        The lower bound of the integral. (Default: 0)
    scl : scalar, optional
        Following each integration the result is *multiplied* by `scl`
        before the integration constant is added. (Default: 1)
    axis : int, optional
        Axis over which the integral is taken. (Default: 0).

        .. versionadded:: 1.7.0

    Returns
    -------
    S : ndarray
        C-series coefficients of the integral.

    Raises
    ------
    ValueError
        If ``m < 1``, ``len(k) > m``, ``np.ndim(lbnd) != 0``, or
        ``np.ndim(scl) != 0``.

    See Also
    --------
    chebder

    Notes
    -----
    Note that the result of each integration is *multiplied* by `scl`.
    Why is this important to note?  Say one is making a linear change of
    variable :math:`u = ax + b` in an integral relative to `x`.  Then
    :math:`dx = du/a`, so one will need to set `scl` equal to
    :math:`1/a`- perhaps not what one would have first thought.

    Also note that, in general, the result of integrating a C-series needs
    to be "reprojected" onto the C-series basis set.  Thus, typically,
    the result of this function is "unintuitive," albeit correct; see
    Examples section below.

    Examples
    --------
    >>> from numpy.polynomial import chebyshev as C
    >>> c = (1,2,3)
    >>> C.chebint(c)
    array([ 0.5, -0.5,  0.5,  0.5])
    >>> C.chebint(c,3)
    array([ 0.03125   , -0.1875    ,  0.04166667, -0.05208333,  0.01041667, # may vary
        0.00625   ])
    >>> C.chebint(c, k=3)
    array([ 3.5, -0.5,  0.5,  0.5])
    >>> C.chebint(c,lbnd=-2)
    array([ 8.5, -0.5,  0.5,  0.5])
    >>> C.chebint(c,scl=-2)
    array([-1.,  1., -1., -1.])

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if not np.iterable(k):
        k = [k]
    cnt = pu._deprecate_as_int(m, "the order of integration")
    iaxis = pu._deprecate_as_int(axis, "the axis")
    if cnt < 0:
        raise ValueError("The order of integration must be non-negative")
    if len(k) > cnt:
        raise ValueError("Too many integration constants")
    if np.ndim(lbnd) != 0:
        raise ValueError("lbnd must be a scalar.")
    if np.ndim(scl) != 0:
        raise ValueError("scl must be a scalar.")
    iaxis = normalize_axis_index(iaxis, c.ndim)

    if cnt == 0:
        return c

    c = np.moveaxis(c, iaxis, 0)
    k = list(k) + [0]*(cnt - len(k))
    for i in range(cnt):
        n = len(c)
        c *= scl
        if n == 1 and np.all(c[0] == 0):
            c[0] += k[i]
        else:
            tmp = np.empty((n + 1,) + c.shape[1:], dtype=c.dtype)
            tmp[0] = c[0]*0
            tmp[1] = c[0]
            if n > 1:
                tmp[2] = c[1]/4
            for j in range(2, n):
                tmp[j + 1] = c[j]/(2*(j + 1))
                tmp[j - 1] -= c[j]/(2*(j - 1))
            tmp[0] += k[i] - chebval(lbnd, tmp)
            c = tmp
    c = np.moveaxis(c, 0, iaxis)
    return c


def chebval(x, c, tensor=True):
    """
    Evaluate a Chebyshev series at points x.

    If `c` is of length `n + 1`, this function returns the value:

    .. math:: p(x) = c_0 * T_0(x) + c_1 * T_1(x) + ... + c_n * T_n(x)

    The parameter `x` is converted to an array only if it is a tuple or a
    list, otherwise it is treated as a scalar. In either case, either `x`
    or its elements must support multiplication and addition both with
    themselves and with the elements of `c`.

    If `c` is a 1-D array, then `p(x)` will have the same shape as `x`.  If
    `c` is multidimensional, then the shape of the result depends on the
    value of `tensor`. If `tensor` is true the shape will be c.shape[1:] +
    x.shape. If `tensor` is false the shape will be c.shape[1:]. Note that
    scalars have shape (,).

    Trailing zeros in the coefficients will be used in the evaluation, so
    they should be avoided if efficiency is a concern.

    Parameters
    ----------
    x : array_like, compatible object
        If `x` is a list or tuple, it is converted to an ndarray, otherwise
        it is left unchanged and treated as a scalar. In either case, `x`
        or its elements must support addition and multiplication with
        themselves and with the elements of `c`.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree n are contained in c[n]. If `c` is multidimensional the
        remaining indices enumerate multiple polynomials. In the two
        dimensional case the coefficients may be thought of as stored in
        the columns of `c`.
    tensor : boolean, optional
        If True, the shape of the coefficient array is extended with ones
        on the right, one for each dimension of `x`. Scalars have dimension 0
        for this action. The result is that every column of coefficients in
        `c` is evaluated for every element of `x`. If False, `x` is broadcast
        over the columns of `c` for the evaluation.  This keyword is useful
        when `c` is multidimensional. The default value is True.

        .. versionadded:: 1.7.0

    Returns
    -------
    values : ndarray, algebra_like
        The shape of the return value is described above.

    See Also
    --------
    chebval2d, chebgrid2d, chebval3d, chebgrid3d

    Notes
    -----
    The evaluation uses Clenshaw recursion, aka synthetic division.

    """
    c = np.array(c, ndmin=1, copy=True)
    if c.dtype.char in '?bBhHiIlLqQpP':
        c = c.astype(np.double)
    if isinstance(x, (tuple, list)):
        x = np.asarray(x)
    if isinstance(x, np.ndarray) and tensor:
        c = c.reshape(c.shape + (1,)*x.ndim)

    if len(c) == 1:
        c0 = c[0]
        c1 = 0
    elif len(c) == 2:
        c0 = c[0]
        c1 = c[1]
    else:
        x2 = 2*x
        c0 = c[-2]
        c1 = c[-1]
        for i in range(3, len(c) + 1):
            tmp = c0
            c0 = c[-i] - c1
            c1 = tmp + c1*x2
    return c0 + c1*x


def chebval2d(x, y, c):
    """
    Evaluate a 2-D Chebyshev series at points (x, y).

    This function returns the values:

    .. math:: p(x,y) = \\sum_{i,j} c_{i,j} * T_i(x) * T_j(y)

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars and they
    must have the same shape after conversion. In either case, either `x`
    and `y` or their elements must support multiplication and addition both
    with themselves and with the elements of `c`.

    If `c` is a 1-D array a one is implicitly appended to its shape to make
    it 2-D. The shape of the result will be c.shape[2:] + x.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points `(x, y)`,
        where `x` and `y` must have the same shape. If `x` or `y` is a list
        or tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and if it isn't an ndarray it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term
        of multi-degree i,j is contained in ``c[i,j]``. If `c` has
        dimension greater than 2 the remaining indices enumerate multiple
        sets of coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional Chebyshev series at points formed
        from pairs of corresponding values from `x` and `y`.

    See Also
    --------
    chebval, chebgrid2d, chebval3d, chebgrid3d

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    return pu._valnd(chebval, c, x, y)


def chebgrid2d(x, y, c):
    """
    Evaluate a 2-D Chebyshev series on the Cartesian product of x and y.

    This function returns the values:

    .. math:: p(a,b) = \\sum_{i,j} c_{i,j} * T_i(a) * T_j(b),

    where the points `(a, b)` consist of all pairs formed by taking
    `a` from `x` and `b` from `y`. The resulting points form a grid with
    `x` in the first dimension and `y` in the second.

    The parameters `x` and `y` are converted to arrays only if they are
    tuples or a lists, otherwise they are treated as a scalars. In either
    case, either `x` and `y` or their elements must support multiplication
    and addition both with themselves and with the elements of `c`.

    If `c` has fewer than two dimensions, ones are implicitly appended to
    its shape to make it 2-D. The shape of the result will be c.shape[2:] +
    x.shape + y.shape.

    Parameters
    ----------
    x, y : array_like, compatible objects
        The two dimensional series is evaluated at the points in the
        Cartesian product of `x` and `y`.  If `x` or `y` is a list or
        tuple, it is first converted to an ndarray, otherwise it is left
        unchanged and, if it isn't an ndarray, it is treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j is contained in `c[i,j]`. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional Chebyshev series at points in the
        Cartesian product of `x` and `y`.

    See Also
    --------
    chebval, chebval2d, chebval3d, chebgrid3d

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    return pu._gridnd(chebval, c, x, y)


def chebval3d(x, y, z, c):
    """
    Evaluate a 3-D Chebyshev series at points (x, y, z).

    This function returns the values:

    .. math:: p(x,y,z) = \\sum_{i,j,k} c_{i,j,k} * T_i(x) * T_j(y) * T_k(z)

    The parameters `x`, `y`, and `z` are converted to arrays only if
    they are tuples or a lists, otherwise they are treated as a scalars and
    they must have the same shape after conversion. In either case, either
    `x`, `y`, and `z` or their elements must support multiplication and
    addition both with themselves and with the elements of `c`.

    If `c` has fewer than 3 dimensions, ones are implicitly appended to its
    shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible object
        The three dimensional series is evaluated at the points
        `(x, y, z)`, where `x`, `y`, and `z` must have the same shape.  If
        any of `x`, `y`, or `z` is a list or tuple, it is first converted
        to an ndarray, otherwise it is left unchanged and if it isn't an
        ndarray it is  treated as a scalar.
    c : array_like
        Array of coefficients ordered so that the coefficient of the term of
        multi-degree i,j,k is contained in ``c[i,j,k]``. If `c` has dimension
        greater than 3 the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the multidimensional polynomial on points formed with
        triples of corresponding values from `x`, `y`, and `z`.

    See Also
    --------
    chebval, chebval2d, chebgrid2d, chebgrid3d

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    return pu._valnd(chebval, c, x, y, z)


def chebgrid3d(x, y, z, c):
    """
    Evaluate a 3-D Chebyshev series on the Cartesian product of x, y, and z.

    This function returns the values:

    .. math:: p(a,b,c) = \\sum_{i,j,k} c_{i,j,k} * T_i(a) * T_j(b) * T_k(c)

    where the points `(a, b, c)` consist of all triples formed by taking
    `a` from `x`, `b` from `y`, and `c` from `z`. The resulting points form
    a grid with `x` in the first dimension, `y` in the second, and `z` in
    the third.

    The parameters `x`, `y`, and `z` are converted to arrays only if they
    are tuples or a lists, otherwise they are treated as a scalars. In
    either case, either `x`, `y`, and `z` or their elements must support
    multiplication and addition both with themselves and with the elements
    of `c`.

    If `c` has fewer than three dimensions, ones are implicitly appended to
    its shape to make it 3-D. The shape of the result will be c.shape[3:] +
    x.shape + y.shape + z.shape.

    Parameters
    ----------
    x, y, z : array_like, compatible objects
        The three dimensional series is evaluated at the points in the
        Cartesian product of `x`, `y`, and `z`.  If `x`,`y`, or `z` is a
        list or tuple, it is first converted to an ndarray, otherwise it is
        left unchanged and, if it isn't an ndarray, it is treated as a
        scalar.
    c : array_like
        Array of coefficients ordered so that the coefficients for terms of
        degree i,j are contained in ``c[i,j]``. If `c` has dimension
        greater than two the remaining indices enumerate multiple sets of
        coefficients.

    Returns
    -------
    values : ndarray, compatible object
        The values of the two dimensional polynomial at points in the Cartesian
        product of `x` and `y`.

    See Also
    --------
    chebval, chebval2d, chebgrid2d, chebval3d

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    return pu._gridnd(chebval, c, x, y, z)


def chebvander(x, deg):
    """Pseudo-Vandermonde matrix of given degree.

    Returns the pseudo-Vandermonde matrix of degree `deg` and sample points
    `x`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., i] = T_i(x),

    where `0 <= i <= deg`. The leading indices of `V` index the elements of
    `x` and the last index is the degree of the Chebyshev polynomial.

    If `c` is a 1-D array of coefficients of length `n + 1` and `V` is the
    matrix ``V = chebvander(x, n)``, then ``np.dot(V, c)`` and
    ``chebval(x, c)`` are the same up to roundoff.  This equivalence is
    useful both for least squares fitting and for the evaluation of a large
    number of Chebyshev series of the same degree and sample points.

    Parameters
    ----------
    x : array_like
        Array of points. The dtype is converted to float64 or complex128
        depending on whether any of the elements are complex. If `x` is
        scalar it is converted to a 1-D array.
    deg : int
        Degree of the resulting matrix.

    Returns
    -------
    vander : ndarray
        The pseudo Vandermonde matrix. The shape of the returned matrix is
        ``x.shape + (deg + 1,)``, where The last index is the degree of the
        corresponding Chebyshev polynomial.  The dtype will be the same as
        the converted `x`.

    """
    ideg = pu._deprecate_as_int(deg, "deg")
    if ideg < 0:
        raise ValueError("deg must be non-negative")

    x = np.array(x, copy=False, ndmin=1) + 0.0
    dims = (ideg + 1,) + x.shape
    dtyp = x.dtype
    v = np.empty(dims, dtype=dtyp)
    # Use forward recursion to generate the entries.
    v[0] = x*0 + 1
    if ideg > 0:
        x2 = 2*x
        v[1] = x
        for i in range(2, ideg + 1):
            v[i] = v[i-1]*x2 - v[i-2]
    return np.moveaxis(v, 0, -1)


def chebvander2d(x, y, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points `(x, y)`. The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (deg[1] + 1)*i + j] = T_i(x) * T_j(y),

    where `0 <= i <= deg[0]` and `0 <= j <= deg[1]`. The leading indices of
    `V` index the points `(x, y)` and the last index encodes the degrees of
    the Chebyshev polynomials.

    If ``V = chebvander2d(x, y, [xdeg, ydeg])``, then the columns of `V`
    correspond to the elements of a 2-D coefficient array `c` of shape
    (xdeg + 1, ydeg + 1) in the order

    .. math:: c_{00}, c_{01}, c_{02} ... , c_{10}, c_{11}, c_{12} ...

    and ``np.dot(V, c.flat)`` and ``chebval2d(x, y, c)`` will be the same
    up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 2-D Chebyshev
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y : array_like
        Arrays of point coordinates, all of the same shape. The dtypes
        will be converted to either float64 or complex128 depending on
        whether any of the elements are complex. Scalars are converted to
        1-D arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg].

    Returns
    -------
    vander2d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg[1]+1)`.  The dtype will be the same
        as the converted `x` and `y`.

    See Also
    --------
    chebvander, chebvander3d, chebval2d, chebval3d

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    return pu._vander_nd_flat((chebvander, chebvander), (x, y), deg)


def chebvander3d(x, y, z, deg):
    """Pseudo-Vandermonde matrix of given degrees.

    Returns the pseudo-Vandermonde matrix of degrees `deg` and sample
    points `(x, y, z)`. If `l, m, n` are the given degrees in `x, y, z`,
    then The pseudo-Vandermonde matrix is defined by

    .. math:: V[..., (m+1)(n+1)i + (n+1)j + k] = T_i(x)*T_j(y)*T_k(z),

    where `0 <= i <= l`, `0 <= j <= m`, and `0 <= j <= n`.  The leading
    indices of `V` index the points `(x, y, z)` and the last index encodes
    the degrees of the Chebyshev polynomials.

    If ``V = chebvander3d(x, y, z, [xdeg, ydeg, zdeg])``, then the columns
    of `V` correspond to the elements of a 3-D coefficient array `c` of
    shape (xdeg + 1, ydeg + 1, zdeg + 1) in the order

    .. math:: c_{000}, c_{001}, c_{002},... , c_{010}, c_{011}, c_{012},...

    and ``np.dot(V, c.flat)`` and ``chebval3d(x, y, z, c)`` will be the
    same up to roundoff. This equivalence is useful both for least squares
    fitting and for the evaluation of a large number of 3-D Chebyshev
    series of the same degrees and sample points.

    Parameters
    ----------
    x, y, z : array_like
        Arrays of point coordinates, all of the same shape. The dtypes will
        be converted to either float64 or complex128 depending on whether
        any of the elements are complex. Scalars are converted to 1-D
        arrays.
    deg : list of ints
        List of maximum degrees of the form [x_deg, y_deg, z_deg].

    Returns
    -------
    vander3d : ndarray
        The shape of the returned matrix is ``x.shape + (order,)``, where
        :math:`order = (deg[0]+1)*(deg[1]+1)*(deg[2]+1)`.  The dtype will
        be the same as the converted `x`, `y`, and `z`.

    See Also
    --------
    chebvander, chebvander3d, chebval2d, chebval3d

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    return pu._vander_nd_flat((chebvander, chebvander, chebvander), (x, y, z), deg)


def chebfit(x, y, deg, rcond=None, full=False, w=None):
    """
    Least squares fit of Chebyshev series to data.

    Return the coefficients of a Chebyshev series of degree `deg` that is the
    least squares fit to the data values `y` given at points `x`. If `y` is
    1-D the returned coefficients will also be 1-D. If `y` is 2-D multiple
    fits are done, one for each column of `y`, and the resulting
    coefficients are stored in the corresponding columns of a 2-D return.
    The fitted polynomial(s) are in the form

    .. math::  p(x) = c_0 + c_1 * T_1(x) + ... + c_n * T_n(x),

    where `n` is `deg`.

    Parameters
    ----------
    x : array_like, shape (M,)
        x-coordinates of the M sample points ``(x[i], y[i])``.
    y : array_like, shape (M,) or (M, K)
        y-coordinates of the sample points. Several data sets of sample
        points sharing the same x-coordinates can be fitted at once by
        passing in a 2D-array that contains one dataset per column.
    deg : int or 1-D array_like
        Degree(s) of the fitting polynomials. If `deg` is a single integer,
        all terms up to and including the `deg`'th term are included in the
        fit. For NumPy versions >= 1.11.0 a list of integers specifying the
        degrees of the terms to include may be used instead.
    rcond : float, optional
        Relative condition number of the fit. Singular values smaller than
        this relative to the largest singular value will be ignored. The
        default value is len(x)*eps, where eps is the relative precision of
        the float type, about 2e-16 in most cases.
    full : bool, optional
        Switch determining nature of return value. When it is False (the
        default) just the coefficients are returned, when True diagnostic
        information from the singular value decomposition is also returned.
    w : array_like, shape (`M`,), optional
        Weights. If not None, the weight ``w[i]`` applies to the unsquared
        residual ``y[i] - y_hat[i]`` at ``x[i]``. Ideally the weights are
        chosen so that the errors of the products ``w[i]*y[i]`` all have the
        same variance.  When using inverse-variance weighting, use
        ``w[i] = 1/sigma(y[i])``.  The default value is None.

        .. versionadded:: 1.5.0

    Returns
    -------
    coef : ndarray, shape (M,) or (M, K)
        Chebyshev coefficients ordered from low to high. If `y` was 2-D,
        the coefficients for the data in column k  of `y` are in column
        `k`.

    [residuals, rank, singular_values, rcond] : list
        These values are only returned if ``full == True``

        - residuals -- sum of squared residuals of the least squares fit
        - rank -- the numerical rank of the scaled Vandermonde matrix
        - singular_values -- singular values of the scaled Vandermonde matrix
        - rcond -- value of `rcond`.

        For more details, see `numpy.linalg.lstsq`.

    Warns
    -----
    RankWarning
        The rank of the coefficient matrix in the least-squares fit is
        deficient. The warning is only raised if ``full == False``.  The
        warnings can be turned off by

        >>> import warnings
        >>> warnings.simplefilter('ignore', np.RankWarning)

    See Also
    --------
    numpy.polynomial.polynomial.polyfit
    numpy.polynomial.legendre.legfit
    numpy.polynomial.laguerre.lagfit
    numpy.polynomial.hermite.hermfit
    numpy.polynomial.hermite_e.hermefit
    chebval : Evaluates a Chebyshev series.
    chebvander : Vandermonde matrix of Chebyshev series.
    chebweight : Chebyshev weight function.
    numpy.linalg.lstsq : Computes a least-squares fit from the matrix.
    scipy.interpolate.UnivariateSpline : Computes spline fits.

    Notes
    -----
    The solution is the coefficients of the Chebyshev series `p` that
    minimizes the sum of the weighted squared errors

    .. math:: E = \\sum_j w_j^2 * |y_j - p(x_j)|^2,

    where :math:`w_j` are the weights. This problem is solved by setting up
    as the (typically) overdetermined matrix equation

    .. math:: V(x) * c = w * y,

    where `V` is the weighted pseudo Vandermonde matrix of `x`, `c` are the
    coefficients to be solved for, `w` are the weights, and `y` are the
    observed values.  This equation is then solved using the singular value
    decomposition of `V`.

    If some of the singular values of `V` are so small that they are
    neglected, then a `RankWarning` will be issued. This means that the
    coefficient values may be poorly determined. Using a lower order fit
    will usually get rid of the warning.  The `rcond` parameter can also be
    set to a value smaller than its default, but the resulting fit may be
    spurious and have large contributions from roundoff error.

    Fits using Chebyshev series are usually better conditioned than fits
    using power series, but much can depend on the distribution of the
    sample points and the smoothness of the data. If the quality of the fit
    is inadequate splines may be a good alternative.

    References
    ----------
    .. [1] Wikipedia, "Curve fitting",
           https://en.wikipedia.org/wiki/Curve_fitting

    Examples
    --------

    """
    return pu._fit(chebvander, x, y, deg, rcond, full, w)


def chebcompanion(c):
    """Return the scaled companion matrix of c.

    The basis polynomials are scaled so that the companion matrix is
    symmetric when `c` is a Chebyshev basis polynomial. This provides
    better eigenvalue estimates than the unscaled case and for basis
    polynomials the eigenvalues are guaranteed to be real if
    `numpy.linalg.eigvalsh` is used to obtain them.

    Parameters
    ----------
    c : array_like
        1-D array of Chebyshev series coefficients ordered from low to high
        degree.

    Returns
    -------
    mat : ndarray
        Scaled companion matrix of dimensions (deg, deg).

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        raise ValueError('Series must have maximum degree of at least 1.')
    if len(c) == 2:
        return np.array([[-c[0]/c[1]]])

    n = len(c) - 1
    mat = np.zeros((n, n), dtype=c.dtype)
    scl = np.array([1.] + [np.sqrt(.5)]*(n-1))
    top = mat.reshape(-1)[1::n+1]
    bot = mat.reshape(-1)[n::n+1]
    top[0] = np.sqrt(.5)
    top[1:] = 1/2
    bot[...] = top
    mat[:, -1] -= (c[:-1]/c[-1])*(scl/scl[-1])*.5
    return mat


def chebroots(c):
    """
    Compute the roots of a Chebyshev series.

    Return the roots (a.k.a. "zeros") of the polynomial

    .. math:: p(x) = \\sum_i c[i] * T_i(x).

    Parameters
    ----------
    c : 1-D array_like
        1-D array of coefficients.

    Returns
    -------
    out : ndarray
        Array of the roots of the series. If all the roots are real,
        then `out` is also real, otherwise it is complex.

    See Also
    --------
    numpy.polynomial.polynomial.polyroots
    numpy.polynomial.legendre.legroots
    numpy.polynomial.laguerre.lagroots
    numpy.polynomial.hermite.hermroots
    numpy.polynomial.hermite_e.hermeroots

    Notes
    -----
    The root estimates are obtained as the eigenvalues of the companion
    matrix, Roots far from the origin of the complex plane may have large
    errors due to the numerical instability of the series for such
    values. Roots with multiplicity greater than 1 will also show larger
    errors as the value of the series near such points is relatively
    insensitive to errors in the roots. Isolated roots near the origin can
    be improved by a few iterations of Newton's method.

    The Chebyshev series basis polynomials aren't powers of `x` so the
    results of this function may seem unintuitive.

    Examples
    --------
    >>> import numpy.polynomial.chebyshev as cheb
    >>> cheb.chebroots((-1, 1,-1, 1)) # T3 - T2 + T1 - T0 has real roots
    array([ -5.00000000e-01,   2.60860684e-17,   1.00000000e+00]) # may vary

    """
    # c is a trimmed copy
    [c] = pu.as_series([c])
    if len(c) < 2:
        return np.array([], dtype=c.dtype)
    if len(c) == 2:
        return np.array([-c[0]/c[1]])

    # rotated companion matrix reduces error
    m = chebcompanion(c)[::-1,::-1]
    r = la.eigvals(m)
    r.sort()
    return r


def chebinterpolate(func, deg, args=()):
    """Interpolate a function at the Chebyshev points of the first kind.

    Returns the Chebyshev series that interpolates `func` at the Chebyshev
    points of the first kind in the interval [-1, 1]. The interpolating
    series tends to a minmax approximation to `func` with increasing `deg`
    if the function is continuous in the interval.

    .. versionadded:: 1.14.0

    Parameters
    ----------
    func : function
        The function to be approximated. It must be a function of a single
        variable of the form ``f(x, a, b, c...)``, where ``a, b, c...`` are
        extra arguments passed in the `args` parameter.
    deg : int
        Degree of the interpolating polynomial
    args : tuple, optional
        Extra arguments to be used in the function call. Default is no extra
        arguments.

    Returns
    -------
    coef : ndarray, shape (deg + 1,)
        Chebyshev coefficients of the interpolating series ordered from low to
        high.

    Examples
    --------
    >>> import numpy.polynomial.chebyshev as C
    >>> C.chebfromfunction(lambda x: np.tanh(x) + 0.5, 8)
    array([  5.00000000e-01,   8.11675684e-01,  -9.86864911e-17,
            -5.42457905e-02,  -2.71387850e-16,   4.51658839e-03,
             2.46716228e-17,  -3.79694221e-04,  -3.26899002e-16])

    Notes
    -----

    The Chebyshev polynomials used in the interpolation are orthogonal when
    sampled at the Chebyshev points of the first kind. If it is desired to
    constrain some of the coefficients they can simply be set to the desired
    value after the interpolation, no new interpolation or fit is needed. This
    is especially useful if it is known apriori that some of coefficients are
    zero. For instance, if the function is even then the coefficients of the
    terms of odd degree in the result can be set to zero.

    """
    deg = np.asarray(deg)

    # check arguments.
    if deg.ndim > 0 or deg.dtype.kind not in 'iu' or deg.size == 0:
        raise TypeError("deg must be an int")
    if deg < 0:
        raise ValueError("expected deg >= 0")

    order = deg + 1
    xcheb = chebpts1(order)
    yfunc = func(xcheb, *args)
    m = chebvander(xcheb, deg)
    c = np.dot(m.T, yfunc)
    c[0] /= order
    c[1:] /= 0.5*order

    return c


def chebgauss(deg):
    """
    Gauss-Chebyshev quadrature.

    Computes the sample points and weights for Gauss-Chebyshev quadrature.
    These sample points and weights will correctly integrate polynomials of
    degree :math:`2*deg - 1` or less over the interval :math:`[-1, 1]` with
    the weight function :math:`f(x) = 1/\\sqrt{1 - x^2}`.

    Parameters
    ----------
    deg : int
        Number of sample points and weights. It must be >= 1.

    Returns
    -------
    x : ndarray
        1-D ndarray containing the sample points.
    y : ndarray
        1-D ndarray containing the weights.

    Notes
    -----

    .. versionadded:: 1.7.0

    The results have only been tested up to degree 100, higher degrees may
    be problematic. For Gauss-Chebyshev there are closed form solutions for
    the sample points and weights. If n = `deg`, then

    .. math:: x_i = \\cos(\\pi (2 i - 1) / (2 n))

    .. math:: w_i = \\pi / n

    """
    ideg = pu._deprecate_as_int(deg, "deg")
    if ideg <= 0:
        raise ValueError("deg must be a positive integer")

    x = np.cos(np.pi * np.arange(1, 2*ideg, 2) / (2.0*ideg))
    w = np.ones(ideg)*(np.pi/ideg)

    return x, w


def chebweight(x):
    """
    The weight function of the Chebyshev polynomials.

    The weight function is :math:`1/\\sqrt{1 - x^2}` and the interval of
    integration is :math:`[-1, 1]`. The Chebyshev polynomials are
    orthogonal, but not normalized, with respect to this weight function.

    Parameters
    ----------
    x : array_like
       Values at which the weight function will be computed.

    Returns
    -------
    w : ndarray
       The weight function at `x`.

    Notes
    -----

    .. versionadded:: 1.7.0

    """
    w = 1./(np.sqrt(1. + x) * np.sqrt(1. - x))
    return w


def chebpts1(npts):
    """
    Chebyshev points of the first kind.

    The Chebyshev points of the first kind are the points ``cos(x)``,
    where ``x = [pi*(k + .5)/npts for k in range(npts)]``.

    Parameters
    ----------
    npts : int
        Number of sample points desired.

    Returns
    -------
    pts : ndarray
        The Chebyshev points of the first kind.

    See Also
    --------
    chebpts2

    Notes
    -----

    .. versionadded:: 1.5.0

    """
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 1:
        raise ValueError("npts must be >= 1")

    x = 0.5 * np.pi / _npts * np.arange(-_npts+1, _npts+1, 2)
    return np.sin(x)


def chebpts2(npts):
    """
    Chebyshev points of the second kind.

    The Chebyshev points of the second kind are the points ``cos(x)``,
    where ``x = [pi*k/(npts - 1) for k in range(npts)]`` sorted in ascending
    order.

    Parameters
    ----------
    npts : int
        Number of sample points desired.

    Returns
    -------
    pts : ndarray
        The Chebyshev points of the second kind.

    Notes
    -----

    .. versionadded:: 1.5.0

    """
    _npts = int(npts)
    if _npts != npts:
        raise ValueError("npts must be integer")
    if _npts < 2:
        raise ValueError("npts must be >= 2")

    x = np.linspace(-np.pi, 0, _npts)
    return np.cos(x)


#
# Chebyshev series class
#

class Chebyshev(ABCPolyBase):
    """A Chebyshev series class.

    The Chebyshev class provides the standard Python numerical methods
    '+', '-', '*', '//', '%', 'divmod', '**', and '()' as well as the
    methods listed below.

    Parameters
    ----------
    coef : array_like
        Chebyshev coefficients in order of increasing degree, i.e.,
        ``(1, 2, 3)`` gives ``1*T_0(x) + 2*T_1(x) + 3*T_2(x)``.
    domain : (2,) array_like, optional
        Domain to use. The interval ``[domain[0], domain[1]]`` is mapped
        to the interval ``[window[0], window[1]]`` by shifting and scaling.
        The default value is [-1, 1].
    window : (2,) array_like, optional
        Window, see `domain` for its use. The default value is [-1, 1].

        .. versionadded:: 1.6.0

    """
    # Virtual Functions
    _add = staticmethod(chebadd)
    _sub = staticmethod(chebsub)
    _mul = staticmethod(chebmul)
    _div = staticmethod(chebdiv)
    _pow = staticmethod(chebpow)
    _val = staticmethod(chebval)
    _int = staticmethod(chebint)
    _der = staticmethod(chebder)
    _fit = staticmethod(chebfit)
    _line = staticmethod(chebline)
    _roots = staticmethod(chebroots)
    _fromroots = staticmethod(chebfromroots)

    @classmethod
    def interpolate(cls, func, deg, domain=None, args=()):
        """Interpolate a function at the Chebyshev points of the first kind.

        Returns the series that interpolates `func` at the Chebyshev points of
        the first kind scaled and shifted to the `domain`. The resulting series
        tends to a minmax approximation of `func` when the function is
        continuous in the domain.

        .. versionadded:: 1.14.0

        Parameters
        ----------
        func : function
            The function to be interpolated. It must be a function of a single
            variable of the form ``f(x, a, b, c...)``, where ``a, b, c...`` are
            extra arguments passed in the `args` parameter.
        deg : int
            Degree of the interpolating polynomial.
        domain : {None, [beg, end]}, optional
            Domain over which `func` is interpolated. The default is None, in
            which case the domain is [-1, 1].
        args : tuple, optional
            Extra arguments to be used in the function call. Default is no
            extra arguments.

        Returns
        -------
        polynomial : Chebyshev instance
            Interpolating Chebyshev instance.

        Notes
        -----
        See `numpy.polynomial.chebfromfunction` for more details.

        """
        if domain is None:
            domain = cls.domain
        xfunc = lambda x: func(pu.mapdomain(x, cls.window, domain), *args)
        coef = chebinterpolate(xfunc, deg)
        return cls(coef, domain=domain)

    # Virtual properties
    domain = np.array(chebdomain)
    window = np.array(chebdomain)
    basis_name = 'T'
