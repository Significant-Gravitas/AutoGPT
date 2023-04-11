"""
This module contains a set of functions for vectorized string
operations and methods.

.. note::
   The `chararray` class exists for backwards compatibility with
   Numarray, it is not recommended for new development. Starting from numpy
   1.4, if one needs arrays of strings, it is recommended to use arrays of
   `dtype` `object_`, `string_` or `unicode_`, and use the free functions
   in the `numpy.char` module for fast vectorized string operations.

Some methods will only be available if the corresponding string method is
available in your version of Python.

The preferred alias for `defchararray` is `numpy.char`.

"""
import functools
from .numerictypes import (
    string_, unicode_, integer, int_, object_, bool_, character)
from .numeric import ndarray, compare_chararrays
from .numeric import array as narray
from numpy.core.multiarray import _vec_string
from numpy.core.overrides import set_module
from numpy.core import overrides
from numpy.compat import asbytes
import numpy

__all__ = [
    'equal', 'not_equal', 'greater_equal', 'less_equal',
    'greater', 'less', 'str_len', 'add', 'multiply', 'mod', 'capitalize',
    'center', 'count', 'decode', 'encode', 'endswith', 'expandtabs',
    'find', 'index', 'isalnum', 'isalpha', 'isdigit', 'islower', 'isspace',
    'istitle', 'isupper', 'join', 'ljust', 'lower', 'lstrip', 'partition',
    'replace', 'rfind', 'rindex', 'rjust', 'rpartition', 'rsplit',
    'rstrip', 'split', 'splitlines', 'startswith', 'strip', 'swapcase',
    'title', 'translate', 'upper', 'zfill', 'isnumeric', 'isdecimal',
    'array', 'asarray'
    ]


_globalvar = 0

array_function_dispatch = functools.partial(
    overrides.array_function_dispatch, module='numpy.char')


def _use_unicode(*args):
    """
    Helper function for determining the output type of some string
    operations.

    For an operation on two ndarrays, if at least one is unicode, the
    result should be unicode.
    """
    for x in args:
        if (isinstance(x, str) or
                issubclass(numpy.asarray(x).dtype.type, unicode_)):
            return unicode_
    return string_

def _to_string_or_unicode_array(result):
    """
    Helper function to cast a result back into a string or unicode array
    if an object array must be used as an intermediary.
    """
    return numpy.asarray(result.tolist())

def _clean_args(*args):
    """
    Helper function for delegating arguments to Python string
    functions.

    Many of the Python string operations that have optional arguments
    do not use 'None' to indicate a default value.  In these cases,
    we need to remove all None arguments, and those following them.
    """
    newargs = []
    for chk in args:
        if chk is None:
            break
        newargs.append(chk)
    return newargs

def _get_num_chars(a):
    """
    Helper function that returns the number of characters per field in
    a string or unicode array.  This is to abstract out the fact that
    for a unicode array this is itemsize / 4.
    """
    if issubclass(a.dtype.type, unicode_):
        return a.itemsize // 4
    return a.itemsize


def _binary_op_dispatcher(x1, x2):
    return (x1, x2)


@array_function_dispatch(_binary_op_dispatcher)
def equal(x1, x2):
    """
    Return (x1 == x2) element-wise.

    Unlike `numpy.equal`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    not_equal, greater_equal, less_equal, greater, less
    """
    return compare_chararrays(x1, x2, '==', True)


@array_function_dispatch(_binary_op_dispatcher)
def not_equal(x1, x2):
    """
    Return (x1 != x2) element-wise.

    Unlike `numpy.not_equal`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, greater_equal, less_equal, greater, less
    """
    return compare_chararrays(x1, x2, '!=', True)


@array_function_dispatch(_binary_op_dispatcher)
def greater_equal(x1, x2):
    """
    Return (x1 >= x2) element-wise.

    Unlike `numpy.greater_equal`, this comparison is performed by
    first stripping whitespace characters from the end of the string.
    This behavior is provided for backward-compatibility with
    numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, not_equal, less_equal, greater, less
    """
    return compare_chararrays(x1, x2, '>=', True)


@array_function_dispatch(_binary_op_dispatcher)
def less_equal(x1, x2):
    """
    Return (x1 <= x2) element-wise.

    Unlike `numpy.less_equal`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, not_equal, greater_equal, greater, less
    """
    return compare_chararrays(x1, x2, '<=', True)


@array_function_dispatch(_binary_op_dispatcher)
def greater(x1, x2):
    """
    Return (x1 > x2) element-wise.

    Unlike `numpy.greater`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, not_equal, greater_equal, less_equal, less
    """
    return compare_chararrays(x1, x2, '>', True)


@array_function_dispatch(_binary_op_dispatcher)
def less(x1, x2):
    """
    Return (x1 < x2) element-wise.

    Unlike `numpy.greater`, this comparison is performed by first
    stripping whitespace characters from the end of the string.  This
    behavior is provided for backward-compatibility with numarray.

    Parameters
    ----------
    x1, x2 : array_like of str or unicode
        Input arrays of the same shape.

    Returns
    -------
    out : ndarray
        Output array of bools.

    See Also
    --------
    equal, not_equal, greater_equal, less_equal, greater
    """
    return compare_chararrays(x1, x2, '<', True)


def _unary_op_dispatcher(a):
    return (a,)


@array_function_dispatch(_unary_op_dispatcher)
def str_len(a):
    """
    Return len(a) element-wise.

    Parameters
    ----------
    a : array_like of str or unicode

    Returns
    -------
    out : ndarray
        Output array of integers

    See Also
    --------
    builtins.len

    Examples
    --------
    >>> a = np.array(['Grace Hopper Conference', 'Open Source Day'])
    >>> np.char.str_len(a)
    array([23, 15])
    >>> a = np.array([u'\u0420', u'\u043e'])
    >>> np.char.str_len(a)
    array([1, 1])
    >>> a = np.array([['hello', 'world'], [u'\u0420', u'\u043e']])
    >>> np.char.str_len(a)
    array([[5, 5], [1, 1]])
    """
    # Note: __len__, etc. currently return ints, which are not C-integers.
    # Generally intp would be expected for lengths, although int is sufficient
    # due to the dtype itemsize limitation.
    return _vec_string(a, int_, '__len__')


@array_function_dispatch(_binary_op_dispatcher)
def add(x1, x2):
    """
    Return element-wise string concatenation for two arrays of str or unicode.

    Arrays `x1` and `x2` must have the same shape.

    Parameters
    ----------
    x1 : array_like of str or unicode
        Input array.
    x2 : array_like of str or unicode
        Input array.

    Returns
    -------
    add : ndarray
        Output array of `string_` or `unicode_`, depending on input types
        of the same shape as `x1` and `x2`.

    """
    arr1 = numpy.asarray(x1)
    arr2 = numpy.asarray(x2)
    out_size = _get_num_chars(arr1) + _get_num_chars(arr2)
    dtype = _use_unicode(arr1, arr2)
    return _vec_string(arr1, (dtype, out_size), '__add__', (arr2,))


def _multiply_dispatcher(a, i):
    return (a,)


@array_function_dispatch(_multiply_dispatcher)
def multiply(a, i):
    """
    Return (a * i), that is string multiple concatenation,
    element-wise.

    Values in `i` of less than 0 are treated as 0 (which yields an
    empty string).

    Parameters
    ----------
    a : array_like of str or unicode

    i : array_like of ints

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input types
    
    Examples
    --------
    >>> a = np.array(["a", "b", "c"])
    >>> np.char.multiply(x, 3)
    array(['aaa', 'bbb', 'ccc'], dtype='<U3')
    >>> i = np.array([1, 2, 3])
    >>> np.char.multiply(a, i)
    array(['a', 'bb', 'ccc'], dtype='<U3')
    >>> np.char.multiply(np.array(['a']), i)
    array(['a', 'aa', 'aaa'], dtype='<U3')
    >>> a = np.array(['a', 'b', 'c', 'd', 'e', 'f']).reshape((2, 3))
    >>> np.char.multiply(a, 3)
    array([['aaa', 'bbb', 'ccc'],
           ['ddd', 'eee', 'fff']], dtype='<U3')
    >>> np.char.multiply(a, i)
    array([['a', 'bb', 'ccc'],
           ['d', 'ee', 'fff']], dtype='<U3')
    """
    a_arr = numpy.asarray(a)
    i_arr = numpy.asarray(i)
    if not issubclass(i_arr.dtype.type, integer):
        raise ValueError("Can only multiply by integers")
    out_size = _get_num_chars(a_arr) * max(int(i_arr.max()), 0)
    return _vec_string(
        a_arr, (a_arr.dtype.type, out_size), '__mul__', (i_arr,))


def _mod_dispatcher(a, values):
    return (a, values)


@array_function_dispatch(_mod_dispatcher)
def mod(a, values):
    """
    Return (a % i), that is pre-Python 2.6 string formatting
    (interpolation), element-wise for a pair of array_likes of str
    or unicode.

    Parameters
    ----------
    a : array_like of str or unicode

    values : array_like of values
       These values will be element-wise interpolated into the string.

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input types

    See Also
    --------
    str.__mod__

    """
    return _to_string_or_unicode_array(
        _vec_string(a, object_, '__mod__', (values,)))


@array_function_dispatch(_unary_op_dispatcher)
def capitalize(a):
    """
    Return a copy of `a` with only the first character of each element
    capitalized.

    Calls `str.capitalize` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array_like of str or unicode
        Input array of strings to capitalize.

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input
        types

    See Also
    --------
    str.capitalize

    Examples
    --------
    >>> c = np.array(['a1b2','1b2a','b2a1','2a1b'],'S4'); c
    array(['a1b2', '1b2a', 'b2a1', '2a1b'],
        dtype='|S4')
    >>> np.char.capitalize(c)
    array(['A1b2', '1b2a', 'B2a1', '2a1b'],
        dtype='|S4')

    """
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'capitalize')


def _center_dispatcher(a, width, fillchar=None):
    return (a,)


@array_function_dispatch(_center_dispatcher)
def center(a, width, fillchar=' '):
    """
    Return a copy of `a` with its elements centered in a string of
    length `width`.

    Calls `str.center` element-wise.

    Parameters
    ----------
    a : array_like of str or unicode

    width : int
        The length of the resulting strings
    fillchar : str or unicode, optional
        The padding character to use (default is space).

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input
        types

    See Also
    --------
    str.center
    
    Notes
    -----
    This function is intended to work with arrays of strings.  The
    fill character is not applied to numeric types.

    Examples
    --------
    >>> c = np.array(['a1b2','1b2a','b2a1','2a1b']); c
    array(['a1b2', '1b2a', 'b2a1', '2a1b'], dtype='<U4')
    >>> np.char.center(c, width=9)
    array(['   a1b2  ', '   1b2a  ', '   b2a1  ', '   2a1b  '], dtype='<U9')
    >>> np.char.center(c, width=9, fillchar='*')
    array(['***a1b2**', '***1b2a**', '***b2a1**', '***2a1b**'], dtype='<U9')
    >>> np.char.center(c, width=1)
    array(['a', '1', 'b', '2'], dtype='<U1')

    """
    a_arr = numpy.asarray(a)
    width_arr = numpy.asarray(width)
    size = int(numpy.max(width_arr.flat))
    if numpy.issubdtype(a_arr.dtype, numpy.string_):
        fillchar = asbytes(fillchar)
    return _vec_string(
        a_arr, (a_arr.dtype.type, size), 'center', (width_arr, fillchar))


def _count_dispatcher(a, sub, start=None, end=None):
    return (a,)


@array_function_dispatch(_count_dispatcher)
def count(a, sub, start=0, end=None):
    """
    Returns an array with the number of non-overlapping occurrences of
    substring `sub` in the range [`start`, `end`].

    Calls `str.count` element-wise.

    Parameters
    ----------
    a : array_like of str or unicode

    sub : str or unicode
       The substring to search for.

    start, end : int, optional
       Optional arguments `start` and `end` are interpreted as slice
       notation to specify the range in which to count.

    Returns
    -------
    out : ndarray
        Output array of ints.

    See Also
    --------
    str.count

    Examples
    --------
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.char.count(c, 'A')
    array([3, 1, 1])
    >>> np.char.count(c, 'aA')
    array([3, 1, 0])
    >>> np.char.count(c, 'A', start=1, end=4)
    array([2, 1, 1])
    >>> np.char.count(c, 'A', start=1, end=3)
    array([1, 0, 0])

    """
    return _vec_string(a, int_, 'count', [sub, start] + _clean_args(end))


def _code_dispatcher(a, encoding=None, errors=None):
    return (a,)


@array_function_dispatch(_code_dispatcher)
def decode(a, encoding=None, errors=None):
    r"""
    Calls ``bytes.decode`` element-wise.

    The set of available codecs comes from the Python standard library,
    and may be extended at runtime.  For more information, see the
    :mod:`codecs` module.

    Parameters
    ----------
    a : array_like of str or unicode

    encoding : str, optional
       The name of an encoding

    errors : str, optional
       Specifies how to handle encoding errors

    Returns
    -------
    out : ndarray

    See Also
    --------
    :py:meth:`bytes.decode`

    Notes
    -----
    The type of the result will depend on the encoding specified.

    Examples
    --------
    >>> c = np.array([b'\x81\xc1\x81\xc1\x81\xc1', b'@@\x81\xc1@@',
    ...               b'\x81\x82\xc2\xc1\xc2\x82\x81'])
    >>> c
    array([b'\x81\xc1\x81\xc1\x81\xc1', b'@@\x81\xc1@@',
    ...    b'\x81\x82\xc2\xc1\xc2\x82\x81'], dtype='|S7')
    >>> np.char.decode(c, encoding='cp037')
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')

    """
    return _to_string_or_unicode_array(
        _vec_string(a, object_, 'decode', _clean_args(encoding, errors)))


@array_function_dispatch(_code_dispatcher)
def encode(a, encoding=None, errors=None):
    """
    Calls `str.encode` element-wise.

    The set of available codecs comes from the Python standard library,
    and may be extended at runtime. For more information, see the codecs
    module.

    Parameters
    ----------
    a : array_like of str or unicode

    encoding : str, optional
       The name of an encoding

    errors : str, optional
       Specifies how to handle encoding errors

    Returns
    -------
    out : ndarray

    See Also
    --------
    str.encode

    Notes
    -----
    The type of the result will depend on the encoding specified.

    """
    return _to_string_or_unicode_array(
        _vec_string(a, object_, 'encode', _clean_args(encoding, errors)))


def _endswith_dispatcher(a, suffix, start=None, end=None):
    return (a,)


@array_function_dispatch(_endswith_dispatcher)
def endswith(a, suffix, start=0, end=None):
    """
    Returns a boolean array which is `True` where the string element
    in `a` ends with `suffix`, otherwise `False`.

    Calls `str.endswith` element-wise.

    Parameters
    ----------
    a : array_like of str or unicode

    suffix : str

    start, end : int, optional
        With optional `start`, test beginning at that position. With
        optional `end`, stop comparing at that position.

    Returns
    -------
    out : ndarray
        Outputs an array of bools.

    See Also
    --------
    str.endswith

    Examples
    --------
    >>> s = np.array(['foo', 'bar'])
    >>> s[0] = 'foo'
    >>> s[1] = 'bar'
    >>> s
    array(['foo', 'bar'], dtype='<U3')
    >>> np.char.endswith(s, 'ar')
    array([False,  True])
    >>> np.char.endswith(s, 'a', start=1, end=2)
    array([False,  True])

    """
    return _vec_string(
        a, bool_, 'endswith', [suffix, start] + _clean_args(end))


def _expandtabs_dispatcher(a, tabsize=None):
    return (a,)


@array_function_dispatch(_expandtabs_dispatcher)
def expandtabs(a, tabsize=8):
    """
    Return a copy of each string element where all tab characters are
    replaced by one or more spaces.

    Calls `str.expandtabs` element-wise.

    Return a copy of each string element where all tab characters are
    replaced by one or more spaces, depending on the current column
    and the given `tabsize`. The column number is reset to zero after
    each newline occurring in the string. This doesn't understand other
    non-printing characters or escape sequences.

    Parameters
    ----------
    a : array_like of str or unicode
        Input array
    tabsize : int, optional
        Replace tabs with `tabsize` number of spaces.  If not given defaults
        to 8 spaces.

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input type

    See Also
    --------
    str.expandtabs

    """
    return _to_string_or_unicode_array(
        _vec_string(a, object_, 'expandtabs', (tabsize,)))


@array_function_dispatch(_count_dispatcher)
def find(a, sub, start=0, end=None):
    """
    For each element, return the lowest index in the string where
    substring `sub` is found.

    Calls `str.find` element-wise.

    For each element, return the lowest index in the string where
    substring `sub` is found, such that `sub` is contained in the
    range [`start`, `end`].

    Parameters
    ----------
    a : array_like of str or unicode

    sub : str or unicode

    start, end : int, optional
        Optional arguments `start` and `end` are interpreted as in
        slice notation.

    Returns
    -------
    out : ndarray or int
        Output array of ints.  Returns -1 if `sub` is not found.

    See Also
    --------
    str.find

    Examples
    --------
    >>> a = np.array(["NumPy is a Python library"])
    >>> np.char.find(a, "Python", start=0, end=None)
    array([11])

    """
    return _vec_string(
        a, int_, 'find', [sub, start] + _clean_args(end))


@array_function_dispatch(_count_dispatcher)
def index(a, sub, start=0, end=None):
    """
    Like `find`, but raises `ValueError` when the substring is not found.

    Calls `str.index` element-wise.

    Parameters
    ----------
    a : array_like of str or unicode

    sub : str or unicode

    start, end : int, optional

    Returns
    -------
    out : ndarray
        Output array of ints.  Returns -1 if `sub` is not found.

    See Also
    --------
    find, str.find

    Examples
    --------
    >>> a = np.array(["Computer Science"])
    >>> np.char.index(a, "Science", start=0, end=None)
    array([9])

    """
    return _vec_string(
        a, int_, 'index', [sub, start] + _clean_args(end))


@array_function_dispatch(_unary_op_dispatcher)
def isalnum(a):
    """
    Returns true for each element if all characters in the string are
    alphanumeric and there is at least one character, false otherwise.

    Calls `str.isalnum` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array_like of str or unicode

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input type

    See Also
    --------
    str.isalnum
    """
    return _vec_string(a, bool_, 'isalnum')


@array_function_dispatch(_unary_op_dispatcher)
def isalpha(a):
    """
    Returns true for each element if all characters in the string are
    alphabetic and there is at least one character, false otherwise.

    Calls `str.isalpha` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array_like of str or unicode

    Returns
    -------
    out : ndarray
        Output array of bools

    See Also
    --------
    str.isalpha
    """
    return _vec_string(a, bool_, 'isalpha')


@array_function_dispatch(_unary_op_dispatcher)
def isdigit(a):
    """
    Returns true for each element if all characters in the string are
    digits and there is at least one character, false otherwise.

    Calls `str.isdigit` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array_like of str or unicode

    Returns
    -------
    out : ndarray
        Output array of bools

    See Also
    --------
    str.isdigit

    Examples
    --------
    >>> a = np.array(['a', 'b', '0'])
    >>> np.char.isdigit(a)
    array([False, False,  True])
    >>> a = np.array([['a', 'b', '0'], ['c', '1', '2']])
    >>> np.char.isdigit(a)
    array([[False, False,  True], [False,  True,  True]])
    """
    return _vec_string(a, bool_, 'isdigit')


@array_function_dispatch(_unary_op_dispatcher)
def islower(a):
    """
    Returns true for each element if all cased characters in the
    string are lowercase and there is at least one cased character,
    false otherwise.

    Calls `str.islower` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array_like of str or unicode

    Returns
    -------
    out : ndarray
        Output array of bools

    See Also
    --------
    str.islower
    """
    return _vec_string(a, bool_, 'islower')


@array_function_dispatch(_unary_op_dispatcher)
def isspace(a):
    """
    Returns true for each element if there are only whitespace
    characters in the string and there is at least one character,
    false otherwise.

    Calls `str.isspace` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array_like of str or unicode

    Returns
    -------
    out : ndarray
        Output array of bools

    See Also
    --------
    str.isspace
    """
    return _vec_string(a, bool_, 'isspace')


@array_function_dispatch(_unary_op_dispatcher)
def istitle(a):
    """
    Returns true for each element if the element is a titlecased
    string and there is at least one character, false otherwise.

    Call `str.istitle` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array_like of str or unicode

    Returns
    -------
    out : ndarray
        Output array of bools

    See Also
    --------
    str.istitle
    """
    return _vec_string(a, bool_, 'istitle')


@array_function_dispatch(_unary_op_dispatcher)
def isupper(a):
    """
    Return true for each element if all cased characters in the
    string are uppercase and there is at least one character, false
    otherwise.

    Call `str.isupper` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array_like of str or unicode

    Returns
    -------
    out : ndarray
        Output array of bools

    See Also
    --------
    str.isupper

    Examples
    --------
    >>> str = "GHC"
    >>> np.char.isupper(str)
    array(True)     
    >>> a = np.array(["hello", "HELLO", "Hello"])
    >>> np.char.isupper(a)
    array([False,  True, False]) 

    """
    return _vec_string(a, bool_, 'isupper')


def _join_dispatcher(sep, seq):
    return (sep, seq)


@array_function_dispatch(_join_dispatcher)
def join(sep, seq):
    """
    Return a string which is the concatenation of the strings in the
    sequence `seq`.

    Calls `str.join` element-wise.

    Parameters
    ----------
    sep : array_like of str or unicode
    seq : array_like of str or unicode

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input types

    See Also
    --------
    str.join

    Examples
    --------
    >>> np.char.join('-', 'osd')
    array('o-s-d', dtype='<U5')

    >>> np.char.join(['-', '.'], ['ghc', 'osd'])
    array(['g-h-c', 'o.s.d'], dtype='<U5')

    """
    return _to_string_or_unicode_array(
        _vec_string(sep, object_, 'join', (seq,)))



def _just_dispatcher(a, width, fillchar=None):
    return (a,)


@array_function_dispatch(_just_dispatcher)
def ljust(a, width, fillchar=' '):
    """
    Return an array with the elements of `a` left-justified in a
    string of length `width`.

    Calls `str.ljust` element-wise.

    Parameters
    ----------
    a : array_like of str or unicode

    width : int
        The length of the resulting strings
    fillchar : str or unicode, optional
        The character to use for padding

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input type

    See Also
    --------
    str.ljust

    """
    a_arr = numpy.asarray(a)
    width_arr = numpy.asarray(width)
    size = int(numpy.max(width_arr.flat))
    if numpy.issubdtype(a_arr.dtype, numpy.string_):
        fillchar = asbytes(fillchar)
    return _vec_string(
        a_arr, (a_arr.dtype.type, size), 'ljust', (width_arr, fillchar))


@array_function_dispatch(_unary_op_dispatcher)
def lower(a):
    """
    Return an array with the elements converted to lowercase.

    Call `str.lower` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array_like, {str, unicode}
        Input array.

    Returns
    -------
    out : ndarray, {str, unicode}
        Output array of str or unicode, depending on input type

    See Also
    --------
    str.lower

    Examples
    --------
    >>> c = np.array(['A1B C', '1BCA', 'BCA1']); c
    array(['A1B C', '1BCA', 'BCA1'], dtype='<U5')
    >>> np.char.lower(c)
    array(['a1b c', '1bca', 'bca1'], dtype='<U5')

    """
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'lower')


def _strip_dispatcher(a, chars=None):
    return (a,)


@array_function_dispatch(_strip_dispatcher)
def lstrip(a, chars=None):
    """
    For each element in `a`, return a copy with the leading characters
    removed.

    Calls `str.lstrip` element-wise.

    Parameters
    ----------
    a : array-like, {str, unicode}
        Input array.

    chars : {str, unicode}, optional
        The `chars` argument is a string specifying the set of
        characters to be removed. If omitted or None, the `chars`
        argument defaults to removing whitespace. The `chars` argument
        is not a prefix; rather, all combinations of its values are
        stripped.

    Returns
    -------
    out : ndarray, {str, unicode}
        Output array of str or unicode, depending on input type

    See Also
    --------
    str.lstrip

    Examples
    --------
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')

    The 'a' variable is unstripped from c[1] because whitespace leading.

    >>> np.char.lstrip(c, 'a')
    array(['AaAaA', '  aA  ', 'bBABba'], dtype='<U7')


    >>> np.char.lstrip(c, 'A') # leaves c unchanged
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> (np.char.lstrip(c, ' ') == np.char.lstrip(c, '')).all()
    ... # XXX: is this a regression? This used to return True
    ... # np.char.lstrip(c,'') does not modify c at all.
    False
    >>> (np.char.lstrip(c, ' ') == np.char.lstrip(c, None)).all()
    True

    """
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'lstrip', (chars,))


def _partition_dispatcher(a, sep):
    return (a,)


@array_function_dispatch(_partition_dispatcher)
def partition(a, sep):
    """
    Partition each element in `a` around `sep`.

    Calls `str.partition` element-wise.

    For each element in `a`, split the element as the first
    occurrence of `sep`, and return 3 strings containing the part
    before the separator, the separator itself, and the part after
    the separator. If the separator is not found, return 3 strings
    containing the string itself, followed by two empty strings.

    Parameters
    ----------
    a : array_like, {str, unicode}
        Input array
    sep : {str, unicode}
        Separator to split each string element in `a`.

    Returns
    -------
    out : ndarray, {str, unicode}
        Output array of str or unicode, depending on input type.
        The output array will have an extra dimension with 3
        elements per input element.

    See Also
    --------
    str.partition

    """
    return _to_string_or_unicode_array(
        _vec_string(a, object_, 'partition', (sep,)))


def _replace_dispatcher(a, old, new, count=None):
    return (a,)


@array_function_dispatch(_replace_dispatcher)
def replace(a, old, new, count=None):
    """
    For each element in `a`, return a copy of the string with all
    occurrences of substring `old` replaced by `new`.

    Calls `str.replace` element-wise.

    Parameters
    ----------
    a : array-like of str or unicode

    old, new : str or unicode

    count : int, optional
        If the optional argument `count` is given, only the first
        `count` occurrences are replaced.

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input type

    See Also
    --------
    str.replace
    
    Examples
    --------
    >>> a = np.array(["That is a mango", "Monkeys eat mangos"])
    >>> np.char.replace(a, 'mango', 'banana')
    array(['That is a banana', 'Monkeys eat bananas'], dtype='<U19')

    >>> a = np.array(["The dish is fresh", "This is it"])
    >>> np.char.replace(a, 'is', 'was')
    array(['The dwash was fresh', 'Thwas was it'], dtype='<U19')
    """
    return _to_string_or_unicode_array(
        _vec_string(
            a, object_, 'replace', [old, new] + _clean_args(count)))


@array_function_dispatch(_count_dispatcher)
def rfind(a, sub, start=0, end=None):
    """
    For each element in `a`, return the highest index in the string
    where substring `sub` is found, such that `sub` is contained
    within [`start`, `end`].

    Calls `str.rfind` element-wise.

    Parameters
    ----------
    a : array-like of str or unicode

    sub : str or unicode

    start, end : int, optional
        Optional arguments `start` and `end` are interpreted as in
        slice notation.

    Returns
    -------
    out : ndarray
       Output array of ints.  Return -1 on failure.

    See Also
    --------
    str.rfind

    """
    return _vec_string(
        a, int_, 'rfind', [sub, start] + _clean_args(end))


@array_function_dispatch(_count_dispatcher)
def rindex(a, sub, start=0, end=None):
    """
    Like `rfind`, but raises `ValueError` when the substring `sub` is
    not found.

    Calls `str.rindex` element-wise.

    Parameters
    ----------
    a : array-like of str or unicode

    sub : str or unicode

    start, end : int, optional

    Returns
    -------
    out : ndarray
       Output array of ints.

    See Also
    --------
    rfind, str.rindex

    """
    return _vec_string(
        a, int_, 'rindex', [sub, start] + _clean_args(end))


@array_function_dispatch(_just_dispatcher)
def rjust(a, width, fillchar=' '):
    """
    Return an array with the elements of `a` right-justified in a
    string of length `width`.

    Calls `str.rjust` element-wise.

    Parameters
    ----------
    a : array_like of str or unicode

    width : int
        The length of the resulting strings
    fillchar : str or unicode, optional
        The character to use for padding

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input type

    See Also
    --------
    str.rjust

    """
    a_arr = numpy.asarray(a)
    width_arr = numpy.asarray(width)
    size = int(numpy.max(width_arr.flat))
    if numpy.issubdtype(a_arr.dtype, numpy.string_):
        fillchar = asbytes(fillchar)
    return _vec_string(
        a_arr, (a_arr.dtype.type, size), 'rjust', (width_arr, fillchar))


@array_function_dispatch(_partition_dispatcher)
def rpartition(a, sep):
    """
    Partition (split) each element around the right-most separator.

    Calls `str.rpartition` element-wise.

    For each element in `a`, split the element as the last
    occurrence of `sep`, and return 3 strings containing the part
    before the separator, the separator itself, and the part after
    the separator. If the separator is not found, return 3 strings
    containing the string itself, followed by two empty strings.

    Parameters
    ----------
    a : array_like of str or unicode
        Input array
    sep : str or unicode
        Right-most separator to split each element in array.

    Returns
    -------
    out : ndarray
        Output array of string or unicode, depending on input
        type.  The output array will have an extra dimension with
        3 elements per input element.

    See Also
    --------
    str.rpartition

    """
    return _to_string_or_unicode_array(
        _vec_string(a, object_, 'rpartition', (sep,)))


def _split_dispatcher(a, sep=None, maxsplit=None):
    return (a,)


@array_function_dispatch(_split_dispatcher)
def rsplit(a, sep=None, maxsplit=None):
    """
    For each element in `a`, return a list of the words in the
    string, using `sep` as the delimiter string.

    Calls `str.rsplit` element-wise.

    Except for splitting from the right, `rsplit`
    behaves like `split`.

    Parameters
    ----------
    a : array_like of str or unicode

    sep : str or unicode, optional
        If `sep` is not specified or None, any whitespace string
        is a separator.
    maxsplit : int, optional
        If `maxsplit` is given, at most `maxsplit` splits are done,
        the rightmost ones.

    Returns
    -------
    out : ndarray
       Array of list objects

    See Also
    --------
    str.rsplit, split

    """
    # This will return an array of lists of different sizes, so we
    # leave it as an object array
    return _vec_string(
        a, object_, 'rsplit', [sep] + _clean_args(maxsplit))


def _strip_dispatcher(a, chars=None):
    return (a,)


@array_function_dispatch(_strip_dispatcher)
def rstrip(a, chars=None):
    """
    For each element in `a`, return a copy with the trailing
    characters removed.

    Calls `str.rstrip` element-wise.

    Parameters
    ----------
    a : array-like of str or unicode

    chars : str or unicode, optional
       The `chars` argument is a string specifying the set of
       characters to be removed. If omitted or None, the `chars`
       argument defaults to removing whitespace. The `chars` argument
       is not a suffix; rather, all combinations of its values are
       stripped.

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input type

    See Also
    --------
    str.rstrip

    Examples
    --------
    >>> c = np.array(['aAaAaA', 'abBABba'], dtype='S7'); c
    array(['aAaAaA', 'abBABba'],
        dtype='|S7')
    >>> np.char.rstrip(c, b'a')
    array(['aAaAaA', 'abBABb'],
        dtype='|S7')
    >>> np.char.rstrip(c, b'A')
    array(['aAaAa', 'abBABba'],
        dtype='|S7')

    """
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'rstrip', (chars,))


@array_function_dispatch(_split_dispatcher)
def split(a, sep=None, maxsplit=None):
    """
    For each element in `a`, return a list of the words in the
    string, using `sep` as the delimiter string.

    Calls `str.split` element-wise.

    Parameters
    ----------
    a : array_like of str or unicode

    sep : str or unicode, optional
       If `sep` is not specified or None, any whitespace string is a
       separator.

    maxsplit : int, optional
        If `maxsplit` is given, at most `maxsplit` splits are done.

    Returns
    -------
    out : ndarray
        Array of list objects

    See Also
    --------
    str.split, rsplit

    """
    # This will return an array of lists of different sizes, so we
    # leave it as an object array
    return _vec_string(
        a, object_, 'split', [sep] + _clean_args(maxsplit))


def _splitlines_dispatcher(a, keepends=None):
    return (a,)


@array_function_dispatch(_splitlines_dispatcher)
def splitlines(a, keepends=None):
    """
    For each element in `a`, return a list of the lines in the
    element, breaking at line boundaries.

    Calls `str.splitlines` element-wise.

    Parameters
    ----------
    a : array_like of str or unicode

    keepends : bool, optional
        Line breaks are not included in the resulting list unless
        keepends is given and true.

    Returns
    -------
    out : ndarray
        Array of list objects

    See Also
    --------
    str.splitlines

    """
    return _vec_string(
        a, object_, 'splitlines', _clean_args(keepends))


def _startswith_dispatcher(a, prefix, start=None, end=None):
    return (a,)


@array_function_dispatch(_startswith_dispatcher)
def startswith(a, prefix, start=0, end=None):
    """
    Returns a boolean array which is `True` where the string element
    in `a` starts with `prefix`, otherwise `False`.

    Calls `str.startswith` element-wise.

    Parameters
    ----------
    a : array_like of str or unicode

    prefix : str

    start, end : int, optional
        With optional `start`, test beginning at that position. With
        optional `end`, stop comparing at that position.

    Returns
    -------
    out : ndarray
        Array of booleans

    See Also
    --------
    str.startswith

    """
    return _vec_string(
        a, bool_, 'startswith', [prefix, start] + _clean_args(end))


@array_function_dispatch(_strip_dispatcher)
def strip(a, chars=None):
    """
    For each element in `a`, return a copy with the leading and
    trailing characters removed.

    Calls `str.strip` element-wise.

    Parameters
    ----------
    a : array-like of str or unicode

    chars : str or unicode, optional
       The `chars` argument is a string specifying the set of
       characters to be removed. If omitted or None, the `chars`
       argument defaults to removing whitespace. The `chars` argument
       is not a prefix or suffix; rather, all combinations of its
       values are stripped.

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input type

    See Also
    --------
    str.strip

    Examples
    --------
    >>> c = np.array(['aAaAaA', '  aA  ', 'abBABba'])
    >>> c
    array(['aAaAaA', '  aA  ', 'abBABba'], dtype='<U7')
    >>> np.char.strip(c)
    array(['aAaAaA', 'aA', 'abBABba'], dtype='<U7')
    >>> np.char.strip(c, 'a') # 'a' unstripped from c[1] because whitespace leads
    array(['AaAaA', '  aA  ', 'bBABb'], dtype='<U7')
    >>> np.char.strip(c, 'A') # 'A' unstripped from c[1] because (unprinted) ws trails
    array(['aAaAa', '  aA  ', 'abBABba'], dtype='<U7')

    """
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'strip', _clean_args(chars))


@array_function_dispatch(_unary_op_dispatcher)
def swapcase(a):
    """
    Return element-wise a copy of the string with
    uppercase characters converted to lowercase and vice versa.

    Calls `str.swapcase` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array_like, {str, unicode}
        Input array.

    Returns
    -------
    out : ndarray, {str, unicode}
        Output array of str or unicode, depending on input type

    See Also
    --------
    str.swapcase

    Examples
    --------
    >>> c=np.array(['a1B c','1b Ca','b Ca1','cA1b'],'S5'); c
    array(['a1B c', '1b Ca', 'b Ca1', 'cA1b'],
        dtype='|S5')
    >>> np.char.swapcase(c)
    array(['A1b C', '1B cA', 'B cA1', 'Ca1B'],
        dtype='|S5')

    """
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'swapcase')


@array_function_dispatch(_unary_op_dispatcher)
def title(a):
    """
    Return element-wise title cased version of string or unicode.

    Title case words start with uppercase characters, all remaining cased
    characters are lowercase.

    Calls `str.title` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array_like, {str, unicode}
        Input array.

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input type

    See Also
    --------
    str.title

    Examples
    --------
    >>> c=np.array(['a1b c','1b ca','b ca1','ca1b'],'S5'); c
    array(['a1b c', '1b ca', 'b ca1', 'ca1b'],
        dtype='|S5')
    >>> np.char.title(c)
    array(['A1B C', '1B Ca', 'B Ca1', 'Ca1B'],
        dtype='|S5')

    """
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'title')


def _translate_dispatcher(a, table, deletechars=None):
    return (a,)


@array_function_dispatch(_translate_dispatcher)
def translate(a, table, deletechars=None):
    """
    For each element in `a`, return a copy of the string where all
    characters occurring in the optional argument `deletechars` are
    removed, and the remaining characters have been mapped through the
    given translation table.

    Calls `str.translate` element-wise.

    Parameters
    ----------
    a : array-like of str or unicode

    table : str of length 256

    deletechars : str

    Returns
    -------
    out : ndarray
        Output array of str or unicode, depending on input type

    See Also
    --------
    str.translate

    """
    a_arr = numpy.asarray(a)
    if issubclass(a_arr.dtype.type, unicode_):
        return _vec_string(
            a_arr, a_arr.dtype, 'translate', (table,))
    else:
        return _vec_string(
            a_arr, a_arr.dtype, 'translate', [table] + _clean_args(deletechars))


@array_function_dispatch(_unary_op_dispatcher)
def upper(a):
    """
    Return an array with the elements converted to uppercase.

    Calls `str.upper` element-wise.

    For 8-bit strings, this method is locale-dependent.

    Parameters
    ----------
    a : array_like, {str, unicode}
        Input array.

    Returns
    -------
    out : ndarray, {str, unicode}
        Output array of str or unicode, depending on input type

    See Also
    --------
    str.upper

    Examples
    --------
    >>> c = np.array(['a1b c', '1bca', 'bca1']); c
    array(['a1b c', '1bca', 'bca1'], dtype='<U5')
    >>> np.char.upper(c)
    array(['A1B C', '1BCA', 'BCA1'], dtype='<U5')

    """
    a_arr = numpy.asarray(a)
    return _vec_string(a_arr, a_arr.dtype, 'upper')


def _zfill_dispatcher(a, width):
    return (a,)


@array_function_dispatch(_zfill_dispatcher)
def zfill(a, width):
    """
    Return the numeric string left-filled with zeros

    Calls `str.zfill` element-wise.

    Parameters
    ----------
    a : array_like, {str, unicode}
        Input array.
    width : int
        Width of string to left-fill elements in `a`.

    Returns
    -------
    out : ndarray, {str, unicode}
        Output array of str or unicode, depending on input type

    See Also
    --------
    str.zfill

    """
    a_arr = numpy.asarray(a)
    width_arr = numpy.asarray(width)
    size = int(numpy.max(width_arr.flat))
    return _vec_string(
        a_arr, (a_arr.dtype.type, size), 'zfill', (width_arr,))


@array_function_dispatch(_unary_op_dispatcher)
def isnumeric(a):
    """
    For each element, return True if there are only numeric
    characters in the element.

    Calls `unicode.isnumeric` element-wise.

    Numeric characters include digit characters, and all characters
    that have the Unicode numeric value property, e.g. ``U+2155,
    VULGAR FRACTION ONE FIFTH``.

    Parameters
    ----------
    a : array_like, unicode
        Input array.

    Returns
    -------
    out : ndarray, bool
        Array of booleans of same shape as `a`.

    See Also
    --------
    unicode.isnumeric

    Examples
    --------
    >>> np.char.isnumeric(['123', '123abc', '9.0', '1/4', 'VIII'])
    array([ True, False, False, False, False])

    """
    if _use_unicode(a) != unicode_:
        raise TypeError("isnumeric is only available for Unicode strings and arrays")
    return _vec_string(a, bool_, 'isnumeric')


@array_function_dispatch(_unary_op_dispatcher)
def isdecimal(a):
    """
    For each element, return True if there are only decimal
    characters in the element.

    Calls `unicode.isdecimal` element-wise.

    Decimal characters include digit characters, and all characters
    that can be used to form decimal-radix numbers,
    e.g. ``U+0660, ARABIC-INDIC DIGIT ZERO``.

    Parameters
    ----------
    a : array_like, unicode
        Input array.

    Returns
    -------
    out : ndarray, bool
        Array of booleans identical in shape to `a`.

    See Also
    --------
    unicode.isdecimal

    Examples
    --------
    >>> np.char.isdecimal(['12345', '4.99', '123ABC', ''])
    array([ True, False, False, False])

    """ 
    if _use_unicode(a) != unicode_:
        raise TypeError("isnumeric is only available for Unicode strings and arrays")
    return _vec_string(a, bool_, 'isdecimal')


@set_module('numpy')
class chararray(ndarray):
    """
    chararray(shape, itemsize=1, unicode=False, buffer=None, offset=0,
              strides=None, order=None)

    Provides a convenient view on arrays of string and unicode values.

    .. note::
       The `chararray` class exists for backwards compatibility with
       Numarray, it is not recommended for new development. Starting from numpy
       1.4, if one needs arrays of strings, it is recommended to use arrays of
       `dtype` `object_`, `string_` or `unicode_`, and use the free functions
       in the `numpy.char` module for fast vectorized string operations.

    Versus a regular NumPy array of type `str` or `unicode`, this
    class adds the following functionality:

      1) values automatically have whitespace removed from the end
         when indexed

      2) comparison operators automatically remove whitespace from the
         end when comparing values

      3) vectorized string operations are provided as methods
         (e.g. `.endswith`) and infix operators (e.g. ``"+", "*", "%"``)

    chararrays should be created using `numpy.char.array` or
    `numpy.char.asarray`, rather than this constructor directly.

    This constructor creates the array, using `buffer` (with `offset`
    and `strides`) if it is not ``None``. If `buffer` is ``None``, then
    constructs a new array with `strides` in "C order", unless both
    ``len(shape) >= 2`` and ``order='F'``, in which case `strides`
    is in "Fortran order".

    Methods
    -------
    astype
    argsort
    copy
    count
    decode
    dump
    dumps
    encode
    endswith
    expandtabs
    fill
    find
    flatten
    getfield
    index
    isalnum
    isalpha
    isdecimal
    isdigit
    islower
    isnumeric
    isspace
    istitle
    isupper
    item
    join
    ljust
    lower
    lstrip
    nonzero
    put
    ravel
    repeat
    replace
    reshape
    resize
    rfind
    rindex
    rjust
    rsplit
    rstrip
    searchsorted
    setfield
    setflags
    sort
    split
    splitlines
    squeeze
    startswith
    strip
    swapaxes
    swapcase
    take
    title
    tofile
    tolist
    tostring
    translate
    transpose
    upper
    view
    zfill

    Parameters
    ----------
    shape : tuple
        Shape of the array.
    itemsize : int, optional
        Length of each array element, in number of characters. Default is 1.
    unicode : bool, optional
        Are the array elements of type unicode (True) or string (False).
        Default is False.
    buffer : object exposing the buffer interface or str, optional
        Memory address of the start of the array data.  Default is None,
        in which case a new array is created.
    offset : int, optional
        Fixed stride displacement from the beginning of an axis?
        Default is 0. Needs to be >=0.
    strides : array_like of ints, optional
        Strides for the array (see `ndarray.strides` for full description).
        Default is None.
    order : {'C', 'F'}, optional
        The order in which the array data is stored in memory: 'C' ->
        "row major" order (the default), 'F' -> "column major"
        (Fortran) order.

    Examples
    --------
    >>> charar = np.chararray((3, 3))
    >>> charar[:] = 'a'
    >>> charar
    chararray([[b'a', b'a', b'a'],
               [b'a', b'a', b'a'],
               [b'a', b'a', b'a']], dtype='|S1')

    >>> charar = np.chararray(charar.shape, itemsize=5)
    >>> charar[:] = 'abc'
    >>> charar
    chararray([[b'abc', b'abc', b'abc'],
               [b'abc', b'abc', b'abc'],
               [b'abc', b'abc', b'abc']], dtype='|S5')

    """
    def __new__(subtype, shape, itemsize=1, unicode=False, buffer=None,
                offset=0, strides=None, order='C'):
        global _globalvar

        if unicode:
            dtype = unicode_
        else:
            dtype = string_

        # force itemsize to be a Python int, since using NumPy integer
        # types results in itemsize.itemsize being used as the size of
        # strings in the new array.
        itemsize = int(itemsize)

        if isinstance(buffer, str):
            # unicode objects do not have the buffer interface
            filler = buffer
            buffer = None
        else:
            filler = None

        _globalvar = 1
        if buffer is None:
            self = ndarray.__new__(subtype, shape, (dtype, itemsize),
                                   order=order)
        else:
            self = ndarray.__new__(subtype, shape, (dtype, itemsize),
                                   buffer=buffer,
                                   offset=offset, strides=strides,
                                   order=order)
        if filler is not None:
            self[...] = filler
        _globalvar = 0
        return self

    def __array_finalize__(self, obj):
        # The b is a special case because it is used for reconstructing.
        if not _globalvar and self.dtype.char not in 'SUbc':
            raise ValueError("Can only create a chararray from string data.")

    def __getitem__(self, obj):
        val = ndarray.__getitem__(self, obj)

        if isinstance(val, character):
            temp = val.rstrip()
            if len(temp) == 0:
                val = ''
            else:
                val = temp

        return val

    # IMPLEMENTATION NOTE: Most of the methods of this class are
    # direct delegations to the free functions in this module.
    # However, those that return an array of strings should instead
    # return a chararray, so some extra wrapping is required.

    def __eq__(self, other):
        """
        Return (self == other) element-wise.

        See Also
        --------
        equal
        """
        return equal(self, other)

    def __ne__(self, other):
        """
        Return (self != other) element-wise.

        See Also
        --------
        not_equal
        """
        return not_equal(self, other)

    def __ge__(self, other):
        """
        Return (self >= other) element-wise.

        See Also
        --------
        greater_equal
        """
        return greater_equal(self, other)

    def __le__(self, other):
        """
        Return (self <= other) element-wise.

        See Also
        --------
        less_equal
        """
        return less_equal(self, other)

    def __gt__(self, other):
        """
        Return (self > other) element-wise.

        See Also
        --------
        greater
        """
        return greater(self, other)

    def __lt__(self, other):
        """
        Return (self < other) element-wise.

        See Also
        --------
        less
        """
        return less(self, other)

    def __add__(self, other):
        """
        Return (self + other), that is string concatenation,
        element-wise for a pair of array_likes of str or unicode.

        See Also
        --------
        add
        """
        return asarray(add(self, other))

    def __radd__(self, other):
        """
        Return (other + self), that is string concatenation,
        element-wise for a pair of array_likes of `string_` or `unicode_`.

        See Also
        --------
        add
        """
        return asarray(add(numpy.asarray(other), self))

    def __mul__(self, i):
        """
        Return (self * i), that is string multiple concatenation,
        element-wise.

        See Also
        --------
        multiply
        """
        return asarray(multiply(self, i))

    def __rmul__(self, i):
        """
        Return (self * i), that is string multiple concatenation,
        element-wise.

        See Also
        --------
        multiply
        """
        return asarray(multiply(self, i))

    def __mod__(self, i):
        """
        Return (self % i), that is pre-Python 2.6 string formatting
        (interpolation), element-wise for a pair of array_likes of `string_`
        or `unicode_`.

        See Also
        --------
        mod
        """
        return asarray(mod(self, i))

    def __rmod__(self, other):
        return NotImplemented

    def argsort(self, axis=-1, kind=None, order=None):
        """
        Return the indices that sort the array lexicographically.

        For full documentation see `numpy.argsort`, for which this method is
        in fact merely a "thin wrapper."

        Examples
        --------
        >>> c = np.array(['a1b c', '1b ca', 'b ca1', 'Ca1b'], 'S5')
        >>> c = c.view(np.chararray); c
        chararray(['a1b c', '1b ca', 'b ca1', 'Ca1b'],
              dtype='|S5')
        >>> c[c.argsort()]
        chararray(['1b ca', 'Ca1b', 'a1b c', 'b ca1'],
              dtype='|S5')

        """
        return self.__array__().argsort(axis, kind, order)
    argsort.__doc__ = ndarray.argsort.__doc__

    def capitalize(self):
        """
        Return a copy of `self` with only the first character of each element
        capitalized.

        See Also
        --------
        char.capitalize

        """
        return asarray(capitalize(self))

    def center(self, width, fillchar=' '):
        """
        Return a copy of `self` with its elements centered in a
        string of length `width`.

        See Also
        --------
        center
        """
        return asarray(center(self, width, fillchar))

    def count(self, sub, start=0, end=None):
        """
        Returns an array with the number of non-overlapping occurrences of
        substring `sub` in the range [`start`, `end`].

        See Also
        --------
        char.count

        """
        return count(self, sub, start, end)

    def decode(self, encoding=None, errors=None):
        """
        Calls ``bytes.decode`` element-wise.

        See Also
        --------
        char.decode

        """
        return decode(self, encoding, errors)

    def encode(self, encoding=None, errors=None):
        """
        Calls `str.encode` element-wise.

        See Also
        --------
        char.encode

        """
        return encode(self, encoding, errors)

    def endswith(self, suffix, start=0, end=None):
        """
        Returns a boolean array which is `True` where the string element
        in `self` ends with `suffix`, otherwise `False`.

        See Also
        --------
        char.endswith

        """
        return endswith(self, suffix, start, end)

    def expandtabs(self, tabsize=8):
        """
        Return a copy of each string element where all tab characters are
        replaced by one or more spaces.

        See Also
        --------
        char.expandtabs

        """
        return asarray(expandtabs(self, tabsize))

    def find(self, sub, start=0, end=None):
        """
        For each element, return the lowest index in the string where
        substring `sub` is found.

        See Also
        --------
        char.find

        """
        return find(self, sub, start, end)

    def index(self, sub, start=0, end=None):
        """
        Like `find`, but raises `ValueError` when the substring is not found.

        See Also
        --------
        char.index

        """
        return index(self, sub, start, end)

    def isalnum(self):
        """
        Returns true for each element if all characters in the string
        are alphanumeric and there is at least one character, false
        otherwise.

        See Also
        --------
        char.isalnum

        """
        return isalnum(self)

    def isalpha(self):
        """
        Returns true for each element if all characters in the string
        are alphabetic and there is at least one character, false
        otherwise.

        See Also
        --------
        char.isalpha

        """
        return isalpha(self)

    def isdigit(self):
        """
        Returns true for each element if all characters in the string are
        digits and there is at least one character, false otherwise.

        See Also
        --------
        char.isdigit

        """
        return isdigit(self)

    def islower(self):
        """
        Returns true for each element if all cased characters in the
        string are lowercase and there is at least one cased character,
        false otherwise.

        See Also
        --------
        char.islower

        """
        return islower(self)

    def isspace(self):
        """
        Returns true for each element if there are only whitespace
        characters in the string and there is at least one character,
        false otherwise.

        See Also
        --------
        char.isspace

        """
        return isspace(self)

    def istitle(self):
        """
        Returns true for each element if the element is a titlecased
        string and there is at least one character, false otherwise.

        See Also
        --------
        char.istitle

        """
        return istitle(self)

    def isupper(self):
        """
        Returns true for each element if all cased characters in the
        string are uppercase and there is at least one character, false
        otherwise.

        See Also
        --------
        char.isupper

        """
        return isupper(self)

    def join(self, seq):
        """
        Return a string which is the concatenation of the strings in the
        sequence `seq`.

        See Also
        --------
        char.join

        """
        return join(self, seq)

    def ljust(self, width, fillchar=' '):
        """
        Return an array with the elements of `self` left-justified in a
        string of length `width`.

        See Also
        --------
        char.ljust

        """
        return asarray(ljust(self, width, fillchar))

    def lower(self):
        """
        Return an array with the elements of `self` converted to
        lowercase.

        See Also
        --------
        char.lower

        """
        return asarray(lower(self))

    def lstrip(self, chars=None):
        """
        For each element in `self`, return a copy with the leading characters
        removed.

        See Also
        --------
        char.lstrip

        """
        return asarray(lstrip(self, chars))

    def partition(self, sep):
        """
        Partition each element in `self` around `sep`.

        See Also
        --------
        partition
        """
        return asarray(partition(self, sep))

    def replace(self, old, new, count=None):
        """
        For each element in `self`, return a copy of the string with all
        occurrences of substring `old` replaced by `new`.

        See Also
        --------
        char.replace

        """
        return asarray(replace(self, old, new, count))

    def rfind(self, sub, start=0, end=None):
        """
        For each element in `self`, return the highest index in the string
        where substring `sub` is found, such that `sub` is contained
        within [`start`, `end`].

        See Also
        --------
        char.rfind

        """
        return rfind(self, sub, start, end)

    def rindex(self, sub, start=0, end=None):
        """
        Like `rfind`, but raises `ValueError` when the substring `sub` is
        not found.

        See Also
        --------
        char.rindex

        """
        return rindex(self, sub, start, end)

    def rjust(self, width, fillchar=' '):
        """
        Return an array with the elements of `self`
        right-justified in a string of length `width`.

        See Also
        --------
        char.rjust

        """
        return asarray(rjust(self, width, fillchar))

    def rpartition(self, sep):
        """
        Partition each element in `self` around `sep`.

        See Also
        --------
        rpartition
        """
        return asarray(rpartition(self, sep))

    def rsplit(self, sep=None, maxsplit=None):
        """
        For each element in `self`, return a list of the words in
        the string, using `sep` as the delimiter string.

        See Also
        --------
        char.rsplit

        """
        return rsplit(self, sep, maxsplit)

    def rstrip(self, chars=None):
        """
        For each element in `self`, return a copy with the trailing
        characters removed.

        See Also
        --------
        char.rstrip

        """
        return asarray(rstrip(self, chars))

    def split(self, sep=None, maxsplit=None):
        """
        For each element in `self`, return a list of the words in the
        string, using `sep` as the delimiter string.

        See Also
        --------
        char.split

        """
        return split(self, sep, maxsplit)

    def splitlines(self, keepends=None):
        """
        For each element in `self`, return a list of the lines in the
        element, breaking at line boundaries.

        See Also
        --------
        char.splitlines

        """
        return splitlines(self, keepends)

    def startswith(self, prefix, start=0, end=None):
        """
        Returns a boolean array which is `True` where the string element
        in `self` starts with `prefix`, otherwise `False`.

        See Also
        --------
        char.startswith

        """
        return startswith(self, prefix, start, end)

    def strip(self, chars=None):
        """
        For each element in `self`, return a copy with the leading and
        trailing characters removed.

        See Also
        --------
        char.strip

        """
        return asarray(strip(self, chars))

    def swapcase(self):
        """
        For each element in `self`, return a copy of the string with
        uppercase characters converted to lowercase and vice versa.

        See Also
        --------
        char.swapcase

        """
        return asarray(swapcase(self))

    def title(self):
        """
        For each element in `self`, return a titlecased version of the
        string: words start with uppercase characters, all remaining cased
        characters are lowercase.

        See Also
        --------
        char.title

        """
        return asarray(title(self))

    def translate(self, table, deletechars=None):
        """
        For each element in `self`, return a copy of the string where
        all characters occurring in the optional argument
        `deletechars` are removed, and the remaining characters have
        been mapped through the given translation table.

        See Also
        --------
        char.translate

        """
        return asarray(translate(self, table, deletechars))

    def upper(self):
        """
        Return an array with the elements of `self` converted to
        uppercase.

        See Also
        --------
        char.upper

        """
        return asarray(upper(self))

    def zfill(self, width):
        """
        Return the numeric string left-filled with zeros in a string of
        length `width`.

        See Also
        --------
        char.zfill

        """
        return asarray(zfill(self, width))

    def isnumeric(self):
        """
        For each element in `self`, return True if there are only
        numeric characters in the element.

        See Also
        --------
        char.isnumeric

        """
        return isnumeric(self)

    def isdecimal(self):
        """
        For each element in `self`, return True if there are only
        decimal characters in the element.

        See Also
        --------
        char.isdecimal

        """
        return isdecimal(self)


@set_module("numpy.char")
def array(obj, itemsize=None, copy=True, unicode=None, order=None):
    """
    Create a `chararray`.

    .. note::
       This class is provided for numarray backward-compatibility.
       New code (not concerned with numarray compatibility) should use
       arrays of type `string_` or `unicode_` and use the free functions
       in :mod:`numpy.char <numpy.core.defchararray>` for fast
       vectorized string operations instead.

    Versus a regular NumPy array of type `str` or `unicode`, this
    class adds the following functionality:

      1) values automatically have whitespace removed from the end
         when indexed

      2) comparison operators automatically remove whitespace from the
         end when comparing values

      3) vectorized string operations are provided as methods
         (e.g. `str.endswith`) and infix operators (e.g. ``+, *, %``)

    Parameters
    ----------
    obj : array of str or unicode-like

    itemsize : int, optional
        `itemsize` is the number of characters per scalar in the
        resulting array.  If `itemsize` is None, and `obj` is an
        object array or a Python list, the `itemsize` will be
        automatically determined.  If `itemsize` is provided and `obj`
        is of type str or unicode, then the `obj` string will be
        chunked into `itemsize` pieces.

    copy : bool, optional
        If true (default), then the object is copied.  Otherwise, a copy
        will only be made if __array__ returns a copy, if obj is a
        nested sequence, or if a copy is needed to satisfy any of the other
        requirements (`itemsize`, unicode, `order`, etc.).

    unicode : bool, optional
        When true, the resulting `chararray` can contain Unicode
        characters, when false only 8-bit characters.  If unicode is
        None and `obj` is one of the following:

          - a `chararray`,
          - an ndarray of type `str` or `unicode`
          - a Python str or unicode object,

        then the unicode setting of the output array will be
        automatically determined.

    order : {'C', 'F', 'A'}, optional
        Specify the order of the array.  If order is 'C' (default), then the
        array will be in C-contiguous order (last-index varies the
        fastest).  If order is 'F', then the returned array
        will be in Fortran-contiguous order (first-index varies the
        fastest).  If order is 'A', then the returned array may
        be in any order (either C-, Fortran-contiguous, or even
        discontiguous).
    """
    if isinstance(obj, (bytes, str)):
        if unicode is None:
            if isinstance(obj, str):
                unicode = True
            else:
                unicode = False

        if itemsize is None:
            itemsize = len(obj)
        shape = len(obj) // itemsize

        return chararray(shape, itemsize=itemsize, unicode=unicode,
                         buffer=obj, order=order)

    if isinstance(obj, (list, tuple)):
        obj = numpy.asarray(obj)

    if isinstance(obj, ndarray) and issubclass(obj.dtype.type, character):
        # If we just have a vanilla chararray, create a chararray
        # view around it.
        if not isinstance(obj, chararray):
            obj = obj.view(chararray)

        if itemsize is None:
            itemsize = obj.itemsize
            # itemsize is in 8-bit chars, so for Unicode, we need
            # to divide by the size of a single Unicode character,
            # which for NumPy is always 4
            if issubclass(obj.dtype.type, unicode_):
                itemsize //= 4

        if unicode is None:
            if issubclass(obj.dtype.type, unicode_):
                unicode = True
            else:
                unicode = False

        if unicode:
            dtype = unicode_
        else:
            dtype = string_

        if order is not None:
            obj = numpy.asarray(obj, order=order)
        if (copy or
                (itemsize != obj.itemsize) or
                (not unicode and isinstance(obj, unicode_)) or
                (unicode and isinstance(obj, string_))):
            obj = obj.astype((dtype, int(itemsize)))
        return obj

    if isinstance(obj, ndarray) and issubclass(obj.dtype.type, object):
        if itemsize is None:
            # Since no itemsize was specified, convert the input array to
            # a list so the ndarray constructor will automatically
            # determine the itemsize for us.
            obj = obj.tolist()
            # Fall through to the default case

    if unicode:
        dtype = unicode_
    else:
        dtype = string_

    if itemsize is None:
        val = narray(obj, dtype=dtype, order=order, subok=True)
    else:
        val = narray(obj, dtype=(dtype, itemsize), order=order, subok=True)
    return val.view(chararray)


@set_module("numpy.char")
def asarray(obj, itemsize=None, unicode=None, order=None):
    """
    Convert the input to a `chararray`, copying the data only if
    necessary.

    Versus a regular NumPy array of type `str` or `unicode`, this
    class adds the following functionality:

      1) values automatically have whitespace removed from the end
         when indexed

      2) comparison operators automatically remove whitespace from the
         end when comparing values

      3) vectorized string operations are provided as methods
         (e.g. `str.endswith`) and infix operators (e.g. ``+``, ``*``,``%``)

    Parameters
    ----------
    obj : array of str or unicode-like

    itemsize : int, optional
        `itemsize` is the number of characters per scalar in the
        resulting array.  If `itemsize` is None, and `obj` is an
        object array or a Python list, the `itemsize` will be
        automatically determined.  If `itemsize` is provided and `obj`
        is of type str or unicode, then the `obj` string will be
        chunked into `itemsize` pieces.

    unicode : bool, optional
        When true, the resulting `chararray` can contain Unicode
        characters, when false only 8-bit characters.  If unicode is
        None and `obj` is one of the following:

          - a `chararray`,
          - an ndarray of type `str` or 'unicode`
          - a Python str or unicode object,

        then the unicode setting of the output array will be
        automatically determined.

    order : {'C', 'F'}, optional
        Specify the order of the array.  If order is 'C' (default), then the
        array will be in C-contiguous order (last-index varies the
        fastest).  If order is 'F', then the returned array
        will be in Fortran-contiguous order (first-index varies the
        fastest).
    """
    return array(obj, itemsize, copy=False,
                 unicode=unicode, order=order)
