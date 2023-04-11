"""
Decorators for labeling and modifying behavior of test objects.

Decorators that merely return a modified version of the original
function object are straightforward. Decorators that return a new
function object need to use
::

  nose.tools.make_decorator(original_function)(decorator)

in returning the decorator, in order to preserve meta-data such as
function name, setup and teardown functions and so on - see
``nose.tools`` for more information.

"""
import collections.abc
import warnings

from .utils import SkipTest, assert_warns, HAS_REFCOUNT

__all__ = ['slow', 'setastest', 'skipif', 'knownfailureif', 'deprecated',
           'parametrize', '_needs_refcount',]


def slow(t):
    """
    .. deprecated:: 1.21
        This decorator is retained for compatibility with the nose testing framework, which is being phased out.
        Please use the nose2 or pytest frameworks instead.

    Label a test as 'slow'.

    The exact definition of a slow test is obviously both subjective and
    hardware-dependent, but in general any individual test that requires more
    than a second or two should be labeled as slow (the whole suite consists of
    thousands of tests, so even a second is significant).

    Parameters
    ----------
    t : callable
        The test to label as slow.

    Returns
    -------
    t : callable
        The decorated test `t`.

    Examples
    --------
    The `numpy.testing` module includes ``import decorators as dec``.
    A test can be decorated as slow like this::

      from numpy.testing import *

      @dec.slow
      def test_big(self):
          print('Big, slow test')

    """
    # Numpy 1.21, 2020-12-20
    warnings.warn('the np.testing.dec decorators are included for nose support, and are '
                'deprecated since NumPy v1.21. Use the nose2 or pytest frameworks instead.', DeprecationWarning, stacklevel=2)

    t.slow = True
    return t

def setastest(tf=True):
    """
    .. deprecated:: 1.21
        This decorator is retained for compatibility with the nose testing framework, which is being phased out.
        Please use the nose2 or pytest frameworks instead.

    Signals to nose that this function is or is not a test.

    Parameters
    ----------
    tf : bool
        If True, specifies that the decorated callable is a test.
        If False, specifies that the decorated callable is not a test.
        Default is True.

    Notes
    -----
    This decorator can't use the nose namespace, because it can be
    called from a non-test module. See also ``istest`` and ``nottest`` in
    ``nose.tools``.

    Examples
    --------
    `setastest` can be used in the following way::

      from numpy.testing import dec

      @dec.setastest(False)
      def func_with_test_in_name(arg1, arg2):
          pass

    """
    # Numpy 1.21, 2020-12-20
    warnings.warn('the np.testing.dec decorators are included for nose support, and are '
            'deprecated since NumPy v1.21. Use the nose2 or pytest frameworks instead.', DeprecationWarning, stacklevel=2)
    def set_test(t):
        t.__test__ = tf
        return t
    return set_test

def skipif(skip_condition, msg=None):
    """
    .. deprecated:: 1.21
        This decorator is retained for compatibility with the nose testing framework, which is being phased out.
        Please use the nose2 or pytest frameworks instead.

    Make function raise SkipTest exception if a given condition is true.

    If the condition is a callable, it is used at runtime to dynamically
    make the decision. This is useful for tests that may require costly
    imports, to delay the cost until the test suite is actually executed.

    Parameters
    ----------
    skip_condition : bool or callable
        Flag to determine whether to skip the decorated test.
    msg : str, optional
        Message to give on raising a SkipTest exception. Default is None.

    Returns
    -------
    decorator : function
        Decorator which, when applied to a function, causes SkipTest
        to be raised when `skip_condition` is True, and the function
        to be called normally otherwise.

    Notes
    -----
    The decorator itself is decorated with the ``nose.tools.make_decorator``
    function in order to transmit function name, and various other metadata.

    """

    def skip_decorator(f):
        # Local import to avoid a hard nose dependency and only incur the
        # import time overhead at actual test-time.
        import nose

        # Numpy 1.21, 2020-12-20
        warnings.warn('the np.testing.dec decorators are included for nose support, and are '
            'deprecated since NumPy v1.21. Use the nose2 or pytest frameworks instead.', DeprecationWarning, stacklevel=2)

        # Allow for both boolean or callable skip conditions.
        if isinstance(skip_condition, collections.abc.Callable):
            skip_val = lambda: skip_condition()
        else:
            skip_val = lambda: skip_condition

        def get_msg(func,msg=None):
            """Skip message with information about function being skipped."""
            if msg is None:
                out = 'Test skipped due to test condition'
            else:
                out = msg

            return f'Skipping test: {func.__name__}: {out}'

        # We need to define *two* skippers because Python doesn't allow both
        # return with value and yield inside the same function.
        def skipper_func(*args, **kwargs):
            """Skipper for normal test functions."""
            if skip_val():
                raise SkipTest(get_msg(f, msg))
            else:
                return f(*args, **kwargs)

        def skipper_gen(*args, **kwargs):
            """Skipper for test generators."""
            if skip_val():
                raise SkipTest(get_msg(f, msg))
            else:
                yield from f(*args, **kwargs)

        # Choose the right skipper to use when building the actual decorator.
        if nose.util.isgenerator(f):
            skipper = skipper_gen
        else:
            skipper = skipper_func

        return nose.tools.make_decorator(f)(skipper)

    return skip_decorator


def knownfailureif(fail_condition, msg=None):
    """
    .. deprecated:: 1.21
        This decorator is retained for compatibility with the nose testing framework, which is being phased out.
        Please use the nose2 or pytest frameworks instead.

    Make function raise KnownFailureException exception if given condition is true.

    If the condition is a callable, it is used at runtime to dynamically
    make the decision. This is useful for tests that may require costly
    imports, to delay the cost until the test suite is actually executed.

    Parameters
    ----------
    fail_condition : bool or callable
        Flag to determine whether to mark the decorated test as a known
        failure (if True) or not (if False).
    msg : str, optional
        Message to give on raising a KnownFailureException exception.
        Default is None.

    Returns
    -------
    decorator : function
        Decorator, which, when applied to a function, causes
        KnownFailureException to be raised when `fail_condition` is True,
        and the function to be called normally otherwise.

    Notes
    -----
    The decorator itself is decorated with the ``nose.tools.make_decorator``
    function in order to transmit function name, and various other metadata.

    """
    # Numpy 1.21, 2020-12-20
    warnings.warn('the np.testing.dec decorators are included for nose support, and are '
            'deprecated since NumPy v1.21. Use the nose2 or pytest frameworks instead.', DeprecationWarning, stacklevel=2)

    if msg is None:
        msg = 'Test skipped due to known failure'

    # Allow for both boolean or callable known failure conditions.
    if isinstance(fail_condition, collections.abc.Callable):
        fail_val = lambda: fail_condition()
    else:
        fail_val = lambda: fail_condition

    def knownfail_decorator(f):
        # Local import to avoid a hard nose dependency and only incur the
        # import time overhead at actual test-time.
        import nose
        from .noseclasses import KnownFailureException

        def knownfailer(*args, **kwargs):
            if fail_val():
                raise KnownFailureException(msg)
            else:
                return f(*args, **kwargs)
        return nose.tools.make_decorator(f)(knownfailer)

    return knownfail_decorator

def deprecated(conditional=True):
    """
    .. deprecated:: 1.21
        This decorator is retained for compatibility with the nose testing framework, which is being phased out.
        Please use the nose2 or pytest frameworks instead.

    Filter deprecation warnings while running the test suite.

    This decorator can be used to filter DeprecationWarning's, to avoid
    printing them during the test suite run, while checking that the test
    actually raises a DeprecationWarning.

    Parameters
    ----------
    conditional : bool or callable, optional
        Flag to determine whether to mark test as deprecated or not. If the
        condition is a callable, it is used at runtime to dynamically make the
        decision. Default is True.

    Returns
    -------
    decorator : function
        The `deprecated` decorator itself.

    Notes
    -----
    .. versionadded:: 1.4.0

    """
    def deprecate_decorator(f):
        # Local import to avoid a hard nose dependency and only incur the
        # import time overhead at actual test-time.
        import nose

        # Numpy 1.21, 2020-12-20
        warnings.warn('the np.testing.dec decorators are included for nose support, and are '
            'deprecated since NumPy v1.21. Use the nose2 or pytest frameworks instead.', DeprecationWarning, stacklevel=2)

        def _deprecated_imp(*args, **kwargs):
            # Poor man's replacement for the with statement
            with assert_warns(DeprecationWarning):
                f(*args, **kwargs)

        if isinstance(conditional, collections.abc.Callable):
            cond = conditional()
        else:
            cond = conditional
        if cond:
            return nose.tools.make_decorator(f)(_deprecated_imp)
        else:
            return f
    return deprecate_decorator


def parametrize(vars, input):
    """
    .. deprecated:: 1.21
        This decorator is retained for compatibility with the nose testing framework, which is being phased out.
        Please use the nose2 or pytest frameworks instead.

    Pytest compatibility class. This implements the simplest level of
    pytest.mark.parametrize for use in nose as an aid in making the transition
    to pytest. It achieves that by adding a dummy var parameter and ignoring
    the doc_func parameter of the base class. It does not support variable
    substitution by name, nor does it support nesting or classes. See the
    pytest documentation for usage.

    .. versionadded:: 1.14.0

    """
    from .parameterized import parameterized

    # Numpy 1.21, 2020-12-20
    warnings.warn('the np.testing.dec decorators are included for nose support, and are '
            'deprecated since NumPy v1.21. Use the nose2 or pytest frameworks instead.', DeprecationWarning, stacklevel=2)

    return parameterized(input)

_needs_refcount = skipif(not HAS_REFCOUNT, "python has no sys.getrefcount")
