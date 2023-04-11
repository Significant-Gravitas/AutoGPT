"""Implementation of __array_function__ overrides from NEP-18."""
import collections
import functools
import os

from numpy.core._multiarray_umath import (
    add_docstring, implement_array_function, _get_implementing_args)
from numpy.compat._inspect import getargspec


ARRAY_FUNCTION_ENABLED = bool(
    int(os.environ.get('NUMPY_EXPERIMENTAL_ARRAY_FUNCTION', 1)))

array_function_like_doc = (
    """like : array_like, optional
        Reference object to allow the creation of arrays which are not
        NumPy arrays. If an array-like passed in as ``like`` supports
        the ``__array_function__`` protocol, the result will be defined
        by it. In this case, it ensures the creation of an array object
        compatible with that passed in via this argument."""
)

def set_array_function_like_doc(public_api):
    if public_api.__doc__ is not None:
        public_api.__doc__ = public_api.__doc__.replace(
            "${ARRAY_FUNCTION_LIKE}",
            array_function_like_doc,
        )
    return public_api


add_docstring(
    implement_array_function,
    """
    Implement a function with checks for __array_function__ overrides.

    All arguments are required, and can only be passed by position.

    Parameters
    ----------
    implementation : function
        Function that implements the operation on NumPy array without
        overrides when called like ``implementation(*args, **kwargs)``.
    public_api : function
        Function exposed by NumPy's public API originally called like
        ``public_api(*args, **kwargs)`` on which arguments are now being
        checked.
    relevant_args : iterable
        Iterable of arguments to check for __array_function__ methods.
    args : tuple
        Arbitrary positional arguments originally passed into ``public_api``.
    kwargs : dict
        Arbitrary keyword arguments originally passed into ``public_api``.

    Returns
    -------
    Result from calling ``implementation()`` or an ``__array_function__``
    method, as appropriate.

    Raises
    ------
    TypeError : if no implementation is found.
    """)


# exposed for testing purposes; used internally by implement_array_function
add_docstring(
    _get_implementing_args,
    """
    Collect arguments on which to call __array_function__.

    Parameters
    ----------
    relevant_args : iterable of array-like
        Iterable of possibly array-like arguments to check for
        __array_function__ methods.

    Returns
    -------
    Sequence of arguments with __array_function__ methods, in the order in
    which they should be called.
    """)


ArgSpec = collections.namedtuple('ArgSpec', 'args varargs keywords defaults')


def verify_matching_signatures(implementation, dispatcher):
    """Verify that a dispatcher function has the right signature."""
    implementation_spec = ArgSpec(*getargspec(implementation))
    dispatcher_spec = ArgSpec(*getargspec(dispatcher))

    if (implementation_spec.args != dispatcher_spec.args or
            implementation_spec.varargs != dispatcher_spec.varargs or
            implementation_spec.keywords != dispatcher_spec.keywords or
            (bool(implementation_spec.defaults) !=
             bool(dispatcher_spec.defaults)) or
            (implementation_spec.defaults is not None and
             len(implementation_spec.defaults) !=
             len(dispatcher_spec.defaults))):
        raise RuntimeError('implementation and dispatcher for %s have '
                           'different function signatures' % implementation)

    if implementation_spec.defaults is not None:
        if dispatcher_spec.defaults != (None,) * len(dispatcher_spec.defaults):
            raise RuntimeError('dispatcher functions can only use None for '
                               'default argument values')


def set_module(module):
    """Decorator for overriding __module__ on a function or class.

    Example usage::

        @set_module('numpy')
        def example():
            pass

        assert example.__module__ == 'numpy'
    """
    def decorator(func):
        if module is not None:
            func.__module__ = module
        return func
    return decorator


def array_function_dispatch(dispatcher, module=None, verify=True,
                            docs_from_dispatcher=False):
    """Decorator for adding dispatch with the __array_function__ protocol.

    See NEP-18 for example usage.

    Parameters
    ----------
    dispatcher : callable
        Function that when called like ``dispatcher(*args, **kwargs)`` with
        arguments from the NumPy function call returns an iterable of
        array-like arguments to check for ``__array_function__``.
    module : str, optional
        __module__ attribute to set on new function, e.g., ``module='numpy'``.
        By default, module is copied from the decorated function.
    verify : bool, optional
        If True, verify the that the signature of the dispatcher and decorated
        function signatures match exactly: all required and optional arguments
        should appear in order with the same names, but the default values for
        all optional arguments should be ``None``. Only disable verification
        if the dispatcher's signature needs to deviate for some particular
        reason, e.g., because the function has a signature like
        ``func(*args, **kwargs)``.
    docs_from_dispatcher : bool, optional
        If True, copy docs from the dispatcher function onto the dispatched
        function, rather than from the implementation. This is useful for
        functions defined in C, which otherwise don't have docstrings.

    Returns
    -------
    Function suitable for decorating the implementation of a NumPy function.
    """

    if not ARRAY_FUNCTION_ENABLED:
        def decorator(implementation):
            if docs_from_dispatcher:
                add_docstring(implementation, dispatcher.__doc__)
            if module is not None:
                implementation.__module__ = module
            return implementation
        return decorator

    def decorator(implementation):
        if verify:
            verify_matching_signatures(implementation, dispatcher)

        if docs_from_dispatcher:
            add_docstring(implementation, dispatcher.__doc__)

        @functools.wraps(implementation)
        def public_api(*args, **kwargs):
            try:
                relevant_args = dispatcher(*args, **kwargs)
            except TypeError as exc:
                # Try to clean up a signature related TypeError.  Such an
                # error will be something like:
                #     dispatcher.__name__() got an unexpected keyword argument
                #
                # So replace the dispatcher name in this case.  In principle
                # TypeErrors may be raised from _within_ the dispatcher, so
                # we check that the traceback contains a string that starts
                # with the name.  (In principle we could also check the
                # traceback length, as it would be deeper.)
                msg = exc.args[0]
                disp_name = dispatcher.__name__
                if not isinstance(msg, str) or not msg.startswith(disp_name):
                    raise

                # Replace with the correct name and re-raise:
                new_msg = msg.replace(disp_name, public_api.__name__)
                raise TypeError(new_msg) from None

            return implement_array_function(
                implementation, public_api, relevant_args, args, kwargs)

        public_api.__code__ = public_api.__code__.replace(
                co_name=implementation.__name__,
                co_filename='<__array_function__ internals>')
        if module is not None:
            public_api.__module__ = module

        public_api._implementation = implementation

        return public_api

    return decorator


def array_function_from_dispatcher(
        implementation, module=None, verify=True, docs_from_dispatcher=True):
    """Like array_function_dispatcher, but with function arguments flipped."""

    def decorator(dispatcher):
        return array_function_dispatch(
            dispatcher, module, verify=verify,
            docs_from_dispatcher=docs_from_dispatcher)(implementation)
    return decorator
