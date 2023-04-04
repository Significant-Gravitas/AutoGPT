# SPDX-License-Identifier: MIT


import inspect
import platform
import sys
import threading
import types
import warnings

from collections.abc import Mapping, Sequence  # noqa


PYPY = platform.python_implementation() == "PyPy"
PY310 = sys.version_info[:2] >= (3, 10)
PY_3_12_PLUS = sys.version_info[:2] >= (3, 12)


def just_warn(*args, **kw):
    warnings.warn(
        "Running interpreter doesn't sufficiently support code object "
        "introspection.  Some features like bare super() or accessing "
        "__class__ will not work with slotted classes.",
        RuntimeWarning,
        stacklevel=2,
    )


class _AnnotationExtractor:
    """
    Extract type annotations from a callable, returning None whenever there
    is none.
    """

    __slots__ = ["sig"]

    def __init__(self, callable):
        try:
            self.sig = inspect.signature(callable)
        except (ValueError, TypeError):  # inspect failed
            self.sig = None

    def get_first_param_type(self):
        """
        Return the type annotation of the first argument if it's not empty.
        """
        if not self.sig:
            return None

        params = list(self.sig.parameters.values())
        if params and params[0].annotation is not inspect.Parameter.empty:
            return params[0].annotation

        return None

    def get_return_type(self):
        """
        Return the return type if it's not empty.
        """
        if (
            self.sig
            and self.sig.return_annotation is not inspect.Signature.empty
        ):
            return self.sig.return_annotation

        return None


def make_set_closure_cell():
    """Return a function of two arguments (cell, value) which sets
    the value stored in the closure cell `cell` to `value`.
    """
    # pypy makes this easy. (It also supports the logic below, but
    # why not do the easy/fast thing?)
    if PYPY:

        def set_closure_cell(cell, value):
            cell.__setstate__((value,))

        return set_closure_cell

    # Otherwise gotta do it the hard way.

    # Create a function that will set its first cellvar to `value`.
    def set_first_cellvar_to(value):
        x = value
        return

        # This function will be eliminated as dead code, but
        # not before its reference to `x` forces `x` to be
        # represented as a closure cell rather than a local.
        def force_x_to_be_a_cell():  # pragma: no cover
            return x

    try:
        # Extract the code object and make sure our assumptions about
        # the closure behavior are correct.
        co = set_first_cellvar_to.__code__
        if co.co_cellvars != ("x",) or co.co_freevars != ():
            raise AssertionError  # pragma: no cover

        # Convert this code object to a code object that sets the
        # function's first _freevar_ (not cellvar) to the argument.
        if sys.version_info >= (3, 8):

            def set_closure_cell(cell, value):
                cell.cell_contents = value

        else:
            args = [co.co_argcount]
            args.append(co.co_kwonlyargcount)
            args.extend(
                [
                    co.co_nlocals,
                    co.co_stacksize,
                    co.co_flags,
                    co.co_code,
                    co.co_consts,
                    co.co_names,
                    co.co_varnames,
                    co.co_filename,
                    co.co_name,
                    co.co_firstlineno,
                    co.co_lnotab,
                    # These two arguments are reversed:
                    co.co_cellvars,
                    co.co_freevars,
                ]
            )
            set_first_freevar_code = types.CodeType(*args)

            def set_closure_cell(cell, value):
                # Create a function using the set_first_freevar_code,
                # whose first closure cell is `cell`. Calling it will
                # change the value of that cell.
                setter = types.FunctionType(
                    set_first_freevar_code, {}, "setter", (), (cell,)
                )
                # And call it to set the cell.
                setter(value)

        # Make sure it works on this interpreter:
        def make_func_with_cell():
            x = None

            def func():
                return x  # pragma: no cover

            return func

        cell = make_func_with_cell().__closure__[0]
        set_closure_cell(cell, 100)
        if cell.cell_contents != 100:
            raise AssertionError  # pragma: no cover

    except Exception:
        return just_warn
    else:
        return set_closure_cell


set_closure_cell = make_set_closure_cell()

# Thread-local global to track attrs instances which are already being repr'd.
# This is needed because there is no other (thread-safe) way to pass info
# about the instances that are already being repr'd through the call stack
# in order to ensure we don't perform infinite recursion.
#
# For instance, if an instance contains a dict which contains that instance,
# we need to know that we're already repr'ing the outside instance from within
# the dict's repr() call.
#
# This lives here rather than in _make.py so that the functions in _make.py
# don't have a direct reference to the thread-local in their globals dict.
# If they have such a reference, it breaks cloudpickle.
repr_context = threading.local()
