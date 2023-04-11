"""
tl;dr: all code is licensed under simplified BSD, unless stated otherwise.

Unless stated otherwise in the source files, all code is copyright 2010 David
Wolever <david@wolever.net>. All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

   1. Redistributions of source code must retain the above copyright notice,
   this list of conditions and the following disclaimer.

   2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY <COPYRIGHT HOLDER> ``AS IS'' AND ANY EXPRESS OR
IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF
MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO
EVENT SHALL <COPYRIGHT HOLDER> OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT,
INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE,
DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF
ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

The views and conclusions contained in the software and documentation are those
of the authors and should not be interpreted as representing official policies,
either expressed or implied, of David Wolever.

"""
import re
import inspect
import warnings
from functools import wraps
from types import MethodType
from collections import namedtuple

from unittest import TestCase

_param = namedtuple("param", "args kwargs")

class param(_param):
    """ Represents a single parameter to a test case.

        For example::

            >>> p = param("foo", bar=16)
            >>> p
            param("foo", bar=16)
            >>> p.args
            ('foo', )
            >>> p.kwargs
            {'bar': 16}

        Intended to be used as an argument to ``@parameterized``::

            @parameterized([
                param("foo", bar=16),
            ])
            def test_stuff(foo, bar=16):
                pass
        """

    def __new__(cls, *args , **kwargs):
        return _param.__new__(cls, args, kwargs)

    @classmethod
    def explicit(cls, args=None, kwargs=None):
        """ Creates a ``param`` by explicitly specifying ``args`` and
            ``kwargs``::

                >>> param.explicit([1,2,3])
                param(*(1, 2, 3))
                >>> param.explicit(kwargs={"foo": 42})
                param(*(), **{"foo": "42"})
            """
        args = args or ()
        kwargs = kwargs or {}
        return cls(*args, **kwargs)

    @classmethod
    def from_decorator(cls, args):
        """ Returns an instance of ``param()`` for ``@parameterized`` argument
            ``args``::

                >>> param.from_decorator((42, ))
                param(args=(42, ), kwargs={})
                >>> param.from_decorator("foo")
                param(args=("foo", ), kwargs={})
            """
        if isinstance(args, param):
            return args
        elif isinstance(args, (str,)):
            args = (args, )
        try:
            return cls(*args)
        except TypeError as e:
            if "after * must be" not in str(e):
                raise
            raise TypeError(
                "Parameters must be tuples, but %r is not (hint: use '(%r, )')"
                %(args, args),
            )

    def __repr__(self):
        return "param(*%r, **%r)" %self


def parameterized_argument_value_pairs(func, p):
    """Return tuples of parameterized arguments and their values.

        This is useful if you are writing your own doc_func
        function and need to know the values for each parameter name::

            >>> def func(a, foo=None, bar=42, **kwargs): pass
            >>> p = param(1, foo=7, extra=99)
            >>> parameterized_argument_value_pairs(func, p)
            [("a", 1), ("foo", 7), ("bar", 42), ("**kwargs", {"extra": 99})]

        If the function's first argument is named ``self`` then it will be
        ignored::

            >>> def func(self, a): pass
            >>> p = param(1)
            >>> parameterized_argument_value_pairs(func, p)
            [("a", 1)]

        Additionally, empty ``*args`` or ``**kwargs`` will be ignored::

            >>> def func(foo, *args): pass
            >>> p = param(1)
            >>> parameterized_argument_value_pairs(func, p)
            [("foo", 1)]
            >>> p = param(1, 16)
            >>> parameterized_argument_value_pairs(func, p)
            [("foo", 1), ("*args", (16, ))]
    """
    argspec = inspect.getargspec(func)
    arg_offset = 1 if argspec.args[:1] == ["self"] else 0

    named_args = argspec.args[arg_offset:]

    result = list(zip(named_args, p.args))
    named_args = argspec.args[len(result) + arg_offset:]
    varargs = p.args[len(result):]

    result.extend([
        (name, p.kwargs.get(name, default))
        for (name, default)
        in zip(named_args, argspec.defaults or [])
    ])

    seen_arg_names = {n for (n, _) in result}
    keywords = dict(sorted([
        (name, p.kwargs[name])
        for name in p.kwargs
        if name not in seen_arg_names
    ]))

    if varargs:
        result.append(("*%s" %(argspec.varargs, ), tuple(varargs)))

    if keywords:
        result.append(("**%s" %(argspec.keywords, ), keywords))

    return result

def short_repr(x, n=64):
    """ A shortened repr of ``x`` which is guaranteed to be ``unicode``::

            >>> short_repr("foo")
            u"foo"
            >>> short_repr("123456789", n=4)
            u"12...89"
    """

    x_repr = repr(x)
    if isinstance(x_repr, bytes):
        try:
            x_repr = str(x_repr, "utf-8")
        except UnicodeDecodeError:
            x_repr = str(x_repr, "latin1")
    if len(x_repr) > n:
        x_repr = x_repr[:n//2] + "..." + x_repr[len(x_repr) - n//2:]
    return x_repr

def default_doc_func(func, num, p):
    if func.__doc__ is None:
        return None

    all_args_with_values = parameterized_argument_value_pairs(func, p)

    # Assumes that the function passed is a bound method.
    descs = [f'{n}={short_repr(v)}' for n, v in all_args_with_values]

    # The documentation might be a multiline string, so split it
    # and just work with the first string, ignoring the period
    # at the end if there is one.
    first, nl, rest = func.__doc__.lstrip().partition("\n")
    suffix = ""
    if first.endswith("."):
        suffix = "."
        first = first[:-1]
    args = "%s[with %s]" %(len(first) and " " or "", ", ".join(descs))
    return "".join([first.rstrip(), args, suffix, nl, rest])

def default_name_func(func, num, p):
    base_name = func.__name__
    name_suffix = "_%s" %(num, )
    if len(p.args) > 0 and isinstance(p.args[0], (str,)):
        name_suffix += "_" + parameterized.to_safe_name(p.args[0])
    return base_name + name_suffix


# force nose for numpy purposes.
_test_runner_override = 'nose'
_test_runner_guess = False
_test_runners = set(["unittest", "unittest2", "nose", "nose2", "pytest"])
_test_runner_aliases = {
    "_pytest": "pytest",
}

def set_test_runner(name):
    global _test_runner_override
    if name not in _test_runners:
        raise TypeError(
            "Invalid test runner: %r (must be one of: %s)"
            %(name, ", ".join(_test_runners)),
        )
    _test_runner_override = name

def detect_runner():
    """ Guess which test runner we're using by traversing the stack and looking
        for the first matching module. This *should* be reasonably safe, as
        it's done during test discovery where the test runner should be the
        stack frame immediately outside. """
    if _test_runner_override is not None:
        return _test_runner_override
    global _test_runner_guess
    if _test_runner_guess is False:
        stack = inspect.stack()
        for record in reversed(stack):
            frame = record[0]
            module = frame.f_globals.get("__name__").partition(".")[0]
            if module in _test_runner_aliases:
                module = _test_runner_aliases[module]
            if module in _test_runners:
                _test_runner_guess = module
                break
        else:
            _test_runner_guess = None
    return _test_runner_guess

class parameterized:
    """ Parameterize a test case::

            class TestInt:
                @parameterized([
                    ("A", 10),
                    ("F", 15),
                    param("10", 42, base=42)
                ])
                def test_int(self, input, expected, base=16):
                    actual = int(input, base=base)
                    assert_equal(actual, expected)

            @parameterized([
                (2, 3, 5)
                (3, 5, 8),
            ])
            def test_add(a, b, expected):
                assert_equal(a + b, expected)
        """

    def __init__(self, input, doc_func=None):
        self.get_input = self.input_as_callable(input)
        self.doc_func = doc_func or default_doc_func

    def __call__(self, test_func):
        self.assert_not_in_testcase_subclass()

        @wraps(test_func)
        def wrapper(test_self=None):
            test_cls = test_self and type(test_self)

            original_doc = wrapper.__doc__
            for num, args in enumerate(wrapper.parameterized_input):
                p = param.from_decorator(args)
                unbound_func, nose_tuple = self.param_as_nose_tuple(test_self, test_func, num, p)
                try:
                    wrapper.__doc__ = nose_tuple[0].__doc__
                    # Nose uses `getattr(instance, test_func.__name__)` to get
                    # a method bound to the test instance (as opposed to a
                    # method bound to the instance of the class created when
                    # tests were being enumerated). Set a value here to make
                    # sure nose can get the correct test method.
                    if test_self is not None:
                        setattr(test_cls, test_func.__name__, unbound_func)
                    yield nose_tuple
                finally:
                    if test_self is not None:
                        delattr(test_cls, test_func.__name__)
                    wrapper.__doc__ = original_doc
        wrapper.parameterized_input = self.get_input()
        wrapper.parameterized_func = test_func
        test_func.__name__ = "_parameterized_original_%s" %(test_func.__name__, )
        return wrapper

    def param_as_nose_tuple(self, test_self, func, num, p):
        nose_func = wraps(func)(lambda *args: func(*args[:-1], **args[-1]))
        nose_func.__doc__ = self.doc_func(func, num, p)
        # Track the unbound function because we need to setattr the unbound
        # function onto the class for nose to work (see comments above), and
        # Python 3 doesn't let us pull the function out of a bound method.
        unbound_func = nose_func
        if test_self is not None:
            nose_func = MethodType(nose_func, test_self)
        return unbound_func, (nose_func, ) + p.args + (p.kwargs or {}, )

    def assert_not_in_testcase_subclass(self):
        parent_classes = self._terrible_magic_get_defining_classes()
        if any(issubclass(cls, TestCase) for cls in parent_classes):
            raise Exception("Warning: '@parameterized' tests won't work "
                            "inside subclasses of 'TestCase' - use "
                            "'@parameterized.expand' instead.")

    def _terrible_magic_get_defining_classes(self):
        """ Returns the list of parent classes of the class currently being defined.
            Will likely only work if called from the ``parameterized`` decorator.
            This function is entirely @brandon_rhodes's fault, as he suggested
            the implementation: http://stackoverflow.com/a/8793684/71522
            """
        stack = inspect.stack()
        if len(stack) <= 4:
            return []
        frame = stack[4]
        code_context = frame[4] and frame[4][0].strip()
        if not (code_context and code_context.startswith("class ")):
            return []
        _, _, parents = code_context.partition("(")
        parents, _, _ = parents.partition(")")
        return eval("[" + parents + "]", frame[0].f_globals, frame[0].f_locals)

    @classmethod
    def input_as_callable(cls, input):
        if callable(input):
            return lambda: cls.check_input_values(input())
        input_values = cls.check_input_values(input)
        return lambda: input_values

    @classmethod
    def check_input_values(cls, input_values):
        # Explicitly convert non-list inputs to a list so that:
        # 1. A helpful exception will be raised if they aren't iterable, and
        # 2. Generators are unwrapped exactly once (otherwise `nosetests
        #    --processes=n` has issues; see:
        #    https://github.com/wolever/nose-parameterized/pull/31)
        if not isinstance(input_values, list):
            input_values = list(input_values)
        return [ param.from_decorator(p) for p in input_values ]

    @classmethod
    def expand(cls, input, name_func=None, doc_func=None, **legacy):
        """ A "brute force" method of parameterizing test cases. Creates new
            test cases and injects them into the namespace that the wrapped
            function is being defined in. Useful for parameterizing tests in
            subclasses of 'UnitTest', where Nose test generators don't work.

            >>> @parameterized.expand([("foo", 1, 2)])
            ... def test_add1(name, input, expected):
            ...     actual = add1(input)
            ...     assert_equal(actual, expected)
            ...
            >>> locals()
            ... 'test_add1_foo_0': <function ...> ...
            >>>
            """

        if "testcase_func_name" in legacy:
            warnings.warn("testcase_func_name= is deprecated; use name_func=",
                          DeprecationWarning, stacklevel=2)
            if not name_func:
                name_func = legacy["testcase_func_name"]

        if "testcase_func_doc" in legacy:
            warnings.warn("testcase_func_doc= is deprecated; use doc_func=",
                          DeprecationWarning, stacklevel=2)
            if not doc_func:
                doc_func = legacy["testcase_func_doc"]

        doc_func = doc_func or default_doc_func
        name_func = name_func or default_name_func

        def parameterized_expand_wrapper(f, instance=None):
            stack = inspect.stack()
            frame = stack[1]
            frame_locals = frame[0].f_locals

            parameters = cls.input_as_callable(input)()
            for num, p in enumerate(parameters):
                name = name_func(f, num, p)
                frame_locals[name] = cls.param_as_standalone_func(p, f, name)
                frame_locals[name].__doc__ = doc_func(f, num, p)

            f.__test__ = False
        return parameterized_expand_wrapper

    @classmethod
    def param_as_standalone_func(cls, p, func, name):
        @wraps(func)
        def standalone_func(*a):
            return func(*(a + p.args), **p.kwargs)
        standalone_func.__name__ = name

        # place_as is used by py.test to determine what source file should be
        # used for this test.
        standalone_func.place_as = func

        # Remove __wrapped__ because py.test will try to look at __wrapped__
        # to determine which parameters should be used with this test case,
        # and obviously we don't need it to do any parameterization.
        try:
            del standalone_func.__wrapped__
        except AttributeError:
            pass
        return standalone_func

    @classmethod
    def to_safe_name(cls, s):
        return str(re.sub("[^a-zA-Z0-9_]+", "_", s))
