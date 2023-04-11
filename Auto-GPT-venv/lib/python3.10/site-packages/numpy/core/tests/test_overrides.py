import inspect
import sys
import os
import tempfile
from io import StringIO
from unittest import mock

import numpy as np
from numpy.testing import (
    assert_, assert_equal, assert_raises, assert_raises_regex)
from numpy.core.overrides import (
    _get_implementing_args, array_function_dispatch,
    verify_matching_signatures, ARRAY_FUNCTION_ENABLED)
from numpy.compat import pickle
import pytest


requires_array_function = pytest.mark.skipif(
    not ARRAY_FUNCTION_ENABLED,
    reason="__array_function__ dispatch not enabled.")


def _return_not_implemented(self, *args, **kwargs):
    return NotImplemented


# need to define this at the top level to test pickling
@array_function_dispatch(lambda array: (array,))
def dispatched_one_arg(array):
    """Docstring."""
    return 'original'


@array_function_dispatch(lambda array1, array2: (array1, array2))
def dispatched_two_arg(array1, array2):
    """Docstring."""
    return 'original'


class TestGetImplementingArgs:

    def test_ndarray(self):
        array = np.array(1)

        args = _get_implementing_args([array])
        assert_equal(list(args), [array])

        args = _get_implementing_args([array, array])
        assert_equal(list(args), [array])

        args = _get_implementing_args([array, 1])
        assert_equal(list(args), [array])

        args = _get_implementing_args([1, array])
        assert_equal(list(args), [array])

    def test_ndarray_subclasses(self):

        class OverrideSub(np.ndarray):
            __array_function__ = _return_not_implemented

        class NoOverrideSub(np.ndarray):
            pass

        array = np.array(1).view(np.ndarray)
        override_sub = np.array(1).view(OverrideSub)
        no_override_sub = np.array(1).view(NoOverrideSub)

        args = _get_implementing_args([array, override_sub])
        assert_equal(list(args), [override_sub, array])

        args = _get_implementing_args([array, no_override_sub])
        assert_equal(list(args), [no_override_sub, array])

        args = _get_implementing_args(
            [override_sub, no_override_sub])
        assert_equal(list(args), [override_sub, no_override_sub])

    def test_ndarray_and_duck_array(self):

        class Other:
            __array_function__ = _return_not_implemented

        array = np.array(1)
        other = Other()

        args = _get_implementing_args([other, array])
        assert_equal(list(args), [other, array])

        args = _get_implementing_args([array, other])
        assert_equal(list(args), [array, other])

    def test_ndarray_subclass_and_duck_array(self):

        class OverrideSub(np.ndarray):
            __array_function__ = _return_not_implemented

        class Other:
            __array_function__ = _return_not_implemented

        array = np.array(1)
        subarray = np.array(1).view(OverrideSub)
        other = Other()

        assert_equal(_get_implementing_args([array, subarray, other]),
                     [subarray, array, other])
        assert_equal(_get_implementing_args([array, other, subarray]),
                     [subarray, array, other])

    def test_many_duck_arrays(self):

        class A:
            __array_function__ = _return_not_implemented

        class B(A):
            __array_function__ = _return_not_implemented

        class C(A):
            __array_function__ = _return_not_implemented

        class D:
            __array_function__ = _return_not_implemented

        a = A()
        b = B()
        c = C()
        d = D()

        assert_equal(_get_implementing_args([1]), [])
        assert_equal(_get_implementing_args([a]), [a])
        assert_equal(_get_implementing_args([a, 1]), [a])
        assert_equal(_get_implementing_args([a, a, a]), [a])
        assert_equal(_get_implementing_args([a, d, a]), [a, d])
        assert_equal(_get_implementing_args([a, b]), [b, a])
        assert_equal(_get_implementing_args([b, a]), [b, a])
        assert_equal(_get_implementing_args([a, b, c]), [b, c, a])
        assert_equal(_get_implementing_args([a, c, b]), [c, b, a])

    def test_too_many_duck_arrays(self):
        namespace = dict(__array_function__=_return_not_implemented)
        types = [type('A' + str(i), (object,), namespace) for i in range(33)]
        relevant_args = [t() for t in types]

        actual = _get_implementing_args(relevant_args[:32])
        assert_equal(actual, relevant_args[:32])

        with assert_raises_regex(TypeError, 'distinct argument types'):
            _get_implementing_args(relevant_args)


class TestNDArrayArrayFunction:

    @requires_array_function
    def test_method(self):

        class Other:
            __array_function__ = _return_not_implemented

        class NoOverrideSub(np.ndarray):
            pass

        class OverrideSub(np.ndarray):
            __array_function__ = _return_not_implemented

        array = np.array([1])
        other = Other()
        no_override_sub = array.view(NoOverrideSub)
        override_sub = array.view(OverrideSub)

        result = array.__array_function__(func=dispatched_two_arg,
                                          types=(np.ndarray,),
                                          args=(array, 1.), kwargs={})
        assert_equal(result, 'original')

        result = array.__array_function__(func=dispatched_two_arg,
                                          types=(np.ndarray, Other),
                                          args=(array, other), kwargs={})
        assert_(result is NotImplemented)

        result = array.__array_function__(func=dispatched_two_arg,
                                          types=(np.ndarray, NoOverrideSub),
                                          args=(array, no_override_sub),
                                          kwargs={})
        assert_equal(result, 'original')

        result = array.__array_function__(func=dispatched_two_arg,
                                          types=(np.ndarray, OverrideSub),
                                          args=(array, override_sub),
                                          kwargs={})
        assert_equal(result, 'original')

        with assert_raises_regex(TypeError, 'no implementation found'):
            np.concatenate((array, other))

        expected = np.concatenate((array, array))
        result = np.concatenate((array, no_override_sub))
        assert_equal(result, expected.view(NoOverrideSub))
        result = np.concatenate((array, override_sub))
        assert_equal(result, expected.view(OverrideSub))

    def test_no_wrapper(self):
        # This shouldn't happen unless a user intentionally calls
        # __array_function__ with invalid arguments, but check that we raise
        # an appropriate error all the same.
        array = np.array(1)
        func = lambda x: x
        with assert_raises_regex(AttributeError, '_implementation'):
            array.__array_function__(func=func, types=(np.ndarray,),
                                     args=(array,), kwargs={})


@requires_array_function
class TestArrayFunctionDispatch:

    def test_pickle(self):
        for proto in range(2, pickle.HIGHEST_PROTOCOL + 1):
            roundtripped = pickle.loads(
                    pickle.dumps(dispatched_one_arg, protocol=proto))
            assert_(roundtripped is dispatched_one_arg)

    def test_name_and_docstring(self):
        assert_equal(dispatched_one_arg.__name__, 'dispatched_one_arg')
        if sys.flags.optimize < 2:
            assert_equal(dispatched_one_arg.__doc__, 'Docstring.')

    def test_interface(self):

        class MyArray:
            def __array_function__(self, func, types, args, kwargs):
                return (self, func, types, args, kwargs)

        original = MyArray()
        (obj, func, types, args, kwargs) = dispatched_one_arg(original)
        assert_(obj is original)
        assert_(func is dispatched_one_arg)
        assert_equal(set(types), {MyArray})
        # assert_equal uses the overloaded np.iscomplexobj() internally
        assert_(args == (original,))
        assert_equal(kwargs, {})

    def test_not_implemented(self):

        class MyArray:
            def __array_function__(self, func, types, args, kwargs):
                return NotImplemented

        array = MyArray()
        with assert_raises_regex(TypeError, 'no implementation found'):
            dispatched_one_arg(array)


@requires_array_function
class TestVerifyMatchingSignatures:

    def test_verify_matching_signatures(self):

        verify_matching_signatures(lambda x: 0, lambda x: 0)
        verify_matching_signatures(lambda x=None: 0, lambda x=None: 0)
        verify_matching_signatures(lambda x=1: 0, lambda x=None: 0)

        with assert_raises(RuntimeError):
            verify_matching_signatures(lambda a: 0, lambda b: 0)
        with assert_raises(RuntimeError):
            verify_matching_signatures(lambda x: 0, lambda x=None: 0)
        with assert_raises(RuntimeError):
            verify_matching_signatures(lambda x=None: 0, lambda y=None: 0)
        with assert_raises(RuntimeError):
            verify_matching_signatures(lambda x=1: 0, lambda y=1: 0)

    def test_array_function_dispatch(self):

        with assert_raises(RuntimeError):
            @array_function_dispatch(lambda x: (x,))
            def f(y):
                pass

        # should not raise
        @array_function_dispatch(lambda x: (x,), verify=False)
        def f(y):
            pass


def _new_duck_type_and_implements():
    """Create a duck array type and implements functions."""
    HANDLED_FUNCTIONS = {}

    class MyArray:
        def __array_function__(self, func, types, args, kwargs):
            if func not in HANDLED_FUNCTIONS:
                return NotImplemented
            if not all(issubclass(t, MyArray) for t in types):
                return NotImplemented
            return HANDLED_FUNCTIONS[func](*args, **kwargs)

    def implements(numpy_function):
        """Register an __array_function__ implementations."""
        def decorator(func):
            HANDLED_FUNCTIONS[numpy_function] = func
            return func
        return decorator

    return (MyArray, implements)


@requires_array_function
class TestArrayFunctionImplementation:

    def test_one_arg(self):
        MyArray, implements = _new_duck_type_and_implements()

        @implements(dispatched_one_arg)
        def _(array):
            return 'myarray'

        assert_equal(dispatched_one_arg(1), 'original')
        assert_equal(dispatched_one_arg(MyArray()), 'myarray')

    def test_optional_args(self):
        MyArray, implements = _new_duck_type_and_implements()

        @array_function_dispatch(lambda array, option=None: (array,))
        def func_with_option(array, option='default'):
            return option

        @implements(func_with_option)
        def my_array_func_with_option(array, new_option='myarray'):
            return new_option

        # we don't need to implement every option on __array_function__
        # implementations
        assert_equal(func_with_option(1), 'default')
        assert_equal(func_with_option(1, option='extra'), 'extra')
        assert_equal(func_with_option(MyArray()), 'myarray')
        with assert_raises(TypeError):
            func_with_option(MyArray(), option='extra')

        # but new options on implementations can't be used
        result = my_array_func_with_option(MyArray(), new_option='yes')
        assert_equal(result, 'yes')
        with assert_raises(TypeError):
            func_with_option(MyArray(), new_option='no')

    def test_not_implemented(self):
        MyArray, implements = _new_duck_type_and_implements()

        @array_function_dispatch(lambda array: (array,), module='my')
        def func(array):
            return array

        array = np.array(1)
        assert_(func(array) is array)
        assert_equal(func.__module__, 'my')

        with assert_raises_regex(
                TypeError, "no implementation found for 'my.func'"):
            func(MyArray())

    def test_signature_error_message(self):
        # The lambda function will be named "<lambda>", but the TypeError
        # should show the name as "func"
        def _dispatcher():
            return ()

        @array_function_dispatch(_dispatcher)
        def func():
            pass

        try:
            func(bad_arg=3)
        except TypeError as e:
            expected_exception = e

        try:
            func(bad_arg=3)
            raise AssertionError("must fail")
        except TypeError as exc:
            assert exc.args == expected_exception.args

    @pytest.mark.parametrize("value", [234, "this func is not replaced"])
    def test_dispatcher_error(self, value):
        # If the dispatcher raises an error, we must not attempt to mutate it
        error = TypeError(value)

        def dispatcher():
            raise error

        @array_function_dispatch(dispatcher)
        def func():
            return 3

        try:
            func()
            raise AssertionError("must fail")
        except TypeError as exc:
            assert exc is error  # unmodified exception


class TestNDArrayMethods:

    def test_repr(self):
        # gh-12162: should still be defined even if __array_function__ doesn't
        # implement np.array_repr()

        class MyArray(np.ndarray):
            def __array_function__(*args, **kwargs):
                return NotImplemented

        array = np.array(1).view(MyArray)
        assert_equal(repr(array), 'MyArray(1)')
        assert_equal(str(array), '1')


class TestNumPyFunctions:

    def test_set_module(self):
        assert_equal(np.sum.__module__, 'numpy')
        assert_equal(np.char.equal.__module__, 'numpy.char')
        assert_equal(np.fft.fft.__module__, 'numpy.fft')
        assert_equal(np.linalg.solve.__module__, 'numpy.linalg')

    def test_inspect_sum(self):
        signature = inspect.signature(np.sum)
        assert_('axis' in signature.parameters)

    @requires_array_function
    def test_override_sum(self):
        MyArray, implements = _new_duck_type_and_implements()

        @implements(np.sum)
        def _(array):
            return 'yes'

        assert_equal(np.sum(MyArray()), 'yes')

    @requires_array_function
    def test_sum_on_mock_array(self):

        # We need a proxy for mocks because __array_function__ is only looked
        # up in the class dict
        class ArrayProxy:
            def __init__(self, value):
                self.value = value
            def __array_function__(self, *args, **kwargs):
                return self.value.__array_function__(*args, **kwargs)
            def __array__(self, *args, **kwargs):
                return self.value.__array__(*args, **kwargs)

        proxy = ArrayProxy(mock.Mock(spec=ArrayProxy))
        proxy.value.__array_function__.return_value = 1
        result = np.sum(proxy)
        assert_equal(result, 1)
        proxy.value.__array_function__.assert_called_once_with(
            np.sum, (ArrayProxy,), (proxy,), {})
        proxy.value.__array__.assert_not_called()

    @requires_array_function
    def test_sum_forwarding_implementation(self):

        class MyArray(np.ndarray):

            def sum(self, axis, out):
                return 'summed'

            def __array_function__(self, func, types, args, kwargs):
                return super().__array_function__(func, types, args, kwargs)

        # note: the internal implementation of np.sum() calls the .sum() method
        array = np.array(1).view(MyArray)
        assert_equal(np.sum(array), 'summed')


class TestArrayLike:
    def setup_method(self):
        class MyArray():
            def __init__(self, function=None):
                self.function = function

            def __array_function__(self, func, types, args, kwargs):
                assert func is getattr(np, func.__name__)
                try:
                    my_func = getattr(self, func.__name__)
                except AttributeError:
                    return NotImplemented
                return my_func(*args, **kwargs)

        self.MyArray = MyArray

        class MyNoArrayFunctionArray():
            def __init__(self, function=None):
                self.function = function

        self.MyNoArrayFunctionArray = MyNoArrayFunctionArray

    def add_method(self, name, arr_class, enable_value_error=False):
        def _definition(*args, **kwargs):
            # Check that `like=` isn't propagated downstream
            assert 'like' not in kwargs

            if enable_value_error and 'value_error' in kwargs:
                raise ValueError

            return arr_class(getattr(arr_class, name))
        setattr(arr_class, name, _definition)

    def func_args(*args, **kwargs):
        return args, kwargs

    @requires_array_function
    def test_array_like_not_implemented(self):
        self.add_method('array', self.MyArray)

        ref = self.MyArray.array()

        with assert_raises_regex(TypeError, 'no implementation found'):
            array_like = np.asarray(1, like=ref)

    _array_tests = [
        ('array', *func_args((1,))),
        ('asarray', *func_args((1,))),
        ('asanyarray', *func_args((1,))),
        ('ascontiguousarray', *func_args((2, 3))),
        ('asfortranarray', *func_args((2, 3))),
        ('require', *func_args((np.arange(6).reshape(2, 3),),
                               requirements=['A', 'F'])),
        ('empty', *func_args((1,))),
        ('full', *func_args((1,), 2)),
        ('ones', *func_args((1,))),
        ('zeros', *func_args((1,))),
        ('arange', *func_args(3)),
        ('frombuffer', *func_args(b'\x00' * 8, dtype=int)),
        ('fromiter', *func_args(range(3), dtype=int)),
        ('fromstring', *func_args('1,2', dtype=int, sep=',')),
        ('loadtxt', *func_args(lambda: StringIO('0 1\n2 3'))),
        ('genfromtxt', *func_args(lambda: StringIO('1,2.1'),
                                  dtype=[('int', 'i8'), ('float', 'f8')],
                                  delimiter=',')),
    ]

    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    @pytest.mark.parametrize('numpy_ref', [True, False])
    @requires_array_function
    def test_array_like(self, function, args, kwargs, numpy_ref):
        self.add_method('array', self.MyArray)
        self.add_method(function, self.MyArray)
        np_func = getattr(np, function)
        my_func = getattr(self.MyArray, function)

        if numpy_ref is True:
            ref = np.array(1)
        else:
            ref = self.MyArray.array()

        like_args = tuple(a() if callable(a) else a for a in args)
        array_like = np_func(*like_args, **kwargs, like=ref)

        if numpy_ref is True:
            assert type(array_like) is np.ndarray

            np_args = tuple(a() if callable(a) else a for a in args)
            np_arr = np_func(*np_args, **kwargs)

            # Special-case np.empty to ensure values match
            if function == "empty":
                np_arr.fill(1)
                array_like.fill(1)

            assert_equal(array_like, np_arr)
        else:
            assert type(array_like) is self.MyArray
            assert array_like.function is my_func

    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    @pytest.mark.parametrize('ref', [1, [1], "MyNoArrayFunctionArray"])
    @requires_array_function
    def test_no_array_function_like(self, function, args, kwargs, ref):
        self.add_method('array', self.MyNoArrayFunctionArray)
        self.add_method(function, self.MyNoArrayFunctionArray)
        np_func = getattr(np, function)

        # Instantiate ref if it's the MyNoArrayFunctionArray class
        if ref == "MyNoArrayFunctionArray":
            ref = self.MyNoArrayFunctionArray.array()

        like_args = tuple(a() if callable(a) else a for a in args)

        with assert_raises_regex(TypeError,
                'The `like` argument must be an array-like that implements'):
            np_func(*like_args, **kwargs, like=ref)

    @pytest.mark.parametrize('numpy_ref', [True, False])
    def test_array_like_fromfile(self, numpy_ref):
        self.add_method('array', self.MyArray)
        self.add_method("fromfile", self.MyArray)

        if numpy_ref is True:
            ref = np.array(1)
        else:
            ref = self.MyArray.array()

        data = np.random.random(5)

        with tempfile.TemporaryDirectory() as tmpdir:
            fname = os.path.join(tmpdir, "testfile")
            data.tofile(fname)

            array_like = np.fromfile(fname, like=ref)
            if numpy_ref is True:
                assert type(array_like) is np.ndarray
                np_res = np.fromfile(fname, like=ref)
                assert_equal(np_res, data)
                assert_equal(array_like, np_res)
            else:
                assert type(array_like) is self.MyArray
                assert array_like.function is self.MyArray.fromfile

    @requires_array_function
    def test_exception_handling(self):
        self.add_method('array', self.MyArray, enable_value_error=True)

        ref = self.MyArray.array()

        with assert_raises(TypeError):
            # Raises the error about `value_error` being invalid first
            np.array(1, value_error=True, like=ref)

    @pytest.mark.parametrize('function, args, kwargs', _array_tests)
    def test_like_as_none(self, function, args, kwargs):
        self.add_method('array', self.MyArray)
        self.add_method(function, self.MyArray)
        np_func = getattr(np, function)

        like_args = tuple(a() if callable(a) else a for a in args)
        # required for loadtxt and genfromtxt to init w/o error.
        like_args_exp = tuple(a() if callable(a) else a for a in args)

        array_like = np_func(*like_args, **kwargs, like=None)
        expected = np_func(*like_args_exp, **kwargs)
        # Special-case np.empty to ensure values match
        if function == "empty":
            array_like.fill(1)
            expected.fill(1)
        assert_equal(array_like, expected)
