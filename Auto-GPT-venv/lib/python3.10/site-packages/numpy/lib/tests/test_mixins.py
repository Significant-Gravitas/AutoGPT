import numbers
import operator

import numpy as np
from numpy.testing import assert_, assert_equal, assert_raises


# NOTE: This class should be kept as an exact copy of the example from the
# docstring for NDArrayOperatorsMixin.

class ArrayLike(np.lib.mixins.NDArrayOperatorsMixin):
    def __init__(self, value):
        self.value = np.asarray(value)

    # One might also consider adding the built-in list type to this
    # list, to support operations like np.add(array_like, list)
    _HANDLED_TYPES = (np.ndarray, numbers.Number)

    def __array_ufunc__(self, ufunc, method, *inputs, **kwargs):
        out = kwargs.get('out', ())
        for x in inputs + out:
            # Only support operations with instances of _HANDLED_TYPES.
            # Use ArrayLike instead of type(self) for isinstance to
            # allow subclasses that don't override __array_ufunc__ to
            # handle ArrayLike objects.
            if not isinstance(x, self._HANDLED_TYPES + (ArrayLike,)):
                return NotImplemented

        # Defer to the implementation of the ufunc on unwrapped values.
        inputs = tuple(x.value if isinstance(x, ArrayLike) else x
                       for x in inputs)
        if out:
            kwargs['out'] = tuple(
                x.value if isinstance(x, ArrayLike) else x
                for x in out)
        result = getattr(ufunc, method)(*inputs, **kwargs)

        if type(result) is tuple:
            # multiple return values
            return tuple(type(self)(x) for x in result)
        elif method == 'at':
            # no return value
            return None
        else:
            # one return value
            return type(self)(result)

    def __repr__(self):
        return '%s(%r)' % (type(self).__name__, self.value)


def wrap_array_like(result):
    if type(result) is tuple:
        return tuple(ArrayLike(r) for r in result)
    else:
        return ArrayLike(result)


def _assert_equal_type_and_value(result, expected, err_msg=None):
    assert_equal(type(result), type(expected), err_msg=err_msg)
    if isinstance(result, tuple):
        assert_equal(len(result), len(expected), err_msg=err_msg)
        for result_item, expected_item in zip(result, expected):
            _assert_equal_type_and_value(result_item, expected_item, err_msg)
    else:
        assert_equal(result.value, expected.value, err_msg=err_msg)
        assert_equal(getattr(result.value, 'dtype', None),
                     getattr(expected.value, 'dtype', None), err_msg=err_msg)


_ALL_BINARY_OPERATORS = [
    operator.lt,
    operator.le,
    operator.eq,
    operator.ne,
    operator.gt,
    operator.ge,
    operator.add,
    operator.sub,
    operator.mul,
    operator.truediv,
    operator.floordiv,
    operator.mod,
    divmod,
    pow,
    operator.lshift,
    operator.rshift,
    operator.and_,
    operator.xor,
    operator.or_,
]


class TestNDArrayOperatorsMixin:

    def test_array_like_add(self):

        def check(result):
            _assert_equal_type_and_value(result, ArrayLike(0))

        check(ArrayLike(0) + 0)
        check(0 + ArrayLike(0))

        check(ArrayLike(0) + np.array(0))
        check(np.array(0) + ArrayLike(0))

        check(ArrayLike(np.array(0)) + 0)
        check(0 + ArrayLike(np.array(0)))

        check(ArrayLike(np.array(0)) + np.array(0))
        check(np.array(0) + ArrayLike(np.array(0)))

    def test_inplace(self):
        array_like = ArrayLike(np.array([0]))
        array_like += 1
        _assert_equal_type_and_value(array_like, ArrayLike(np.array([1])))

        array = np.array([0])
        array += ArrayLike(1)
        _assert_equal_type_and_value(array, ArrayLike(np.array([1])))

    def test_opt_out(self):

        class OptOut:
            """Object that opts out of __array_ufunc__."""
            __array_ufunc__ = None

            def __add__(self, other):
                return self

            def __radd__(self, other):
                return self

        array_like = ArrayLike(1)
        opt_out = OptOut()

        # supported operations
        assert_(array_like + opt_out is opt_out)
        assert_(opt_out + array_like is opt_out)

        # not supported
        with assert_raises(TypeError):
            # don't use the Python default, array_like = array_like + opt_out
            array_like += opt_out
        with assert_raises(TypeError):
            array_like - opt_out
        with assert_raises(TypeError):
            opt_out - array_like

    def test_subclass(self):

        class SubArrayLike(ArrayLike):
            """Should take precedence over ArrayLike."""

        x = ArrayLike(0)
        y = SubArrayLike(1)
        _assert_equal_type_and_value(x + y, y)
        _assert_equal_type_and_value(y + x, y)

    def test_object(self):
        x = ArrayLike(0)
        obj = object()
        with assert_raises(TypeError):
            x + obj
        with assert_raises(TypeError):
            obj + x
        with assert_raises(TypeError):
            x += obj

    def test_unary_methods(self):
        array = np.array([-1, 0, 1, 2])
        array_like = ArrayLike(array)
        for op in [operator.neg,
                   operator.pos,
                   abs,
                   operator.invert]:
            _assert_equal_type_and_value(op(array_like), ArrayLike(op(array)))

    def test_forward_binary_methods(self):
        array = np.array([-1, 0, 1, 2])
        array_like = ArrayLike(array)
        for op in _ALL_BINARY_OPERATORS:
            expected = wrap_array_like(op(array, 1))
            actual = op(array_like, 1)
            err_msg = 'failed for operator {}'.format(op)
            _assert_equal_type_and_value(expected, actual, err_msg=err_msg)

    def test_reflected_binary_methods(self):
        for op in _ALL_BINARY_OPERATORS:
            expected = wrap_array_like(op(2, 1))
            actual = op(2, ArrayLike(1))
            err_msg = 'failed for operator {}'.format(op)
            _assert_equal_type_and_value(expected, actual, err_msg=err_msg)

    def test_matmul(self):
        array = np.array([1, 2], dtype=np.float64)
        array_like = ArrayLike(array)
        expected = ArrayLike(np.float64(5))
        _assert_equal_type_and_value(expected, np.matmul(array_like, array))
        _assert_equal_type_and_value(
            expected, operator.matmul(array_like, array))
        _assert_equal_type_and_value(
            expected, operator.matmul(array, array_like))

    def test_ufunc_at(self):
        array = ArrayLike(np.array([1, 2, 3, 4]))
        assert_(np.negative.at(array, np.array([0, 1])) is None)
        _assert_equal_type_and_value(array, ArrayLike([-1, -2, 3, 4]))

    def test_ufunc_two_outputs(self):
        mantissa, exponent = np.frexp(2 ** -3)
        expected = (ArrayLike(mantissa), ArrayLike(exponent))
        _assert_equal_type_and_value(
            np.frexp(ArrayLike(2 ** -3)), expected)
        _assert_equal_type_and_value(
            np.frexp(ArrayLike(np.array(2 ** -3))), expected)
