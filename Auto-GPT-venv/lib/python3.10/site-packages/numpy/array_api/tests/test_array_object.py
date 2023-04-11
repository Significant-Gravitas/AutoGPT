import operator

from numpy.testing import assert_raises
import numpy as np
import pytest

from .. import ones, asarray, reshape, result_type, all, equal
from .._array_object import Array
from .._dtypes import (
    _all_dtypes,
    _boolean_dtypes,
    _floating_dtypes,
    _integer_dtypes,
    _integer_or_boolean_dtypes,
    _numeric_dtypes,
    int8,
    int16,
    int32,
    int64,
    uint64,
    bool as bool_,
)


def test_validate_index():
    # The indexing tests in the official array API test suite test that the
    # array object correctly handles the subset of indices that are required
    # by the spec. But the NumPy array API implementation specifically
    # disallows any index not required by the spec, via Array._validate_index.
    # This test focuses on testing that non-valid indices are correctly
    # rejected. See
    # https://data-apis.org/array-api/latest/API_specification/indexing.html
    # and the docstring of Array._validate_index for the exact indexing
    # behavior that should be allowed. This does not test indices that are
    # already invalid in NumPy itself because Array will generally just pass
    # such indices directly to the underlying np.ndarray.

    a = ones((3, 4))

    # Out of bounds slices are not allowed
    assert_raises(IndexError, lambda: a[:4])
    assert_raises(IndexError, lambda: a[:-4])
    assert_raises(IndexError, lambda: a[:3:-1])
    assert_raises(IndexError, lambda: a[:-5:-1])
    assert_raises(IndexError, lambda: a[4:])
    assert_raises(IndexError, lambda: a[-4:])
    assert_raises(IndexError, lambda: a[4::-1])
    assert_raises(IndexError, lambda: a[-4::-1])

    assert_raises(IndexError, lambda: a[...,:5])
    assert_raises(IndexError, lambda: a[...,:-5])
    assert_raises(IndexError, lambda: a[...,:5:-1])
    assert_raises(IndexError, lambda: a[...,:-6:-1])
    assert_raises(IndexError, lambda: a[...,5:])
    assert_raises(IndexError, lambda: a[...,-5:])
    assert_raises(IndexError, lambda: a[...,5::-1])
    assert_raises(IndexError, lambda: a[...,-5::-1])

    # Boolean indices cannot be part of a larger tuple index
    assert_raises(IndexError, lambda: a[a[:,0]==1,0])
    assert_raises(IndexError, lambda: a[a[:,0]==1,...])
    assert_raises(IndexError, lambda: a[..., a[0]==1])
    assert_raises(IndexError, lambda: a[[True, True, True]])
    assert_raises(IndexError, lambda: a[(True, True, True),])

    # Integer array indices are not allowed (except for 0-D)
    idx = asarray([[0, 1]])
    assert_raises(IndexError, lambda: a[idx])
    assert_raises(IndexError, lambda: a[idx,])
    assert_raises(IndexError, lambda: a[[0, 1]])
    assert_raises(IndexError, lambda: a[(0, 1), (0, 1)])
    assert_raises(IndexError, lambda: a[[0, 1]])
    assert_raises(IndexError, lambda: a[np.array([[0, 1]])])

    # Multiaxis indices must contain exactly as many indices as dimensions
    assert_raises(IndexError, lambda: a[()])
    assert_raises(IndexError, lambda: a[0,])
    assert_raises(IndexError, lambda: a[0])
    assert_raises(IndexError, lambda: a[:])

def test_operators():
    # For every operator, we test that it works for the required type
    # combinations and raises TypeError otherwise
    binary_op_dtypes = {
        "__add__": "numeric",
        "__and__": "integer_or_boolean",
        "__eq__": "all",
        "__floordiv__": "numeric",
        "__ge__": "numeric",
        "__gt__": "numeric",
        "__le__": "numeric",
        "__lshift__": "integer",
        "__lt__": "numeric",
        "__mod__": "numeric",
        "__mul__": "numeric",
        "__ne__": "all",
        "__or__": "integer_or_boolean",
        "__pow__": "numeric",
        "__rshift__": "integer",
        "__sub__": "numeric",
        "__truediv__": "floating",
        "__xor__": "integer_or_boolean",
    }

    # Recompute each time because of in-place ops
    def _array_vals():
        for d in _integer_dtypes:
            yield asarray(1, dtype=d)
        for d in _boolean_dtypes:
            yield asarray(False, dtype=d)
        for d in _floating_dtypes:
            yield asarray(1.0, dtype=d)

    for op, dtypes in binary_op_dtypes.items():
        ops = [op]
        if op not in ["__eq__", "__ne__", "__le__", "__ge__", "__lt__", "__gt__"]:
            rop = "__r" + op[2:]
            iop = "__i" + op[2:]
            ops += [rop, iop]
        for s in [1, 1.0, False]:
            for _op in ops:
                for a in _array_vals():
                    # Test array op scalar. From the spec, the following combinations
                    # are supported:

                    # - Python bool for a bool array dtype,
                    # - a Python int within the bounds of the given dtype for integer array dtypes,
                    # - a Python int or float for floating-point array dtypes

                    # We do not do bounds checking for int scalars, but rather use the default
                    # NumPy behavior for casting in that case.

                    if ((dtypes == "all"
                         or dtypes == "numeric" and a.dtype in _numeric_dtypes
                         or dtypes == "integer" and a.dtype in _integer_dtypes
                         or dtypes == "integer_or_boolean" and a.dtype in _integer_or_boolean_dtypes
                         or dtypes == "boolean" and a.dtype in _boolean_dtypes
                         or dtypes == "floating" and a.dtype in _floating_dtypes
                        )
                        # bool is a subtype of int, which is why we avoid
                        # isinstance here.
                        and (a.dtype in _boolean_dtypes and type(s) == bool
                             or a.dtype in _integer_dtypes and type(s) == int
                             or a.dtype in _floating_dtypes and type(s) in [float, int]
                        )):
                        # Only test for no error
                        getattr(a, _op)(s)
                    else:
                        assert_raises(TypeError, lambda: getattr(a, _op)(s))

                # Test array op array.
                for _op in ops:
                    for x in _array_vals():
                        for y in _array_vals():
                            # See the promotion table in NEP 47 or the array
                            # API spec page on type promotion. Mixed kind
                            # promotion is not defined.
                            if (x.dtype == uint64 and y.dtype in [int8, int16, int32, int64]
                                or y.dtype == uint64 and x.dtype in [int8, int16, int32, int64]
                                or x.dtype in _integer_dtypes and y.dtype not in _integer_dtypes
                                or y.dtype in _integer_dtypes and x.dtype not in _integer_dtypes
                                or x.dtype in _boolean_dtypes and y.dtype not in _boolean_dtypes
                                or y.dtype in _boolean_dtypes and x.dtype not in _boolean_dtypes
                                or x.dtype in _floating_dtypes and y.dtype not in _floating_dtypes
                                or y.dtype in _floating_dtypes and x.dtype not in _floating_dtypes
                                ):
                                assert_raises(TypeError, lambda: getattr(x, _op)(y))
                            # Ensure in-place operators only promote to the same dtype as the left operand.
                            elif (
                                _op.startswith("__i")
                                and result_type(x.dtype, y.dtype) != x.dtype
                            ):
                                assert_raises(TypeError, lambda: getattr(x, _op)(y))
                            # Ensure only those dtypes that are required for every operator are allowed.
                            elif (dtypes == "all" and (x.dtype in _boolean_dtypes and y.dtype in _boolean_dtypes
                                                      or x.dtype in _numeric_dtypes and y.dtype in _numeric_dtypes)
                                or (dtypes == "numeric" and x.dtype in _numeric_dtypes and y.dtype in _numeric_dtypes)
                                or dtypes == "integer" and x.dtype in _integer_dtypes and y.dtype in _numeric_dtypes
                                or dtypes == "integer_or_boolean" and (x.dtype in _integer_dtypes and y.dtype in _integer_dtypes
                                                                       or x.dtype in _boolean_dtypes and y.dtype in _boolean_dtypes)
                                or dtypes == "boolean" and x.dtype in _boolean_dtypes and y.dtype in _boolean_dtypes
                                or dtypes == "floating" and x.dtype in _floating_dtypes and y.dtype in _floating_dtypes
                            ):
                                getattr(x, _op)(y)
                            else:
                                assert_raises(TypeError, lambda: getattr(x, _op)(y))

    unary_op_dtypes = {
        "__abs__": "numeric",
        "__invert__": "integer_or_boolean",
        "__neg__": "numeric",
        "__pos__": "numeric",
    }
    for op, dtypes in unary_op_dtypes.items():
        for a in _array_vals():
            if (
                dtypes == "numeric"
                and a.dtype in _numeric_dtypes
                or dtypes == "integer_or_boolean"
                and a.dtype in _integer_or_boolean_dtypes
            ):
                # Only test for no error
                getattr(a, op)()
            else:
                assert_raises(TypeError, lambda: getattr(a, op)())

    # Finally, matmul() must be tested separately, because it works a bit
    # different from the other operations.
    def _matmul_array_vals():
        for a in _array_vals():
            yield a
        for d in _all_dtypes:
            yield ones((3, 4), dtype=d)
            yield ones((4, 2), dtype=d)
            yield ones((4, 4), dtype=d)

    # Scalars always error
    for _op in ["__matmul__", "__rmatmul__", "__imatmul__"]:
        for s in [1, 1.0, False]:
            for a in _matmul_array_vals():
                if (type(s) in [float, int] and a.dtype in _floating_dtypes
                    or type(s) == int and a.dtype in _integer_dtypes):
                    # Type promotion is valid, but @ is not allowed on 0-D
                    # inputs, so the error is a ValueError
                    assert_raises(ValueError, lambda: getattr(a, _op)(s))
                else:
                    assert_raises(TypeError, lambda: getattr(a, _op)(s))

    for x in _matmul_array_vals():
        for y in _matmul_array_vals():
            if (x.dtype == uint64 and y.dtype in [int8, int16, int32, int64]
                or y.dtype == uint64 and x.dtype in [int8, int16, int32, int64]
                or x.dtype in _integer_dtypes and y.dtype not in _integer_dtypes
                or y.dtype in _integer_dtypes and x.dtype not in _integer_dtypes
                or x.dtype in _floating_dtypes and y.dtype not in _floating_dtypes
                or y.dtype in _floating_dtypes and x.dtype not in _floating_dtypes
                or x.dtype in _boolean_dtypes
                or y.dtype in _boolean_dtypes
                ):
                assert_raises(TypeError, lambda: x.__matmul__(y))
                assert_raises(TypeError, lambda: y.__rmatmul__(x))
                assert_raises(TypeError, lambda: x.__imatmul__(y))
            elif x.shape == () or y.shape == () or x.shape[1] != y.shape[0]:
                assert_raises(ValueError, lambda: x.__matmul__(y))
                assert_raises(ValueError, lambda: y.__rmatmul__(x))
                if result_type(x.dtype, y.dtype) != x.dtype:
                    assert_raises(TypeError, lambda: x.__imatmul__(y))
                else:
                    assert_raises(ValueError, lambda: x.__imatmul__(y))
            else:
                x.__matmul__(y)
                y.__rmatmul__(x)
                if result_type(x.dtype, y.dtype) != x.dtype:
                    assert_raises(TypeError, lambda: x.__imatmul__(y))
                elif y.shape[0] != y.shape[1]:
                    # This one fails because x @ y has a different shape from x
                    assert_raises(ValueError, lambda: x.__imatmul__(y))
                else:
                    x.__imatmul__(y)


def test_python_scalar_construtors():
    b = asarray(False)
    i = asarray(0)
    f = asarray(0.0)

    assert bool(b) == False
    assert int(i) == 0
    assert float(f) == 0.0
    assert operator.index(i) == 0

    # bool/int/float should only be allowed on 0-D arrays.
    assert_raises(TypeError, lambda: bool(asarray([False])))
    assert_raises(TypeError, lambda: int(asarray([0])))
    assert_raises(TypeError, lambda: float(asarray([0.0])))
    assert_raises(TypeError, lambda: operator.index(asarray([0])))

    # bool/int/float should only be allowed on arrays of the corresponding
    # dtype
    assert_raises(ValueError, lambda: bool(i))
    assert_raises(ValueError, lambda: bool(f))

    assert_raises(ValueError, lambda: int(b))
    assert_raises(ValueError, lambda: int(f))

    assert_raises(ValueError, lambda: float(b))
    assert_raises(ValueError, lambda: float(i))

    assert_raises(TypeError, lambda: operator.index(b))
    assert_raises(TypeError, lambda: operator.index(f))


def test_device_property():
    a = ones((3, 4))
    assert a.device == 'cpu'

    assert all(equal(a.to_device('cpu'), a))
    assert_raises(ValueError, lambda: a.to_device('gpu'))

    assert all(equal(asarray(a, device='cpu'), a))
    assert_raises(ValueError, lambda: asarray(a, device='gpu'))

def test_array_properties():
    a = ones((1, 2, 3))
    b = ones((2, 3))
    assert_raises(ValueError, lambda: a.T)

    assert isinstance(b.T, Array)
    assert b.T.shape == (3, 2)

    assert isinstance(a.mT, Array)
    assert a.mT.shape == (1, 3, 2)
    assert isinstance(b.mT, Array)
    assert b.mT.shape == (3, 2)

def test___array__():
    a = ones((2, 3), dtype=int16)
    assert np.asarray(a) is a._array
    b = np.asarray(a, dtype=np.float64)
    assert np.all(np.equal(b, np.ones((2, 3), dtype=np.float64)))
    assert b.dtype == np.float64

def test_allow_newaxis():
    a = ones(5)
    indexed_a = a[None, :]
    assert indexed_a.shape == (1, 5)

def test_disallow_flat_indexing_with_newaxis():
    a = ones((3, 3, 3))
    with pytest.raises(IndexError):
        a[None, 0, 0]

def test_disallow_mask_with_newaxis():
    a = ones((3, 3, 3))
    with pytest.raises(IndexError):
        a[None, asarray(True)]

@pytest.mark.parametrize("shape", [(), (5,), (3, 3, 3)])
@pytest.mark.parametrize("index", ["string", False, True])
def test_error_on_invalid_index(shape, index):
    a = ones(shape)
    with pytest.raises(IndexError):
        a[index]

def test_mask_0d_array_without_errors():
    a = ones(())
    a[asarray(True)]

@pytest.mark.parametrize(
    "i", [slice(5), slice(5, 0), asarray(True), asarray([0, 1])]
)
def test_error_on_invalid_index_with_ellipsis(i):
    a = ones((3, 3, 3))
    with pytest.raises(IndexError):
        a[..., i]
    with pytest.raises(IndexError):
        a[i, ...]

def test_array_keys_use_private_array():
    """
    Indexing operations convert array keys before indexing the internal array

    Fails when array_api array keys are not converted into NumPy-proper arrays
    in __getitem__(). This is achieved by passing array_api arrays with 0-sized
    dimensions, which NumPy-proper treats erroneously - not sure why!

    TODO: Find and use appropriate __setitem__() case.
    """
    a = ones((0, 0), dtype=bool_)
    assert a[a].shape == (0,)

    a = ones((0,), dtype=bool_)
    key = ones((0, 0), dtype=bool_)
    with pytest.raises(IndexError):
        a[key]
