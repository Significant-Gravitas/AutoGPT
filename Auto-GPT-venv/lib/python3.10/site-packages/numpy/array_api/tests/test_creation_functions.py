from numpy.testing import assert_raises
import numpy as np

from .. import all
from .._creation_functions import (
    asarray,
    arange,
    empty,
    empty_like,
    eye,
    full,
    full_like,
    linspace,
    meshgrid,
    ones,
    ones_like,
    zeros,
    zeros_like,
)
from .._dtypes import float32, float64
from .._array_object import Array


def test_asarray_errors():
    # Test various protections against incorrect usage
    assert_raises(TypeError, lambda: Array([1]))
    assert_raises(TypeError, lambda: asarray(["a"]))
    assert_raises(ValueError, lambda: asarray([1.0], dtype=np.float16))
    assert_raises(OverflowError, lambda: asarray(2**100))
    # Preferably this would be OverflowError
    # assert_raises(OverflowError, lambda: asarray([2**100]))
    assert_raises(TypeError, lambda: asarray([2**100]))
    asarray([1], device="cpu")  # Doesn't error
    assert_raises(ValueError, lambda: asarray([1], device="gpu"))

    assert_raises(ValueError, lambda: asarray([1], dtype=int))
    assert_raises(ValueError, lambda: asarray([1], dtype="i"))


def test_asarray_copy():
    a = asarray([1])
    b = asarray(a, copy=True)
    a[0] = 0
    assert all(b[0] == 1)
    assert all(a[0] == 0)
    a = asarray([1])
    b = asarray(a, copy=np._CopyMode.ALWAYS)
    a[0] = 0
    assert all(b[0] == 1)
    assert all(a[0] == 0)
    a = asarray([1])
    b = asarray(a, copy=np._CopyMode.NEVER)
    a[0] = 0
    assert all(b[0] == 0)
    assert_raises(NotImplementedError, lambda: asarray(a, copy=False))
    assert_raises(NotImplementedError,
                  lambda: asarray(a, copy=np._CopyMode.IF_NEEDED))


def test_arange_errors():
    arange(1, device="cpu")  # Doesn't error
    assert_raises(ValueError, lambda: arange(1, device="gpu"))
    assert_raises(ValueError, lambda: arange(1, dtype=int))
    assert_raises(ValueError, lambda: arange(1, dtype="i"))


def test_empty_errors():
    empty((1,), device="cpu")  # Doesn't error
    assert_raises(ValueError, lambda: empty((1,), device="gpu"))
    assert_raises(ValueError, lambda: empty((1,), dtype=int))
    assert_raises(ValueError, lambda: empty((1,), dtype="i"))


def test_empty_like_errors():
    empty_like(asarray(1), device="cpu")  # Doesn't error
    assert_raises(ValueError, lambda: empty_like(asarray(1), device="gpu"))
    assert_raises(ValueError, lambda: empty_like(asarray(1), dtype=int))
    assert_raises(ValueError, lambda: empty_like(asarray(1), dtype="i"))


def test_eye_errors():
    eye(1, device="cpu")  # Doesn't error
    assert_raises(ValueError, lambda: eye(1, device="gpu"))
    assert_raises(ValueError, lambda: eye(1, dtype=int))
    assert_raises(ValueError, lambda: eye(1, dtype="i"))


def test_full_errors():
    full((1,), 0, device="cpu")  # Doesn't error
    assert_raises(ValueError, lambda: full((1,), 0, device="gpu"))
    assert_raises(ValueError, lambda: full((1,), 0, dtype=int))
    assert_raises(ValueError, lambda: full((1,), 0, dtype="i"))


def test_full_like_errors():
    full_like(asarray(1), 0, device="cpu")  # Doesn't error
    assert_raises(ValueError, lambda: full_like(asarray(1), 0, device="gpu"))
    assert_raises(ValueError, lambda: full_like(asarray(1), 0, dtype=int))
    assert_raises(ValueError, lambda: full_like(asarray(1), 0, dtype="i"))


def test_linspace_errors():
    linspace(0, 1, 10, device="cpu")  # Doesn't error
    assert_raises(ValueError, lambda: linspace(0, 1, 10, device="gpu"))
    assert_raises(ValueError, lambda: linspace(0, 1, 10, dtype=float))
    assert_raises(ValueError, lambda: linspace(0, 1, 10, dtype="f"))


def test_ones_errors():
    ones((1,), device="cpu")  # Doesn't error
    assert_raises(ValueError, lambda: ones((1,), device="gpu"))
    assert_raises(ValueError, lambda: ones((1,), dtype=int))
    assert_raises(ValueError, lambda: ones((1,), dtype="i"))


def test_ones_like_errors():
    ones_like(asarray(1), device="cpu")  # Doesn't error
    assert_raises(ValueError, lambda: ones_like(asarray(1), device="gpu"))
    assert_raises(ValueError, lambda: ones_like(asarray(1), dtype=int))
    assert_raises(ValueError, lambda: ones_like(asarray(1), dtype="i"))


def test_zeros_errors():
    zeros((1,), device="cpu")  # Doesn't error
    assert_raises(ValueError, lambda: zeros((1,), device="gpu"))
    assert_raises(ValueError, lambda: zeros((1,), dtype=int))
    assert_raises(ValueError, lambda: zeros((1,), dtype="i"))


def test_zeros_like_errors():
    zeros_like(asarray(1), device="cpu")  # Doesn't error
    assert_raises(ValueError, lambda: zeros_like(asarray(1), device="gpu"))
    assert_raises(ValueError, lambda: zeros_like(asarray(1), dtype=int))
    assert_raises(ValueError, lambda: zeros_like(asarray(1), dtype="i"))

def test_meshgrid_dtype_errors():
    # Doesn't raise
    meshgrid()
    meshgrid(asarray([1.], dtype=float32))
    meshgrid(asarray([1.], dtype=float32), asarray([1.], dtype=float32))

    assert_raises(ValueError, lambda: meshgrid(asarray([1.], dtype=float32), asarray([1.], dtype=float64)))
