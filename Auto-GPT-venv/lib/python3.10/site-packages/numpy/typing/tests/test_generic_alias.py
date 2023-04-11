from __future__ import annotations

import sys
import copy
import types
import pickle
import weakref
from typing import TypeVar, Any, Union, Callable

import pytest
import numpy as np
from numpy._typing._generic_alias import _GenericAlias
from typing_extensions import Unpack

ScalarType = TypeVar("ScalarType", bound=np.generic, covariant=True)
T1 = TypeVar("T1")
T2 = TypeVar("T2")
DType = _GenericAlias(np.dtype, (ScalarType,))
NDArray = _GenericAlias(np.ndarray, (Any, DType))

# NOTE: The `npt._GenericAlias` *class* isn't quite stable on python >=3.11.
# This is not a problem during runtime (as it's 3.8-exclusive), but we still
# need it for the >=3.9 in order to verify its semantics match
# `types.GenericAlias` replacement. xref numpy/numpy#21526
if sys.version_info >= (3, 9):
    DType_ref = types.GenericAlias(np.dtype, (ScalarType,))
    NDArray_ref = types.GenericAlias(np.ndarray, (Any, DType_ref))
    FuncType = Callable[["_GenericAlias | types.GenericAlias"], Any]
else:
    DType_ref = Any
    NDArray_ref = Any
    FuncType = Callable[["_GenericAlias"], Any]

GETATTR_NAMES = sorted(set(dir(np.ndarray)) - _GenericAlias._ATTR_EXCEPTIONS)

BUFFER = np.array([1], dtype=np.int64)
BUFFER.setflags(write=False)

def _get_subclass_mro(base: type) -> tuple[type, ...]:
    class Subclass(base):  # type: ignore[misc,valid-type]
        pass
    return Subclass.__mro__[1:]


class TestGenericAlias:
    """Tests for `numpy._typing._generic_alias._GenericAlias`."""

    @pytest.mark.parametrize("name,func", [
        ("__init__", lambda n: n),
        ("__init__", lambda n: _GenericAlias(np.ndarray, Any)),
        ("__init__", lambda n: _GenericAlias(np.ndarray, (Any,))),
        ("__init__", lambda n: _GenericAlias(np.ndarray, (Any, Any))),
        ("__init__", lambda n: _GenericAlias(np.ndarray, T1)),
        ("__init__", lambda n: _GenericAlias(np.ndarray, (T1,))),
        ("__init__", lambda n: _GenericAlias(np.ndarray, (T1, T2))),
        ("__origin__", lambda n: n.__origin__),
        ("__args__", lambda n: n.__args__),
        ("__parameters__", lambda n: n.__parameters__),
        ("__mro_entries__", lambda n: n.__mro_entries__([object])),
        ("__hash__", lambda n: hash(n)),
        ("__repr__", lambda n: repr(n)),
        ("__getitem__", lambda n: n[np.float64]),
        ("__getitem__", lambda n: n[ScalarType][np.float64]),
        ("__getitem__", lambda n: n[Union[np.int64, ScalarType]][np.float64]),
        ("__getitem__", lambda n: n[Union[T1, T2]][np.float32, np.float64]),
        ("__eq__", lambda n: n == n),
        ("__ne__", lambda n: n != np.ndarray),
        ("__call__", lambda n: n((1,), np.int64, BUFFER)),
        ("__call__", lambda n: n(shape=(1,), dtype=np.int64, buffer=BUFFER)),
        ("subclassing", lambda n: _get_subclass_mro(n)),
        ("pickle", lambda n: n == pickle.loads(pickle.dumps(n))),
    ])
    def test_pass(self, name: str, func: FuncType) -> None:
        """Compare `types.GenericAlias` with its numpy-based backport.

        Checker whether ``func`` runs as intended and that both `GenericAlias`
        and `_GenericAlias` return the same result.

        """
        value = func(NDArray)

        if sys.version_info >= (3, 9):
            value_ref = func(NDArray_ref)
            assert value == value_ref

    @pytest.mark.parametrize("name,func", [
        ("__copy__", lambda n: n == copy.copy(n)),
        ("__deepcopy__", lambda n: n == copy.deepcopy(n)),
    ])
    def test_copy(self, name: str, func: FuncType) -> None:
        value = func(NDArray)

        # xref bpo-45167
        GE_398 = (
            sys.version_info[:2] == (3, 9) and sys.version_info >= (3, 9, 8)
        )
        if GE_398 or sys.version_info >= (3, 10, 1):
            value_ref = func(NDArray_ref)
            assert value == value_ref

    def test_dir(self) -> None:
        value = dir(NDArray)
        if sys.version_info < (3, 9):
            return

        # A number attributes only exist in `types.GenericAlias` in >= 3.11
        if sys.version_info < (3, 11, 0, "beta", 3):
            value.remove("__typing_unpacked_tuple_args__")
        if sys.version_info < (3, 11, 0, "beta", 1):
            value.remove("__unpacked__")
        assert value == dir(NDArray_ref)

    @pytest.mark.parametrize("name,func,dev_version", [
        ("__iter__", lambda n: len(list(n)), ("beta", 1)),
        ("__iter__", lambda n: next(iter(n)), ("beta", 1)),
        ("__unpacked__", lambda n: n.__unpacked__, ("beta", 1)),
        ("Unpack", lambda n: Unpack[n], ("beta", 1)),

        # The right operand should now have `__unpacked__ = True`,
        # and they are thus now longer equivalent
        ("__ne__", lambda n: n != next(iter(n)), ("beta", 1)),

        # >= beta3
        ("__typing_unpacked_tuple_args__",
         lambda n: n.__typing_unpacked_tuple_args__, ("beta", 3)),

        # >= beta4
        ("__class__", lambda n: n.__class__ == type(n), ("beta", 4)),
    ])
    def test_py311_features(
        self,
        name: str,
        func: FuncType,
        dev_version: tuple[str, int],
    ) -> None:
        """Test Python 3.11 features."""
        value = func(NDArray)

        if sys.version_info >= (3, 11, 0, *dev_version):
            value_ref = func(NDArray_ref)
            assert value == value_ref

    def test_weakref(self) -> None:
        """Test ``__weakref__``."""
        value = weakref.ref(NDArray)()

        if sys.version_info >= (3, 9, 1):  # xref bpo-42332
            value_ref = weakref.ref(NDArray_ref)()
            assert value == value_ref

    @pytest.mark.parametrize("name", GETATTR_NAMES)
    def test_getattr(self, name: str) -> None:
        """Test that `getattr` wraps around the underlying type,
        aka ``__origin__``.

        """
        value = getattr(NDArray, name)
        value_ref1 = getattr(np.ndarray, name)

        if sys.version_info >= (3, 9):
            value_ref2 = getattr(NDArray_ref, name)
            assert value == value_ref1 == value_ref2
        else:
            assert value == value_ref1

    @pytest.mark.parametrize("name,exc_type,func", [
        ("__getitem__", TypeError, lambda n: n[()]),
        ("__getitem__", TypeError, lambda n: n[Any, Any]),
        ("__getitem__", TypeError, lambda n: n[Any][Any]),
        ("isinstance", TypeError, lambda n: isinstance(np.array(1), n)),
        ("issublass", TypeError, lambda n: issubclass(np.ndarray, n)),
        ("setattr", AttributeError, lambda n: setattr(n, "__origin__", int)),
        ("setattr", AttributeError, lambda n: setattr(n, "test", int)),
        ("getattr", AttributeError, lambda n: getattr(n, "test")),
    ])
    def test_raise(
        self,
        name: str,
        exc_type: type[BaseException],
        func: FuncType,
    ) -> None:
        """Test operations that are supposed to raise."""
        with pytest.raises(exc_type):
            func(NDArray)

        if sys.version_info >= (3, 9):
            with pytest.raises(exc_type):
                func(NDArray_ref)
