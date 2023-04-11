"""
This file tests the generic aspects of ArrayMethod.  At the time of writing
this is private API, but when added, public API may be added here.
"""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest

import numpy as np
from numpy.core._multiarray_umath import _get_castingimpl as get_castingimpl


class TestResolveDescriptors:
    # Test mainly error paths of the resolve_descriptors function,
    # note that the `casting_unittests` tests exercise this non-error paths.

    # Casting implementations are the main/only current user:
    method = get_castingimpl(type(np.dtype("d")), type(np.dtype("f")))

    @pytest.mark.parametrize("args", [
        (True,),  # Not a tuple.
        ((None,)),  # Too few elements
        ((None, None, None),),  # Too many
        ((None, None),),  # Input dtype is None, which is invalid.
        ((np.dtype("d"), True),),  # Output dtype is not a dtype
        ((np.dtype("f"), None),),  # Input dtype does not match method
    ])
    def test_invalid_arguments(self, args):
        with pytest.raises(TypeError):
            self.method._resolve_descriptors(*args)


class TestSimpleStridedCall:
    # Test mainly error paths of the resolve_descriptors function,
    # note that the `casting_unittests` tests exercise this non-error paths.

    # Casting implementations are the main/only current user:
    method = get_castingimpl(type(np.dtype("d")), type(np.dtype("f")))

    @pytest.mark.parametrize(["args", "error"], [
        ((True,), TypeError),  # Not a tuple
        (((None,),), TypeError),  # Too few elements
        ((None, None), TypeError),  # Inputs are not arrays.
        (((None, None, None),), TypeError),  # Too many
        (((np.arange(3), np.arange(3)),), TypeError),  # Incorrect dtypes
        (((np.ones(3, dtype=">d"), np.ones(3, dtype="<f")),),
         TypeError),  # Does not support byte-swapping
        (((np.ones((2, 2), dtype="d"), np.ones((2, 2), dtype="f")),),
         ValueError),  # not 1-D
        (((np.ones(3, dtype="d"), np.ones(4, dtype="f")),),
          ValueError),  # different length
        (((np.frombuffer(b"\0x00"*3*2, dtype="d"),
           np.frombuffer(b"\0x00"*3, dtype="f")),),
         ValueError),  # output not writeable
    ])
    def test_invalid_arguments(self, args, error):
        # This is private API, which may be modified freely
        with pytest.raises(error):
            self.method._simple_strided_call(*args)


@pytest.mark.skipif(sys.version_info < (3, 9), reason="Requires python 3.9")
@pytest.mark.parametrize(
    "cls", [np.ndarray, np.recarray, np.chararray, np.matrix, np.memmap]
)
class TestClassGetItem:
    def test_class_getitem(self, cls: type[np.ndarray]) -> None:
        """Test `ndarray.__class_getitem__`."""
        alias = cls[Any, Any]
        assert isinstance(alias, types.GenericAlias)
        assert alias.__origin__ is cls

    @pytest.mark.parametrize("arg_len", range(4))
    def test_subscript_tup(self, cls: type[np.ndarray], arg_len: int) -> None:
        arg_tup = (Any,) * arg_len
        if arg_len in (1, 2):
            assert cls[arg_tup]
        else:
            match = f"Too {'few' if arg_len == 0 else 'many'} arguments"
            with pytest.raises(TypeError, match=match):
                cls[arg_tup]


@pytest.mark.skipif(sys.version_info >= (3, 9), reason="Requires python 3.8")
def test_class_getitem_38() -> None:
    match = "Type subscription requires python >= 3.9"
    with pytest.raises(TypeError, match=match):
        np.ndarray[Any, Any]
