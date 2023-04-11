from typing import Any, TypeVar, overload, Generic
import ctypes as ct

from numpy import ndarray
from numpy.ctypeslib import c_intp

_CastT = TypeVar("_CastT", bound=ct._CanCastTo)  # Copied from `ctypes.cast`
_CT = TypeVar("_CT", bound=ct._CData)
_PT = TypeVar("_PT", bound=None | int)

# TODO: Let the likes of `shape_as` and `strides_as` return `None`
# for 0D arrays once we've got shape-support

class _ctypes(Generic[_PT]):
    @overload
    def __new__(cls, array: ndarray[Any, Any], ptr: None = ...) -> _ctypes[None]: ...
    @overload
    def __new__(cls, array: ndarray[Any, Any], ptr: _PT) -> _ctypes[_PT]: ...
    @property
    def data(self) -> _PT: ...
    @property
    def shape(self) -> ct.Array[c_intp]: ...
    @property
    def strides(self) -> ct.Array[c_intp]: ...
    @property
    def _as_parameter_(self) -> ct.c_void_p: ...

    def data_as(self, obj: type[_CastT]) -> _CastT: ...
    def shape_as(self, obj: type[_CT]) -> ct.Array[_CT]: ...
    def strides_as(self, obj: type[_CT]) -> ct.Array[_CT]: ...
