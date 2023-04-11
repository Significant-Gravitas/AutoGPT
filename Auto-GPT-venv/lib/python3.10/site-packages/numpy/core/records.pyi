import os
from collections.abc import Sequence, Iterable
from typing import (
    Any,
    TypeVar,
    overload,
    Protocol,
)

from numpy import (
    format_parser as format_parser,
    record as record,
    recarray as recarray,
    dtype,
    generic,
    void,
    _ByteOrder,
    _SupportsBuffer,
)

from numpy._typing import (
    ArrayLike,
    DTypeLike,
    NDArray,
    _ShapeLike,
    _ArrayLikeVoid_co,
    _NestedSequence,
)

_SCT = TypeVar("_SCT", bound=generic)

_RecArray = recarray[Any, dtype[_SCT]]

class _SupportsReadInto(Protocol):
    def seek(self, offset: int, whence: int, /) -> object: ...
    def tell(self, /) -> int: ...
    def readinto(self, buffer: memoryview, /) -> int: ...

__all__: list[str]

@overload
def fromarrays(
    arrayList: Iterable[ArrayLike],
    dtype: DTypeLike = ...,
    shape: None | _ShapeLike = ...,
    formats: None = ...,
    names: None = ...,
    titles: None = ...,
    aligned: bool = ...,
    byteorder: None = ...,
) -> _RecArray[Any]: ...
@overload
def fromarrays(
    arrayList: Iterable[ArrayLike],
    dtype: None = ...,
    shape: None | _ShapeLike = ...,
    *,
    formats: DTypeLike,
    names: None | str | Sequence[str] = ...,
    titles: None | str | Sequence[str] = ...,
    aligned: bool = ...,
    byteorder: None | _ByteOrder = ...,
) -> _RecArray[record]: ...

@overload
def fromrecords(
    recList: _ArrayLikeVoid_co | tuple[Any, ...] | _NestedSequence[tuple[Any, ...]],
    dtype: DTypeLike = ...,
    shape: None | _ShapeLike = ...,
    formats: None = ...,
    names: None = ...,
    titles: None = ...,
    aligned: bool = ...,
    byteorder: None = ...,
) -> _RecArray[record]: ...
@overload
def fromrecords(
    recList: _ArrayLikeVoid_co | tuple[Any, ...] | _NestedSequence[tuple[Any, ...]],
    dtype: None = ...,
    shape: None | _ShapeLike = ...,
    *,
    formats: DTypeLike,
    names: None | str | Sequence[str] = ...,
    titles: None | str | Sequence[str] = ...,
    aligned: bool = ...,
    byteorder: None | _ByteOrder = ...,
) -> _RecArray[record]: ...

@overload
def fromstring(
    datastring: _SupportsBuffer,
    dtype: DTypeLike,
    shape: None | _ShapeLike = ...,
    offset: int = ...,
    formats: None = ...,
    names: None = ...,
    titles: None = ...,
    aligned: bool = ...,
    byteorder: None = ...,
) -> _RecArray[record]: ...
@overload
def fromstring(
    datastring: _SupportsBuffer,
    dtype: None = ...,
    shape: None | _ShapeLike = ...,
    offset: int = ...,
    *,
    formats: DTypeLike,
    names: None | str | Sequence[str] = ...,
    titles: None | str | Sequence[str] = ...,
    aligned: bool = ...,
    byteorder: None | _ByteOrder = ...,
) -> _RecArray[record]: ...

@overload
def fromfile(
    fd: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _SupportsReadInto,
    dtype: DTypeLike,
    shape: None | _ShapeLike = ...,
    offset: int = ...,
    formats: None = ...,
    names: None = ...,
    titles: None = ...,
    aligned: bool = ...,
    byteorder: None = ...,
) -> _RecArray[Any]: ...
@overload
def fromfile(
    fd: str | bytes | os.PathLike[str] | os.PathLike[bytes] | _SupportsReadInto,
    dtype: None = ...,
    shape: None | _ShapeLike = ...,
    offset: int = ...,
    *,
    formats: DTypeLike,
    names: None | str | Sequence[str] = ...,
    titles: None | str | Sequence[str] = ...,
    aligned: bool = ...,
    byteorder: None | _ByteOrder = ...,
) -> _RecArray[record]: ...

@overload
def array(
    obj: _SCT | NDArray[_SCT],
    dtype: None = ...,
    shape: None | _ShapeLike = ...,
    offset: int = ...,
    formats: None = ...,
    names: None = ...,
    titles: None = ...,
    aligned: bool = ...,
    byteorder: None = ...,
    copy: bool = ...,
) -> _RecArray[_SCT]: ...
@overload
def array(
    obj: ArrayLike,
    dtype: DTypeLike,
    shape: None | _ShapeLike = ...,
    offset: int = ...,
    formats: None = ...,
    names: None = ...,
    titles: None = ...,
    aligned: bool = ...,
    byteorder: None = ...,
    copy: bool = ...,
) -> _RecArray[Any]: ...
@overload
def array(
    obj: ArrayLike,
    dtype: None = ...,
    shape: None | _ShapeLike = ...,
    offset: int = ...,
    *,
    formats: DTypeLike,
    names: None | str | Sequence[str] = ...,
    titles: None | str | Sequence[str] = ...,
    aligned: bool = ...,
    byteorder: None | _ByteOrder = ...,
    copy: bool = ...,
) -> _RecArray[record]: ...
@overload
def array(
    obj: None,
    dtype: DTypeLike,
    shape: _ShapeLike,
    offset: int = ...,
    formats: None = ...,
    names: None = ...,
    titles: None = ...,
    aligned: bool = ...,
    byteorder: None = ...,
    copy: bool = ...,
) -> _RecArray[Any]: ...
@overload
def array(
    obj: None,
    dtype: None = ...,
    *,
    shape: _ShapeLike,
    offset: int = ...,
    formats: DTypeLike,
    names: None | str | Sequence[str] = ...,
    titles: None | str | Sequence[str] = ...,
    aligned: bool = ...,
    byteorder: None | _ByteOrder = ...,
    copy: bool = ...,
) -> _RecArray[record]: ...
@overload
def array(
    obj: _SupportsReadInto,
    dtype: DTypeLike,
    shape: None | _ShapeLike = ...,
    offset: int = ...,
    formats: None = ...,
    names: None = ...,
    titles: None = ...,
    aligned: bool = ...,
    byteorder: None = ...,
    copy: bool = ...,
) -> _RecArray[Any]: ...
@overload
def array(
    obj: _SupportsReadInto,
    dtype: None = ...,
    shape: None | _ShapeLike = ...,
    offset: int = ...,
    *,
    formats: DTypeLike,
    names: None | str | Sequence[str] = ...,
    titles: None | str | Sequence[str] = ...,
    aligned: bool = ...,
    byteorder: None | _ByteOrder = ...,
    copy: bool = ...,
) -> _RecArray[record]: ...
