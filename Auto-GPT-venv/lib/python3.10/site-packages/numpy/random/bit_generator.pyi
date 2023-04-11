import abc
from threading import Lock
from collections.abc import Callable, Mapping, Sequence
from typing import (
    Any,
    NamedTuple,
    TypedDict,
    TypeVar,
    Union,
    overload,
    Literal,
)

from numpy import dtype, ndarray, uint32, uint64
from numpy._typing import _ArrayLikeInt_co, _ShapeLike, _SupportsDType, _UInt32Codes, _UInt64Codes

_T = TypeVar("_T")

_DTypeLikeUint32 = Union[
    dtype[uint32],
    _SupportsDType[dtype[uint32]],
    type[uint32],
    _UInt32Codes,
]
_DTypeLikeUint64 = Union[
    dtype[uint64],
    _SupportsDType[dtype[uint64]],
    type[uint64],
    _UInt64Codes,
]

class _SeedSeqState(TypedDict):
    entropy: None | int | Sequence[int]
    spawn_key: tuple[int, ...]
    pool_size: int
    n_children_spawned: int

class _Interface(NamedTuple):
    state_address: Any
    state: Any
    next_uint64: Any
    next_uint32: Any
    next_double: Any
    bit_generator: Any

class ISeedSequence(abc.ABC):
    @abc.abstractmethod
    def generate_state(
        self, n_words: int, dtype: _DTypeLikeUint32 | _DTypeLikeUint64 = ...
    ) -> ndarray[Any, dtype[uint32 | uint64]]: ...

class ISpawnableSeedSequence(ISeedSequence):
    @abc.abstractmethod
    def spawn(self: _T, n_children: int) -> list[_T]: ...

class SeedlessSeedSequence(ISpawnableSeedSequence):
    def generate_state(
        self, n_words: int, dtype: _DTypeLikeUint32 | _DTypeLikeUint64 = ...
    ) -> ndarray[Any, dtype[uint32 | uint64]]: ...
    def spawn(self: _T, n_children: int) -> list[_T]: ...

class SeedSequence(ISpawnableSeedSequence):
    entropy: None | int | Sequence[int]
    spawn_key: tuple[int, ...]
    pool_size: int
    n_children_spawned: int
    pool: ndarray[Any, dtype[uint32]]
    def __init__(
        self,
        entropy: None | int | Sequence[int] | _ArrayLikeInt_co = ...,
        *,
        spawn_key: Sequence[int] = ...,
        pool_size: int = ...,
        n_children_spawned: int = ...,
    ) -> None: ...
    def __repr__(self) -> str: ...
    @property
    def state(
        self,
    ) -> _SeedSeqState: ...
    def generate_state(
        self, n_words: int, dtype: _DTypeLikeUint32 | _DTypeLikeUint64 = ...
    ) -> ndarray[Any, dtype[uint32 | uint64]]: ...
    def spawn(self, n_children: int) -> list[SeedSequence]: ...

class BitGenerator(abc.ABC):
    lock: Lock
    def __init__(self, seed: None | _ArrayLikeInt_co | SeedSequence = ...) -> None: ...
    def __getstate__(self) -> dict[str, Any]: ...
    def __setstate__(self, state: dict[str, Any]) -> None: ...
    def __reduce__(
        self,
    ) -> tuple[Callable[[str], BitGenerator], tuple[str], tuple[dict[str, Any]]]: ...
    @abc.abstractmethod
    @property
    def state(self) -> Mapping[str, Any]: ...
    @state.setter
    def state(self, value: Mapping[str, Any]) -> None: ...
    @overload
    def random_raw(self, size: None = ..., output: Literal[True] = ...) -> int: ...  # type: ignore[misc]
    @overload
    def random_raw(self, size: _ShapeLike = ..., output: Literal[True] = ...) -> ndarray[Any, dtype[uint64]]: ...  # type: ignore[misc]
    @overload
    def random_raw(self, size: None | _ShapeLike = ..., output: Literal[False] = ...) -> None: ...  # type: ignore[misc]
    def _benchmark(self, cnt: int, method: str = ...) -> None: ...
    @property
    def ctypes(self) -> _Interface: ...
    @property
    def cffi(self) -> _Interface: ...
