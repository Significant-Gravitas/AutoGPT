from typing import Any, TypedDict

from numpy import dtype, ndarray, uint32
from numpy.random.bit_generator import BitGenerator, SeedSequence
from numpy._typing import _ArrayLikeInt_co

class _MT19937Internal(TypedDict):
    key: ndarray[Any, dtype[uint32]]
    pos: int

class _MT19937State(TypedDict):
    bit_generator: str
    state: _MT19937Internal

class MT19937(BitGenerator):
    def __init__(self, seed: None | _ArrayLikeInt_co | SeedSequence = ...) -> None: ...
    def _legacy_seeding(self, seed: _ArrayLikeInt_co) -> None: ...
    def jumped(self, jumps: int = ...) -> MT19937: ...
    @property
    def state(self) -> _MT19937State: ...
    @state.setter
    def state(self, value: _MT19937State) -> None: ...
