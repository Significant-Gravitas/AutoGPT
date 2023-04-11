from typing import Any, Generic, TypeVar

from frozenlist import FrozenList

__all__ = ("Signal",)

_T = TypeVar("_T")

class Signal(FrozenList[_T], Generic[_T]):
    def __init__(self, owner: Any) -> None: ...
    def __repr__(self) -> str: ...
    async def send(self, *args: Any, **kwargs: Any) -> None: ...
