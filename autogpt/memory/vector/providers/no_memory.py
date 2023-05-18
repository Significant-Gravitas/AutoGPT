"""A class that does not store any data. This is the default memory provider."""
from __future__ import annotations

from typing import Any, Iterator, Sequence

from .. import MemoryItem
from .base import VectorMemoryProvider


class NoMemory(VectorMemoryProvider):
    """
    A class that does not store any data. This is the default memory provider.
    """

    def __iter__(self) -> Iterator[MemoryItem]:
        return iter([])

    def __contains__(self, x: MemoryItem) -> bool:
        return False

    def __len__(self) -> int:
        return 0

    def add(self, item: MemoryItem):
        pass

    def discard(self, item: MemoryItem):
        pass

    def clear(self):
        pass
