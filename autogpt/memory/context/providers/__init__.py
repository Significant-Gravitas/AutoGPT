from __future__ import annotations

import abc

from autogpt.config import Config
from autogpt.logs import logger
from autogpt.singleton import AbstractSingleton

from ..memory_item import MemoryItem
from .json_file import JSONFileMemory
from .no_memory import NoMemory
from .pinecone import PineconeMemory
from .redis import RedisMemory


class ContextMemoryProvider(AbstractSingleton):
    @abc.abstractmethod
    def add(self, item: MemoryItem):
        """Adds a MemoryItem to the memory index"""
        pass

    @abc.abstractmethod
    def get(self, query: str) -> MemoryItem | None:
        """Gets an item from memory based on the query"""
        pass

    @abc.abstractmethod
    def get_relevant(self, query: str, num_relevant=5) -> list[MemoryItem]:
        """Gets relevant memory items for the query"""
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        """Clears memory"""
        pass

    @abc.abstractmethod
    def get_stats(self):
        """Get stats from memory"""
        pass


__all__ = [
    "ContextMemoryProvider",
    "JSONFileMemory",
    "NoMemory",
    "PineconeMemory",
    "RedisMemory",
]

# add backends requiring libraries that are not installed by default
try:
    from .milvus import MilvusMemory

    __all__.append("MilvusMemory")
except ImportError:
    pass

try:
    from .weaviate import WeaviateMemory

    __all__.append("WeaviateMemory")
except ImportError:
    pass
