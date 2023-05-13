import abc

from autogpt.memory.context import MemoryItem
from autogpt.singleton import AbstractSingleton


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
