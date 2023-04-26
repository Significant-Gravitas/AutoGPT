"""Base class for memory providers."""
import abc

from autogpt.config import Config
from autogpt.singleton import AbstractSingleton

cfg = Config()


class MemoryProviderSingleton(AbstractSingleton):
    @abc.abstractmethod
    def add(self, data):
        """Adds to memory"""
        pass

    @abc.abstractmethod
    def get(self, data):
        """Gets from memory"""
        pass

    @abc.abstractmethod
    def clear(self):
        """Clears memory"""
        pass

    @abc.abstractmethod
    def get_relevant(self, data, num_relevant=5):
        """Gets relevant memory for"""
        pass

    @abc.abstractmethod
    def get_stats(self):
        """Get stats from memory"""
        pass
