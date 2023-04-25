"""Base class for memory providers."""
import abc

from autogpt.config import Config
from autogpt.singleton import AbstractSingleton

cfg = Config()


class MemoryProviderSingleton(AbstractSingleton):
    @abc.abstractmethod
    def add(self, agent_name : str,  data):
        pass

    @abc.abstractmethod
    def get(self, agent_name : str, data):
        pass

    @abc.abstractmethod
    def clear(self, agent_name : str):
        pass

    @abc.abstractmethod
    def get_relevant(self, agent_name : str, data, num_relevant=5):
        pass

    @abc.abstractmethod
    def get_stats(self):
        pass
