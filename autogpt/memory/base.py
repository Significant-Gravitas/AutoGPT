"""Base class for memory providers."""
import abc

import openai

from autogpt.api_manager import api_manager
from autogpt.config import AbstractSingleton, Config

cfg = Config()


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    return api_manager.embedding_create(
        text_list=[text], model="text-embedding-ada-002"
    )


class MemoryProviderSingleton(AbstractSingleton):
    @abc.abstractmethod
    def add(self, data):
        pass

    @abc.abstractmethod
    def get(self, data):
        pass

    @abc.abstractmethod
    def clear(self):
        pass

    @abc.abstractmethod
    def get_relevant(self, data, num_relevant=5):
        pass

    @abc.abstractmethod
    def get_stats(self):
        pass
