"""Base class for memory providers."""
import abc

import openai

from autogpt.config import AbstractSingleton, Config

cfg = Config()


def get_ada_embedding(text):
    text = text.replace("\n", " ")
    if cfg.use_azure:
        return openai.Embedding.create(
            input=[text],
            engine=cfg.get_azure_deployment_id_for_model("text-embedding-ada-002"),
        )["data"][0]["embedding"]
    else:
        return openai.Embedding.create(input=[text], model="text-embedding-ada-002")[
            "data"
        ][0]["embedding"]


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
