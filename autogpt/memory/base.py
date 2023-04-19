"""Base class for memory providers."""
import abc

import openai

from autogpt.config import AbstractSingleton, Config

cfg = Config()
embedding_cache = {}

def get_ada_embedding(text):
    text = text.replace("\n", " ")
    if text not in embedding_cache:
        if cfg.use_azure:
            embedding_cache[text] = openai.Embedding.create(
                input=[text],
                engine=cfg.get_azure_deployment_id_for_model("text-embedding-ada-002"),
            )["data"][0]["embedding"]
        else:
            embedding_cache[text] = openai.Embedding.create(
                input=[text], model="text-embedding-ada-002"
            )["data"][0]["embedding"]
    return embedding_cache[text]

class MemoryProviderSingleton(AbstractSingleton):
    @staticmethod
    @abc.abstractmethod
    def add(data):
        pass

    @staticmethod
    @abc.abstractmethod
    def get(data):
        pass

    @staticmethod
    @abc.abstractmethod
    def clear():
        pass

    @staticmethod
    @abc.abstractmethod
    def get_relevant(data, num_relevant=5):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_stats():
        pass
