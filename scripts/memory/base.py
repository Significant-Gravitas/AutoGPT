import abc
from config import AbstractSingleton, Config
import openai

class MemoryProviderSingleton(AbstractSingleton):
    """Base class for memory providers."""

    def __init__(self, config: Config):
        self.config = config

    def get_ada_embedding(self, text: str):
        """
        Get the ADA embedding of a given text.

        :param text: The input text to get the embedding for.
        :return: The ADA embedding for the input text.
        """
        text = text.replace("\n", " ")
        if self.config.use_azure:
            return openai.Embedding.create(input=[text], engine=self.config.get_azure_deployment_id_for_model("text-embedding-ada-002"))["data"][0]["embedding"]
        else:
            return openai.Embedding.create(input=[text], model="text-embedding-ada-002")["data"][0]["embedding"]

    @abc.abstractmethod
    def add(self, data):
        """Add data to the memory provider."""
        pass

    @abc.abstractmethod
    def get(self, data):
        """Get data from the memory provider."""
        pass

    @abc.abstractmethod
    def clear(self):
        """Clear the memory provider."""
        pass

    @abc.abstractmethod
    def get_relevant(self, data, num_relevant=5):
        """Get relevant data from the memory provider."""
        pass

    @abc.abstractmethod
    def get_stats(self):
        """Get stats of the memory provider."""
        pass
    