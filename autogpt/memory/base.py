"""Base class for memory providers."""
import abc
from typing import Any, Dict, List

import openai

from autogpt.api_manager import api_manager
from autogpt.config import AbstractSingleton, Config

cfg = Config()


def get_ada_embedding(text: str) -> Dict:
    text = text.replace("\n", " ")
    return api_manager.embedding_create(
        text_list=[text], model="text-embedding-ada-002"
    )


class MemoryProviderSingleton(AbstractSingleton):
    @abc.abstractmethod
    def add(self, data: Any) -> None:
        pass

    @abc.abstractmethod
    def get(self, data: Any) -> Any:
        pass

    @abc.abstractmethod
    def clear(self) -> None:
        pass

    @abc.abstractmethod
    def get_relevant(self, data: Any, num_relevant: int = 5) -> List[Any]:
        pass

    @abc.abstractmethod
    def get_stats(self) -> Dict[str, Any]:
        pass
