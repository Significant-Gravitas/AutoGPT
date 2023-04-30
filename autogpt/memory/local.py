from __future__ import annotations

import dataclasses
from pathlib import Path
from typing import Any, List

import numpy as np
import orjson

from autogpt.llm import get_ada_embedding
from autogpt.memory.base import MemoryProviderSingleton

EMBED_DIM = 1536
SAVE_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS


def create_default_embeddings(embed_dim: int):
    return np.zeros((0, embed_dim)).astype(np.float32)


@dataclasses.dataclass
class CacheContent:
    texts: List[str] = dataclasses.field(default_factory=list)
    embeddings: np.ndarray = dataclasses.field(
        default_factory=create_default_embeddings(EMBED_DIM)
    )

    def __init__(self, embed_dim: int) -> None:
        """Initialize a class instance

        Args:
            embed_dim: Embedding dimension

        Returns:
            None
        """
        self.texts: List[str] = []
        self.embeddings: np.ndarray = create_default_embeddings(embed_dim)


class LocalCache(MemoryProviderSingleton):
    """A class that stores the memory in a local file"""

    def __init__(self, cfg) -> None:
        """Initialize a class instance

        Args:
            cfg: Config object

        Returns:
            None
        """
        workspace_path = Path(cfg.workspace_path)
        self.filename = workspace_path / f"{cfg.memory_index}.json"

        self.filename.touch(exist_ok=True)

        file_content = b"{}"
        with self.filename.open("w+b") as f:
            f.write(file_content)

        self.embed_dim = cfg.embed_dim

        self.data = CacheContent(self.embed_dim)

    def add(self, text: str):
        """
        Add text to our list of texts, add embedding as row to our
            embeddings-matrix

        Args:
            text: str

        Returns: None
        """
        if "Command Error:" in text:
            return ""
        self.data.texts.append(text)

        embedding = get_ada_embedding(text)

        vector = np.array(embedding).astype(np.float32)
        vector = vector[np.newaxis, :]
        self.data.embeddings = np.concatenate(
            [
                self.data.embeddings,
                vector,
            ],
            axis=0,
        )

        with open(self.filename, "wb") as f:
            out = orjson.dumps(self.data, option=SAVE_OPTIONS)
            f.write(out)
        return text

    def clear(self) -> str:
        """
        Clears the data in memory.

        Returns: A message indicating that the memory has been cleared.
        """
        self.data = CacheContent(self.embed_dim)
        return "Obliviated"

    def get(self, data: str) -> list[Any] | None:
        """
        Gets the data from the memory that is most relevant to the given data.

        Args:
            data: The data to compare to.

        Returns: The most relevant data.
        """
        return self.get_relevant(data, 1)

    def get_relevant(self, text: str, k: int) -> list[Any]:
        """
        matrix-vector mult to find score-for-each-row-of-matrix
         get indices for top-k winning scores
         return texts for those indices
        Args:
            text: str
            k: int

        Returns: List[str]
        """
        embedding = get_ada_embedding(text)

        scores = np.dot(self.data.embeddings, embedding)

        top_k_indices = np.argsort(scores)[-k:][::-1]

        return [self.data.texts[i] for i in top_k_indices]

    def get_stats(self) -> tuple[int, tuple[int, ...]]:
        """
        Returns: The stats of the local cache.
        """
        return len(self.data.texts), self.data.embeddings.shape
