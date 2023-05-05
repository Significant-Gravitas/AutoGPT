from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import orjson

from autogpt.config import Config

from ..memory_item import MemoryItem
from ..utils import Embedding, get_embedding
from .base import MemoryProviderSingleton


class JSONFileMemory(MemoryProviderSingleton):
    """Memory backend that stores memories in a JSON file"""

    SAVE_OPTIONS = orjson.OPT_SERIALIZE_NUMPY | orjson.OPT_SERIALIZE_DATACLASS

    file_path: Path
    memories: list[MemoryItem]

    def __init__(self, cfg: Config) -> None:
        """Initialize a class instance

        Args:
            cfg: Config object

        Returns:
            None
        """
        workspace_path = Path(cfg.workspace_path)
        self.file_path = workspace_path / f"{cfg.memory_index}.json"
        self.file_path.touch()

        self.memories = []
        self.save_index()

    def add(self, document: str):
        """
        Add text to our list of texts, add embedding as row to our
            embeddings-matrix

        Args:
            text: str

        Returns: None
        """
        if "Command Error:" in document:
            return None

        new_memory = MemoryItem.from_text(document)

        self.memories.append(new_memory)
        self.save_index()
        return new_memory

    def get(self, query: str) -> MemoryItem | None:
        """
        Gets the data from the memory that is most relevant to the given query.

        Args:
            data: The data to compare to.

        Returns: The most relevant Memory
        """
        return self.get_relevant(query, 1)[0]

    def get_relevant(self, query: str, k: int) -> list[MemoryItem]:
        """
        matrix-vector mult to find score-for-each-row-of-matrix
         get indices for top-k winning scores
         return texts for those indices
        Args:
            text: str
            k: int

        Returns: list[Memory] containing the top [k] relevant memories
        """
        e_query: Embedding = get_embedding(query)

        scores: list[float] = np.dot([m.e_summary for m in self.memories], e_query)
        # scores: list[float] = np.dot([m.e for m in self.memories], e_query)

        top_k_indices = np.argsort(scores)[-k:][::-1]  # take last 5 items and reverse

        return [self.memories[i] for i in top_k_indices]

    def get_stats(self) -> tuple[int, int]:
        """
        Returns: The stats of the local cache in a tuple (n_memories, n_chunks)
        """
        return len(self.memories), functools.reduce(
            lambda t, m: t + len(m.e_chunks), self.memories, 0
        )

    def clear(self):
        """Clears the data in memory."""
        self.memories = []
        self.save_index()

    def save_index(self):
        with self.file_path.open("wb") as f:
            f.write(orjson.dumps(self.memories, option=self.SAVE_OPTIONS))
