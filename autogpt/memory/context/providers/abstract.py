import abc
import functools
from typing import MutableSet

import numpy as np

from autogpt.logs import logger
from autogpt.memory.context import MemoryItem, MemoryItemRelevance
from autogpt.memory.context.utils import Embedding, get_embedding
from autogpt.singleton import AbstractSingleton


class ContextMemoryProvider(MutableSet[MemoryItem], AbstractSingleton):
    def get(self, query: str) -> MemoryItemRelevance | None:
        """
        Gets the data from the memory that is most relevant to the given query.

        Args:
            data: The data to compare to.

        Returns: The most relevant Memory
        """
        result = self.get_relevant(query, 1)
        return result[0] if result else None

    def get_relevant(self, query: str, k: int) -> list[MemoryItemRelevance]:
        """
        Returns the top-k most relevant memories for the given query

        Args:
            query: the query to compare stored memories to
            k: the number of relevant memories to fetch

        Returns:
            list[MemoryItemRelevance] containing the top [k] relevant memories
        """
        if len(self) < 1:
            return []

        e_query: Embedding = get_embedding(query)

        logger.debug(f"Searching for {k} relevant items; {len(self)} in index")

        relevances = [m.relevance_for(query, e_query) for m in self]
        logger.debug(f"Memory match scores: {[str(r) for r in relevances]}")

        # take last k items and reverse
        top_k_indices = np.argsort([r.score for r in relevances])[-k:][::-1]

        return [relevances[i] for i in top_k_indices]

    def get_stats(self) -> tuple[int, int]:
        """
        Returns: The stats of the memory in a tuple (n_memories, n_chunks)
        """
        return len(self), functools.reduce(
            lambda t, m: t + len(m.e_chunks), self, 0
        )
