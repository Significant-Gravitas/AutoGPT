import abc
import functools
import logging
from typing import MutableSet, Sequence

import numpy as np
from forge.config.config import Config

from .. import MemoryItem, MemoryItemRelevance
from ..utils import Embedding, get_embedding

logger = logging.getLogger(__name__)


class VectorMemoryProvider(MutableSet[MemoryItem]):
    @abc.abstractmethod
    def __init__(self, config: Config):
        pass

    def get(self, query: str, config: Config) -> MemoryItemRelevance | None:
        """
        Gets the data from the memory that is most relevant to the given query.

        Args:
            query: The query used to retrieve information.
            config: The config Object.

        Returns: The most relevant Memory
        """
        result = self.get_relevant(query, 1, config)
        return result[0] if result else None

    def get_relevant(
        self, query: str, k: int, config: Config
    ) -> Sequence[MemoryItemRelevance]:
        """
        Returns the top-k most relevant memories for the given query

        Args:
            query: the query to compare stored memories to
            k: the number of relevant memories to fetch
            config: The config Object.

        Returns:
            list[MemoryItemRelevance] containing the top [k] relevant memories
        """
        if len(self) < 1:
            return []

        logger.debug(
            f"Searching for {k} relevant memories for query '{query}'; "
            f"{len(self)} memories in index"
        )

        relevances = self.score_memories_for_relevance(query, config)
        logger.debug(f"Memory relevance scores: {[str(r) for r in relevances]}")

        # take last k items and reverse
        top_k_indices = np.argsort([r.score for r in relevances])[-k:][::-1]

        return [relevances[i] for i in top_k_indices]

    def score_memories_for_relevance(
        self, for_query: str, config: Config
    ) -> Sequence[MemoryItemRelevance]:
        """
        Returns MemoryItemRelevance for every memory in the index.
        Implementations may override this function for performance purposes.
        """
        e_query: Embedding = get_embedding(for_query, config)
        return [m.relevance_for(for_query, e_query) for m in self]

    def get_stats(self) -> tuple[int, int]:
        """
        Returns:
            tuple (n_memories: int, n_chunks: int): the stats of the memory index
        """
        return len(self), functools.reduce(lambda t, m: t + len(m.e_chunks), self, 0)
