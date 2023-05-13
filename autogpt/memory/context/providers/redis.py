"""Redis memory provider."""
from __future__ import annotations

from typing import Any

import numpy as np
import redis
from colorama import Fore, Style
from redis.commands.json.commands import JSONCommands
from redis.commands.search.field import Field, TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from autogpt.config import Config
from autogpt.logs import logger

from ..memory_item import MemoryItem
from ..utils import Embedding, get_embedding
from .abstract import ContextMemoryProvider


class RedisMemory(ContextMemoryProvider):
    cfg: Config

    redis: redis.Redis
    dimension: int
    index_name: str

    n_memories: int
    """Number of items currently in memory"""

    SCHEMA = [
        TextField("raw_content"),
        VectorField(
            "embedding",
            "HNSW",
            {"TYPE": "FLOAT32", "DIM": 1536, "DISTANCE_METRIC": "IP"},
        ),
    ]

    def __init__(self, cfg: Config):
        """
        Initializes the Redis memory provider.

        Args:
            cfg: The config object.

        Returns: None
        """
        self.cfg = cfg

        redis_host = cfg.redis_host
        redis_port = cfg.redis_port
        redis_password = cfg.redis_password
        self.redis = redis.Redis(
            host=redis_host,
            port=redis_port,
            password=redis_password,
            db=0,  # Cannot be changed
        )
        self.dimension = 1536
        self.index_name = cfg.memory_index

        # Check redis connection
        try:
            self.redis.ping()
        except redis.ConnectionError as e:
            logger.typewriter_log(
                "FAILED TO CONNECT TO REDIS",
                Fore.RED,
                Style.BRIGHT + str(e) + Style.RESET_ALL,
            )
            logger.double_check(
                "Please ensure you have setup and configured Redis properly for use. "
                + f"You can check out {Fore.CYAN + Style.BRIGHT}"
                f"https://github.com/Torantulino/Auto-GPT#redis-setup{Style.RESET_ALL}"
                " to ensure you've set up everything correctly."
            )
            exit(1)

        if cfg.wipe_redis_on_start:
            self.clear()
        try:
            self.redis.ft(f"{self.index_name}").create_index(
                fields=self.SCHEMA,
                definition=IndexDefinition(
                    prefix=[f"{self.index_name}:"], index_type=IndexType.HASH
                ),
            )
        except Exception as e:
            logger.warn("Error creating Redis search index: ", e)

        n_existing_memories: bytes | None = self.redis.get(
            f"{self.index_name}-n_memories"
        )
        self.n_memories = (
            int(n_existing_memories.decode("utf-8")) if n_existing_memories else 0
        )

    def add(self, document: str) -> MemoryItem | None:
        """
        Adds a data point to the memory.

        Args:
            data: The data to add.

        Returns: Message indicating that the data has been added.
        """
        if "Command Error:" in document:
            return None

        memory = MemoryItem.from_text(document)

        data_dict = {"raw_content": document, "embedding": memory.e_summary.tobytes()}

        pipe = self.redis.pipeline()
        pipe.hset(f"{self.index_name}:{self.n_memories}", mapping=data_dict)
        logger.debug(
            f"Inserting data into memory at index: {self.n_memories}:\n"
            f"data: {document}"
        )
        self.n_memories += 1
        pipe.set(f"{self.index_name}-n_memories", self.n_memories)
        pipe.execute()
        return memory

    def get(self, data: str) -> list[Any] | None:
        """
        Gets the data from the memory that is most relevant to the given data.

        Args:
            data: The data to compare to.

        Returns: The most relevant data.
        """
        return self.get_relevant(data, 1)

    def get_relevant(self, query: str, num_relevant: int = 5) -> list[Any] | None:
        """
        Returns all the data in the memory that is relevant to the given data.
        Args:
            data: The data to compare to.
            num_relevant: The number of relevant data to return.

        Returns: A list of the most relevant data.
        """
        query_embedding = get_embedding(query)
        base_query = f"*=>[KNN {num_relevant} @embedding $vector AS vector_score]"
        query = (
            Query(base_query)
            .return_fields("raw_content", "vector_score")
            .sort_by("vector_score")
            .dialect(2)
        )
        query_vector = query_embedding.tobytes()

        try:
            results = self.redis.ft(f"{self.index_name}").search(
                query, query_params={"vector": query_vector}
            )
        except Exception as e:
            logger.warn("Error calling Redis search: ", e)
            return None
        return [result.data for result in results.docs]

    def get_stats(self):
        """
        Returns: The stats of the memory index.
        """
        return self.redis.ft(f"{self.index_name}").info()

    def clear(self) -> None:
        """Clears the Redis database."""
        self.redis.flushall()
