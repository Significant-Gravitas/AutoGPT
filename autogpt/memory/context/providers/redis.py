"""Redis memory provider."""
from __future__ import annotations

from typing import Any, Iterator

import redis
from colorama import Fore, Style
from redis.commands.json.commands import JSONCommands
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition, IndexType
from redis.commands.search.query import Query

from autogpt.config import Config
from autogpt.logs import logger
from autogpt.memory.context import MemoryItemRelevance

from ..memory_item import MemoryItem
from ..utils import Embedding, get_embedding
from .abstract import ContextMemoryProvider


class RedisMemory(ContextMemoryProvider):
    cfg: Config

    redis: redis.Redis
    index_name: str

    id_seq: int
    """Last sequential ID in index"""

    @property
    def id_seq_key(self):
        return f"{self.index_name}-id_seq"

    DIMENSION = 1536
    SCHEMA = [
        TextField("content"),
        TextField("summary"),
        VectorField(
            "embedding",
            "HNSW",
            {"TYPE": "FLOAT32", "DIM": DIMENSION, "DISTANCE_METRIC": "IP"},
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
            decode_responses=True,
        )
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
                f"https://docs.agpt.co/configuration/memory/#redis-setup{Style.RESET_ALL}"
                " to ensure you've set up everything correctly."
            )
            exit(1)

        if cfg.wipe_redis_on_start:
            self.clear()
        try:
            self.redis.ft(self.index_name).create_index(
                fields=self.SCHEMA,
                definition=IndexDefinition(
                    prefix=[f"{self.index_name}:"], index_type=IndexType.HASH
                ),
            )
        except Exception as e:
            logger.warn(f"Error creating Redis search index: {e}")

        stored_id_seq: str | None = self.redis.get(self.id_seq_key)
        self.id_seq = int(stored_id_seq) if stored_id_seq else 0

    def __iter__(self) -> Iterator[MemoryItem]:
        for hkey in self.redis.keys(f"{self.index_name}:*[0-9]_header"):
            meta = self.redis.hget(hkey, "metadata")
            memory = self._retrieve(meta["memory_id"])
            if not memory:
                continue
            yield memory

    def __contains__(self, x: MemoryItem) -> bool:
        return bool(self._find(x))

    def __len__(self) -> int:
        return len(self.redis.keys(f"{self.index_name}:*[0-9]_header"))

    def add(self, item: MemoryItem):
        id = self._increment_id_seq()
        self._set(
            f"{id}_header",
            item.raw_content,
            item.summary,
            item.e_summary,
            item.metadata | {"memory_id": id},
        )

        for j, chunk in enumerate(item.chunks):
            self._set(
                f"{id}_chunk-{j}",
                chunk,
                item.chunk_summaries[j],
                item.e_chunks[j],
                {"memory_id": id, "chunk_pos": j},
            )

    def _set(
        self,
        key: str,
        content: str,
        summary: str,
        embedding: Embedding,
        metadata: dict[str, Any] | None = None,
    ):
        data_dict = {
            "content": content,
            "summary": summary,
            "embedding": embedding.tobytes(),
            "metadata": metadata,
        }

        self.redis.hset(f"{self.index_name}:{key}", mapping=data_dict)
        logger.debug(
            f"Inserting item into Redis collection at key '{key}':\n"
            f"summary: {summary}\n"
            f"metadata: {metadata}\n"
            f"content: {content}"
        )

    def _retrieve(self, id: int) -> MemoryItem | None:
        hkey = f"{self.index_name}:{id}_header"
        if not self.redis.exists(hkey):
            return None

        header = self.redis.hgetall(hkey)
        chunks = []
        for ckey in self.redis.keys(f"{self.index_name}:{id}_chunk-*[0-9]"):
            chunks.append(self.redis.hgetall(ckey))

        return MemoryItem(
            raw_content=header["content"],
            summary=header["summary"],
            metadata=header["metadata"],
            e_summary=header["embedding"],
            e_chunks=[c["embedding"] for c in chunks],
            chunks=[c["content"] for c in chunks],
            chunk_summaries=[c["summary"] for c in chunks],
        )

    def _find(self, item: MemoryItem) -> MemoryItem | None:
        results = self.redis.ft(self.index_name).search(
            Query("@content:$content").return_field("metadata"),
            {"content": item.raw_content},
        )
        if not results.docs:
            return None
        return self._retrieve(results.docs[0]["metadata"]["memory_id"])

    def _increment_id_seq(self):
        self.id_seq = self.redis.incrby(self.id_seq_key)
        return self.id_seq

    def get_relevant(self, query: str, k: int) -> list[MemoryItemRelevance]:
        v_query = get_embedding(query).tobytes()

        base_search_query = f"*=>[KNN @embedding $v_query AS relevance_score]"
        search_query = (
            Query(base_search_query)
            .return_fields("metadata", "relevance_score")
            .sort_by("relevance_score")
            .dialect(2)
        )

        try:
            search_result = self.redis.ft(self.index_name).search(
                search_query, query_params={"v_query": v_query}
            )
        except Exception as e:
            logger.warn(f"Error calling Redis search: {e}")
            return []

        results: list[MemoryItemRelevance] = []
        result_ids: list[int] = []
        for r in search_result.docs:
            id = r["metadata"]["memory_id"]
            if id in result_ids:
                continue

            pieces = filter(
                lambda d: d["metadata"]["memory_id"] == id, search_result.docs
            )
            header = next(filter(lambda p: "chunk_pos" not in p["metadata"], pieces))
            chunks = list(filter(lambda p: "chunk_pos" in p["metadata"], pieces))
            chunks.sort(key=lambda c: c["metadata"]["chunk_pos"])
            results.append(
                MemoryItemRelevance(
                    memory_item=self._retrieve(id),
                    for_query=query,
                    summary_relevance_score=header["relevance_score"],
                    chunk_relevance_scores=[cr["relevance_score"] for cr in chunks],
                )
            )
            result_ids.append(id)

            if len(results) == k:
                break
        return results

    def discard(self, item: MemoryItem):
        if not item.metadata["memory_id"]:
            item = self._find(item)
        if item:
            id = item.metadata["memory_id"]
            self.redis.delete(self.redis.keys(f"{self.index_name}:{id}_[hc][eh][au][dn][ek]?*"))

    def clear(self) -> None:
        """Clears the Redis database."""
        self.redis.flushdb()
