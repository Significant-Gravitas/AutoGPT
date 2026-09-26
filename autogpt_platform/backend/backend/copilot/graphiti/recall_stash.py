"""The forget stash: what each recent forget set on an edge, in Redis, where
an ingestion in any process can read it.

graphiti's ``add_episode`` saves the edges and entities it read when it
started (``SET r = edge``, ``SET n = node``), so a forget that lands while an
ingestion runs can be overwritten by that older copy, and the ingestion loop
is per process, so it cannot simply be queued behind the forget.
``recall_forget`` therefore writes a ``ForgetRecord`` for every edge before it
touches the graph, and every ingestion reads the stash when it is done and
applies again each forget that landed while it ran (``recall_ingest.py``). A
repair that failed leaves its record here too, so the next ingestion or the
next forget of that edge finishes it.

A record lives for ``STASH_TTL`` after it was written. Redis is best effort:
a failed or slow call is logged and the forget or ingestion goes on without
it, which reopens, for that forget, the window the stash closes.
"""

import asyncio
import logging
from collections.abc import Awaitable
from datetime import datetime, timedelta, timezone
from typing import cast

from pydantic import BaseModel, ValidationError

from backend.data.redis_client import get_redis_async

from .memory_model import MemoryStatus
from .recall import FORGOTTEN_FACT, USER_FORGET_REASON
from .scope import forget_stash_key

logger = logging.getLogger(__name__)

STASH_TTL = timedelta(hours=1)
# Every Redis call is bounded: ``get_redis_async`` retries a lost connection
# for minutes, and neither a forget nor an ingestion may wait on that. The
# bound is ``asyncio.timeout``, not ``wait_for``, which on Python 3.11 can
# swallow the cancellation of the ingestion worker calling it.
_REDIS_TIMEOUT_SECONDS = 2.0


class ForgetRecord(BaseModel):
    """The fields a forget owns on one edge, as it set them.

    ``episodes`` are the edge's sources when it was forgotten,
    ``redacted_episodes`` the episodes the forget hid. ``dropped_episodes``
    are episodes that stated the fact again after the forget but that a
    failed repair left on the edge. The endpoints are kept by uuid and by
    name, for a hard forget whose entities an ingestion saved again.
    """

    uuid: str
    hard: bool = False
    forgotten_at: str
    status: str = MemoryStatus.retracted.value
    expiration_reason: str = USER_FORGET_REASON
    expired_at: str
    invalid_at: str | None = None
    valid_at: str | None = None
    fact: str = FORGOTTEN_FACT
    fact_redacted: str | None = None
    name: str = FORGOTTEN_FACT
    name_redacted: str | None = None
    episodes: list[str] = []
    redacted_episodes: list[str] = []
    dropped_episodes: list[str] = []
    source: str | None = None
    target: str | None = None
    source_name: str | None = None
    target_name: str | None = None
    stashed_at: str


async def stash_forgets(group_id: str, records: list[ForgetRecord]) -> bool:
    """Write ``records`` into graph ``group_id``'s stash, replacing any older
    record of the same edge; False when Redis could not take them."""
    if not records:
        return True
    try:
        async with asyncio.timeout(_REDIS_TIMEOUT_SECONDS):
            await _write(group_id, records)
    except Exception:
        logger.warning(
            f"Forget stash write failed for graph {group_id[:20]}: an ingestion "
            "running now can undo this forget",
            exc_info=True,
        )
        return False
    return True


async def read_forgets(group_id: str) -> dict[str, ForgetRecord]:
    """Graph ``group_id``'s records younger than ``STASH_TTL``, by edge uuid;
    empty when Redis could not be read."""
    try:
        async with asyncio.timeout(_REDIS_TIMEOUT_SECONDS):
            records = await _read(group_id)
    except Exception:
        logger.warning(
            f"Forget stash read failed for graph {group_id[:20]}", exc_info=True
        )
        return {}
    return {record.uuid: record for record in records}


async def _write(group_id: str, records: list[ForgetRecord]) -> None:
    redis = await get_redis_async()
    key = forget_stash_key(group_id)
    async with redis.pipeline(transaction=True) as pipe:
        pipe.hset(key, mapping={r.uuid: r.model_dump_json() for r in records})
        pipe.expire(key, int(STASH_TTL.total_seconds()))
        await pipe.execute()


async def _read(group_id: str) -> list[ForgetRecord]:
    """The fresh records; stale and unreadable ones are deleted on the way."""
    redis = await get_redis_async()
    key = forget_stash_key(group_id)
    stored = await cast(Awaitable[dict[str, str]], redis.hgetall(key))
    cutoff = datetime.now(timezone.utc) - STASH_TTL
    fresh = {
        uuid: record
        for uuid, value in stored.items()
        if (record := _parse(value)) is not None
        and datetime.fromisoformat(record.stashed_at) >= cutoff
    }
    stale = [uuid for uuid in stored if uuid not in fresh]
    if stale:
        await cast(Awaitable[int], redis.hdel(key, *stale))
    return list(fresh.values())


def _parse(value: str) -> ForgetRecord | None:
    try:
        return ForgetRecord.model_validate_json(value)
    except ValidationError:
        logger.warning("Unreadable forget stash record skipped")
        return None
