from unittest.mock import AsyncMock

import pytest
from fakeredis.aioredis import FakeRedis

from backend.data.redis_helpers import hash_compare_and_set


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "record,updates",
    [
        (("meta:{s}", "value", 60), None),
        (("terminal:{s}", "value", 0), None),
        (("terminal:{other}", "value", 60), None),
        (None, {"status": "clobber"}),
    ],
)
async def test_invalid_companions_cannot_partially_complete(record, updates):
    redis = AsyncMock()
    with pytest.raises(ValueError):
        await hash_compare_and_set(
            redis,
            "meta:{s}",
            "status",
            expected="running",
            new="failed",
            record=record,
            updates=updates,
        )
    redis.eval.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("decoded", [False, True])
async def test_terminal_cas_lua_ownership_and_independent_ttl(decoded):
    async with FakeRedis(decode_responses=decoded) as redis:
        meta, first, second = "meta:{s}", "terminal:{s}:a", "terminal:{s}:b"
        await redis.hset(meta, mapping={"status": "running", "turn_id": "a"})
        assert await hash_compare_and_set(
            redis,
            meta,
            "status",
            expected="running",
            new="failed",
            guard=("turn_id", "a"),
            updates={"error_message": "first failure"},
            record=(first, "first failure", 60),
        )
        assert 0 < await redis.ttl(first) <= 60
        await redis.expire(first, 10)
        await redis.hset(meta, mapping={"status": "running", "turn_id": "b"})
        assert not await hash_compare_and_set(
            redis,
            meta,
            "status",
            expected="running",
            new="failed",
            guard=("turn_id", "a"),
            updates={"error_message": "stale"},
            record=(first, "stale", 60),
        )
        assert await hash_compare_and_set(
            redis,
            meta,
            "status",
            expected="running",
            new="completed",
            guard=("turn_id", "b"),
            updates={"error_message": ""},
            record=(second, "done", 60),
        )
        assert 0 < await redis.ttl(first) <= 10
        assert 10 < await redis.ttl(second) <= 60
        assert await redis.get(first) == (
            "first failure" if decoded else b"first failure"
        )
        assert not await hash_compare_and_set(
            redis,
            meta,
            "status",
            expected="running",
            new="failed",
            guard=("turn_id", "b"),
            record=(second, "overwrite", 60),
        )
        assert await redis.get(second) == ("done" if decoded else b"done")
