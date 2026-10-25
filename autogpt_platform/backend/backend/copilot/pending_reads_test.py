"""Exercise the pending queue's pop and take Lua against the configured Redis cluster."""

import json
from uuid import uuid4

import pytest
from redis.asyncio.cluster import RedisCluster
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from backend.copilot.pending_reads import (
    READ_TTL_SECONDS,
    pop_recording_reads,
    put_back_unread,
    take_recording_read,
)
from backend.data import redis_client
from backend.util.testing import is_tcp_port_reachable

BEFORE, IMAGE, AFTER = (json.dumps({"id": i, "content": i}) for i in ("a", "b", "c"))


@pytest.fixture
async def queue():
    if not is_tcp_port_reachable(redis_client.HOST, redis_client.PORT):
        pytest.skip("local Redis cluster not reachable")
    async with RedisCluster(
        host=redis_client.HOST,
        port=redis_client.PORT,
        password=redis_client.PASSWORD,
        address_remap=redis_client._address_remap,
        decode_responses=True,
        socket_timeout=2,
        socket_connect_timeout=2,
        retry=Retry(NoBackoff(), 0),
    ) as client:
        tag = uuid4().hex
        keys = {
            "queue_key": f"test:pending:{{{tag}}}",
            "read_key": f"test:pending:read:{{{tag}}}",
            "held_key": f"test:pending:held:{{{tag}}}",
        }
        try:
            await client.rpush(keys["queue_key"], BEFORE, IMAGE, AFTER)
            await client.pexpire(keys["queue_key"], 60_000)
            await client.set(keys["held_key"], "error|0")
            yield client, keys
        finally:
            await client.delete(*keys.values())


async def _read(client, keys) -> set[str]:
    return await client.smembers(keys["read_key"])


async def test_a_drain_records_every_id_it_pops_and_ends_the_hold(queue):
    client, keys = queue
    await client.rpush(keys["queue_key"], "not json")

    popped = await pop_recording_reads(client, **keys, count=10)

    assert popped == [BEFORE, IMAGE, AFTER, "not json"]
    assert await client.exists(keys["queue_key"]) == 0
    assert await _read(client, keys) == {"a", "b", "c"}
    assert 0 < await client.ttl(keys["read_key"]) <= READ_TTL_SECONDS
    assert await client.exists(keys["held_key"]) == 0


async def test_a_prefix_pop_keeps_appended_messages_and_expiry(queue):
    client, keys = queue
    expires_at = await client.pexpiretime(keys["queue_key"])
    await client.rpush(keys["queue_key"], "appended")

    popped = await pop_recording_reads(client, **keys, count=10, prefix=[BEFORE])

    assert popped == [BEFORE]
    assert await client.lrange(keys["queue_key"], 0, -1) == [IMAGE, AFTER, "appended"]
    assert await client.pexpiretime(keys["queue_key"]) == expires_at
    assert await _read(client, keys) == {"a"}


@pytest.mark.parametrize(
    "prefix", [[], ["changed"], [BEFORE, "changed"], [BEFORE, IMAGE, AFTER, "extra"]]
)
async def test_a_stale_or_empty_prefix_pops_and_records_nothing(queue, prefix):
    client, keys = queue
    expires_at = await client.pexpiretime(keys["queue_key"])

    assert await pop_recording_reads(client, **keys, count=10, prefix=prefix) == []
    assert await client.lrange(keys["queue_key"], 0, -1) == [BEFORE, IMAGE, AFTER]
    assert await client.pexpiretime(keys["queue_key"]) == expires_at
    assert await _read(client, keys) == set()
    assert await client.exists(keys["held_key"]) == 1


async def test_an_empty_queue_pops_nothing_and_keeps_the_hold(queue):
    client, keys = queue
    await client.delete(keys["queue_key"])

    assert await pop_recording_reads(client, **keys, count=10) == []
    assert await client.exists(keys["held_key"]) == 1


async def test_a_take_removes_that_entry_once_and_records_it(queue):
    client, keys = queue
    take = {"queue_key": keys["queue_key"], "read_key": keys["read_key"]}

    assert await take_recording_read(client, **take, payload=IMAGE, pending_id="b")
    assert not await take_recording_read(client, **take, payload=IMAGE, pending_id="b")
    assert await client.lrange(keys["queue_key"], 0, -1) == [BEFORE, AFTER]
    assert await _read(client, keys) == {"b"}


async def test_a_put_back_queues_it_again_unread(queue):
    client, keys = queue
    take = {"queue_key": keys["queue_key"], "read_key": keys["read_key"]}
    assert await take_recording_read(client, **take, payload=IMAGE, pending_id="b")

    length = await put_back_unread(
        client,
        **take,
        payload=IMAGE,
        pending_id="b",
        enqueued_at=0.0,
        ttl_seconds=60,
    )

    assert length == 3
    assert await client.lrange(keys["queue_key"], 0, -1) == [BEFORE, AFTER, IMAGE]
    assert await _read(client, keys) == set()
    assert 0 < await client.ttl(keys["queue_key"]) <= 60


async def test_a_put_back_returns_it_to_its_place_by_enqueue_time(queue):
    client, keys = queue
    take = {"queue_key": keys["queue_key"], "read_key": keys["read_key"]}
    timed = [
        json.dumps({"id": i, "enqueued_at": t})
        for i, t in (("x", 1.0), ("y", 2.0), ("z", 3.0))
    ]
    await client.delete(keys["queue_key"])
    await client.rpush(keys["queue_key"], *timed)
    assert await take_recording_read(client, **take, payload=timed[1], pending_id="y")

    await put_back_unread(
        client,
        **take,
        payload=timed[1],
        pending_id="y",
        enqueued_at=2.0,
        ttl_seconds=60,
    )

    assert await client.lrange(keys["queue_key"], 0, -1) == timed


async def test_a_put_back_into_a_full_queue_evicts_nothing(queue):
    client, keys = queue
    take = {"queue_key": keys["queue_key"], "read_key": keys["read_key"]}
    full = [json.dumps({"id": f"m{n}", "enqueued_at": float(n)}) for n in range(10)]
    await client.delete(keys["queue_key"])
    await client.rpush(keys["queue_key"], *full)
    assert await take_recording_read(client, **take, payload=full[0], pending_id="m0")
    late = json.dumps({"id": "late", "enqueued_at": 99.0})
    await client.rpush(keys["queue_key"], late)

    length = await put_back_unread(
        client,
        **take,
        payload=full[0],
        pending_id="m0",
        enqueued_at=0.0,
        ttl_seconds=60,
    )

    assert length == 11
    assert await client.lrange(keys["queue_key"], 0, -1) == [*full, late]
