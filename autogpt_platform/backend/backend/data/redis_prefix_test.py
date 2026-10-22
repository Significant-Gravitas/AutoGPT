"""Exercise the production prefix-pop Lua against the configured Redis cluster."""

from uuid import uuid4

import pytest
from redis.asyncio.cluster import RedisCluster
from redis.asyncio.retry import Retry
from redis.backoff import NoBackoff

from backend.data import redis_client
from backend.data.redis_helpers import pop_matching_list_prefix
from backend.util.testing import is_tcp_port_reachable


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
        key = f"test:prefix-pop:{{{uuid4().hex}}}"
        try:
            await client.rpush(key, "before", "image", "after")
            await client.pexpire(key, 60_000)
            yield client, key
        finally:
            await client.delete(key)


async def test_pop_prefix_preserves_appended_messages_and_expiry(queue):
    client, key = queue
    expires_at = await client.pexpiretime(key)
    await client.rpush(key, "appended")

    assert await pop_matching_list_prefix(client, key, ["before"]) == ["before"]
    assert await client.lrange(key, 0, -1) == ["image", "after", "appended"]
    assert await client.pexpiretime(key) == expires_at


async def test_pop_full_prefix_removes_the_list(queue):
    client, key = queue
    expected = ["before", "image", "after"]

    assert await pop_matching_list_prefix(client, key, expected) == expected
    assert await client.exists(key) == 0
    assert await client.ttl(key) == -2
    assert await pop_matching_list_prefix(client, key, expected) == []


@pytest.mark.parametrize(
    "expected",
    [[], ["changed"], ["before", "changed"], ["before", "image", "after", "extra"]],
)
async def test_mismatched_or_empty_prefix_preserves_the_queue(queue, expected):
    client, key = queue
    expires_at = await client.pexpiretime(key)

    assert await pop_matching_list_prefix(client, key, expected) == []
    assert await client.lrange(key, 0, -1) == ["before", "image", "after"]
    assert await client.pexpiretime(key) == expires_at
