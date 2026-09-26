"""Pin the semantics of the conftest's in-memory Redis stand-in.

The dream tests lean on ``FakeAsyncRedis`` for locks, markers, hit counters
and batch bookkeeping, so its behaviour on the calls the dream code makes
(NX/XX sets, TTLs, hashes and the two single-key lock scripts) is worth
asserting directly rather than only through the tests that happen to use
each call.
"""

import pytest

from .conftest import FakeAsyncRedis
from .locks import _EXTEND_SCRIPT, _UNLOCK_SCRIPT


@pytest.mark.asyncio
async def test_set_honours_nx_xx_and_ttl_arguments():
    redis = FakeAsyncRedis()
    assert await redis.set("k", "v1", nx=True, ex=30) is True
    assert await redis.set("k", "v2", nx=True) is None
    assert await redis.get("k") == "v1"
    assert await redis.ttl("k") == 30
    assert await redis.set("missing", "v", xx=True) is None
    assert await redis.set("k", "v3", xx=True, px=5000) is True
    assert await redis.get("k") == "v3"
    assert await redis.ttl("k") == 5
    assert await redis.set("b", b"bytes") is True
    assert await redis.get("b") == "bytes"


@pytest.mark.asyncio
async def test_delete_expire_exists_and_incr():
    redis = FakeAsyncRedis()
    assert await redis.expire("nope", 10) is False
    assert await redis.ttl("nope") == -2
    assert await redis.incr("counter") == 1
    assert await redis.incr("counter") == 2
    assert await redis.ttl("counter") == -1
    assert await redis.expire("counter", 60) is True
    assert await redis.exists("counter", "nope") == 1
    assert await redis.delete("counter", "nope") == 1
    assert await redis.exists("counter") == 0


@pytest.mark.asyncio
async def test_hash_commands():
    redis = FakeAsyncRedis()
    assert await redis.hset("h", "a", 1) == 1
    assert await redis.hset("h", mapping={"a": "x", "b": b"y"}) == 1
    assert await redis.hget("h", "a") == "x"
    assert await redis.hgetall("h") == {"a": "x", "b": "y"}
    assert await redis.hdel("h", "a", "zzz") == 1
    assert await redis.exists("h") == 1
    assert await redis.delete("h") == 1
    assert await redis.hgetall("h") == {}


@pytest.mark.asyncio
async def test_eval_models_the_lock_scripts():
    redis = FakeAsyncRedis()
    await redis.set("dream:inflight:u1", "token-a", nx=True, ex=1800)
    assert await redis.eval(_EXTEND_SCRIPT, 1, "dream:inflight:u1", "token-b", 60) == 0
    assert await redis.eval(_EXTEND_SCRIPT, 1, "dream:inflight:u1", "token-a", 60) == 1
    assert await redis.ttl("dream:inflight:u1") == 60
    assert await redis.eval(_UNLOCK_SCRIPT, 1, "dream:inflight:u1", "token-b") == 0
    assert await redis.eval(_UNLOCK_SCRIPT, 1, "dream:inflight:u1", "token-a") == 1
    assert await redis.get("dream:inflight:u1") is None
    with pytest.raises(NotImplementedError):
        await redis.eval("return 1", 2, "a", "b", "c")
    await redis.set("k", "t")
    with pytest.raises(NotImplementedError):
        await redis.eval('redis.call("incr", KEYS[1])', 1, "k", "t")
    await redis.aclose()
