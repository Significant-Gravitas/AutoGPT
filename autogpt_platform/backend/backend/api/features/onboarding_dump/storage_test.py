"""Unit tests for the Redis part buffer.

Every other suite in this package replaces ``storage`` with ``AsyncMock``s,
so the ordering and accounting here — the two things standing between a
long recording and a silently corrupted or wrongly rejected upload — are
exercised against a fake hash rather than mocked away.
"""

from typing import Any

import pytest
from pytest_mock import MockerFixture
from redis.cluster import key_slot

from backend.api.features.onboarding_dump import storage

USER_ID = "user-1"
RECORDING_ID = "rec-1"


class FakeRedis:
    """The handful of hash commands the buffer uses."""

    def __init__(self) -> None:
        self.hashes: dict[str, dict[str, str]] = {}
        self.expiries: dict[str, int] = {}

    async def hgetall(self, key: str) -> dict[str, str]:
        return dict(self.hashes.get(key, {}))

    async def hvals(self, key: str) -> list[str]:
        return list(self.hashes.get(key, {}).values())

    async def hset(self, key: str, field: str, value: str) -> int:
        entries = self.hashes.setdefault(key, {})
        added = field not in entries
        entries[field] = value
        return int(added)

    async def expire(self, key: str, seconds: int) -> None:
        self.expiries[key] = seconds

    async def delete(self, *keys: str) -> None:
        for key in keys:
            self.hashes.pop(key, None)
            self.expiries.pop(key, None)

    def pipeline(self, transaction: bool = True) -> "FakePipeline":
        return FakePipeline(self)


class FakePipeline:
    """Queues commands like redis-py: sync on the way in, awaited on exec."""

    def __init__(self, redis: FakeRedis) -> None:
        self.redis = redis
        self.queued: list[tuple[str, tuple[Any, ...]]] = []

    async def __aenter__(self) -> "FakePipeline":
        return self

    async def __aexit__(self, *exc_info: object) -> None:
        return None

    def hset(self, *args: Any) -> None:
        self.queued.append(("hset", args))

    def expire(self, *args: Any) -> None:
        self.queued.append(("expire", args))

    def hvals(self, *args: Any) -> None:
        self.queued.append(("hvals", args))

    async def execute(self) -> list[Any]:
        return [await getattr(self.redis, name)(*args) for name, args in self.queued]


@pytest.fixture
def redis(mocker: MockerFixture) -> FakeRedis:
    fake = FakeRedis()

    async def get_redis_async() -> FakeRedis:
        return fake

    mocker.patch.object(storage.redis_client, "get_redis_async", new=get_redis_async)
    return fake


async def _append(redis: FakeRedis, part_index: int, content: bytes) -> int:
    return await storage.append_part(USER_ID, RECORDING_ID, part_index, content)


def test_buffer_keys_share_a_redis_cluster_slot():
    for user_id, recording_id in [
        (USER_ID, RECORDING_ID),
        ("c0933014-3c5a-499a-91db-10e020d527b1", "recording_42"),
    ]:
        parts_key = storage._parts_key(user_id, recording_id)
        sizes_key = storage._sizes_key(user_id, recording_id)

        assert key_slot(parts_key.encode()) == key_slot(sizes_key.encode())


@pytest.mark.asyncio
async def test_parts_are_assembled_in_numeric_not_lexicographic_order(
    redis: FakeRedis,
):
    """Redis hash fields are strings, so ``"10" < "2"`` when sorted raw.

    A recording longer than ten timeslices would then concatenate as
    1, 10, 11, 2, … — a stream that is still valid enough to store and
    transcribe, and completely scrambled.
    """
    for index in range(12):
        await _append(redis, index, f"<{index}>".encode())

    assembled = await storage.assemble_parts(USER_ID, RECORDING_ID)

    assert assembled == b"".join(f"<{index}>".encode() for index in range(12))


@pytest.mark.asyncio
async def test_an_empty_buffer_assembles_to_nothing(redis: FakeRedis):
    assert await storage.assemble_parts(USER_ID, RECORDING_ID) == b""
    assert await storage.buffered_size(USER_ID, RECORDING_ID) == 0


@pytest.mark.asyncio
@pytest.mark.parametrize("length", [3, 4, 5, 300, 301, 302])
async def test_the_running_total_is_the_exact_decoded_byte_count(
    redis: FakeRedis, length: int
):
    """Base64 padding varies with ``len(payload) % 3``; the cap does not."""
    content = b"x" * length

    total = await _append(redis, 0, content)

    assert total == length
    assert await storage.buffered_size(USER_ID, RECORDING_ID) == length
    assert await storage.assemble_parts(USER_ID, RECORDING_ID) == content


@pytest.mark.asyncio
async def test_replaying_a_part_overwrites_it_instead_of_double_counting(
    redis: FakeRedis,
):
    """The client's retry queue replays parts it isn't sure landed."""
    await _append(redis, 0, b"a" * 10)
    await _append(redis, 1, b"b" * 20)

    total = await _append(redis, 1, b"b" * 20)

    assert total == 30
    assert await storage.buffered_size(USER_ID, RECORDING_ID) == 30


@pytest.mark.asyncio
async def test_the_buffer_and_its_size_index_expire_and_are_dropped_together(
    redis: FakeRedis,
):
    await _append(redis, 0, b"chunk")
    assert set(redis.expiries.values()) == {storage.PART_BUFFER_TTL_SECONDS}

    await storage.discard_parts(USER_ID, RECORDING_ID)

    assert redis.hashes == {}
    assert await storage.buffered_size(USER_ID, RECORDING_ID) == 0
