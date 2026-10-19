"""Unit tests for the Redis part buffer.

Every other suite in this package replaces ``storage`` with ``AsyncMock``s,
so the ordering and accounting here — the two things standing between a
long recording and a silently corrupted or wrongly rejected upload — are
exercised against a fake hash rather than mocked away.
"""

from contextlib import asynccontextmanager
from datetime import datetime, timedelta, timezone
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import aiohttp
import pytest
from pytest_mock import MockerFixture
from redis.cluster import key_slot

from backend.api.features.onboarding_dump import storage
from backend.util.gcs_utils_test import _URL_CONTAINING_404, _gcs_http_error

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


def _handler(mocker: MockerFixture, bucket: str = "private-bucket") -> MagicMock:
    handler = MagicMock()
    handler.config.gcs_bucket_name = bucket
    handler.store_file = AsyncMock(return_value="gcs://private-bucket/x")
    mocker.patch.object(
        storage, "get_cloud_storage_handler", AsyncMock(return_value=handler)
    )
    return handler


@pytest.mark.asyncio
async def test_recordings_are_stored_marked_with_an_expiry(
    mocker: MockerFixture,
):
    handler = _handler(mocker)

    await storage.store_audio(USER_ID, b"audio", "brain-dump-rec-1.webm")

    kwargs = handler.store_file.call_args.kwargs
    # A day past the sweep, so the generic cleanup never races it.
    assert kwargs["expiration_hours"] == 48 + 24
    assert kwargs["delete_when_expired"] is True
    assert kwargs["custom_metadata"] == {
        storage.RECORDING_MARKER_KEY: storage.RECORDING_MARKER_VALUE
    }
    assert kwargs["user_id"] == USER_ID


def test_a_recording_due_just_after_one_sweep_goes_within_48_hours():
    """The sweep runs once per ``RECORDING_PURGE_INTERVAL``, so a recording
    just short of the max age at one run goes at the next, and that run
    gets half an hour to reach it."""
    assert storage.audio_max_age() == timedelta(hours=46, minutes=30)

    deleted_by = storage.audio_max_age() + storage.RECORDING_PURGE_INTERVAL

    assert deleted_by + timedelta(minutes=30) <= timedelta(hours=48)


NOW = datetime(2026, 10, 13, 12, 0, tzinfo=timezone.utc)
AUDIO_PATH = "gcs://private-bucket/uploads/users/user-1/u/brain-dump-rec-1.webm"


@pytest.mark.asyncio
async def test_audio_available_checks_the_configured_bucket(mocker: MockerFixture):
    _handler(mocker)
    created = mocker.patch.object(
        storage, "blob_created_at", AsyncMock(return_value=None)
    )

    assert await storage.audio_available(AUDIO_PATH, now=NOW) is False
    created.assert_awaited_once_with(
        "private-bucket", "uploads/users/user-1/u/brain-dump-rec-1.webm"
    )


@pytest.mark.asyncio
async def test_audio_due_for_deletion_is_no_longer_served(mocker: MockerFixture):
    """Still in the bucket until the next sweep, but already past the policy."""
    _handler(mocker)
    created = mocker.patch.object(storage, "blob_created_at", AsyncMock())

    created.return_value = NOW - storage.audio_max_age() + timedelta(minutes=1)
    assert await storage.audio_available(AUDIO_PATH, now=NOW) is True

    created.return_value = NOW - storage.audio_max_age()
    assert await storage.audio_available(AUDIO_PATH, now=NOW) is False


@pytest.mark.asyncio
async def test_audio_in_another_bucket_counts_as_gone(mocker: MockerFixture):
    _handler(mocker)
    created = mocker.patch.object(storage, "blob_created_at", AsyncMock())

    path = "gcs://legacy-bucket/uploads/users/user-1/u/brain-dump-rec-1.webm"
    assert await storage.audio_available(path) is False
    created.assert_not_awaited()


def _gcs_client(mocker: MockerFixture) -> MagicMock:
    client = MagicMock()
    client.delete = AsyncMock()

    @asynccontextmanager
    async def fresh_client():
        yield client

    mocker.patch.object(storage, "fresh_gcs_client", fresh_client)
    return client


@pytest.mark.asyncio
async def test_delete_audio_removes_the_recording_from_the_bucket(
    mocker: MockerFixture,
):
    _handler(mocker)
    client = _gcs_client(mocker)

    await storage.delete_audio(AUDIO_PATH)

    client.delete.assert_awaited_once_with(
        "private-bucket", "uploads/users/user-1/u/brain-dump-rec-1.webm"
    )


@pytest.mark.asyncio
async def test_delete_audio_counts_a_missing_recording_as_deleted(
    mocker: MockerFixture,
):
    _handler(mocker)
    client = _gcs_client(mocker)
    client.delete.side_effect = _gcs_http_error(404, _URL_CONTAINING_404)

    await storage.delete_audio(AUDIO_PATH)


@pytest.mark.asyncio
async def test_delete_audio_raises_any_other_failure(mocker: MockerFixture):
    """So the caller can log it and leave the file to the retention sweep."""
    _handler(mocker)
    client = _gcs_client(mocker)
    client.delete.side_effect = _gcs_http_error(503, _URL_CONTAINING_404)

    with pytest.raises(aiohttp.ClientResponseError):
        await storage.delete_audio(AUDIO_PATH)


@pytest.mark.asyncio
async def test_delete_audio_never_touches_another_bucket(mocker: MockerFixture):
    _handler(mocker)
    client = _gcs_client(mocker)

    path = "gcs://legacy-bucket/uploads/users/user-1/u/brain-dump-rec-1.webm"
    with pytest.raises(ValueError):
        await storage.delete_audio(path)
    client.delete.assert_not_awaited()
