"""Unit tests for the forget stash (``recall_stash``) on the in-memory Redis
every graphiti test gets (``conftest.forget_stash``).

The live cross-process run is ``recall_inflight_integration_test.py``.
"""

import asyncio
from datetime import datetime, timedelta, timezone
from unittest.mock import AsyncMock, patch

import pytest

from . import recall_stash
from .recall_fake_redis import FakeRedis
from .recall_stash import STASH_TTL, ForgetRecord, read_forgets, stash_forgets
from .scope import forget_stash_key

_GROUP = "user_abc"


def _record(uuid: str = "e1", *, age: timedelta = timedelta()) -> ForgetRecord:
    when = (datetime.now(timezone.utc) - age).isoformat()
    return ForgetRecord(
        uuid=uuid,
        forgotten_at=when,
        expired_at=when,
        fact_redacted="Alice works on Atlas",
        stashed_at=when,
    )


class TestStash:
    @pytest.mark.asyncio
    async def test_a_record_written_is_read_back_by_its_edge(
        self, forget_stash: FakeRedis
    ) -> None:
        record = _record()

        assert await stash_forgets(_GROUP, [record])

        assert await read_forgets(_GROUP) == {"e1": record}
        assert set(forget_stash.hashes) == {forget_stash_key(_GROUP)}

    @pytest.mark.asyncio
    async def test_a_newer_record_of_the_same_edge_replaces_the_older(self) -> None:
        await stash_forgets(_GROUP, [_record()])
        newer = _record().model_copy(update={"hard": True})

        await stash_forgets(_GROUP, [newer])

        assert (await read_forgets(_GROUP))["e1"].hard

    @pytest.mark.asyncio
    async def test_graphs_do_not_share_a_stash(self) -> None:
        await stash_forgets(_GROUP, [_record()])

        assert await read_forgets("user_other") == {}

    @pytest.mark.asyncio
    async def test_stale_and_unreadable_records_are_dropped(
        self, forget_stash: FakeRedis
    ) -> None:
        await stash_forgets(
            _GROUP, [_record("fresh"), _record("stale", age=STASH_TTL * 2)]
        )
        forget_stash.hashes[forget_stash_key(_GROUP)]["broken"] = "{not json"

        assert list(await read_forgets(_GROUP)) == ["fresh"]
        assert list(forget_stash.hashes[forget_stash_key(_GROUP)]) == ["fresh"]

    @pytest.mark.asyncio
    async def test_nothing_to_stash_touches_no_redis(self) -> None:
        connect = AsyncMock()
        with patch.object(recall_stash, "get_redis_async", connect):
            assert await stash_forgets(_GROUP, [])

        connect.assert_not_awaited()


class TestRedisDown:
    @pytest.mark.asyncio
    async def test_a_failed_write_is_reported_not_raised(self) -> None:
        down = AsyncMock(side_effect=ConnectionError("redis down"))
        with patch.object(recall_stash, "get_redis_async", down):
            assert await stash_forgets(_GROUP, [_record()]) is False

    @pytest.mark.asyncio
    async def test_a_failed_read_is_an_empty_stash(self) -> None:
        down = AsyncMock(side_effect=ConnectionError("redis down"))
        with patch.object(recall_stash, "get_redis_async", down):
            assert await read_forgets(_GROUP) == {}

    @pytest.mark.asyncio
    async def test_a_slow_redis_is_not_waited_on(self) -> None:
        """``get_redis_async`` retries a lost connection for minutes."""

        async def hang() -> FakeRedis:
            await asyncio.sleep(10)
            return FakeRedis()

        with (
            patch.object(recall_stash, "get_redis_async", hang),
            patch.object(recall_stash, "_REDIS_TIMEOUT_SECONDS", 0.05),
        ):
            assert await stash_forgets(_GROUP, [_record()]) is False
            assert await read_forgets(_GROUP) == {}
