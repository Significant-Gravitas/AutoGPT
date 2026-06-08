"""Redis SETNX lock for the dream pass."""

from __future__ import annotations

from unittest.mock import AsyncMock

import pytest

from .locks import (
    BATCH_LOCK_TTL_SECONDS,
    DEFAULT_LOCK_TTL_SECONDS,
    DreamLockHeld,
    dream_lock,
    release_dream_lock,
)


@pytest.mark.asyncio
async def test_dream_lock_releases_on_exit(mocker):
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)  # SETNX wins
    redis.delete = AsyncMock(return_value=1)
    mocker.patch(
        "backend.data.redis_client.get_redis_async",
        AsyncMock(return_value=redis),
    )

    async with dream_lock("user-a", ttl_seconds=DEFAULT_LOCK_TTL_SECONDS):
        pass

    redis.set.assert_awaited_once()
    args, kwargs = redis.set.call_args
    assert args[0] == "dream:inflight:user-a"
    assert kwargs == {"nx": True, "ex": DEFAULT_LOCK_TTL_SECONDS}
    redis.delete.assert_awaited_once_with("dream:inflight:user-a")


@pytest.mark.asyncio
async def test_dream_lock_raises_when_already_held(mocker):
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=False)  # SETNX lost — someone else holds
    redis.delete = AsyncMock()
    mocker.patch(
        "backend.data.redis_client.get_redis_async",
        AsyncMock(return_value=redis),
    )

    with pytest.raises(DreamLockHeld):
        async with dream_lock("user-b"):
            pytest.fail("should not enter the body when lock is held")

    # And critically: we don't try to delete a key that wasn't ours.
    redis.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_dream_lock_swallows_delete_failure(mocker, caplog):
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(side_effect=Exception("redis down"))
    mocker.patch(
        "backend.data.redis_client.get_redis_async",
        AsyncMock(return_value=redis),
    )

    # The TTL is the fallback release, so a delete failure must not
    # propagate up to the orchestrator.
    async with dream_lock("user-c"):
        pass

    assert "Failed to release dream lock" in caplog.text


@pytest.mark.asyncio
async def test_dream_lock_disown_skips_release_and_extends_ttl(mocker):
    """The batch path extends the TTL to the batch window and disowns the
    lock, so the context manager must NOT delete it on exit — the batch
    callback (or the TTL) owns release."""
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    redis.delete = AsyncMock(return_value=1)
    redis.expire = AsyncMock(return_value=True)
    mocker.patch(
        "backend.data.redis_client.get_redis_async",
        AsyncMock(return_value=redis),
    )

    async with dream_lock("user-d") as handle:
        await handle.extend(BATCH_LOCK_TTL_SECONDS)
        handle.disown()

    redis.expire.assert_awaited_once_with(
        "dream:inflight:user-d", BATCH_LOCK_TTL_SECONDS
    )
    redis.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_dream_lock_deletes_key(mocker):
    redis = AsyncMock()
    redis.delete = AsyncMock(return_value=1)
    mocker.patch(
        "backend.data.redis_client.get_redis_async",
        AsyncMock(return_value=redis),
    )

    await release_dream_lock("user-e")
    redis.delete.assert_awaited_once_with("dream:inflight:user-e")
