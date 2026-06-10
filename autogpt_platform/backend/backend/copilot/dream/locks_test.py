"""Redis SETNX lock for the dream pass — token-owned compare-and-delete."""

from __future__ import annotations

import uuid
from unittest.mock import AsyncMock

import pytest

from .locks import (
    BATCH_LOCK_TTL_SECONDS,
    DEFAULT_LOCK_TTL_SECONDS,
    DreamLockHeld,
    dream_lock,
    read_dream_lock_token,
    release_dream_lock,
)


def _redis_mock(**overrides) -> AsyncMock:
    redis = AsyncMock()
    redis.set = AsyncMock(return_value=True)
    redis.eval = AsyncMock(return_value=1)
    redis.delete = AsyncMock(return_value=1)
    for name, value in overrides.items():
        setattr(redis, name, value)
    return redis


def _patch_redis(mocker, redis: AsyncMock) -> None:
    mocker.patch(
        "backend.data.redis_client.get_redis_async",
        AsyncMock(return_value=redis),
    )


@pytest.mark.asyncio
async def test_dream_lock_stores_uuid_token_and_releases_via_compare_and_delete(
    mocker,
):
    redis = _redis_mock()
    _patch_redis(mocker, redis)

    async with dream_lock("user-a", ttl_seconds=DEFAULT_LOCK_TTL_SECONDS) as handle:
        token = handle.token

    # Acquire: SETNX with a uuid4 ownership token as the value, not "1".
    redis.set.assert_awaited_once()
    args, kwargs = redis.set.call_args
    assert args[0] == "dream:inflight:user-a"
    assert args[1] == token
    assert uuid.UUID(token)  # parseable uuid, unique per acquire
    assert kwargs == {"nx": True, "ex": DEFAULT_LOCK_TTL_SECONDS}
    # Release: single-key Lua compare-and-delete on OUR token — never a
    # blind DEL that could take out a newer pass's lock.
    redis.eval.assert_awaited_once()
    eval_args = redis.eval.call_args.args
    assert eval_args[1] == 1  # single key — routes on Redis Cluster
    assert eval_args[2] == "dream:inflight:user-a"
    assert eval_args[3] == token
    redis.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_dream_lock_raises_when_already_held(mocker):
    redis = _redis_mock(set=AsyncMock(return_value=False))  # SETNX lost
    _patch_redis(mocker, redis)

    with pytest.raises(DreamLockHeld):
        async with dream_lock("user-b"):
            pytest.fail("should not enter the body when lock is held")

    # And critically: we don't try to delete a key that wasn't ours.
    redis.eval.assert_not_awaited()
    redis.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_dream_lock_swallows_release_failure(mocker, caplog):
    redis = _redis_mock(eval=AsyncMock(side_effect=Exception("redis down")))
    _patch_redis(mocker, redis)

    # The TTL is the fallback release, so a release failure must not
    # propagate up to the orchestrator.
    async with dream_lock("user-c"):
        pass

    assert "Failed to release dream lock" in caplog.text


@pytest.mark.asyncio
async def test_release_on_exit_skips_delete_when_token_mismatch(mocker, caplog):
    """A late exit (pass outlived its TTL, key re-acquired by a newer pass)
    must leave the new holder's lock alone — the Lua compare returns 0."""
    redis = _redis_mock(eval=AsyncMock(return_value=0))
    _patch_redis(mocker, redis)

    async with dream_lock("user-d"):
        pass

    redis.eval.assert_awaited_once()
    redis.delete.assert_not_awaited()
    assert "no longer held our token" in caplog.text


@pytest.mark.asyncio
async def test_dream_lock_disown_skips_release_and_extends_ttl(mocker):
    """The batch path extends the TTL to the batch window and disowns the
    lock, so the context manager must NOT delete it on exit — the batch
    callback (or the TTL) owns release."""
    redis = _redis_mock()
    _patch_redis(mocker, redis)

    async with dream_lock("user-e") as handle:
        await handle.extend(BATCH_LOCK_TTL_SECONDS)
        handle.disown()

    # extend re-asserts OUR token with SET XX (only-if-exists) — an
    # expired lock is never resurrected, unlike a plain SET would.
    assert redis.set.await_count == 2
    extend_args, extend_kwargs = redis.set.call_args
    assert extend_args == ("dream:inflight:user-e", handle.token)
    assert extend_kwargs == {"xx": True, "ex": BATCH_LOCK_TTL_SECONDS}
    redis.eval.assert_not_awaited()
    redis.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_extend_warns_when_lock_already_expired(mocker, caplog):
    """SET XX returns nil when the key expired — extend must surface that
    (ownership is lost) instead of silently recreating the lock."""
    redis = _redis_mock(set=AsyncMock(side_effect=[True, None]))
    _patch_redis(mocker, redis)

    async with dream_lock("user-f") as handle:
        await handle.extend(BATCH_LOCK_TTL_SECONDS)
        handle.disown()

    assert "expired before extend" in caplog.text


@pytest.mark.asyncio
async def test_release_dream_lock_compare_and_deletes_with_token(mocker):
    redis = _redis_mock()
    _patch_redis(mocker, redis)

    await release_dream_lock("user-g", "tok-g")

    redis.eval.assert_awaited_once()
    eval_args = redis.eval.call_args.args
    assert eval_args[1] == 1
    assert eval_args[2] == "dream:inflight:user-g"
    assert eval_args[3] == "tok-g"
    redis.delete.assert_not_awaited()


@pytest.mark.asyncio
async def test_release_skips_delete_when_token_mismatch(mocker, caplog):
    """A batch callback landing after the lock expired and was re-acquired
    must not release the new holder's lock — the compare fails and the key
    is left alone."""
    redis = _redis_mock(eval=AsyncMock(return_value=0))
    _patch_redis(mocker, redis)

    await release_dream_lock("user-h", "stale-token")

    redis.eval.assert_awaited_once()
    redis.delete.assert_not_awaited()
    assert "no longer held our token" in caplog.text


@pytest.mark.asyncio
async def test_release_without_token_leaves_lock_for_ttl(mocker, caplog):
    """When the token is unknown (input bundle TTL'd out / corrupted) the
    release must not blind-delete — the TTL clears the key instead."""
    redis = _redis_mock()
    _patch_redis(mocker, redis)

    await release_dream_lock("user-i", None)

    redis.eval.assert_not_awaited()
    redis.delete.assert_not_awaited()
    assert "leaving it for the TTL" in caplog.text


@pytest.mark.asyncio
async def test_release_dream_lock_swallows_redis_failure(mocker, caplog):
    redis = _redis_mock(eval=AsyncMock(side_effect=Exception("redis down")))
    _patch_redis(mocker, redis)

    await release_dream_lock("user-j", "tok-j")

    assert "Failed to release disowned dream lock" in caplog.text


@pytest.mark.asyncio
async def test_read_dream_lock_token_decodes_current_holder(mocker):
    redis = _redis_mock(get=AsyncMock(return_value=b"tok-bytes"))
    _patch_redis(mocker, redis)

    assert await read_dream_lock_token("user-k") == "tok-bytes"
    redis.get.assert_awaited_once_with("dream:inflight:user-k")


@pytest.mark.asyncio
async def test_read_dream_lock_token_none_when_unheld(mocker):
    redis = _redis_mock(get=AsyncMock(return_value=None))
    _patch_redis(mocker, redis)

    assert await read_dream_lock_token("user-l") is None
