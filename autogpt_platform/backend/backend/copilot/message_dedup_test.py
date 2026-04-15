"""Unit tests for backend.copilot.message_dedup."""

from unittest.mock import AsyncMock

import pytest
import pytest_mock

from backend.copilot.message_dedup import _KEY_PREFIX, acquire_dedup_lock


def _patch_redis(mocker: pytest_mock.MockerFixture, *, set_returns):
    mock_redis = AsyncMock()
    mock_redis.set = AsyncMock(return_value=set_returns)
    mocker.patch(
        "backend.copilot.message_dedup.get_redis_async",
        new_callable=AsyncMock,
        return_value=mock_redis,
    )
    return mock_redis


@pytest.mark.asyncio
async def test_acquire_returns_none_when_no_message_no_files(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Nothing to deduplicate — no Redis call made, None returned."""
    mock_redis = _patch_redis(mocker, set_returns=True)
    result = await acquire_dedup_lock("sess-1", None, None)
    assert result is None
    mock_redis.set.assert_not_called()


@pytest.mark.asyncio
async def test_acquire_returns_lock_on_first_request(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """First request acquires the lock and returns a _DedupLock."""
    mock_redis = _patch_redis(mocker, set_returns=True)
    lock = await acquire_dedup_lock("sess-1", "hello", None)
    assert lock is not None
    mock_redis.set.assert_called_once()
    key_arg = mock_redis.set.call_args.args[0]
    assert key_arg.startswith(f"{_KEY_PREFIX}:sess-1:")


@pytest.mark.asyncio
async def test_acquire_returns_none_on_duplicate(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """Duplicate request (NX fails) returns None to signal the caller."""
    _patch_redis(mocker, set_returns=None)
    result = await acquire_dedup_lock("sess-1", "hello", None)
    assert result is None


@pytest.mark.asyncio
async def test_acquire_key_stable_across_file_order(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """File IDs are sorted before hashing so order doesn't affect the key."""
    mock_redis_1 = _patch_redis(mocker, set_returns=True)
    await acquire_dedup_lock("sess-1", "msg", ["b", "a"])
    key_ab = mock_redis_1.set.call_args.args[0]

    mock_redis_2 = _patch_redis(mocker, set_returns=True)
    await acquire_dedup_lock("sess-1", "msg", ["a", "b"])
    key_ba = mock_redis_2.set.call_args.args[0]

    assert key_ab == key_ba


@pytest.mark.asyncio
async def test_release_deletes_key(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """release() calls Redis delete exactly once."""
    mock_redis = _patch_redis(mocker, set_returns=True)
    lock = await acquire_dedup_lock("sess-1", "hello", None)
    assert lock is not None
    await lock.release()
    mock_redis.delete.assert_called_once()


@pytest.mark.asyncio
async def test_release_swallows_redis_error(
    mocker: pytest_mock.MockerFixture,
) -> None:
    """release() must not raise even when Redis delete fails."""
    mock_redis = _patch_redis(mocker, set_returns=True)
    mock_redis.delete = AsyncMock(side_effect=RuntimeError("redis down"))
    lock = await acquire_dedup_lock("sess-1", "hello", None)
    assert lock is not None
    await lock.release()  # must not raise
    mock_redis.delete.assert_called_once()
