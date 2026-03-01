"""Tests for e2b_sandbox: get_or_create_sandbox, _try_reconnect, kill_sandbox.

Uses mock Redis and mock AsyncSandbox — no external dependencies.
Tests are synchronous (using asyncio.run) to avoid conflicts with the
session-scoped event loop in conftest.py.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .e2b_sandbox import (
    _CREATING,
    _SANDBOX_REDIS_PREFIX,
    _try_reconnect,
    get_or_create_sandbox,
    kill_sandbox,
)

_KEY = f"{_SANDBOX_REDIS_PREFIX}sess-123"
_API_KEY = "test-api-key"
_TIMEOUT = 300


def _mock_sandbox(sandbox_id: str = "sb-abc", running: bool = True) -> MagicMock:
    sb = MagicMock()
    sb.sandbox_id = sandbox_id
    sb.is_running = AsyncMock(return_value=running)
    return sb


def _mock_redis(get_val: str | bytes | None = None, set_nx_result: bool = True):
    r = AsyncMock()
    r.get = AsyncMock(return_value=get_val)
    r.set = AsyncMock(return_value=set_nx_result)
    r.setex = AsyncMock()
    r.delete = AsyncMock()
    r.expire = AsyncMock()
    return r


def _patch_redis(redis):
    return patch(
        "backend.copilot.tools.e2b_sandbox.get_redis_async",
        new_callable=AsyncMock,
        return_value=redis,
    )


# ---------------------------------------------------------------------------
# _try_reconnect
# ---------------------------------------------------------------------------


class TestTryReconnect:
    def test_reconnect_success(self):
        sb = _mock_sandbox()
        redis = _mock_redis()
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(_try_reconnect("sb-abc", _API_KEY, _KEY, _TIMEOUT))

        assert result is sb
        redis.expire.assert_awaited_once_with(_KEY, _TIMEOUT)
        redis.delete.assert_not_awaited()

    def test_reconnect_not_running_clears_key(self):
        sb = _mock_sandbox(running=False)
        redis = _mock_redis()
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(_try_reconnect("sb-abc", _API_KEY, _KEY, _TIMEOUT))

        assert result is None
        redis.delete.assert_awaited_once_with(_KEY)

    def test_reconnect_exception_clears_key(self):
        redis = _mock_redis()
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(side_effect=ConnectionError("gone"))
            result = asyncio.run(_try_reconnect("sb-abc", _API_KEY, _KEY, _TIMEOUT))

        assert result is None
        redis.delete.assert_awaited_once_with(_KEY)


# ---------------------------------------------------------------------------
# get_or_create_sandbox
# ---------------------------------------------------------------------------


class TestGetOrCreateSandbox:
    def test_reconnect_existing(self):
        """When Redis has a valid sandbox_id, reconnect to it."""
        sb = _mock_sandbox()
        redis = _mock_redis(get_val="sb-abc")
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(
                get_or_create_sandbox("sess-123", _API_KEY, timeout=_TIMEOUT)
            )

        assert result is sb
        mock_cls.create.assert_not_called()

    def test_create_new_when_no_key(self):
        """When Redis is empty, claim lock and create a new sandbox."""
        sb = _mock_sandbox("sb-new")
        redis = _mock_redis(get_val=None, set_nx_result=True)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.create = AsyncMock(return_value=sb)
            result = asyncio.run(
                get_or_create_sandbox("sess-123", _API_KEY, timeout=_TIMEOUT)
            )

        assert result is sb
        redis.setex.assert_awaited_once_with(_KEY, _TIMEOUT, "sb-new")

    def test_create_failure_clears_lock(self):
        """If sandbox creation fails, the Redis lock is deleted."""
        redis = _mock_redis(get_val=None, set_nx_result=True)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.create = AsyncMock(side_effect=RuntimeError("quota"))
            with pytest.raises(RuntimeError, match="quota"):
                asyncio.run(
                    get_or_create_sandbox("sess-123", _API_KEY, timeout=_TIMEOUT)
                )

        redis.delete.assert_awaited_once_with(_KEY)

    def test_wait_for_lock_then_reconnect(self):
        """When another process holds the lock, wait and reconnect."""
        sb = _mock_sandbox("sb-other")
        redis = _mock_redis()
        redis.get = AsyncMock(side_effect=[_CREATING, "sb-other"])
        redis.set = AsyncMock(return_value=False)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
            patch(
                "backend.copilot.tools.e2b_sandbox.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(
                get_or_create_sandbox("sess-123", _API_KEY, timeout=_TIMEOUT)
            )

        assert result is sb

    def test_stale_reconnect_clears_and_creates(self):
        """When stored sandbox is stale, clear key and create a new one."""
        stale_sb = _mock_sandbox("sb-stale", running=False)
        new_sb = _mock_sandbox("sb-fresh")
        redis = _mock_redis(get_val="sb-stale", set_nx_result=True)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=stale_sb)
            mock_cls.create = AsyncMock(return_value=new_sb)
            result = asyncio.run(
                get_or_create_sandbox("sess-123", _API_KEY, timeout=_TIMEOUT)
            )

        assert result is new_sb
        redis.delete.assert_awaited()


# ---------------------------------------------------------------------------
# kill_sandbox
# ---------------------------------------------------------------------------


class TestKillSandbox:
    def test_kill_existing_sandbox(self):
        """Kill a running sandbox and clean up Redis."""
        sb = _mock_sandbox()
        sb.kill = AsyncMock()
        redis = _mock_redis(get_val="sb-abc")
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(kill_sandbox("sess-123", _API_KEY))

        assert result is True
        redis.delete.assert_awaited_once_with(_KEY)
        sb.kill.assert_awaited_once()

    def test_kill_no_sandbox(self):
        """No-op when no sandbox exists in Redis."""
        redis = _mock_redis(get_val=None)
        with _patch_redis(redis):
            result = asyncio.run(kill_sandbox("sess-123", _API_KEY))

        assert result is False
        redis.delete.assert_not_awaited()

    def test_kill_creating_state(self):
        """Clears Redis key but returns False when sandbox is still being created."""
        redis = _mock_redis(get_val=_CREATING)
        with _patch_redis(redis):
            result = asyncio.run(kill_sandbox("sess-123", _API_KEY))

        assert result is False
        redis.delete.assert_awaited_once_with(_KEY)

    def test_kill_connect_failure(self):
        """Returns False and cleans Redis if connect/kill fails."""
        redis = _mock_redis(get_val="sb-abc")
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(side_effect=ConnectionError("gone"))
            result = asyncio.run(kill_sandbox("sess-123", _API_KEY))

        assert result is False
        redis.delete.assert_awaited_once_with(_KEY)

    def test_kill_with_bytes_redis_value(self):
        """Redis may return bytes — kill_sandbox should decode correctly."""
        sb = _mock_sandbox()
        sb.kill = AsyncMock()
        redis = _mock_redis(get_val=b"sb-abc")
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(kill_sandbox("sess-123", _API_KEY))

        assert result is True
        sb.kill.assert_awaited_once()
