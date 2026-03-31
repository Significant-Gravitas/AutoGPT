"""Tests for e2b_sandbox: get_or_create_sandbox, _try_reconnect, kill_sandbox.

sandbox_id is stored in Redis under _SANDBOX_KEY_PREFIX + session_id.
The same key doubles as a creation lock via a "creating" sentinel value.

Tests mock:
- ``get_redis_async`` (sandbox key storage + creation lock sentinel)
- ``AsyncSandbox`` (E2B SDK)

Tests are synchronous (using asyncio.run) to avoid conflicts with the
session-scoped event loop in conftest.py.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from .e2b_sandbox import (
    _CREATING_SENTINEL,
    _try_reconnect,
    get_or_create_sandbox,
    kill_sandbox,
    pause_sandbox,
    pause_sandbox_direct,
)

_SESSION_ID = "sess-123"
_API_KEY = "test-api-key"
_SANDBOX_ID = "sb-abc"
_TIMEOUT = 300


def _mock_sandbox(sandbox_id: str = _SANDBOX_ID, running: bool = True) -> MagicMock:
    sb = MagicMock()
    sb.sandbox_id = sandbox_id
    sb.is_running = AsyncMock(return_value=running)
    sb.pause = AsyncMock()
    sb.kill = AsyncMock()
    return sb


def _mock_redis(
    set_nx_result: bool = True,
    stored_sandbox_id: str | None = None,
) -> AsyncMock:
    """Create a mock redis client.

    *stored_sandbox_id* is returned by ``get()`` calls (simulates the sandbox_id
    stored under the ``_SANDBOX_KEY_PREFIX`` key).  ``set_nx_result`` controls
    whether the creation-slot ``SET NX`` succeeds.

    If *stored_sandbox_id* is None the key is absent (no sandbox, no lock).
    """
    r = AsyncMock()
    raw = stored_sandbox_id.encode() if stored_sandbox_id else None
    r.get = AsyncMock(return_value=raw)
    r.set = AsyncMock(return_value=set_nx_result)
    r.delete = AsyncMock()
    return r


def _patch_redis(redis: AsyncMock):
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
        """Returns the sandbox when it connects and is running; refreshes Redis TTL."""
        sb = _mock_sandbox()
        redis = _mock_redis()
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(_try_reconnect(_SANDBOX_ID, _SESSION_ID, _API_KEY))

        assert result is sb
        redis.delete.assert_not_awaited()
        # TTL must be refreshed so an active session cannot lose its key at expiry.
        redis.set.assert_awaited_once()

    def test_reconnect_not_running_clears_redis(self):
        """Clears sandbox_id in Redis when the sandbox is no longer running."""
        sb = _mock_sandbox(running=False)
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(_try_reconnect(_SANDBOX_ID, _SESSION_ID, _API_KEY))

        assert result is None
        redis.delete.assert_awaited_once()

    def test_reconnect_exception_clears_redis(self):
        """Clears sandbox_id in Redis when connect raises an exception."""
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(side_effect=ConnectionError("gone"))
            result = asyncio.run(_try_reconnect(_SANDBOX_ID, _SESSION_ID, _API_KEY))

        assert result is None
        redis.delete.assert_awaited_once()


# ---------------------------------------------------------------------------
# get_or_create_sandbox
# ---------------------------------------------------------------------------


class TestGetOrCreateSandbox:
    def test_reconnect_existing(self):
        """When Redis has a valid sandbox_id, reconnect to it."""
        sb = _mock_sandbox()
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(
                get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
            )

        assert result is sb
        mock_cls.create.assert_not_called()
        # redis.set called once to refresh TTL, not to claim a creation slot
        redis.set.assert_awaited_once()

    def test_create_new_when_no_stored_id(self):
        """When Redis has no sandbox_id, claim slot and create a new sandbox."""
        new_sb = _mock_sandbox("sb-new")
        redis = _mock_redis(set_nx_result=True, stored_sandbox_id=None)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.create = AsyncMock(return_value=new_sb)
            result = asyncio.run(
                get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
            )

        assert result is new_sb
        mock_cls.create.assert_awaited_once()
        # Verify lifecycle: pause + auto_resume enabled
        _, kwargs = mock_cls.create.call_args
        assert kwargs.get("lifecycle") == {
            "on_timeout": "pause",
            "auto_resume": True,
        }
        # sandbox_id should be saved to Redis
        redis.set.assert_awaited()

    def test_create_with_on_timeout_kill(self):
        """on_timeout='kill' disables auto_resume automatically."""
        new_sb = _mock_sandbox("sb-new")
        redis = _mock_redis(set_nx_result=True, stored_sandbox_id=None)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.create = AsyncMock(return_value=new_sb)
            asyncio.run(
                get_or_create_sandbox(
                    _SESSION_ID, _API_KEY, timeout=_TIMEOUT, on_timeout="kill"
                )
            )

        _, kwargs = mock_cls.create.call_args
        assert kwargs.get("lifecycle") == {
            "on_timeout": "kill",
            "auto_resume": False,
        }

    def test_create_failure_releases_slot(self):
        """If sandbox creation fails, the Redis creation slot is deleted."""
        redis = _mock_redis(set_nx_result=True, stored_sandbox_id=None)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.create = AsyncMock(side_effect=RuntimeError("quota"))
            with pytest.raises(RuntimeError, match="quota"):
                asyncio.run(
                    get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
                )

        redis.delete.assert_awaited_once()

    def test_redis_save_failure_kills_sandbox_and_releases_slot(self):
        """If Redis save fails after creation, sandbox is killed and slot released."""
        new_sb = _mock_sandbox("sb-new")
        redis = _mock_redis(set_nx_result=True, stored_sandbox_id=None)
        # First set() call = creation slot SET NX (returns True).
        # Second set() call = sandbox_id save (raises).
        call_count = 0

        async def _set_side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return True  # creation slot claimed
            raise RuntimeError("redis error")

        redis.set = AsyncMock(side_effect=_set_side_effect)

        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.create = AsyncMock(return_value=new_sb)
            with pytest.raises(RuntimeError, match="redis error"):
                asyncio.run(
                    get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
                )

        # Sandbox must be killed to avoid leaking it
        new_sb.kill.assert_awaited_once()
        # Creation slot must always be released
        redis.delete.assert_awaited_once()

    def test_wait_for_creating_sentinel_then_reconnect(self):
        """When the key holds the 'creating' sentinel, wait then reconnect."""
        sb = _mock_sandbox("sb-other")
        # First get() returns the sentinel; second returns the real ID.
        redis = AsyncMock()
        creating_raw = _CREATING_SENTINEL.encode()
        redis.get = AsyncMock(side_effect=[creating_raw, b"sb-other"])
        redis.set = AsyncMock(return_value=False)
        redis.delete = AsyncMock()

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
                get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
            )

        assert result is sb

    def test_stale_reconnect_clears_and_creates(self):
        """When stored sandbox is stale (not running), clear it and create a new one."""
        stale_sb = _mock_sandbox("sb-stale", running=False)
        new_sb = _mock_sandbox("sb-fresh")
        # First get() returns stale id (for reconnect check), then None (after clear).
        redis = AsyncMock()
        redis.get = AsyncMock(side_effect=[b"sb-stale", None])
        redis.set = AsyncMock(return_value=True)
        redis.delete = AsyncMock()

        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=stale_sb)
            mock_cls.create = AsyncMock(return_value=new_sb)
            result = asyncio.run(
                get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
            )

        assert result is new_sb
        # Redis delete called at least once to clear stale id
        redis.delete.assert_awaited()


# ---------------------------------------------------------------------------
# kill_sandbox
# ---------------------------------------------------------------------------


class TestKillSandbox:
    def test_kill_existing_sandbox(self):
        """Kill a running sandbox and clear its Redis entry."""
        sb = _mock_sandbox()
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))

        assert result is True
        sb.kill.assert_awaited_once()
        # Redis key cleared after successful kill
        redis.delete.assert_awaited_once()

    def test_kill_no_sandbox(self):
        """No-op when Redis has no sandbox_id."""
        redis = _mock_redis(stored_sandbox_id=None)
        with _patch_redis(redis):
            result = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))

        assert result is False

    def test_kill_connect_failure_keeps_redis(self):
        """Returns False and leaves Redis entry intact when connect/kill fails.

        Keeping the sandbox_id in Redis allows the kill to be retried.
        """
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(side_effect=ConnectionError("gone"))
            result = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))

        assert result is False
        redis.delete.assert_not_awaited()

    def test_kill_timeout_keeps_redis(self):
        """Returns False and leaves Redis entry intact when the E2B call times out."""
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with (
            _patch_redis(redis),
            patch(
                "backend.copilot.tools.e2b_sandbox.asyncio.wait_for",
                new_callable=AsyncMock,
                side_effect=asyncio.TimeoutError,
            ),
        ):
            result = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))

        assert result is False
        redis.delete.assert_not_awaited()

    def test_kill_creating_sentinel_returns_false(self):
        """No-op when the key holds the 'creating' sentinel (no real sandbox yet)."""
        redis = _mock_redis(stored_sandbox_id=_CREATING_SENTINEL)
        with _patch_redis(redis):
            result = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))

        assert result is False


# ---------------------------------------------------------------------------
# pause_sandbox
# ---------------------------------------------------------------------------


class TestPauseSandbox:
    def test_pause_existing_sandbox(self):
        """Pause a running sandbox; Redis sandbox_id is preserved."""
        sb = _mock_sandbox()
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(pause_sandbox(_SESSION_ID, _API_KEY))

        assert result is True
        sb.pause.assert_awaited_once()
        # sandbox_id should remain in Redis (not cleared on pause)
        redis.delete.assert_not_awaited()

    def test_pause_no_sandbox(self):
        """No-op when Redis has no sandbox_id."""
        redis = _mock_redis(stored_sandbox_id=None)
        with _patch_redis(redis):
            result = asyncio.run(pause_sandbox(_SESSION_ID, _API_KEY))

        assert result is False

    def test_pause_connect_failure(self):
        """Returns False if connect fails."""
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(side_effect=ConnectionError("gone"))
            result = asyncio.run(pause_sandbox(_SESSION_ID, _API_KEY))

        assert result is False

    def test_pause_creating_sentinel_returns_false(self):
        """No-op when the key holds the 'creating' sentinel (no real sandbox yet)."""
        redis = _mock_redis(stored_sandbox_id=_CREATING_SENTINEL)
        with _patch_redis(redis):
            result = asyncio.run(pause_sandbox(_SESSION_ID, _API_KEY))

        assert result is False

    def test_pause_timeout_returns_false(self):
        """Returns False and preserves Redis entry when the E2B API call times out."""
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with (
            _patch_redis(redis),
            patch(
                "backend.copilot.tools.e2b_sandbox.asyncio.wait_for",
                new_callable=AsyncMock,
                side_effect=asyncio.TimeoutError,
            ),
        ):
            result = asyncio.run(pause_sandbox(_SESSION_ID, _API_KEY))

        assert result is False
        # sandbox_id must remain in Redis so the next turn can reconnect
        redis.delete.assert_not_awaited()

    def test_pause_then_reconnect_reuses_sandbox(self):
        """After pause, get_or_create_sandbox reconnects the same sandbox.

        Covers the pause->reconnect cycle: connect() auto-resumes a paused
        sandbox, and is_running() returns True once resume completes, so the
        same sandbox_id is reused rather than a new one being created.
        """
        sb = _mock_sandbox(_SANDBOX_ID)
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)

            # Step 1: pause the sandbox
            paused = asyncio.run(pause_sandbox(_SESSION_ID, _API_KEY))
            assert paused is True
            sb.pause.assert_awaited_once()

            # Step 2: reconnect on next turn -- same sandbox should be returned
            result = asyncio.run(
                get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
            )

        assert result is sb
        mock_cls.create.assert_not_called()


# ---------------------------------------------------------------------------
# pause_sandbox_direct
# ---------------------------------------------------------------------------


class TestPauseSandboxDirect:
    def test_pause_direct_success(self):
        """Pauses the sandbox directly without a Redis lookup or reconnect."""
        sb = _mock_sandbox()
        result = asyncio.run(pause_sandbox_direct(sb, _SESSION_ID))

        assert result is True
        sb.pause.assert_awaited_once()

    def test_pause_direct_failure_returns_false(self):
        """Returns False when sandbox.pause() raises."""
        sb = _mock_sandbox()
        sb.pause = AsyncMock(side_effect=RuntimeError("e2b error"))
        result = asyncio.run(pause_sandbox_direct(sb, _SESSION_ID))

        assert result is False

    def test_pause_direct_timeout_returns_false(self):
        """Returns False when sandbox.pause() exceeds the 10s timeout."""
        sb = _mock_sandbox()
        with patch(
            "backend.copilot.tools.e2b_sandbox.asyncio.wait_for",
            new_callable=AsyncMock,
            side_effect=asyncio.TimeoutError,
        ):
            result = asyncio.run(pause_sandbox_direct(sb, _SESSION_ID))

        assert result is False
