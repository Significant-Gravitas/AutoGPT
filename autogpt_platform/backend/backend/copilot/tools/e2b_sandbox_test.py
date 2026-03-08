"""Tests for e2b_sandbox: get_or_create_sandbox, _try_reconnect, kill_sandbox.

sandbox_id is now stored in ChatSession.metadata (DB) via ChatSessionMetadata.
Redis is only used for the short-lived creation lock.

Tests mock:
- ``get_session_metadata`` / ``update_session_metadata`` (DB layer)
- ``get_redis_async`` (creation lock only)
- ``AsyncSandbox`` (E2B SDK)

Tests are synchronous (using asyncio.run) to avoid conflicts with the
session-scoped event loop in conftest.py.
"""

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend.copilot.model import ChatSessionMetadata

from .e2b_sandbox import (
    _try_reconnect,
    get_or_create_sandbox,
    kill_sandbox,
    pause_sandbox,
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


def _mock_redis(set_nx_result: bool = True) -> AsyncMock:
    r = AsyncMock()
    r.set = AsyncMock(return_value=set_nx_result)
    r.delete = AsyncMock()
    return r


def _patch_redis(redis: AsyncMock):
    return patch(
        "backend.copilot.tools.e2b_sandbox.get_redis_async",
        new_callable=AsyncMock,
        return_value=redis,
    )


def _patch_get_metadata(metadata: ChatSessionMetadata):
    return patch(
        "backend.copilot.tools.e2b_sandbox.get_session_metadata",
        new_callable=AsyncMock,
        return_value=metadata,
    )


def _patch_update_metadata():
    return patch(
        "backend.copilot.tools.e2b_sandbox.update_session_metadata",
        new_callable=AsyncMock,
    )


# ---------------------------------------------------------------------------
# _try_reconnect
# ---------------------------------------------------------------------------


class TestTryReconnect:
    def test_reconnect_success(self):
        """Returns the sandbox when it connects and is running."""
        sb = _mock_sandbox()
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_update_metadata() as mock_update,
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(_try_reconnect(_SANDBOX_ID, _SESSION_ID, _API_KEY))

        assert result is sb
        mock_update.assert_not_awaited()

    def test_reconnect_not_running_clears_metadata(self):
        """Clears sandbox_id in metadata when the sandbox is no longer running."""
        sb = _mock_sandbox(running=False)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_get_metadata(ChatSessionMetadata(e2b_sandbox_id=_SANDBOX_ID)),
            _patch_update_metadata() as mock_update,
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(_try_reconnect(_SANDBOX_ID, _SESSION_ID, _API_KEY))

        assert result is None
        mock_update.assert_awaited_once()
        _, updated_meta = mock_update.call_args.args
        assert updated_meta.e2b_sandbox_id is None

    def test_reconnect_exception_clears_metadata(self):
        """Clears sandbox_id in metadata when connect raises an exception."""
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_get_metadata(ChatSessionMetadata(e2b_sandbox_id=_SANDBOX_ID)),
            _patch_update_metadata() as mock_update,
        ):
            mock_cls.connect = AsyncMock(side_effect=ConnectionError("gone"))
            result = asyncio.run(_try_reconnect(_SANDBOX_ID, _SESSION_ID, _API_KEY))

        assert result is None
        mock_update.assert_awaited_once()


# ---------------------------------------------------------------------------
# get_or_create_sandbox
# ---------------------------------------------------------------------------


class TestGetOrCreateSandbox:
    def test_reconnect_existing(self):
        """When metadata has a valid sandbox_id, reconnect to it."""
        sb = _mock_sandbox()
        redis = _mock_redis()
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
            _patch_get_metadata(ChatSessionMetadata(e2b_sandbox_id=_SANDBOX_ID)),
            _patch_update_metadata(),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(
                get_or_create_sandbox(_SESSION_ID, _API_KEY, pause_timeout=_TIMEOUT)
            )

        assert result is sb
        mock_cls.create.assert_not_called()
        redis.set.assert_not_called()

    def test_create_new_when_no_metadata(self):
        """When metadata has no sandbox_id, claim lock and create a new sandbox."""
        new_sb = _mock_sandbox("sb-new")
        redis = _mock_redis(set_nx_result=True)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
            _patch_get_metadata(ChatSessionMetadata(e2b_sandbox_id=None)),
            _patch_update_metadata() as mock_update,
        ):
            mock_cls.create = AsyncMock(return_value=new_sb)
            result = asyncio.run(
                get_or_create_sandbox(_SESSION_ID, _API_KEY, pause_timeout=_TIMEOUT)
            )

        assert result is new_sb
        mock_cls.create.assert_awaited_once()
        # Verify lifecycle param is set
        _, kwargs = mock_cls.create.call_args
        assert kwargs.get("lifecycle") == {"on_timeout": "pause"}
        # Metadata should be updated with the new sandbox_id
        mock_update.assert_awaited_once()
        _, saved_meta = mock_update.call_args.args
        assert saved_meta.e2b_sandbox_id == "sb-new"
        # Lock released
        redis.delete.assert_awaited_once()

    def test_create_failure_clears_lock(self):
        """If sandbox creation fails, the Redis lock is deleted."""
        redis = _mock_redis(set_nx_result=True)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
            _patch_get_metadata(ChatSessionMetadata(e2b_sandbox_id=None)),
            _patch_update_metadata(),
        ):
            mock_cls.create = AsyncMock(side_effect=RuntimeError("quota"))
            with pytest.raises(RuntimeError, match="quota"):
                asyncio.run(
                    get_or_create_sandbox(_SESSION_ID, _API_KEY, pause_timeout=_TIMEOUT)
                )

        redis.delete.assert_awaited_once()

    def test_wait_for_lock_then_reconnect(self):
        """When another process holds the lock, wait then reconnect to the created sandbox."""
        sb = _mock_sandbox("sb-other")
        redis = _mock_redis(set_nx_result=False)
        meta_with_sandbox = ChatSessionMetadata(e2b_sandbox_id="sb-other")

        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
            patch(
                "backend.copilot.tools.e2b_sandbox.get_session_metadata",
                new_callable=AsyncMock,
                side_effect=[
                    ChatSessionMetadata(e2b_sandbox_id=None),  # initial check
                    meta_with_sandbox,  # poll in wait loop
                ],
            ),
            _patch_update_metadata(),
            patch(
                "backend.copilot.tools.e2b_sandbox.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(
                get_or_create_sandbox(_SESSION_ID, _API_KEY, pause_timeout=_TIMEOUT)
            )

        assert result is sb

    def test_stale_reconnect_clears_and_creates(self):
        """When stored sandbox is stale (not running), clear it and create a new one."""
        stale_sb = _mock_sandbox("sb-stale", running=False)
        new_sb = _mock_sandbox("sb-fresh")
        redis = _mock_redis(set_nx_result=True)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
            _patch_get_metadata(ChatSessionMetadata(e2b_sandbox_id="sb-stale")),
            _patch_update_metadata() as mock_update,
        ):
            mock_cls.connect = AsyncMock(return_value=stale_sb)
            mock_cls.create = AsyncMock(return_value=new_sb)
            result = asyncio.run(
                get_or_create_sandbox(_SESSION_ID, _API_KEY, pause_timeout=_TIMEOUT)
            )

        assert result is new_sb
        # update_session_metadata called at least once (clear stale, then save new)
        assert mock_update.await_count >= 1


# ---------------------------------------------------------------------------
# kill_sandbox
# ---------------------------------------------------------------------------


class TestKillSandbox:
    def test_kill_existing_sandbox(self):
        """Kill a running sandbox and clear its metadata entry."""
        sb = _mock_sandbox()
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_get_metadata(ChatSessionMetadata(e2b_sandbox_id=_SANDBOX_ID)),
            _patch_update_metadata() as mock_update,
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))

        assert result is True
        sb.kill.assert_awaited_once()
        # Metadata cleared after successful kill
        mock_update.assert_awaited_once()
        _, cleared_meta = mock_update.call_args.args
        assert cleared_meta.e2b_sandbox_id is None

    def test_kill_no_sandbox(self):
        """No-op when metadata has no sandbox_id."""
        with _patch_get_metadata(ChatSessionMetadata(e2b_sandbox_id=None)):
            result = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))

        assert result is False

    def test_kill_clear_metadata_false_skips_metadata_update(self):
        """When clear_metadata=False, metadata is NOT updated after kill.

        Used by the delete-session route where the session row is already gone,
        so attempting to write metadata would fail with a not-found error.
        """
        sb = _mock_sandbox()
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_update_metadata() as mock_update,
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(
                kill_sandbox(
                    _SESSION_ID,
                    _API_KEY,
                    e2b_sandbox_id=_SANDBOX_ID,
                    clear_metadata=False,
                )
            )

        assert result is True
        sb.kill.assert_awaited_once()
        mock_update.assert_not_awaited()

    def test_kill_connect_failure_keeps_metadata(self):
        """Returns False and leaves metadata intact when connect/kill fails.

        Keeping the sandbox_id in metadata allows the kill to be retried.
        """
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_get_metadata(ChatSessionMetadata(e2b_sandbox_id=_SANDBOX_ID)),
            _patch_update_metadata() as mock_update,
        ):
            mock_cls.connect = AsyncMock(side_effect=ConnectionError("gone"))
            result = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))

        assert result is False
        mock_update.assert_not_awaited()

    def test_kill_timeout_keeps_metadata(self):
        """Returns False and leaves metadata intact when the E2B call times out."""
        with (
            _patch_get_metadata(ChatSessionMetadata(e2b_sandbox_id=_SANDBOX_ID)),
            _patch_update_metadata() as mock_update,
            patch(
                "backend.copilot.tools.e2b_sandbox.asyncio.wait_for",
                new_callable=AsyncMock,
                side_effect=asyncio.TimeoutError,
            ),
        ):
            result = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))

        assert result is False
        mock_update.assert_not_awaited()


# ---------------------------------------------------------------------------
# pause_sandbox
# ---------------------------------------------------------------------------


class TestPauseSandbox:
    def test_pause_existing_sandbox(self):
        """Pause a running sandbox; metadata sandbox_id is preserved."""
        sb = _mock_sandbox()
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_get_metadata(ChatSessionMetadata(e2b_sandbox_id=_SANDBOX_ID)),
            _patch_update_metadata() as mock_update,
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(pause_sandbox(_SESSION_ID, _API_KEY))

        assert result is True
        sb.pause.assert_awaited_once()
        # sandbox_id should remain in metadata (not cleared on pause)
        mock_update.assert_not_awaited()

    def test_pause_no_sandbox(self):
        """No-op when metadata has no sandbox_id."""
        with _patch_get_metadata(ChatSessionMetadata(e2b_sandbox_id=None)):
            result = asyncio.run(pause_sandbox(_SESSION_ID, _API_KEY))

        assert result is False

    def test_pause_connect_failure(self):
        """Returns False if connect fails."""
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_get_metadata(ChatSessionMetadata(e2b_sandbox_id=_SANDBOX_ID)),
            _patch_update_metadata(),
        ):
            mock_cls.connect = AsyncMock(side_effect=ConnectionError("gone"))
            result = asyncio.run(pause_sandbox(_SESSION_ID, _API_KEY))

        assert result is False

    def test_pause_then_reconnect_reuses_sandbox(self):
        """After pause, get_or_create_sandbox reconnects the same sandbox.

        Covers the pause→reconnect cycle: connect() auto-resumes a paused
        sandbox, and is_running() returns True once resume completes, so the
        same sandbox_id is reused rather than a new one being created.
        """
        sb = _mock_sandbox(_SANDBOX_ID)
        redis = _mock_redis()
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
            _patch_get_metadata(ChatSessionMetadata(e2b_sandbox_id=_SANDBOX_ID)),
            _patch_update_metadata(),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)

            # Step 1: pause the sandbox
            paused = asyncio.run(pause_sandbox(_SESSION_ID, _API_KEY))
            assert paused is True
            sb.pause.assert_awaited_once()

            # Step 2: reconnect on next turn — same sandbox should be returned
            result = asyncio.run(
                get_or_create_sandbox(_SESSION_ID, _API_KEY, pause_timeout=_TIMEOUT)
            )

        assert result is sb
        mock_cls.create.assert_not_called()
