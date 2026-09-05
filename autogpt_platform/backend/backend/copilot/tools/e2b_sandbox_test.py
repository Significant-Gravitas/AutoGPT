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
    _SANDBOX_CREATE_MAX_RETRIES,
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

    def test_create_retries_on_timeout_then_succeeds(self):
        """On first-attempt timeout, retries and succeeds on second attempt."""
        new_sb = _mock_sandbox("sb-retry")
        redis = _mock_redis(set_nx_result=True, stored_sandbox_id=None)

        call_count = 0

        async def _create_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise asyncio.TimeoutError
            return new_sb

        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
            patch(
                "backend.copilot.tools.e2b_sandbox.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_cls.create = AsyncMock(side_effect=_create_side_effect)
            result = asyncio.run(
                get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
            )

        assert result is new_sb
        assert call_count == 2

    def test_create_exhausts_all_retries_then_raises(self):
        """When all retry attempts fail, the last exception is re-raised."""
        redis = _mock_redis(set_nx_result=True, stored_sandbox_id=None)

        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
            patch(
                "backend.copilot.tools.e2b_sandbox.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_cls.create = AsyncMock(side_effect=asyncio.TimeoutError)
            with pytest.raises(asyncio.TimeoutError):
                asyncio.run(
                    get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
                )

        assert mock_cls.create.await_count == _SANDBOX_CREATE_MAX_RETRIES
        # Creation slot must be released even after full retry exhaustion
        redis.delete.assert_awaited_once()

    def test_create_non_timeout_exception_also_retried(self):
        """Non-timeout exceptions (e.g., network errors) are also retried."""
        new_sb = _mock_sandbox("sb-net-retry")
        redis = _mock_redis(set_nx_result=True, stored_sandbox_id=None)

        call_count = 0

        async def _create_side_effect(**kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise ConnectionError("temporary network blip")
            return new_sb

        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
            patch(
                "backend.copilot.tools.e2b_sandbox.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_cls.create = AsyncMock(side_effect=_create_side_effect)
            result = asyncio.run(
                get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
            )

        assert result is new_sb
        assert call_count == 2

    def test_create_cancellation_releases_creation_slot(self):
        """CancelledError during creation must release the Redis sentinel."""
        redis = _mock_redis(set_nx_result=True, stored_sandbox_id=None)

        async def _create_side_effect(**kwargs):
            raise asyncio.CancelledError

        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
            patch(
                "backend.copilot.tools.e2b_sandbox.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_cls.create = AsyncMock(side_effect=_create_side_effect)
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(
                    get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
                )

        # Sentinel must be released even on task cancellation
        redis.delete.assert_awaited_once()

    def test_post_create_cancellation_kills_sandbox(self):
        """CancelledError during _set_stored_sandbox_id must kill the already-created sandbox."""
        redis = _mock_redis(set_nx_result=True, stored_sandbox_id=None)
        created_sb = _mock_sandbox()

        async def _set_side_effect(*_args, **_kwargs):
            raise asyncio.CancelledError

        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            patch(
                "backend.copilot.tools.e2b_sandbox._set_stored_sandbox_id",
                side_effect=_set_side_effect,
            ),
            _patch_redis(redis),
            patch(
                "backend.copilot.tools.e2b_sandbox.asyncio.sleep",
                new_callable=AsyncMock,
            ),
        ):
            mock_cls.create = AsyncMock(return_value=created_sb)
            with pytest.raises(asyncio.CancelledError):
                asyncio.run(
                    get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
                )

        # Sandbox must be killed and Redis sentinel cleared on post-create cancellation
        created_sb.kill.assert_awaited_once()
        redis.delete.assert_awaited_once()

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


# ---------------------------------------------------------------------------
# Expert boxes: one persistent sandbox per hired expert
# ---------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone  # noqa: E402
from types import SimpleNamespace  # noqa: E402

from e2b import SandboxState  # noqa: E402

from backend.blocks.desktop._api import SHARED_PATH, WORKSPACE_PATH  # noqa: E402
from backend.blocks.desktop._common import (  # noqa: E402
    expert_volume_name,
    user_volume_name,
    workspace_volume_mounts,
)

from .e2b_sandbox import (  # noqa: E402
    SandboxOwner,
    find_owned_sandbox_id,
    kill_expert_sandboxes,
)

_EXPERT_ID = "exp-777"
_USER_ID = "user-42"
_EXPERT_SHELL_KEY = f"copilot:e2b:expert:{_EXPERT_ID}:shell"
_EXPERT_DESKTOP_KEY = f"copilot:e2b:expert:{_EXPERT_ID}:desktop"
_EXPERT_ACTIVE_KEY = f"{_EXPERT_SHELL_KEY}:active"


def _info(
    sandbox_id: str, state: SandboxState, age_seconds: int = 0
) -> SimpleNamespace:
    return SimpleNamespace(
        sandbox_id=sandbox_id,
        state=state,
        started_at=datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
    )


def _mock_list(infos: list) -> MagicMock:
    """``AsyncSandbox.list`` is sync and returns an async paginator."""
    paginator = MagicMock()
    paginator.next_items = AsyncMock(return_value=infos)
    return MagicMock(return_value=paginator)


def _keyed_redis(values: dict[str, str | None], decr_result: int = 0) -> AsyncMock:
    """Redis mock answering ``get`` per key (bytes, like the real client)."""
    r = AsyncMock()
    r.get = AsyncMock(side_effect=lambda key: (values.get(key) or "").encode() or None)
    r.set = AsyncMock(return_value=True)
    r.delete = AsyncMock()
    r.incr = AsyncMock(return_value=1)
    r.expire = AsyncMock()
    r.decr = AsyncMock(return_value=decr_result)
    return r


class TestSandboxOwner:
    def test_expert_session_is_owned_by_the_expert(self):
        owner = SandboxOwner.for_session(_SESSION_ID, _EXPERT_ID)
        assert owner == SandboxOwner("expert", _EXPERT_ID)
        assert owner.is_expert
        assert owner.key() == _EXPERT_SHELL_KEY
        assert owner.key("desktop") == _EXPERT_DESKTOP_KEY

    def test_plain_session_keys_are_unchanged(self):
        owner = SandboxOwner.for_session(_SESSION_ID, None)
        assert not owner.is_expert
        assert owner.key() == f"copilot:e2b:sandbox:{_SESSION_ID}"
        assert owner.key("desktop") == f"copilot:e2b:desktop:{_SESSION_ID}"

    def test_expert_cache_outlives_session_cache(self):
        assert (
            SandboxOwner("expert", _EXPERT_ID).ttl
            > SandboxOwner("session", _SESSION_ID).ttl
        )

    def test_metadata_identifies_owner_and_kind(self):
        assert SandboxOwner("expert", _EXPERT_ID).metadata("desktop") == {
            "autogpt_owner": f"expert:{_EXPERT_ID}",
            "autogpt_kind": "desktop",
        }


class TestFindOwnedSandboxId:
    def test_session_owner_never_hits_the_api(self):
        with patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls:
            result = asyncio.run(
                find_owned_sandbox_id(
                    SandboxOwner("session", _SESSION_ID), "shell", _API_KEY
                )
            )
        assert result is None
        mock_cls.list.assert_not_called()

    def test_prefers_running_box_over_newer_paused_one(self):
        infos = [
            _info("sb-paused-new", SandboxState.PAUSED, age_seconds=10),
            _info("sb-running-old", SandboxState.RUNNING, age_seconds=500),
        ]
        with patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls:
            mock_cls.list = _mock_list(infos)
            result = asyncio.run(
                find_owned_sandbox_id(
                    SandboxOwner("expert", _EXPERT_ID), "shell", _API_KEY
                )
            )
        assert result == "sb-running-old"
        query = mock_cls.list.call_args.kwargs["query"]
        assert query.metadata == {
            "autogpt_owner": f"expert:{_EXPERT_ID}",
            "autogpt_kind": "shell",
        }
        assert set(query.state) == {SandboxState.RUNNING, SandboxState.PAUSED}

    def test_newest_paused_box_when_none_running(self):
        infos = [
            _info("sb-old", SandboxState.PAUSED, age_seconds=900),
            _info("sb-new", SandboxState.PAUSED, age_seconds=5),
        ]
        with patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls:
            mock_cls.list = _mock_list(infos)
            result = asyncio.run(
                find_owned_sandbox_id(
                    SandboxOwner("expert", _EXPERT_ID), "desktop", _API_KEY
                )
            )
        assert result == "sb-new"

    def test_api_failure_is_a_miss_not_an_error(self):
        with patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls:
            mock_cls.list = MagicMock(side_effect=RuntimeError("e2b down"))
            result = asyncio.run(
                find_owned_sandbox_id(
                    SandboxOwner("expert", _EXPERT_ID), "shell", _API_KEY
                )
            )
        assert result is None


class TestExpertShellBox:
    def test_recovers_expert_box_from_e2b_when_redis_is_empty(self):
        """Redis is only a cache for an expert's box; E2B metadata is the record."""
        sb = _mock_sandbox("sb-expert")
        redis = _keyed_redis({})
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.list = _mock_list([_info("sb-expert", SandboxState.PAUSED)])
            mock_cls.connect = AsyncMock(return_value=sb)
            mock_cls.create = AsyncMock()
            result = asyncio.run(
                get_or_create_sandbox(
                    _SESSION_ID, _API_KEY, timeout=_TIMEOUT, expert_id=_EXPERT_ID
                )
            )

        assert result is sb
        mock_cls.connect.assert_awaited_once_with("sb-expert", api_key=_API_KEY)
        mock_cls.create.assert_not_awaited()
        # Re-cached under the expert key, never the session key.
        keys = {call.args[0] for call in redis.set.await_args_list}
        assert _EXPERT_SHELL_KEY in keys
        assert f"copilot:e2b:sandbox:{_SESSION_ID}" not in keys
        # This turn is counted so a concurrent turn's end cannot pause the box.
        redis.incr.assert_awaited_once_with(_EXPERT_ACTIVE_KEY)

    def test_creates_expert_box_with_home_and_shared_volumes(self):
        sb = _mock_sandbox("sb-expert-new")
        sb.commands = MagicMock()
        sb.commands.run = AsyncMock()
        redis = _keyed_redis({})
        mounts = workspace_volume_mounts(_USER_ID, _EXPERT_ID)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            patch("backend.copilot.tools.e2b_sandbox.AsyncVolume") as mock_volume,
            _patch_redis(redis),
        ):
            mock_cls.list = _mock_list([])
            mock_cls.create = AsyncMock(return_value=sb)
            # Volumes already exist -> mounted by name.
            mock_volume.create = AsyncMock(side_effect=RuntimeError("exists"))
            result = asyncio.run(
                get_or_create_sandbox(
                    _SESSION_ID,
                    _API_KEY,
                    timeout=_TIMEOUT,
                    volume_mounts=mounts,
                    expert_id=_EXPERT_ID,
                )
            )

        assert result is sb
        kwargs = mock_cls.create.call_args.kwargs
        assert kwargs["volume_mounts"] == {
            WORKSPACE_PATH: expert_volume_name(_EXPERT_ID),
            SHARED_PATH: user_volume_name(_USER_ID),
        }
        assert kwargs["metadata"] == {
            "autogpt_owner": f"expert:{_EXPERT_ID}",
            "autogpt_kind": "shell",
        }
        # Creation lock and cached id both live under the expert key.
        lock_call = redis.set.await_args_list[0]
        assert lock_call.args[0] == _EXPERT_SHELL_KEY
        assert lock_call.args[1] == _CREATING_SENTINEL
        # Both mount points exist before the first command runs.
        mkdir = sb.commands.run.await_args.args[0]
        assert WORKSPACE_PATH in mkdir and SHARED_PATH in mkdir

    def test_plain_session_create_is_tagged_but_untouched_otherwise(self):
        sb = _mock_sandbox("sb-plain")
        redis = _mock_redis(set_nx_result=True, stored_sandbox_id=None)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.create = AsyncMock(return_value=sb)
            asyncio.run(get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT))

        kwargs = mock_cls.create.call_args.kwargs
        assert kwargs["metadata"] == {
            "autogpt_owner": f"session:{_SESSION_ID}",
            "autogpt_kind": "shell",
        }
        assert kwargs["volume_mounts"] is None
        mock_cls.list.assert_not_called()
        redis.incr.assert_not_awaited()


class TestExpertPause:
    def test_last_turn_pauses_the_box(self):
        sb = _mock_sandbox()
        redis = _keyed_redis({}, decr_result=0)
        with _patch_redis(redis):
            ok = asyncio.run(
                pause_sandbox_direct(sb, _SESSION_ID, expert_id=_EXPERT_ID)
            )
        assert ok is True
        sb.pause.assert_awaited_once()
        redis.decr.assert_awaited_once_with(_EXPERT_ACTIVE_KEY)
        redis.delete.assert_awaited_with(_EXPERT_ACTIVE_KEY)

    def test_concurrent_turn_keeps_the_box_running(self):
        """Pausing under another session of the same expert would sever its
        command stream, so the box stays up until the last turn ends."""
        sb = _mock_sandbox()
        redis = _keyed_redis({}, decr_result=1)
        with _patch_redis(redis):
            ok = asyncio.run(
                pause_sandbox_direct(sb, _SESSION_ID, expert_id=_EXPERT_ID)
            )
        assert ok is False
        sb.pause.assert_not_awaited()

    def test_lookup_pause_honours_the_counter_too(self):
        redis = _keyed_redis({_EXPERT_SHELL_KEY: _SANDBOX_ID}, decr_result=2)
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock()
            ok = asyncio.run(pause_sandbox(_SESSION_ID, _API_KEY, expert_id=_EXPERT_ID))
        assert ok is False
        mock_cls.connect.assert_not_awaited()

    def test_session_pause_never_touches_the_counter(self):
        sb = _mock_sandbox()
        redis = _keyed_redis({})
        with _patch_redis(redis):
            ok = asyncio.run(pause_sandbox_direct(sb, _SESSION_ID))
        assert ok is True
        redis.decr.assert_not_awaited()


class TestExpertKill:
    def test_deleting_an_expert_chat_leaves_the_expert_box_alone(self):
        """kill_sandbox only knows session keys; an expert box has none."""
        redis = _keyed_redis({_EXPERT_SHELL_KEY: "sb-expert"})
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock()
            ok = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))
        assert ok is False
        mock_cls.connect.assert_not_awaited()

    def test_archive_kills_shell_and_desktop_and_clears_cache(self):
        shell, desktop = _mock_sandbox("sb-shell"), _mock_sandbox("sb-desktop")
        redis = _keyed_redis(
            {_EXPERT_SHELL_KEY: "sb-shell", _EXPERT_DESKTOP_KEY: "sb-desktop"}
        )
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(
                side_effect=lambda sid, api_key: {
                    "sb-shell": shell,
                    "sb-desktop": desktop,
                }[sid]
            )
            killed = asyncio.run(kill_expert_sandboxes(_EXPERT_ID, _API_KEY))

        assert killed == 2
        shell.kill.assert_awaited_once()
        desktop.kill.assert_awaited_once()
        deleted = {call.args[0] for call in redis.delete.await_args_list}
        assert deleted == {_EXPERT_SHELL_KEY, _EXPERT_DESKTOP_KEY, _EXPERT_ACTIVE_KEY}
        mock_cls.list.assert_not_called()

    def test_archive_falls_back_to_e2b_metadata_for_forgotten_boxes(self):
        shell = _mock_sandbox("sb-shell")
        redis = _keyed_redis({})
        lists = {
            "shell": [_info("sb-shell", SandboxState.PAUSED)],
            "desktop": [],
        }
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):

            def _list(query, **_):
                paginator = MagicMock()
                paginator.next_items = AsyncMock(
                    return_value=lists[query.metadata["autogpt_kind"]]
                )
                return paginator

            mock_cls.list = MagicMock(side_effect=_list)
            mock_cls.connect = AsyncMock(return_value=shell)
            killed = asyncio.run(kill_expert_sandboxes(_EXPERT_ID, _API_KEY))

        assert killed == 1
        shell.kill.assert_awaited_once()
        assert mock_cls.list.call_count == 2

    def test_failed_kill_keeps_cache_for_retry(self):
        redis = _keyed_redis({_EXPERT_SHELL_KEY: "sb-shell"})
        with (
            patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.list = _mock_list([])
            mock_cls.connect = AsyncMock(side_effect=RuntimeError("boom"))
            killed = asyncio.run(kill_expert_sandboxes(_EXPERT_ID, _API_KEY))
        assert killed == 0
        deleted = {call.args[0] for call in redis.delete.await_args_list}
        assert _EXPERT_SHELL_KEY not in deleted
