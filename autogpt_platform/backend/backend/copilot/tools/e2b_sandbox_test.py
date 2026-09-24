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
import contextlib
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from e2b import SandboxState
from e2b.exceptions import SandboxNotFoundException

from backend.blocks.desktop._api import _STOP_STREAM, SHARED_PATH, WORKSPACE_PATH
from backend.blocks.desktop._common import (
    expert_volume_name,
    user_volume_name,
    workspace_volume_mounts,
)
from backend.util.sandbox_metadata import deployment_env

from .e2b_sandbox import (
    _CREATING_SENTINEL,
    _SANDBOX_CREATE_MAX_RETRIES,
    SandboxLookupError,
    SandboxNotOwnedError,
    SandboxOwner,
    _try_reconnect,
    connect_owned,
    count_expert_turn,
    find_owned_sandbox_id,
    get_or_create_sandbox,
    kill_expert_sandbox,
    kill_sandbox,
    pause_sandbox,
    pause_sandbox_direct,
)

_SESSION_ID = "sess-123"
_API_KEY = "test-api-key"
_SANDBOX_ID = "sb-abc"
_TIMEOUT = 300


def _mock_sandbox(
    sandbox_id: str = _SANDBOX_ID,
    running: bool = True,
    *,
    owner: SandboxOwner | None = None,
) -> MagicMock:
    """A connected box stamped as *owner*'s (the session's shell box by default).

    ``connect_owned`` reads the stamp back through ``get_info``; a box with
    the wrong stamp is refused, so every test that reconnects says whose box
    it is.
    """
    sb = MagicMock()
    sb.sandbox_id = sandbox_id
    sb.is_running = AsyncMock(return_value=running)
    sb.pause = AsyncMock()
    sb.kill = AsyncMock()
    sb.commands.run = AsyncMock()
    stamped = (owner or SandboxOwner(kind="session", id=_SESSION_ID)).metadata()
    sb.get_info = AsyncMock(return_value=MagicMock(metadata=stamped))
    _STAMPS[sandbox_id] = sb.get_info.return_value
    return sb


# sandbox_id -> the info the SDK reports for it, filled by ``_mock_sandbox``.
_STAMPS: dict[str, MagicMock] = {}
_SESSION_SHELL_STAMP = SandboxOwner(kind="session", id=_SESSION_ID).metadata()


@pytest.fixture(autouse=True)
def _fresh_stamps():
    _STAMPS.clear()
    yield


@contextlib.contextmanager
def _patch_sdk():
    """The E2B SDK, with ``get_info`` answering for the boxes a test built.

    ``connect_owned`` reads a box's stamp through the *static* ``get_info``
    before it connects, so the class mock has to know the stamp too: a box
    from ``_mock_sandbox`` answers with its own, and an id no test built is
    taken to be the session's shell box, which is what most tests reconnect.
    """
    with patch("backend.copilot.tools.e2b_sandbox.AsyncSandbox") as mock_cls:

        async def _get_info(sandbox_id: str, **_):
            return _STAMPS.get(sandbox_id) or MagicMock(metadata=_SESSION_SHELL_STAMP)

        mock_cls.get_info = AsyncMock(side_effect=_get_info)
        # Nothing listed and kills by id succeed, unless a test says otherwise;
        # every kill path now sweeps for pre-one-box desktops through these.
        mock_cls.list = _mock_list([])
        mock_cls.kill = AsyncMock(return_value=True)
        yield mock_cls


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
    shell_key = f"copilot:e2b:sandbox:{_SESSION_ID}"
    # Only the session's box key holds the id.
    r.get = AsyncMock(side_effect=lambda key: raw if key == shell_key else None)
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


@pytest.fixture(autouse=True)
def _no_proxy_credentials():
    """Pause and kill revoke the box's proxy credential (``e2b_network``),
    which has its own Redis handle; ``TestProxyCredentialIsRevoked`` covers it."""
    with patch("backend.copilot.tools.e2b_sandbox.forget_sandbox", AsyncMock()):
        yield


class TestTryReconnect:
    def test_reconnect_refuses_a_box_stamped_for_someone_else(self):
        """A cached id that resolves to another owner's box is dropped, not used."""
        sb = _mock_sandbox(owner=SandboxOwner(kind="session", id="other-session"))
        redis = _mock_redis()
        with (
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(_try_reconnect(_SANDBOX_ID, _SESSION_ID, _API_KEY))

        assert result is None
        # Connecting would resume the other owner's box; it is never touched.
        mock_cls.connect.assert_not_awaited()
        sb.kill.assert_not_awaited()
        redis.delete.assert_awaited_once()

    def test_reconnect_success(self):
        """Returns the sandbox when it connects and is running; refreshes Redis TTL."""
        sb = _mock_sandbox()
        redis = _mock_redis()
        with (
            _patch_sdk() as mock_cls,
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
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(_try_reconnect(_SANDBOX_ID, _SESSION_ID, _API_KEY))

        assert result is None
        redis.delete.assert_awaited_once()

    def test_reconnect_gone_box_clears_redis(self):
        """A box E2B no longer has is given up on before any connect."""
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with (
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.get_info = AsyncMock(side_effect=SandboxNotFoundException("404"))
            mock_cls.connect = AsyncMock()
            result = asyncio.run(_try_reconnect(_SANDBOX_ID, _SESSION_ID, _API_KEY))

        assert result is None
        mock_cls.connect.assert_not_awaited()
        redis.delete.assert_awaited_once()

    def test_reconnect_transient_error_surfaces_and_keeps_redis(self):
        """A 5xx says nothing about the box; it must not be replaced over one."""
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with (
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(side_effect=ConnectionError("blip"))
            with pytest.raises(ConnectionError):
                asyncio.run(_try_reconnect(_SANDBOX_ID, _SESSION_ID, _API_KEY))

        redis.delete.assert_not_awaited()

    def test_reconnect_rearms_the_running_time_limit(self):
        sb = _mock_sandbox()
        with _patch_sdk() as mock_cls, _patch_redis(_mock_redis()):
            mock_cls.connect = AsyncMock(return_value=sb)
            asyncio.run(
                _try_reconnect(_SANDBOX_ID, _SESSION_ID, _API_KEY, timeout=_TIMEOUT)
            )
        mock_cls.connect.assert_awaited_once_with(
            _SANDBOX_ID, api_key=_API_KEY, timeout=_TIMEOUT
        )


# ---------------------------------------------------------------------------
# get_or_create_sandbox
# ---------------------------------------------------------------------------


class TestGetOrCreateSandbox:
    def test_reconnect_existing(self):
        """When Redis has a valid sandbox_id, reconnect to it."""
        sb = _mock_sandbox()
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with (
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(
                get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
            )

        assert result is sb
        # A box whose screen was never on pays no command on the way back.
        sb.commands.run.assert_not_awaited()
        mock_cls.create.assert_not_called()
        # redis.set called once to refresh TTL, not to claim a creation slot
        redis.set.assert_awaited_once()

    def test_create_new_when_no_stored_id(self):
        """When Redis has no sandbox_id, claim slot and create a new sandbox."""
        new_sb = _mock_sandbox("sb-new")
        redis = _mock_redis(set_nx_result=True, stored_sandbox_id=None)
        with (
            _patch_sdk() as mock_cls,
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

    def test_create_ensures_our_template_exists_first(self):
        """The managed image is built on the team before the first create."""
        order: list[str] = []
        new_sb = _mock_sandbox("sb-new")
        redis = _mock_redis(set_nx_result=True, stored_sandbox_id=None)

        async def fake_ensure(template: str, api_key: str) -> None:
            order.append(f"ensure:{template}:{api_key}")

        async def fake_create(**kwargs):
            order.append("create")
            return new_sb

        async def fake_set(key, value, **kwargs):
            if value == _CREATING_SENTINEL:
                order.append("claim")
            return True

        redis.set = AsyncMock(side_effect=fake_set)

        with (
            _patch_sdk() as mock_cls,
            patch(
                "backend.copilot.tools.e2b_sandbox.ensure_template",
                side_effect=fake_ensure,
            ),
            _patch_redis(redis),
        ):
            mock_cls.create = AsyncMock(side_effect=fake_create)
            asyncio.run(
                get_or_create_sandbox(
                    _SESSION_ID, _API_KEY, timeout=_TIMEOUT, template="agpt-desktop-1x2"
                )
            )

        # The build can take longer than the creation slot's TTL, so it must
        # finish before the slot is claimed.
        assert order == [f"ensure:agpt-desktop-1x2:{_API_KEY}", "claim", "create"]

    def test_create_with_on_timeout_kill(self):
        """on_timeout='kill' disables auto_resume automatically."""
        new_sb = _mock_sandbox("sb-new")
        redis = _mock_redis(set_nx_result=True, stored_sandbox_id=None)
        with (
            _patch_sdk() as mock_cls,
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
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.create = AsyncMock(side_effect=RuntimeError("quota"))
            with (
                pytest.raises(RuntimeError, match="quota"),
                patch("backend.copilot.tools.e2b_sandbox.forget_template") as forget,
            ):
                asyncio.run(
                    get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
                )

        redis.delete.assert_awaited_once()
        # The template is re-checked next time in case it went away.
        forget.assert_called_once_with("base", _API_KEY)

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
            _patch_sdk() as mock_cls,
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
            _patch_sdk() as mock_cls,
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
            _patch_sdk() as mock_cls,
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
            _patch_sdk() as mock_cls,
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
            _patch_sdk() as mock_cls,
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
            _patch_sdk() as mock_cls,
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
            _patch_sdk() as mock_cls,
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
            _patch_sdk() as mock_cls,
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
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))

        assert result is True
        sb.kill.assert_awaited_once()
        # The cached id is cleared after a successful kill, then the screen
        # flag and turn counter that belonged to the box.
        deleted = [key for call in redis.delete.await_args_list for key in call.args]
        assert deleted[0] == f"copilot:e2b:sandbox:{_SESSION_ID}"
        assert set(deleted[1:]) == {
            f"copilot:e2b:sandbox:{_SESSION_ID}:display",
            f"copilot:e2b:sandbox:{_SESSION_ID}:stream",
            f"copilot:e2b:sandbox:{_SESSION_ID}:active",
            f"copilot:e2b:desktop:{_SESSION_ID}",
        }

    def test_kill_refuses_a_foreign_box_and_forgets_it(self):
        """A cached id stamped for someone else is dropped so the next
        pause or kill does not fail the same way; the box is never touched."""
        foreign = _mock_sandbox(owner=SandboxOwner(kind="session", id="other"))
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with _patch_sdk() as mock_cls, _patch_redis(redis):
            mock_cls.connect = AsyncMock(return_value=foreign)
            result = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))

        assert result is False
        mock_cls.connect.assert_not_awaited()
        foreign.kill.assert_not_awaited()
        deleted = {key for call in redis.delete.await_args_list for key in call.args}
        assert f"copilot:e2b:sandbox:{_SESSION_ID}" in deleted
        assert f"copilot:e2b:sandbox:{_SESSION_ID}:display" not in deleted

    def test_kill_no_sandbox(self):
        """No-op when Redis has no sandbox_id."""
        redis = _mock_redis(stored_sandbox_id=None)
        with _patch_sdk(), _patch_redis(redis):
            result = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))

        assert result is False

    def test_kill_connect_failure_keeps_redis(self):
        """Returns False and leaves Redis entry intact when connect/kill fails.

        Keeping the sandbox_id in Redis allows the kill to be retried.
        """
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with (
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(side_effect=ConnectionError("gone"))
            result = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))

        assert result is False
        deleted = {key for call in redis.delete.await_args_list for key in call.args}
        assert f"copilot:e2b:sandbox:{_SESSION_ID}" not in deleted

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
        with _patch_sdk(), _patch_redis(redis):
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
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(pause_sandbox(_SESSION_ID, _API_KEY))

        assert result is True
        sb.pause.assert_awaited_once()
        # The id stays cached; only the stream password goes, so the next
        # open issues a new one and a URL that leaked stops working.
        deleted = {key for call in redis.delete.await_args_list for key in call.args}
        assert deleted == {f"copilot:e2b:sandbox:{_SESSION_ID}:stream"}

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
            _patch_sdk() as mock_cls,
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
            _patch_sdk() as mock_cls,
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
        redis = _mock_redis()
        with _patch_redis(redis):
            result = asyncio.run(pause_sandbox_direct(sb, _SESSION_ID))

        assert result is True
        sb.pause.assert_awaited_once()
        # The screen was never on: one flag read, and no command in the box.
        redis.get.assert_awaited_once_with(f"copilot:e2b:sandbox:{_SESSION_ID}:display")
        sb.commands.run.assert_not_awaited()
        # The turn-end pause is where the stream password is dropped.
        redis.delete.assert_awaited_once_with(
            f"copilot:e2b:sandbox:{_SESSION_ID}:stream"
        )

    def test_pause_direct_failure_returns_false(self):
        """Returns False when sandbox.pause() raises; the password stays."""
        sb = _mock_sandbox()
        sb.pause = AsyncMock(side_effect=RuntimeError("e2b error"))
        redis = _mock_redis()
        with _patch_redis(redis):
            result = asyncio.run(pause_sandbox_direct(sb, _SESSION_ID))

        assert result is False
        redis.delete.assert_not_awaited()

    def test_pause_direct_timeout_returns_false(self):
        """Returns False when sandbox.pause() exceeds the 10s timeout."""
        sb = _mock_sandbox()
        with (
            _patch_redis(_mock_redis()),
            patch(
                "backend.copilot.tools.e2b_sandbox.asyncio.wait_for",
                new_callable=AsyncMock,
                side_effect=asyncio.TimeoutError,
            ),
        ):
            result = asyncio.run(pause_sandbox_direct(sb, _SESSION_ID))

        assert result is False


# ---------------------------------------------------------------------------
# A pause ends the stream: the password is forgotten *and* nothing serves it
# ---------------------------------------------------------------------------

_SHELL_KEY = f"copilot:e2b:sandbox:{_SESSION_ID}"
_DISPLAY_KEY = f"{_SHELL_KEY}:display"
_STREAM_KEY = f"{_SHELL_KEY}:stream"
_DISPLAY_LOCK_KEY = f"{_DISPLAY_KEY}:lock"


def _screen_redis(**keys: str) -> AsyncMock:
    """Redis holding the box id plus the given screen keys, as bytes.

    Unlike ``_keyed_redis`` an empty value is a value: the stopped-stream
    marker is an empty string, and a missing key is ``None``.
    """
    values = {_SHELL_KEY: _SANDBOX_ID, **keys}
    r = AsyncMock()
    r.get = AsyncMock(
        side_effect=lambda key: values[key].encode() if key in values else None
    )
    r.set = AsyncMock(return_value=True)
    r.delete = AsyncMock()
    return r


def _stream_key_writes(redis: AsyncMock) -> list:
    return [c for c in redis.set.await_args_list if c.args[0] == _STREAM_KEY]


def _assert_stream_stopped_as_root(sb: MagicMock) -> None:
    sb.commands.run.assert_awaited_once()
    call = sb.commands.run.await_args
    assert call.args[0] == _STOP_STREAM
    assert call.kwargs["user"] == "root"


class TestPauseStopsTheStream:
    def test_pause_stops_the_stream_before_the_box_pauses(self):
        sb = _mock_sandbox()
        order: list[str] = []
        sb.commands.run = AsyncMock(side_effect=lambda *_, **__: order.append("stop"))
        sb.pause = AsyncMock(side_effect=lambda: order.append("pause"))
        redis = _screen_redis(**{_DISPLAY_KEY: _SANDBOX_ID, _STREAM_KEY: "issued"})
        with _patch_sdk() as mock_cls, _patch_redis(redis):
            mock_cls.connect = AsyncMock(return_value=sb)
            assert asyncio.run(pause_sandbox(_SESSION_ID, _API_KEY)) is True

        assert order == ["stop", "pause"]
        _assert_stream_stopped_as_root(sb)
        # The password is replaced by the stopped marker, not just deleted,
        # so the next reconnect knows there is nothing left to stop.
        [write] = _stream_key_writes(redis)
        assert write.args[1] == ""
        redis.delete.assert_not_awaited()

    def test_direct_pause_stops_the_stream_before_the_box_pauses(self):
        sb = _mock_sandbox()
        order: list[str] = []
        sb.commands.run = AsyncMock(side_effect=lambda *_, **__: order.append("stop"))
        sb.pause = AsyncMock(side_effect=lambda: order.append("pause"))
        redis = _screen_redis(**{_DISPLAY_KEY: _SANDBOX_ID, _STREAM_KEY: "issued"})
        with _patch_redis(redis):
            assert asyncio.run(pause_sandbox_direct(sb, _SESSION_ID)) is True

        assert order == ["stop", "pause"]
        _assert_stream_stopped_as_root(sb)
        [write] = _stream_key_writes(redis)
        assert write.args[1] == ""
        redis.delete.assert_not_awaited()

    def test_a_screen_flag_for_a_replaced_box_stops_nothing(self):
        sb = _mock_sandbox()
        redis = _screen_redis(**{_DISPLAY_KEY: "sb-old", _STREAM_KEY: "issued"})
        with _patch_redis(redis):
            assert asyncio.run(pause_sandbox_direct(sb, _SESSION_ID)) is True

        sb.commands.run.assert_not_awaited()
        redis.delete.assert_awaited_once_with(_STREAM_KEY)

    @pytest.mark.parametrize("direct", [True, False])
    def test_a_stop_that_fails_does_not_block_the_pause(self, direct: bool):
        sb = _mock_sandbox()
        sb.commands.run = AsyncMock(side_effect=RuntimeError("box not answering"))
        redis = _screen_redis(**{_DISPLAY_KEY: _SANDBOX_ID, _STREAM_KEY: "issued"})
        with _patch_sdk() as mock_cls, _patch_redis(redis):
            mock_cls.connect = AsyncMock(return_value=sb)
            paused = asyncio.run(
                pause_sandbox_direct(sb, _SESSION_ID)
                if direct
                else pause_sandbox(_SESSION_ID, _API_KEY)
            )

        assert paused is True
        sb.commands.run.assert_awaited_once()
        sb.pause.assert_awaited_once()
        # Not known to be stopped: the password is dropped with no marker, so
        # the box's next reconnect tries the stop again.
        assert _stream_key_writes(redis) == []
        redis.delete.assert_awaited_once_with(_STREAM_KEY)

    def test_a_stop_that_hangs_is_cut_off_and_the_box_still_pauses(self):
        async def _hang(*_, **__):
            await asyncio.sleep(30)

        sb = _mock_sandbox()
        sb.commands.run = AsyncMock(side_effect=_hang)
        redis = _screen_redis(**{_DISPLAY_KEY: _SANDBOX_ID, _STREAM_KEY: "issued"})
        with (
            _patch_redis(redis),
            patch(
                "backend.copilot.tools.e2b_sandbox._STOP_STREAM_TIMEOUT_SECONDS", 0.01
            ),
        ):
            assert asyncio.run(pause_sandbox_direct(sb, _SESSION_ID)) is True

        sb.commands.run.assert_awaited_once()
        sb.pause.assert_awaited_once()
        redis.delete.assert_awaited_once_with(_STREAM_KEY)

    def test_the_stop_does_not_eat_into_the_time_the_pause_has(self):
        """The lookup pause runs connect, stop and pause under one deadline:
        the stop's own limit is added to it rather than taken out of it."""
        sb = _mock_sandbox()
        redis = _screen_redis(**{_DISPLAY_KEY: _SANDBOX_ID, _STREAM_KEY: "issued"})
        real_wait_for = asyncio.wait_for
        budgets: list[float] = []

        async def _recording_wait_for(awaitable, timeout):
            budgets.append(timeout)
            return await real_wait_for(awaitable, timeout)

        with (
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
            patch(
                "backend.copilot.tools.e2b_sandbox.asyncio.wait_for",
                _recording_wait_for,
            ),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            assert asyncio.run(pause_sandbox(_SESSION_ID, _API_KEY)) is True

        assert budgets == [15, 5]

    def test_unreadable_screen_state_does_not_block_the_pause(self):
        sb = _mock_sandbox()
        redis = _screen_redis()
        redis.get = AsyncMock(side_effect=ConnectionError("redis down"))
        with _patch_redis(redis):
            assert asyncio.run(pause_sandbox_direct(sb, _SESSION_ID)) is True

        redis.get.assert_awaited_once_with(_DISPLAY_KEY)
        sb.pause.assert_awaited_once()


class TestReconnectSettlesTheStream:
    """A stream that outlived its running stretch is stopped on reconnect."""

    def _reconnect(self, redis: AsyncMock, sb: MagicMock, *, paused: bool):
        """Reconnect to *sb*, which E2B reports as *paused* (or running) just
        before the connect wakes it."""
        _STAMPS[sb.sandbox_id].state = (
            SandboxState.PAUSED if paused else SandboxState.RUNNING
        )
        with _patch_sdk() as mock_cls, _patch_redis(redis):
            mock_cls.connect = AsyncMock(return_value=sb)
            return asyncio.run(
                get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
            )

    @pytest.mark.parametrize(
        "keys",
        [
            pytest.param({_DISPLAY_KEY: _SANDBOX_ID}, id="password-gone"),
            # E2B's own timeout pause can land before the password expires.
            pytest.param(
                {_DISPLAY_KEY: _SANDBOX_ID, _STREAM_KEY: "issued"},
                id="password-still-remembered",
            ),
            # The opener's own connect is what resumed the box: nobody can be
            # part-way through starting a stream in a box that was paused.
            pytest.param(
                {_DISPLAY_KEY: _SANDBOX_ID, _DISPLAY_LOCK_KEY: "token"},
                id="open-in-progress",
            ),
        ],
    )
    def test_a_box_that_was_paused_comes_back_with_its_stream_stopped(self, keys: dict):
        sb = _mock_sandbox()
        redis = _screen_redis(**keys)

        assert self._reconnect(redis, sb, paused=True) is sb

        _assert_stream_stopped_as_root(sb)
        [write] = _stream_key_writes(redis)
        assert write.args[1] == ""

    def test_a_running_box_with_no_stream_key_was_resumed_behind_our_back(self):
        """The password outlived the box's running limit, so the box did pause
        and something other than us woke it: its stream is a leftover."""
        sb = _mock_sandbox()
        redis = _screen_redis(**{_DISPLAY_KEY: _SANDBOX_ID})

        assert self._reconnect(redis, sb, paused=False) is sb

        _assert_stream_stopped_as_root(sb)
        [write] = _stream_key_writes(redis)
        assert write.args[1] == ""

    def test_a_watched_stream_on_a_box_that_never_paused_is_kept_alive(self):
        """An expert's overlapping turns keep the box running past the
        password's first expiry: each connect pushes the expiry out with the
        box's re-armed limit instead of letting the stream read as a leftover."""
        sb = _mock_sandbox()
        redis = _screen_redis(**{_DISPLAY_KEY: _SANDBOX_ID, _STREAM_KEY: "issued"})

        assert self._reconnect(redis, sb, paused=False) is sb

        sb.commands.run.assert_not_awaited()
        redis.expire.assert_awaited_once_with(_STREAM_KEY, _TIMEOUT)
        assert _stream_key_writes(redis) == []

    def test_a_stop_that_fails_still_hands_the_box_back(self):
        sb = _mock_sandbox()
        sb.commands.run = AsyncMock(side_effect=RuntimeError("box not answering"))
        redis = _screen_redis(**{_DISPLAY_KEY: _SANDBOX_ID})

        assert self._reconnect(redis, sb, paused=True) is sb

        sb.commands.run.assert_awaited_once()
        assert _stream_key_writes(redis) == []

    def test_unreadable_screen_state_still_hands_the_box_back(self):
        sb = _mock_sandbox()
        redis = _screen_redis()
        shell_get = redis.get.side_effect

        def _get(key: str):
            if key == _DISPLAY_KEY:
                raise ConnectionError("redis down")
            return shell_get(key)

        redis.get = AsyncMock(side_effect=_get)

        assert self._reconnect(redis, sb, paused=True) is sb

        sb.commands.run.assert_not_awaited()

    @pytest.mark.parametrize(
        "paused, keys",
        [
            pytest.param(
                True, {_DISPLAY_KEY: _SANDBOX_ID, _STREAM_KEY: ""}, id="already-stopped"
            ),
            pytest.param(
                False,
                {_DISPLAY_KEY: _SANDBOX_ID, _STREAM_KEY: ""},
                id="already-stopped-and-running",
            ),
            pytest.param(
                False,
                {_DISPLAY_KEY: _SANDBOX_ID, _DISPLAY_LOCK_KEY: "token"},
                id="open-in-progress-on-a-running-box",
            ),
            pytest.param(True, {_DISPLAY_KEY: "sb-old"}, id="flag-for-a-replaced-box"),
            pytest.param(True, {}, id="screen-never-on"),
        ],
    )
    def test_a_stream_that_is_accounted_for_is_left_alone(
        self, paused: bool, keys: dict
    ):
        sb = _mock_sandbox()

        assert self._reconnect(_screen_redis(**keys), sb, paused=paused) is sb

        sb.commands.run.assert_not_awaited()


# ---------------------------------------------------------------------------
# Expert boxes: one persistent sandbox per hired expert
# ---------------------------------------------------------------------------

_EXPERT_ID = "exp-777"
_USER_ID = "user-42"
_EXPERT_SHELL_KEY = f"copilot:e2b:expert:{_EXPERT_ID}:shell"
_EXPERT_DISPLAY_KEY = f"copilot:e2b:expert:{_EXPERT_ID}:shell:display"
_EXPERT_STREAM_KEY = f"{_EXPERT_SHELL_KEY}:stream"
_EXPERT_ACTIVE_KEY = f"{_EXPERT_SHELL_KEY}:active"
_EXPERT_LEGACY_DESKTOP_KEY = f"copilot:e2b:expert:{_EXPERT_ID}:desktop"


def _info(
    sandbox_id: str, state: SandboxState, age_seconds: int = 0, kind: str = "shell"
) -> SimpleNamespace:
    return SimpleNamespace(
        sandbox_id=sandbox_id,
        state=state,
        started_at=datetime.now(timezone.utc) - timedelta(seconds=age_seconds),
        metadata={"autogpt_kind": kind},
    )


def _mock_list(infos: list) -> MagicMock:
    """``AsyncSandbox.list`` is sync and returns an async paginator.

    Answers per query kind: the owner's shell box and its pre-one-box
    desktop are looked up through the same call with different stamps.
    """

    def _list(query=None, **_):
        kind = (query.metadata or {}).get("autogpt_kind") if query else None
        paginator = MagicMock()
        paginator.next_items = AsyncMock(
            return_value=[
                i for i in infos if kind in (None, i.metadata["autogpt_kind"])
            ]
        )
        return paginator

    return MagicMock(side_effect=_list)


def _listed_kinds(mock_cls: MagicMock) -> list[str]:
    """Which stamps ``AsyncSandbox.list`` was asked for, in order."""
    return [
        c.kwargs["query"].metadata["autogpt_kind"] for c in mock_cls.list.call_args_list
    ]


def _keyed_redis(values: dict[str, str | None], decr_result: int = 0) -> AsyncMock:
    """Redis mock answering ``get`` per key (bytes, like the real client)."""
    r = AsyncMock()
    r.get = AsyncMock(side_effect=lambda key: (values.get(key) or "").encode() or None)
    r.set = AsyncMock(return_value=True)
    r.delete = AsyncMock()
    # Two scripts: ``_acquire_turn`` (INCR + EXPIRE) and ``_release_turn``
    # (DECR, and DEL when nothing is left).
    r.eval = AsyncMock(
        side_effect=lambda script, *_: 1 if "incr" in script else max(decr_result, 0)
    )
    return r


def _turn_acquires(redis: AsyncMock) -> list[str]:
    """Keys the acquire script ran against, in order."""
    return [
        call.args[2] for call in redis.eval.await_args_list if "incr" in call.args[0]
    ]


class TestConnectOwned:
    def test_checks_the_stamp_then_connects(self):
        owner = SandboxOwner(kind="expert", id=_EXPERT_ID)
        sb = _mock_sandbox("sb-expert", owner=owner)
        with _patch_sdk() as mock_cls:
            mock_cls.connect = AsyncMock(return_value=sb)
            result = asyncio.run(connect_owned("sb-expert", owner, _API_KEY))
        assert result is sb
        mock_cls.get_info.assert_awaited_once_with("sb-expert", api_key=_API_KEY)
        mock_cls.connect.assert_awaited_once_with(
            "sb-expert", api_key=_API_KEY, timeout=None
        )

    def test_timeout_reaches_the_connect(self):
        """A resumed box gets its own running-time limit, not the SDK's."""
        owner = SandboxOwner(kind="expert", id=_EXPERT_ID)
        sb = _mock_sandbox("sb-expert", owner=owner)
        with _patch_sdk() as mock_cls:
            mock_cls.connect = AsyncMock(return_value=sb)
            asyncio.run(connect_owned("sb-expert", owner, _API_KEY, timeout=900))
        mock_cls.connect.assert_awaited_once_with(
            "sb-expert", api_key=_API_KEY, timeout=900
        )

    @pytest.mark.parametrize(
        "reconnecting, pinned_for",
        [("user-a", "user-a"), ("user-b", None), (None, None)],
        ids=["its own user", "another user", "nobody"],
    )
    def test_a_box_is_pinned_only_for_the_user_it_was_created_for(
        self, reconnecting, pinned_for
    ):
        """Its creator's processes may still be running in it: re-pinned for
        whoever reconnects, they would get to act with that user's accounts.
        A mismatch keeps the box's egress and swaps nothing in."""
        owner = SandboxOwner(kind="expert", id=_EXPERT_ID)
        stamp = owner.creation_metadata(user_id="user-a")
        sb = MagicMock()
        with (
            _patch_sdk() as mock_cls,
            patch(
                "backend.copilot.tools.e2b_sandbox.connect_sandbox",
                AsyncMock(return_value=sb),
            ) as connect,
        ):
            mock_cls.get_info = AsyncMock(return_value=MagicMock(metadata=stamp))
            asyncio.run(connect_owned("sb-1", owner, _API_KEY, user_id=reconnecting))
        egress_owner = connect.await_args.args[2]
        assert egress_owner.user_id == pinned_for
        assert egress_owner.label == f"expert:{_EXPERT_ID}"

    @pytest.mark.parametrize(
        "stamp",
        [
            {"autogpt_owner": "expert:someone-else", "autogpt_kind": "shell"},
            {"autogpt_owner": f"expert:{_EXPERT_ID}", "autogpt_kind": "desktop"},
            {},
        ],
        ids=["another owner", "wrong kind", "unstamped"],
    )
    def test_refuses_a_box_that_is_not_the_owners_without_waking_it(self, stamp):
        """Any id connects under the platform key; the stamp is the only record.

        A connect resumes a paused box and extends its life, so the refusal
        has to come from the stamp alone, before any connect.
        """
        owner = SandboxOwner(kind="expert", id=_EXPERT_ID)
        with _patch_sdk() as mock_cls:
            mock_cls.get_info = AsyncMock(return_value=MagicMock(metadata=stamp))
            mock_cls.connect = AsyncMock()
            with pytest.raises(SandboxNotOwnedError):
                asyncio.run(connect_owned("sb-x", owner, _API_KEY))
        mock_cls.connect.assert_not_awaited()


class TestSandboxOwner:
    def test_expert_session_is_owned_by_the_expert(self):
        owner = SandboxOwner.for_session(_SESSION_ID, _EXPERT_ID)
        assert owner == SandboxOwner(kind="expert", id=_EXPERT_ID)
        assert owner.is_expert
        assert owner.key() == _EXPERT_SHELL_KEY
        assert owner.display_key() == _EXPERT_DISPLAY_KEY

    def test_plain_session_keys_are_unchanged(self):
        owner = SandboxOwner.for_session(_SESSION_ID, None)
        assert not owner.is_expert
        assert owner.key() == f"copilot:e2b:sandbox:{_SESSION_ID}"
        assert owner.display_key() == f"copilot:e2b:sandbox:{_SESSION_ID}:display"

    def test_expert_cache_outlives_session_cache(self):
        assert (
            SandboxOwner(kind="expert", id=_EXPERT_ID).ttl
            > SandboxOwner(kind="session", id=_SESSION_ID).ttl
        )

    def test_metadata_identifies_owner_and_kind(self):
        assert SandboxOwner(kind="expert", id=_EXPERT_ID).metadata() == {
            "autogpt_owner": f"expert:{_EXPERT_ID}",
            "autogpt_kind": "shell",
        }


class TestFindOwnedSandboxId:
    def test_session_owner_never_hits_the_api(self):
        with _patch_sdk() as mock_cls:
            result = asyncio.run(
                find_owned_sandbox_id(
                    SandboxOwner(kind="session", id=_SESSION_ID), _API_KEY
                )
            )
        assert result is None
        mock_cls.list.assert_not_called()

    def test_prefers_running_box_over_newer_paused_one(self):
        infos = [
            _info("sb-paused-new", SandboxState.PAUSED, age_seconds=10),
            _info("sb-running-old", SandboxState.RUNNING, age_seconds=500),
        ]
        with _patch_sdk() as mock_cls:
            mock_cls.list = _mock_list(infos)
            result = asyncio.run(
                find_owned_sandbox_id(
                    SandboxOwner(kind="expert", id=_EXPERT_ID), _API_KEY
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
        with _patch_sdk() as mock_cls:
            mock_cls.list = _mock_list(infos)
            result = asyncio.run(
                find_owned_sandbox_id(
                    SandboxOwner(kind="expert", id=_EXPERT_ID), _API_KEY
                )
            )
        assert result == "sb-new"

    def test_api_failure_is_an_error_not_a_miss(self):
        """ "Unknown" must never be read as "none": the caller would create a
        second box and fork the expert's durable state."""
        with _patch_sdk() as mock_cls:
            mock_cls.list = MagicMock(side_effect=RuntimeError("e2b down"))
            with pytest.raises(SandboxLookupError, match="e2b down"):
                asyncio.run(
                    find_owned_sandbox_id(
                        SandboxOwner(kind="expert", id=_EXPERT_ID), _API_KEY
                    )
                )


class TestExpertShellBox:
    def test_recovers_expert_box_from_e2b_when_redis_is_empty(self):
        """Redis is only a cache for an expert's box; E2B metadata is the record."""
        sb = _mock_sandbox(
            "sb-expert", owner=SandboxOwner(kind="expert", id=_EXPERT_ID)
        )
        redis = _keyed_redis({})
        with (
            _patch_sdk() as mock_cls,
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
        # Resumed under the box's own running-time limit, not the SDK's.
        mock_cls.connect.assert_awaited_once_with(
            "sb-expert", api_key=_API_KEY, timeout=_TIMEOUT
        )
        mock_cls.create.assert_not_awaited()
        # Re-cached under the expert key, never the session key.
        keys = {call.args[0] for call in redis.set.await_args_list}
        assert _EXPERT_SHELL_KEY in keys
        assert f"copilot:e2b:sandbox:{_SESSION_ID}" not in keys
        # This turn is counted so a concurrent turn's end cannot pause the box.
        assert _turn_acquires(redis) == [_EXPERT_ACTIVE_KEY]

    def test_lookup_failure_never_creates_a_second_expert_box(self):
        redis = _keyed_redis({})
        with (
            _patch_sdk() as mock_cls,
            patch("backend.copilot.tools.e2b_sandbox.ensure_template", AsyncMock()),
            _patch_redis(redis),
        ):
            mock_cls.list = MagicMock(side_effect=RuntimeError("e2b down"))
            mock_cls.create = AsyncMock()
            with pytest.raises(SandboxLookupError):
                asyncio.run(
                    get_or_create_sandbox(
                        _SESSION_ID, _API_KEY, timeout=_TIMEOUT, expert_id=_EXPERT_ID
                    )
                )
        mock_cls.create.assert_not_awaited()

    def test_listed_expert_box_gets_a_second_reconnect_before_being_forked(self):
        """One transient connect failure must not leave the durable box behind."""
        sb = _mock_sandbox(
            "sb-expert", owner=SandboxOwner(kind="expert", id=_EXPERT_ID)
        )
        redis = _keyed_redis({})
        with (
            _patch_sdk() as mock_cls,
            patch("backend.copilot.tools.e2b_sandbox.asyncio.sleep", AsyncMock()),
            _patch_redis(redis),
        ):
            mock_cls.list = _mock_list([_info("sb-expert", SandboxState.PAUSED)])
            mock_cls.connect = AsyncMock(side_effect=[RuntimeError("502"), sb])
            mock_cls.create = AsyncMock()
            result = asyncio.run(
                get_or_create_sandbox(
                    _SESSION_ID, _API_KEY, timeout=_TIMEOUT, expert_id=_EXPERT_ID
                )
            )
        assert result is sb
        assert mock_cls.connect.await_count == 2
        mock_cls.create.assert_not_awaited()

    def test_count_turn_false_defers_the_turn_count(self):
        sb = _mock_sandbox(
            "sb-expert", owner=SandboxOwner(kind="expert", id=_EXPERT_ID)
        )
        redis = _keyed_redis({_EXPERT_SHELL_KEY: "sb-expert"})
        with (
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            asyncio.run(
                get_or_create_sandbox(
                    _SESSION_ID,
                    _API_KEY,
                    timeout=_TIMEOUT,
                    expert_id=_EXPERT_ID,
                    count_turn=False,
                )
            )
            assert _turn_acquires(redis) == []
            asyncio.run(count_expert_turn(_SESSION_ID, _EXPERT_ID))
        assert _turn_acquires(redis) == [_EXPERT_ACTIVE_KEY]

    def test_counting_a_turn_arms_the_expiry_in_the_same_script(self):
        """An INCR without its EXPIRE would be a count nothing releases."""
        redis = _keyed_redis({})
        with _patch_redis(redis):
            asyncio.run(count_expert_turn(_SESSION_ID, _EXPERT_ID))
        script, nkeys, key, ttl = redis.eval.await_args.args
        assert nkeys == 1 and key == _EXPERT_ACTIVE_KEY
        assert "incr" in script and "expire" in script and ttl > 0

    def test_a_turn_that_cannot_be_counted_does_not_get_the_box(self):
        """An uncounted turn's release would decrement someone else's count
        and could pause the box under them, so the count failure surfaces."""
        sb = _mock_sandbox(
            "sb-expert", owner=SandboxOwner(kind="expert", id=_EXPERT_ID)
        )
        redis = _keyed_redis({_EXPERT_SHELL_KEY: "sb-expert"})
        redis.eval = AsyncMock(side_effect=ConnectionError("redis down"))
        with (
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            with pytest.raises(ConnectionError):
                asyncio.run(
                    get_or_create_sandbox(
                        _SESSION_ID, _API_KEY, timeout=_TIMEOUT, expert_id=_EXPERT_ID
                    )
                )
        # The failed acquire was the only script; no release ran for it.
        assert [
            c.args[0] for c in redis.eval.await_args_list if "decr" in c.args[0]
        ] == []

    def test_creates_expert_box_with_home_and_shared_volumes(self):
        sb = _mock_sandbox("sb-expert-new")
        sb.commands = MagicMock()
        sb.commands.run = AsyncMock()
        redis = _keyed_redis({})
        mounts = workspace_volume_mounts(_USER_ID, _EXPERT_ID)
        with (
            _patch_sdk() as mock_cls,
            # Volumes already exist -> mounted by name.  Patched where the
            # module looks it up, so no real E2B call can sneak through.
            patch(
                "backend.copilot.tools.e2b_sandbox.resolve_volume",
                AsyncMock(side_effect=lambda name, key: name),
            ),
            _patch_redis(redis),
        ):
            mock_cls.list = _mock_list([])
            mock_cls.create = AsyncMock(return_value=sb)
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
            "service": "autogpt-platform",
            "autogpt_owner": f"expert:{_EXPERT_ID}",
            "autogpt_kind": "shell",
            "autogpt_source": "copilot",
            "autogpt_env": deployment_env(),
            "autogpt_session": _SESSION_ID,
            "autogpt_expert": _EXPERT_ID,
            "autogpt_template": "base",
            "autogpt_mounts": "attached",
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
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.create = AsyncMock(return_value=sb)
            asyncio.run(get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT))

        kwargs = mock_cls.create.call_args.kwargs
        assert kwargs["metadata"] == {
            "service": "autogpt-platform",
            "autogpt_owner": f"session:{_SESSION_ID}",
            "autogpt_kind": "shell",
            "autogpt_source": "copilot",
            "autogpt_env": deployment_env(),
            "autogpt_session": _SESSION_ID,
            "autogpt_template": "base",
            "autogpt_mounts": "none",
        }
        assert kwargs["volume_mounts"] is None
        mock_cls.list.assert_not_called()
        assert _turn_acquires(redis) == []


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
        # One script does the DECR and, at zero, the DEL: no window for a turn
        # that starts in between to lose its count.
        script, _, key = redis.eval.await_args.args
        assert key == _EXPERT_ACTIVE_KEY and "decr" in script and "del" in script

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
            _patch_sdk() as mock_cls,
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
        redis.eval.assert_not_awaited()


class TestExpertKill:
    def test_deleting_an_expert_chat_leaves_the_expert_box_alone(self):
        """kill_sandbox only knows session keys; an expert box has none."""
        redis = _keyed_redis({_EXPERT_SHELL_KEY: "sb-expert"})
        with (
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock()
            ok = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))
        assert ok is False
        mock_cls.connect.assert_not_awaited()

    def test_archive_kills_the_box_and_clears_its_state(self):
        expert = SandboxOwner(kind="expert", id=_EXPERT_ID)
        box = _mock_sandbox("sb-box", owner=expert)
        redis = _keyed_redis({_EXPERT_SHELL_KEY: "sb-box"})
        with (
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=box)
            killed = asyncio.run(kill_expert_sandbox(_EXPERT_ID, _API_KEY))

        assert killed is True
        box.kill.assert_awaited_once()
        deleted = {key for call in redis.delete.await_args_list for key in call.args}
        # The cached id, and with it the screen flag, stream password and
        # turn counter; the pre-one-box desktop key is swept along.
        assert deleted == {
            _EXPERT_SHELL_KEY,
            _EXPERT_DISPLAY_KEY,
            _EXPERT_STREAM_KEY,
            _EXPERT_ACTIVE_KEY,
            _EXPERT_LEGACY_DESKTOP_KEY,
        }
        # The cached id was enough; no lookup for the shell box.
        assert "shell" not in _listed_kinds(mock_cls)

    def test_archive_falls_back_to_e2b_metadata_for_a_forgotten_box(self):
        box = _mock_sandbox("sb-box", owner=SandboxOwner(kind="expert", id=_EXPERT_ID))
        redis = _keyed_redis({})
        with (
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.list = _mock_list([_info("sb-box", SandboxState.PAUSED)])
            mock_cls.connect = AsyncMock(return_value=box)
            killed = asyncio.run(kill_expert_sandbox(_EXPERT_ID, _API_KEY))

        assert killed is True
        box.kill.assert_awaited_once()
        assert _listed_kinds(mock_cls).count("shell") == 1

    def test_failed_kill_keeps_cache_for_retry(self):
        _mock_sandbox("sb-box", owner=SandboxOwner(kind="expert", id=_EXPERT_ID))
        redis = _keyed_redis({_EXPERT_SHELL_KEY: "sb-box"})
        with (
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.list = _mock_list([])
            mock_cls.connect = AsyncMock(side_effect=RuntimeError("boom"))
            killed = asyncio.run(kill_expert_sandbox(_EXPERT_ID, _API_KEY))
        assert killed is False
        deleted = {key for call in redis.delete.await_args_list for key in call.args}
        assert _EXPERT_SHELL_KEY not in deleted


class TestKillSandboxForgetsScreen:
    def test_deleting_a_chat_drops_its_screen_flag_and_turn_counter(self):
        """The screen lives in the same box, so nothing else needs killing,
        but the flag that says it was on must not outlive the box."""
        box = _mock_sandbox("sb-box")
        redis = _keyed_redis({f"copilot:e2b:sandbox:{_SESSION_ID}": "sb-box"})
        with (
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(return_value=box)
            ok = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))
        assert ok is True
        box.kill.assert_awaited_once()
        deleted = {key for call in redis.delete.await_args_list for key in call.args}
        assert deleted == {
            f"copilot:e2b:sandbox:{_SESSION_ID}",
            f"copilot:e2b:sandbox:{_SESSION_ID}:display",
            f"copilot:e2b:sandbox:{_SESSION_ID}:stream",
            f"copilot:e2b:sandbox:{_SESSION_ID}:active",
            f"copilot:e2b:desktop:{_SESSION_ID}",
        }


class TestLegacyDesktopSweep:
    """Boxes from before one box per owner: a separate desktop, stamped
    ``autogpt_kind=desktop``, that nothing opens any more.  The kill paths
    take it along so it does not sit paused on E2B, browser profile and all."""

    def test_deleting_a_chat_kills_its_old_desktop_by_id_without_waking_it(self):
        box = _mock_sandbox("sb-box")
        redis = _keyed_redis({f"copilot:e2b:sandbox:{_SESSION_ID}": "sb-box"})
        with _patch_sdk() as mock_cls, _patch_redis(redis):
            mock_cls.list = _mock_list(
                [_info("sb-old-desktop", SandboxState.PAUSED, kind="desktop")]
            )
            mock_cls.connect = AsyncMock(return_value=box)
            ok = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))
        assert ok is True
        box.kill.assert_awaited_once()
        mock_cls.kill.assert_awaited_once_with("sb-old-desktop", api_key=_API_KEY)
        # Connected once, to the shell box; the desktop was killed by id.
        mock_cls.connect.assert_awaited_once()
        query = mock_cls.list.call_args.kwargs["query"].metadata
        assert query["autogpt_kind"] == "desktop"
        assert query["autogpt_owner"] == f"session:{_SESSION_ID}"

    def test_archiving_an_expert_sweeps_its_old_desktop_too(self):
        expert = SandboxOwner(kind="expert", id=_EXPERT_ID)
        box = _mock_sandbox("sb-box", owner=expert)
        redis = _keyed_redis({_EXPERT_SHELL_KEY: "sb-box"})
        with _patch_sdk() as mock_cls, _patch_redis(redis):
            mock_cls.list = _mock_list(
                [_info("sb-old-desktop", SandboxState.PAUSED, kind="desktop")]
            )
            mock_cls.connect = AsyncMock(return_value=box)
            killed = asyncio.run(kill_expert_sandbox(_EXPERT_ID, _API_KEY))
        assert killed is True
        box.kill.assert_awaited_once()
        mock_cls.kill.assert_awaited_once_with("sb-old-desktop", api_key=_API_KEY)
        deleted = {key for call in redis.delete.await_args_list for key in call.args}
        assert _EXPERT_LEGACY_DESKTOP_KEY in deleted

    def test_an_expert_with_only_an_old_desktop_still_counts_as_cleaned(self):
        redis = _keyed_redis({})
        with _patch_sdk() as mock_cls, _patch_redis(redis):
            mock_cls.list = _mock_list(
                [_info("sb-old-desktop", SandboxState.PAUSED, kind="desktop")]
            )
            mock_cls.connect = AsyncMock()
            killed = asyncio.run(kill_expert_sandbox(_EXPERT_ID, _API_KEY))
        assert killed is True
        mock_cls.kill.assert_awaited_once_with("sb-old-desktop", api_key=_API_KEY)
        mock_cls.connect.assert_not_awaited()

    def test_a_swept_old_desktop_does_not_hide_a_failed_shell_kill(self):
        """The current box is what the caller asked about; reporting it
        killed because an old desktop went would end its retries."""
        redis = _keyed_redis({f"copilot:e2b:sandbox:{_SESSION_ID}": "sb-box"})
        with _patch_sdk() as mock_cls, _patch_redis(redis):
            mock_cls.list = _mock_list(
                [_info("sb-old-desktop", SandboxState.PAUSED, kind="desktop")]
            )
            mock_cls.connect = AsyncMock(side_effect=ConnectionError("gone"))
            ok = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))
        assert ok is False
        mock_cls.kill.assert_awaited_once_with("sb-old-desktop", api_key=_API_KEY)

    def test_an_expert_lookup_failure_is_not_done_whatever_was_swept(self):
        redis = _keyed_redis({})
        with _patch_sdk() as mock_cls, _patch_redis(redis):
            mock_cls.list = MagicMock(side_effect=RuntimeError("e2b down"))
            killed = asyncio.run(kill_expert_sandbox(_EXPERT_ID, _API_KEY))
        assert killed is False

    def test_a_failed_sweep_does_not_stop_the_shell_kill(self):
        box = _mock_sandbox("sb-box")
        redis = _keyed_redis({f"copilot:e2b:sandbox:{_SESSION_ID}": "sb-box"})
        with _patch_sdk() as mock_cls, _patch_redis(redis):
            mock_cls.list = MagicMock(side_effect=RuntimeError("e2b down"))
            mock_cls.connect = AsyncMock(return_value=box)
            ok = asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY))
        assert ok is True
        box.kill.assert_awaited_once()
        mock_cls.kill.assert_not_awaited()


class TestExpertBoxRecovery:
    def test_listed_box_that_is_gone_falls_through_to_create(self):
        """E2B may still list a box mid-teardown; the loop must create a
        replacement instead of spinning on the same id."""
        fresh = _mock_sandbox("sb-fresh")
        # Stamped as the expert's, so the refusal is not what stops the reconnect.
        _mock_sandbox("sb-dead", owner=SandboxOwner(kind="expert", id=_EXPERT_ID))
        redis = _keyed_redis({})
        with (
            _patch_sdk() as mock_cls,
            patch("backend.copilot.tools.e2b_sandbox.asyncio.sleep", AsyncMock()),
            _patch_redis(redis),
        ):
            mock_cls.list = _mock_list([_info("sb-dead", SandboxState.PAUSED)])
            mock_cls.connect = AsyncMock(side_effect=SandboxNotFoundException("gone"))
            mock_cls.create = AsyncMock(return_value=fresh)
            result = asyncio.run(
                get_or_create_sandbox(
                    _SESSION_ID, _API_KEY, timeout=_TIMEOUT, expert_id=_EXPERT_ID
                )
            )
        assert result is fresh
        mock_cls.connect.assert_awaited_once_with(
            "sb-dead", api_key=_API_KEY, timeout=_TIMEOUT
        )
        mock_cls.create.assert_awaited_once()
        assert _listed_kinds(mock_cls).count("shell") <= 2

    def test_a_transient_error_twice_surfaces_instead_of_replacing_the_box(self):
        """Whoever owns it: after the one retry the caller hears about it,
        and the box, with everything on it, is still theirs next time."""
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with (
            _patch_sdk() as mock_cls,
            patch("backend.copilot.tools.e2b_sandbox.asyncio.sleep", AsyncMock()),
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(side_effect=RuntimeError("502"))
            mock_cls.create = AsyncMock()
            with pytest.raises(RuntimeError, match="502"):
                asyncio.run(
                    get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
                )
        assert mock_cls.connect.await_count == 2
        mock_cls.create.assert_not_awaited()
        redis.delete.assert_not_awaited()

    def test_a_session_box_also_gets_its_one_retry(self):
        """The session's box carries its screen now, so a blip must not
        cost it the box any more than it would an expert."""
        sb = _mock_sandbox()
        redis = _mock_redis(stored_sandbox_id=_SANDBOX_ID)
        with (
            _patch_sdk() as mock_cls,
            patch("backend.copilot.tools.e2b_sandbox.asyncio.sleep", AsyncMock()),
            _patch_redis(redis),
        ):
            mock_cls.connect = AsyncMock(side_effect=[RuntimeError("502"), sb])
            mock_cls.create = AsyncMock()
            result = asyncio.run(
                get_or_create_sandbox(_SESSION_ID, _API_KEY, timeout=_TIMEOUT)
            )
        assert result is sb
        mock_cls.create.assert_not_awaited()

    def test_mounts_survive_transient_create_failures(self):
        """Only the final attempt goes volume-less; a slow first attempt must
        not quietly cost an expert its durable home for 30 days."""
        sb = _mock_sandbox("sb-late")
        sb.commands = MagicMock()
        sb.commands.run = AsyncMock()
        redis = _keyed_redis({})
        mounts = workspace_volume_mounts(_USER_ID, _EXPERT_ID)
        with (
            _patch_sdk() as mock_cls,
            patch(
                "backend.copilot.tools.e2b_sandbox.resolve_volume",
                new=AsyncMock(side_effect=lambda name, key: name),
            ),
            patch("backend.copilot.tools.e2b_sandbox.asyncio.sleep", new=AsyncMock()),
            _patch_redis(redis),
        ):
            mock_cls.list = _mock_list([])
            mock_cls.create = AsyncMock(
                side_effect=[asyncio.TimeoutError(), RuntimeError("busy"), sb]
            )
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
        attempts = [c.kwargs for c in mock_cls.create.call_args_list]
        assert len(attempts) == _SANDBOX_CREATE_MAX_RETRIES
        assert attempts[0]["volume_mounts"] == mounts
        assert attempts[1]["volume_mounts"] == mounts
        assert attempts[2]["volume_mounts"] is None
        assert attempts[0]["metadata"]["autogpt_mounts"] == "attached"
        assert attempts[2]["metadata"]["autogpt_mounts"] == "none"

    def test_release_fails_closed_when_redis_is_unavailable(self):
        """Without the counter we cannot rule out a concurrent turn, so the
        box is left running for the lifecycle timeout to pause."""
        sb = _mock_sandbox()
        redis = _keyed_redis({})
        redis.eval = AsyncMock(side_effect=ConnectionError("redis down"))
        with _patch_redis(redis):
            ok = asyncio.run(
                pause_sandbox_direct(sb, _SESSION_ID, expert_id=_EXPERT_ID)
            )
        assert ok is False
        sb.pause.assert_not_awaited()


class TestProxyCredentialIsRevoked:
    """A box that is paused or killed will not present its proxy credential
    again; left alone, the record would keep naming its owner to the proxy."""

    _FORGET = "backend.copilot.tools.e2b_sandbox.forget_sandbox"

    def test_on_kill(self):
        sb = _mock_sandbox()
        with (
            _patch_sdk() as mock_cls,
            _patch_redis(_mock_redis(stored_sandbox_id=_SANDBOX_ID)),
            patch(self._FORGET, AsyncMock()) as forget,
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            assert asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY)) is True
        forget.assert_awaited_once_with(_SANDBOX_ID)

    def test_on_pause(self):
        sb = _mock_sandbox()
        with (
            _patch_sdk() as mock_cls,
            _patch_redis(_mock_redis(stored_sandbox_id=_SANDBOX_ID)),
            patch(self._FORGET, AsyncMock()) as forget,
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            assert asyncio.run(pause_sandbox(_SESSION_ID, _API_KEY)) is True
        forget.assert_awaited_once_with(_SANDBOX_ID)

    def test_on_the_turn_end_pause(self):
        sb = _mock_sandbox()
        with (
            _patch_redis(_mock_redis()),
            patch(self._FORGET, AsyncMock()) as forget,
        ):
            assert asyncio.run(pause_sandbox_direct(sb, _SESSION_ID)) is True
        forget.assert_awaited_once_with(sb.sandbox_id)

    def test_not_when_the_kill_failed(self):
        """The box is still there and may yet egress: keep its credential."""
        sb = _mock_sandbox()
        sb.kill = AsyncMock(side_effect=RuntimeError("e2b error"))
        with (
            _patch_sdk() as mock_cls,
            _patch_redis(_mock_redis(stored_sandbox_id=_SANDBOX_ID)),
            patch(self._FORGET, AsyncMock()) as forget,
        ):
            mock_cls.connect = AsyncMock(return_value=sb)
            assert asyncio.run(kill_sandbox(_SESSION_ID, _API_KEY)) is False
        forget.assert_not_awaited()


# ---------------------------------------------------------------------------
# The box's own environment behind the swap proxy
# ---------------------------------------------------------------------------


class TestCreationEnvBehindTheSwapProxy:
    """A new box behind the proxy starts with placeholders in its own env, for
    what does not run through a command, and those credentials are granted to
    it; without the proxy it starts with no integration env at all, as
    before."""

    def _create(self, *, proxy: str | None, user_id: str | None) -> dict:
        new_sb = _mock_sandbox("sb-new")
        redis = _mock_redis(set_nx_result=True, stored_sandbox_id=None)
        with (
            _patch_sdk() as mock_cls,
            _patch_redis(redis),
            patch(
                "backend.copilot.tools.e2b_sandbox.proxy_address", return_value=proxy
            ),
            patch(
                "backend.copilot.tools.e2b_sandbox.placeholder_grants",
                AsyncMock(return_value={"github": "cred-default"}),
            ) as lookup,
            patch(
                "backend.copilot.tools.e2b_sandbox.grant_to_box", AsyncMock()
            ) as grant,
        ):
            mock_cls.create = AsyncMock(return_value=new_sb)
            asyncio.run(
                get_or_create_sandbox(
                    _SESSION_ID, _API_KEY, timeout=_TIMEOUT, user_id=user_id
                )
            )
        _, kwargs = mock_cls.create.call_args
        self.lookup, self.grant = lookup, grant
        return kwargs

    def test_behind_the_proxy_the_box_starts_with_granted_placeholders(self):
        kwargs = self._create(proxy="proxy:1080", user_id="user-a")
        assert kwargs["envs"]["GH_TOKEN"] == "hsurr:github:cred-default"
        assert kwargs["envs"]["GITHUB_TOKEN"] == "hsurr:github:cred-default"
        self.lookup.assert_awaited_once_with("user-a")
        self.grant.assert_awaited_once_with("sb-new", {"github": "cred-default"})

    def test_without_the_proxy_the_create_call_is_unchanged(self):
        kwargs = self._create(proxy=None, user_id="user-a")
        assert "envs" not in kwargs
        self.lookup.assert_not_awaited()
        self.grant.assert_not_awaited()

    def test_a_box_with_no_user_gets_none(self):
        kwargs = self._create(proxy="proxy:1080", user_id=None)
        assert "envs" not in kwargs
        self.lookup.assert_not_awaited()
