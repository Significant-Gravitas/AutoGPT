"""Tests for backend.copilot.computer: describe without waking, screen on in place."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from e2b import SandboxState

from backend.blocks.desktop._api import DesktopStream
from backend.blocks.desktop._common import SHARED_PATH, WORKSPACE_PATH
from backend.copilot.computer import (
    ComputerInfo,
    computer_owner,
    describe_computer,
    mounts_for,
    open_desktop,
    screen_is_on,
)
from backend.copilot.tools.e2b_sandbox import SandboxLookupError, SandboxOwner

_C = "backend.copilot.computer"
_USER, _EXPERT, _SESSION = "user-1", "exp-1", "sess-1"


def _info(sandbox_id: str, state: SandboxState, mounts: str = "attached"):
    return SimpleNamespace(
        sandbox_id=sandbox_id,
        state=state,
        started_at=datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc),
        cpu_count=1,
        memory_mb=2048,
        template_id="agpt-desktop-1x2",
        metadata={"autogpt_kind": "shell", "autogpt_mounts": mounts},
    )


@pytest.fixture(autouse=True)
def _lock_release_runs_on_the_mock():
    """Route the desktop lock's release script to the mocked client."""
    with patch(
        f"{_C}.delete_if_owner",
        lambda client, **kwargs: client.delete_if_owner(**kwargs),
    ):
        yield


def _redis(display: str | None, lock_free: bool = True, stream: str | None = None):
    r = MagicMock()
    r.get = AsyncMock(
        side_effect=lambda key: stream if key.endswith(":stream") else display
    )
    r.set = AsyncMock(return_value=lock_free)
    r.delete = AsyncMock()
    r.delete_if_owner = AsyncMock(return_value=1)
    return r


def _sandbox(sandbox_id: str = "sb-1"):
    sb = MagicMock()
    sb.sandbox_id = sandbox_id
    return sb


def _desktop(sandbox_id: str = "sb-1", mounted: bool = True):
    d = MagicMock()
    d.ensure_display = AsyncMock()
    d.ensure_persistent_home = AsyncMock()
    d.is_workspace_mounted = AsyncMock(return_value=mounted)
    d.start_stream = AsyncMock(return_value=(_LIVE_STREAM(sandbox_id), "secret"))
    return d


def _LIVE_STREAM(sandbox_id: str) -> DesktopStream:
    return DesktopStream(
        url="https://6080-x.e2b.app/vnc.html?password=secret", sandbox_id=sandbox_id
    )


@pytest.fixture(autouse=True)
def _owner_bound_links():
    with patch(
        f"{_C}.create_preview_link",
        side_effect=lambda user_id, url: f"preview://{user_id}/token",
    ) as link:
        yield link


class TestComputerOwner:
    def test_expert_session_is_the_experts_computer(self):
        assert computer_owner(_SESSION, _EXPERT) == SandboxOwner(
            kind="expert", id=_EXPERT
        )
        assert computer_owner(_SESSION, None) == SandboxOwner(
            kind="session", id=_SESSION
        )

    def test_mounts_follow_the_shells_rule(self):
        # An expert always has its home; the shared volume needs a user.
        assert set(mounts_for(None, _EXPERT)) == {WORKSPACE_PATH}
        assert set(mounts_for(_USER, _EXPERT)) == {WORKSPACE_PATH, SHARED_PATH}
        assert set(mounts_for(_USER, None)) == {WORKSPACE_PATH}
        assert mounts_for(None, None) == {}


class TestDescribeComputer:
    @pytest.mark.asyncio
    async def test_reports_the_box_and_its_screen_without_connecting(self):
        owner = SandboxOwner(kind="expert", id=_EXPERT)
        with (
            patch(f"{_C}.chat_config") as cfg,
            patch(
                f"{_C}.list_owned_sandboxes",
                AsyncMock(return_value=[_info("sb-1", SandboxState.PAUSED)]),
            ) as list_mock,
            patch(f"{_C}.get_redis_async", AsyncMock(return_value=_redis("sb-1"))),
            patch(f"{_C}.DesktopSession") as desktop_cls,
        ):
            cfg.active_e2b_api_key = "k"
            info = await describe_computer(owner, mounts_for(_USER, _EXPERT))

        assert isinstance(info, ComputerInfo)
        assert info.owner_kind == "expert" and info.e2b_active
        assert info.box and info.box.state == "paused" and info.box.mounts_attached
        assert info.box.cpu_count == 1 and info.box.memory_mb == 2048
        assert info.screen_on is True
        assert info.mounts[SHARED_PATH].startswith("autogpt-user-")
        list_mock.assert_awaited_once_with(owner, "k")
        # Describing must never resume a paused box.
        desktop_cls.assert_not_called()

    @pytest.mark.asyncio
    async def test_screen_flag_for_a_replaced_box_does_not_count(self):
        owner = SandboxOwner(kind="session", id=_SESSION)
        with (
            patch(f"{_C}.chat_config") as cfg,
            patch(
                f"{_C}.list_owned_sandboxes",
                AsyncMock(return_value=[_info("sb-new", SandboxState.RUNNING)]),
            ),
            patch(f"{_C}.get_redis_async", AsyncMock(return_value=_redis("sb-old"))),
        ):
            cfg.active_e2b_api_key = "k"
            info = await describe_computer(owner, {})
        assert info.box and info.box.state == "running"
        assert info.screen_on is False

    @pytest.mark.asyncio
    async def test_a_failed_listing_shows_nothing_rather_than_failing(self):
        owner = SandboxOwner(kind="expert", id=_EXPERT)
        with (
            patch(f"{_C}.chat_config") as cfg,
            patch(
                f"{_C}.list_owned_sandboxes",
                AsyncMock(side_effect=SandboxLookupError("e2b down")),
            ),
        ):
            cfg.active_e2b_api_key = "k"
            info = await describe_computer(owner, {})
        assert info.e2b_active and info.box is None and info.screen_on is False

    @pytest.mark.asyncio
    async def test_without_e2b_it_says_so_and_lists_nothing(self):
        owner = SandboxOwner(kind="session", id=_SESSION)
        with (
            patch(f"{_C}.chat_config") as cfg,
            patch(f"{_C}.list_owned_sandboxes", AsyncMock()) as list_mock,
        ):
            cfg.active_e2b_api_key = None
            info = await describe_computer(owner, {})
        assert info.e2b_active is False and info.box is None
        assert info.screen_on is False
        list_mock.assert_not_awaited()


class TestOpenDesktop:
    def _patches(self, redis, sandbox, desktop):
        desktop_cls = MagicMock(return_value=desktop)
        return (
            patch(f"{_C}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_C}.get_or_create_owner_sandbox", AsyncMock(return_value=sandbox)),
            patch(f"{_C}.DesktopSession", desktop_cls),
            patch(f"{_C}.chat_config"),
        )

    @pytest.mark.asyncio
    async def test_needs_a_user_to_issue_the_link_to(self):
        with pytest.raises(ValueError, match="authenticated user"):
            await open_desktop(
                SandboxOwner(kind="expert", id=_EXPERT), {}, "k", user_id=None
            )

    @pytest.mark.asyncio
    async def test_turns_the_screen_on_in_the_owners_own_box(self):
        owner = SandboxOwner(kind="expert", id=_EXPERT)
        mounts = mounts_for(_USER, _EXPERT)
        redis, sandbox, desktop = _redis(None), _sandbox("sb-1"), _desktop("sb-1")
        redis_p, get_p, cls_p, cfg_p = self._patches(redis, sandbox, desktop)
        with redis_p, get_p as get_mock, cls_p as desktop_cls, cfg_p as cfg:
            cfg.e2b_sandbox_timeout = 420
            cfg.e2b_sandbox_template = "agpt-desktop-1x2"
            cfg.e2b_sandbox_on_timeout = "pause"
            stream, first_time, shared = await open_desktop(
                owner, mounts, "k", user_id=_USER, session_id=_SESSION
            )

        assert first_time and shared and stream.sandbox_id == "sb-1"
        # The same box a turn would use, found or created the same way, but
        # not counted as a turn so the agent's turn-end pause still fires.
        get_mock.assert_awaited_once_with(
            owner,
            "k",
            timeout=420,
            template="agpt-desktop-1x2",
            on_timeout="pause",
            volume_mounts=mounts,
            user_id=_USER,
            session_id=_SESSION,
            count_turn=False,
        )
        desktop_cls.assert_called_once_with(sandbox)
        desktop.ensure_display.assert_awaited_once()
        desktop.ensure_persistent_home.assert_awaited_once()
        desktop.start_stream.assert_awaited_once()
        # The screen flag is remembered against this box's id, under the owner.
        flag = [c for c in redis.set.await_args_list if c.args[0].endswith(":display")]
        assert flag and flag[0].args[:2] == (
            f"copilot:e2b:expert:{_EXPERT}:shell:display",
            "sb-1",
        )

    @pytest.mark.asyncio
    async def test_hands_out_an_owner_bound_link_never_the_password_url(
        self, _owner_bound_links
    ):
        owner = SandboxOwner(kind="session", id=_SESSION)
        redis, sandbox, desktop = _redis(None), _sandbox("sb-1"), _desktop("sb-1")
        redis_p, get_p, cls_p, cfg_p = self._patches(redis, sandbox, desktop)
        with redis_p, get_p, cls_p, cfg_p:
            stream, _, _ = await open_desktop(owner, {}, "k", user_id=_USER)
        # The live URL went into the link for this user, and only the link
        # comes out.
        _owner_bound_links.assert_called_once_with(_USER, _LIVE_STREAM("sb-1").url)
        assert stream.url == f"preview://{_USER}/token"
        assert stream.requires_auth is True
        assert "secret" not in stream.model_dump_json()

    @pytest.mark.asyncio
    async def test_first_open_starts_a_stream_under_a_new_password(self):
        owner = SandboxOwner(kind="session", id=_SESSION)
        redis, sandbox, desktop = _redis(None), _sandbox("sb-1"), _desktop("sb-1")
        redis_p, get_p, cls_p, cfg_p = self._patches(redis, sandbox, desktop)
        with redis_p, get_p, cls_p, cfg_p as cfg:
            cfg.e2b_sandbox_timeout = 420
            await open_desktop(owner, {}, "k", user_id=_USER)
        desktop.start_stream.assert_awaited_once_with(None)
        # Remembered off the box, for as long as the box could keep running.
        redis.set.assert_any_await(
            f"copilot:e2b:sandbox:{_SESSION}:stream", "secret", ex=420
        )

    @pytest.mark.asyncio
    async def test_reopen_hands_back_the_same_stream_while_the_box_kept_running(
        self,
    ):
        owner = SandboxOwner(kind="session", id=_SESSION)
        redis = _redis("sb-1", stream="issued-before")
        sandbox, desktop = _sandbox("sb-1"), _desktop("sb-1")
        redis_p, get_p, cls_p, cfg_p = self._patches(redis, sandbox, desktop)
        with redis_p, get_p, cls_p, cfg_p:
            await open_desktop(owner, {}, "k", user_id=_USER)
        desktop.start_stream.assert_awaited_once_with("issued-before")

    @pytest.mark.asyncio
    async def test_a_stream_stopped_at_the_pause_reopens_under_a_new_password(self):
        """The pause leaves an empty marker where the password was: to an open
        that is no password, so the stack restarts under a fresh one."""
        owner = SandboxOwner(kind="session", id=_SESSION)
        redis = _redis("sb-1", stream="")
        sandbox, desktop = _sandbox("sb-1"), _desktop("sb-1")
        redis_p, get_p, cls_p, cfg_p = self._patches(redis, sandbox, desktop)
        with redis_p, get_p, cls_p, cfg_p:
            await open_desktop(owner, {}, "k", user_id=_USER)
        desktop.start_stream.assert_awaited_once_with(None)

    @pytest.mark.asyncio
    async def test_a_password_left_over_from_a_replaced_box_is_not_reused(self):
        """The screen flag names another box: whatever password Redis still
        holds belonged to that one."""
        owner = SandboxOwner(kind="session", id=_SESSION)
        redis = _redis("sb-old", stream="issued-before")
        sandbox, desktop = _sandbox("sb-new"), _desktop("sb-new")
        redis_p, get_p, cls_p, cfg_p = self._patches(redis, sandbox, desktop)
        with redis_p, get_p, cls_p, cfg_p:
            await open_desktop(owner, {}, "k", user_id=_USER)
        desktop.start_stream.assert_awaited_once_with(None)

    @pytest.mark.asyncio
    async def test_second_open_only_refreshes_the_stream(self):
        owner = SandboxOwner(kind="session", id=_SESSION)
        redis, sandbox, desktop = _redis("sb-1"), _sandbox("sb-1"), _desktop("sb-1")
        redis_p, get_p, cls_p, cfg_p = self._patches(redis, sandbox, desktop)
        with redis_p, get_p, cls_p, cfg_p:
            stream, first_time, shared = await open_desktop(
                owner, {}, "k", user_id=_USER
            )

        assert not first_time and stream.sandbox_id == "sb-1"
        # ensure_display is idempotent and cheap; the home redirect is not
        # repeated once the screen has been on.
        desktop.ensure_display.assert_awaited_once()
        desktop.ensure_persistent_home.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_no_volume_means_nothing_to_redirect(self):
        owner = SandboxOwner(kind="session", id=_SESSION)
        redis, sandbox = _redis(None), _sandbox("sb-1")
        desktop = _desktop("sb-1", mounted=False)
        redis_p, get_p, cls_p, cfg_p = self._patches(redis, sandbox, desktop)
        with redis_p, get_p, cls_p, cfg_p:
            _stream, first_time, shared = await open_desktop(
                owner, {}, "k", user_id=_USER
            )
        assert first_time and not shared
        desktop.ensure_persistent_home.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_second_opener_waits_for_the_first(self):
        """The panel's button and the model's start_desktop can race; only one
        may start the display stack, the other waits and reuses the stream."""
        owner = SandboxOwner(kind="expert", id=_EXPERT)
        redis, sandbox, desktop = _redis("sb-1"), _sandbox("sb-1"), _desktop("sb-1")
        # Lock taken on the first attempt; free on the second; then the
        # screen flag and the stream password.
        redis.set = AsyncMock(side_effect=[False, True, True, True])
        redis_p, get_p, cls_p, cfg_p = self._patches(redis, sandbox, desktop)
        lock_key = f"copilot:e2b:expert:{_EXPERT}:shell:display:lock"
        with redis_p, get_p, cls_p, cfg_p, patch(f"{_C}._DESKTOP_LOCK_POLL_SECONDS", 0):
            stream, first_time, _ = await open_desktop(owner, {}, "k", user_id=_USER)

        assert not first_time and stream.sandbox_id == "sb-1"
        # It went back for the lock rather than starting a second display stack.
        attempts = [c for c in redis.set.await_args_list if c.args[0] == lock_key]
        assert len(attempts) == 2
        # The lock is released by token, never a bare delete of the key.
        assert redis.delete_if_owner.await_args.kwargs == {
            "key": lock_key,
            "token": attempts[1].args[1],
        }

    @pytest.mark.asyncio
    async def test_the_open_is_cut_off_before_the_lock_can_lapse(self):
        owner = SandboxOwner(kind="expert", id=_EXPERT)
        redis = _redis(None)

        async def never(*_args, **_kwargs):
            await asyncio.Event().wait()

        with (
            patch(f"{_C}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_C}.get_or_create_owner_sandbox", AsyncMock(side_effect=never)),
            patch(f"{_C}.chat_config"),
            patch(f"{_C}._DESKTOP_OPEN_DEADLINE_SECONDS", 0.01),
        ):
            with pytest.raises(asyncio.TimeoutError):
                await open_desktop(owner, {}, "k", user_id=_USER)
        from backend.copilot import computer

        assert (
            computer._DESKTOP_OPEN_DEADLINE_SECONDS < computer._DESKTOP_LOCK_TTL_SECONDS
        )
        # The lock is still released on the way out.
        redis.delete_if_owner.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_gives_up_when_the_lock_never_frees(self):
        owner = SandboxOwner(kind="expert", id=_EXPERT)
        redis = _redis(None, lock_free=False)
        with (
            patch(f"{_C}.get_redis_async", AsyncMock(return_value=redis)),
            patch(f"{_C}._DESKTOP_LOCK_POLL_SECONDS", 0.01),
            patch(f"{_C}._DESKTOP_LOCK_WAIT_SECONDS", 0.015),
            patch(f"{_C}.get_or_create_owner_sandbox", AsyncMock()) as get_mock,
        ):
            with pytest.raises(RuntimeError, match="still opening"):
                await open_desktop(owner, {}, "k", user_id=_USER)
        get_mock.assert_not_awaited()
        redis.delete_if_owner.assert_not_awaited()


class TestScreenIsOn:
    @pytest.mark.asyncio
    async def test_matches_only_the_current_box(self):
        owner = SandboxOwner(kind="session", id=_SESSION)
        with patch(f"{_C}.get_redis_async", AsyncMock(return_value=_redis(b"sb-1"))):
            assert await screen_is_on(owner, "sb-1") is True
            assert await screen_is_on(owner, "sb-2") is False
