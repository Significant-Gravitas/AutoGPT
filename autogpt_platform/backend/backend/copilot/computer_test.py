"""Tests for backend.copilot.computer: describe without waking, screen on in place."""

import asyncio
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch
from uuid import uuid4

import pytest
from e2b import SandboxState

from backend.blocks.desktop._api import DesktopStream
from backend.blocks.desktop._common import SHARED_PATH, WORKSPACE_PATH
from backend.copilot.computer import (
    ComputerInfo,
    DesktopRefused,
    computer_owner,
    describe_computer,
    mounts_for,
    open_desktop,
    open_desktop_from_ui,
    screen_is_on,
)
from backend.copilot.tools.e2b_sandbox import (
    KEEP_CEILING,
    SandboxLookupError,
    SandboxOwner,
)
from backend.data.redis_client import get_redis_async
from backend.util.sandbox_limits import SandboxLimitError

_C = "backend.copilot.computer"
_USER, _EXPERT, _SESSION = "user-1", "exp-1", "sess-1"


def _info(sandbox_id: str, state: SandboxState, mounts: str = "attached"):
    return SimpleNamespace(
        sandbox_id=sandbox_id,
        state=state,
        started_at=datetime(2026, 9, 5, 12, 0, tzinfo=timezone.utc),
        cpu_count=1,
        memory_mb=2048,
        template_id="agpt-desktop-1x2-cfc276e0",
        metadata={"autogpt_kind": "shell", "autogpt_mounts": mounts},
    )


def _redis(display: str | None, lock_free: bool = True, stream: str | None = None):
    r = MagicMock()
    r.get = AsyncMock(
        side_effect=lambda key: stream if key.endswith(":stream") else display
    )
    r.set = AsyncMock(return_value=lock_free)
    r.delete = AsyncMock()
    r.eval = AsyncMock(return_value=1)
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


class TestOpenDesktopFromUI:
    """An open from the Computer tab or side panel, which no turn stands behind."""

    @pytest.fixture(autouse=True)
    def _setup(self, monkeypatch: pytest.MonkeyPatch):
        self.prefix = f"test:{uuid4().hex}:desktop-charged-until:"
        monkeypatch.setattr(f"{_C}._CHARGED_UNTIL_KEY_PREFIX", self.prefix)
        with patch(
            "backend.copilot.trial_cost_context.get_ready_usage_state",
            AsyncMock(return_value=SimpleNamespace(trial_id=None, generation=None)),
        ):
            yield

    async def test_a_spent_allowance_refuses_before_the_box_is_touched_or_charged(
        self,
    ):
        owner = SandboxOwner(kind="session", id=_SESSION)
        with (
            patch(
                f"{_C}.platform_spend_refusal",
                AsyncMock(return_value="You've reached your daily usage limit."),
            ) as allowance,
            patch(f"{_C}.open_desktop", AsyncMock()) as opened,
            patch(f"{_C}.persist_and_record_usage", AsyncMock()) as charged,
        ):
            with pytest.raises(DesktopRefused, match="daily usage limit"):
                await open_desktop_from_ui(
                    owner, {}, "e2b-key", user_id=_USER, session_id=_SESSION
                )
        allowance.assert_awaited_once_with(_USER, surface="chat")
        opened.assert_not_awaited()
        charged.assert_not_awaited()

    async def test_each_open_is_charged_only_for_the_running_time_it_adds(self):
        """Against the stack's Redis, which runs the charge script."""
        owner = SandboxOwner(kind="expert", id=f"exp-{uuid4().hex}")
        clock = MagicMock()
        cfg = MagicMock(
            e2b_sandbox_timeout=420,
            e2b_desktop_usd_per_hour=0.36,
            e2b_sandbox_template="tpl",
        )
        try:
            with (
                patch(f"{_C}.platform_spend_refusal", AsyncMock(return_value=None)),
                patch(f"{_C}.open_desktop", AsyncMock(return_value="stream")) as opened,
                patch(f"{_C}.persist_and_record_usage", AsyncMock()) as charged,
                patch(f"{_C}.chat_config", cfg),
                patch(f"{_C}.time", clock),
            ):
                # The first open arms 420 s; 60 s on, a re-open moves the end out
                # by 60 s; the same second again adds nothing; once the armed
                # time has run out, an open arms a whole 420 s again.
                for now in (1_000, 1_060, 1_060, 2_000):
                    clock.time.return_value = now
                    result = await open_desktop_from_ui(
                        owner, {"/w": "vol"}, "e2b-key", user_id=_USER
                    )
                    assert result == "stream"
        finally:
            await (await get_redis_async()).delete(
                f"{self.prefix}{owner.kind}:{owner.id}"
            )
        assert opened.await_count == 4
        opened.assert_awaited_with(
            owner, {"/w": "vol"}, "e2b-key", user_id=_USER, session_id=None
        )
        charges = [c.kwargs for c in charged.await_args_list]
        assert [c["extra_metadata"]["seconds"] for c in charges] == [420, 60, 420]
        assert [c["cost_usd"] for c in charges] == pytest.approx([0.042, 0.006, 0.042])
        assert {(c["user_id"], c["provider"], c["expert_id"]) for c in charges} == {
            (_USER, "e2b", owner.id)
        }

    async def test_an_open_that_fails_after_arming_the_box_is_still_charged(self):
        """The connect re-arms the box before the display can fail, so the time
        it armed is charged whether or not the open then succeeds."""
        owner = SandboxOwner(kind="session", id=f"sess-{uuid4().hex}")
        try:
            with (
                patch(f"{_C}.platform_spend_refusal", AsyncMock(return_value=None)),
                patch(
                    f"{_C}.open_desktop",
                    AsyncMock(side_effect=RuntimeError("display did not start")),
                ),
                patch(f"{_C}.persist_and_record_usage", AsyncMock()) as charged,
            ):
                with pytest.raises(RuntimeError):
                    await open_desktop_from_ui(
                        owner, {}, "e2b-key", user_id=_USER, session_id=owner.id
                    )
        finally:
            await (await get_redis_async()).delete(
                f"{self.prefix}{owner.kind}:{owner.id}"
            )
        charged.assert_awaited_once()
        assert charged.await_args.kwargs["chat_session_id_override"] == owner.id

    async def test_a_start_the_sandbox_limits_refuse_is_not_charged(self):
        """The limits refuse before any box is started or woken, so nothing ran."""
        owner = SandboxOwner(kind="session", id=f"sess-{uuid4().hex}")
        key = f"{self.prefix}{owner.kind}:{owner.id}"
        with (
            patch(f"{_C}.platform_spend_refusal", AsyncMock(return_value=None)),
            patch(
                f"{_C}.open_desktop",
                AsyncMock(side_effect=SandboxLimitError("Too many sandboxes.")),
            ),
            patch(f"{_C}.persist_and_record_usage", AsyncMock()) as charged,
        ):
            with pytest.raises(SandboxLimitError):
                await open_desktop_from_ui(owner, {}, "e2b-key", user_id=_USER)
        charged.assert_not_awaited()
        # Nothing is recorded as paid for, so the next open is charged in full.
        assert await (await get_redis_async()).get(key) is None


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
            cfg.e2b_sandbox_template = "agpt-desktop-1x2-cfc276e0"
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
            template="agpt-desktop-1x2-cfc276e0",
            on_timeout="pause",
            volume_mounts=mounts,
            user_id=_USER,
            session_id=_SESSION,
            count_turn=False,
            # The UI has no turn, so it keeps the box's ceiling.
            providers=KEEP_CEILING,
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
        _script, _, released_key, token = redis.eval.await_args.args
        assert released_key == lock_key
        assert token == attempts[1].args[1]

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
        redis.eval.assert_awaited_once()

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
        redis.eval.assert_not_awaited()


class TestScreenIsOn:
    @pytest.mark.asyncio
    async def test_matches_only_the_current_box(self):
        owner = SandboxOwner(kind="session", id=_SESSION)
        with patch(f"{_C}.get_redis_async", AsyncMock(return_value=_redis(b"sb-1"))):
            assert await screen_is_on(owner, "sb-1") is True
            assert await screen_is_on(owner, "sb-2") is False
