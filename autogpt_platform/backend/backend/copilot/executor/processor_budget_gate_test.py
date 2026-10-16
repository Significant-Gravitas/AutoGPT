"""The executor checks the allowance of every platform turn it picks up.

Whatever queued the turn -- the chat route, a schedule, the AutoPilot block, a
sub-session, a chat bot -- a turn the allowance can't cover never runs: it
ends as a not-started pause, live and on a marker row, without an error. The
one exception is the AutoPilot block's turn, which runs on the credit balance
instead when the balance covers it.
"""

import logging
import threading
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, PropertyMock, patch

import pytest

from backend.copilot.config import ChatConfig
from backend.copilot.continuation import KEEP_GOING_NOTE, RESUMED_AT_RESET_NOTE
from backend.copilot.executor.processor import (
    RATE_LIMIT_UNAVAILABLE_CODE,
    RATE_LIMIT_UNAVAILABLE_MESSAGE,
    CoPilotProcessor,
)
from backend.copilot.executor.utils import CoPilotExecutionEntry, CoPilotLogMetadata
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.rate_limit import RateLimitUnavailable
from backend.copilot.response_model import StreamStart, StreamTurnPaused
from backend.copilot.route_table import PlatformRoute, platform_row
from backend.copilot.token_tracking import persist_and_record_usage
from backend.copilot.tree import root_envelope
from backend.copilot.turn_pause import (
    TurnPause,
    append_pause_marker,
    continues_tainted,
    pause_of,
)

_PROC = "backend.copilot.executor.processor"
PAUSE = TurnPause(
    reason="daily",
    resets_at=datetime(2026, 10, 15, tzinfo=UTC),
    checkpoint_saved=False,
    started=False,
)


@pytest.fixture(autouse=True)
def no_trial_attribution(mocker):
    store = MagicMock()
    store.get_usage_activation_state = AsyncMock(
        side_effect=lambda user_id: SimpleNamespace(
            user_id=user_id, generation=None, trial_id=None, ready=True
        )
    )
    mocker.patch(
        "backend.copilot.usage_activation.pro_activation_db", return_value=store
    )


class _Turn:
    """Runs ``_execute_async`` with the registry, session store and engine
    replaced, recording what the turn published and how it ended."""

    def __init__(self, session: ChatSession) -> None:
        self.session = session
        self.publish = AsyncMock()
        self.complete = AsyncMock()
        self.save = AsyncMock()
        self.engine = MagicMock(return_value=MagicMock())
        self.stream = MagicMock(return_value=_EmptyStream())

    async def run(self, entry: CoPilotExecutionEntry, refusal: AsyncMock) -> None:
        with (
            patch(f"{_PROC}.refusal_for", refusal),
            patch(f"{_PROC}.stream_registry.publish_chunk", self.publish),
            patch(f"{_PROC}.stream_registry.mark_session_completed", self.complete),
            patch(f"{_PROC}.stream_registry.stream_and_publish", self.stream),
            patch(f"{_PROC}.ChatConfig", return_value=MagicMock(test_mode=True)),
            patch(f"{_PROC}.stream_chat_completion_dummy", self.engine),
            patch(
                "backend.copilot.model.get_chat_session",
                AsyncMock(return_value=self.session),
            ),
            patch("backend.copilot.budget_gate.upsert_chat_session", self.save),
        ):
            await CoPilotProcessor()._execute_async(
                entry,
                threading.Event(),
                MagicMock(),
                CoPilotLogMetadata(logger=logging.getLogger("test-copilot")),
            )

    def published(self) -> list:
        return [call.args[1] for call in self.publish.await_args_list]


class _EmptyStream:
    def __aiter__(self):
        return self

    async def __anext__(self):
        raise StopAsyncIteration

    async def aclose(self) -> None:
        return None


def _entry(**update) -> CoPilotExecutionEntry:
    entry = CoPilotExecutionEntry(
        session_id="sess-1",
        turn_id="turn-1",
        user_id="user-1",
        message="summarise my inbox",
    )
    return entry.model_copy(update=update)


def _session(*messages: ChatMessage) -> ChatSession:
    session = ChatSession.new("user-1", dry_run=False)
    session.session_id = "sess-1"
    session.messages = list(messages)
    return session


@pytest.mark.asyncio
async def test_a_refused_turn_ends_paused_without_running():
    turn = _Turn(_session(ChatMessage(role="assistant", content="Earlier.")))
    refusal = AsyncMock(return_value=PAUSE)

    await turn.run(_entry(), refusal)

    refusal.assert_awaited_once_with("user-1")
    turn.engine.assert_not_called()
    turn.stream.assert_not_called()
    paused = [c for c in turn.published() if isinstance(c, StreamTurnPaused)]
    assert [c.pause for c in paused] == [PAUSE.as_part()]
    assert any(isinstance(c, StreamStart) for c in turn.published())
    for call in turn.publish.await_args_list:
        assert call.args[0] == "turn-1"
    # The chat keeps what was asked and why it didn't run.
    asked, marker = turn.session.messages[-2:]
    assert (asked.role, asked.content) == ("user", "summarise my inbox")
    assert pause_of(marker) == PAUSE
    turn.save.assert_awaited_once_with(turn.session)
    turn.complete.assert_awaited_once_with(
        "sess-1", error_message=None, error_code=None, turn_id="turn-1"
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "origin, tainted",
    [
        (dict(from_chat_platform=True), True),
        (dict(envelope=root_envelope("turn-1", tainted=True)), True),
        (dict(envelope=root_envelope("turn-1")), False),
        ({}, False),
    ],
    ids=["a-bots-message", "a-tainted-tree", "the-owners-message", "no-envelope"],
)
async def test_a_refused_turns_marker_keeps_the_taint_it_would_have_run_with(
    origin: dict, tainted: bool
):
    """The resume at the budget reset picks this pause up, in a web chat too."""
    turn = _Turn(_session(ChatMessage(role="assistant", content="Earlier.")))

    await turn.run(_entry(**origin), AsyncMock(return_value=PAUSE))

    turn.engine.assert_not_called()
    assert pause_of(turn.session.messages[-1]) == PAUSE
    assert continues_tainted(turn.session.messages) is tainted


@pytest.mark.asyncio
@pytest.mark.parametrize("note", [KEEP_GOING_NOTE, RESUMED_AT_RESET_NOTE])
@pytest.mark.parametrize(
    "paused_tainted", [True, False], ids=["paused-tainted", "paused-untainted"]
)
async def test_a_refused_pickup_keeps_the_paused_turns_taint(
    note: str, paused_tainted: bool
):
    """Keep going, or the resume at the reset, queued without the taint (by an
    API or a scheduler from before the field): the not-started marker still
    says how the paused turn ran, for the resume that picks this one up."""
    session = _session(ChatMessage(role="user", content="Build the report"))
    append_pause_marker(
        session, TurnPause(reason="steps", steps=450), tainted=paused_tainted
    )
    turn = _Turn(session)

    await turn.run(
        _entry(message=note, is_user_message=False, envelope=root_envelope("turn-1")),
        AsyncMock(return_value=PAUSE),
    )

    turn.engine.assert_not_called()
    assert pause_of(turn.session.messages[-1]) == PAUSE
    assert continues_tainted(turn.session.messages) is paused_tainted


@pytest.mark.asyncio
async def test_a_message_the_route_already_saved_is_not_saved_twice():
    turn = _Turn(_session(ChatMessage(role="user", content="summarise my inbox")))

    await turn.run(_entry(), AsyncMock(return_value=PAUSE))

    assert [m.content for m in turn.session.messages if m.role == "user"] == [
        "summarise my inbox"
    ]
    assert pause_of(turn.session.messages[-1]) == PAUSE


@pytest.mark.asyncio
async def test_an_unreadable_allowance_fails_the_turn_closed():
    turn = _Turn(_session())

    await turn.run(_entry(), AsyncMock(side_effect=RateLimitUnavailable()))

    turn.engine.assert_not_called()
    turn.stream.assert_not_called()
    assert not any(isinstance(c, StreamTurnPaused) for c in turn.published())
    turn.complete.assert_awaited_once_with(
        "sess-1",
        error_message=RATE_LIMIT_UNAVAILABLE_MESSAGE,
        error_code=RATE_LIMIT_UNAVAILABLE_CODE,
        turn_id="turn-1",
    )


@pytest.mark.asyncio
async def test_a_turn_the_allowance_covers_runs():
    turn = _Turn(_session())

    await turn.run(_entry(), AsyncMock(return_value=None))

    turn.engine.assert_called_once()
    turn.stream.assert_called_once()
    turn.save.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_turn_on_the_users_own_subscription_is_not_checked():
    session = ChatSession.new(
        "user-1",
        dry_run=False,
        llm_auth_provider="microsoft_365_copilot",
        llm_credential_id="cred-microsoft",
    )
    session.session_id = "sess-1"
    turn = _Turn(session)
    refusal = AsyncMock(return_value=PAUSE)
    manager = MagicMock(acquire_lease=AsyncMock(side_effect=ValueError))

    with patch(f"{_PROC}.IntegrationCredentialsManager", return_value=manager):
        with pytest.raises(RuntimeError):
            await turn.run(
                _entry(
                    llm_auth_provider="microsoft_365_copilot",
                    llm_credential_id="cred-microsoft",
                ),
                refusal,
            )

    refusal.assert_not_awaited()
    manager.acquire_lease.assert_awaited_once()


class _SpendingStream:
    """A turn that records its cost the way an engine does, then ends."""

    def __init__(self, cost_usd: float) -> None:
        self.cost_usd = cost_usd
        self.done = False

    def __aiter__(self):
        return self

    async def __anext__(self):
        if not self.done:
            self.done = True
            await persist_and_record_usage(
                session=None,
                user_id="user-1",
                prompt_tokens=1_000,
                completion_tokens=100,
                cost_usd=self.cost_usd,
            )
        raise StopAsyncIteration

    async def aclose(self) -> None:
        return None


_CREDITS_ON = "backend.copilot.credit_billing.settings.config.enable_credit"


@pytest.fixture
def credits(mocker) -> MagicMock:
    db = MagicMock()
    db.get_credits = AsyncMock(return_value=500)
    db.spend_credits = AsyncMock(return_value=0)
    db.get_org_credits = AsyncMock(return_value=0)
    db.spend_org_credits = AsyncMock(return_value=0)
    mocker.patch("backend.copilot.credit_billing.credit_db", return_value=db)
    # A deployment that bills credits, on a route whose engine caps a turn's
    # dollars.
    mocker.patch(_CREDITS_ON, True)
    mocker.patch.object(
        ChatConfig,
        "route_row",
        new_callable=PropertyMock,
        return_value=platform_row("openrouter"),
    )
    return db


@pytest.fixture
def windows(mocker) -> AsyncMock:
    """What records spend on the plan's daily and weekly windows."""
    mocker.patch("backend.copilot.token_tracking._schedule_cost_log")
    return mocker.patch(
        "backend.copilot.token_tracking.record_cost_usage", new=AsyncMock()
    )


def _paused(turn: _Turn) -> list:
    return [c.pause for c in turn.published() if isinstance(c, StreamTurnPaused)]


@pytest.mark.asyncio
async def test_a_block_turn_past_its_allowance_runs_on_credits_and_pays_its_cost(
    credits, windows
):
    turn = _Turn(_session())
    turn.stream = MagicMock(return_value=_SpendingStream(cost_usd=0.1234))

    await turn.run(_entry(tool_name="autopilot_block"), AsyncMock(return_value=PAUSE))

    turn.engine.assert_called_once()
    assert _paused(turn) == []
    # $0.1234 is 1,234 tenths of a cent: 13 credits, rounded up.
    credits.spend_credits.assert_awaited_once()
    charge = credits.spend_credits.await_args.kwargs
    assert (charge["user_id"], charge["cost"]) == ("user-1", 13)
    assert charge["metadata"].chat_session_id == "sess-1"
    assert charge["metadata"].block == "AutoPilotBlock"
    windows.assert_not_awaited()
    turn.complete.assert_awaited_once_with(
        "sess-1", error_message=None, error_code=None, turn_id="turn-1"
    )


@pytest.mark.asyncio
async def test_an_organizations_block_turn_is_charged_to_the_organization(
    credits, windows
):
    credits.get_org_credits.return_value = 300
    turn = _Turn(_session())
    turn.stream = MagicMock(return_value=_SpendingStream(cost_usd=0.1234))

    await turn.run(
        _entry(tool_name="autopilot_block", organization_id="org-1", team_id="team-1"),
        AsyncMock(return_value=PAUSE),
    )

    credits.get_org_credits.assert_awaited_once_with(org_id="org-1")
    credits.spend_credits.assert_not_awaited()
    charge = credits.spend_org_credits.await_args.kwargs
    assert (charge["org_id"], charge["team_id"], charge["cost"]) == (
        "org-1",
        "team-1",
        13,
    )


@pytest.mark.asyncio
async def test_a_turn_the_allowance_covers_still_spends_from_its_windows(
    credits, windows
):
    turn = _Turn(_session())
    turn.stream = MagicMock(return_value=_SpendingStream(cost_usd=0.1234))

    await turn.run(_entry(tool_name="autopilot_block"), AsyncMock(return_value=None))

    windows.assert_awaited_once()
    assert windows.await_args.kwargs["cost_microdollars"] == 123_400
    credits.get_credits.assert_not_awaited()
    credits.spend_credits.assert_not_awaited()


@pytest.mark.asyncio
async def test_a_block_turn_the_balance_cant_cover_is_refused_and_charged_nothing(
    credits, windows
):
    credits.get_credits.return_value = 49
    turn = _Turn(_session())

    await turn.run(_entry(tool_name="autopilot_block"), AsyncMock(return_value=PAUSE))

    turn.engine.assert_not_called()
    turn.stream.assert_not_called()
    assert _paused(turn) == [PAUSE.as_part()]
    credits.spend_credits.assert_not_awaited()
    windows.assert_not_awaited()


@pytest.mark.asyncio
@pytest.mark.parametrize("tool_name", [None, "chat", "run_sub_session"])
async def test_a_turn_the_block_didnt_start_is_refused_whatever_the_balance(
    credits, windows, tool_name
):
    credits.get_credits.return_value = 5_000
    turn = _Turn(_session())

    await turn.run(_entry(tool_name=tool_name), AsyncMock(return_value=PAUSE))

    turn.engine.assert_not_called()
    assert _paused(turn) == [PAUSE.as_part()]
    credits.get_credits.assert_not_awaited()
    credits.spend_credits.assert_not_awaited()


@pytest.mark.asyncio
async def test_an_unreadable_allowance_fails_a_block_turn_closed_too(credits):
    turn = _Turn(_session())

    await turn.run(
        _entry(tool_name="autopilot_block"),
        AsyncMock(side_effect=RateLimitUnavailable()),
    )

    turn.engine.assert_not_called()
    assert _paused(turn) == []
    credits.get_credits.assert_not_awaited()
    turn.complete.assert_awaited_once_with(
        "sess-1",
        error_message=RATE_LIMIT_UNAVAILABLE_MESSAGE,
        error_code=RATE_LIMIT_UNAVAILABLE_CODE,
        turn_id="turn-1",
    )


@pytest.mark.asyncio
async def test_an_unreadable_balance_fails_a_block_turn_closed(credits):
    credits.get_credits.side_effect = ConnectionError("db down")
    turn = _Turn(_session())

    await turn.run(_entry(tool_name="autopilot_block"), AsyncMock(return_value=PAUSE))

    turn.engine.assert_not_called()
    assert _paused(turn) == []
    credits.spend_credits.assert_not_awaited()
    turn.complete.assert_awaited_once_with(
        "sess-1",
        error_message=RATE_LIMIT_UNAVAILABLE_MESSAGE,
        error_code=RATE_LIMIT_UNAVAILABLE_CODE,
        turn_id="turn-1",
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "route, on_credits",
    [
        ("openrouter", True),
        ("direct_anthropic", True),
        ("subscription", True),
        ("local", False),
    ],
)
async def test_only_a_route_that_caps_a_turns_dollars_runs_a_block_turn_on_credits(
    credits, windows, mocker, route: PlatformRoute, on_credits: bool
):
    """A turn on the credit balance mustn't overdraw it, so it runs only on a
    route whose engine gives the turn a dollar ceiling; elsewhere it is refused
    as any other turn past the allowance is."""
    mocker.patch.object(
        ChatConfig,
        "route_row",
        new_callable=PropertyMock,
        return_value=platform_row(route),
    )
    turn = _Turn(_session())
    turn.stream = MagicMock(return_value=_SpendingStream(cost_usd=0.1234))

    await turn.run(_entry(tool_name="autopilot_block"), AsyncMock(return_value=PAUSE))

    if on_credits:
        turn.engine.assert_called_once()
        assert _paused(turn) == []
        credits.spend_credits.assert_awaited_once()
    else:
        turn.engine.assert_not_called()
        assert _paused(turn) == [PAUSE.as_part()]
        credits.get_credits.assert_not_awaited()
        credits.spend_credits.assert_not_awaited()
