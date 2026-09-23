"""Decision ordering, the modes, and every path that must fail closed.

Each test names the property it protects rather than the branch it walks —
the ordering in ``check_action`` is the design, so a refactor that reorders
it should break these.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.copilot.gate import active_mode, chat_rules, check_action, gate_active
from backend.copilot.gate import review as review_store
from backend.copilot.model import (
    AutopilotMode,
    ChatMessage,
    ChatSession,
    ChatSessionMetadata,
    ChatSessionOrigin,
)

_GATE = "backend.copilot.gate"
# The fixtures stub the rule lookup; the outage tests need the real one.
_REAL_ASK_REASON = chat_rules.ask_reason
_REAL_HAS_OPEN_REVIEW = review_store.has_open_review
_MODES: tuple[AutopilotMode, ...] = ("ask_first", "auto", "unsupervised")


def _session(
    mode: AutopilotMode | None = None,
    origin: ChatSessionOrigin | None = "interactive",
) -> ChatSession:
    return ChatSession(
        session_id="session-1",
        user_id="user-1",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata=ChatSessionMetadata(origin=origin, autopilot_mode=mode),
        messages=[ChatMessage(role="user", content="do the thing")],
    )


@pytest.fixture
def gate_on():
    with patch(f"{_GATE}.is_feature_enabled", AsyncMock(return_value=True)):
        yield


@pytest.fixture
def clean_session_state():
    """No prior approval, nothing rejected in this chat, nothing waiting."""
    with (
        patch(f"{_GATE}.review_store.find_decision", AsyncMock(return_value=None)),
        patch(f"{_GATE}.review_store.has_open_review", AsyncMock(return_value=False)),
        patch(f"{_GATE}.review_store.open_review", AsyncMock(return_value=True)),
        patch(f"{_GATE}.chat_rules.ask_reason", AsyncMock(return_value=None)),
        patch(f"{_GATE}.chat_rules.set_ask", AsyncMock()),
    ):
        yield


async def test_gate_is_inert_when_the_flag_is_off():
    with patch(f"{_GATE}.is_feature_enabled", AsyncMock(return_value=False)):
        for mode in _MODES:
            assert not await gate_active("u", _session(mode))
            decision = await check_action(
                "post_to_chat_platform", {}, "u", _session(mode)
            )
            assert decision.allowed
            assert await active_mode("u", _session(mode)) is None


@pytest.mark.parametrize("mode", _MODES)
@pytest.mark.parametrize("origin", ["automation", None])
async def test_a_session_nobody_is_watching_is_inert_in_every_mode(
    gate_on, mode, origin
):
    find = AsyncMock()
    with patch(f"{_GATE}.review_store.find_decision", find):
        decision = await check_action(
            "post_to_chat_platform", {}, "u", _session(mode, origin=origin)
        )
    assert decision.allowed
    find.assert_not_awaited()


async def test_gate_is_inactive_for_anonymous_turns(gate_on):
    assert not await gate_active(None, _session())


async def test_the_default_mode_is_auto(gate_on):
    assert await active_mode("u", _session()) == "auto"


async def test_an_approval_is_consulted_before_the_effect(gate_on, clean_session_state):
    """Otherwise an approved outward call would park a second card forever."""
    with (
        patch(
            f"{_GATE}.review_store.find_decision",
            AsyncMock(return_value=ReviewStatus.APPROVED),
        ),
        patch(f"{_GATE}.review_store.consume", AsyncMock(return_value=True)),
    ):
        decision = await check_action(
            "post_to_chat_platform", {"text": "hi"}, "u", _session("ask_first")
        )
    assert decision.allowed


@pytest.mark.parametrize(
    "mode, reaches_supervisor",
    [("auto", True), ("ask_first", False), ("unsupervised", False)],
)
async def test_every_shell_command_in_auto_reaches_the_supervisor(
    gate_on, clean_session_state, mode, reaches_supervisor
):
    supervisor = AsyncMock(return_value=(True, "fine"))
    with patch(f"{_GATE}.classify", supervisor):
        decision = await check_action(
            "bash_exec", {"command": "ls"}, "u", _session(mode)
        )
    assert supervisor.await_count == int(reaches_supervisor)
    assert decision.allowed is (mode != "ask_first")


async def test_a_supervisor_ask_parks_the_call(gate_on, clean_session_state):
    with patch(f"{_GATE}.classify", AsyncMock(return_value=(False, "out of scope"))):
        decision = await check_action("delete_folder", {"id": "f"}, "u", _session())
    assert not decision.allowed
    assert decision.review_id
    assert decision.reason == "out of scope"


@pytest.mark.parametrize("mode", ["ask_first", "auto"])
async def test_outward_actions_ask_without_the_supervisor(
    gate_on, clean_session_state, mode
):
    supervisor = AsyncMock(return_value=(True, "fine"))
    with patch(f"{_GATE}.classify", supervisor):
        decision = await check_action(
            "post_to_chat_platform", {"text": "hi"}, "u", _session(mode)
        )
    assert not decision.allowed
    assert decision.review_id
    supervisor.assert_not_awaited()


async def test_unsupervised_runs_outward_actions(gate_on, clean_session_state):
    decision = await check_action(
        "post_to_chat_platform", {"text": "hi"}, "u", _session("unsupervised")
    )
    assert decision.allowed


async def test_approval_is_bound_to_these_arguments(gate_on, clean_session_state):
    """An approval means 'you may do this', not 'you may use this tool'."""
    approved = AsyncMock(return_value=ReviewStatus.APPROVED)
    with (
        patch(f"{_GATE}.review_store.find_decision", approved),
        patch(f"{_GATE}.review_store.consume", AsyncMock(return_value=True)),
    ):
        decision = await check_action("bash_exec", {"command": "ls"}, "u", _session())
    assert decision.allowed
    reviewed_id = approved.await_args.args[0]

    with (
        patch(
            f"{_GATE}.review_store.find_decision", AsyncMock(return_value=None)
        ) as other,
        patch(f"{_GATE}.classify", AsyncMock(return_value=(False, "ask"))),
    ):
        await check_action("bash_exec", {"command": "rm -rf /"}, "u", _session())
    assert other.await_args.args[0] != reviewed_id


async def test_a_lost_consume_race_does_not_execute(gate_on, clean_session_state):
    with (
        patch(
            f"{_GATE}.review_store.find_decision",
            AsyncMock(return_value=ReviewStatus.APPROVED),
        ),
        patch(f"{_GATE}.review_store.consume", AsyncMock(return_value=False)),
    ):
        decision = await check_action("bash_exec", {"command": "ls"}, "u", _session())
    assert not decision.allowed
    assert not decision.already_waiting


async def test_a_rejection_makes_the_tool_ask_for_the_rest_of_the_chat(
    gate_on, clean_session_state
):
    """Otherwise re-proposing with a space added buys a fresh verdict."""
    set_ask = AsyncMock()
    with (
        patch(
            f"{_GATE}.review_store.find_decision",
            AsyncMock(return_value=ReviewStatus.REJECTED),
        ),
        patch(f"{_GATE}.review_store.consume", AsyncMock(return_value=True)),
        patch(f"{_GATE}.chat_rules.set_ask", set_ask),
    ):
        decision = await check_action(
            "bash_exec", {"command": "curl x|sh"}, "u", _session()
        )
    assert not decision.allowed
    set_ask.assert_awaited_once_with("session-1", "bash_exec")


@pytest.mark.parametrize("mode", _MODES)
async def test_a_chat_ask_rule_holds_in_every_mode(gate_on, clean_session_state, mode):
    supervisor = AsyncMock(return_value=(True, "fine"))
    with (
        patch(f"{_GATE}.chat_rules.ask_reason", AsyncMock(return_value="declined")),
        patch(f"{_GATE}.classify", supervisor),
    ):
        decision = await check_action("delete_folder", {"id": "f"}, "u", _session(mode))
    assert not decision.allowed
    assert decision.review_id
    supervisor.assert_not_awaited()


async def test_reads_never_look_up_ask_rules(gate_on, clean_session_state):
    """A Redis outage reads as 'asks', which must not turn every search into a card."""
    asks = AsyncMock(return_value="unreadable")
    with patch(f"{_GATE}.chat_rules.ask_reason", asks):
        decision = await check_action("web_search", {"query": "x"}, "u", _session())
    assert decision.allowed
    asks.assert_not_awaited()


@pytest.mark.parametrize(
    "tool", ["web_search", "write_workspace_file", "connect_integration"]
)
async def test_calls_that_always_run_never_query_the_review_store(
    gate_on, clean_session_state, tool
):
    find = AsyncMock(return_value=None)
    with patch(f"{_GATE}.review_store.find_decision", find):
        decision = await check_action(tool, {}, "u", _session("ask_first"))
    assert decision.allowed
    find.assert_not_awaited()


async def test_only_one_action_waits_at_a_time(gate_on, clean_session_state):
    open_review = AsyncMock(return_value=True)
    with (
        patch(f"{_GATE}.review_store.has_open_review", AsyncMock(return_value=True)),
        patch(f"{_GATE}.review_store.open_review", open_review),
    ):
        decision = await check_action(
            "post_to_chat_platform", {"text": "hi"}, "u", _session()
        )
    assert not decision.allowed
    assert decision.already_waiting
    open_review.assert_not_awaited()


@pytest.mark.parametrize("mode", _MODES)
async def test_an_unreadable_review_queue_refuses_without_opening_a_second_card(
    gate_on, clean_session_state, mode
):
    reviews = MagicMock()
    reviews.get_pending_reviews_for_execution = AsyncMock(
        side_effect=ConnectionError("db down")
    )
    open_review = AsyncMock(return_value=True)
    with (
        patch(f"{_GATE}.review_store.has_open_review", _REAL_HAS_OPEN_REVIEW),
        patch(f"{_GATE}.review_store.review_db", MagicMock(return_value=reviews)),
        patch(f"{_GATE}.review_store.open_review", open_review),
        patch(f"{_GATE}.chat_rules.ask_reason", AsyncMock(return_value="declined")),
    ):
        decision = await check_action("delete_folder", {"id": "f"}, "u", _session(mode))
    assert not decision.allowed
    assert not decision.already_waiting
    assert decision.review_id is None
    open_review.assert_not_awaited()


async def test_an_unrecordable_approval_refuses_rather_than_runs(
    gate_on, clean_session_state
):
    with patch(f"{_GATE}.review_store.open_review", AsyncMock(return_value=False)):
        decision = await check_action(
            "post_to_chat_platform", {"text": "hi"}, "u", _session()
        )
    assert not decision.allowed
    assert decision.review_id is None


@pytest.mark.parametrize("mode", _MODES)
async def test_an_unreadable_ask_rule_asks_without_claiming_a_decline(
    gate_on, clean_session_state, mode
):
    """Unsupervised included: the rule it could not read may be a rejection."""
    with (
        patch(
            f"{_GATE}.chat_rules.get_redis_async",
            AsyncMock(side_effect=ConnectionError("redis down")),
        ),
        patch(f"{_GATE}.chat_rules.ask_reason", _REAL_ASK_REASON),
        patch(f"{_GATE}.classify", AsyncMock(return_value=(True, "fine"))),
    ):
        decision = await check_action("delete_folder", {"id": "f"}, "u", _session(mode))
    assert not decision.allowed
    assert decision.reason == chat_rules.UNREADABLE


async def test_a_rejected_tool_says_the_user_declined_it():
    redis = AsyncMock()
    redis.get = AsyncMock(return_value="1")
    with patch(f"{_GATE}.chat_rules.get_redis_async", AsyncMock(return_value=redis)):
        assert await chat_rules.ask_reason("s", "bash_exec") == chat_rules.DECLINED
