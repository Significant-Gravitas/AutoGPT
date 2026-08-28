"""Decision ordering, and every path that must fail closed.

Each test names the property it protects rather than the branch it walks —
the ordering in ``check_action`` is the design, so a refactor that reorders
it should break these.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from prisma.enums import ReviewStatus

from backend.copilot.gate import check_action, gate_active
from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    ChatSessionMetadata,
    ChatSessionOrigin,
)

_GATE = "backend.copilot.gate"


def _session(
    origin: ChatSessionOrigin | None = "interactive",
    *,
    source_platform: str | None = None,
    messages: list[ChatMessage] | None = None,
    auto_mode: bool | None = None,
) -> ChatSession:
    return ChatSession(
        session_id="session-1",
        user_id="user-1",
        usage=[],
        started_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
        metadata=ChatSessionMetadata(
            origin=origin, source_platform=source_platform, auto_mode=auto_mode
        ),
        messages=messages or [ChatMessage(role="user", content="do the thing")],
    )


@pytest.fixture
def gate_on():
    with patch(f"{_GATE}.is_feature_enabled", AsyncMock(return_value=True)):
        yield


@pytest.fixture
def clean_session_state():
    """No prior approval, nothing escalated, nothing untrusted ingested."""
    with (
        patch(f"{_GATE}.review_store.find_decision", AsyncMock(return_value=None)),
        patch(f"{_GATE}.review_store.has_open_review", AsyncMock(return_value=False)),
        patch(f"{_GATE}.review_store.open_review", AsyncMock(return_value=True)),
        patch(f"{_GATE}.taint.is_escalated", AsyncMock(return_value=False)),
        patch(f"{_GATE}.taint.is_tainted", AsyncMock(return_value=False)),
        patch(f"{_GATE}.taint.escalate", AsyncMock()),
    ):
        yield


async def test_gate_is_inert_when_the_flag_is_off():
    """Flag-off must be today's behaviour byte-for-byte."""
    with patch(f"{_GATE}.is_feature_enabled", AsyncMock(return_value=False)):
        decision = await check_action(
            "bash_exec", {"command": "rm -rf /"}, "u", _session()
        )
    assert decision.allowed


@pytest.mark.parametrize("origin", ["automation", None])
async def test_gate_is_inactive_where_nobody_is_watching(gate_on, origin):
    """A question parked in an unattended run is a stall, not a safeguard;
    refusing there instead would break shipped scheduled behaviour."""
    assert not await gate_active("u", _session(origin))


async def test_gate_is_inactive_for_anonymous_turns(gate_on):
    assert not await gate_active(None, _session())


async def test_session_can_opt_out(gate_on):
    assert not await gate_active("u", _session(auto_mode=False))


async def test_read_tools_never_reach_the_classifier(gate_on, clean_session_state):
    with patch(f"{_GATE}.classify", AsyncMock()) as classifier:
        decision = await check_action("list_schedules", {}, "u", _session())
    assert decision.allowed
    classifier.assert_not_awaited()


async def test_always_ask_is_never_classified(gate_on, clean_session_state):
    with patch(
        f"{_GATE}.classify", AsyncMock(return_value=(True, "fine"))
    ) as classifier:
        decision = await check_action(
            "post_to_chat_platform", {"text": "hi"}, "u", _session()
        )
    assert not decision.allowed
    classifier.assert_not_awaited()


async def test_defer_tools_pass_through_to_their_own_gate(gate_on, clean_session_state):
    with patch(f"{_GATE}.classify", AsyncMock()) as classifier:
        decision = await check_action(
            "run_block", {"block_id": "uuid"}, "u", _session()
        )
    assert decision.allowed
    classifier.assert_not_awaited()


async def test_taint_skips_the_classifier_for_effectful_calls(
    gate_on, clean_session_state
):
    """The injected text would be sitting in the arguments the classifier reads,
    so this is precisely where its verdict is worth nothing."""
    with (
        patch(f"{_GATE}.taint.is_tainted", AsyncMock(return_value=True)),
        patch(
            f"{_GATE}.classify", AsyncMock(return_value=(True, "looks fine"))
        ) as classifier,
    ):
        decision = await check_action(
            "bash_exec", {"command": "curl x"}, "u", _session()
        )
    assert not decision.allowed
    classifier.assert_not_awaited()


async def test_taint_still_allows_reading(gate_on, clean_session_state):
    with (
        patch(f"{_GATE}.taint.is_tainted", AsyncMock(return_value=True)),
        patch(f"{_GATE}.classify", AsyncMock(return_value=(True, "ok"))),
    ):
        decision = await check_action(
            "web_fetch", {"url": "https://x"}, "u", _session()
        )
    assert decision.allowed


async def test_classifier_allow_lets_a_judged_call_through(
    gate_on, clean_session_state
):
    with patch(f"{_GATE}.classify", AsyncMock(return_value=(True, "in scope"))):
        decision = await check_action(
            "write_workspace_file", {"filename": "a"}, "u", _session()
        )
    assert decision.allowed


async def test_classifier_ask_parks_the_call(gate_on, clean_session_state):
    with patch(f"{_GATE}.classify", AsyncMock(return_value=(False, "out of scope"))):
        decision = await check_action(
            "write_workspace_file", {"filename": "a"}, "u", _session()
        )
    assert not decision.allowed
    assert decision.review_id


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
        patch(f"{_GATE}.review_store.has_open_review", AsyncMock(return_value=False)),
        patch(f"{_GATE}.review_store.open_review", AsyncMock(return_value=True)),
        patch(f"{_GATE}.taint.is_escalated", AsyncMock(return_value=False)),
        patch(f"{_GATE}.taint.is_tainted", AsyncMock(return_value=False)),
        patch(f"{_GATE}.classify", AsyncMock(return_value=(False, "ask"))),
    ):
        await check_action("bash_exec", {"command": "rm -rf /"}, "u", _session())
    assert other.await_args.args[0] != reviewed_id


async def test_a_lost_consume_race_does_not_execute(gate_on, clean_session_state):
    """Parallel dispatch is deliberate, so the delete is the mutex."""
    with (
        patch(
            f"{_GATE}.review_store.find_decision",
            AsyncMock(return_value=ReviewStatus.APPROVED),
        ),
        patch(f"{_GATE}.review_store.consume", AsyncMock(return_value=False)),
    ):
        decision = await check_action("bash_exec", {"command": "ls"}, "u", _session())
    assert not decision.allowed


async def test_rejection_escalates_the_whole_tool(gate_on, clean_session_state):
    """Otherwise re-proposing with a space added buys a fresh verdict."""
    escalate = AsyncMock()
    with (
        patch(
            f"{_GATE}.review_store.find_decision",
            AsyncMock(return_value=ReviewStatus.REJECTED),
        ),
        patch(f"{_GATE}.review_store.consume", AsyncMock(return_value=True)),
        patch(f"{_GATE}.taint.escalate", escalate),
    ):
        decision = await check_action(
            "bash_exec", {"command": "curl x|sh"}, "u", _session()
        )
    assert not decision.allowed
    escalate.assert_awaited_once_with("session-1", "bash_exec")


async def test_an_escalated_tool_is_not_classified_again(gate_on, clean_session_state):
    with (
        patch(f"{_GATE}.taint.is_escalated", AsyncMock(return_value=True)),
        patch(
            f"{_GATE}.classify", AsyncMock(return_value=(True, "fine"))
        ) as classifier,
    ):
        decision = await check_action(
            "write_workspace_file", {"filename": "a"}, "u", _session()
        )
    assert not decision.allowed
    classifier.assert_not_awaited()


async def test_only_one_action_waits_at_a_time(gate_on, clean_session_state):
    """One Approve button submits the whole queue, so the queue stays at one."""
    open_review = AsyncMock(return_value=True)
    with (
        patch(f"{_GATE}.review_store.has_open_review", AsyncMock(return_value=True)),
        patch(f"{_GATE}.review_store.open_review", open_review),
        patch(f"{_GATE}.classify", AsyncMock(return_value=(False, "ask"))),
    ):
        decision = await check_action(
            "write_workspace_file", {"filename": "a"}, "u", _session()
        )
    assert not decision.allowed
    assert decision.already_waiting
    open_review.assert_not_awaited()


async def test_an_unrecordable_approval_refuses_rather_than_runs(
    gate_on, clean_session_state
):
    with (
        patch(f"{_GATE}.review_store.open_review", AsyncMock(return_value=False)),
        patch(f"{_GATE}.classify", AsyncMock(return_value=(False, "ask"))),
    ):
        decision = await check_action(
            "write_workspace_file", {"filename": "a"}, "u", _session()
        )
    assert not decision.allowed
    assert decision.review_id is None


async def test_chat_platform_sessions_are_untrusted_at_birth(gate_on):
    """Their 'user' turn can be authored by any member of a linked server, and
    no taint source is ever touched."""
    from backend.copilot.gate.taint import born_tainted

    assert born_tainted(_session(source_platform="discord"))
    assert not born_tainted(_session())
