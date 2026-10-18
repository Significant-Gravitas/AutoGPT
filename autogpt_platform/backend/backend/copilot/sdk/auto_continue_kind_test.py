"""An auto-continue that picks up the owner's reply to a hire's kickoff.

The reply is the latest user row only once the continuation has saved it, so
whether the continuation answers the kickoff is read then: the owner's reply
gets the ordinary tool set, not the onboarding card's. The dispatch's own facts
carry over to the continuation, except that it is not spoken.

Drives the real ``stream_chat_completion_sdk`` through two CLI queries, the
kickoff turn and its continuation, with the CLI, persistence and the first-turn
context stubbed.
"""

import contextlib
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest

from backend.copilot.expert_kickoff import expert_kickoff_metadata
from backend.copilot.model import ChatMessage, ChatSession
from backend.copilot.pending_messages import PendingMessage
from backend.copilot.sdk import service as sdk
from backend.copilot.sdk.continued_turn_query_test import _client
from backend.copilot.sdk.retry_scenarios_test import _make_sdk_patches
from backend.copilot.sdk.turn_summary import TurnSummaryCollector
from backend.copilot.session_kind import SessionKind
from backend.copilot.tree import root_envelope

_SVC = "backend.copilot.sdk.service"
_REPLY = "Now run my report"


@pytest.mark.asyncio
async def test_auto_continue_after_kickoff_recomputes_kind_after_followup(
    tmp_path: Path,
):
    session = _kickoff_session()

    turns = await _kickoff_then_reply(tmp_path, session)

    assert [t["denied"] for t in turns] == [
        True,
        False,
    ], "the owner's queued reply got the kickoff's onboarding-only tools"
    assert [m.content for m in session.messages if m.role == "user"][-1] == _REPLY


@pytest.mark.asyncio
async def test_the_continuation_keeps_the_dispatchs_facts(tmp_path: Path):
    session = _kickoff_session()
    dispatched = SessionKind.for_turn(
        session, scheduled=True, voice=True, from_chat_platform=True
    )

    turns = await _kickoff_then_reply(tmp_path, session, session_kind=dispatched)

    first, second = (t["kind"] for t in turns)
    assert first == dispatched
    assert second == dispatched.model_copy(
        update={"expert_kickoff": False, "voice": False}
    )


def _kickoff_session() -> ChatSession:
    session = ChatSession.new("test-user", dry_run=False)
    session.expert_id = "e"
    session.metadata.origin = "interactive"
    session.messages = [
        ChatMessage(
            role="user",
            content="You were hired",
            metadata=expert_kickoff_metadata("e"),
            sequence=0,
        )
    ]
    return session


async def _kickoff_then_reply(
    tmp_path: Path,
    session: ChatSession,
    *,
    session_kind: SessionKind | None = None,
) -> list[dict[str, Any]]:
    """Run the kickoff turn while the owner's reply waits in the queue; the
    turn auto-continues with it. Returns, per query, whether ``read_skill``
    was hidden and the kind the turn recorded."""
    turns: list[dict[str, Any]] = []
    build_options = sdk._build_sdk_tool_options
    record_turn = TurnSummaryCollector.record_turn

    def tool_options(*args, **kwargs):
        turns[-1]["denied"] = "read_skill" in kwargs["denied_tools"]
        return build_options(*args, **kwargs)

    def recorded(collector, **kwargs):
        turns.append({"kind": kwargs["kind"]})
        return record_turn(collector, **kwargs)

    async def persist(s, *args, **kwargs):
        for i, message in enumerate(s.messages):
            message.sequence = i
        return s

    drains = 0

    async def drain(*args):
        # The turn-start drain finds nothing; the reply arrives while the
        # kickoff turn runs, so the drain after it picks the reply up.
        nonlocal drains
        drains += 1
        return [PendingMessage(content=_REPLY)] if drains == 2 else []

    cli = AsyncMock()
    cli.__aenter__.return_value = _client()
    cli.__aexit__.return_value = None
    patches = _make_sdk_patches(session, "", None, lambda *a, **k: cli) + [
        (
            f"{_SVC}.build_expert_identity_suffix",
            dict(new=AsyncMock(return_value="Expert identity")),
        ),
        (f"{_SVC}._make_sdk_cwd", dict(return_value=str(tmp_path))),
        (f"{_SVC}._build_sdk_tool_options", dict(side_effect=tool_options)),
        (f"{_SVC}.drain_pending_safe", dict(new=drain)),
        (f"{_SVC}._update_title_async", dict(new=AsyncMock())),
        (f"{_SVC}.build_turn_budget_block", dict(new=AsyncMock(return_value=""))),
        (f"{_SVC}.build_skills_update_notice", dict(new=AsyncMock(return_value=""))),
        (f"{_SVC}._first_turn_context_blocks", dict(new=AsyncMock(return_value=[]))),
        (f"{_SVC}._rebuild_first_turn_context", dict(new=AsyncMock(return_value=[]))),
        (f"{_SVC}.persist_session_safe", dict(new=persist)),
        (
            "backend.copilot.pending_message_helpers.persist_session_safe",
            dict(new=persist),
        ),
        (f"{_SVC}.persist_and_record_usage", dict(new=AsyncMock())),
        (f"{_SVC}._graphiti_ingest_allowed", dict(return_value=False)),
        (f"{_SVC}.capture_chat_turn", dict(new=AsyncMock())),
        (f"{_SVC}.TurnSummaryCollector.record_turn", dict(new=recorded)),
    ]
    with contextlib.ExitStack() as stack:
        for target, kwargs in patches:
            stack.enter_context(patch(target, **kwargs))
        async for _ in sdk.stream_chat_completion_sdk(
            session_id=session.session_id,
            session=session,
            user_id="test-user",
            message="You were hired",
            envelope=root_envelope("kickoff-turn"),
            session_kind=session_kind,
        ):
            pass
    return turns
