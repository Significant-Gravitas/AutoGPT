"""A turn's kind, read once from the session's stored facts and the dispatch's."""

import pytest

from backend.copilot.expert_kickoff import expert_kickoff_metadata
from backend.copilot.model import ChatMessage, ChatSession, ChatSessionOrigin
from backend.copilot.session_kind import SessionKind
from backend.copilot.tools import kickoff_turn_disabled_tools, origin_disabled_tools


def _session(
    origin: ChatSessionOrigin | None = "interactive", *, rows=()
) -> ChatSession:
    session = ChatSession.new("user-1", dry_run=False)
    session.metadata.origin = origin
    session.messages = list(rows)
    return session


def _kickoff_row() -> ChatMessage:
    return ChatMessage(
        role="user",
        content="You were just hired.",
        metadata=expert_kickoff_metadata("e"),
    )


@pytest.mark.parametrize(
    "origin, scheduled, unattended",
    [
        ("automation", False, True),
        ("interactive", True, True),
        ("interactive", False, False),
        (None, False, False),
    ],
    ids=["automation chat", "scheduled into a user chat", "user chat", "legacy chat"],
)
def test_who_is_watching(origin, scheduled, unattended):
    kind = SessionKind.for_turn(_session(origin), scheduled=scheduled)

    assert kind.unattended is unattended


def test_a_turn_answering_the_kickoff_row_is_the_kickoff():
    assert SessionKind.for_turn(_session(rows=[_kickoff_row()])).expert_kickoff


def test_a_typed_reply_after_the_kickoff_is_not():
    rows = [
        _kickoff_row(),
        ChatMessage(role="assistant", content="Hi"),
        ChatMessage(role="user", content="Hello"),
    ]

    assert not SessionKind.for_turn(_session(rows=rows)).expert_kickoff


def test_a_kind_read_before_the_turns_rows_reads_the_kickoff_again_after():
    """An auto-continue's queued reply, or a message the engine saves itself,
    is the latest user row only once it is in: the kickoff is read again
    then, and the dispatch's facts stay as they were."""
    session = _session(rows=[_kickoff_row()])
    kind = SessionKind.for_turn(
        session, scheduled=True, voice=True, from_chat_platform=True
    )
    assert kind.expert_kickoff

    session.messages.append(ChatMessage(role="user", content="Now run my report"))
    after = kind.with_turn_rows(session)

    assert not after.expert_kickoff
    assert after == kind.model_copy(update={"expert_kickoff": False})


def test_the_stored_facts_and_the_dispatch_facts_are_read_once():
    session = _session()
    session.metadata.source_platform = "slack"
    session.metadata.builder_graph_id = "graph-1"
    session.metadata.delegated_by_session_id = "parent"

    kind = SessionKind.for_turn(session, voice=True)

    assert (kind.from_chat_platform, kind.builder, kind.delegated, kind.voice) == (
        True,
        True,
        True,
        True,
    )
    assert SessionKind.for_turn(_session()).model_dump() == {
        "origin": "interactive",
        "unattended": False,
        "expert_kickoff": False,
        "from_chat_platform": False,
        "builder": False,
        "delegated": False,
        "voice": False,
    }


def test_a_bot_prompt_counts_as_one_even_in_a_chat_started_on_the_web():
    web = _session()

    assert not SessionKind.for_turn(web).from_chat_platform
    assert SessionKind.for_turn(web, from_chat_platform=True).from_chat_platform


@pytest.mark.parametrize(
    "origin, kickoff, hidden",
    [
        ("interactive", False, frozenset()),
        ("interactive", True, kickoff_turn_disabled_tools()),
        ("automation", False, origin_disabled_tools("automation")),
        (None, False, origin_disabled_tools(None)),
        (
            "automation",
            True,
            kickoff_turn_disabled_tools() | origin_disabled_tools("automation"),
        ),
    ],
    ids=["typed", "kickoff", "automation", "legacy", "automation kickoff"],
)
def test_the_hidden_tools_are_the_union_of_both_rules(origin, kickoff, hidden):
    rows = [_kickoff_row()] if kickoff else []

    assert SessionKind.for_turn(_session(origin, rows=rows)).hidden_tools() == hidden
