from backend.copilot.model import ChatMessage

from .chat_evidence import build_chat_evidence
from .contract import LearningScope, SourceRevision


def test_a_note_for_the_owner_is_never_evidence():
    """A scheduled chat's note is the platform talking to its owner. Learnt
    from, "usage limit reached" would read as something about the user."""
    source = SourceRevision(
        source_id="source-1",
        source_kind="chat_session",
        source_ref="session-1",
        scope=LearningScope(user_id="user-1", owner_key="user:user-1"),
        revision="3",
        epoch=0,
    )
    messages = [
        ChatMessage(role="user", content="brief me", sequence=0),
        ChatMessage(
            role="assistant",
            content="Skipped: daily usage limit reached",
            sequence=1,
            metadata={"kind": "scheduled_chat_notice", "hidden_from_model": True},
        ),
        ChatMessage(role="user", content="brief me again", sequence=2),
        ChatMessage(role="assistant", content="here is your brief", sequence=3),
    ]

    bundle = build_chat_evidence(source, messages, last_sequence=3, max_chars=10_000)

    assert [span.ref for span in bundle.spans] == ["msg:0", "msg:2", "msg:3"]
