from backend.copilot.expert_kickoff import (
    expert_kickoff_message_id,
    expert_kickoff_metadata,
    is_expert_kickoff_turn,
    is_hidden_chat_message,
    scoped_client_message_id,
)
from backend.copilot.model import ChatMessage, ChatSession

EXPERT_ID = "1a5b1a10-6d10-4d7c-9d0d-2f6f1d9c0f11"


def _expert_session(*messages: ChatMessage) -> ChatSession:
    session = ChatSession.new(user_id="alice", dry_run=False, expert_id=EXPERT_ID)
    session.messages.extend(messages)
    return session


def _kickoff_message() -> ChatMessage:
    return ChatMessage(
        role="user",
        content="You were just hired.",
        metadata=expert_kickoff_metadata(EXPERT_ID),
    )


def test_message_id_is_stable_and_scoped_to_owner_and_expert() -> None:
    message_id = expert_kickoff_message_id("user-a", "session-a", "expert-a")

    assert message_id == expert_kickoff_message_id("user-a", "session-a", "expert-a")
    assert message_id != expert_kickoff_message_id("user-b", "session-a", "expert-a")
    assert message_id != expert_kickoff_message_id("user-a", "session-b", "expert-a")
    assert message_id != expert_kickoff_message_id("user-a", "session-a", "expert-b")
    assert len(message_id) == 36


def test_client_message_ids_are_stable_and_tenant_scoped() -> None:
    message_id = scoped_client_message_id("user-a", "session-a", "click-a")

    assert message_id == scoped_client_message_id("user-a", "session-a", "click-a")
    assert message_id != scoped_client_message_id("user-b", "session-a", "click-a")
    assert message_id != scoped_client_message_id("user-a", "session-b", "click-a")
    assert message_id != scoped_client_message_id("user-a", "session-a", "click-b")


def test_client_cannot_preclaim_an_expert_kickoff_primary_key() -> None:
    kickoff_id = expert_kickoff_message_id(
        "target-user", "target-session", "target-expert"
    )

    assert (
        scoped_client_message_id("attacker", "attacker-session", kickoff_id)
        != kickoff_id
    )


def test_metadata_marks_kickoff_hidden_without_hiding_other_messages() -> None:
    metadata = expert_kickoff_metadata("expert-a")

    assert metadata == {
        "hidden": True,
        "kind": "expert_kickoff",
        "expert_id": "expert-a",
    }
    assert is_hidden_chat_message(metadata)
    assert not is_hidden_chat_message(None)
    assert not is_hidden_chat_message({"hidden": False})


def test_kickoff_turn_is_the_one_answering_the_server_written_message() -> None:
    assert is_expert_kickoff_turn(_expert_session(_kickoff_message()))


def test_a_session_with_no_messages_is_not_a_kickoff_turn() -> None:
    assert not is_expert_kickoff_turn(_expert_session())


def test_a_user_typed_first_message_is_not_a_kickoff_turn() -> None:
    session = _expert_session(ChatMessage(role="user", content="hi"))

    assert not is_expert_kickoff_turn(session)


def test_the_gate_lifts_once_the_user_answers_the_card() -> None:
    session = _expert_session(
        _kickoff_message(),
        ChatMessage(role="assistant", content="Here are a few questions."),
        ChatMessage(role="user", content="Weekly digest, connect Linear."),
    )

    assert not is_expert_kickoff_turn(session)


def test_assistant_and_tool_rows_after_the_kickoff_do_not_lift_the_gate() -> None:
    """The kickoff turn's own tool round must stay gated to the end."""
    session = _expert_session(
        _kickoff_message(),
        ChatMessage(role="assistant", content="", tool_calls=[{"id": "call-1"}]),
        ChatMessage(role="tool", content="{}", tool_call_id="call-1"),
    )

    assert is_expert_kickoff_turn(session)
