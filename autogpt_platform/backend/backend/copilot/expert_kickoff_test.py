from backend.copilot.expert_kickoff import (
    expert_kickoff_message_id,
    expert_kickoff_metadata,
    is_hidden_chat_message,
    scoped_client_message_id,
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
