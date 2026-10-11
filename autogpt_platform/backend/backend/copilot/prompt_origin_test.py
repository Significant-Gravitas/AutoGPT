import json
from datetime import UTC, datetime

import pytest
from prisma.models import ChatMessage as PrismaChatMessage
from pytest_mock import MockerFixture

from backend.copilot.model import (
    ChatMessage,
    ChatSession,
    _save_session_to_db,
    maybe_append_user_message,
)
from backend.copilot.prompt_origin import (
    PROMPT_ORIGIN_METADATA_KEY,
    PromptOrigin,
    split_prompt_origin,
    with_prompt_origin,
)
from backend.copilot.sharing.models import sanitize_chat_message

FIRED_AT = datetime(2026, 10, 5, 9, 0, tzinfo=UTC)
ORIGIN = PromptOrigin(
    kind="scheduled_routine",
    schedule_id="job-1",
    routine_id="routine-1",
    fired_at=FIRED_AT,
)


def _prisma_row(metadata: dict | None) -> PrismaChatMessage:
    return PrismaChatMessage(
        id="msg-7",
        createdAt=FIRED_AT,
        sessionId="sess-1",
        role="assistant",
        content="Check the inbox and summarise anything new.",
        sequence=7,
        toolCalls=None,
        name=None,
        toolCallId=None,
        refusal=None,
        functionCall=None,
        metadata=None if metadata is None else json.dumps(metadata),
    )


def _session() -> ChatSession:
    return ChatSession.new(user_id="user-1", dry_run=False)


# ---- with_prompt_origin / split_prompt_origin --------------------------------


def test_round_trip_through_metadata():
    stored = with_prompt_origin(None, ORIGIN)

    assert stored == {
        PROMPT_ORIGIN_METADATA_KEY: {
            "kind": "scheduled_routine",
            "schedule_id": "job-1",
            "routine_id": "routine-1",
            "fired_at": "2026-10-05T09:00:00Z",
        }
    }
    assert split_prompt_origin(stored) == (None, ORIGIN)


def test_other_keys_survive_both_directions():
    original = {"from_session_id": "s-2"}

    stored = with_prompt_origin(original, ORIGIN)

    assert original == {"from_session_id": "s-2"}
    assert split_prompt_origin(stored) == ({"from_session_id": "s-2"}, ORIGIN)


def test_no_origin_leaves_metadata_alone():
    original = {"hidden": True}

    assert with_prompt_origin(original, None) is original
    assert with_prompt_origin(None, None) is None
    assert split_prompt_origin(original) == (original, None)
    assert split_prompt_origin(None) == (None, None)


def test_malformed_stamp_is_dropped_not_raised():
    rest, origin = split_prompt_origin(
        {PROMPT_ORIGIN_METADATA_KEY: {"kind": "not-a-kind"}, "hidden": True}
    )

    assert origin is None
    assert rest == {"hidden": True}


# ---- ChatMessage: the stamp never reaches a serialized payload ---------------


def test_from_db_lifts_the_stamp_out_of_metadata():
    stamped = ChatMessage.from_db(_prisma_row(with_prompt_origin(None, ORIGIN)))
    plain = ChatMessage.from_db(_prisma_row(None))

    assert stamped.prompt_origin == ORIGIN
    assert stamped.metadata is None
    assert stamped.role == "assistant"
    assert stamped.content == plain.content
    assert stamped.model_dump() == plain.model_dump()
    assert PROMPT_ORIGIN_METADATA_KEY not in stamped.model_dump_json()


def test_from_db_keeps_other_metadata_keys():
    row = _prisma_row(with_prompt_origin({"from_session_id": "s-2"}, ORIGIN))

    message = ChatMessage.from_db(row)

    assert message.metadata == {"from_session_id": "s-2"}
    assert message.prompt_origin == ORIGIN


def test_share_page_payload_is_unchanged_by_the_stamp():
    stamped = ChatMessage.from_db(_prisma_row(with_prompt_origin(None, ORIGIN)))
    plain = ChatMessage.from_db(_prisma_row(None))

    assert sanitize_chat_message(stamped) == sanitize_chat_message(plain)


# ---- the writer: maybe_append_user_message ----------------------------------


@pytest.mark.parametrize("is_user_message,role", [(False, "assistant"), (True, "user")])
def test_appended_prompt_row_carries_the_stamp(is_user_message: bool, role: str):
    session = _session()

    appended = maybe_append_user_message(
        session,
        "Check the inbox.",
        is_user_message,
        with_prompt_origin(None, ORIGIN),
    )

    assert appended is True
    row = session.messages[-1]
    assert row.role == role
    assert row.content == "Check the inbox."
    assert row.prompt_origin == ORIGIN
    assert row.metadata is None


def test_appended_row_without_stamp_is_unchanged():
    session = _session()

    maybe_append_user_message(session, "hi", True, {"from_session_id": "s-2"})

    row = session.messages[-1]
    assert row.prompt_origin is None
    assert row.metadata == {"from_session_id": "s-2"}


# ---- repeat fires get their own row; a redelivered fire does not ------------

NEXT_FIRE = ORIGIN.model_copy(
    update={"fired_at": datetime(2026, 10, 6, 9, 0, tzinfo=UTC)}
)


def _thread_after_a_fire(previous_fire: PromptOrigin | None) -> ChatSession:
    """A chat a routine fires into, after one fire's prompt and its reply.

    The previous prompt's stamp is ``None`` where it cannot be read, as on
    every row the executor loads from the Redis cache or over RPC.
    """
    session = _session()
    session.messages = [
        ChatMessage(role="user", content="Watch my inbox every morning."),
        ChatMessage(role="assistant", content="Will do."),
        ChatMessage(
            role="assistant", content="Check the inbox.", prompt_origin=previous_fire
        ),
        ChatMessage(role="assistant", content="Nothing new."),
    ]
    return session


def _fire(session: ChatSession, origin: PromptOrigin) -> bool:
    return maybe_append_user_message(
        session, "Check the inbox.", False, with_prompt_origin(None, origin)
    )


@pytest.mark.parametrize(
    "previous_fire", [None, ORIGIN], ids=["stamp-not-read", "stamp-read"]
)
def test_a_repeat_fire_appends_its_own_stamped_row(previous_fire):
    session = _thread_after_a_fire(previous_fire)

    assert _fire(session, NEXT_FIRE) is True

    assert len(session.messages) == 5
    row = session.messages[-1]
    assert (row.role, row.content) == ("assistant", "Check the inbox.")
    assert row.prompt_origin == NEXT_FIRE
    assert row.metadata is None


def test_unstamped_messages_keep_the_trailing_same_role_check():
    session = _thread_after_a_fire(None)

    assert maybe_append_user_message(session, "Check the inbox.", False) is False
    assert len(session.messages) == 4


@pytest.mark.parametrize(
    "stored_stamp", [None, NEXT_FIRE], ids=["stamp-not-read", "stamp-read"]
)
def test_a_redelivered_fire_does_not_append_a_second_row(stored_stamp):
    session = _thread_after_a_fire(ORIGIN)
    session.messages.append(
        ChatMessage(
            role="assistant", content="Check the inbox.", prompt_origin=stored_stamp
        )
    )

    assert _fire(session, NEXT_FIRE) is False
    assert len(session.messages) == 5


def test_a_readable_stamp_finds_a_redelivered_fire_behind_its_reply():
    session = _thread_after_a_fire(ORIGIN)
    session.messages += [
        ChatMessage(
            role="assistant", content="Check the inbox.", prompt_origin=NEXT_FIRE
        ),
        ChatMessage(role="reasoning", content="Look at the unread folder."),
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[{"id": "call-1", "type": "function", "function": {}}],
        ),
        ChatMessage(role="tool", content="0 unread", tool_call_id="call-1"),
        ChatMessage(role="assistant", content="Still nothing new."),
    ]

    assert _fire(session, NEXT_FIRE) is False
    assert len(session.messages) == 9


def test_a_last_row_stamped_with_another_fire_is_not_a_redelivery():
    session = _thread_after_a_fire(ORIGIN)
    session.messages.pop()

    assert _fire(session, NEXT_FIRE) is True
    assert session.messages[-1].prompt_origin == NEXT_FIRE


def test_stamps_without_a_fire_time_are_not_matched_to_each_other():
    queued_before_fired_at = ORIGIN.model_copy(update={"fired_at": None})
    session = _thread_after_a_fire(queued_before_fired_at)

    assert _fire(session, queued_before_fired_at) is True
    assert len(session.messages) == 5


def test_a_later_fire_after_a_user_role_first_fire_appends_its_row():
    """A thread routine's first fire opens the chat as a user-role row; its
    later fires are assistant-role and must still get their own row."""
    session = _thread_after_a_fire(None)
    session.messages.append(ChatMessage(role="user", content="Check the inbox."))

    assert _fire(session, NEXT_FIRE) is True
    assert session.messages[-1].role == "assistant"


def test_a_fire_without_a_fire_time_matches_the_same_last_prompt():
    """An entry queued before fire times existed cannot be told apart from the
    previous fire, so the same prompt as the last row is taken as already
    written, even when that row's stamp can be read."""
    session = _thread_after_a_fire(ORIGIN)
    session.messages.pop()

    assert _fire(session, ORIGIN.model_copy(update={"fired_at": None})) is False
    assert len(session.messages) == 3


# ---- persistence: the insert folds the stamp back into the metadata column ---


@pytest.mark.asyncio(loop_scope="session")
async def test_insert_writes_the_stamp_to_the_metadata_column(mocker: MockerFixture):
    session = _session()
    maybe_append_user_message(
        session, "Check the inbox.", False, with_prompt_origin(None, ORIGIN)
    )
    maybe_append_user_message(session, "plain", True, None)
    mock_db = mocker.MagicMock()
    mock_db.update_chat_session = mocker.AsyncMock()
    mock_db.add_chat_messages_batch = mocker.AsyncMock(return_value=3)
    mocker.patch("backend.copilot.model.chat_db", return_value=mock_db)

    await _save_session_to_db(
        session, existing_message_count=3, skip_existence_check=True
    )

    rows = mock_db.add_chat_messages_batch.call_args.kwargs["messages"]
    assert rows[0]["metadata"] == with_prompt_origin(None, ORIGIN)
    assert rows[1]["metadata"] is None
    assert split_prompt_origin(rows[0]["metadata"]) == (None, ORIGIN)
