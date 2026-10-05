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
