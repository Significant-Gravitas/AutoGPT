"""The session view a client seeds a running turn from, on the real database."""

import json
import uuid

import pytest
from prisma.models import User

from backend.api.features.chat.routes import _with_checkpointed_rows
from backend.copilot.db import get_chat_messages_paginated
from backend.copilot.model import ChatMessage, upsert_chat_session
from backend.copilot.stream_checkpoint import rows_digest, turn_checkpoint
from backend.copilot.stream_registry import TurnCheckpoint
from backend.data import db
from backend.data.user import get_or_create_user

from .recording import persisted_session

# The session GET's default ``limit``.
PAGE = 50


@pytest.fixture
async def user_id():
    await db.connect()
    user_id = str(uuid.uuid4())
    await get_or_create_user({"sub": user_id, "email": f"{user_id}@drift.invalid"})
    yield user_id
    await User.prisma().delete_many(where={"id": user_id})


@pytest.mark.parametrize("tool_rounds", [30, 140])
async def test_a_running_turns_view_holds_the_rows_its_checkpoint_counts(
    user_id: str, tool_rounds: int
) -> None:
    session = await persisted_session(user_id, "Check every page")
    turn_start = len(session.messages)
    for round_number in range(tool_rounds):
        session.messages.extend(_tool_round(round_number))
    await upsert_chat_session(session)
    published = turn_checkpoint(session.messages, turn_start)
    assert published is not None and published.rows > PAGE
    checkpoint = TurnCheckpoint(
        entry_id="1-0", rows=published.rows, sequence=published.sequence
    )
    page = await get_chat_messages_paginated(session.session_id, PAGE, user_id=user_id)
    assert page is not None

    view = await _with_checkpointed_rows(page, checkpoint, user_id, None)

    counted = range(checkpoint.sequence, checkpoint.sequence + checkpoint.rows)
    seeded = [m for m in view.messages if m.sequence in counted]
    assert rows_digest(seeded) == published.digest
    sequences = [m.sequence for m in view.messages]
    assert sequences == list(range(checkpoint.sequence, len(session.messages)))
    assert view.oldest_sequence == checkpoint.sequence
    assert view.has_more is True


async def test_a_view_read_before_the_checkpoints_last_rows_landed_gains_them(
    user_id: str,
) -> None:
    session = await persisted_session(user_id, "Check one page")
    turn_start = len(session.messages)
    session.messages.extend(_tool_round(0))
    await upsert_chat_session(session)
    stale = await get_chat_messages_paginated(session.session_id, PAGE, user_id=user_id)
    assert stale is not None
    session.messages.extend(_tool_round(1))
    await upsert_chat_session(session)
    published = turn_checkpoint(session.messages, turn_start)
    assert published is not None
    checkpoint = TurnCheckpoint(
        entry_id="1-0", rows=published.rows, sequence=published.sequence
    )

    view = await _with_checkpointed_rows(stale, checkpoint, user_id, None)

    assert [m.sequence for m in view.messages] == list(range(len(session.messages)))
    assert rows_digest(view.messages[turn_start:]) == published.digest


def _tool_round(number: int) -> list[ChatMessage]:
    call_id = f"call-{number}"
    return [
        ChatMessage(
            role="assistant",
            content="",
            tool_calls=[
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": "web_fetch",
                        "arguments": json.dumps({"page": number}),
                    },
                }
            ],
        ),
        ChatMessage(role="tool", content=f"page {number}", tool_call_id=call_id),
    ]
