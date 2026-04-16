"""Unit tests for copilot.db — paginated message queries."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from prisma.models import ChatMessage as PrismaChatMessage
from prisma.models import ChatSession as PrismaChatSession

from backend.copilot.db import (
    PaginatedMessages,
    get_chat_messages_paginated,
    set_turn_duration,
    update_message_content_by_sequence,
)
from backend.copilot.model import ChatMessage as CopilotChatMessage
from backend.copilot.model import ChatSession, get_chat_session, upsert_chat_session


def _make_msg(
    sequence: int,
    role: str = "assistant",
    content: str | None = "hello",
    tool_calls: Any = None,
) -> PrismaChatMessage:
    """Build a minimal PrismaChatMessage for testing."""
    return PrismaChatMessage(
        id=f"msg-{sequence}",
        createdAt=datetime.now(UTC),
        sessionId="sess-1",
        role=role,
        content=content,
        sequence=sequence,
        toolCalls=tool_calls,
        name=None,
        toolCallId=None,
        refusal=None,
        functionCall=None,
    )


def _make_session(
    session_id: str = "sess-1",
    user_id: str = "user-1",
    messages: list[PrismaChatMessage] | None = None,
) -> PrismaChatSession:
    """Build a minimal PrismaChatSession for testing."""
    now = datetime.now(UTC)
    session = PrismaChatSession.model_construct(
        id=session_id,
        createdAt=now,
        updatedAt=now,
        userId=user_id,
        credentials={},
        successfulAgentRuns={},
        successfulAgentSchedules={},
        totalPromptTokens=0,
        totalCompletionTokens=0,
        title=None,
        metadata={},
        Messages=messages or [],
    )
    return session


SESSION_ID = "sess-1"


@pytest.fixture()
def mock_db():
    """Patch ChatSession.prisma().find_first and ChatMessage.prisma().find_many.

    find_first is used for the main query (session + included messages).
    find_many is used only for boundary expansion queries.
    """
    with (
        patch.object(PrismaChatSession, "prisma") as mock_session_prisma,
        patch.object(PrismaChatMessage, "prisma") as mock_msg_prisma,
    ):
        find_first = AsyncMock()
        mock_session_prisma.return_value.find_first = find_first

        find_many = AsyncMock(return_value=[])
        mock_msg_prisma.return_value.find_many = find_many

        yield find_first, find_many


# ---------- Basic pagination ----------


@pytest.mark.asyncio
async def test_basic_page_returns_messages_ascending(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """Messages are returned in ascending sequence order."""
    find_first, _ = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(3), _make_msg(2), _make_msg(1)],
    )

    page = await get_chat_messages_paginated(SESSION_ID, limit=5)

    assert isinstance(page, PaginatedMessages)
    assert [m.sequence for m in page.messages] == [1, 2, 3]
    assert page.has_more is False
    assert page.oldest_sequence == 1


@pytest.mark.asyncio
async def test_has_more_when_results_exceed_limit(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """has_more is True when DB returns more than limit items."""
    find_first, _ = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(3), _make_msg(2), _make_msg(1)],
    )

    page = await get_chat_messages_paginated(SESSION_ID, limit=2)

    assert page is not None
    assert page.has_more is True
    assert len(page.messages) == 2
    assert [m.sequence for m in page.messages] == [2, 3]


@pytest.mark.asyncio
async def test_empty_session_returns_no_messages(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    find_first, _ = mock_db
    find_first.return_value = _make_session(messages=[])

    page = await get_chat_messages_paginated(SESSION_ID, limit=50)

    assert page is not None
    assert page.messages == []
    assert page.has_more is False
    assert page.oldest_sequence is None


@pytest.mark.asyncio
async def test_before_sequence_filters_correctly(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """before_sequence is passed as a where filter inside the Messages include."""
    find_first, _ = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(2), _make_msg(1)],
    )

    await get_chat_messages_paginated(SESSION_ID, limit=50, before_sequence=5)

    call_kwargs = find_first.call_args
    include = call_kwargs.kwargs.get("include") or call_kwargs[1].get("include")
    assert include["Messages"]["where"] == {"sequence": {"lt": 5}}


@pytest.mark.asyncio
async def test_no_where_on_messages_without_before_sequence(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """Without before_sequence, the Messages include has no where clause."""
    find_first, _ = mock_db
    find_first.return_value = _make_session(messages=[_make_msg(1)])

    await get_chat_messages_paginated(SESSION_ID, limit=50)

    call_kwargs = find_first.call_args
    include = call_kwargs.kwargs.get("include") or call_kwargs[1].get("include")
    assert "where" not in include["Messages"]


# ---------- Forward pagination (from_start / after_sequence) ----------


@pytest.mark.asyncio
async def test_from_start_uses_asc_order_no_where(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """from_start=True queries messages in ASC order with no where filter."""
    find_first, _ = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(0), _make_msg(1), _make_msg(2)],
    )

    await get_chat_messages_paginated(SESSION_ID, limit=50, from_start=True)

    call_kwargs = find_first.call_args
    include = call_kwargs.kwargs.get("include") or call_kwargs[1].get("include")
    assert include["Messages"]["order_by"] == {"sequence": "asc"}
    assert "where" not in include["Messages"]


@pytest.mark.asyncio
async def test_from_start_returns_messages_ascending(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """from_start=True returns messages in ascending sequence order."""
    find_first, _ = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(0), _make_msg(1), _make_msg(2)],
    )

    page = await get_chat_messages_paginated(SESSION_ID, limit=50, from_start=True)

    assert page is not None
    assert [m.sequence for m in page.messages] == [0, 1, 2]
    assert (
        page.oldest_sequence is None
    )  # None in forward mode — not a valid backward cursor
    assert page.newest_sequence == 2
    assert page.has_more is False


@pytest.mark.asyncio
async def test_from_start_has_more_when_results_exceed_limit(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """from_start=True sets has_more when DB returns more than limit items."""
    find_first, _ = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(0), _make_msg(1), _make_msg(2)],
    )

    page = await get_chat_messages_paginated(SESSION_ID, limit=2, from_start=True)

    assert page is not None
    assert page.has_more is True
    assert [m.sequence for m in page.messages] == [0, 1]
    assert page.newest_sequence == 1


@pytest.mark.asyncio
async def test_after_sequence_uses_gt_filter_asc_order(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """after_sequence adds a sequence > N where clause and uses ASC order."""
    find_first, _ = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(11), _make_msg(12)],
    )

    await get_chat_messages_paginated(SESSION_ID, limit=50, after_sequence=10)

    call_kwargs = find_first.call_args
    include = call_kwargs.kwargs.get("include") or call_kwargs[1].get("include")
    assert include["Messages"]["order_by"] == {"sequence": "asc"}
    assert include["Messages"]["where"] == {"sequence": {"gt": 10}}


@pytest.mark.asyncio
async def test_after_sequence_returns_messages_in_order(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """after_sequence returns only messages with sequence > cursor, ascending."""
    find_first, _ = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(11), _make_msg(12), _make_msg(13)],
    )

    page = await get_chat_messages_paginated(SESSION_ID, limit=50, after_sequence=10)

    assert page is not None
    assert [m.sequence for m in page.messages] == [11, 12, 13]
    assert (
        page.oldest_sequence is None
    )  # None in forward mode — not a valid backward cursor
    assert page.newest_sequence == 13
    assert page.has_more is False


@pytest.mark.asyncio
async def test_newest_sequence_none_for_backward_mode(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """newest_sequence is None in backward mode — it is not a valid forward cursor."""
    find_first, _ = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(5), _make_msg(4), _make_msg(3)],
    )

    page = await get_chat_messages_paginated(SESSION_ID, limit=50)

    assert page is not None
    assert page.newest_sequence is None
    assert page.oldest_sequence == 3


@pytest.mark.asyncio
async def test_forward_mode_no_boundary_expansion(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """Forward pagination never triggers backward boundary expansion."""
    find_first, find_many = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(0, role="tool"), _make_msg(1, role="tool")],
    )

    await get_chat_messages_paginated(SESSION_ID, limit=50, from_start=True)

    assert find_many.call_count == 0


@pytest.mark.asyncio
async def test_forward_tail_boundary_trims_trailing_tool_messages(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """Forward pages that end with tool messages are trimmed to the owning
    assistant so the next after_sequence page doesn't start mid-tool-group."""
    find_first, _ = mock_db
    # DB returns 4 messages ASC: assistant at 0, tool at 1, tool at 2, tool at 3
    find_first.return_value = _make_session(
        messages=[
            _make_msg(0, role="assistant"),
            _make_msg(1, role="tool"),
            _make_msg(2, role="tool"),
            _make_msg(3, role="tool"),
        ],
    )

    page = await get_chat_messages_paginated(SESSION_ID, limit=10, from_start=True)

    assert page is not None
    # Page should be trimmed to end at the assistant message
    assert [m.sequence for m in page.messages] == [0]
    assert page.newest_sequence == 0
    # has_more must be True so the client fetches the tool messages on next page
    assert page.has_more is True


@pytest.mark.asyncio
async def test_forward_tail_boundary_no_trim_when_last_not_tool(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """Forward pages that end with a non-tool message are not trimmed."""
    find_first, _ = mock_db
    find_first.return_value = _make_session(
        messages=[
            _make_msg(0, role="user"),
            _make_msg(1, role="assistant"),
            _make_msg(2, role="tool"),
            _make_msg(3, role="assistant"),
        ],
    )

    page = await get_chat_messages_paginated(SESSION_ID, limit=10, from_start=True)

    assert page is not None
    assert [m.sequence for m in page.messages] == [0, 1, 2, 3]
    assert page.newest_sequence == 3
    assert page.has_more is False


@pytest.mark.asyncio
async def test_user_id_filter_applied_to_session_where(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """user_id adds a userId filter to the session-level where clause."""
    find_first, _ = mock_db
    find_first.return_value = _make_session(messages=[_make_msg(1)])

    await get_chat_messages_paginated(SESSION_ID, limit=50, user_id="user-abc")

    call_kwargs = find_first.call_args
    where = call_kwargs.kwargs.get("where") or call_kwargs[1].get("where")
    assert where["userId"] == "user-abc"


@pytest.mark.asyncio
async def test_session_not_found_returns_none(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """Returns None when session doesn't exist or user doesn't own it."""
    find_first, _ = mock_db
    find_first.return_value = None

    page = await get_chat_messages_paginated(SESSION_ID, limit=50)

    assert page is None


@pytest.mark.asyncio
async def test_session_info_included_in_result(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """PaginatedMessages includes session metadata."""
    find_first, _ = mock_db
    find_first.return_value = _make_session(messages=[_make_msg(1)])

    page = await get_chat_messages_paginated(SESSION_ID, limit=50)

    assert page is not None
    assert page.session.session_id == SESSION_ID


# ---------- Backward boundary expansion ----------


@pytest.mark.asyncio
async def test_boundary_expansion_includes_assistant(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """When page starts with a tool message, expand backward to include
    the owning assistant message."""
    find_first, find_many = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(5, role="tool"), _make_msg(4, role="tool")],
    )
    find_many.return_value = [_make_msg(3, role="assistant")]

    page = await get_chat_messages_paginated(SESSION_ID, limit=5)

    assert page is not None
    assert [m.sequence for m in page.messages] == [3, 4, 5]
    assert page.messages[0].role == "assistant"
    assert page.oldest_sequence == 3


@pytest.mark.asyncio
async def test_boundary_expansion_includes_multiple_tool_msgs(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """Boundary expansion scans past consecutive tool messages to find
    the owning assistant."""
    find_first, find_many = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(7, role="tool")],
    )
    find_many.return_value = [
        _make_msg(6, role="tool"),
        _make_msg(5, role="tool"),
        _make_msg(4, role="assistant"),
    ]

    page = await get_chat_messages_paginated(SESSION_ID, limit=5)

    assert page is not None
    assert [m.sequence for m in page.messages] == [4, 5, 6, 7]
    assert page.messages[0].role == "assistant"


@pytest.mark.asyncio
async def test_boundary_expansion_sets_has_more_when_not_at_start(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """After boundary expansion, has_more=True if expanded msgs aren't at seq 0."""
    find_first, find_many = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(3, role="tool")],
    )
    find_many.return_value = [_make_msg(2, role="assistant")]

    page = await get_chat_messages_paginated(SESSION_ID, limit=5)

    assert page is not None
    assert page.has_more is True


@pytest.mark.asyncio
async def test_boundary_expansion_no_has_more_at_conversation_start(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """has_more stays False when boundary expansion reaches seq 0."""
    find_first, find_many = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(1, role="tool")],
    )
    find_many.return_value = [_make_msg(0, role="assistant")]

    page = await get_chat_messages_paginated(SESSION_ID, limit=5)

    assert page is not None
    assert page.has_more is False
    assert page.oldest_sequence == 0


@pytest.mark.asyncio
async def test_no_boundary_expansion_when_first_msg_not_tool(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """No boundary expansion when the first message is not a tool message."""
    find_first, find_many = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(3, role="user"), _make_msg(2, role="assistant")],
    )

    page = await get_chat_messages_paginated(SESSION_ID, limit=5)

    assert page is not None
    assert find_many.call_count == 0
    assert [m.sequence for m in page.messages] == [2, 3]


@pytest.mark.asyncio
async def test_boundary_expansion_warns_when_no_owner_found(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """When boundary scan doesn't find a non-tool message, a warning is logged
    and the boundary messages are still included."""
    find_first, find_many = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(10, role="tool")],
    )
    find_many.return_value = [_make_msg(i, role="tool") for i in range(9, -1, -1)]

    with patch("backend.copilot.db.logger") as mock_logger:
        page = await get_chat_messages_paginated(SESSION_ID, limit=5)
        mock_logger.warning.assert_called_once()

    assert page is not None
    assert page.messages[0].role == "tool"
    assert len(page.messages) > 1


# ---------- Turn duration (integration tests) ----------


@pytest.mark.asyncio(loop_scope="session")
async def test_set_turn_duration_updates_cache_in_place(setup_test_user, test_user_id):
    """set_turn_duration patches the cached session without invalidation.

    Verifies that after calling set_turn_duration the Redis-cached session
    reflects the updated durationMs on the last assistant message, without
    the cache having been deleted and re-populated (which could race with
    concurrent get_chat_session calls).
    """
    session = ChatSession.new(user_id=test_user_id, dry_run=False)
    session.messages = [
        CopilotChatMessage(role="user", content="hello"),
        CopilotChatMessage(role="assistant", content="hi there"),
    ]
    session = await upsert_chat_session(session)

    # Ensure the session is in cache
    cached = await get_chat_session(session.session_id, test_user_id)
    assert cached is not None
    assert cached.messages[-1].duration_ms is None

    # Update turn duration — should patch cache in-place
    await set_turn_duration(session.session_id, 1234)

    # Read from cache (not DB) — the cache should already have the update
    updated = await get_chat_session(session.session_id, test_user_id)
    assert updated is not None
    assistant_msgs = [m for m in updated.messages if m.role == "assistant"]
    assert len(assistant_msgs) == 1
    assert assistant_msgs[0].duration_ms == 1234


@pytest.mark.asyncio(loop_scope="session")
async def test_set_turn_duration_no_assistant_message(setup_test_user, test_user_id):
    """set_turn_duration is a no-op when there are no assistant messages."""
    session = ChatSession.new(user_id=test_user_id, dry_run=False)
    session.messages = [
        CopilotChatMessage(role="user", content="hello"),
    ]
    session = await upsert_chat_session(session)

    # Should not raise
    await set_turn_duration(session.session_id, 5678)

    cached = await get_chat_session(session.session_id, test_user_id)
    assert cached is not None
    # User message should not have durationMs
    assert cached.messages[0].duration_ms is None


# ---------- update_message_content_by_sequence ----------


@pytest.mark.asyncio
async def test_update_message_content_by_sequence_success():
    """Returns True when update_many reports exactly one row updated."""
    with (
        patch.object(PrismaChatMessage, "prisma") as mock_prisma,
        patch("backend.copilot.db.sanitize_string", side_effect=lambda x: x),
    ):
        mock_prisma.return_value.update_many = AsyncMock(return_value=1)

        result = await update_message_content_by_sequence("sess-1", 0, "new content")

    assert result is True
    mock_prisma.return_value.update_many.assert_called_once_with(
        where={"sessionId": "sess-1", "sequence": 0},
        data={"content": "new content"},
    )


@pytest.mark.asyncio
async def test_update_message_content_by_sequence_not_found():
    """Returns False and logs a warning when no rows are updated."""
    with (
        patch.object(PrismaChatMessage, "prisma") as mock_prisma,
        patch("backend.copilot.db.logger") as mock_logger,
    ):
        mock_prisma.return_value.update_many = AsyncMock(return_value=0)

        result = await update_message_content_by_sequence("sess-1", 99, "content")

    assert result is False
    mock_logger.warning.assert_called_once()


@pytest.mark.asyncio
async def test_update_message_content_by_sequence_db_error():
    """Returns False and logs an error when the DB raises an exception."""
    with (
        patch.object(PrismaChatMessage, "prisma") as mock_prisma,
        patch("backend.copilot.db.logger") as mock_logger,
    ):
        mock_prisma.return_value.update_many = AsyncMock(
            side_effect=RuntimeError("db error")
        )

        result = await update_message_content_by_sequence("sess-1", 0, "content")

    assert result is False
    mock_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_update_message_content_by_sequence_multi_row_logs_error():
    """Returns True but logs an error when update_many touches more than one row."""
    with (
        patch.object(PrismaChatMessage, "prisma") as mock_prisma,
        patch("backend.copilot.db.logger") as mock_logger,
    ):
        mock_prisma.return_value.update_many = AsyncMock(return_value=2)

        result = await update_message_content_by_sequence("sess-1", 0, "content")

    assert result is True
    mock_logger.error.assert_called_once()


@pytest.mark.asyncio
async def test_update_message_content_by_sequence_sanitizes_content():
    """Verifies sanitize_string is applied to content before the DB write."""
    with (
        patch.object(PrismaChatMessage, "prisma") as mock_prisma,
        patch(
            "backend.copilot.db.sanitize_string", return_value="sanitized"
        ) as mock_sanitize,
    ):
        mock_prisma.return_value.update_many = AsyncMock(return_value=1)

        await update_message_content_by_sequence("sess-1", 0, "raw content")

    mock_sanitize.assert_called_once_with("raw content")
    mock_prisma.return_value.update_many.assert_called_once_with(
        where={"sessionId": "sess-1", "sequence": 0},
        data={"content": "sanitized"},
    )
