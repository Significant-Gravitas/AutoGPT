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
        chatStatus="idle",
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


# ---------- Visibility guarantee ----------


@pytest.mark.asyncio
async def test_visibility_expands_when_all_tool_messages(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """When the entire page is tool messages, expand backward to find
    at least one visible (user/assistant) message so the chat isn't blank."""
    find_first, find_many = mock_db
    # Newest 3 messages are all tool messages (DESC → reversed to ASC)
    find_first.return_value = _make_session(
        messages=[
            _make_msg(12, role="tool"),
            _make_msg(11, role="tool"),
            _make_msg(10, role="tool"),
        ],
    )
    # Boundary expansion finds the owning assistant first (boundary fix),
    # then visibility expansion finds a user message further back
    find_many.side_effect = [
        # First call: boundary fix (oldest msg is tool → find owner)
        [_make_msg(9, role="tool"), _make_msg(8, role="tool")],
        # Second call: visibility expansion (still all tool → find visible)
        [_make_msg(7, role="tool"), _make_msg(6, role="assistant")],
    ]

    page = await get_chat_messages_paginated(SESSION_ID, limit=3)

    assert page is not None
    # Should include the expanded messages + original tool messages
    roles = [m.role for m in page.messages]
    assert "assistant" in roles or "user" in roles
    assert page.has_more is True


@pytest.mark.asyncio
async def test_no_visibility_expansion_when_visible_messages_present(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """No visibility expansion needed when page already has visible messages."""
    find_first, find_many = mock_db
    # Page has an assistant message among tool messages
    find_first.return_value = _make_session(
        messages=[
            _make_msg(5, role="tool"),
            _make_msg(4, role="assistant"),
            _make_msg(3, role="user"),
        ],
    )

    page = await get_chat_messages_paginated(SESSION_ID, limit=3)

    assert page is not None
    # Boundary expansion might fire (oldest is tool), but NOT visibility
    assert [m.sequence for m in page.messages][0] <= 3


@pytest.mark.asyncio
async def test_visibility_no_expansion_when_no_earlier_messages(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """When the page is all tool messages but there are no earlier messages
    in the DB, visibility expansion returns early without changes."""
    find_first, find_many = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(1, role="tool"), _make_msg(0, role="tool")],
    )
    # Boundary expansion: no earlier messages
    # Visibility expansion: no earlier messages
    find_many.side_effect = [[], []]

    page = await get_chat_messages_paginated(SESSION_ID, limit=2)

    assert page is not None
    assert all(m.role == "tool" for m in page.messages)


@pytest.mark.asyncio
async def test_visibility_expansion_reaches_seq_zero(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """When visibility expansion finds a visible message at sequence 0,
    has_more should be False."""
    find_first, find_many = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(5, role="tool"), _make_msg(4, role="tool")],
    )
    find_many.side_effect = [
        # Boundary expansion
        [_make_msg(3, role="tool")],
        # Visibility expansion — finds user at seq 0
        [
            _make_msg(2, role="tool"),
            _make_msg(1, role="tool"),
            _make_msg(0, role="user"),
        ],
    ]

    page = await get_chat_messages_paginated(SESSION_ID, limit=2)

    assert page is not None
    assert page.messages[0].role == "user"
    assert page.messages[0].sequence == 0
    assert page.has_more is False


@pytest.mark.asyncio
async def test_visibility_expansion_with_user_id(
    mock_db: tuple[AsyncMock, AsyncMock],
):
    """Visibility expansion passes user_id filter to the boundary query."""
    find_first, find_many = mock_db
    find_first.return_value = _make_session(
        messages=[_make_msg(10, role="tool")],
    )
    find_many.side_effect = [
        # Boundary expansion
        [_make_msg(9, role="tool")],
        # Visibility expansion
        [_make_msg(8, role="assistant")],
    ]

    await get_chat_messages_paginated(SESSION_ID, limit=1, user_id="user-abc")

    # Both find_many calls should include the user_id session filter
    for call in find_many.call_args_list:
        where = call.kwargs.get("where") or call[1].get("where")
        assert "Session" in where
        assert where["Session"] == {"is": {"userId": "user-abc"}}


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
        # Two warnings: boundary expansion + visibility expansion (all tool msgs)
        assert mock_logger.warning.call_count == 2

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


# NOTE: previously this file had a separate suite for ``db.get_chat_session``
# (windowed eager-load). That function was removed in favour of going through
# ``get_chat_messages_paginated`` directly — see ``model._get_session_from_db``.
# Cap-hit + tool-pair boundary behaviour is now covered by the paginated tests
# above and the integration coverage in ``model_test.py``.


# ChatSession lifecycle primitives + add_chat_message.


@pytest.mark.asyncio
async def test_count_chat_sessions_by_status_filters_by_user_and_status() -> None:
    from backend.copilot.db import count_chat_sessions_by_status

    count = AsyncMock(return_value=3)
    with patch.object(PrismaChatSession, "prisma", return_value=AsyncMock(count=count)):
        result = await count_chat_sessions_by_status(user_id="u1", status="running")
    assert result == 3
    where = count.call_args.kwargs["where"]
    assert where == {"userId": "u1", "chatStatus": "running"}


@pytest.mark.asyncio
async def test_list_chat_sessions_by_status_returns_app_models_oldest_first() -> None:
    """Returns RPC-safe ``ChatSessionInfo`` rows (not raw Prisma) so the
    function can serve the dispatcher across the DatabaseManager RPC
    boundary from the CoPilotExecutor subprocess."""
    from backend.copilot.db import list_chat_sessions_by_status

    now = datetime.now(UTC)
    rows = [
        PrismaChatSession(
            id="s1",
            userId="u1",
            chatStatus="queued",
            createdAt=now,
            updatedAt=now,
            credentials="{}",
            successfulAgentRuns="{}",
            successfulAgentSchedules="{}",
            metadata="{}",
            totalPromptTokens=0,
            totalCompletionTokens=0,
        ),
        PrismaChatSession(
            id="s2",
            userId="u1",
            chatStatus="queued",
            createdAt=now,
            updatedAt=now,
            credentials="{}",
            successfulAgentRuns="{}",
            successfulAgentSchedules="{}",
            metadata="{}",
            totalPromptTokens=0,
            totalCompletionTokens=0,
        ),
    ]
    find_many = AsyncMock(return_value=rows)
    with patch.object(
        PrismaChatSession, "prisma", return_value=AsyncMock(find_many=find_many)
    ):
        result = await list_chat_sessions_by_status(user_id="u1", status="queued")
    assert [r.session_id for r in result] == ["s1", "s2"]
    assert all(r.chat_status == "queued" for r in result)
    kwargs = find_many.call_args.kwargs
    assert kwargs["where"] == {"userId": "u1", "chatStatus": "queued"}
    assert kwargs["order"] == {"updatedAt": "asc"}


@pytest.mark.asyncio
async def test_update_chat_session_status_owner_gated_returns_true_on_match() -> None:
    """User-initiated transition (cancel) gates on both ``chatStatus``
    AND ``userId`` — both guards in one atomic update."""
    from backend.copilot.db import update_chat_session_status

    update_many = AsyncMock(return_value=1)
    with patch.object(
        PrismaChatSession, "prisma", return_value=AsyncMock(update_many=update_many)
    ):
        ok = await update_chat_session_status(
            session_id="s1",
            expect_status="queued",
            status="idle",
            user_id="u1",
        )
    assert ok is True
    kwargs = update_many.call_args.kwargs
    assert kwargs["where"] == {"id": "s1", "chatStatus": "queued", "userId": "u1"}
    assert kwargs["data"] == {"chatStatus": "idle"}


@pytest.mark.asyncio
async def test_update_chat_session_status_returns_false_when_cas_fails() -> None:
    from backend.copilot.db import update_chat_session_status

    update_many = AsyncMock(return_value=0)
    with patch.object(
        PrismaChatSession, "prisma", return_value=AsyncMock(update_many=update_many)
    ):
        ok = await update_chat_session_status(
            session_id="s1", expect_status="queued", status="idle"
        )
    assert ok is False


@pytest.mark.asyncio
async def test_update_chat_session_status_dispatcher_path_omits_user_gate() -> None:
    """Dispatcher-initiated transitions (claim / restore / release) don't
    pass ``user_id`` — they act on the session regardless of owner."""
    from backend.copilot.db import update_chat_session_status

    update_many = AsyncMock(return_value=1)
    with patch.object(
        PrismaChatSession, "prisma", return_value=AsyncMock(update_many=update_many)
    ):
        await update_chat_session_status(
            session_id="s1", expect_status="queued", status="running"
        )
    where = update_many.call_args.kwargs["where"]
    assert "userId" not in where
    assert where == {"id": "s1", "chatStatus": "queued"}


@pytest.mark.asyncio
async def test_add_chat_message_serialises_metadata_via_safejson() -> None:
    from backend.copilot.db import add_chat_message

    create = AsyncMock(return_value=_make_msg(sequence=1))
    session_update = AsyncMock()
    with (
        patch.object(
            PrismaChatMessage, "prisma", return_value=AsyncMock(create=create)
        ),
        patch.object(
            PrismaChatSession, "prisma", return_value=AsyncMock(update=session_update)
        ),
    ):
        await add_chat_message(
            message_id="m1",
            session_id="s1",
            role="user",
            content="hi",
            sequence=1,
            metadata={"mode": "extended_thinking"},
        )
    data = create.call_args.kwargs["data"]
    assert data["id"] == "m1"
    metadata = data["metadata"]
    inner = getattr(metadata, "data", metadata)
    assert inner == {"mode": "extended_thinking"}


@pytest.mark.asyncio
async def test_add_chat_message_omits_metadata_when_none() -> None:
    from backend.copilot.db import add_chat_message

    create = AsyncMock(return_value=_make_msg(sequence=1))
    session_update = AsyncMock()
    with (
        patch.object(
            PrismaChatMessage, "prisma", return_value=AsyncMock(create=create)
        ),
        patch.object(
            PrismaChatSession, "prisma", return_value=AsyncMock(update=session_update)
        ),
    ):
        await add_chat_message(
            message_id="m1",
            session_id="s1",
            role="user",
            content="hi",
            sequence=1,
            metadata=None,
        )
    data = create.call_args.kwargs["data"]
    assert "metadata" not in data
