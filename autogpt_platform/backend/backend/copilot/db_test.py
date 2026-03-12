"""Unit tests for copilot.db — paginated message queries."""

from __future__ import annotations

from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from prisma.models import ChatMessage as PrismaChatMessage

from backend.copilot.db import PaginatedMessages, get_chat_messages_paginated


def _make_msg(
    sequence: int,
    role: str = "assistant",
    content: str | None = "hello",
    tool_calls: object | None = None,
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
        # Nullable fields
        name=None,
        toolCallId=None,
        refusal=None,
        functionCall=None,
    )


SESSION_ID = "sess-1"


@pytest.fixture()
def mock_find_many():
    """Patch PrismaChatMessage.prisma().find_many as an AsyncMock."""
    with patch.object(PrismaChatMessage, "prisma") as mock_prisma:
        find_many = AsyncMock()
        mock_prisma.return_value.find_many = find_many
        yield find_many


# ---------- Basic pagination ----------


@pytest.mark.asyncio
async def test_basic_page_returns_messages_ascending(mock_find_many: AsyncMock):
    """Messages are returned in ascending sequence order."""
    # Prisma returns desc order; function reverses to asc
    mock_find_many.return_value = [_make_msg(3), _make_msg(2), _make_msg(1)]

    page = await get_chat_messages_paginated(SESSION_ID, limit=5)

    assert isinstance(page, PaginatedMessages)
    assert [m.sequence for m in page.messages] == [1, 2, 3]
    assert page.has_more is False
    assert page.oldest_sequence == 1


@pytest.mark.asyncio
async def test_has_more_when_results_exceed_limit(mock_find_many: AsyncMock):
    """has_more is True when DB returns more than limit items."""
    # limit=2, so take=3; return 3 items → has_more=True, keep first 2
    mock_find_many.return_value = [_make_msg(3), _make_msg(2), _make_msg(1)]

    page = await get_chat_messages_paginated(SESSION_ID, limit=2)

    assert page.has_more is True
    assert len(page.messages) == 2
    # After desc→asc reversal of the first 2 (seq 3,2): [2, 3]
    assert [m.sequence for m in page.messages] == [2, 3]


@pytest.mark.asyncio
async def test_empty_session_returns_no_messages(mock_find_many: AsyncMock):
    mock_find_many.return_value = []

    page = await get_chat_messages_paginated(SESSION_ID, limit=50)

    assert page.messages == []
    assert page.has_more is False
    assert page.oldest_sequence is None


@pytest.mark.asyncio
async def test_before_sequence_filters_correctly(mock_find_many: AsyncMock):
    """before_sequence is passed as a lt filter to Prisma."""
    mock_find_many.return_value = [_make_msg(2), _make_msg(1)]

    await get_chat_messages_paginated(SESSION_ID, limit=50, before_sequence=5)

    call_kwargs = mock_find_many.call_args
    where = call_kwargs.kwargs.get("where") or call_kwargs[1].get("where")
    assert where["sequence"] == {"lt": 5}


# ---------- Backward boundary expansion ----------


@pytest.mark.asyncio
async def test_boundary_expansion_includes_assistant(mock_find_many: AsyncMock):
    """When page starts with a tool message, expand backward to include
    the owning assistant message."""
    # First call: main query returns [tool(5), tool(4)] (desc), reversed → [4, 5]
    # But wait — after limit slice and reverse, first msg is tool → triggers expansion
    # Second call: boundary query returns [assistant(3)]
    mock_find_many.side_effect = [
        # Main query (desc order)
        [_make_msg(5, role="tool"), _make_msg(4, role="tool")],
        # Boundary expansion query (desc order)
        [_make_msg(3, role="assistant")],
    ]

    page = await get_chat_messages_paginated(SESSION_ID, limit=5)

    assert [m.sequence for m in page.messages] == [3, 4, 5]
    assert page.messages[0].role == "assistant"
    assert page.oldest_sequence == 3


@pytest.mark.asyncio
async def test_boundary_expansion_includes_multiple_tool_msgs(
    mock_find_many: AsyncMock,
):
    """Boundary expansion scans past consecutive tool messages to find
    the owning assistant."""
    mock_find_many.side_effect = [
        # Main query: starts with tool after reversal
        [_make_msg(7, role="tool")],
        # Boundary: tool(6), tool(5), assistant(4)
        [
            _make_msg(6, role="tool"),
            _make_msg(5, role="tool"),
            _make_msg(4, role="assistant"),
        ],
    ]

    page = await get_chat_messages_paginated(SESSION_ID, limit=5)

    assert [m.sequence for m in page.messages] == [4, 5, 6, 7]
    assert page.messages[0].role == "assistant"


@pytest.mark.asyncio
async def test_boundary_expansion_sets_has_more_when_not_at_start(
    mock_find_many: AsyncMock,
):
    """After boundary expansion, has_more=True if expanded msgs aren't at seq 0."""
    mock_find_many.side_effect = [
        [_make_msg(3, role="tool")],
        [_make_msg(2, role="assistant")],  # seq 2 > 0
    ]

    page = await get_chat_messages_paginated(SESSION_ID, limit=5)

    assert page.has_more is True


@pytest.mark.asyncio
async def test_boundary_expansion_no_has_more_at_conversation_start(
    mock_find_many: AsyncMock,
):
    """has_more stays False when boundary expansion reaches seq 0."""
    mock_find_many.side_effect = [
        [_make_msg(1, role="tool")],
        [_make_msg(0, role="assistant")],  # seq 0 = conversation start
    ]

    page = await get_chat_messages_paginated(SESSION_ID, limit=5)

    assert page.has_more is False
    assert page.oldest_sequence == 0


@pytest.mark.asyncio
async def test_no_boundary_expansion_when_first_msg_not_tool(
    mock_find_many: AsyncMock,
):
    """No boundary expansion when the first message is not a tool message."""
    mock_find_many.return_value = [
        _make_msg(3, role="user"),
        _make_msg(2, role="assistant"),
    ]

    page = await get_chat_messages_paginated(SESSION_ID, limit=5)

    # Only one find_many call (no boundary expansion)
    assert mock_find_many.call_count == 1
    assert [m.sequence for m in page.messages] == [2, 3]


@pytest.mark.asyncio
async def test_boundary_expansion_warns_when_no_owner_found(
    mock_find_many: AsyncMock,
):
    """When boundary scan doesn't find a non-tool message, a warning is logged
    and the boundary messages are still included."""
    mock_find_many.side_effect = [
        [_make_msg(10, role="tool")],
        # All tool messages — no assistant found within scan limit
        [_make_msg(i, role="tool") for i in range(9, -1, -1)],
    ]

    with patch("backend.copilot.db.logger") as mock_logger:
        page = await get_chat_messages_paginated(SESSION_ID, limit=5)
        mock_logger.warning.assert_called_once()

    # Boundary messages are still prepended
    assert page.messages[0].role == "tool"
    assert len(page.messages) > 1
