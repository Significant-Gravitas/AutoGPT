"""A shared chat serves files and linked runs only while they are live."""

from unittest.mock import AsyncMock, patch

import pytest
from prisma.models import ChatLinkedShare, SharedChatFile

from backend.copilot.sharing.db import _resolve_linked_executions, get_shared_chat_file

TOKEN = "550e8400-e29b-41d4-a716-446655440000"
FILE_ID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"


@pytest.mark.asyncio
async def test_an_allowlisted_file_needs_the_chat_still_shared_under_its_token():
    with (
        patch.object(SharedChatFile, "prisma") as file_prisma,
        patch("backend.copilot.sharing.db.PrismaChatSession.prisma") as session_prisma,
    ):
        file_prisma.return_value.find_first = AsyncMock(return_value=None)
        session_prisma.return_value.find_first = AsyncMock(return_value=None)

        assert await get_shared_chat_file(TOKEN, FILE_ID) is None

    where = file_prisma.return_value.find_first.await_args.kwargs["where"]
    assert where == {
        "shareToken": TOKEN,
        "fileId": FILE_ID,
        "Session": {"is": {"shareToken": TOKEN, "isShared": True}},
    }
    session_where = session_prisma.return_value.find_first.await_args.kwargs["where"]
    assert session_where == {"shareToken": TOKEN, "isShared": True}


@pytest.mark.asyncio
async def test_a_shared_chat_does_not_list_deleted_runs():
    with patch.object(ChatLinkedShare, "prisma") as linked_prisma:
        linked_prisma.return_value.find_many = AsyncMock(return_value=[])

        assert await _resolve_linked_executions(session_id="chat-1") == []

    where = linked_prisma.return_value.find_many.await_args.kwargs["where"]
    assert where == {"sessionId": "chat-1", "Execution": {"is": {"isDeleted": False}}}
