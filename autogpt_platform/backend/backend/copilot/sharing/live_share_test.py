"""A shared chat serves files and linked runs only while they are live."""

from contextlib import asynccontextmanager
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.models import ChatLinkedShare, SharedChatFile

from backend.copilot.sharing.db import (
    _resolve_linked_executions,
    shared_chat_file_access,
)

TOKEN = "550e8400-e29b-41d4-a716-446655440000"
FILE_ID = "6ba7b810-9dad-11d1-80b4-00c04fd430c8"


@pytest.mark.asyncio
async def test_an_allowlisted_file_needs_the_chat_still_shared_under_its_token():
    tx = MagicMock()
    tx.query_raw = AsyncMock(return_value=[])

    @asynccontextmanager
    async def fake_transaction():
        yield tx

    with (
        patch("backend.copilot.sharing.db.transaction", fake_transaction),
        patch.object(SharedChatFile, "prisma") as file_prisma,
    ):
        async with shared_chat_file_access(TOKEN, FILE_ID) as file:
            assert file is None

    # The share row is locked by token and live flag before any file is read,
    # so a chat unshared (or re-shared under a new token) serves nothing.
    query, token = tx.query_raw.await_args.args
    assert '"shareToken" = $1' in query
    assert '"isShared" = true' in query
    assert "FOR SHARE" in query
    assert token == TOKEN
    file_prisma.assert_not_called()


@pytest.mark.asyncio
async def test_a_shared_chat_does_not_list_deleted_runs():
    with patch.object(ChatLinkedShare, "prisma") as linked_prisma:
        linked_prisma.return_value.find_many = AsyncMock(return_value=[])

        assert await _resolve_linked_executions(session_id="chat-1") == []

    where = linked_prisma.return_value.find_many.await_args.kwargs["where"]
    assert where == {"sessionId": "chat-1", "Execution": {"is": {"isDeleted": False}}}
