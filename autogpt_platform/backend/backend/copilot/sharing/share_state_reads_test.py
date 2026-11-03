"""The share modal's three disclosure counts are read side by side, and only
for a session the user owns."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from prisma.models import ChatMessage as PrismaChatMessage
from prisma.models import ChatSession as PrismaChatSession

from backend.copilot.sharing import db as sharing_db
from backend.util.testing import Rendezvous

SESSION_ID = "session-1"
USER_ID = "user-1"


def _owned_session() -> PrismaChatSession:
    now = datetime.now(UTC)
    return PrismaChatSession.model_construct(
        id=SESSION_ID,
        createdAt=now,
        updatedAt=now,
        userId=USER_ID,
        isShared=True,
        shareToken="token-A",
        autoShareExecutions=False,
    )


@pytest.mark.asyncio
async def test_the_disclosure_counts_are_read_side_by_side() -> None:
    meeting = Rendezvous(3)
    session_prisma = MagicMock()
    session_prisma.find_first = AsyncMock(return_value=_owned_session())
    message_prisma = MagicMock()
    message_prisma.count = AsyncMock(side_effect=meeting.returning(12))
    with (
        patch.object(PrismaChatSession, "prisma", return_value=session_prisma),
        patch.object(PrismaChatMessage, "prisma", return_value=message_prisma),
        patch.object(
            sharing_db,
            "_collect_execution_ids_from_messages",
            AsyncMock(side_effect=meeting.returning(["run-1", "run-2"])),
        ),
        patch.object(
            sharing_db,
            "_count_referenced_workspace_files",
            AsyncMock(side_effect=meeting.returning(3)),
        ),
    ):
        state = await sharing_db.get_chat_share_state(SESSION_ID, USER_ID)

    assert state.is_shared is True
    assert state.share_token == "token-A"
    assert (state.message_count, state.linked_run_count, state.file_count) == (
        12,
        2,
        3,
    )


@pytest.mark.asyncio
async def test_a_session_the_user_does_not_own_reads_none_of_its_messages() -> None:
    session_prisma = MagicMock()
    session_prisma.find_first = AsyncMock(return_value=None)
    message_prisma = MagicMock()
    message_prisma.count = AsyncMock(return_value=12)
    collect = AsyncMock(return_value=["run-1"])
    files = AsyncMock(return_value=3)
    with (
        patch.object(PrismaChatSession, "prisma", return_value=session_prisma),
        patch.object(PrismaChatMessage, "prisma", return_value=message_prisma),
        patch.object(sharing_db, "_collect_execution_ids_from_messages", collect),
        patch.object(sharing_db, "_count_referenced_workspace_files", files),
    ):
        state = await sharing_db.get_chat_share_state(SESSION_ID, "someone-else")

    assert state.is_shared is False
    message_prisma.count.assert_not_awaited()
    collect.assert_not_awaited()
    files.assert_not_awaited()
