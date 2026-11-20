"""The share modal's disclosure counts: the messages are counted first, then
the two reference scans run side by side, and only for a session the user
owns."""

import asyncio
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
async def test_the_messages_are_counted_before_the_two_scans_run_side_by_side() -> None:
    meeting = Rendezvous(2)
    counted: list[int] = []
    counted_when_a_scan_started: list[bool] = []

    async def count_messages(*_args: object, **_kwargs: object) -> int:
        await asyncio.sleep(0)  # a real count yields before it answers
        counted.append(12)
        return 12

    async def collect_runs(*_args: object, **_kwargs: object) -> list[str]:
        counted_when_a_scan_started.append(bool(counted))
        await meeting.arrive()
        return ["run-1", "run-2"]

    async def count_files(*_args: object, **_kwargs: object) -> int:
        counted_when_a_scan_started.append(bool(counted))
        await meeting.arrive()
        return 3

    session_prisma = MagicMock()
    session_prisma.find_first = AsyncMock(return_value=_owned_session())
    message_prisma = MagicMock()
    message_prisma.count = AsyncMock(side_effect=count_messages)
    with (
        patch.object(PrismaChatSession, "prisma", return_value=session_prisma),
        patch.object(PrismaChatMessage, "prisma", return_value=message_prisma),
        patch.object(sharing_db, "_collect_execution_ids_from_messages", collect_runs),
        patch.object(sharing_db, "_count_referenced_workspace_files", count_files),
    ):
        state = await sharing_db.get_chat_share_state(SESSION_ID, USER_ID)

    assert counted_when_a_scan_started == [True, True]
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
