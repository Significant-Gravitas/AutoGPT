"""The chat list reads its page of chats and their total side by side: each
stand-in answers only once both are being awaited."""

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from pytest_mock import MockerFixture

from backend.copilot import model as chat_model
from backend.util.testing import Rendezvous


@pytest.mark.asyncio
async def test_a_sessions_page_and_its_total_are_read_side_by_side(
    mocker: MockerFixture,
) -> None:
    meeting = Rendezvous(2)
    db = SimpleNamespace(
        get_user_chat_sessions=AsyncMock(side_effect=meeting.returning(["s1", "s2"])),
        get_user_session_count=AsyncMock(side_effect=meeting.returning(7)),
    )
    mocker.patch.object(chat_model, "chat_db", return_value=db)

    sessions, total = await chat_model.get_user_sessions(
        "user-1", 2, 0, organization_id="org-1", expert_id="expert-1"
    )

    assert (sessions, total) == (["s1", "s2"], 7)
    db.get_user_chat_sessions.assert_awaited_once_with(
        "user-1",
        2,
        0,
        organization_id="org-1",
        title_contains=None,
        expert_id="expert-1",
        pinned_first=True,
    )
    db.get_user_session_count.assert_awaited_once_with(
        "user-1", organization_id="org-1", expert_id="expert-1"
    )
