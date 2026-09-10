from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from . import fetch as fetch_mod


@pytest.mark.asyncio
async def test_autopilot_dream_fetches_only_unscoped_sessions(mocker):
    database = SimpleNamespace(get_user_chat_sessions=AsyncMock(return_value=[]))
    mocker.patch.object(fetch_mod, "chat_db", return_value=database)

    await fetch_mod._fetch_recent_sessions("user-1", datetime.now(timezone.utc), 10)

    database.get_user_chat_sessions.assert_awaited_once_with(
        "user-1", limit=10, autopilot_only=True
    )


@pytest.mark.asyncio
async def test_expert_dream_fetches_only_that_experts_sessions(mocker):
    database = SimpleNamespace(get_user_chat_sessions=AsyncMock(return_value=[]))
    mocker.patch.object(fetch_mod, "chat_db", return_value=database)

    await fetch_mod._fetch_recent_sessions(
        "user-1", datetime.now(timezone.utc), 10, "expert-1"
    )

    database.get_user_chat_sessions.assert_awaited_once_with(
        "user-1", limit=10, expert_id="expert-1"
    )
