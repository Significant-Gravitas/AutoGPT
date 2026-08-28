from unittest.mock import AsyncMock, MagicMock

import pytest

from backend.copilot import service


@pytest.mark.asyncio
async def test_assign_user_claims_database_before_refreshing_cache(mocker):
    session = MagicMock(user_id=None)
    mocker.patch.object(
        service, "get_chat_session", new=AsyncMock(return_value=session)
    )
    client = MagicMock(claim_chat_session=AsyncMock())
    mocker.patch.object(service, "chat_db", return_value=client)
    cache = mocker.patch.object(service, "cache_chat_session", new=AsyncMock())

    result = await service.assign_user_to_session("session-1", "user-1")

    client.claim_chat_session.assert_awaited_once_with("session-1", "user-1")
    assert session.user_id == "user-1"
    cache.assert_awaited_once_with(session)
    assert result is session
